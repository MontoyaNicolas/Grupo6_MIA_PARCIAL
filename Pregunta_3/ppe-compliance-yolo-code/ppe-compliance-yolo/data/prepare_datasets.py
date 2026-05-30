"""
Preparación y armonización de los conjuntos de datos.

Unifica tres fuentes públicas con convenciones de anotación distintas en una sola
taxonomía y en formato YOLO (un .txt por imagen, coordenadas normalizadas):

    Índice  Clase          Significado
    ------  -------------  -----------------------------------------------------
      0     persona        cuerpo completo de un trabajador
      1     casco          casco de seguridad puesto
      2     cabeza         cabeza descubierta (equivale a "sin casco")
      3     chaleco        chaleco de alta visibilidad puesto
      4     sin_chaleco    torso sin chaleco

El cumplimiento NO es una clase: lo deriva el verificador por reglas a partir de
estas detecciones (ver `compliance_verifier.py`).

Fuentes soportadas (se omiten en silencio las que no estén presentes):
  * SHWD  (Safety-Helmet-Wearing-Dataset) — formato Pascal VOC. Ojo: su clase
    "person" marca cabezas SIN casco, por lo que se reasigna a `cabeza`.
  * Safety Helmet Detection (Roboflow/Kaggle) — formato YOLO. Clases
    helmet/head/person.
  * PPE / Construction Site Safety (Roboflow) — formato YOLO. Clases de tipo
    Hardhat / NO-Hardhat / Safety Vest / NO-Safety Vest / Person.

El índice de oclusión por trabajador es un *proxy* calibrado: la fracción máxima
del área de una caja `persona` cubierta por otra caja. Se guarda como metadato
por imagen para el análisis de robustez (`occlusion_analysis.py`).

Uso:
    python data/prepare_datasets.py --raw datasets_raw --out datasets \
        --val 0.15 --test 0.15 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Taxonomía unificada
# --------------------------------------------------------------------------- #
UNIFIED = {"persona": 0, "casco": 1, "cabeza": 2, "chaleco": 3, "sin_chaleco": 4}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Mapeo de cada fuente a la taxonomía unificada. Las claves se comparan en
# minúsculas y sin espacios. Lo que no aparezca aquí se descarta.
SOURCE_MAPS: Dict[str, Dict[str, str]] = {
    "shwd": {
        "hat": "casco",
        "person": "cabeza",  # en SHWD "person" = cabeza sin casco
    },
    "helmet": {
        "helmet": "casco",
        "head": "cabeza",
        "person": "persona",
    },
    "ppe": {
        "hardhat": "casco",
        "helmet": "casco",
        "no-hardhat": "cabeza",
        "nohardhat": "cabeza",
        "head": "cabeza",
        "safety vest": "chaleco",
        "safety-vest": "chaleco",
        "vest": "chaleco",
        "no-safety vest": "sin_chaleco",
        "no-safety-vest": "sin_chaleco",
        "no-vest": "sin_chaleco",
        "person": "persona",
    },
}


# --------------------------------------------------------------------------- #
# Lectores de anotaciones
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    return name.strip().lower()


def read_voc(xml_path: Path) -> Tuple[int, int, List[Tuple[str, List[float]]]]:
    """Lee un XML de Pascal VOC. Devuelve (w, h, [(clase, [x1,y1,x2,y2])])."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.find("width").text))
    h = int(float(size.find("height").text))
    objs = []
    for obj in root.findall("object"):
        name = _norm(obj.find("name").text)
        b = obj.find("bndbox")
        x1 = float(b.find("xmin").text)
        y1 = float(b.find("ymin").text)
        x2 = float(b.find("xmax").text)
        y2 = float(b.find("ymax").text)
        objs.append((name, [x1, y1, x2, y2]))
    return w, h, objs


def read_yolo(txt_path: Path, names: List[str], w: int, h: int) -> List[Tuple[str, List[float]]]:
    """Lee un .txt YOLO (cls cx cy bw bh normalizados). Devuelve px xyxy."""
    objs = []
    if not txt_path.exists():
        return objs
    for line in txt_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        name = _norm(names[c]) if 0 <= c < len(names) else str(c)
        objs.append((name, [x1, y1, x2, y2]))
    return objs


def load_data_yaml_names(root: Path) -> List[str]:
    """Lee los nombres de clase de un data.yaml de Roboflow si existe."""
    for cand in ("data.yaml", "data.yml"):
        p = root / cand
        if p.exists():
            names: List[str] = []
            for line in p.read_text().splitlines():
                line = line.strip()
                if line.startswith("names:") and "[" in line:
                    inner = line.split("[", 1)[1].rsplit("]", 1)[0]
                    names = [s.strip().strip("'\"") for s in inner.split(",")]
                    break
            if names:
                return names
    return []


# --------------------------------------------------------------------------- #
# Oclusión (proxy)
# --------------------------------------------------------------------------- #
def overlap_fraction(box: List[float], others: List[List[float]]) -> float:
    """Fracción máxima del área de `box` cubierta por alguna caja de `others`."""
    ax1, ay1, ax2, ay2 = box
    area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    if area <= 0:
        return 0.0
    best = 0.0
    for o in others:
        ix1, iy1 = max(ax1, o[0]), max(ay1, o[1])
        ix2, iy2 = min(ax2, o[2]), min(ay2, o[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        best = max(best, (iw * ih) / area)
    return best


def occlusion_level(objs_px: List[Tuple[str, List[float]]]) -> str:
    """Nivel de oclusión de una imagen a partir de los solapamientos.

    Se toma como referencia la caja `persona` (o, en su defecto, cualquier caja)
    con mayor fracción de área cubierta por otras cajas.
    """
    persons = [b for n, b in objs_px if UNIFIED.get(n_to_unified(n), -1) == 0]
    boxes = [b for _, b in objs_px]
    ref = persons if persons else boxes
    if not ref:
        return "low"
    worst = 0.0
    for b in ref:
        worst = max(worst, overlap_fraction(b, [o for o in boxes if o is not b]))
    if worst < 0.15:
        return "low"
    if worst < 0.40:
        return "medium"
    return "high"


def n_to_unified(name: str) -> str:
    """Devuelve el nombre unificado de una clase ya mapeada (identidad si lo es)."""
    return name if name in UNIFIED else name


# --------------------------------------------------------------------------- #
# Conversión de una fuente
# --------------------------------------------------------------------------- #
def detect_format(root: Path) -> str:
    """'voc' si hay XMLs de anotación, 'yolo' si hay .txt de etiquetas."""
    if any(root.rglob("*.xml")):
        return "voc"
    return "yolo"


def find_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]


def label_path_for(img: Path) -> Path:
    """Ruta del .txt YOLO asociado a una imagen (estructura images/->labels/)."""
    parts = list(img.parts)
    for i, seg in enumerate(parts):
        if seg.lower() == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def voc_xml_for(img: Path) -> Path:
    for cand in (img.with_suffix(".xml"),
                 img.parent.parent / "Annotations" / (img.stem + ".xml"),
                 img.parent / "annotations" / (img.stem + ".xml")):
        if cand.exists():
            return cand
    return img.with_suffix(".xml")


def image_size(path: Path) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.width, im.height


def convert_source(name: str, root: Path) -> List[Dict]:
    """Convierte una fuente a registros unificados en memoria.

    Cada registro: {img, w, h, lines (yolo unificado), occ}
    """
    cmap = SOURCE_MAPS[name]
    fmt = detect_format(root)
    yolo_names = load_data_yaml_names(root) if fmt == "yolo" else []
    records: List[Dict] = []

    for img in find_images(root):
        try:
            if fmt == "voc":
                xml = voc_xml_for(img)
                if not xml.exists():
                    continue
                w, h, raw = read_voc(xml)
            else:
                w, h = image_size(img)
                raw = read_yolo(label_path_for(img), yolo_names, w, h)
        except Exception:
            continue

        objs_unified: List[Tuple[str, List[float]]] = []
        lines: List[str] = []
        for cls_name, box in raw:
            uni = cmap.get(cls_name)
            if uni is None:
                continue
            cid = UNIFIED[uni]
            objs_unified.append((uni, box))
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            continue
        records.append(
            {"img": img, "w": w, "h": h, "lines": lines, "occ": occlusion_level(objs_unified)}
        )
    print(f"  [{name}] {len(records)} imágenes con etiquetas útiles ({fmt}).")
    return records


# --------------------------------------------------------------------------- #
# Escritura de splits
# --------------------------------------------------------------------------- #
def write_split(records: List[Dict], out: Path, split: str) -> Dict[str, int]:
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    occ_map: Dict[str, str] = {}
    counts = {k: 0 for k in UNIFIED}
    for i, r in enumerate(records):
        stem = f"{split}_{i:06d}"
        dst_img = out / "images" / split / (stem + r["img"].suffix.lower())
        dst_lbl = out / "labels" / split / (stem + ".txt")
        shutil.copy(r["img"], dst_img)
        dst_lbl.write_text("\n".join(r["lines"]) + "\n")
        occ_map[dst_img.name] = r["occ"]
        for ln in r["lines"]:
            cid = int(ln.split()[0])
            counts[[k for k, v in UNIFIED.items() if v == cid][0]] += 1
    (out / f"occlusion_{split}.json").write_text(json.dumps(occ_map, indent=2))
    return counts


def main():
    ap = argparse.ArgumentParser(description="Armoniza los datasets a la taxonomía unificada.")
    ap.add_argument("--raw", default="datasets_raw",
                    help="carpeta con subcarpetas shwd/ helmet/ ppe/")
    ap.add_argument("--out", default="datasets", help="carpeta de salida")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    random.seed(args.seed)

    all_records: List[Dict] = []
    print("Convirtiendo fuentes…")
    for name in SOURCE_MAPS:
        root = raw / name
        if root.exists():
            all_records += convert_source(name, root)
        else:
            print(f"  [{name}] no encontrada en {root} (se omite).")

    if not all_records:
        raise SystemExit(
            "No se encontraron datos. Coloca las fuentes en "
            f"{raw}/shwd, {raw}/helmet, {raw}/ppe y vuelve a ejecutar."
        )

    random.shuffle(all_records)
    n = len(all_records)
    n_test = int(n * args.test)
    n_val = int(n * args.val)
    test_r = all_records[:n_test]
    val_r = all_records[n_test:n_test + n_val]
    train_r = all_records[n_test + n_val:]

    print(f"\nTotal: {n} imágenes -> train {len(train_r)} / val {len(val_r)} / test {len(test_r)}")
    for split, recs in (("train", train_r), ("val", val_r), ("test", test_r)):
        counts = write_split(recs, out, split)
        print(f"  {split}: instancias por clase -> {counts}")

    # data.yaml para Ultralytics.
    names_sorted = [k for k, _ in sorted(UNIFIED.items(), key=lambda kv: kv[1])]
    yaml_text = (
        f"# Generado por prepare_datasets.py\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(UNIFIED)}\n"
        f"names: {names_sorted}\n"
    )
    (out / "ppe.yaml").write_text(yaml_text)
    print(f"\nEscrito {out/'ppe.yaml'}. Listo para entrenar.")


if __name__ == "__main__":
    main()
