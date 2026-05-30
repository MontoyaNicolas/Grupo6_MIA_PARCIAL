"""
Análisis de robustez ante oclusión.

Particiona la partición de prueba en tres niveles de oclusión —bajo, medio,
alto— usando el metadato generado por `prepare_datasets.py`, evalúa el detector
en cada subconjunto por separado y reporta la caída de rendimiento. El índice de
oclusión es un proxy calibrado (fracción del área de la persona cubierta por
otras cajas), tal y como se discute en el informe.

Ejemplo:
    python occlusion_analysis.py --weights runs/ppe_detector/weights/best.pt \
        --dataset datasets --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

LEVELS = ["low", "medium", "high"]


def build_subset(dataset: Path, split: str, level: str, occ_map: dict, work: Path) -> Path:
    """Crea (con enlaces simbólicos) un subconjunto con las imágenes de un nivel
    de oclusión y devuelve la ruta de su data.yaml temporal."""
    img_dir = work / level / "images" / split
    lbl_dir = work / level / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    src_img = dataset / "images" / split
    src_lbl = dataset / "labels" / split
    n = 0
    for img_name, lv in occ_map.items():
        if lv != level:
            continue
        s_img = src_img / img_name
        s_lbl = src_lbl / (Path(img_name).stem + ".txt")
        if not s_img.exists():
            continue
        d_img = img_dir / img_name
        d_lbl = lbl_dir / s_lbl.name
        for d, s in ((d_img, s_img), (d_lbl, s_lbl)):
            if d.exists() or d.is_symlink():
                d.unlink()
            if s.exists():
                os.symlink(s.resolve(), d)
        n += 1

    yaml_path = work / level / "data.yaml"
    yaml_path.write_text(
        f"path: {(work / level).resolve()}\n"
        f"train: images/{split}\n"
        f"val: images/{split}\n"
        f"nc: 5\n"
        f"names: ['persona', 'casco', 'cabeza', 'chaleco', 'sin_chaleco']\n"
    )
    return yaml_path, n


def main():
    ap = argparse.ArgumentParser(description="Mide la robustez del detector ante oclusión.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", default="datasets", help="raíz del dataset unificado")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/occlusion_metrics.json")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Ultralytics no está instalado. Ejecuta: pip install -r requirements.txt")

    dataset = Path(args.dataset)
    occ_file = dataset / f"occlusion_{args.split}.json"
    if not occ_file.exists():
        raise SystemExit(
            f"No se encontró {occ_file}. Ejecuta antes data/prepare_datasets.py."
        )
    occ_map = json.loads(occ_file.read_text())

    model = YOLO(args.weights)
    work = Path("runs") / "occlusion_subsets"
    results = {}

    for level in LEVELS:
        yaml_path, n = build_subset(dataset, args.split, level, occ_map, work)
        if n == 0:
            print(f"[{level}] sin imágenes en este nivel; se omite.")
            continue
        print(f"\n== Nivel de oclusión: {level} ({n} imágenes) ==")
        m = model.val(data=str(yaml_path), split="val", imgsz=args.imgsz,
                      device=args.device, plots=False)
        results[level] = {
            "n_images": n,
            "mAP@0.5": round(float(m.box.map50), 4),
            "mAP@0.5:0.95": round(float(m.box.map), 4),
            "recall": round(float(m.box.mr), 4),
        }

    # Caída de rendimiento bajo -> alto.
    if "low" in results and "high" in results:
        drop50 = results["low"]["mAP@0.5"] - results["high"]["mAP@0.5"]
        results["degradation_low_to_high"] = {
            "mAP@0.5_points": round(drop50 * 100, 2),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n== Resumen por nivel de oclusión ==")
    print(f"  {'nivel':8s} {'N':>6s} {'mAP@0.5':>9s} {'mAP@.5:.95':>11s} {'recall':>8s}")
    for level in LEVELS:
        if level in results:
            r = results[level]
            print(f"  {level:8s} {r['n_images']:>6d} {r['mAP@0.5']:>9.3f} "
                  f"{r['mAP@0.5:0.95']:>11.3f} {r['recall']:>8.3f}")
    if "degradation_low_to_high" in results:
        print(f"\n  Caída mAP@0.5 (bajo -> alto): "
              f"{results['degradation_low_to_high']['mAP@0.5_points']} puntos")
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    main()
