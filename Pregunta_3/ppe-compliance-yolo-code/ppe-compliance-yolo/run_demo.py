"""
Demostración del verificador de cumplimiento sobre escenas no vistas.

Ejecuta el detector entrenado y el verificador por reglas sobre una imagen, una
carpeta de imágenes o un vídeo, y guarda el resultado anotado:
  * Trabajadores conformes en verde, no conformes en rojo con el motivo.
  * EPP detectado (casco, chaleco) en verde; ausencias (cabeza, sin_chaleco) en
    rojo.
  * Superpuesta, la tasa de cumplimiento del fotograma y su versión suavizada.

Como el detector se entrena con la taxonomía unificada (data/ppe.yaml), los
índices de clase de Ultralytics ya coinciden con los del verificador.

Ejemplos:
    python run_demo.py --weights runs/ppe_detector/weights/best.pt \
        --source ejemplos/obra.jpg --out results/demo
    python run_demo.py --weights runs/ppe_detector/weights/best.pt \
        --source ejemplos/clip.mp4 --out results/demo --no-vest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from compliance_verifier import ComplianceVerifier, CLASS_NAMES

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
VID_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# Colores BGR.
GREEN = (60, 180, 75)
RED = (40, 40, 220)
GRAY = (160, 160, 160)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Por clase, color de la caja "normal" (las no conformes se repintan en rojo).
CLASS_COLOR = {0: (235, 170, 60), 1: GREEN, 2: RED, 3: GREEN, 4: RED}


def draw_box(img, box, color, label="", thick=2):
    import cv2
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1,
                    cv2.LINE_AA)


def overlay_panel(img, frame_res):
    """Dibuja el panel de cumplimiento en la esquina superior izquierda."""
    import cv2
    h, w = img.shape[:2]
    rate = frame_res.compliance_rate
    smooth = frame_res.smoothed_rate
    color = GREEN if smooth >= 0.999 else (RED if smooth < 0.8 else (40, 170, 235))
    lines = [
        f"Trabajadores: {frame_res.n_workers}  Conformes: {frame_res.n_compliant}",
        f"Cumplimiento: {rate*100:5.1f}%   (suavizado {smooth*100:5.1f}%)",
    ]
    if frame_res.orphan_violations:
        lines.append(f"Incumplimientos sin persona: {frame_res.orphan_violations}")
    pw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for t in lines) + 20
    ph = 26 * len(lines) + 12
    panel = img.copy()
    cv2.rectangle(panel, (8, 8), (8 + pw, 8 + ph), BLACK, -1)
    cv2.addWeighted(panel, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (8, 8), (8 + pw, 8 + ph), color, 2)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (18, 34 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2,
                    cv2.LINE_AA)


def annotate(img, detections, frame_res):
    """Pinta detecciones y juicio de cumplimiento sobre la imagen."""
    flagged_ids = {id(f["xyxy"]) if isinstance(f["xyxy"], np.ndarray) else None
                   for f in frame_res.flagged_boxes}

    # Cajas de EPP y personas (color por clase).
    for box, cid, conf in detections:
        name = CLASS_NAMES.get(int(cid), str(cid))
        color = CLASS_COLOR.get(int(cid), GRAY)
        draw_box(img, box, color, f"{name} {conf:.2f}", thick=2)

    # Personas no conformes: borde rojo grueso + motivo.
    for w in frame_res.workers:
        if not w.compliant:
            draw_box(img, w.box, RED, "NO CONFORME: " + ", ".join(w.reasons), thick=3)
        else:
            draw_box(img, w.box, GREEN, "conforme", thick=3)

    # Incumplimientos huérfanos (sin persona).
    for f in frame_res.flagged_boxes:
        if f.get("kind") == "huerfano":
            draw_box(img, f["xyxy"], RED, f["label"], thick=2)

    overlay_panel(img, frame_res)
    return img


def to_detections(result):
    """Convierte un objeto Results de Ultralytics en (box_xyxy, cls, conf)."""
    out = []
    if result.boxes is None:
        return out
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    for i in range(len(cls)):
        out.append((xyxy[i], int(cls[i]), float(conf[i])))
    return out


def run_images(model, verifier, sources, out_dir, imgsz):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sources:
        verifier.reset()
        res = model.predict(str(src), imgsz=imgsz, conf=verifier.conf_thres, verbose=False)[0]
        dets = to_detections(res)
        frame_res = verifier.verify_frame(
            [d[0] for d in dets], [d[1] for d in dets], [d[2] for d in dets]
        )
        img = cv2.imread(str(src))
        annotate(img, dets, frame_res)
        dst = out_dir / f"annotated_{Path(src).name}"
        cv2.imwrite(str(dst), img)
        print(f"  {Path(src).name}: {frame_res.n_compliant}/{frame_res.n_workers} conformes "
              f"({frame_res.compliance_rate*100:.0f}%) -> {dst}")


def run_video(model, verifier, src, out_dir, imgsz):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el vídeo {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst = out_dir / f"annotated_{Path(src).stem}.mp4"
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    verifier.reset()
    rates = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, imgsz=imgsz, conf=verifier.conf_thres, verbose=False)[0]
        dets = to_detections(res)
        frame_res = verifier.verify_frame(
            [d[0] for d in dets], [d[1] for d in dets], [d[2] for d in dets]
        )
        rates.append(frame_res.smoothed_rate)
        annotate(frame, dets, frame_res)
        writer.write(frame)
        fi += 1
    cap.release()
    writer.release()
    avg = float(np.mean(rates)) if rates else 1.0
    print(f"  {fi} fotogramas procesados. Cumplimiento medio (suavizado): {avg*100:.1f}%")
    print(f"  Vídeo anotado -> {dst}")


def main():
    ap = argparse.ArgumentParser(description="Demostración del verificador de EPP.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="imagen, carpeta o vídeo")
    ap.add_argument("--out", default="results/demo")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-vest", action="store_true",
                    help="no exigir chaleco (solo casco) para el cumplimiento")
    ap.add_argument("--smooth-window", type=int, default=15)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Ultralytics no está instalado. Ejecuta: pip install -r requirements.txt")

    model = YOLO(args.weights)
    verifier = ComplianceVerifier(
        require_vest=not args.no_vest, conf_thres=args.conf, smooth_window=args.smooth_window
    )
    out_dir = Path(args.out)
    src = Path(args.source)

    if src.is_dir():
        imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
        print(f"Procesando {len(imgs)} imágenes…")
        run_images(model, verifier, imgs, out_dir, args.imgsz)
    elif src.suffix.lower() in VID_EXTS:
        print("Procesando vídeo…")
        run_video(model, verifier, src, out_dir, args.imgsz)
    elif src.suffix.lower() in IMG_EXTS:
        run_images(model, verifier, [src], out_dir, args.imgsz)
    else:
        raise SystemExit(f"Formato no reconocido: {src}")


if __name__ == "__main__":
    main()
