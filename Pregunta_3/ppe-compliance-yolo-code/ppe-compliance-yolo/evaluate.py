"""
Evaluación cuantitativa del detector.

Calcula mAP@0.5 y mAP@0.5:0.95 globales y su desglose por clase sobre la
partición indicada (por defecto, test). Guarda los resultados en un JSON para
poder reproducir las tablas del informe.

Ejemplo:
    python evaluate.py --weights runs/ppe_detector/weights/best.pt \
        --data data/ppe.yaml --split test --imgsz 640
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Evalúa mAP por clase del detector de EPP.")
    ap.add_argument("--weights", required=True, help="ruta a best.pt")
    ap.add_argument("--data", default="data/ppe.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/eval_metrics.json")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Ultralytics no está instalado. Ejecuta: pip install -r requirements.txt")

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data, split=args.split, imgsz=args.imgsz,
        batch=args.batch, device=args.device, plots=True,
    )

    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    box = metrics.box

    report = {
        "split": args.split,
        "imgsz": args.imgsz,
        "global": {
            "mAP@0.5": round(float(box.map50), 4),
            "mAP@0.5:0.95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
        },
        "per_class": {},
    }

    # Desglose por clase (orden de box.ap_class_index).
    for i, c in enumerate(box.ap_class_index):
        cls_name = names.get(int(c), str(int(c)))
        report["per_class"][cls_name] = {
            "AP@0.5": round(float(box.ap50[i]), 4),
            "AP@0.5:0.95": round(float(box.ap[i]), 4),
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Impresión legible.
    print("\n== Resultados globales ==")
    for k, v in report["global"].items():
        print(f"  {k:14s}: {v}")
    print("\n== Desglose por clase ==")
    print(f"  {'clase':14s} {'AP@0.5':>8s} {'AP@.5:.95':>10s} {'P':>7s} {'R':>7s}")
    for cls_name, m in report["per_class"].items():
        print(f"  {cls_name:14s} {m['AP@0.5']:>8.3f} {m['AP@0.5:0.95']:>10.3f} "
              f"{m['precision']:>7.3f} {m['recall']:>7.3f}")
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    main()
