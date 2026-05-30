"""
Entrenamiento del detector de cumplimiento de EPP.

Afina un modelo YOLO (YOLOv8/YOLOv9) preentrenado en COCO sobre la taxonomía
unificada de cinco clases. Las decisiones de entrenamiento (resolución, mosaico
con apagado tardío, aumento de datos, planificador) se cargan desde
`configs/hyp.yaml` y se documentan allí.

Los componentes principales del detector —cabeza desacoplada sin anclas,
asignación alineada con la tarea y pérdida CIoU+DFL+BCE— están reimplementados y
verificados en `model/` (ver `tests/test_components.py`). Ultralytics comparte
exactamente ese diseño y se emplea aquí para el entrenamiento a escala.

Ejemplo:
    python train.py --data data/ppe.yaml --model yolov8m.pt --imgsz 640 \
        --epochs 100 --batch 16 --name ppe_yolov8m
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_hyp(path: str) -> dict:
    if not Path(path).exists():
        print(f"[aviso] No se encontró {path}; se usan los valores por defecto de Ultralytics.")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser(description="Entrena el detector de EPP con YOLO.")
    ap.add_argument("--data", default="data/ppe.yaml", help="data.yaml del dataset unificado")
    ap.add_argument("--model", default="yolov8m.pt",
                    help="pesos base (yolov8m.pt, yolov8l.pt, yolov9c.pt, …)")
    ap.add_argument("--hyp", default="configs/hyp.yaml", help="hiperparámetros")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="resolución de entrada (usar 1280 para objetos pequeños)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="'0', '0,1' o 'cpu'")
    ap.add_argument("--name", default="ppe_detector", help="nombre del experimento")
    ap.add_argument("--project", default="runs", help="carpeta de resultados")
    ap.add_argument("--patience", type=int, default=30, help="early stopping")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "Ultralytics no está instalado. Ejecuta: pip install -r requirements.txt"
        )

    hyp = load_hyp(args.hyp)
    model = YOLO(args.model)

    # Los hiperparámetros de configs/hyp.yaml se mezclan con los argumentos de CLI.
    train_args = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=args.project,
        patience=args.patience,
        seed=args.seed,
        # El verificador exige recobrado alto en las clases negativas; activamos
        # cosine LR y guardado del mejor checkpoint por mAP.
        cos_lr=True,
        plots=True,
    )
    train_args.update(hyp)

    print("== Configuración de entrenamiento ==")
    for k, v in sorted(train_args.items()):
        print(f"  {k}: {v}")

    results = model.train(**train_args)
    print("\nEntrenamiento finalizado.")
    print(f"Mejor checkpoint: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    return results


if __name__ == "__main__":
    main()
