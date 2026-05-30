#!/usr/bin/env bash
# =============================================================================
# Pipeline de extremo a extremo: preparación -> entrenamiento -> evaluación ->
# análisis de oclusión -> demostración.
#
# Uso:
#   bash run_all.sh
#
# Variables de entorno opcionales (con sus valores por defecto):
#   RAW=datasets_raw     carpeta con las fuentes shwd/ helmet/ ppe/
#   OUT=datasets         carpeta del dataset unificado
#   MODEL=yolov8m.pt     pesos base
#   IMGSZ=640            resolución (usar 1280 para objetos pequeños)
#   EPOCHS=100
#   BATCH=16
#   NAME=ppe_detector
#   DEMO_SOURCE=ejemplos demostración (imagen, carpeta o vídeo no vistos)
# =============================================================================
set -euo pipefail

RAW="${RAW:-datasets_raw}"
OUT="${OUT:-datasets}"
MODEL="${MODEL:-yolov8m.pt}"
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-16}"
NAME="${NAME:-ppe_detector}"
DEMO_SOURCE="${DEMO_SOURCE:-ejemplos}"

WEIGHTS="runs/${NAME}/weights/best.pt"

echo "=================================================================="
echo " 0) Verificación de los componentes reimplementados"
echo "=================================================================="
python tests/test_components.py

echo "=================================================================="
echo " 1) Armonización de los datasets a la taxonomía unificada"
echo "=================================================================="
python data/prepare_datasets.py --raw "${RAW}" --out "${OUT}" --seed 0

echo "=================================================================="
echo " 2) Entrenamiento del detector (${MODEL}, imgsz=${IMGSZ})"
echo "=================================================================="
python train.py --data "${OUT}/ppe.yaml" --model "${MODEL}" \
    --imgsz "${IMGSZ}" --epochs "${EPOCHS}" --batch "${BATCH}" --name "${NAME}"

echo "=================================================================="
echo " 3) Evaluación cuantitativa (mAP por clase, partición test)"
echo "=================================================================="
python evaluate.py --weights "${WEIGHTS}" --data "${OUT}/ppe.yaml" \
    --split test --imgsz "${IMGSZ}"

echo "=================================================================="
echo " 4) Análisis de robustez ante oclusión"
echo "=================================================================="
python occlusion_analysis.py --weights "${WEIGHTS}" --dataset "${OUT}" \
    --split test --imgsz "${IMGSZ}"

echo "=================================================================="
echo " 5) Demostración del verificador en escenas no vistas"
echo "=================================================================="
if [ -e "${DEMO_SOURCE}" ]; then
    python run_demo.py --weights "${WEIGHTS}" --source "${DEMO_SOURCE}" \
        --out results/demo --imgsz "${IMGSZ}"
else
    echo "  (Omitida: coloca imágenes o un vídeo en '${DEMO_SOURCE}' para la demo.)"
fi

echo "=================================================================="
echo " Pipeline completo. Resultados en results/ y runs/${NAME}/."
echo "=================================================================="
