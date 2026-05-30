# Detección de Cumplimiento de Normas de Seguridad en Obras de Construcción con YOLO

Sistema completo para detectar trabajadores y verificar el uso de Equipo de
Protección Personal (EPP) —casco y chaleco de alta visibilidad— en imágenes y
vídeo de obras de construcción. Combina un **detector YOLO** afinado con un
**verificador de cumplimiento basado en reglas** que produce una tasa de
cumplimiento auditable por fotograma y marca a los trabajadores no conformes.

Este repositorio acompaña al informe del proyecto y reproduce todos sus
resultados: entrenamiento, evaluación por clase (mAP@0.5 y mAP@0.5:0.95),
análisis de robustez ante oclusión y la demostración del verificador.

---

## Taxonomía unificada

Las tres fuentes de datos usan convenciones distintas; se armonizan en cinco
clases. El cumplimiento **no** es una clase: lo deriva el verificador.

| Índice | Clase         | Significado                                   |
|:------:|---------------|-----------------------------------------------|
| 0      | `persona`     | cuerpo completo del trabajador                |
| 1      | `casco`       | casco de seguridad puesto                     |
| 2      | `cabeza`      | cabeza descubierta (equivale a "sin casco")   |
| 3      | `chaleco`     | chaleco de alta visibilidad puesto            |
| 4      | `sin_chaleco` | torso sin chaleco                             |

> Nota sobre SHWD: su clase `person` anota **cabezas sin casco**, por lo que se
> reasigna a `cabeza` durante la armonización.

---

## Instalación

Requiere Python 3.9+ (probado en 3.10–3.12).

```bash
git clone https://github.com/<equipo>/ppe-compliance-yolo.git
cd ppe-compliance-yolo
pip install -r requirements.txt
```

El entrenamiento usa GPU automáticamente si `torch` detecta CUDA; en CPU también
funciona (más lento).

---

## Datos

Descarga las tres fuentes públicas y colócalas bajo `datasets_raw/` así:

```
datasets_raw/
├── shwd/      # Safety-Helmet-Wearing-Dataset  (formato Pascal VOC)
├── helmet/    # Safety Helmet Detection (Roboflow/Kaggle, formato YOLO)
└── ppe/       # PPE / Construction Site Safety (Roboflow, formato YOLO)
```

Fuentes:
- **Safety Helmet Detection** — Roboflow Universe / Kaggle (~5.000 imágenes, CC BY 4.0).
- **PPE / Construction Site Safety** — Roboflow Universe (~10.000 imágenes).
- **SHWD** — github.com/njvisionpower/Safety-Helmet-Wearing-Dataset (ejemplos negativos "sin casco").

El script de preparación detecta el formato (VOC o YOLO), aplica el mapeo de
clases, genera las particiones train/val/test y calcula el metadato de oclusión.
Se omiten en silencio las fuentes que no estén presentes, de modo que puedes
empezar con una sola.

---

## Ejecución de extremo a extremo (un solo comando)

```bash
bash run_all.sh
```

Esto encadena: (0) verificación de componentes → (1) armonización de datos →
(2) entrenamiento → (3) evaluación por clase → (4) análisis de oclusión →
(5) demostración del verificador.

Parámetros configurables por variables de entorno (con sus valores por defecto):

```bash
MODEL=yolov8m.pt IMGSZ=640 EPOCHS=100 BATCH=16 NAME=ppe_detector \
DEMO_SOURCE=ejemplos bash run_all.sh
```

Para el experimento de objetos pequeños del informe, repite con `IMGSZ=1280`.

---

## Uso por etapas

**1. Preparar los datos**
```bash
python data/prepare_datasets.py --raw datasets_raw --out datasets --seed 0
```

**2. Entrenar** (decisiones documentadas en `configs/hyp.yaml`)
```bash
python train.py --data datasets/ppe.yaml --model yolov8m.pt \
    --imgsz 640 --epochs 100 --batch 16 --name ppe_detector
```

**3. Evaluar** (mAP@0.5 y mAP@0.5:0.95 con desglose por clase)
```bash
python evaluate.py --weights runs/ppe_detector/weights/best.pt \
    --data datasets/ppe.yaml --split test --imgsz 640
```

**4. Robustez ante oclusión** (particiones bajo/medio/alto y caída de mAP)
```bash
python occlusion_analysis.py --weights runs/ppe_detector/weights/best.pt \
    --dataset datasets --split test --imgsz 640
```

**5. Demostración del verificador** (imagen, carpeta o vídeo no vistos)
```bash
python run_demo.py --weights runs/ppe_detector/weights/best.pt \
    --source ejemplos/obra.mp4 --out results/demo
# Para exigir solo casco (sin chaleco): añade --no-vest
```

---

## El verificador de cumplimiento

`compliance_verifier.py` posprocesa las detecciones del detector con reglas
explícitas y trazables:

1. **Asociación.** Cada pieza de EPP se asocia al trabajador (`persona`) que la
   contiene; el casco/cabeza se buscan en la franja superior de la persona.
2. **Estado del trabajador.** Cumple si tiene casco y —cuando se exige— chaleco.
   La evidencia de mayor confianza decide entre casco/cabeza y chaleco/sin
   chaleco.
3. **Criterio conservador.** El estado `desconocido` (EPP no visible) **no** se
   da por cumplido: en seguridad, un falso negativo es más costoso que una
   alarma de más.
4. **Tasa por fotograma + suavizado temporal.** Una ventana deslizante de
   `W=15` fotogramas estabiliza la tasa frente a detecciones intermitentes.
5. **Marcado.** Devuelve las cajas no conformes (y los incumplimientos sin
   persona asociada, típicos de oclusión severa) para resaltarlas en rojo.

El módulo solo depende de numpy y su lógica se valida de forma aislada.

---

## Componentes reimplementados

Los componentes principales del detector están reimplementados y verificados
numéricamente en `model/`:

- `model/head.py` — cabeza desacoplada **sin anclas** con regresión de caja por
  distribución discreta + integral DFL.
- `model/assigner.py` — **asignación de etiquetas alineada con la tarea** (TOOD):
  selección dinámica de positivos por la métrica `s^α · u^β`.
- `model/loss.py` — pérdida **CIoU + DFL + BCE** con objetivo de clasificación
  suave alineado con el IoU.

El entrenamiento a escala se apoya en Ultralytics, que comparte exactamente este
diseño. Para validar los componentes:

```bash
python -m pytest tests/ -q       # o:  python tests/test_components.py
```

---

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── run_all.sh                 # pipeline de extremo a extremo (un comando)
├── train.py                   # entrenamiento (Ultralytics + hiperparámetros documentados)
├── evaluate.py                # mAP por clase
├── occlusion_analysis.py      # robustez ante oclusión
├── compliance_verifier.py     # verificador por reglas (numpy)
├── run_demo.py                # demostración sobre imágenes/vídeo
├── configs/
│   └── hyp.yaml               # hiperparámetros, con justificación
├── data/
│   ├── prepare_datasets.py    # armonización a la taxonomía unificada
│   └── ppe.yaml               # config del dataset (Ultralytics)
├── model/                     # componentes reimplementados
│   ├── head.py
│   ├── assigner.py
│   └── loss.py
└── tests/
    └── test_components.py     # pruebas numéricas
```

---

## Reproducibilidad

Las semillas se fijan en la preparación de datos y el entrenamiento (`--seed 0`).
Cada etapa guarda sus artefactos: los checkpoints en `runs/<name>/weights/`, las
métricas en `results/*.json` y la demostración anotada en `results/demo/`. Para
reproducir las tablas del informe, ejecuta el pipeline completo y toma los
valores de `results/eval_metrics.json` y `results/occlusion_metrics.json`.

---

## Licencia

Código bajo licencia MIT (ver `LICENSE`). Los conjuntos de datos conservan sus
licencias de origen; revísalas antes de redistribuir.
