"""Componentes del detector reimplementados por el equipo.

Estos módulos (cabeza desacoplada sin anclas, asignador alineado con la tarea y
pérdida CIoU+DFL+BCE) constituyen los componentes principales del detector y se
validan numéricamente de forma independiente. El entrenamiento a escala del
informe se apoya en el framework Ultralytics, que comparte exactamente este
diseño.
"""

from .head import DetectHead, DFL, make_anchors, dist2bbox
from .assigner import TaskAlignedAssigner, bbox_iou_pairwise
from .loss import DetectionLoss, bbox_ciou, df_loss

__all__ = [
    "DetectHead",
    "DFL",
    "make_anchors",
    "dist2bbox",
    "TaskAlignedAssigner",
    "bbox_iou_pairwise",
    "DetectionLoss",
    "bbox_ciou",
    "df_loss",
]
