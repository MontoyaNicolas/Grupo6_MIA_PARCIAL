"""
Cabeza de detección desacoplada y sin anclas (anchor-free).

Esta es una reimplementación didáctica del cabezal de detección estilo YOLOv8.
Se utiliza para verificar numéricamente nuestra comprensión de los componentes
principales (ramas desacopladas de clasificación y regresión, regresión de caja
mediante distribución discreta + DFL, y decodificación libre de anclas). El
entrenamiento a gran escala del informe se realiza sobre el framework Ultralytics,
pero estos módulos son ejecutables y se validan con tensores aleatorios en
`tests/test_components.py`.

Convenciones:
  - Entrada: lista de mapas de características de un FPN, p. ej. P3, P4, P5 con
    strides {8, 16, 32}.
  - La rama de regresión predice, por cada lado de la caja (l, t, r, b), una
    distribución discreta sobre `reg_max + 1` posiciones. El valor esperado de
    esa distribución (Distribution Focal Loss, DFL) da la distancia continua al
    borde, expresada en unidades de celda.
  - La decodificación a (x1, y1, x2, y2) usa los centros de las celdas del
    "grid" multiplicados por el stride correspondiente.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(in_ch: int, out_ch: int, k: int = 3, s: int = 1) -> nn.Sequential:
    """Bloque Conv-BN-SiLU estándar."""
    pad = k // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )


class DFL(nn.Module):
    """Integral de la distribución discreta de distancias (Distribution Focal Loss).

    Convierte los logits de forma (B, 4*(reg_max+1), A) en distancias continuas
    (B, 4, A) tomando la esperanza de la softmax sobre el eje de bins.
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        # Proyección fija con los índices 0..reg_max (no entrenable).
        self.register_buffer("project", torch.arange(reg_max + 1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        x = x.view(b, 4, self.reg_max + 1, a).softmax(dim=2)
        # Esperanza sobre los bins -> distancia en unidades de celda.
        return torch.einsum("bcna,n->bca", x, self.project)


def make_anchors(
    feats: List[torch.Tensor], strides: List[int], offset: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Genera los puntos-ancla (centros de celda) y el vector de strides.

    Devuelve:
        points:  (A, 2) coordenadas (x, y) de los centros, en unidades de celda.
        strides: (A, 1) stride asociado a cada punto.
    """
    points, stride_tensor = [], []
    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        sx = torch.arange(w, device=feat.device, dtype=feat.dtype) + offset
        sy = torch.arange(h, device=feat.device, dtype=feat.dtype) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        points.append(torch.stack((sx, sy), dim=-1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride, device=feat.device, dtype=feat.dtype)
        )
    return torch.cat(points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Convierte distancias (l, t, r, b) y centros de celda en cajas xyxy."""
    lt, rb = distance.chunk(2, dim=-1)
    x1y1 = points - lt
    x2y2 = points + rb
    return torch.cat((x1y1, x2y2), dim=-1)


class DetectHead(nn.Module):
    """Cabezal de detección desacoplado y sin anclas.

    Args:
        num_classes: número de clases de la taxonomía unificada (por defecto 5:
            persona, casco, cabeza, chaleco, sin_chaleco).
        in_channels: canales de cada nivel del FPN (P3, P4, P5).
        reg_max: número de bins menos uno para la distribución de cajas (DFL).
        strides: strides de cada nivel del FPN.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: Tuple[int, ...] = (256, 512, 512),
        reg_max: int = 16,
        strides: Tuple[int, ...] = (8, 16, 32),
    ) -> None:
        super().__init__()
        self.nc = num_classes
        self.nl = len(in_channels)
        self.reg_max = reg_max
        self.no = num_classes + 4 * (reg_max + 1)  # salidas por celda
        self.strides = list(strides)

        # Canales intermedios de cada rama (siguiendo el diseño de YOLOv8).
        c2 = max(16, in_channels[0] // 4, 4 * (reg_max + 1))
        c3 = max(in_channels[0], min(num_classes, 100))

        # Rama de regresión (cajas): produce 4*(reg_max+1) canales por celda.
        self.reg_branch = nn.ModuleList(
            nn.Sequential(_conv(c, c2), _conv(c2, c2), nn.Conv2d(c2, 4 * (reg_max + 1), 1))
            for c in in_channels
        )
        # Rama de clasificación: produce `num_classes` canales por celda.
        self.cls_branch = nn.ModuleList(
            nn.Sequential(_conv(c, c3), _conv(c3, c3), nn.Conv2d(c3, num_classes, 1))
            for c in in_channels
        )
        self.dfl = DFL(reg_max)
        self._init_bias()

    def _init_bias(self) -> None:
        """Inicializa el sesgo de la rama de clasificación para estabilizar el
        arranque (probabilidad a priori baja, como en RetinaNet)."""
        prior = 0.01
        for m in self.cls_branch:
            b = m[-1].bias.view(-1)
            b.data.fill_(-math.log((1 - prior) / prior))

    def forward(self, feats: List[torch.Tensor]):
        """Pase hacia delante.

        En entrenamiento devuelve los mapas crudos por nivel (para la pérdida).
        En inferencia devuelve cajas decodificadas y scores por clase.
        """
        raw = []
        for i in range(self.nl):
            reg = self.reg_branch[i](feats[i])
            cls = self.cls_branch[i](feats[i])
            raw.append(torch.cat((reg, cls), dim=1))

        if self.training:
            return raw

        # ---- Decodificación para inferencia ----
        points, stride_tensor = make_anchors(feats, self.strides)
        # Aplanar y concatenar todos los niveles: (B, no, A)
        x = torch.cat([r.view(r.shape[0], self.no, -1) for r in raw], dim=2)
        box_logits, cls_logits = x.split((4 * (self.reg_max + 1), self.nc), dim=1)
        distance = self.dfl(box_logits)  # (B, 4, A) en unidades de celda
        # points -> (1, A, 2); distance -> (B, A, 4)
        boxes_cells = dist2bbox(distance.permute(0, 2, 1), points.unsqueeze(0))
        boxes = boxes_cells * stride_tensor.view(1, -1, 1)  # a píxeles
        scores = cls_logits.permute(0, 2, 1).sigmoid()  # (B, A, nc)
        return boxes, scores


if __name__ == "__main__":
    # Comprobación rápida de formas con un FPN sintético.
    torch.manual_seed(0)
    head = DetectHead(num_classes=5, in_channels=(64, 128, 256))
    feats = [
        torch.randn(2, 64, 80, 80),
        torch.randn(2, 128, 40, 40),
        torch.randn(2, 256, 20, 20),
    ]
    head.train()
    raw = head(feats)
    print("Entrenamiento: niveles =", [tuple(r.shape) for r in raw])
    head.eval()
    with torch.no_grad():
        boxes, scores = head(feats)
    print("Inferencia: cajas =", tuple(boxes.shape), "scores =", tuple(scores.shape))
