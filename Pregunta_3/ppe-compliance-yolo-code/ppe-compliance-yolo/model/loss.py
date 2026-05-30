"""
Función de pérdida del detector: CIoU + DFL + BCE.

Reimplementación didáctica del objetivo de entrenamiento de un detector sin
anclas estilo YOLOv8. La pérdida total combina tres términos:

  * Clasificación  -> entropía cruzada binaria (BCE) contra un objetivo suave
    proporcionado por el asignador alineado con la tarea.
  * Localización    -> Complete IoU (CIoU), que añade penalizaciones de distancia
    de centros y de relación de aspecto al IoU clásico.
  * Distribución    -> Distribution Focal Loss (DFL), que entrena la distribución
    discreta de distancias a los bordes para que su esperanza coincida con la
    distancia real.

Solo los términos de localización y DFL se aplican sobre las anclas positivas;
la BCE se aplica sobre todas las anclas con el objetivo suave (cero en el fondo).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assigner import TaskAlignedAssigner


def bbox_ciou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """CIoU por pares alineados (N, 4) vs (N, 4) en formato xyxy. Devuelve (N,)."""
    # Intersección.
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area_p = (pred[:, 2] - pred[:, 0]).clamp(0) * (pred[:, 3] - pred[:, 1]).clamp(0)
    area_t = (target[:, 2] - target[:, 0]).clamp(0) * (target[:, 3] - target[:, 1]).clamp(0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    # Caja envolvente mínima.
    cw = torch.max(pred[:, 2], target[:, 2]) - torch.min(pred[:, 0], target[:, 0])
    ch = torch.max(pred[:, 3], target[:, 3]) - torch.min(pred[:, 1], target[:, 1])
    c2 = cw.pow(2) + ch.pow(2) + eps

    # Distancia entre centros.
    px = (pred[:, 0] + pred[:, 2]) / 2
    py = (pred[:, 1] + pred[:, 3]) / 2
    tx = (target[:, 0] + target[:, 2]) / 2
    ty = (target[:, 1] + target[:, 3]) / 2
    rho2 = (px - tx).pow(2) + (py - ty).pow(2)

    # Término de relación de aspecto.
    wp = (pred[:, 2] - pred[:, 0]).clamp(min=eps)
    hp = (pred[:, 3] - pred[:, 1]).clamp(min=eps)
    wt = (target[:, 2] - target[:, 0]).clamp(min=eps)
    ht = (target[:, 3] - target[:, 1]).clamp(min=eps)
    v = (4 / (torch.pi ** 2)) * (torch.atan(wt / ht) - torch.atan(wp / hp)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2 + alpha * v)


def df_loss(pred_dist: torch.Tensor, target: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Distribution Focal Loss para una de las cuatro distancias.

    pred_dist: (N, reg_max+1) logits de la distribución.
    target:    (N,) distancia real continua en unidades de celda, en [0, reg_max].
    """
    target = target.clamp(0, reg_max - 1e-3)
    tl = target.long()              # bin inferior
    tr = tl + 1                     # bin superior
    wl = tr.float() - target        # peso del bin inferior
    wr = 1 - wl                     # peso del bin superior
    loss = (
        F.cross_entropy(pred_dist, tl, reduction="none") * wl
        + F.cross_entropy(pred_dist, tr, reduction="none") * wr
    )
    return loss


class DetectionLoss(nn.Module):
    """Pérdida total del detector.

    Args:
        num_classes: clases de la taxonomía unificada.
        reg_max: bins de la distribución de cajas (debe coincidir con la cabeza).
        gains: pesos relativos (box, cls, dfl).
    """

    def __init__(
        self,
        num_classes: int = 5,
        reg_max: int = 16,
        gains: tuple = (7.5, 0.5, 1.5),
    ) -> None:
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.box_gain, self.cls_gain, self.dfl_gain = gains
        self.assigner = TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        cls_logits: torch.Tensor,    # (A, C)
        box_dist_logits: torch.Tensor,  # (A, 4*(reg_max+1))
        pred_boxes: torch.Tensor,    # (A, 4) cajas decodificadas (píxeles)
        points: torch.Tensor,        # (A, 2) centros de celda (píxeles)
        strides: torch.Tensor,       # (A, 1) stride por ancla
        gt_labels: torch.Tensor,     # (G,)
        gt_boxes: torch.Tensor,      # (G, 4) xyxy en píxeles
    ):
        device = cls_logits.device
        # El asignador compara cajas e IoU en píxeles, por lo que necesita los
        # centros de las celdas también en píxeles (points * stride). La pérdida
        # DFL, en cambio, trabaja en unidades de celda (más abajo).
        anchor_centers = points * strides
        target_labels, target_boxes, target_scores, fg_mask = self.assigner(
            cls_logits.sigmoid().detach(),
            pred_boxes.detach(),
            anchor_centers,
            gt_labels,
            gt_boxes,
        )
        target_sum = target_scores.sum().clamp(min=1.0)

        # --- Clasificación (todas las anclas) ---
        loss_cls = self.bce(cls_logits, target_scores).sum() / target_sum

        loss_box = torch.zeros(1, device=device)
        loss_dfl = torch.zeros(1, device=device)

        if fg_mask.any():
            pos = torch.where(fg_mask)[0]
            weight = target_scores[pos].sum(dim=1)

            # --- Localización (CIoU) ---
            ciou = bbox_ciou(pred_boxes[pos], target_boxes[pos])
            loss_box = ((1.0 - ciou) * weight).sum() / target_sum

            # --- DFL ---
            # Distancias objetivo (l, t, r, b) en unidades de celda.
            pts = points[pos]
            st = strides[pos]
            tgt = target_boxes[pos] / st  # a unidades de celda
            lt = pts - tgt[:, :2]
            rb = tgt[:, 2:] - pts
            tgt_ltrb = torch.cat((lt, rb), dim=1)  # (P, 4)

            dist_logits = box_dist_logits[pos].view(-1, 4, self.reg_max + 1)
            dfl = torch.zeros(pos.numel(), device=device)
            for side in range(4):
                dfl = dfl + df_loss(dist_logits[:, side, :], tgt_ltrb[:, side], self.reg_max)
            loss_dfl = (dfl * weight).sum() / target_sum

        total = self.box_gain * loss_box + self.cls_gain * loss_cls + self.dfl_gain * loss_dfl
        return total.squeeze(), {
            "box": float(loss_box.detach().mean()),
            "cls": float(loss_cls.detach().mean()),
            "dfl": float(loss_dfl.detach().mean()),
        }


if __name__ == "__main__":
    from .head import DetectHead, make_anchors

    torch.manual_seed(0)
    reg_max = 16
    head = DetectHead(num_classes=5, in_channels=(64, 128, 256), reg_max=reg_max)
    feats = [
        torch.randn(1, 64, 80, 80),
        torch.randn(1, 128, 40, 40),
        torch.randn(1, 256, 20, 20),
    ]
    head.eval()
    with torch.no_grad():
        pred_boxes, _ = head(feats)          # (1, 8400, 4) en píxeles
    pred_boxes = pred_boxes[0]
    points, strides = make_anchors(feats, head.strides)  # cells, stride
    A = points.shape[0]

    cls_logits = torch.randn(A, 5, requires_grad=True)
    box_dist = torch.randn(A, 4 * (reg_max + 1), requires_grad=True)
    gt_labels = torch.tensor([0, 2, 3])
    gt_boxes = torch.tensor(
        [[40, 40, 160, 360], [300, 100, 360, 200], [420, 420, 560, 620.0]]
    )

    crit = DetectionLoss(num_classes=5, reg_max=reg_max)
    loss, parts = crit(cls_logits, box_dist, pred_boxes, points, strides, gt_labels, gt_boxes)
    loss.backward()
    print("pérdida total:", round(float(loss.detach()), 4), "| componentes:", parts)
    print("positivos:", int((crit.assigner(cls_logits.sigmoid().detach(), pred_boxes.detach(), points * strides, gt_labels, gt_boxes)[3]).sum()))
    print("gradiente cls ok:", cls_logits.grad is not None, "| gradiente dfl ok:", box_dist.grad is not None)
