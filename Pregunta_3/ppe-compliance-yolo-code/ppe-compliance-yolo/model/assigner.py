"""
Asignador de etiquetas alineado con la tarea (Task-Aligned Assigner, TOOD).

Reimplementación didáctica de la asignación dinámica de etiquetas empleada por
los detectores modernos de una sola etapa. En lugar de reglas fijas basadas en
IoU con anclas, cada predicción recibe una *métrica de alineación*

    t = s^alpha * u^beta

donde `s` es la probabilidad de la clase correcta y `u` el IoU con la caja real.
Para cada objeto se seleccionan los `topk` candidatos con mayor `t` entre los
puntos-ancla cuyo centro cae dentro de la caja, y se normaliza el objetivo de
clasificación para que el pico coincida con el mayor IoU del objeto. Esto alinea
las dos ramas (clasificación y localización) y mejora notablemente el reciclaje
de positivos en objetos pequeños y ocluidos.

Referencia conceptual: Feng et al., "TOOD: Task-aligned One-stage Object
Detection", ICCV 2021.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bbox_iou_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU por pares entre dos conjuntos de cajas xyxy.

    boxes1: (N, 4), boxes2: (M, 4) -> (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter + 1e-9
    return inter / union


def select_candidates_in_gts(points: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Máscara (G, A) que indica qué centros de celda caen dentro de cada GT."""
    n_anchors = points.shape[0]
    g = gt_boxes.shape[0]
    pts = points.unsqueeze(0).expand(g, n_anchors, 2)
    x1y1 = gt_boxes[:, None, :2]
    x2y2 = gt_boxes[:, None, 2:]
    deltas = torch.cat((pts - x1y1, x2y2 - pts), dim=-1)  # (G, A, 4)
    return deltas.amin(dim=-1) > 1e-6


class TaskAlignedAssigner:
    """Asignador alineado con la tarea.

    Args:
        topk:  número de candidatos por objeto.
        alpha: exponente del término de clasificación.
        beta:  exponente del término de IoU.
        eps:   estabilizador numérico.
    """

    def __init__(self, topk: int = 10, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def __call__(
        self,
        pred_scores: torch.Tensor,   # (A, C) probabilidades por clase
        pred_boxes: torch.Tensor,    # (A, 4) cajas xyxy decodificadas (píxeles)
        points: torch.Tensor,        # (A, 2) centros de celda (píxeles)
        gt_labels: torch.Tensor,     # (G,) índice de clase de cada objeto
        gt_boxes: torch.Tensor,      # (G, 4) cajas reales xyxy
    ):
        """Devuelve los objetivos por ancla.

        target_labels: (A,) clase asignada (-1 si es fondo)
        target_boxes:  (A, 4) caja objetivo
        target_scores: (A, C) objetivo suave de clasificación (alineado con IoU)
        fg_mask:       (A,) booleano de positivos
        """
        num_anchors = pred_scores.shape[0]
        num_gt = gt_boxes.shape[0]
        device = pred_scores.device

        if num_gt == 0:
            return (
                torch.full((num_anchors,), -1, dtype=torch.long, device=device),
                torch.zeros((num_anchors, 4), device=device),
                torch.zeros_like(pred_scores),
                torch.zeros(num_anchors, dtype=torch.bool, device=device),
            )

        # IoU y score de la clase correcta por cada par (objeto, ancla).
        ious = bbox_iou_pairwise(gt_boxes, pred_boxes).clamp(0)        # (G, A)
        cls_scores = pred_scores[:, gt_labels].transpose(0, 1)         # (G, A)
        # Métrica de alineación.
        align_metric = cls_scores.pow(self.alpha) * ious.pow(self.beta)

        # Solo son candidatos los puntos dentro de la caja.
        in_gts = select_candidates_in_gts(points, gt_boxes)            # (G, A)
        align_metric = align_metric * in_gts

        # topk candidatos por objeto.
        topk = min(self.topk, align_metric.shape[1])
        _, topk_idx = align_metric.topk(topk, dim=1)
        cand_mask = torch.zeros_like(align_metric, dtype=torch.bool)
        cand_mask.scatter_(1, topk_idx, True)
        cand_mask &= in_gts

        # Resolver anclas reclamadas por varios objetos -> nos quedamos con el
        # de mayor IoU.
        overlap = cand_mask.float().sum(dim=0)                          # (A,)
        conflict = overlap > 1
        if conflict.any():
            best_gt = ious.argmax(dim=0)                                # (A,)
            new_mask = torch.zeros_like(cand_mask)
            new_mask[best_gt[conflict], torch.where(conflict)[0]] = True
            cand_mask = torch.where(conflict.unsqueeze(0), new_mask, cand_mask)

        fg_mask = cand_mask.any(dim=0)                                  # (A,)
        assigned_gt = cand_mask.float().argmax(dim=0)                   # (A,)

        target_labels = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
        target_labels[fg_mask] = gt_labels[assigned_gt[fg_mask]]
        target_boxes = gt_boxes[assigned_gt]

        # Normalización del objetivo suave: el pico por objeto se escala al
        # IoU máximo que ese objeto alcanza (alineación clasificación-IoU).
        align_per_gt = align_metric * cand_mask
        max_align = align_per_gt.amax(dim=1, keepdim=True)
        max_iou = (ious * cand_mask).amax(dim=1, keepdim=True)
        norm_align = (align_per_gt / (max_align + self.eps) * max_iou)  # (G, A)
        norm_align = norm_align.amax(dim=0)                             # (A,)

        target_scores = torch.zeros_like(pred_scores)
        pos_idx = torch.where(fg_mask)[0]
        target_scores[pos_idx, target_labels[pos_idx]] = norm_align[pos_idx]

        return target_labels, target_boxes, target_scores, fg_mask


if __name__ == "__main__":
    torch.manual_seed(0)
    A, C, G = 200, 5, 3
    points = torch.rand(A, 2) * 640
    pred_boxes = torch.cat([points - 10, points + 10], dim=1)
    pred_scores = torch.rand(A, C)
    gt_labels = torch.tensor([0, 1, 3])
    gt_boxes = torch.tensor([[50, 50, 120, 200], [300, 100, 340, 160], [400, 400, 500, 620.0]])
    assigner = TaskAlignedAssigner()
    tl, tb, ts, fg = assigner(pred_scores, pred_boxes, points, gt_labels, gt_boxes)
    print("positivos asignados:", int(fg.sum().item()))
    print("clases presentes:", torch.unique(tl[fg]).tolist())
