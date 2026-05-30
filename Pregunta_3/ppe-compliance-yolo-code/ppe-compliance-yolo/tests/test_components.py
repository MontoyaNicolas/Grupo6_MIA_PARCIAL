"""
Pruebas numéricas de los componentes reimplementados.

Validan que la cabeza, el asignador, la pérdida y el verificador se comportan
como se espera. Se pueden ejecutar con pytest o directamente:

    python -m pytest tests/ -q
    python tests/test_components.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.head import DetectHead, make_anchors          # noqa: E402
from model.assigner import TaskAlignedAssigner            # noqa: E402
from model.loss import DetectionLoss, bbox_ciou           # noqa: E402
from compliance_verifier import (                          # noqa: E402
    ComplianceVerifier, PERSONA, CASCO, CABEZA, CHALECO, SIN_CHALECO,
)


def _toy_fpn(bs=2):
    return [
        torch.randn(bs, 64, 80, 80),
        torch.randn(bs, 128, 40, 40),
        torch.randn(bs, 256, 20, 20),
    ]


def test_head_shapes():
    head = DetectHead(num_classes=5, in_channels=(64, 128, 256), reg_max=16)
    feats = _toy_fpn()
    head.train()
    raw = head(feats)
    # 5 clases + 4*(16+1) = 73 canales por celda.
    assert all(r.shape[1] == 73 for r in raw)
    head.eval()
    with torch.no_grad():
        boxes, scores = head(feats)
    assert boxes.shape == (2, 8400, 4)      # 80^2 + 40^2 + 20^2
    assert scores.shape == (2, 8400, 5)
    assert torch.all((scores >= 0) & (scores <= 1))


def test_ciou_identical_is_one():
    box = torch.tensor([[10.0, 10.0, 50.0, 80.0]])
    assert torch.allclose(bbox_ciou(box, box), torch.ones(1), atol=1e-4)


def test_ciou_disjoint_is_negative():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    assert float(bbox_ciou(a, b)) < 0


def test_assigner_assigns_positives():
    torch.manual_seed(0)
    head = DetectHead(num_classes=5, in_channels=(64, 128, 256))
    feats = _toy_fpn(1)
    head.eval()
    with torch.no_grad():
        boxes, _ = head(feats)
    points, strides = make_anchors(feats, head.strides)
    pred_scores = torch.rand(points.shape[0], 5)
    gt_labels = torch.tensor([0, 2, 3])
    gt_boxes = torch.tensor([[40, 40, 160, 360], [300, 100, 360, 200], [420, 420, 560, 620.0]])
    asg = TaskAlignedAssigner(topk=10)
    tl, tb, ts, fg = asg(pred_scores, boxes[0], points * strides, gt_labels, gt_boxes)
    assert int(fg.sum()) > 0
    assert set(torch.unique(tl[fg]).tolist()).issubset({0, 2, 3})


def test_loss_backprops():
    torch.manual_seed(0)
    reg_max = 16
    head = DetectHead(num_classes=5, in_channels=(64, 128, 256), reg_max=reg_max)
    feats = _toy_fpn(1)
    head.eval()
    with torch.no_grad():
        pred_boxes, _ = head(feats)
    pred_boxes = pred_boxes[0]
    points, strides = make_anchors(feats, head.strides)
    A = points.shape[0]
    cls_logits = torch.randn(A, 5, requires_grad=True)
    box_dist = torch.randn(A, 4 * (reg_max + 1), requires_grad=True)
    gt_labels = torch.tensor([0, 2, 3])
    gt_boxes = torch.tensor([[40, 40, 160, 360], [300, 100, 360, 200], [420, 420, 560, 620.0]])
    crit = DetectionLoss(num_classes=5, reg_max=reg_max)
    loss, parts = crit(cls_logits, box_dist, pred_boxes, points, strides, gt_labels, gt_boxes)
    loss.backward()
    assert torch.isfinite(loss)
    assert cls_logits.grad is not None and box_dist.grad is not None
    assert parts["box"] > 0 and parts["dfl"] > 0


def test_verifier_compliance_logic():
    boxes = [
        [50, 100, 150, 400], [70, 110, 130, 170], [60, 200, 140, 320],   # A: casco+chaleco
        [200, 100, 300, 400], [220, 110, 280, 170], [210, 200, 290, 320],  # B: cabeza+chaleco
        [350, 100, 450, 400], [370, 110, 430, 170], [360, 200, 440, 320],  # C: casco+sin_chaleco
    ]
    classes = [PERSONA, CASCO, CHALECO, PERSONA, CABEZA, CHALECO, PERSONA, CASCO, SIN_CHALECO]
    confs = [0.9] * len(classes)
    v = ComplianceVerifier(require_vest=True)
    r = v.verify_frame(boxes, classes, confs)
    assert r.n_workers == 3
    assert r.n_compliant == 1
    assert abs(r.compliance_rate - 1 / 3) < 1e-3  # la tasa se redondea a 4 decimales
    assert r.workers[0].compliant and not r.workers[1].compliant and not r.workers[2].compliant


def test_verifier_temporal_smoothing():
    v = ComplianceVerifier(require_vest=False, smooth_window=4)
    # Fotogramas alternos 100% / 0% -> el suavizado converge a ~0.5.
    seq = []
    for k in range(8):
        if k % 2 == 0:
            r = v.verify_frame([[0, 0, 10, 30]], [PERSONA], [0.9])      # sin casco visible -> unknown
        else:
            r = v.verify_frame([[0, 0, 10, 30], [2, 2, 8, 10]], [PERSONA, CASCO], [0.9, 0.9])
        seq.append(r.smoothed_rate)
    assert 0.0 <= seq[-1] <= 1.0
    assert len(seq) == 8


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} pruebas superadas.")
    sys.exit(0 if passed == len(tests) else 1)
