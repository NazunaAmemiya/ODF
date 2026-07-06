"""Common metric primitives shared by detection and segmentation metrics."""

from __future__ import annotations

from typing import Tuple

import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    boxes = boxes.float()
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for boxes in xyxy format."""

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def mask_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """IoU for binary masks, reduced per mask in the batch."""

    pred = pred.bool()
    target = target.bool()
    dims = tuple(range(1, pred.ndim))
    inter = (pred & target).sum(dim=dims).float()
    union = (pred | target).sum(dim=dims).float()
    return inter / union.clamp(min=eps)


def dice_score(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Dice score for binary masks, reduced per mask in the batch."""

    pred = pred.float()
    target = target.float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return (2 * inter + eps) / (denom + eps)


def precision_recall(
    tp: int,
    fp: int,
    fn: int,
    eps: float = 1e-9,
) -> Tuple[float, float, float]:
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    f1 = 2 * precision * recall / max(precision + recall, eps)
    return precision, recall, f1
