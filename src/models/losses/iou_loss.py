"""IoU-based losses and YOLO detection loss config container."""

from __future__ import annotations

import torch
from torch import nn

from src.utils.metrics import box_iou
from src.utils.registry import LOSSES


def bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Aligned IoU for xyxy boxes."""

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((0,))
    ious = box_iou(boxes1, boxes2)
    if ious.shape[0] == ious.shape[1]:
        return ious.diag()
    return ious.max(dim=1).values


@LOSSES.register_module(name=["IoULoss", "GIoULoss", "CIoULoss"])
class IoULoss(nn.Module):
    """Simple 1 - IoU loss for aligned boxes."""

    def __init__(self, reduction: str = "mean", loss_weight: float = 1.0, **kwargs) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = float(loss_weight)
        self.extra_cfg = kwargs

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 1.0 - bbox_iou(pred, target)
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            return loss * self.loss_weight
        else:
            loss = loss.mean() if loss.numel() else pred.sum() * 0.0
        return loss * self.loss_weight


@LOSSES.register_module(name=["YOLOv8Loss"])
class YOLOv8Loss(nn.Module):
    """Configurable weights for the compact YOLO-style head loss."""

    def __init__(
        self,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        obj_weight: float = 1.0,
        dfl_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.box_weight = float(box_weight)
        self.cls_weight = float(cls_weight)
        self.obj_weight = float(obj_weight)
        self.dfl_weight = float(dfl_weight)
        self.extra_cfg = kwargs

    def forward(self, loss_dict):
        total = 0.0
        for value in loss_dict.values():
            total = total + value
        return total
