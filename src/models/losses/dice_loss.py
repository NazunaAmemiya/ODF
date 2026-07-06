"""Dice and segmentation losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.registry import LOSSES


@LOSSES.register_module(name=["DiceLoss"])
class DiceLoss(nn.Module):
    """Soft Dice loss for binary masks."""

    def __init__(self, eps: float = 1e-6, loss_weight: float = 1.0, **kwargs) -> None:
        super().__init__()
        self.eps = float(eps)
        self.loss_weight = float(loss_weight)
        self.extra_cfg = kwargs

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.sigmoid() if pred.min() < 0 or pred.max() > 1 else pred
        target = target.float()
        dims = tuple(range(1, pred.ndim))
        inter = (pred * target).sum(dim=dims)
        denom = pred.sum(dim=dims) + target.sum(dim=dims)
        loss = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)
        return loss.mean() * self.loss_weight


@LOSSES.register_module(name=["BCEDiceLoss"])
class BCEDiceLoss(nn.Module):
    """BCEWithLogits + Dice for binary masks."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, **kwargs) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.dice = DiceLoss()
        self.extra_cfg = kwargs

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target.float())
        dice = self.dice(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


@LOSSES.register_module(name=["YOLOv8SegLoss"])
class YOLOv8SegLoss(nn.Module):
    """Configurable weights for compact YOLO segmentation loss."""

    def __init__(
        self,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        obj_weight: float = 1.0,
        dfl_weight: float = 0.0,
        mask_weight: float = 1.0,
        overlap_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.box_weight = float(box_weight)
        self.cls_weight = float(cls_weight)
        self.obj_weight = float(obj_weight)
        self.dfl_weight = float(dfl_weight)
        self.mask_weight = float(mask_weight)
        self.overlap_weight = float(overlap_weight)
        self.extra_cfg = kwargs

    def forward(self, loss_dict):
        total = 0.0
        for value in loss_dict.values():
            total = total + value
        return total
