"""Segmentation metrics."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from src.utils.metrics import dice_score, mask_iou
from src.utils.registry import METRICS


@METRICS.register_module(name=["SegmentationMetric", "SegMetric"])
class SegmentationMetric:
    """Foreground mask mIoU and Dice metric."""

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        self.threshold = float(threshold)
        self.extra_cfg = kwargs
        self.reset()

    def reset(self) -> None:
        self.iou_sum = 0.0
        self.dice_sum = 0.0
        self.count = 0

    def update(self, predictions: List[Dict[str, torch.Tensor]], targets: Dict[str, torch.Tensor]) -> None:
        gt_masks = targets.get("gt_masks")
        valid = targets.get("valid_mask")
        if gt_masks is None:
            return
        for idx, pred in enumerate(predictions):
            target = gt_masks[idx]
            if valid is not None and valid.shape[1] > 0:
                target = target * valid[idx].float()[:, None, None]
            target = target.sum(dim=0, keepdim=True).clamp(0, 1)
            pred_mask = pred.get("masks")
            if pred_mask is None:
                pred_mask = torch.zeros_like(target)
            pred_mask = pred_mask.detach().float().cpu()
            target = target.detach().float().cpu()
            if pred_mask.ndim == 3:
                pred_mask = pred_mask[:1]
            if pred_mask.shape[-2:] != target.shape[-2:]:
                pred_mask = F.interpolate(pred_mask[None], size=target.shape[-2:], mode="nearest")[0]
            pred_binary = pred_mask >= self.threshold
            self.iou_sum += float(mask_iou(pred_binary, target.bool()).mean().item())
            self.dice_sum += float(dice_score(pred_binary.float(), target.float()).mean().item())
            self.count += 1

    def compute(self) -> Dict[str, float]:
        denom = max(self.count, 1)
        return {
            "mIoU": self.iou_sum / denom,
            "Dice": self.dice_sum / denom,
            "num_samples": float(self.count),
        }
