"""Compact YOLO-style segmentation head."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.backbones.csp_darknet import ConvBNAct
from src.models.heads.yolo_det_head import YOLOv8DetHead
from src.utils.registry import HEADS


@HEADS.register_module(name=["YOLOv8SegHead", "YoloSegHead"])
class YOLOv8SegHead(YOLOv8DetHead):
    """Detection head with a semantic foreground mask branch."""

    def __init__(
        self,
        num_classes: int = 1,
        num_masks: int = 32,
        in_channels: Sequence[int] = (128, 128, 128),
        feat_channels: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes=num_classes, in_channels=in_channels, feat_channels=feat_channels, **kwargs)
        first_channels = self.in_channels[0]
        self.num_masks = int(num_masks)
        self.mask_head = nn.Sequential(
            ConvBNAct(first_channels, feat_channels, 3),
            ConvBNAct(feat_channels, feat_channels, 3),
            nn.Conv2d(feat_channels, 1, 1),
        )

    def forward(self, features: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        outputs = super().forward(features)
        outputs["mask_logits"] = self.mask_head(features[0])
        return outputs

    def loss(self, outputs: Dict[str, List[torch.Tensor]], targets: Dict[str, torch.Tensor], loss_module: Any = None) -> torch.Tensor:
        det_loss = super().loss(outputs, targets, loss_module=loss_module)
        mask_logits = outputs["mask_logits"]
        target_mask = self._make_mask_target(mask_logits, targets)
        mask_bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
        prob = mask_logits.sigmoid()
        dims = tuple(range(1, prob.ndim))
        inter = (prob * target_mask).sum(dim=dims)
        denom = prob.sum(dim=dims) + target_mask.sum(dim=dims)
        dice = 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)
        mask_loss = mask_bce + dice.mean()
        mask_weight = float(getattr(loss_module, "mask_weight", 1.0))
        total = det_loss + mask_weight * mask_loss
        self.last_loss_dict.update(
            {
                "loss_mask": mask_loss.detach(),
                "loss_total": total.detach(),
            }
        )
        return total

    def _make_mask_target(self, mask_logits: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        gt_masks = targets.get("gt_masks")
        if gt_masks is None or gt_masks.numel() == 0:
            return torch.zeros_like(mask_logits)
        gt_masks = gt_masks.to(mask_logits.device).float()
        valid = targets.get("valid_mask")
        if valid is not None and valid.numel() > 0:
            gt_masks = gt_masks * valid.to(mask_logits.device).float()[:, :, None, None]
        union = gt_masks.sum(dim=1, keepdim=True).clamp(0, 1)
        return F.interpolate(union, size=mask_logits.shape[-2:], mode="nearest")
