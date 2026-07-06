"""Compact YOLO-style detection head."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.backbones.csp_darknet import ConvBNAct
from src.utils.registry import HEADS


@HEADS.register_module(name=["YOLOv8DetHead", "YoloDetHead"])
class YOLOv8DetHead(nn.Module):
    """Anchor-free detection head used by the framework baseline."""

    def __init__(
        self,
        num_classes: int = 1,
        in_channels: Sequence[int] = (128, 128, 128),
        feat_channels: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = list(in_channels)
        self.feat_channels = int(feat_channels)
        self.extra_cfg = kwargs
        self.stems = nn.ModuleList([ConvBNAct(c, self.feat_channels, 3) for c in self.in_channels])
        self.cls_convs = nn.ModuleList([nn.Conv2d(self.feat_channels, self.num_classes, 1) for _ in self.in_channels])
        self.obj_convs = nn.ModuleList([nn.Conv2d(self.feat_channels, 1, 1) for _ in self.in_channels])
        self.box_convs = nn.ModuleList([nn.Conv2d(self.feat_channels, 4, 1) for _ in self.in_channels])
        self.last_loss_dict: Dict[str, torch.Tensor] = {}

    def forward(self, features: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        cls_logits: List[torch.Tensor] = []
        obj_logits: List[torch.Tensor] = []
        box_reg: List[torch.Tensor] = []
        for feature, stem, cls_conv, obj_conv, box_conv in zip(
            features, self.stems, self.cls_convs, self.obj_convs, self.box_convs
        ):
            hidden = stem(feature)
            cls_logits.append(cls_conv(hidden))
            obj_logits.append(obj_conv(hidden))
            box_reg.append(box_conv(hidden))
        return {"cls_logits": cls_logits, "obj_logits": obj_logits, "box_reg": box_reg}

    def loss(self, outputs: Dict[str, List[torch.Tensor]], targets: Dict[str, torch.Tensor], loss_module: Any = None) -> torch.Tensor:
        box_weight = float(getattr(loss_module, "box_weight", 7.5))
        cls_weight = float(getattr(loss_module, "cls_weight", 0.5))
        obj_weight = float(getattr(loss_module, "obj_weight", 1.0))

        loss_obj = outputs["obj_logits"][0].sum() * 0.0
        loss_cls = outputs["obj_logits"][0].sum() * 0.0
        loss_box = outputs["obj_logits"][0].sum() * 0.0
        level_count = max(len(outputs["obj_logits"]), 1)

        for cls_logits, obj_logits, box_reg in zip(
            outputs["cls_logits"], outputs["obj_logits"], outputs["box_reg"]
        ):
            obj_target, cls_target, box_target, pos_mask = self._make_targets_for_level(
                cls_logits, targets
            )
            loss_obj = loss_obj + F.binary_cross_entropy_with_logits(obj_logits, obj_target)
            if pos_mask.any():
                cls_pred = cls_logits.permute(0, 2, 3, 1)[pos_mask]
                cls_true = cls_target.permute(0, 2, 3, 1)[pos_mask]
                box_pred = box_reg.sigmoid().permute(0, 2, 3, 1)[pos_mask]
                box_true = box_target.permute(0, 2, 3, 1)[pos_mask]
                loss_cls = loss_cls + F.binary_cross_entropy_with_logits(cls_pred, cls_true)
                loss_box = loss_box + F.smooth_l1_loss(box_pred, box_true)

        loss_obj = loss_obj / level_count
        loss_cls = loss_cls / level_count
        loss_box = loss_box / level_count
        total = obj_weight * loss_obj + cls_weight * loss_cls + box_weight * loss_box
        self.last_loss_dict = {
            "loss_obj": loss_obj.detach(),
            "loss_cls": loss_cls.detach(),
            "loss_box": loss_box.detach(),
            "loss_total": total.detach(),
        }
        return total

    def _make_targets_for_level(
        self,
        cls_logits: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ):
        device = cls_logits.device
        batch_size, _, feat_h, feat_w = cls_logits.shape
        obj_target = torch.zeros((batch_size, 1, feat_h, feat_w), device=device)
        cls_target = torch.zeros((batch_size, self.num_classes, feat_h, feat_w), device=device)
        box_target = torch.zeros((batch_size, 4, feat_h, feat_w), device=device)
        pos_mask = torch.zeros((batch_size, feat_h, feat_w), dtype=torch.bool, device=device)

        boxes = targets.get("gt_bboxes")
        labels = targets.get("gt_labels")
        valid = targets.get("valid_mask")
        img_shape = targets.get("img_shape")
        if boxes is None or labels is None or boxes.numel() == 0:
            return obj_target, cls_target, box_target, pos_mask

        for b in range(batch_size):
            if boxes.shape[1] == 0:
                continue
            valid_b = valid[b] if valid is not None else labels[b] >= 0
            for obj_idx in torch.where(valid_b)[0].tolist():
                box = boxes[b, obj_idx].to(device).float()
                label = int(labels[b, obj_idx].item())
                if label < 0 or label >= self.num_classes:
                    label = 0
                if img_shape is not None:
                    img_h = max(float(img_shape[b, 0].item()), 1.0)
                    img_w = max(float(img_shape[b, 1].item()), 1.0)
                else:
                    img_h = float(feat_h)
                    img_w = float(feat_w)
                cx = ((box[0] + box[2]) / 2.0).clamp(0, img_w - 1)
                cy = ((box[1] + box[3]) / 2.0).clamp(0, img_h - 1)
                bw = (box[2] - box[0]).clamp(min=1.0, max=img_w)
                bh = (box[3] - box[1]).clamp(min=1.0, max=img_h)
                gx = int(torch.clamp(cx / img_w * feat_w, 0, feat_w - 1).item())
                gy = int(torch.clamp(cy / img_h * feat_h, 0, feat_h - 1).item())
                obj_target[b, 0, gy, gx] = 1.0
                cls_target[b, label, gy, gx] = 1.0
                box_target[b, :, gy, gx] = torch.stack([cx / img_w, cy / img_h, bw / img_w, bh / img_h])
                pos_mask[b, gy, gx] = True
        return obj_target, cls_target, box_target, pos_mask
