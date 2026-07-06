"""Segmentation decoder."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from src.models.decoders.det_decoder import DetDecoder
from src.utils.registry import DECODERS


@DECODERS.register_module(name=["SegDecoder", "YOLOv8SegDecoder"])
class SegDecoder(DetDecoder):
    """Decode detections and a semantic foreground mask."""

    def __init__(self, mask_threshold: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask_threshold = float(mask_threshold)

    def decode(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        image_size: Sequence[int],
        conf_thres: float = None,
        iou_thres: float = None,
    ):
        detections = super().decode(outputs, image_size, conf_thres, iou_thres)
        mask_logits = outputs.get("mask_logits")
        if mask_logits is None:
            return detections

        image_h, image_w = int(image_size[0]), int(image_size[1])
        masks = F.interpolate(mask_logits, size=(image_h, image_w), mode="bilinear", align_corners=False)
        masks = masks.sigmoid() >= self.mask_threshold
        for idx, det in enumerate(detections):
            det["masks"] = masks[idx]
        return detections
