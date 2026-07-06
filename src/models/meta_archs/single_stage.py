"""Single-stage model architectures."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
from torch import nn

from src.models.meta_archs.base_model import BaseModel
from src.utils.registry import BACKBONES, DECODERS, HEADS, LOSSES, MODELS, NECKS, build_from_cfg


@MODELS.register_module(name=["SingleStageDetector"])
class SingleStageDetector(BaseModel):
    """Backbone + neck + head + decoder model for detection."""

    default_decoder_type = "DetDecoder"

    def __init__(
        self,
        backbone: Dict[str, Any],
        neck: Optional[Dict[str, Any]] = None,
        head: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        decoder: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.backbone = build_from_cfg(backbone, BACKBONES)

        backbone_channels = getattr(self.backbone, "out_channels", None)
        if neck is not None:
            neck_cfg = copy.deepcopy(neck)
            if backbone_channels is not None:
                neck_cfg.setdefault("in_channels", backbone_channels)
            self.neck = build_from_cfg(neck_cfg, NECKS)
            feature_channels = getattr(self.neck, "out_channels", backbone_channels)
        else:
            self.neck = nn.Identity()
            feature_channels = backbone_channels

        if head is None:
            head = {"type": "YOLOv8DetHead"}
        head_cfg = copy.deepcopy(head)
        if feature_channels is not None:
            head_cfg.setdefault("in_channels", feature_channels)
        self.head = build_from_cfg(head_cfg, HEADS)

        self.loss_module = build_from_cfg(loss, LOSSES) if loss else None
        if decoder is None:
            decoder = {"type": self.default_decoder_type}
        self.decoder = build_from_cfg(decoder, DECODERS)
        self.extra_cfg = kwargs

    def extract_feat(self, images: torch.Tensor):
        features = self.backbone(images)
        if isinstance(features, torch.Tensor):
            features = [features]
        features = self.neck(features)
        if isinstance(features, torch.Tensor):
            features = [features]
        return features

    def forward_head(self, images: torch.Tensor):
        return self.head(self.extract_feat(images))

    def forward_train(self, images: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.forward_head(images)
        return self.head.loss(outputs, targets, loss_module=self.loss_module)

    def forward_test(self, images: torch.Tensor):
        outputs = self.forward_head(images)
        image_size = images.shape[-2:]
        return self.decoder(outputs, image_size=image_size)


@MODELS.register_module(name=["SingleStageSegmentor"])
class SingleStageSegmentor(SingleStageDetector):
    """Single-stage segmentation model using a segmentation head/decoder."""

    default_decoder_type = "SegDecoder"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
