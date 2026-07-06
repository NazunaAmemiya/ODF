"""ResNet backbone wrapper."""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from src.utils.registry import BACKBONES


@BACKBONES.register_module(name=["ResNet", "ResNetBackbone"])
class ResNetBackbone(nn.Module):
    """Feature extractor that exposes C2-C5 maps from torchvision ResNet."""

    def __init__(
        self,
        depth: int = 50,
        pretrained: bool = False,
        frozen_stages: int = -1,
        in_channels: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18, resnet34, resnet50, resnet101
        except Exception as exc:
            raise ImportError("torchvision is required for ResNetBackbone") from exc

        builders = {18: resnet18, 34: resnet34, 50: resnet50, 101: resnet101}
        if depth not in builders:
            raise ValueError("Supported ResNet depths are 18, 34, 50, and 101.")

        # Avoid implicit network downloads. Users can load pretrained weights via checkpoint.
        model = builders[depth](weights=None)
        if in_channels != 3:
            old = model.conv1
            model.conv1 = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.out_channels = [64, 128, 256, 512] if depth in {18, 34} else [256, 512, 1024, 2048]
        self.pretrained = pretrained
        self.extra_cfg = kwargs
        self._freeze_stages(frozen_stages)

    def _freeze_stages(self, frozen_stages: int) -> None:
        modules = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in modules[: max(frozen_stages + 1, 0)]:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]
