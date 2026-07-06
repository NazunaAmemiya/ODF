"""Compact CSP/YOLO-style backbones."""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn

from src.utils.registry import BACKBONES


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + SiLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Bottleneck(nn.Module):
    """Small residual bottleneck used inside CSP blocks."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 8)
        self.conv1 = ConvBNAct(channels, hidden, 1)
        self.conv2 = ConvBNAct(hidden, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class C2f(nn.Module):
    """Simplified YOLOv8 C2f block."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1) -> None:
        super().__init__()
        self.reduce = ConvBNAct(in_channels, out_channels, 1)
        self.blocks = nn.Sequential(*[Bottleneck(out_channels) for _ in range(num_blocks)])
        self.out = ConvBNAct(out_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(self.reduce(x)))


@BACKBONES.register_module(name=["YOLOv8Backbone", "CSPDarknet", "CSPDarknetTiny"])
class YOLOv8Backbone(nn.Module):
    """A compact YOLO-style feature extractor.

    It returns three feature maps with channels ``[64, 128, 256]`` by default,
    matching the repository YOLO configs.
    """

    def __init__(
        self,
        version: str = "n",
        in_channels: int = 3,
        out_channels: Sequence[int] = (64, 128, 256),
        pretrained: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        width_mul = {"n": 1.0, "s": 1.25, "m": 1.5, "l": 2.0, "x": 2.5}.get(version, 1.0)
        c1, c2, c3 = [max(int(c * width_mul), 16) for c in out_channels]
        self.out_channels = [c1, c2, c3]
        stem_channels = max(c1 // 2, 16)
        self.stem = ConvBNAct(in_channels, stem_channels, 3, 2)
        self.stage1 = nn.Sequential(ConvBNAct(stem_channels, c1, 3, 2), C2f(c1, c1, 1))
        self.stage2 = nn.Sequential(ConvBNAct(c1, c2, 3, 2), C2f(c2, c2, 2))
        self.stage3 = nn.Sequential(ConvBNAct(c2, c3, 3, 2), C2f(c3, c3, 2))
        self.pretrained = pretrained
        self.extra_cfg = kwargs

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return [c3, c4, c5]
