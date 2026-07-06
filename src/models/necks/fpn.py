"""Feature Pyramid Network."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.backbones.csp_darknet import ConvBNAct
from src.utils.registry import NECKS


@NECKS.register_module(name=["FPN"])
class FPN(nn.Module):
    """Standard top-down feature pyramid."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 128,
        num_outs: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = [out_channels] * int(num_outs)
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1) for c in self.in_channels]
        )
        self.output_convs = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels, 3) for _ in self.in_channels]
        )
        self.num_outs = int(num_outs)
        self.extra_cfg = kwargs

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(inputs) != len(self.in_channels):
            raise ValueError(f"FPN expected {len(self.in_channels)} inputs, got {len(inputs)}")

        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        for idx in range(len(laterals) - 1, 0, -1):
            laterals[idx - 1] = laterals[idx - 1] + F.interpolate(
                laterals[idx], size=laterals[idx - 1].shape[-2:], mode="nearest"
            )

        outs = [conv(x) for conv, x in zip(self.output_convs, laterals)]
        while len(outs) < self.num_outs:
            outs.append(F.max_pool2d(outs[-1], kernel_size=1, stride=2))
        return outs[: self.num_outs]
