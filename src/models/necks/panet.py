"""PANet / YOLOv8 PAFPN neck."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.backbones.csp_darknet import ConvBNAct
from src.models.necks.fpn import FPN
from src.utils.registry import NECKS


@NECKS.register_module(name=["PANet", "YOLOv8PAFPN"])
class PANet(FPN):
    """FPN with a small bottom-up path aggregation stage."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 128,
        num_outs: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels, num_outs=num_outs, **kwargs)
        self.downsample_convs = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels, 3, 2) for _ in range(max(num_outs - 1, 0))]
        )
        self.pan_convs = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels, 3) for _ in range(max(num_outs - 1, 0))]
        )
        self.out_channels = [out_channels] * int(num_outs)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        outs = super().forward(inputs)
        for idx in range(len(outs) - 1):
            down = self.downsample_convs[idx](outs[idx])
            if down.shape[-2:] != outs[idx + 1].shape[-2:]:
                down = F.interpolate(down, size=outs[idx + 1].shape[-2:], mode="nearest")
            outs[idx + 1] = self.pan_convs[idx](outs[idx + 1] + down)
        return outs
