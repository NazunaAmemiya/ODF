"""Base model interfaces."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn


class BaseModel(nn.Module):
    """Base class for framework models."""

    def forward_train(self, images: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forward_test(self, images: torch.Tensor) -> Any:
        raise NotImplementedError

    def forward(self, images: torch.Tensor, targets: Optional[Dict[str, torch.Tensor]] = None) -> Any:
        if self.training and targets is not None:
            return self.forward_train(images, targets)
        return self.forward_test(images)
