"""Learning-rate scheduler builders."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)


class LinearWarmupCosineLR(LambdaLR):
    """Linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int = 0,
        max_iters: int = 100,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_iters = max(int(warmup_iters), 0)
        self.max_iters = max(int(max_iters), 1)
        self.min_lr_ratio = float(min_lr_ratio)
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, step: int) -> float:
        if self.warmup_iters > 0 and step < self.warmup_iters:
            return float(step + 1) / float(self.warmup_iters)
        progress = (step - self.warmup_iters) / max(self.max_iters - self.warmup_iters, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    steps_per_epoch: Optional[int] = None,
):
    """Build a scheduler from config.

    Supported ``type`` values: ``CosineAnnealingLR``, ``StepLR``,
    ``MultiStepLR``, ``ReduceLROnPlateau``, ``OneCycleLR``,
    ``LinearWarmupCosineLR``, and ``None``.
    """

    cfg = dict(cfg or {})
    scheduler_type = str(cfg.pop("type", "CosineAnnealingLR"))
    if scheduler_type.lower() in {"none", "null", "false"}:
        return None

    if scheduler_type == "StepLR":
        return StepLR(
            optimizer,
            step_size=int(cfg.pop("step_size", 30)),
            gamma=float(cfg.pop("gamma", 0.1)),
            **cfg,
        )
    if scheduler_type == "MultiStepLR":
        return MultiStepLR(
            optimizer,
            milestones=list(cfg.pop("milestones", [60, 90])),
            gamma=float(cfg.pop("gamma", 0.1)),
            **cfg,
        )
    if scheduler_type == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=str(cfg.pop("mode", "min")),
            factor=float(cfg.pop("factor", 0.1)),
            patience=int(cfg.pop("patience", 10)),
            **cfg,
        )
    if scheduler_type == "OneCycleLR":
        if steps_per_epoch is not None:
            cfg.setdefault("steps_per_epoch", steps_per_epoch)
        cfg.setdefault("max_lr", optimizer.param_groups[0]["lr"])
        cfg.setdefault("epochs", 100)
        return OneCycleLR(optimizer, **cfg)
    if scheduler_type == "LinearWarmupCosineLR":
        return LinearWarmupCosineLR(
            optimizer,
            warmup_iters=int(cfg.pop("warmup_iters", 0)),
            max_iters=int(cfg.pop("max_iters", cfg.pop("T_max", 100))),
            min_lr_ratio=float(cfg.pop("min_lr_ratio", 0.0)),
            **cfg,
        )
    return CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.pop("T_max", 100)),
        eta_min=float(cfg.pop("eta_min", 0.0)),
        **cfg,
    )
