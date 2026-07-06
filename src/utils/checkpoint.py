"""Checkpoint save/load helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import torch


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a checkpoint with enough state to resume training."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {"state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if metrics is not None:
        payload["metrics"] = metrics
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_state_dict_flexible(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> Any:
    """Load model weights while tolerating DataParallel's ``module.`` prefix."""

    try:
        return model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        stripped = {
            key.replace("module.", "", 1) if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
        return model.load_state_dict(stripped, strict=strict)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    map_location: Union[str, torch.device] = "cpu",
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load checkpoint and optionally restore optimizer/scheduler state."""

    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    load_state_dict_flexible(model, state_dict, strict=strict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint

