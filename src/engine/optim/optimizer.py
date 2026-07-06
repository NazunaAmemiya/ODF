"""Optimizer builders for config-driven training."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch
from torch import nn
from torch.optim import Adam, AdamW, RMSprop, SGD


def _is_norm_layer(module: nn.Module) -> bool:
    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.LayerNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )
    return isinstance(module, norm_types)


def build_param_groups(
    model: nn.Module,
    weight_decay: float,
    bias_decay: float = 0.0,
    norm_decay: float = 0.0,
) -> List[Dict[str, Any]]:
    """Create parameter groups with lighter decay for bias/norm parameters."""

    decay_params = []
    bias_params = []
    norm_params = []

    module_by_param = {}
    for module in model.modules():
        for param in module.parameters(recurse=False):
            module_by_param[id(param)] = module

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        module = module_by_param.get(id(param))
        if name.endswith(".bias"):
            bias_params.append(param)
        elif module is not None and _is_norm_layer(module):
            norm_params.append(param)
        else:
            decay_params.append(param)

    groups: List[Dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if bias_params:
        groups.append({"params": bias_params, "weight_decay": bias_decay})
    if norm_params:
        groups.append({"params": norm_params, "weight_decay": norm_decay})
    return groups


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build optimizer from config.

    Supported ``type`` values: ``SGD``, ``Adam``, ``AdamW``, ``RMSprop``.
    """

    cfg = dict(cfg or {})
    optim_type = str(cfg.pop("type", "AdamW"))
    lr = float(cfg.pop("lr", 1e-3))
    weight_decay = float(cfg.pop("weight_decay", 5e-4))
    paramwise_cfg = cfg.pop("paramwise_cfg", None)

    if paramwise_cfg:
        params: Iterable[Dict[str, Any]] = build_param_groups(
            model,
            weight_decay=weight_decay,
            bias_decay=float(paramwise_cfg.get("bias_decay", 0.0)),
            norm_decay=float(paramwise_cfg.get("norm_decay", 0.0)),
        )
        optimizer_weight_decay = 0.0
    else:
        params = (p for p in model.parameters() if p.requires_grad)
        optimizer_weight_decay = weight_decay

    optim_key = optim_type.lower()
    if optim_key == "sgd":
        return SGD(
            params,
            lr=lr,
            momentum=float(cfg.pop("momentum", 0.9)),
            nesterov=bool(cfg.pop("nesterov", False)),
            weight_decay=optimizer_weight_decay,
            **cfg,
        )
    if optim_key == "adam":
        return Adam(params, lr=lr, weight_decay=optimizer_weight_decay, **cfg)
    if optim_key == "adamw":
        return AdamW(params, lr=lr, weight_decay=optimizer_weight_decay, **cfg)
    if optim_key == "rmsprop":
        return RMSprop(
            params,
            lr=lr,
            momentum=float(cfg.pop("momentum", 0.0)),
            weight_decay=optimizer_weight_decay,
            **cfg,
        )
    raise ValueError(f"Unsupported optimizer: {optim_type}")
