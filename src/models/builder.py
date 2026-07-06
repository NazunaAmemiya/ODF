"""Model builders."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.utils.registry import BACKBONES, DECODERS, HEADS, LOSSES, MODELS, NECKS, build_from_cfg

# Registry side effects.
from . import backbones as _backbones  # noqa: F401
from . import decoders as _decoders  # noqa: F401
from . import heads as _heads  # noqa: F401
from . import losses as _losses  # noqa: F401
from . import meta_archs as _meta_archs  # noqa: F401
from . import necks as _necks  # noqa: F401


def build_model(cfg: Dict[str, Any]):
    return build_from_cfg(cfg, MODELS)


def build_backbone(cfg: Dict[str, Any]):
    return build_from_cfg(cfg, BACKBONES)


def build_neck(cfg: Optional[Dict[str, Any]]):
    return build_from_cfg(cfg, NECKS)


def build_head(cfg: Dict[str, Any]):
    return build_from_cfg(cfg, HEADS)


def build_loss(cfg: Optional[Dict[str, Any]]):
    return build_from_cfg(cfg, LOSSES)


def build_decoder(cfg: Optional[Dict[str, Any]]):
    return build_from_cfg(cfg, DECODERS)
