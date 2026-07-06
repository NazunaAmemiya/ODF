"""Dataset and dataloader builders."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional

from torch.utils.data import DataLoader

from src.datasets.transforms import Compose, Normalize, Resize, ToTensor, pad_collate_fn
from src.utils.registry import DATASETS, TRANSFORMS, build_from_cfg

# Import modules for registry side effects.
from . import mosquito_det as _mosquito_det  # noqa: F401
from . import mosquito_seg as _mosquito_seg  # noqa: F401
from .transforms import augment as _augment  # noqa: F401
from .transforms import formatting as _formatting  # noqa: F401


def _build_transforms(
    pipeline_cfg: Optional[Dict[str, Any]],
    input_size: List[int],
    is_train: bool,
) -> Compose:
    transform_cfgs: Iterable[Dict[str, Any]] = []
    if pipeline_cfg:
        transform_cfgs = pipeline_cfg.get("transforms", []) or []

    transforms = [Resize(input_size)]
    for cfg in transform_cfgs:
        cfg = copy.deepcopy(cfg)
        transform_type = cfg.get("type")
        if transform_type in {"Mosaic", "MosaicSeg"}:
            transforms.append(build_from_cfg(cfg, TRANSFORMS))
            continue
        if not is_train and transform_type in {"RandomHorizontalFlip", "ColorJitter", "MixUp"}:
            continue
        transforms.append(build_from_cfg(cfg, TRANSFORMS))

    if not any(t.__class__.__name__ == "Normalize" for t in transforms):
        transforms.append(Normalize())
    transforms.append(ToTensor())
    return Compose(transforms)


def build_dataset(
    dataset_cfg: Dict[str, Any],
    pipeline_cfg: Optional[Dict[str, Any]] = None,
    is_train: bool = True,
):
    """Build a dataset from config.

    The repository tools pass only ``cfg['dataset']``. For convenience this
    builder also accepts ``pipeline`` nested inside dataset config.
    """

    cfg = copy.deepcopy(dataset_cfg)
    nested_pipeline = cfg.pop("pipeline", None)
    pipeline_cfg = pipeline_cfg or nested_pipeline or {}
    input_size = cfg.pop("input_size", None) or pipeline_cfg.get("input_size", [640, 640])

    if is_train:
        img_dir = cfg.pop("train_path", cfg.pop("img_dir", cfg.pop("image_dir", None)))
        ann_file = cfg.pop("train_ann", cfg.pop("ann_file", None))
    else:
        img_dir = cfg.pop("val_path", cfg.pop("test_path", cfg.pop("img_dir", cfg.pop("image_dir", None))))
        ann_file = cfg.pop("val_ann", cfg.pop("test_ann", cfg.pop("ann_file", None)))

    transforms = _build_transforms(pipeline_cfg, input_size, is_train)
    default_args = {
        "img_dir": img_dir,
        "ann_file": ann_file,
        "input_size": input_size,
        "transforms": transforms,
        "is_train": is_train,
    }
    return build_from_cfg(cfg, DATASETS, default_args=default_args)


def build_dataloader(
    dataset_cfg: Dict[str, Any],
    dataloader_cfg: Optional[Dict[str, Any]] = None,
    is_train: bool = True,
    pipeline_cfg: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    dataloader_cfg = dataloader_cfg or {}
    dataset = build_dataset(dataset_cfg, pipeline_cfg=pipeline_cfg, is_train=is_train)

    batch_size = int(dataloader_cfg.get("batch_size", 1))
    num_workers = int(dataloader_cfg.get("num_workers", 0))
    shuffle = dataloader_cfg.get("shuffle")
    if shuffle is None:
        shuffle = bool(dataloader_cfg.get("shuffle_train", is_train)) if is_train else False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle),
        num_workers=num_workers,
        pin_memory=bool(dataloader_cfg.get("pin_memory", False)),
        drop_last=bool(dataloader_cfg.get("drop_last", False)),
        collate_fn=pad_collate_fn,
    )
