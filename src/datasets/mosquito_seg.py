"""Mosquito segmentation dataset."""

from __future__ import annotations

from typing import Any

from src.datasets.base_dataset import BaseMosquitoDataset
from src.utils.registry import DATASETS


@DATASETS.register_module(name=["MosquitoSegDataset", "MosquitoSegmentationDataset"])
class MosquitoSegDataset(BaseMosquitoDataset):
    """Dataset for polygon/mask mosquito segmentation."""

    task = "seg"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
