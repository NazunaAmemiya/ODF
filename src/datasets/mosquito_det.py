"""Mosquito detection dataset."""

from __future__ import annotations

from typing import Any

from src.datasets.base_dataset import BaseMosquitoDataset
from src.utils.registry import DATASETS


@DATASETS.register_module(name=["MosquitoDetDataset", "MosquitoDetectionDataset"])
class MosquitoDetDataset(BaseMosquitoDataset):
    """Dataset for bounding-box mosquito detection."""

    task = "det"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
