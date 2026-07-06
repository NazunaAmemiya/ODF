"""Segmentation visualization utilities."""

from __future__ import annotations

from typing import Dict, Sequence

import cv2
import numpy as np
import torch

from src.utils.registry import VISUALIZERS
from src.visualization.det_visualizer import DetVisualizer, _to_numpy


@VISUALIZERS.register_module(name=["SegVisualizer", "SegmentationVisualizer"])
class SegVisualizer(DetVisualizer):
    """Overlay masks and draw detection boxes."""

    def __init__(
        self,
        class_names: Sequence[str] = ("mosquito",),
        color=(0, 255, 0),
        mask_color=(0, 128, 255),
        alpha: float = 0.45,
        **kwargs,
    ) -> None:
        super().__init__(class_names=class_names, color=color, **kwargs)
        self.mask_color = np.asarray(mask_color, dtype=np.uint8)
        self.alpha = float(alpha)

    def draw(self, image: np.ndarray, prediction: Dict, score_thr: float = 0.25) -> np.ndarray:
        out = image.copy()
        masks = prediction.get("masks")
        if masks is not None:
            masks = _to_numpy(masks)
            if masks.ndim == 3:
                mask = masks[0]
            elif masks.ndim == 2:
                mask = masks
            else:
                mask = None
            if mask is not None:
                if mask.shape[:2] != out.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(bool)
                overlay = out.copy()
                overlay[mask] = (overlay[mask] * (1.0 - self.alpha) + self.mask_color * self.alpha).astype(np.uint8)
                out = overlay
        return super().draw(out, prediction, score_thr=score_thr)

    __call__ = draw
