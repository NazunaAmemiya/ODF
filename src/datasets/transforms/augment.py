"""Data augmentation primitives for image, box, and mask samples."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

from src.utils.registry import TRANSFORMS

Sample = Dict[str, Any]


class Compose:
    """Apply transforms in sequence."""

    def __init__(self, transforms: Optional[Iterable[Any]] = None) -> None:
        self.transforms = list(transforms or [])

    def __call__(self, sample: Sample) -> Sample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


@TRANSFORMS.register_module()
class Resize:
    """Resize image, boxes, and masks to a fixed ``(height, width)`` size."""

    def __init__(self, size: Sequence[int] = (640, 640)) -> None:
        if len(size) != 2:
            raise ValueError("Resize size must be [height, width].")
        self.size = (int(size[0]), int(size[1]))

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]
        old_h, old_w = image.shape[:2]
        new_h, new_w = self.size
        sample["image"] = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        boxes = sample.get("boxes")
        if boxes is not None and len(boxes):
            boxes = np.asarray(boxes, dtype=np.float32).copy()
            boxes[:, [0, 2]] *= new_w / max(old_w, 1)
            boxes[:, [1, 3]] *= new_h / max(old_h, 1)
            sample["boxes"] = boxes

        masks = sample.get("masks")
        if masks is not None and len(masks):
            resized = [
                cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                for mask in masks
            ]
            sample["masks"] = np.stack(resized, axis=0).astype(np.uint8)
        return sample


@TRANSFORMS.register_module()
class RandomHorizontalFlip:
    """Horizontally flip a sample with probability ``prob``."""

    def __init__(self, prob: float = 0.5) -> None:
        self.prob = float(prob)

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.prob:
            return sample

        image = np.ascontiguousarray(sample["image"][:, ::-1])
        width = image.shape[1]
        sample["image"] = image

        boxes = sample.get("boxes")
        if boxes is not None and len(boxes):
            boxes = np.asarray(boxes, dtype=np.float32).copy()
            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()
            boxes[:, 0] = width - x2
            boxes[:, 2] = width - x1
            sample["boxes"] = boxes

        masks = sample.get("masks")
        if masks is not None and len(masks):
            sample["masks"] = np.ascontiguousarray(masks[:, :, ::-1])
        return sample


@TRANSFORMS.register_module()
class ColorJitter:
    """Random brightness/contrast jitter for RGB images."""

    def __init__(self, brightness: float = 0.0, contrast: float = 0.0, prob: float = 1.0) -> None:
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.prob = float(prob)

    def __call__(self, sample: Sample) -> Sample:
        if random.random() >= self.prob:
            return sample

        image = sample["image"].astype(np.float32)
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = (image - 127.5) * factor + 127.5
        if self.brightness > 0:
            delta = 255.0 * random.uniform(-self.brightness, self.brightness)
            image = image + delta
        sample["image"] = np.clip(image, 0, 255).astype(np.uint8)
        return sample


@TRANSFORMS.register_module()
class Normalize:
    """Normalize RGB image from 0..255 to float values."""

    def __init__(
        self,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        to_rgb: bool = False,
    ) -> None:
        self.mean = np.asarray(mean if mean is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        self.std = np.asarray(std if std is not None else [1.0, 1.0, 1.0], dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]
        if self.to_rgb:
            image = image[..., ::-1]
        image = image.astype(np.float32) / 255.0
        sample["image"] = (image - self.mean) / np.maximum(self.std, 1e-6)
        return sample


@TRANSFORMS.register_module(name=["Mosaic", "MosaicSeg"])
class Mosaic:
    """Placeholder-compatible Mosaic transform.

    True Mosaic needs access to extra dataset samples. The transform is kept as a
    no-op so existing configs run, and it can be upgraded later without changing
    the public pipeline API.
    """

    def __init__(self, prob: float = 1.0, **kwargs: Any) -> None:
        self.prob = prob
        self.kwargs = kwargs

    def __call__(self, sample: Sample) -> Sample:
        return sample


@TRANSFORMS.register_module()
class MixUp:
    """No-op MixUp hook for config compatibility."""

    def __init__(self, prob: float = 0.0, **kwargs: Any) -> None:
        self.prob = prob
        self.kwargs = kwargs

    def __call__(self, sample: Sample) -> Sample:
        return sample
