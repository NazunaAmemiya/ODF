"""Formatting utilities that turn numpy samples into batched tensors."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from src.utils.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ToTensor:
    """Convert image, boxes, labels, and masks to tensors."""

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"]
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        image = np.ascontiguousarray(image.transpose(2, 0, 1))

        boxes = np.asarray(sample.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(sample.get("labels", []), dtype=np.int64).reshape(-1)
        masks = sample.get("masks")
        if masks is None:
            masks = np.zeros((0, image.shape[1], image.shape[2]), dtype=np.uint8)
        masks = np.asarray(masks, dtype=np.uint8)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.size == 0:
            masks = np.zeros((0, image.shape[1], image.shape[2]), dtype=np.uint8)

        return {
            "img": torch.from_numpy(image).float(),
            "gt_bboxes": torch.from_numpy(boxes).float(),
            "gt_labels": torch.from_numpy(labels).long(),
            "gt_masks": torch.from_numpy(masks).float(),
            "img_shape": torch.tensor([image.shape[1], image.shape[2]], dtype=torch.long),
        }


@TRANSFORMS.register_module(name=["FormatDet", "FormatSeg"])
class FormatSample(ToTensor):
    """Alias kept for config readability."""



def pad_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length boxes/masks so the simple tools can call ``.to()``."""

    images = torch.stack([item["img"] for item in batch], dim=0)
    batch_size, _, height, width = images.shape
    max_instances = max((item["gt_bboxes"].shape[0] for item in batch), default=0)

    boxes = images.new_zeros((batch_size, max_instances, 4))
    labels = torch.full((batch_size, max_instances), -1, dtype=torch.long)
    valid = torch.zeros((batch_size, max_instances), dtype=torch.bool)
    masks = images.new_zeros((batch_size, max_instances, height, width))
    img_shape = torch.stack([item["img_shape"] for item in batch], dim=0)

    for idx, item in enumerate(batch):
        num = item["gt_bboxes"].shape[0]
        if num == 0:
            continue
        boxes[idx, :num] = item["gt_bboxes"]
        labels[idx, :num] = item["gt_labels"]
        valid[idx, :num] = True
        item_masks = item.get("gt_masks")
        if item_masks is not None and item_masks.numel() > 0:
            masks[idx, :num] = item_masks[:num]

    return {
        "img": images,
        "gt_bboxes": boxes,
        "gt_labels": labels,
        "valid_mask": valid,
        "gt_masks": masks,
        "img_shape": img_shape,
    }
