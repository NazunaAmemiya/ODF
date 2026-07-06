"""Base dataset classes for mosquito detection and segmentation."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

from src.datasets.transforms import Compose, Normalize, Resize, ToTensor

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _as_abs(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.abspath(path)


def _clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    boxes = boxes.astype(np.float32)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep]


class BaseMosquitoDataset(Dataset):
    """Read mosquito datasets in COCO JSON or YOLO txt format.

    Returned samples use the normalized framework keys:
    ``img``, ``gt_bboxes``, ``gt_labels``, ``gt_masks``, and ``img_shape``.
    """

    task = "det"

    def __init__(
        self,
        img_dir: Optional[str] = None,
        ann_file: Optional[str] = None,
        input_size: Sequence[int] = (640, 640),
        transforms: Optional[Any] = None,
        class_names: Optional[Sequence[str]] = None,
        num_classes: int = 1,
        is_train: bool = True,
        **kwargs: Any,
    ) -> None:
        self.img_dir = _as_abs(img_dir)
        self.ann_file = _as_abs(ann_file)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.class_names = list(class_names or ["mosquito"])
        self.num_classes = int(num_classes or len(self.class_names) or 1)
        self.is_train = is_train
        self.extra_cfg = kwargs
        self.samples = self._load_samples()
        self.transforms = transforms or Compose([Resize(self.input_size), Normalize(), ToTensor()])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.samples[index]
        image = cv2.imread(item["img_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {item['img_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        boxes = np.asarray(item.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(item.get("labels", []), dtype=np.int64).reshape(-1)
        masks = self._build_masks(item.get("segmentations", []), height, width, boxes)

        if len(boxes) != len(labels):
            labels = np.zeros((len(boxes),), dtype=np.int64)
        if len(masks) and len(masks) != len(boxes):
            masks = masks[: len(boxes)]

        sample = {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
        }
        return self.transforms(sample)

    def _load_samples(self) -> List[Dict[str, Any]]:
        if self.ann_file and os.path.isfile(self.ann_file):
            try:
                return self._load_coco(self.ann_file)
            except Exception as exc:
                raise RuntimeError(f"Failed to read annotation file {self.ann_file}: {exc}") from exc
        return self._load_yolo_or_images()

    def _load_coco(self, ann_file: str) -> List[Dict[str, Any]]:
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        categories = data.get("categories", [])
        cat_to_label = {
            cat.get("id", idx + 1): idx for idx, cat in enumerate(categories)
        }
        if not cat_to_label:
            cat_to_label = {1: 0, 0: 0}

        anns_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for ann in data.get("annotations", []):
            if ann.get("iscrowd", 0):
                continue
            anns_by_image[int(ann["image_id"])].append(ann)

        samples: List[Dict[str, Any]] = []
        for img in data.get("images", []):
            file_name = img.get("file_name") or img.get("path")
            if not file_name:
                continue
            img_path = file_name if os.path.isabs(file_name) else os.path.join(self.img_dir or "", file_name)
            boxes: List[List[float]] = []
            labels: List[int] = []
            segmentations: List[Any] = []
            for ann in anns_by_image.get(int(img["id"]), []):
                bbox = ann.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                x, y, w, h = [float(v) for v in bbox[:4]]
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(cat_to_label.get(ann.get("category_id", 1), 0))
                segmentations.append(ann.get("segmentation"))
            samples.append(
                {
                    "img_path": os.path.abspath(img_path),
                    "boxes": boxes,
                    "labels": labels,
                    "segmentations": segmentations,
                }
            )
        return samples

    def _load_yolo_or_images(self) -> List[Dict[str, Any]]:
        if not self.img_dir or not os.path.isdir(self.img_dir):
            return []

        image_paths: List[str] = []
        for root, _, files in os.walk(self.img_dir):
            for file_name in files:
                if os.path.splitext(file_name)[1].lower() in _IMAGE_EXTS:
                    image_paths.append(os.path.join(root, file_name))

        samples: List[Dict[str, Any]] = []
        for img_path in sorted(image_paths):
            boxes, labels, segments = self._read_yolo_label(img_path)
            samples.append(
                {
                    "img_path": os.path.abspath(img_path),
                    "boxes": boxes,
                    "labels": labels,
                    "segmentations": segments,
                }
            )
        return samples

    def _read_yolo_label(self, img_path: str) -> Tuple[List[List[float]], List[int], List[Any]]:
        label_path = self._guess_label_path(img_path)
        if not label_path or not os.path.isfile(label_path):
            return [], [], []

        image = cv2.imread(img_path)
        if image is None:
            return [], [], []
        h, w = image.shape[:2]
        boxes: List[List[float]] = []
        labels: List[int] = []
        segments: List[Any] = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [float(x) for x in line.strip().split()]
                if len(parts) < 5:
                    continue
                label = int(parts[0])
                coords = parts[1:]
                if len(coords) == 4:
                    cx, cy, bw, bh = coords
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    segments.append(None)
                elif len(coords) >= 6 and len(coords) % 2 == 0:
                    pts = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
                    pts[:, 0] *= w
                    pts[:, 1] *= h
                    x1, y1 = pts.min(axis=0)
                    x2, y2 = pts.max(axis=0)
                    boxes.append([float(x1), float(y1), float(x2), float(y2)])
                    segments.append([pts.reshape(-1).tolist()])
                else:
                    continue
                labels.append(label)
        return boxes, labels, segments

    def _guess_label_path(self, img_path: str) -> Optional[str]:
        stem = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        candidates = [
            os.path.join(os.path.dirname(img_path), stem),
            os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels", stem),
            img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit(".", 1)[0] + ".txt",
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return candidates[0]

    def _build_masks(
        self,
        segmentations: Sequence[Any],
        height: int,
        width: int,
        boxes: np.ndarray,
    ) -> np.ndarray:
        if self.task != "seg" or not len(boxes):
            return np.zeros((0, height, width), dtype=np.uint8)

        masks: List[np.ndarray] = []
        for idx, segmentation in enumerate(segmentations):
            mask = np.zeros((height, width), dtype=np.uint8)
            if isinstance(segmentation, list):
                for polygon in segmentation:
                    arr = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                    if len(arr) >= 3:
                        cv2.fillPoly(mask, [arr.astype(np.int32)], 1)
            elif isinstance(segmentation, dict):
                # RLE decoding needs pycocotools, which is intentionally optional.
                # Fall back to a box mask so training still has a useful target.
                pass

            if mask.sum() == 0 and idx < len(boxes):
                x1, y1, x2, y2 = boxes[idx].astype(int)
                mask[max(y1, 0): max(y2, 0), max(x1, 0): max(x2, 0)] = 1
            masks.append(mask)

        if not masks:
            return np.zeros((0, height, width), dtype=np.uint8)
        return np.stack(masks, axis=0).astype(np.uint8)
