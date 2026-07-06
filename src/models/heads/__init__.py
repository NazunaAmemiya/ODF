"""Prediction heads."""

from .yolo_det_head import YOLOv8DetHead
from .yolo_seg_head import YOLOv8SegHead

__all__ = ["YOLOv8DetHead", "YOLOv8SegHead"]
