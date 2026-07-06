"""Detection metrics."""

from __future__ import annotations

from typing import Dict, List

import torch

from src.utils.metrics import box_iou, precision_recall
from src.utils.registry import METRICS


@METRICS.register_module(name=["DetectionMetric", "DetMetric"])
class DetectionMetric:
    """Greedy IoU metric for quick validation feedback.

    This is intentionally lightweight. It reports AP50-like precision, recall,
    and F1 using greedy matching at one IoU threshold.
    """

    def __init__(self, iou_threshold: float = 0.5, score_threshold: float = 0.0, **kwargs) -> None:
        self.iou_threshold = float(iou_threshold)
        self.score_threshold = float(score_threshold)
        self.extra_cfg = kwargs
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, predictions: List[Dict[str, torch.Tensor]], targets: Dict[str, torch.Tensor]) -> None:
        gt_boxes = targets.get("gt_bboxes")
        valid = targets.get("valid_mask")
        if gt_boxes is None:
            return

        for idx, pred in enumerate(predictions):
            gt = gt_boxes[idx]
            if valid is not None:
                gt = gt[valid[idx]]
            boxes = pred.get("boxes", gt.new_zeros((0, 4))).detach().cpu()
            scores = pred.get("scores", gt.new_zeros((0,))).detach().cpu()
            gt = gt.detach().cpu()
            keep = scores >= self.score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            order = scores.argsort(descending=True)
            boxes = boxes[order]

            if gt.numel() == 0:
                self.fp += int(boxes.shape[0])
                continue
            if boxes.numel() == 0:
                self.fn += int(gt.shape[0])
                continue

            matched = torch.zeros((gt.shape[0],), dtype=torch.bool)
            ious = box_iou(boxes, gt)
            for pred_idx in range(boxes.shape[0]):
                best_iou, best_gt = ious[pred_idx].max(dim=0)
                if best_iou >= self.iou_threshold and not matched[best_gt]:
                    matched[best_gt] = True
                    self.tp += 1
                else:
                    self.fp += 1
            self.fn += int((~matched).sum().item())

    def compute(self) -> Dict[str, float]:
        precision, recall, f1 = precision_recall(self.tp, self.fp, self.fn)
        return {
            "mAP50": precision,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(self.fn),
        }
