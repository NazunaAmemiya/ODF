"""Detection visualization utilities."""

from __future__ import annotations

from typing import Dict, Sequence

import cv2
import numpy as np
import torch

from src.utils.registry import VISUALIZERS


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


@VISUALIZERS.register_module(name=["DetVisualizer", "DetectionVisualizer"])
class DetVisualizer:
    """Draw bounding boxes, labels, and scores on an image."""

    def __init__(self, class_names: Sequence[str] = ("mosquito",), color=(0, 255, 0), **kwargs) -> None:
        self.class_names = list(class_names)
        self.color = tuple(int(c) for c in color)
        self.extra_cfg = kwargs

    def draw(self, image: np.ndarray, prediction: Dict, score_thr: float = 0.25) -> np.ndarray:
        out = image.copy()
        boxes = _to_numpy(prediction.get("boxes", np.zeros((0, 4))))
        scores = _to_numpy(prediction.get("scores", np.zeros((0,))))
        labels = _to_numpy(prediction.get("labels", np.zeros((0,), dtype=np.int64))).astype(int)

        for box, score, label in zip(boxes, scores, labels):
            if float(score) < score_thr:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(out, (x1, y1), (x2, y2), self.color, 2)
            name = self.class_names[label] if 0 <= label < len(self.class_names) else str(label)
            caption = f"{name} {float(score):.2f}"
            (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text = max(y1 - th - baseline, 0)
            cv2.rectangle(out, (x1, y_text), (x1 + tw + 4, y_text + th + baseline + 4), self.color, -1)
            cv2.putText(
                out,
                caption,
                (x1 + 2, y_text + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return out

    __call__ = draw
