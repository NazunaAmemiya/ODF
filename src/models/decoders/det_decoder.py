"""Detection output decoder with NMS."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn

from src.utils.metrics import box_iou
from src.utils.registry import DECODERS


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Torchvision NMS if available, otherwise a small pure PyTorch fallback."""

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    try:
        from torchvision.ops import nms as tv_nms

        return tv_nms(boxes, scores, iou_threshold)
    except Exception:
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            current = order[0]
            keep.append(current)
            if order.numel() == 1:
                break
            ious = box_iou(boxes[current].unsqueeze(0), boxes[order[1:]]).squeeze(0)
            order = order[1:][ious <= iou_threshold]
        return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)


@DECODERS.register_module(name=["DetDecoder", "YOLOv8DetDecoder"])
class DetDecoder(nn.Module):
    """Decode compact YOLO head tensors into boxes/scores/labels."""

    def __init__(
        self,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.max_det = int(max_det)
        self.extra_cfg = kwargs

    def forward(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        image_size: Sequence[int],
        conf_thres: float = None,
        iou_thres: float = None,
    ) -> List[Dict[str, torch.Tensor]]:
        return self.decode(outputs, image_size, conf_thres, iou_thres)

    def decode(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        image_size: Sequence[int],
        conf_thres: float = None,
        iou_thres: float = None,
    ) -> List[Dict[str, torch.Tensor]]:
        conf = self.conf_thres if conf_thres is None else float(conf_thres)
        iou_thr = self.iou_thres if iou_thres is None else float(iou_thres)
        image_h, image_w = int(image_size[0]), int(image_size[1])

        cls_logits = outputs["cls_logits"]
        obj_logits = outputs["obj_logits"]
        box_reg = outputs["box_reg"]
        batch_size = cls_logits[0].shape[0]
        decoded: List[Dict[str, torch.Tensor]] = []

        for batch_idx in range(batch_size):
            boxes_all = []
            scores_all = []
            labels_all = []
            for cls_level, obj_level, box_level in zip(cls_logits, obj_logits, box_reg):
                cls_prob = cls_level[batch_idx].sigmoid().permute(1, 2, 0).reshape(-1, cls_level.shape[1])
                obj_prob = obj_level[batch_idx].sigmoid().permute(1, 2, 0).reshape(-1, 1)
                box = box_level[batch_idx].sigmoid().permute(1, 2, 0).reshape(-1, 4)
                class_scores, labels = cls_prob.max(dim=1)
                scores = class_scores * obj_prob.squeeze(1)
                keep = scores >= conf
                if keep.sum() == 0:
                    continue
                box = box[keep]
                scores = scores[keep]
                labels = labels[keep]
                cx = box[:, 0] * image_w
                cy = box[:, 1] * image_h
                bw = box[:, 2] * image_w
                bh = box[:, 3] * image_h
                xyxy = torch.stack(
                    [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1
                )
                xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, image_w - 1)
                xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, image_h - 1)
                boxes_all.append(xyxy)
                scores_all.append(scores)
                labels_all.append(labels)

            if boxes_all:
                boxes = torch.cat(boxes_all, dim=0)
                scores = torch.cat(scores_all, dim=0)
                labels = torch.cat(labels_all, dim=0)
                keep = nms(boxes, scores, iou_thr)[: self.max_det]
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
            else:
                device = cls_logits[0].device
                boxes = torch.zeros((0, 4), device=device)
                scores = torch.zeros((0,), device=device)
                labels = torch.zeros((0,), dtype=torch.long, device=device)
            decoded.append({"boxes": boxes, "scores": scores, "labels": labels})
        return decoded
