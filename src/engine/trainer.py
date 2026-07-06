"""Reusable trainer for Mosquito-CV models."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

import torch

from src.datasets.builder import build_dataloader
from src.engine.callbacks import Callback, CheckpointHook
from src.engine.optim import build_optimizer, build_scheduler
from src.models.builder import build_model
from src.utils.logger import TensorboardLogger, setup_logger


class Trainer:
    """Config-driven training loop shared by experiments."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        work_dir: Optional[str] = None,
        callbacks: Optional[Iterable[Callback]] = None,
    ) -> None:
        self.cfg = cfg
        self.work_dir = work_dir or cfg.get("work_dir", "./work_dirs")
        os.makedirs(self.work_dir, exist_ok=True)
        self.logger = setup_logger("mosquito_cv_trainer", self.work_dir)
        self.device = torch.device(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(cfg["model"]).to(self.device)
        self.optimizer = build_optimizer(self.model, cfg.get("optimizer", {}))
        self.train_loader = build_dataloader(
            cfg["dataset"], cfg.get("dataloader", {}), is_train=True, pipeline_cfg=cfg.get("pipeline")
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            cfg.get("scheduler", {}),
            steps_per_epoch=max(len(self.train_loader), 1),
        )
        self.val_loader = None
        if cfg.get("validate", False):
            self.val_loader = build_dataloader(
                cfg["dataset"], cfg.get("dataloader", {}), is_train=False, pipeline_cfg=cfg.get("pipeline")
            )
        logging_cfg = cfg.get("logging", {})
        self.tb = TensorboardLogger(
            os.path.join(self.work_dir, "tensorboard"),
            enabled=bool(logging_cfg.get("use_tensorboard", False)),
        )
        ckpt_cfg = cfg.get("checkpoint", {})
        default_callbacks: List[Callback] = [
            CheckpointHook(
                self.work_dir,
                interval=int(ckpt_cfg.get("interval", 1)),
                save_best=bool(ckpt_cfg.get("save_best", True)),
                monitor=str(ckpt_cfg.get("monitor_metric", "loss")),
                mode="min" if str(ckpt_cfg.get("monitor_metric", "loss")).lower().endswith("loss") else "max",
            )
        ]
        self.callbacks = default_callbacks + list(callbacks or [])
        self.global_step = 0
        self.should_stop = False

    def train(self) -> None:
        epochs = int(self.cfg.get("epochs", 100))
        interval = int(self.cfg.get("logging", {}).get("interval", 10))
        for callback in self.callbacks:
            callback.on_train_start(self)

        for epoch in range(1, epochs + 1):
            if self.should_stop:
                break
            for callback in self.callbacks:
                callback.on_epoch_start(self, epoch)
            metrics = self.train_one_epoch(epoch, interval)
            if self.scheduler is not None:
                self.scheduler.step()
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, metrics)

        for callback in self.callbacks:
            callback.on_train_end(self)
        self.tb.close()

    def train_one_epoch(self, epoch: int, log_interval: int = 10) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        for step, batch in enumerate(self.train_loader):
            images = batch["img"].to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k != "img"}
            loss = self.model(images, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            loss_value = float(loss.detach().item())
            total_loss += loss_value
            self.global_step += 1
            self.tb.add_scalar("train/loss", loss_value, self.global_step)
            if step % max(log_interval, 1) == 0:
                self.logger.info("Epoch %s | step %s/%s | loss %.4f", epoch, step, len(self.train_loader), loss_value)
            for callback in self.callbacks:
                callback.on_batch_end(self, epoch, step, loss_value)
        avg_loss = total_loss / max(len(self.train_loader), 1)
        self.logger.info("Epoch %s finished | avg_loss %.4f", epoch, avg_loss)
        return {"loss": avg_loss}
