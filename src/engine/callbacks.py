"""Training callbacks and hooks."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from src.utils.checkpoint import save_checkpoint

class Callback:
    """Base callback with optional lifecycle hooks."""

    def on_train_start(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        pass

    def on_batch_end(self, trainer: Any, epoch: int, step: int, loss: float) -> None:
        pass


class CheckpointHook(Callback):
    """Save latest and optional best checkpoints."""

    def __init__(
        self,
        save_dir: str,
        interval: int = 1,
        save_best: bool = True,
        monitor: str = "loss",
        mode: str = "min",
    ) -> None:
        self.save_dir = save_dir
        self.interval = max(int(interval), 1)
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.best_value: Optional[float] = None

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        if epoch % self.interval == 0:
            save_checkpoint(
                trainer.model,
                os.path.join(self.save_dir, "latest.pth"),
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=epoch,
                metrics=metrics,
            )
        if not self.save_best or self.monitor not in metrics:
            return
        value = float(metrics[self.monitor])
        better = self.best_value is None
        if not better:
            better = value < self.best_value if self.mode == "min" else value > self.best_value
        if better:
            self.best_value = value
            save_checkpoint(
                trainer.model,
                os.path.join(self.save_dir, "best.pth"),
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=epoch,
                metrics=metrics,
            )


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving."""

    def __init__(self, monitor: str = "loss", patience: int = 20, mode: str = "min", min_delta: float = 0.005) -> None:
        self.monitor = monitor
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = min_delta # Thêm ngưỡng tối thiểu
        self.best_value: Optional[float] = None
        self.bad_epochs = 0

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        if self.monitor not in metrics:
            return
            
        value = float(metrics[self.monitor])
        better = self.best_value is None
        
        if not better:
            # Ép mô hình phải giảm Loss sâu hơn mức min_delta thì mới được tính là tốt hơn
            if self.mode == "min":
                better = value < (self.best_value - self.min_delta)
            else:
                better = value > (self.best_value + self.min_delta)
                
        if better:
            self.best_value = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            
        if self.bad_epochs >= self.patience:
            trainer.should_stop = True
