"""Logging helpers."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "mosquito_cv",
    save_dir: Optional[str] = None,
    filename: str = "log.txt",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a console/file logger without duplicating handlers."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(getattr(h, "_mosquito_console", False) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler._mosquito_console = True
        logger.addHandler(console_handler)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, filename)
        abs_file_path = os.path.abspath(file_path)
        if not any(
            getattr(h, "_mosquito_file", None) == abs_file_path
            for h in logger.handlers
        ):
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            file_handler._mosquito_file = abs_file_path
            logger.addHandler(file_handler)

    logger.debug("Logger initialized at %s", datetime.now().isoformat())
    return logger


class TensorboardLogger:
    """Thin optional TensorBoard wrapper used by the trainer."""

    def __init__(self, log_dir: str, enabled: bool = True) -> None:
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=log_dir)
            except Exception:
                self.writer = None

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
