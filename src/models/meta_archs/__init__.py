"""Meta architectures."""

from .base_model import BaseModel
from .single_stage import SingleStageDetector, SingleStageSegmentor

__all__ = ["BaseModel", "SingleStageDetector", "SingleStageSegmentor"]
