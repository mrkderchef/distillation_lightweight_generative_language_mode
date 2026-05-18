"""
Training module for teacher training, student baseline, and distillation.
"""

from src.training.trainer import Trainer
from src.training.distillation import DistillationTrainer

__all__ = ["Trainer", "DistillationTrainer"]
