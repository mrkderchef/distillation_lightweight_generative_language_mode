"""
Evaluation module for computing metrics (perplexity, coherence, etc.).
"""

from src.evaluation.metrics import compute_perplexity, evaluate_model

__all__ = ["compute_perplexity", "evaluate_model"]
