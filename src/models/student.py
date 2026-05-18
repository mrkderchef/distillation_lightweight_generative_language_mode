"""
Student model: Lightweight decoder-only transformer for distillation.
"""

import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config


class StudentModel(nn.Module):
    """Lightweight GPT-2 style student model."""

    def __init__(
        self,
        vocab_size=50257,
        n_layers=4,
        n_heads=4,
        hidden_size=256,
        max_length=512,
    ):
        """
        Args:
            vocab_size: Vocabulary size (default: GPT-2 tokenizer vocab).
            n_layers: Number of transformer layers.
            n_heads: Number of attention heads.
            hidden_size: Hidden dimension size.
            max_length: Maximum sequence length.
        """
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_layer=n_layers,
            n_head=n_heads,
            n_embd=hidden_size,
            n_positions=max_length,
            n_inner=hidden_size * 4,
        )
        self.model = GPT2LMHeadModel(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass returning logits and optional loss."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def get_logits(self, input_ids, attention_mask=None):
        """Get raw logits without computing loss."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())
