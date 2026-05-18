"""
Teacher model: GPT-2 Small for autoregressive language modeling on TinyStories.
"""

import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config


class TeacherModel(nn.Module):
    """GPT-2 Small teacher model wrapper."""

    def __init__(self, pretrained=True, model_name="gpt2"):
        """
        Args:
            pretrained: Whether to load pretrained weights.
            model_name: Model identifier for Hugging Face.
        """
        super().__init__()
        if pretrained:
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            config = GPT2Config()
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
