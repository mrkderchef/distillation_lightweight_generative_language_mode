"""
Tokenizer setup for the project.
"""

from transformers import AutoTokenizer


def get_tokenizer(model_name="gpt2"):
    """
    Load and configure a tokenizer.

    Args:
        model_name: Pretrained model name for tokenizer (default: gpt2).

    Returns:
        Configured tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 tokenizer has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
