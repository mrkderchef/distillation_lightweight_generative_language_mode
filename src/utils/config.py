"""
Configuration loading utilities.
"""

import yaml
from pathlib import Path


DEFAULT_CONFIG = {
    "data": {
        "dataset": "roneneldan/TinyStories",
        "max_length": 512,
        "batch_size": 16,
        "max_samples_train": None,
        "max_samples_val": None,
    },
    "teacher": {
        "model_name": "gpt2",
        "pretrained": True,
    },
    "student": {
        "vocab_size": 50257,
        "n_layers": 4,
        "n_heads": 4,
        "hidden_size": 256,
        "max_length": 512,
    },
    "training": {
        "epochs": 5,
        "lr": 5e-4,
        "weight_decay": 0.01,
        "use_amp": True,
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.5,
        "epochs": 10,
        "lr": 5e-4,
    },
    "generation": {
        "max_new_tokens": 100,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
    },
}


def load_config(config_path=None):
    """
    Load configuration from YAML file or return defaults.

    Args:
        config_path: Path to YAML config file. If None, returns defaults.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        return DEFAULT_CONFIG

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Merge with defaults
    merged = DEFAULT_CONFIG.copy()
    for key, value in config.items():
        if isinstance(value, dict) and key in merged:
            merged[key].update(value)
        else:
            merged[key] = value

    return merged
