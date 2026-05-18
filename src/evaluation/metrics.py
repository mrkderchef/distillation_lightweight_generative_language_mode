"""
Evaluation metrics for generative language models.
"""

import torch
import torch.nn.functional as F
import math
from tqdm import tqdm


def compute_perplexity(model, dataloader, device="cuda"):
    """
    Compute perplexity of a language model on a given dataset.

    Args:
        model: Language model to evaluate.
        dataloader: DataLoader with tokenized data.
        device: Device to run evaluation on.

    Returns:
        Perplexity score (float).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Perplexity"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Count non-padding tokens
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def evaluate_model(model, dataloader, device="cuda"):
    """
    Comprehensive evaluation of a language model.

    Args:
        model: Language model to evaluate.
        dataloader: DataLoader with tokenized data.
        device: Device to run evaluation on.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    # Model size metrics
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "num_parameters": num_params,
        "model_size_mb": model_size_mb,
        "total_tokens_evaluated": total_tokens,
    }
