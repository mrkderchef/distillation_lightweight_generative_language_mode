"""
Text generation utilities for autoregressive language models.
"""

import torch
import torch.nn.functional as F


def generate_text(
    model,
    tokenizer,
    prompt="Once upon a time",
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    device="cuda",
):
    """
    Generate text using autoregressive decoding.

    Args:
        model: Trained language model.
        tokenizer: Tokenizer for encoding/decoding.
        prompt: Text prompt to start generation.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (1.0 = neutral).
        top_k: Top-k filtering parameter.
        top_p: Nucleus sampling parameter.
        device: Device for inference.

    Returns:
        Generated text string.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            if hasattr(outputs, "logits"):
                logits = outputs.logits[:, -1, :]
            else:
                logits = outputs[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text
