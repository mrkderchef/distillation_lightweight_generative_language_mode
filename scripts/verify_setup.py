"""
Script to verify the full setup: models, tokenizer, data pipeline.
Run this to confirm everything is working before training.
"""

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import get_tokenizer
from src.data.dataset import TinyStoriesDataset, get_dataloader
from src.models.teacher import TeacherModel
from src.models.student import StudentModel
from src.generation.generate import generate_text


def verify_setup():
    """Verify all components are working correctly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print()

    # 1. Tokenizer
    print("=" * 50)
    print("1. Testing Tokenizer")
    print("=" * 50)
    tokenizer = get_tokenizer("gpt2")
    test_text = "Once upon a time, there was a little cat."
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"  Input: '{test_text}'")
    print(f"  Token IDs: {tokens['input_ids'][0].tolist()}")
    print(f"  Decoded: '{tokenizer.decode(tokens['input_ids'][0])}'")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    print()

    # 2. Dataset
    print("=" * 50)
    print("2. Testing Dataset")
    print("=" * 50)
    dataset = TinyStoriesDataset(
        split="validation",
        tokenizer=tokenizer,
        max_length=512,
        max_samples=100,
    )
    print(f"  Dataset size: {len(dataset)} samples")
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  labels shape: {sample['labels'].shape}")
    print()

    # 3. DataLoader
    print("=" * 50)
    print("3. Testing DataLoader")
    print("=" * 50)
    dataloader = get_dataloader(
        split="validation",
        tokenizer=tokenizer,
        batch_size=4,
        max_length=512,
        max_samples=100,
    )
    batch = next(iter(dataloader))
    print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"  Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  Batch labels shape: {batch['labels'].shape}")
    print()

    # 4. Teacher Model
    print("=" * 50)
    print("4. Testing Teacher Model (GPT-2 Small)")
    print("=" * 50)
    teacher = TeacherModel(pretrained=True)
    print(f"  Parameters: {teacher.num_parameters:,}")
    print(f"  Size (FP32): {teacher.num_parameters * 4 / 1024**2:.1f} MB")

    teacher = teacher.to(device)
    with torch.no_grad():
        outputs = teacher(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
    print(f"  Output loss: {outputs.loss.item():.4f}")
    print(f"  Logits shape: {outputs.logits.shape}")
    print()

    # 5. Student Model
    print("=" * 50)
    print("5. Testing Student Model (4-layer, 256-hidden)")
    print("=" * 50)
    student = StudentModel(
        vocab_size=50257,
        n_layers=4,
        n_heads=4,
        hidden_size=256,
        max_length=512,
    )
    print(f"  Parameters: {student.num_parameters:,}")
    print(f"  Size (FP32): {student.num_parameters * 4 / 1024**2:.1f} MB")
    print(f"  Compression ratio: {teacher.num_parameters / student.num_parameters:.1f}x")

    student = student.to(device)
    with torch.no_grad():
        outputs = student(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].shape[0] and batch["labels"].to(device),
        )
    print(f"  Output loss: {outputs.loss.item():.4f}")
    print(f"  Logits shape: {outputs.logits.shape}")
    print()

    # 6. Generation
    print("=" * 50)
    print("6. Testing Generation (Teacher)")
    print("=" * 50)
    generated = generate_text(
        model=teacher,
        tokenizer=tokenizer,
        prompt="Once upon a time",
        max_new_tokens=50,
        temperature=0.8,
        device=device,
    )
    print(f"  Generated text:\n  {generated}")
    print()

    # Summary
    print("=" * 50)
    print("SETUP VERIFICATION COMPLETE")
    print("=" * 50)
    print(f"  ✓ Tokenizer working (vocab={tokenizer.vocab_size})")
    print(f"  ✓ Dataset loading (TinyStories)")
    print(f"  ✓ DataLoader batching")
    print(f"  ✓ Teacher model (GPT-2, {teacher.num_parameters:,} params)")
    print(f"  ✓ Student model ({student.num_parameters:,} params)")
    print(f"  ✓ Text generation")
    print(f"\n  Ready for training!")


if __name__ == "__main__":
    verify_setup()
