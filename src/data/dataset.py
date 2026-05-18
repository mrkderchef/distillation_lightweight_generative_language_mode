"""
TinyStories dataset loading and preprocessing.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class TinyStoriesDataset(Dataset):
    """Dataset class for loading and tokenizing TinyStories data."""

    def __init__(self, split="train", tokenizer=None, max_length=512, max_samples=None):
        """
        Args:
            split: Dataset split ('train' or 'validation').
            tokenizer: Hugging Face tokenizer instance.
            max_length: Maximum sequence length for tokenization.
            max_samples: Optional limit on number of samples to load.
        """
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length

        dataset = load_dataset("roneneldan/TinyStories", split=split)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # For causal LM: labels are shifted input_ids
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # ignore padding in loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def get_dataloader(split, tokenizer, batch_size=16, max_length=512, max_samples=None):
    """Create a DataLoader for the TinyStories dataset."""
    dataset = TinyStoriesDataset(
        split=split,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=True,
    )
