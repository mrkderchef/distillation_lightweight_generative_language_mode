"""
Script to download and prepare the TinyStories dataset.
Saves processed data locally for faster loading during training.
"""

import argparse
from pathlib import Path
from datasets import load_dataset


def download_tinystories(output_dir="data", num_proc=4):
    """
    Download TinyStories dataset from Hugging Face and save locally.

    Args:
        output_dir: Directory to save the dataset.
        num_proc: Number of processes for parallel operations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading TinyStories dataset from Hugging Face...")
    print("This may take a few minutes on first run.\n")

    # Load dataset (streams and caches automatically)
    dataset = load_dataset("roneneldan/TinyStories")

    print(f"Dataset loaded successfully!")
    print(f"  Train split: {len(dataset['train']):,} samples")
    print(f"  Validation split: {len(dataset['validation']):,} samples")

    # Save to disk in Arrow format for fast loading
    train_path = output_dir / "train"
    val_path = output_dir / "validation"

    print(f"\nSaving train split to {train_path}...")
    dataset["train"].save_to_disk(str(train_path))

    print(f"Saving validation split to {val_path}...")
    dataset["validation"].save_to_disk(str(val_path))

    print("\nDataset download and preparation complete!")
    print(f"Data saved to: {output_dir.resolve()}")

    # Print sample
    print("\n--- Sample Story ---")
    print(dataset["train"][0]["text"][:500])
    print("---")

    return dataset


def get_dataset_stats(dataset):
    """Print basic statistics about the dataset."""
    print("\n=== Dataset Statistics ===")
    for split_name, split_data in dataset.items():
        texts = split_data["text"]
        lengths = [len(t) for t in texts]
        print(f"\n{split_name}:")
        print(f"  Samples: {len(texts):,}")
        print(f"  Avg character length: {sum(lengths)/len(lengths):.0f}")
        print(f"  Min character length: {min(lengths)}")
        print(f"  Max character length: {max(lengths)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TinyStories dataset")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    args = parser.parse_args()

    dataset = download_tinystories(output_dir=args.output_dir)

    if args.stats:
        get_dataset_stats(dataset)
