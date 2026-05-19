"""
Compare baseline and distilled student checkpoints on the same validation subset.

The script writes machine-readable metrics to JSON and qualitative samples to
Markdown so runs can be compared after training.

Usage:
    python scripts/compare_students.py --config config.yaml --device cuda
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from datasets import Dataset as ArrowDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import get_tokenizer
from src.evaluation.metrics import evaluate_model
from src.generation.generate import generate_text
from src.models.student import StudentModel
from src.utils.config import load_config


DEFAULT_PROMPTS = (
    "Once upon a time",
    "The little dog",
    "She was very happy",
)


class TokenizedTextDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_cached_validation_arrow():
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "roneneldan___tiny_stories"
    matches = sorted(cache_root.glob("default/0.0.0/*/tiny_stories-validation.arrow"))
    return matches[-1] if matches else None


def load_validation_dataset(max_samples):
    arrow_path = find_cached_validation_arrow()
    if arrow_path is None:
        raise FileNotFoundError(
            "Could not find cached TinyStories validation Arrow file. "
            "Run training or download the dataset once before comparing."
        )
    print(f"Loading cached validation data: {arrow_path}", flush=True)
    dataset = ArrowDataset.from_file(str(arrow_path))
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def load_student(config, checkpoint_path, device):
    model = StudentModel(**config["student"]).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def evaluate_checkpoint(name, checkpoint_path, config, dataloader, tokenizer, prompts, args):
    print(f"\nEvaluating {name}: {checkpoint_path}", flush=True)
    model = load_student(config, checkpoint_path, args.device)
    metrics = evaluate_model(model, dataloader, device=args.device)

    samples = {}
    for prompt in prompts:
        torch.manual_seed(args.seed)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)
        samples[prompt] = generate_text(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.generation_temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
        )

    del model
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
        "samples": samples,
    }


def write_reports(results, args):
    args.report_dir.mkdir(parents=True, exist_ok=True)
    run_id = timestamp()

    json_path = args.report_dir / f"student_comparison_{run_id}.json"
    md_path = args.report_dir / f"student_comparison_{run_id}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Student Comparison\n\n")
        f.write(f"- Validation samples: {args.max_samples_val}\n")
        f.write(f"- Device: `{args.device}`\n")
        f.write(f"- Seed: `{args.seed}`\n\n")

        f.write("| Model | Loss | Perplexity | Parameters | Size MB |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for name, result in results["models"].items():
            metrics = result["metrics"]
            f.write(
                f"| {name} | {metrics['loss']:.4f} | {metrics['perplexity']:.2f} | "
                f"{metrics['num_parameters']:,} | {metrics['model_size_mb']:.1f} |\n"
            )

        f.write("\n## Samples\n")
        for prompt in results["prompts"]:
            f.write(f"\n### {prompt}\n\n")
            for name, result in results["models"].items():
                f.write(f"**{name}**\n\n")
                f.write(result["samples"][prompt].strip() + "\n\n")

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Compare student model checkpoints.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples-val", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--baseline-path", type=Path, default=Path("outputs/student_baseline.pt"))
    parser.add_argument("--distilled-path", type=Path, default=None)
    parser.add_argument("--old-distilled-path", type=Path, default=Path("outputs/student_distilled.pt"))
    parser.add_argument("--include-old-distilled", action="store_true")
    parser.add_argument(
        "--extra-checkpoint",
        action="append",
        default=[],
        help="Additional checkpoint to compare, formatted as name=path",
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--generation-temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.max_samples_val = args.max_samples_val or config["data"].get("max_samples_val") or 1000
    args.batch_size = args.batch_size or config["data"]["batch_size"]
    args.distilled_path = args.distilled_path or Path(
        config["distillation"].get("save_path", "outputs/student_distilled.pt")
    )

    checkpoints = [("student_baseline", args.baseline_path), ("student_distilled", args.distilled_path)]
    if args.include_old_distilled:
        checkpoints.append(("student_distilled_old", args.old_distilled_path))
    for item in args.extra_checkpoint:
        if "=" not in item:
            raise ValueError("--extra-checkpoint must be formatted as name=path")
        name, path = item.split("=", 1)
        checkpoints.append((name, Path(path)))

    missing = [str(path) for _, path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {', '.join(missing)}")

    print(f"Using device: {args.device}", flush=True)
    print(f"Validation samples: {args.max_samples_val}", flush=True)

    tokenizer = get_tokenizer(config["teacher"]["model_name"])
    raw_dataset = load_validation_dataset(args.max_samples_val)
    dataset = TokenizedTextDataset(raw_dataset, tokenizer, config["data"]["max_length"])
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    results = {
        "config": str(args.config),
        "device": args.device,
        "max_samples_val": args.max_samples_val,
        "prompts": list(DEFAULT_PROMPTS),
        "models": {},
    }

    for name, checkpoint_path in checkpoints:
        results["models"][name] = evaluate_checkpoint(
            name,
            checkpoint_path,
            config,
            dataloader,
            tokenizer,
            DEFAULT_PROMPTS,
            args,
        )

    json_path, md_path = write_reports(results, args)

    print("\nSummary", flush=True)
    for name, result in results["models"].items():
        metrics = result["metrics"]
        print(f"{name}: loss={metrics['loss']:.4f}, perplexity={metrics['perplexity']:.2f}", flush=True)
    print(f"\nWrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
