"""
Evaluation script: Compare teacher, student baseline, and distilled student.
"""

import argparse
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import get_tokenizer
from src.data.dataset import get_dataloader
from src.models.teacher import TeacherModel
from src.models.student import StudentModel
from src.evaluation.metrics import evaluate_model
from src.generation.generate import generate_text
from src.utils.config import load_config


def evaluate_all(config, device):
    """Evaluate and compare all models."""
    tokenizer = get_tokenizer(config["teacher"]["model_name"])
    val_loader = get_dataloader(
        "validation", tokenizer,
        batch_size=config["data"]["batch_size"],
        max_length=config["data"]["max_length"],
        max_samples=1000,  # Use subset for quick evaluation
    )

    results = {}

    # Teacher
    print("\n" + "=" * 60)
    print("EVALUATING TEACHER (GPT-2 Small)")
    print("=" * 60)
    teacher = TeacherModel(pretrained=True)
    teacher_path = config["training"].get("save_path", "outputs/teacher.pt")
    if Path(teacher_path).exists():
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        print(f"Loaded fine-tuned teacher from {teacher_path}")
    teacher = teacher.to(device)
    results["teacher"] = evaluate_model(teacher, val_loader, device=device)
    print(f"  Perplexity: {results['teacher']['perplexity']:.2f}")
    print(f"  Parameters: {results['teacher']['num_parameters']:,}")
    print(f"  Size: {results['teacher']['model_size_mb']:.1f} MB")

    # Student Baseline
    student_baseline_path = "outputs/student_baseline.pt"
    if Path(student_baseline_path).exists():
        print("\n" + "=" * 60)
        print("EVALUATING STUDENT BASELINE")
        print("=" * 60)
        student_bl = StudentModel(**config["student"]).to(device)
        student_bl.load_state_dict(torch.load(student_baseline_path, map_location=device))
        results["student_baseline"] = evaluate_model(student_bl, val_loader, device=device)
        print(f"  Perplexity: {results['student_baseline']['perplexity']:.2f}")
        print(f"  Parameters: {results['student_baseline']['num_parameters']:,}")

    # Distilled Student
    distilled_path = config["distillation"].get("save_path", "outputs/student_distilled.pt")
    if Path(distilled_path).exists():
        print("\n" + "=" * 60)
        print("EVALUATING DISTILLED STUDENT")
        print("=" * 60)
        student_dist = StudentModel(**config["student"]).to(device)
        student_dist.load_state_dict(torch.load(distilled_path, map_location=device))
        results["student_distilled"] = evaluate_model(student_dist, val_loader, device=device)
        print(f"  Perplexity: {results['student_distilled']['perplexity']:.2f}")
        print(f"  Parameters: {results['student_distilled']['num_parameters']:,}")

    # Generation comparison
    print("\n" + "=" * 60)
    print("GENERATION SAMPLES")
    print("=" * 60)
    prompts = ["Once upon a time", "The little dog", "She was very happy"]

    for prompt in prompts:
        print(f"\nPrompt: \"{prompt}\"")
        print("-" * 40)

        text = generate_text(teacher, tokenizer, prompt=prompt, max_new_tokens=80, device=device)
        print(f"  Teacher: {text}")

        if Path(distilled_path).exists():
            text = generate_text(student_dist, tokenizer, prompt=prompt, max_new_tokens=80, device=device)
            print(f"  Student (distilled): {text}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Params':<15} {'Perplexity':<12} {'Size (MB)':<10}")
    print("-" * 62)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['num_parameters']:>12,}   {metrics['perplexity']:<12.2f} {metrics['model_size_mb']:<10.1f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_all(config, device)
