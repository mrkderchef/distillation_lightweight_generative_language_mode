"""
Main training script for the distillation project.
Orchestrates the full pipeline: data loading → teacher training → student training → distillation.

Usage:
    python scripts/train.py --config config.yaml --stage teacher
    python scripts/train.py --config config.yaml --stage student_baseline
    python scripts/train.py --config config.yaml --stage distillation
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
from src.training.trainer import Trainer
from src.training.distillation import DistillationTrainer
from src.evaluation.metrics import evaluate_model, compute_perplexity
from src.generation.generate import generate_text
from src.utils.config import load_config
from src.utils.logging import setup_logger


def train_teacher(config, device):
    """Train/fine-tune the teacher model on TinyStories."""
    logger = setup_logger("teacher_training")
    logger.info("Starting teacher training...")

    tokenizer = get_tokenizer(config["teacher"]["model_name"])
    train_loader = get_dataloader(
        "train", tokenizer,
        batch_size=config["data"]["batch_size"],
        max_length=config["data"]["max_length"],
        max_samples=config["data"]["max_samples_train"],
    )
    val_loader = get_dataloader(
        "validation", tokenizer,
        batch_size=config["data"]["batch_size"],
        max_length=config["data"]["max_length"],
        max_samples=config["data"]["max_samples_val"],
    )

    teacher = TeacherModel(pretrained=config["teacher"]["pretrained"])
    logger.info(f"Teacher parameters: {teacher.num_parameters:,}")

    trainer = Trainer(
        model=teacher,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        device=device,
        use_amp=config["training"]["use_amp"],
    )

    save_path = config["training"].get("save_path", "outputs/teacher.pt")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.train(num_epochs=config["training"]["epochs"], save_path=save_path)

    # Evaluate
    logger.info("Evaluating teacher model...")
    metrics = evaluate_model(teacher, val_loader, device=device)
    logger.info(f"Teacher perplexity: {metrics['perplexity']:.2f}")

    # Generate sample
    sample = generate_text(teacher, tokenizer, prompt="Once upon a time", device=device)
    logger.info(f"Sample generation:\n{sample}")

    return teacher


def train_student_baseline(config, device):
    """Train student model with standard CE loss (no distillation)."""
    logger = setup_logger("student_baseline")
    logger.info("Starting student baseline training...")

    tokenizer = get_tokenizer(config["teacher"]["model_name"])
    train_loader = get_dataloader(
        "train", tokenizer,
        batch_size=config["data"]["batch_size"],
        max_length=config["data"]["max_length"],
        max_samples=config["data"]["max_samples_train"],
    )
    val_loader = get_dataloader(
        "validation", tokenizer,
        batch_size=config["data"]["batch_size"],
        max_length=config["data"]["max_length"],
        max_samples=config["data"]["max_samples_val"],
    )

    student = StudentModel(**config["student"])
    logger.info(f"Student parameters: {student.num_parameters:,}")

    trainer = Trainer(
        model=student,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        device=device,
        use_amp=config["training"]["use_amp"],
    )

    save_path = "outputs/student_baseline.pt"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    baseline_epochs = config["training"].get(
        "student_baseline_epochs",
        config["distillation"]["epochs"],
    )
    trainer.train(num_epochs=baseline_epochs, save_path=save_path)

    # Evaluate
    metrics = evaluate_model(student, val_loader, device=device)
    logger.info(f"Student baseline perplexity: {metrics['perplexity']:.2f}")

    return student


def train_distillation(config, device):
    """Train student model using knowledge distillation from teacher."""
    logger = setup_logger("distillation")
    logger.info("Starting distillation training...")

    tokenizer = get_tokenizer(config["teacher"]["model_name"])
    train_loader = get_dataloader(
        "train", tokenizer,
        batch_size=config["data"]["batch_size"],
        max_length=config["data"]["max_length"],
        max_samples=config["data"]["max_samples_train"],
    )
    val_loader = get_dataloader(
        "validation", tokenizer,
        batch_size=config["data"]["batch_size"],
        max_length=config["data"]["max_length"],
        max_samples=config["data"]["max_samples_val"],
    )

    # Load trained teacher
    teacher = TeacherModel(pretrained=config["teacher"]["pretrained"])
    teacher_path = config["training"].get("save_path", "outputs/teacher.pt")
    if Path(teacher_path).exists():
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        logger.info(f"Loaded teacher from {teacher_path}")
    else:
        logger.warning("No trained teacher found, using pretrained GPT-2 as teacher")

    # Create student
    student = StudentModel(**config["student"])
    logger.info(f"Teacher: {teacher.num_parameters:,} params")
    logger.info(f"Student: {student.num_parameters:,} params")
    logger.info(f"Compression: {teacher.num_parameters / student.num_parameters:.1f}x")

    # Distillation
    distiller = DistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        temperature=config["distillation"]["temperature"],
        alpha=config["distillation"]["alpha"],
        lr=config["distillation"]["lr"],
        device=device,
        use_amp=config["training"]["use_amp"],
    )

    save_path = config["distillation"].get("save_path", "outputs/student_distilled.pt")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    distiller.train(num_epochs=config["distillation"]["epochs"], save_path=save_path)

    # Evaluate
    metrics = evaluate_model(student, val_loader, device=device)
    logger.info(f"Distilled student perplexity: {metrics['perplexity']:.2f}")

    # Generate sample
    sample = generate_text(student, tokenizer, prompt="Once upon a time", device=device)
    logger.info(f"Sample generation:\n{sample}")

    return student


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["teacher", "student_baseline", "distillation", "all"],
                        help="Training stage to run")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.stage == "teacher" or args.stage == "all":
        train_teacher(config, device)

    if args.stage == "student_baseline" or args.stage == "all":
        train_student_baseline(config, device)

    if args.stage == "distillation" or args.stage == "all":
        train_distillation(config, device)
