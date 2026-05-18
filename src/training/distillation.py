"""
Knowledge distillation trainer for training student from teacher.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


class DistillationTrainer:
    """Trainer implementing knowledge distillation for generative language models."""

    def __init__(
        self,
        teacher_model,
        student_model,
        train_dataloader,
        val_dataloader=None,
        temperature=2.0,
        alpha=0.5,
        lr=5e-4,
        weight_decay=0.01,
        device="cuda",
        use_amp=True,
    ):
        """
        Args:
            teacher_model: Trained teacher model (frozen).
            student_model: Student model to train.
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.
            temperature: Temperature for softening distributions.
            alpha: Weight for distillation loss (1-alpha for CE loss).
            lr: Learning rate.
            weight_decay: AdamW weight decay.
            device: Device to train on.
            use_amp: Whether to use mixed precision.
        """
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        self.use_amp = use_amp

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.optimizer = AdamW(self.student.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = GradScaler() if use_amp else None

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Compute combined distillation + cross-entropy loss.

        L = alpha * T^2 * KL(teacher_soft || student_soft) + (1 - alpha) * CE(student, labels)
        """
        T = self.temperature

        # Soft targets from teacher and student
        teacher_soft = F.log_softmax(teacher_logits / T, dim=-1)
        student_soft = F.log_softmax(student_logits / T, dim=-1)

        # KL divergence loss (scaled by T^2)
        kl_loss = F.kl_div(
            student_soft,
            teacher_soft,
            log_target=True,
            reduction="batchmean",
        ) * (T ** 2)

        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # Combined loss
        loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        return loss, kl_loss.item(), ce_loss.item()

    def train_epoch(self):
        """Train for one epoch with distillation."""
        self.student.train()
        total_loss = 0.0
        total_kl = 0.0
        total_ce = 0.0
        num_batches = 0

        for batch in tqdm(self.train_dataloader, desc="Distillation Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    with torch.no_grad():
                        teacher_logits = self.teacher.get_logits(input_ids, attention_mask)
                    student_logits = self.student.get_logits(input_ids, attention_mask)
                    loss, kl, ce = self.distillation_loss(student_logits, teacher_logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                with torch.no_grad():
                    teacher_logits = self.teacher.get_logits(input_ids, attention_mask)
                student_logits = self.student.get_logits(input_ids, attention_mask)
                loss, kl, ce = self.distillation_loss(student_logits, teacher_logits, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_kl += kl
            total_ce += ce
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_kl = total_kl / num_batches
        avg_ce = total_ce / num_batches
        return avg_loss, avg_kl, avg_ce

    @torch.no_grad()
    def evaluate(self):
        """Evaluate student on validation set."""
        if self.val_dataloader is None:
            return None

        self.student.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.student(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1

        return total_loss / num_batches

    def train(self, num_epochs=10, save_path=None):
        """Full distillation training loop."""
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            avg_loss, avg_kl, avg_ce = self.train_epoch()
            val_loss = self.evaluate()

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | KL: {avg_kl:.4f} | CE: {avg_ce:.4f}",
                end="",
            )
            if val_loss is not None:
                print(f" | Val Loss: {val_loss:.4f}", end="")
            print()

            if save_path and val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.student.state_dict(), save_path)
                print(f"  -> Saved best student to {save_path}")
