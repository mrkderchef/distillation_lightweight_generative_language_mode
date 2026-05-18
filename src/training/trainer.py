"""
Standard trainer for teacher and student baseline training.
"""

import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


class Trainer:
    """Standard autoregressive language model trainer."""

    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        lr=5e-4,
        weight_decay=0.01,
        device="cuda",
        use_amp=True,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.use_amp = use_amp

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = GradScaler() if use_amp else None

    def train_epoch(self):
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.train_dataloader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set. Returns average loss."""
        if self.val_dataloader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1

        return total_loss / num_batches

    def train(self, num_epochs=5, save_path=None):
        """Full training loop."""
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()

            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}", end="")
            if val_loss is not None:
                print(f" | Val Loss: {val_loss:.4f}", end="")
            print()

            if save_path and val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> Saved best model to {save_path}")
