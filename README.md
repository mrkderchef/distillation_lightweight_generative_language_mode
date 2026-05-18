# Knowledge Distillation for Lightweight Generative Language Models

## Overview

This project investigates whether **knowledge distillation** can compress transformer-based generative language models while preserving important language generation capabilities. The focus is on autoregressive language modeling using the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

Unlike simple classification tasks, this project targets genuine generative transformer behavior including grammar, syntax, semantic understanding, contextual coherence, and next-token prediction.

### Research Question

> Can knowledge distillation preserve generative language modeling capabilities in highly compressed transformer architectures suitable for edge or mobile deployment?

---

## Motivation

Modern large language models achieve impressive text generation but require enormous memory, expensive GPUs, high inference cost, and large energy consumption. This project investigates whether smaller transformer models can inherit generative capabilities from larger teacher models through knowledge distillation — enabling efficient AI for edge and mobile deployment.

---

## Project Structure

```
project-root/
├── data/                  # Dataset storage (TinyStories)
├── notebooks/             # Jupyter notebooks for exploration
├── src/
│   ├── data/              # Dataset loading and tokenization
│   ├── models/            # Teacher (GPT-2) and Student architectures
│   ├── training/          # Standard trainer and distillation trainer
│   ├── evaluation/        # Metrics (perplexity, model size, etc.)
│   ├── generation/        # Text generation utilities
│   └── utils/             # Configuration and logging
├── outputs/               # Trained model checkpoints
├── reports/               # Generated reports and figures
├── config.yaml            # Project configuration
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Architecture

### Teacher Model — GPT-2 Small

| Property | Value |
|---|---|
| Parameters | ~124M |
| Layers | 12 |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Context Length | 1024 |
| FP32 Size | ~500MB |

### Student Model — Lightweight Decoder Transformer

| Property | Value |
|---|---|
| Parameters | ~5M–20M |
| Layers | 2–6 |
| Hidden Size | 256–384 |
| Attention Heads | 4 |
| Context Length | 256–512 |
| FP32 Size | ~20–80MB |

---

## Distillation Pipeline

1. **Teacher Training/Fine-Tuning** — Train or fine-tune GPT-2 Small on TinyStories for strong generative quality.
2. **Student Baseline** — Train the student model with standard next-token prediction (cross-entropy only).
3. **Distillation Training** — Train the student using both ground truth tokens and teacher soft distributions.

### Distillation Loss

$$L = \alpha \cdot T^2 \cdot \text{KL}(p_{\text{teacher}} \| p_{\text{student}}) + (1 - \alpha) \cdot L_{\text{CE}}$$

Where:
- $T$ = temperature for softening distributions
- $\alpha$ = weight balancing distillation vs. cross-entropy loss
- $p_{\text{teacher}}$ and $p_{\text{student}}$ are softmax distributions at temperature $T$

---

## Experiments

| Experiment | Description |
|---|---|
| Teacher Evaluation | Evaluate fine-tuned GPT-2 on TinyStories |
| Student Baseline | Train student with CE loss only |
| Distilled Student | Train student with distillation from teacher |
| Temperature Sweep | Vary T ∈ {1, 2, 3, 5, 10} and compare |
| Compression Comparison | Compare parameter count, memory, perplexity |

### Metrics

- Perplexity
- Generation quality (qualitative samples)
- Coherence
- Parameter count
- Memory usage (FP32 / FP16 / INT8)
- Inference efficiency

---

## Installation

```bash
# Clone the repository
git clone https://github.com/mrkderchef/distillation_lightweight_generative_language_mode.git
cd distillation_lightweight_generative_language_mode

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Configuration

Edit `config.yaml` to adjust hyperparameters:

```yaml
data:
  max_length: 512
  batch_size: 16

student:
  n_layers: 4
  n_heads: 4
  hidden_size: 256

distillation:
  temperature: 2.0
  alpha: 0.5
  epochs: 10
```

### Training the Teacher

```python
from src.data.dataset import get_dataloader
from src.data.tokenizer import get_tokenizer
from src.models.teacher import TeacherModel
from src.training.trainer import Trainer

tokenizer = get_tokenizer("gpt2")
train_loader = get_dataloader("train", tokenizer, batch_size=16)
val_loader = get_dataloader("validation", tokenizer, batch_size=16)

teacher = TeacherModel(pretrained=True)
trainer = Trainer(teacher, train_loader, val_loader)
trainer.train(num_epochs=5, save_path="outputs/teacher.pt")
```

### Distillation

```python
from src.models.student import StudentModel
from src.training.distillation import DistillationTrainer

student = StudentModel(n_layers=4, n_heads=4, hidden_size=256)
distiller = DistillationTrainer(
    teacher_model=teacher,
    student_model=student,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    temperature=2.0,
    alpha=0.5,
)
distiller.train(num_epochs=10, save_path="outputs/student_distilled.pt")
```

### Text Generation

```python
from src.generation.generate import generate_text

text = generate_text(
    model=student,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_new_tokens=100,
    temperature=0.8,
)
print(text)
```

---

## Expected Results

- Smaller models lose significant generative quality when trained alone
- Knowledge distillation noticeably improves student model performance
- Real compression tradeoffs become visible
- Generative capabilities can partially survive aggressive compression

---

## Hardware Requirements

- GPU with at least 8GB VRAM recommended (e.g., NVIDIA RTX 3060+)
- CPU training possible but significantly slower
- Mixed precision (FP16) supported for faster training

---

## License

This project is for academic and research purposes.

---

## Author

[@mrkderchef](https://github.com/mrkderchef)
