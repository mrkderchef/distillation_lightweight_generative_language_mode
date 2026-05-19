# Project Decisions Log

This document records all key decisions made during the project setup and implementation, along with rationale for each choice.

---

## 1. Dataset: TinyStories

**Choice:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by Ronen Eldan & Yuanzhi Li (Microsoft Research, 2023)

**Why:**
- Specifically designed for training and evaluating small language models
- Contains ~2.1M synthetically generated short stories with coherent grammar and narrative structure
- Enables meaningful generative experiments on consumer hardware (no cluster needed)
- Published research demonstrates that models as small as 1M parameters can learn grammar and reasoning from this dataset
- Controlled vocabulary and sentence complexity allow clear observation of compression effects
- Well-suited for distillation research: large models excel while small models visibly degrade, making distillation improvements measurable

**Alternatives considered:**
- WikiText-103: Too complex for very small models, would not converge meaningfully
- OpenWebText: Too large and noisy, hard to control experiments
- BookCorpus: Copyright concerns, less controlled complexity

---

## 2. Teacher Model: GPT-2 Small (124M parameters)

**Choice:** GPT-2 Small pretrained, fine-tuned on TinyStories

**Why:**
- Well-understood decoder-only transformer architecture
- 124M parameters provides a meaningful "large" model relative to the student
- Pretrained weights available (reduces training time significantly)
- Widely used in distillation literature — results are comparable to published work
- Tokenizer (BPE, 50257 vocab) works well for English story text
- Can be fine-tuned on a single consumer GPU in reasonable time
- Clear architectural relationship to the student (same family, different scale)

**Alternatives considered:**
- GPT-2 Medium (355M): Too expensive to fine-tune on limited hardware, marginal benefit for TinyStories
- TinyLlama (1.1B): Overkill for this dataset, slow teacher inference during distillation
- Custom transformer: No pretrained weights, adds unnecessary training overhead
- DistilGPT-2: Already distilled — defeats the purpose of studying distillation

---

## 3. Student Model: Custom 4-layer GPT-2 Architecture (~19M parameters)

**Choice:** GPT2Config with 4 layers, 4 attention heads, hidden_size=256, context_length=512

**Why:**
- ~19M parameters = ~6.5x compression ratio vs teacher (meaningful compression)
- 4 layers is minimum for capturing multi-level language patterns (syntax, short-range semantics)
- Hidden size 256 with 4 heads gives 64-dim per head (proven effective in small transformers)
- Same tokenizer as teacher (no token mapping complexity)
- GPT2LMHeadModel architecture ensures compatible logit shapes for distillation
- Context length 512 is sufficient for TinyStories (most stories < 300 tokens)
- Small enough to show clear degradation without distillation, large enough to benefit from it

**Alternatives considered:**
- 2-layer model (~8M): Too weak — may not learn meaningful patterns even with distillation
- 6-layer / 384 hidden (~45M): Too close to teacher, compression ratio not impressive
- Different architecture (e.g., RWKV, Mamba): Would complicate distillation (different output shapes)

---

## 4. Tokenizer: GPT-2 BPE Tokenizer (50257 vocab)

**Choice:** Reuse GPT-2's pretrained BPE tokenizer for both teacher and student

**Why:**
- Shared vocabulary between teacher and student is essential for logit-level distillation
- BPE handles unseen words gracefully via subword decomposition
- 50257 tokens is standard — no need to retrain or reduce vocabulary
- Pad token set to EOS token (standard practice for GPT-2 family)
- Avoids token alignment issues that would arise with different tokenizers

**Alternatives considered:**
- Smaller custom vocabulary: Would require retraining teacher, adds complexity
- Character-level tokenization: Sequences become too long, context window issues

---

## 5. Distillation Strategy: Logit-based KL Divergence with Temperature Scaling

**Choice:** Combined loss: α·T²·KL(teacher_soft || student_soft) + (1-α)·CE(student, labels)

**Why:**
- Standard and well-proven approach from Hinton et al. (2015)
- KL divergence on softened distributions transfers "dark knowledge" (inter-token relationships)
- Temperature scaling reveals teacher's uncertainty and token similarity structure
- Combined loss ensures student still learns from ground truth (prevents collapse)
- T² scaling compensates for gradient magnitude reduction at higher temperatures
- Simple to implement and debug compared to feature-based or attention-based distillation

**Alternatives considered:**
- Feature-based distillation (FitNets): Requires dimension matching layers, harder to tune
- Attention transfer: Teacher/student have different number of heads and layers
- Progressive distillation: More complex training schedule, diminishing returns for this scale
- Patient Knowledge Distillation: Adds complexity with multi-layer matching

---

## 6. Default Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Temperature (T) | 1.5 | Best early results came from slightly softened teacher distributions without making targets too uniform |
| Alpha | 0.2-0.3 | Lower teacher weight worked better than equal weighting; 0.2 is best so far |
| Learning Rate | 2e-4-3e-4 | Lower distillation LR was more stable than the initial 5e-4 setting |
| Weight Decay | 0.01 | Light regularization, standard for language models |
| Batch Size | 16 | Fits in 8GB VRAM with mixed precision |
| Context Length | 512 | Covers 95%+ of TinyStories, balances memory vs coverage |
| Training Epochs (Teacher) | 1 | GPT-2 is pretrained; pilot run fine-tunes quickly on a 30k TinyStories subset |
| Training Epochs (Student Baseline) | 1 | Baseline pilot run is used as a comparable reference |
| Distillation Epochs | 3 | Gives the student more time to absorb teacher signal without making runs too long |
| Mixed Precision | FP16 | 2x memory savings, no quality loss for training |

### Pilot Results Update

The original default table above has been revised by pilot experiments on a
30k-example TinyStories training subset. All student checkpoints below use the
same ~16M parameter architecture and were evaluated on the same 1000-example
TinyStories validation subset.

| Model | Distillation settings | Loss | Perplexity |
|---|---|---:|---:|
| Student baseline | CE only | 2.6797 | 14.58 |
| Distilled old | T=2.0, alpha=0.5, lr=5e-4 | 2.7596 | 15.79 |
| Distilled current config | T=1.5, alpha=0.3, lr=2e-4 | 2.6268 | 13.83 |
| Distilled best so far | T=1.5, alpha=0.2, lr=3e-4 | 2.4668 | 11.78 |

**Decision:** Keep `outputs/student_distilled_t15_a02.pt` as the best checkpoint
so far. The experiment suggests this setup benefits from conservative teacher
weighting; equal CE/KL weighting over-emphasized the teacher signal and performed
worse than the baseline.

---

## 7. Evaluation Metrics

**Choice:** Perplexity as primary metric, supplemented with generation quality samples

**Why:**
- Perplexity is the standard metric for language models (directly measures prediction quality)
- Lower perplexity = better next-token prediction = better language understanding
- Easily comparable across models of different sizes
- Generation samples provide qualitative evidence of coherence, grammar, creativity
- Model size (parameters, MB) quantifies compression ratio

**Alternatives considered:**
- BLEU/ROUGE: Designed for translation/summarization, not open-ended generation
- BERTScore: Requires reference text, not suitable for unconstrained generation
- Human evaluation: Time-consuming, subjective, not reproducible

---

## 8. Hardware & Training Setup

**Choice:** Single GPU training with mixed precision (AMP), PyTorch native

**Why:**
- Project designed for consumer hardware (single RTX-class GPU)
- PyTorch AMP (autocast + GradScaler) provides near-free 2x speedup
- No distributed training complexity needed at this scale
- Hugging Face `datasets` library handles efficient data loading with memory mapping
- Checkpointing best model by validation loss

---

## 9. Framework: PyTorch + Hugging Face Transformers

**Choice:** PyTorch as training framework, Hugging Face for model/tokenizer/data loading

**Why:**
- PyTorch offers full control over training loop (essential for custom distillation loss)
- Hugging Face provides pretrained GPT-2 weights and TinyStories dataset access
- GPT2LMHeadModel allows easy configuration of custom architectures via GPT2Config
- Well-documented, widely used in research, easy to debug
- No need for higher-level frameworks (Lightning, Trainer API) — custom distillation loop is simple enough

**Alternatives considered:**
- TensorFlow/Keras: Less common in current LLM research, fewer pretrained model options
- Hugging Face Trainer API: Too opinionated for custom distillation loss function
- JAX/Flax: Higher performance ceiling but steeper learning curve, overkill for this project
