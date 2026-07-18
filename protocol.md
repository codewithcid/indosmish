# Evaluation Protocol (FROZEN — S2 / G8)

This file is frozen **before any model is trained**. Changes after the first training
run require a dated amendment note at the bottom; results produced under a different
protocol version are not comparable.

## 1. Task

Tri-class single-message classification: `ham` (0), `spam` (1), `smishing` (2).
Input is the raw SMS text only (no sender metadata, no URL resolution).

## 2. Metrics

- **Primary:** macro-averaged F1.
- **Always reported alongside:** accuracy, per-class precision/recall/F1, and
  **smishing-class recall** (highlighted — it is the minority boundary class).
- Never report accuracy alone. No pooled averages across script arms.

## 3. Splits

- 70 / 15 / 15 train / validation / test, stratified by label.
- **Deduplication precedes splitting** (exact-normalized + near-duplicate signature;
  see `src/indosmish/data/dedup_split.py`). This avoids the near-duplicate leakage
  documented for the 2011 SMS Spam Collection.
- Random seed **42** everywhere. Frozen split files live in
  `data/processed/splits/` and are committed; all experiments read those files.
- Augmented (synthetic) messages are **train-only**. The validation and test sets
  contain only human-verified messages. No synthetic message, nor any near-duplicate
  of one, may enter val/test.

## 4. Script arms (controlled variable)

Every model is evaluated separately on:

| Arm | Description |
|-----|-------------|
| `roman` | Malayalam–English code-mix in Latin script (native SMS register) |
| `native` | Malayalam script (transliterated via IndicXlit where not originally native) |
| `xlit-norm` | Roman input normalized to native script by IndicXlit before classification |

Per-arm results are reported in full; cross-arm deltas are the headline comparison
(tests the transliteration-deficit prediction of Krishnan et al. in a security setting).

## 5. Model roster

1. TF-IDF + Random Forest (classical floor)
2. MuRIL-base fine-tuned
3. IndicBERT fine-tuned
4. Qwen2.5-1.5B-Instruct: zero-shot, few-shot (k=8, fixed exemplars), QLoRA fine-tuned
5. Best SLM at FP16 vs 4-bit (GGUF q4_K_M; AWQ if kernel support allows) —
   **quantization delta reported per class and per script arm**

## 6. Deployment reporting (any "deployable" claim)

Report together, on named hardware, at the deployed precision:
model file size (MB), resident memory (MB), mean + p95 latency per message over
≥200 messages after ≥20 warm-up messages, and numerical precision.
Named targets: AMD Ryzen 7 6800H (CPU arm), RTX 3050 Laptop 4 GB (GPU arm),
HF Spaces free CPU tier (demo arm).

## 7. Robustness suite (stretch, G5)

If run: homoglyph substitution, zero-width character injection, and LLM paraphrase,
applied to the test set only, executed **identically at FP16 and 4-bit**.

## 8. Contamination disclosure

For every LLM-era model: state the model's training-data cutoff and note that public
seed corpora (Mishra & Soni 2022, Dravidian Spam SMS 2023) may be in pre-training data.
Synthetic augmented messages postdate all model cutoffs and are contamination-free;
the val/test sets are newly assembled and transformed, which mitigates but does not
eliminate contamination risk. State this in the write-up.

---
*Protocol v1.0 — frozen 2026-07-18. No amendments.*
