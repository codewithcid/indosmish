# IndoSmish — Code-Mixed Malayalam–English Smishing Detection

Implementation of the benchmark specified in:

> Sidhardh S et al., "From SMS Spam Filters to Quantized LLMs: A Structured Survey of
> Smishing Detection for Code-Mixed Indian Languages and Resource-Efficient Deployment,"
> IEEE Access, 2026 (Section XIII, steps S1–S6; gaps G1–G4).

This repo builds the first tri-class (ham / spam / smishing), dual-script
Malayalam–English SMS corpus, establishes Indic encoder baselines (MuRIL, IndicBERT),
fine-tunes and 4-bit-quantizes a small language model (Qwen2.5-1.5B via QLoRA), and
reports deployment-grade profiling on named commodity hardware.

## Pipeline (maps to paper roadmap)

| Step | Paper gap | What | Where it runs |
|------|-----------|------|---------------|
| S2 | G8 | Frozen evaluation protocol → [protocol.md](protocol.md) | — |
| S1 | G1 | Corpus build: source merge → augmentation → transliteration → dedup/split | local CPU + Gemini free API |
| S3 | G2 | TF-IDF+RF floor; MuRIL / IndicBERT fine-tuning, script as controlled variable | local RTX 3050 |
| S4 | G3a | Qwen2.5-1.5B zero/few-shot, then QLoRA fine-tune | Kaggle T4 (free) |
| S5 | G3b | 4-bit PTQ (GGUF q4_K / AWQ); FP16-vs-4-bit delta per class & per script | Kaggle + local |
| S6 | G4 | Latency / RSS / model size at deployed precision on named hardware | local (Ryzen 7 6800H + RTX 3050) |
| demo | — | Gradio app on Hugging Face Spaces | HF free CPU tier |

## Repo layout

```
indosmish/
├── protocol.md              # FROZEN evaluation protocol (S2) — read first
├── configs/default.yaml     # paths, labels, hyperparameters
├── data/
│   ├── DATA.md              # source datasets + manual download steps
│   ├── raw/                 # downloaded source corpora (gitignored)
│   ├── interim/             # unified schema, augmentation output awaiting review
│   └── processed/           # final deduped corpus + frozen splits
├── src/indosmish/
│   ├── data/                # build_corpus, augment, transliterate, dedup_split
│   ├── models/              # classical baseline, encoder fine-tuning, SLM prompting
│   └── eval/                # metrics, device profiling
├── kaggle/qlora_train.py    # QLoRA training script (run on Kaggle T4)
├── app/app.py               # Gradio demo (deploys to HF Spaces)
└── results/                 # metric JSONs + markdown tables (committed)
```

## Quickstart (local)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# GPU torch (CUDA 12.x):
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 1. Follow data/DATA.md to place raw datasets, then:
python -m indosmish.data.build_corpus
python -m indosmish.data.augment        # needs GEMINI_API_KEY env var
python -m indosmish.data.dedup_split

# 2. Baselines
python -m indosmish.models.classical
python -m indosmish.models.train_encoder --model google/muril-base-cased
python -m indosmish.models.train_encoder --model ai4bharat/indic-bert

# 3. Profiling
python -m indosmish.eval.profile_device --model results/best
```

## Hardware (profiling targets — named per protocol)

- **Laptop:** AMD Ryzen 7 6800H, NVIDIA RTX 3050 Laptop 4 GB, Windows 11
- **Cloud (training only):** Kaggle T4 16 GB
- **Demo:** Hugging Face Spaces free CPU tier

## Team

Sidhardh S, Advaith Kamath, Joel G. Bert, T. Amarsainadh — VIT-AP University (SCOPE).
Supervisors: Bileesh P. Babu, Gokul Yenduri.
