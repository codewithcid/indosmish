# Running the SLM half on Kaggle (S4–S5, G3)

The 4-bit QLoRA fine-tuning of Qwen2.5-1.5B needs a GPU with bitsandbytes — it cannot run
on a 4 GB laptop card or a CPU sandbox. A free Kaggle T4 does it in ~1 hour. Everything
else (corpus, encoders, ONNX, transliteration, deployment) already ran locally.

## Steps (one notebook, ~1 hour)

1. Go to **kaggle.com → Create → New Notebook**.
2. **Settings (right panel) → Accelerator → GPU T4 x2**. Turn **Internet: On**.
3. First cell — install:
   ```
   !pip install -q -U transformers peft bitsandbytes trl accelerate datasets scikit-learn
   ```
4. Second cell — paste the entire contents of `indosmish_slm.py`, then add at the bottom:
   ```python
   main()
   ```
   (or upload `indosmish_slm.py` as a notebook data file and `%run indosmish_slm.py`).
5. **Run All.** The script clones the public repo for the frozen splits automatically —
   no manual data upload needed.

## What it produces

- Qwen2.5-1.5B **zero-shot** and **few-shot** (fp16) macro-F1 + smishing recall
- **QLoRA fine-tuned, 4-bit** macro-F1 + smishing recall
- `slm_results.json` (download it)

## After it runs

Download `slm_results.json`, drop it into `results/`, and it slots into the master
comparison alongside the classical / MuRIL / IndicBERT rows. The headline comparison is
**few-shot fp16 vs QLoRA 4-bit** — task adaptation gain at the deployed precision (G3).

The local encoder results already give the strongest numbers (IndicBERT 0.892 / 0.868);
the SLM rows complete the generational picture your survey lays out.
