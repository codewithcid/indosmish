"""IndoSmish — complete SLM pipeline for a Kaggle T4 notebook (paper steps S4-S5, G3).

Runs the quantized small-language-model half of the benchmark that can't run on a 4GB
laptop: Qwen2.5-1.5B zero-shot, few-shot, and QLoRA fine-tuning, then reports full- vs
4-bit-precision deltas per class — the paper's headline quantization measurement.

HOW TO RUN (≈1 hour on a free Kaggle T4):
  1. kaggle.com → Create → New Notebook → Settings → Accelerator = GPU T4 x2.
  2. Paste this file into a cell (or upload it and `%run indosmish_slm.py`).
  3. Run all. Data is pulled from the public GitHub repo automatically.
  4. Copy the printed results table back into results/ and RESULTS.md.

Cell 0 — install (run once):
  !pip install -q -U transformers peft bitsandbytes trl accelerate datasets scikit-learn
"""
import json
import subprocess
import sys

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score, recall_score

REPO = "https://github.com/codewithcid/indosmish"
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
LABELS = ["ham", "spam", "smishing"]
SYSTEM = ("You are an SMS security classifier for code-mixed Malayalam-English messages. "
          "Classify each message as exactly one of: ham, spam, smishing. Answer one word only.")


# ---------- data ----------
def get_data():
    subprocess.run(["git", "clone", "--depth", "1", REPO, "/kaggle/working/repo"], check=False)
    import pandas as pd
    d = "/kaggle/working/repo/data/processed/splits"
    return (pd.read_csv(f"{d}/train.csv"), pd.read_csv(f"{d}/val.csv"), pd.read_csv(f"{d}/test.csv"))


def metrics(y, p):
    rep = classification_report(y, p, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "macro_f1": f1_score(y, p, labels=LABELS, average="macro", zero_division=0),
        "smishing_recall": recall_score(y, p, labels=["smishing"], average="macro", zero_division=0),
        "per_class": {l: {"recall": rep[l]["recall"], "f1": rep[l]["f1-score"]} for l in LABELS},
    }


def parse(txt):
    t = txt.lower()
    for l in LABELS:
        if l in t:
            return l
    return "ham"


# ---------- prompting (S4) ----------
def build_fewshot(train, k=6):
    per = max(1, k // 3); parts = []
    for l in LABELS:
        for t in train[train.label == l].text.head(per):
            parts.append(f"Message: {t}\nLabel: {l}")
    return "\n\n".join(parts) + "\n\n"


def run_prompting(model, tok, test, fewshot=""):
    preds = []
    for t in test.text:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{fewshot}Message: {t}\nLabel:"}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=4, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        preds.append(parse(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)))
    return preds


# ---------- QLoRA (S5) ----------
def train_qlora(train):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments)
    tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.pad_token or tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))

    def fmt(r):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Message: {r['text']}\nLabel:"},
                {"role": "assistant", "content": f" {r['label']}"}]
        return tok.apply_chat_template(msgs, tokenize=False)
    ds = Dataset.from_dict({"text": [fmt(r) for _, r in train.iterrows()]})
    ds = ds.map(lambda b: {**tok(b["text"], truncation=True, max_length=192, padding="max_length"),
                           "labels": tok(b["text"], truncation=True, max_length=192, padding="max_length")["input_ids"]},
                batched=True, remove_columns=["text"])
    Trainer(model=model, args=TrainingArguments(
        output_dir="/kaggle/working/qlora", per_device_train_batch_size=8, gradient_accumulation_steps=2,
        learning_rate=2e-4, num_train_epochs=3, fp16=True, logging_steps=20, report_to="none",
        optim="paged_adamw_8bit", save_strategy="no"), train_dataset=ds).train()
    return model, tok


def main():
    train, val, test = get_data()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    results = {}

    print("\n=== S4: zero-shot & few-shot (fp16) ===")
    tok = AutoTokenizer.from_pretrained(BASE)
    fp16 = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto")
    results["zeroshot_fp16"] = metrics(test.label.tolist(), run_prompting(fp16, tok, test))
    results["fewshot_fp16"] = metrics(test.label.tolist(), run_prompting(fp16, tok, test, build_fewshot(train)))
    del fp16; torch.cuda.empty_cache()

    print("\n=== S5: QLoRA fine-tune, then evaluate (4-bit) ===")
    model, tok = train_qlora(train)
    results["qlora_4bit"] = metrics(test.label.tolist(), run_prompting(model, tok, test))

    print("\n================ SLM RESULTS ================")
    for k, v in results.items():
        print(f"{k:16s} macro-F1={v['macro_f1']:.3f}  smishing-recall={v['smishing_recall']:.3f}")
    print("\nParticularly compare fewshot_fp16 vs qlora_4bit (adaptation gain at 4-bit).")
    with open("/kaggle/working/slm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved -> /kaggle/working/slm_results.json  (download + add to results/)")


if __name__ == "__main__":
    main()
