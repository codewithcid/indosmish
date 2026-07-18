"""S4-S5/G3 — QLoRA fine-tune Qwen2.5-1.5B on Kaggle T4, then export.

Run this in a Kaggle notebook (GPU T4 x2 or P100). Upload data/processed/splits/
as a Kaggle dataset, adjust DATA_DIR, then run top to bottom.

Cell 0 (install):
    !pip install -q -U transformers peft bitsandbytes trl accelerate datasets

This trains a 4-bit NF4 backbone with LoRA adapters (paper's QLoRA, ref [36]),
saves the adapter, and merges to fp16 for later GGUF/AWQ export.
"""
import json
import os

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

DATA_DIR = "/kaggle/input/indosmish-splits"   # <-- adjust to your uploaded dataset
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = "/kaggle/working/qwen-indosmish"
LABELS = ["ham", "spam", "smishing"]

SYSTEM = (
    "You are an SMS security classifier for code-mixed Malayalam-English messages. "
    "Classify each message as exactly one of: ham, spam, smishing. Answer one word only."
)


def load_csv(name):
    import csv
    rows = []
    with open(os.path.join(DATA_DIR, f"{name}.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def to_chat(tok, rows):
    texts = []
    for r in rows:
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Message: {r['text']}\nLabel:"},
            {"role": "assistant", "content": f" {r['label']}"},
        ]
        texts.append(tok.apply_chat_template(msgs, tokenize=False))
    return Dataset.from_dict({"text": texts})


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = to_chat(tok, load_csv("train"))

    def tok_fn(b):
        out = tok(b["text"], truncation=True, max_length=256, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=OUT,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        report_to="none",
        optim="paged_adamw_8bit",
    )
    Trainer(model=model, args=args, train_dataset=ds).train()

    model.save_pretrained(f"{OUT}/adapter")
    tok.save_pretrained(f"{OUT}/adapter")

    # Merge to fp16 for GGUF/AWQ export (do on CPU/fresh load to avoid 4-bit merge issues).
    print("Adapter saved. To export: reload base in fp16, PeftModel.from_pretrained, "
          "merge_and_unload(), save_pretrained -> then llama.cpp convert to GGUF q4_k_m.")


if __name__ == "__main__":
    main()
