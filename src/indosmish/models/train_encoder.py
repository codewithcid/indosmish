"""S3/G2 — Fine-tune an Indic encoder (MuRIL / IndicBERT) for tri-class detection.

Tuned for a 4 GB RTX 3050: max_length=128, batch 16, fp16, gradient accumulation.
Trains on all script arms combined, evaluates per arm (protocol section 4), so the
transliteration deficit is measured rather than assumed.

Run:
  python -m indosmish.models.train_encoder --model google/muril-base-cased --tag muril
  python -m indosmish.models.train_encoder --model ai4bharat/indic-bert --tag indicbert
Output: results/best/<tag>/  (saved model)  +  results/<tag>_{arm}.json
"""
import argparse

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ..config import load_config, resolve
from ..data.schema import LABELS
from ..eval.metrics import compute_metrics, print_summary, save_result
from .data_utils import filter_arm, load_split

L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for l, i in L2I.items()}


def _to_ds(df, tok, max_len):
    ds = Dataset.from_dict({"text": df["text"].tolist(),
                            "label": [L2I[x] for x in df["label"]]})
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len),
                  batched=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    cfg = load_config()
    ecfg = cfg["encoder"]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=I2L, label2id=L2I,
    )

    train, val = load_split("train"), load_split("val")
    ds_tr = _to_ds(train, tok, ecfg["max_length"])
    ds_va = _to_ds(val, tok, ecfg["max_length"])

    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=-1)
        m = compute_metrics([I2L[i] for i in labels], [I2L[i] for i in pred])
        return {"macro_f1": m["macro_f1"], "smishing_recall": m["smishing_recall"]}

    out_dir = resolve(f"results/best/{args.tag}")
    targs = TrainingArguments(
        output_dir=str(resolve(f"runs/{args.tag}")),
        per_device_train_batch_size=ecfg["batch_size"],
        per_device_eval_batch_size=ecfg["batch_size"],
        gradient_accumulation_steps=ecfg["grad_accum"],
        learning_rate=float(ecfg["lr"]),
        num_train_epochs=ecfg["epochs"],
        fp16=ecfg["fp16"] and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=25,
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds_tr,
                      eval_dataset=ds_va, tokenizer=tok, compute_metrics=hf_metrics)
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    # Per-arm test evaluation.
    test = load_split("test")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    for arm in ("roman", "native"):
        part = filter_arm(test, arm)
        if len(part) < 5:
            print(f"[skip] arm={arm}: {len(part)} rows")
            continue
        preds = []
        for i in range(0, len(part), 32):
            batch = part["text"].tolist()[i:i + 32]
            enc = tok(batch, truncation=True, max_length=ecfg["max_length"],
                      padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            preds += [I2L[int(x)] for x in logits.argmax(-1).cpu()]
        m = compute_metrics(part["label"].tolist(), preds)
        m["model"], m["arm"] = args.tag, arm
        print_summary(f"{args.tag} [{arm}]", m)
        save_result(m, f"results/{args.tag}_{arm}.json")


if __name__ == "__main__":
    main()
