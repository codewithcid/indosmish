"""S5/G3 + deployment — Export a fine-tuned encoder to ONNX and int8-quantize it.

Two payoffs:
  1. Deployment artifact: a small int8 ONNX model that runs under onnxruntime (no torch)
     inside Render's free 512MB tier.
  2. Paper result: the FP32-vs-int8 quantization delta on security text (G3), per class
     and per script arm, reported into results/.

Run:  python -m indosmish.models.export_onnx --src results/best/indicbert --tag indicbert
Output:
  app/backend/model/onnx/        fp32 ONNX + tokenizer
  app/backend/model/onnx-int8/   int8 ONNX + tokenizer   (this is what Render serves)
  results/onnx_quant_{tag}_{arm}.json
"""
import argparse
from pathlib import Path

import numpy as np
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

from ..config import resolve
from ..data.schema import LABELS
from ..eval.metrics import compute_metrics, print_summary, save_result
from .data_utils import filter_arm, load_split

I2L = {i: l for i, l in enumerate(LABELS)}


def _load_tokenizer(src: Path, base: str | None):
    """Robust load: some saved ALBERT tokenizers trip a transformers-5.x fast-tokenizer
    bug. Try the saved dir (fast, then slow), then the base hub id if given."""
    for target in (src, base):
        if target is None:
            continue
        for kwargs in ({}, {"use_fast": False}):
            try:
                return AutoTokenizer.from_pretrained(target, **kwargs)
            except (AttributeError, ValueError, TypeError):
                continue
    raise RuntimeError(f"Could not load tokenizer from {src} or base={base}")


def _dir_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in Path(p).rglob("*.onnx")) / 1e6


def _predict_all(model, tok, texts, batch=32):
    preds = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, truncation=True, max_length=128, padding=True, return_tensors="np")
        logits = model(**{k: v for k, v in enc.items()}).logits
        preds += [I2L[int(x)] for x in np.asarray(logits).argmax(-1)]
    return preds


def _eval(model, tok, tag, precision):
    test = load_split("test")
    out = {}
    for arm in ("roman", "native"):
        part = filter_arm(test, arm)
        if len(part) < 5:
            continue
        preds = _predict_all(model, tok, part["text"].tolist())
        m = compute_metrics(part["label"].tolist(), preds)
        m["model"], m["arm"], m["precision"] = tag, arm, precision
        print_summary(f"{tag} ONNX-{precision} [{arm}]", m)
        save_result(m, f"results/onnx_{precision}_{tag}_{arm}.json")
        out[arm] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="fine-tuned HF model dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default=None,
                    help="base model id for tokenizer fallback, e.g. ai4bharat/indic-bert")
    args = ap.parse_args()

    src = resolve(args.src)
    onnx_dir = resolve("app/backend/model/onnx")
    int8_dir = resolve("app/backend/model/onnx-int8")
    onnx_dir.mkdir(parents=True, exist_ok=True)
    int8_dir.mkdir(parents=True, exist_ok=True)

    tok = _load_tokenizer(src, args.base)

    # 1) Export to ONNX (fp32).
    print("Exporting to ONNX (fp32)...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(src, export=True)
    ort_model.save_pretrained(onnx_dir)
    tok.save_pretrained(onnx_dir)

    # 2) Dynamic int8 quantization (CPU, avx2 — matches Render hardware).
    print("Quantizing to int8 (dynamic, avx2)...")
    quantizer = ORTQuantizer.from_pretrained(onnx_dir)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
    quantizer.quantize(save_dir=int8_dir, quantization_config=qconfig)
    tok.save_pretrained(int8_dir)

    print(f"\nfp32 ONNX: {_dir_mb(onnx_dir):.1f} MB   int8 ONNX: {_dir_mb(int8_dir):.1f} MB")

    # 3) Evaluate both and report the quantization delta.
    fp32 = _eval(ORTModelForSequenceClassification.from_pretrained(onnx_dir), tok, args.tag, "fp32")
    int8 = _eval(ORTModelForSequenceClassification.from_pretrained(int8_dir), tok, args.tag, "int8")

    print("\n=== FP32 -> int8 quantization delta (roman arm) ===")
    if "roman" in fp32 and "roman" in int8:
        a, b = fp32["roman"], int8["roman"]
        print(f"  macro-F1:        {a['macro_f1']:.3f} -> {b['macro_f1']:.3f} "
              f"({b['macro_f1'] - a['macro_f1']:+.3f})")
        print(f"  smishing recall: {a['smishing_recall']:.3f} -> {b['smishing_recall']:.3f} "
              f"({b['smishing_recall'] - a['smishing_recall']:+.3f})")
    print(f"\nDeployable int8 model -> {int8_dir}")


if __name__ == "__main__":
    main()
