"""Diagnostic: can int8 retain accuracy with a gentler quant config?

The default avx2 per-channel dynamic quant collapsed smishing recall (0.87->0.37).
Try alternatives and measure smishing recall on the roman test set to see if the drop
is a config artifact (fixable) or fundamental (a real G5 finding).

Run:  python scripts/try_int8_configs.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer  # noqa: E402
from optimum.onnxruntime.configuration import AutoQuantizationConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from indosmish.data.schema import LABELS  # noqa: E402
from indosmish.eval.metrics import compute_metrics  # noqa: E402
from indosmish.models.data_utils import filter_arm, load_split  # noqa: E402

I2L = {i: l for i, l in enumerate(LABELS)}
FP32 = ROOT / "app/backend/model/onnx"
tok = AutoTokenizer.from_pretrained(FP32, local_files_only=True)
test = filter_arm(load_split("test"), "roman")
y = test["label"].tolist()
texts = test["text"].tolist()


def evaluate(model) -> dict:
    preds = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], truncation=True, max_length=128, padding=True, return_tensors="np")
        logits = model(**dict(enc)).logits
        preds += [I2L[int(x)] for x in np.asarray(logits).argmax(-1)]
    return compute_metrics(y, preds)


configs = {
    "avx2_per_channel": AutoQuantizationConfig.avx2(is_static=False, per_channel=True),
    "avx2_no_per_channel": AutoQuantizationConfig.avx2(is_static=False, per_channel=False),
    "avx512_no_per_channel": AutoQuantizationConfig.avx512(is_static=False, per_channel=False),
    "avx2_reduce_range": AutoQuantizationConfig.avx2(is_static=False, per_channel=False, reduce_range=True),
}

print("Baseline fp32:")
m = evaluate(ORTModelForSequenceClassification.from_pretrained(FP32, local_files_only=True))
print(f"  macro-F1 {m['macro_f1']:.3f}  smishing-recall {m['smishing_recall']:.3f}\n")

for name, qcfg in configs.items():
    out = ROOT / f"scratch_int8_{name}"
    try:
        q = ORTQuantizer.from_pretrained(FP32)
        q.quantize(save_dir=out, quantization_config=qcfg)
        tok.save_pretrained(out)
        m = evaluate(ORTModelForSequenceClassification.from_pretrained(out, local_files_only=True))
        size = sum(f.stat().st_size for f in out.glob("*.onnx")) / 1e6
        print(f"{name:24s} macro-F1 {m['macro_f1']:.3f}  smishing-recall {m['smishing_recall']:.3f}  ({size:.0f}MB)")
    except Exception as e:  # noqa: BLE001
        print(f"{name:24s} FAILED: {str(e)[:80]}")
