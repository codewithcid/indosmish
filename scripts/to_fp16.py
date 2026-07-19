"""Convert the fp32 ONNX encoder to fp16 (halves size for GitHub <100MB) and verify
accuracy retention on the roman test set. fp16 is far gentler than int8.

Run:  python scripts/to_fp16.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnxconverter_common import float16

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimum.onnxruntime import ORTModelForSequenceClassification  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from indosmish.data.schema import LABELS  # noqa: E402
from indosmish.eval.metrics import compute_metrics, save_result  # noqa: E402
from indosmish.models.data_utils import filter_arm, load_split  # noqa: E402

I2L = {i: l for i, l in enumerate(LABELS)}
src = ROOT / "app/backend/model/onnx"
dst = ROOT / "app/backend/model/onnx-fp16"
dst.mkdir(parents=True, exist_ok=True)

m16 = float16.convert_float_to_float16(onnx.load(str(src / "model.onnx")), keep_io_types=True)
onnx.save(m16, str(dst / "model.onnx"))
for f in src.glob("*"):
    if f.suffix != ".onnx" and f.is_file():
        shutil.copy(f, dst / f.name)
print(f"fp16 ONNX size: {(dst / 'model.onnx').stat().st_size / 1e6:.0f} MB")

tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
model = ORTModelForSequenceClassification.from_pretrained(dst, local_files_only=True)
test = filter_arm(load_split("test"), "roman")
preds = []
for i in range(0, len(test), 32):
    enc = tok(test["text"].tolist()[i:i + 32], truncation=True, max_length=128,
              padding=True, return_tensors="np")
    preds += [I2L[int(x)] for x in np.asarray(model(**dict(enc)).logits).argmax(-1)]
r = compute_metrics(test["label"].tolist(), preds)
r["model"], r["arm"], r["precision"] = "indicbert", "roman", "fp16"
save_result(r, ROOT / "results/onnx_fp16_indicbert_roman.json")
print(f"fp16: macro-F1 {r['macro_f1']:.3f}  smishing-recall {r['smishing_recall']:.3f}")
