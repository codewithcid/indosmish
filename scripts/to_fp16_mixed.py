"""Convert the fp32 ONNX encoder to fp16 (~half size) for a faster in-browser download.

Uses float16.convert_float_to_float16 with an op_block_list that keeps the attention-mask
and layer-norm ops in fp32 (which is where a naive full cast produced an invalid graph on
ALBERT), while still converting the weight-heavy MatMul/Gather tensors to fp16 — so the
file halves. Validated by loading in onnxruntime and checking accuracy on the test set.

Output: app/backend/model/onnx/model_fp16.onnx  (transformers.js dtype:"fp16")
Run:  python scripts/to_fp16_mixed.py
"""
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from indosmish.config import resolve  # noqa: E402
from indosmish.data.schema import LABELS  # noqa: E402
from indosmish.eval.metrics import compute_metrics, print_summary  # noqa: E402
from indosmish.models.data_utils import filter_arm, load_split  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

I2L = {i: l for i, l in enumerate(LABELS)}
mdir = ROOT / "app/backend/model/onnx"
tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
tok.enable_truncation(max_length=128)


def feed(text: str) -> dict:
    e = tok.encode(str(text))
    return {"input_ids": np.array([e.ids], np.int64),
            "attention_mask": np.array([e.attention_mask], np.int64),
            "token_type_ids": np.array([e.type_ids], np.int64)}


# Keep mask-construction + layer-norm math in fp32 for a valid, stable graph; the big
# weight tensors (MatMul, Gather embedding) still convert to fp16 and halve the file.
BLOCK = ["Where", "Expand", "Equal", "Range", "ConstantOfShape", "Cast", "Sub", "Mul",
         "Add", "Div", "Sqrt", "Pow", "ReduceMean", "Softmax"]

print("Converting to fp16 (weight tensors), keeping mask/norm ops fp32...")
m16 = float16.convert_float_to_float16(onnx.load(str(mdir / "model.onnx")),
                                       keep_io_types=True, op_block_list=BLOCK)
out = mdir / "model_fp16.onnx"
onnx.save(m16, str(out))
print(f"fp32 {(mdir/'model.onnx').stat().st_size/1e6:.0f} MB -> fp16 {out.stat().st_size/1e6:.0f} MB")

so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(str(out), sess_options=so, providers=["CPUExecutionProvider"])
names = {i.name for i in sess.get_inputs()}
test = filter_arm(load_split("test"), "roman")
preds = [I2L[int(np.asarray(sess.run(None, {k: v for k, v in feed(t).items() if k in names})[0]).argmax(-1)[0])]
         for t in test["text"]]
print_summary("IndicBERT fp16 [roman]", compute_metrics(test["label"].tolist(), preds))
print(f"\nfp16 model ready -> {out}")
