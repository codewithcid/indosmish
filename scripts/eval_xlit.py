"""G2 — Evaluate the fine-tuned IndicBERT on the transliterated native-script test arm
and report the transliteration deficit vs the Roman arm (same messages, both scripts).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from indosmish.config import resolve  # noqa: E402
from indosmish.data.schema import LABELS  # noqa: E402
from indosmish.eval.metrics import compute_metrics, print_summary, save_result  # noqa: E402

import onnxruntime as ort  # noqa: E402

I2L = {i: l for i, l in enumerate(LABELS)}
mdir = ROOT / "app/backend/model/onnx"
tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
tok.enable_truncation(max_length=128)
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess = ort.InferenceSession(str(mdir / "model.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
names = {i.name for i in sess.get_inputs()}


def predict(texts):
    preds = []
    for t in texts:
        e = tok.encode(str(t) if t is not None and str(t) != "nan" else "")
        feed = {"input_ids": np.array([e.ids], np.int64),
                "attention_mask": np.array([e.attention_mask], np.int64),
                "token_type_ids": np.array([e.type_ids], np.int64)}
        feed = {k: v for k, v in feed.items() if k in names}
        preds.append(I2L[int(np.asarray(sess.run(None, feed)[0]).argmax(-1)[0])])
    return preds


# Native-script (transliterated) arm — matched to the same messages as the roman arm.
xlit = pd.read_csv(resolve("data/processed/splits") / "test_xlit.csv", encoding="utf-8")
xlit = xlit.dropna(subset=["text"])
xlit = xlit[xlit["text"].astype(str).str.strip() != ""]
m = compute_metrics(xlit["label"].tolist(), predict(xlit["text"].tolist()))
m["model"], m["arm"] = "indicbert", "xlit-native"
print_summary("IndicBERT [xlit-native] (transliterated)", m)
save_result(m, "results/indicbert_xlit-native.json")

# Roman arm on the SAME message subset, for a matched deficit comparison.
test = pd.read_csv(resolve("data/processed/splits") / "test.csv", encoding="utf-8")
roman = test[test["id"].isin(set(xlit["id"]))]
mr = compute_metrics(roman["label"].tolist(), predict(roman["text"].tolist()))
print_summary("IndicBERT [roman] (same messages)", mr)

print("\n=== TRANSLITERATION DEFICIT (G2): roman -> native script, same messages ===")
print(f"  macro-F1:        {mr['macro_f1']:.3f} -> {m['macro_f1']:.3f} ({m['macro_f1']-mr['macro_f1']:+.3f})")
print(f"  smishing recall: {mr['smishing_recall']:.3f} -> {m['smishing_recall']:.3f} ({m['smishing_recall']-mr['smishing_recall']:+.3f})")
