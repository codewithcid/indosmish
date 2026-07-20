"""IndoSmish inference API (memory-lean, for 512MB free tiers).

Serves the fine-tuned IndicBERT encoder as ONNX. Deliberately avoids importing
`transformers` (which alone costs ~150MB RSS) — the tokenizer is loaded directly from
tokenizer.json via the lightweight `tokenizers` library, and onnxruntime is configured
with the CPU memory arena disabled. Total footprint fits comfortably under 512MB.

Env:
  MODEL_DIR     local ONNX dir (default: model/onnx)
  MODEL_REPO    HF repo id to download from if MODEL_DIR is empty (optional)
  ALLOW_ORIGIN  CORS origin for the frontend (default: * )

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
"""
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

LABELS = ["ham", "spam", "smishing"]
MAX_LEN = 128
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model/onnx"))
MODEL_REPO = os.environ.get("MODEL_REPO", "")
ALLOW_ORIGIN = os.environ.get("ALLOW_ORIGIN", "*")

CUES = {
    "urgency": ["urgent", "immediately", "expire", "block", "suspend", "24 hour", "24hr",
                "last chance", "udane", "innu", "rathri", "cut cheyyum"],
    "authority": ["bank", "kyc", "account", "govt", "police", "rbi", "office",
                  "customer care", "verify", "pan"],
    "reward/scarcity": ["win", "won", "prize", "lottery", "free", "reward", "offer",
                        "gift", "cashback", "labhich", "kittum", "award"],
    "action vector": ["click", "link", "http", "bit.ly", "call", "cheyyu", "vilikk",
                      "otp", "upi", "claim"],
}


def _ensure_model() -> Path:
    if (MODEL_DIR / "model.onnx").exists():
        return MODEL_DIR
    if not MODEL_REPO:
        raise RuntimeError(f"No model at {MODEL_DIR} and MODEL_REPO not set.")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=MODEL_REPO))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Message(BaseModel):
    text: str


app = FastAPI(title="IndoSmish API", version="1.1")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"] if ALLOW_ORIGIN == "*" else [ALLOW_ORIGIN],
    allow_methods=["*"], allow_headers=["*"],
)

_tok = None
_sess = None
_input_names: set[str] = set()


@app.on_event("startup")
def _load():
    global _tok, _sess, _input_names
    import onnxruntime as ort
    from tokenizers import Tokenizer

    mdir = _ensure_model()
    _tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
    _tok.enable_truncation(max_length=MAX_LEN)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False      # big RSS saving on 512MB tiers
    so.enable_mem_pattern = False
    _sess = ort.InferenceSession(str(mdir / "model.onnx"), sess_options=so,
                                 providers=["CPUExecutionProvider"])
    _input_names = {i.name for i in _sess.get_inputs()}
    print(f"Model ready from {mdir} (inputs: {_input_names})")


def _find_cues(text: str) -> list[str]:
    t = text.lower()
    return [cue for cue, kws in CUES.items() if any(k in t for k in kws)]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _sess is not None}


@app.post("/classify")
def classify(msg: Message):
    text = msg.text.strip()
    if not text:
        return {"error": "empty message"}
    t0 = time.perf_counter()
    enc = _tok.encode(text)
    feed = {
        "input_ids": np.array([enc.ids], dtype=np.int64),
        "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
    }
    feed = {k: v for k, v in feed.items() if k in _input_names}
    logits = _sess.run(None, feed)[0]
    probs = _softmax(np.asarray(logits))[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    top = int(probs.argmax())
    return {
        "label": LABELS[top],
        "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
        "cues": _find_cues(text),
        "latency_ms": round(latency_ms, 1),
        "precision": "fp32-onnx",
    }
