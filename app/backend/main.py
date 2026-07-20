"""IndoSmish inference API (Render backend).

Serves the fine-tuned IndicBERT encoder as ONNX via onnxruntime (no torch — fits the
free 512MB tier). The model is loaded from a local dir if present, else downloaded once
from a HuggingFace model repo at startup (keeps the git repo small).

Env:
  MODEL_DIR   local ONNX dir (default: model/onnx)
  MODEL_REPO  HF repo id to download from if MODEL_DIR is empty (e.g. codewithcid/indosmish-indicbert-onnx)
  ALLOW_ORIGIN  CORS origin for the Vercel frontend (default: * )

Run locally:  uvicorn main:app --reload --port 8000
"""
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

LABELS = ["ham", "spam", "smishing"]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model/onnx"))
MODEL_REPO = os.environ.get("MODEL_REPO", "")
ALLOW_ORIGIN = os.environ.get("ALLOW_ORIGIN", "*")

# Persuasion / attack cues for the explanation shown in the UI (transparent, not the model).
CUES = {
    "urgency": ["urgent", "immediately", "expire", "block", "suspend", "24 hour", "24hr",
                "last chance", "udane", "innu", "rathri", "cut cheyyum"],
    "authority": ["bank", "kyc", "account", "govt", "police", "rbi", "office", "customer care",
                  "verify", "pan"],
    "reward/scarcity": ["win", "won", "prize", "lottery", "free", "reward", "offer", "gift",
                        "cashback", "labhich", "kittum", "award"],
    "action vector": ["click", "link", "http", "bit.ly", "call", "cheyyu", "vilikk", "otp",
                      "upi", "claim"],
}


def _ensure_model() -> Path:
    if (MODEL_DIR / "model.onnx").exists():
        return MODEL_DIR
    if not MODEL_REPO:
        raise RuntimeError(
            f"No model at {MODEL_DIR} and MODEL_REPO not set. "
            "Set MODEL_REPO to a HF repo id, or bundle the ONNX dir."
        )
    from huggingface_hub import snapshot_download
    print(f"Downloading model from {MODEL_REPO} ...")
    local = snapshot_download(repo_id=MODEL_REPO)
    return Path(local)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Message(BaseModel):
    text: str


app = FastAPI(title="IndoSmish API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=[ALLOW_ORIGIN] if ALLOW_ORIGIN != "*" else ["*"],
    allow_methods=["*"], allow_headers=["*"],
)

_tok = None
_sess = None


@app.on_event("startup")
def _load():
    global _tok, _sess
    import onnxruntime as ort
    from transformers import AutoTokenizer

    mdir = _ensure_model()
    _tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True)
    _sess = ort.InferenceSession(str(mdir / "model.onnx"), providers=["CPUExecutionProvider"])
    print(f"Model ready from {mdir}")


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
    enc = _tok(text, truncation=True, max_length=128, return_tensors="np")
    inputs = {k: v for k, v in enc.items() if k in {i.name for i in _sess.get_inputs()}}
    logits = _sess.run(None, inputs)[0]
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
