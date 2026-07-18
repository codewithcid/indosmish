"""S6/G4 — Deployment profiling on named hardware.

Reports, at the deployed precision: model file size (MB), peak resident memory (MB),
mean + p95 per-message latency over N timed messages after W warm-ups. Works for a
HF encoder dir or a GGUF file (llama.cpp). Records device name in the output JSON.

Run (encoder):  python -m indosmish.eval.profile_device --encoder results/muril_roman
Run (gguf):     python -m indosmish.eval.profile_device --gguf models/qwen-q4_k_m.gguf
"""
import argparse
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from ..config import load_config, resolve


def _dir_size_mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def _sample_messages(n: int) -> list[str]:
    cfg = load_config()
    test = pd.read_csv(resolve(cfg["paths"]["splits"]) / "test.csv", encoding="utf-8")
    texts = test["text"].tolist()
    if not texts:
        texts = ["Your account is blocked, click http://bit.ly/x to verify"]
    # cycle to reach n
    return [texts[i % len(texts)] for i in range(n)]


def _device_label(kind: str) -> str:
    return f"{platform.processor() or platform.machine()} | {platform.system()} {platform.release()} | {kind}"


def profile(predict_fn, kind: str, size_mb: float, precision: str) -> dict:
    cfg = load_config()["profiling"]
    warm, timed = cfg["warmup_messages"], cfg["timed_messages"]
    msgs = _sample_messages(warm + timed)

    proc = psutil.Process()
    for m in msgs[:warm]:
        predict_fn(m)

    peak_rss = proc.memory_info().rss / 1e6
    lat = []
    for m in msgs[warm:]:
        t0 = time.perf_counter()
        predict_fn(m)
        lat.append((time.perf_counter() - t0) * 1000)  # ms
        peak_rss = max(peak_rss, proc.memory_info().rss / 1e6)

    lat = np.array(lat)
    return {
        "device": _device_label(kind),
        "precision": precision,
        "model_size_mb": round(size_mb, 2),
        "peak_rss_mb": round(peak_rss, 1),
        "latency_ms_mean": round(float(lat.mean()), 2),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
        "n_timed": len(lat),
    }


def _encoder_predict_fn(path: Path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    def fn(text: str):
        enc = tok(text, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**enc)

    return fn, "GPU" if device == "cuda" else "CPU"


def _gguf_predict_fn(path: Path):
    from llama_cpp import Llama

    llm = Llama(model_path=str(path), n_ctx=512, verbose=False)

    def fn(text: str):
        llm(f"Classify SMS as ham/spam/smishing: {text}\nLabel:", max_tokens=3)

    return fn, "CPU"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=str)
    ap.add_argument("--gguf", type=str)
    ap.add_argument("--precision", type=str, default=None)
    ap.add_argument("--out", type=str, default="results/profiling.json")
    args = ap.parse_args()

    if args.encoder:
        path = resolve(args.encoder)
        fn, kind = _encoder_predict_fn(path)
        prec = args.precision or "fp32"
    elif args.gguf:
        path = resolve(args.gguf)
        fn, kind = _gguf_predict_fn(path)
        prec = args.precision or "q4_k_m"
    else:
        raise SystemExit("Pass --encoder <dir> or --gguf <file>")

    result = profile(fn, kind, _dir_size_mb(path), prec)
    print("\nProfiling result:")
    for k, v in result.items():
        print(f"  {k:18s}: {v}")

    import json

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else []
    existing.append(result)
    out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nAppended -> {out}")


if __name__ == "__main__":
    main()
