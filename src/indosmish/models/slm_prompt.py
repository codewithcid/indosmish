"""S4/G3 — Qwen2.5-1.5B zero-shot and few-shot tri-class classification.

Runs against a GGUF (llama.cpp, CPU/4 GB-GPU friendly) or a HF checkpoint. Few-shot
uses k fixed exemplars drawn from the TRAIN split (never val/test). Evaluated per arm.

Run (gguf, zero-shot):
  python -m indosmish.models.slm_prompt --gguf models/qwen2.5-1.5b-q4_k_m.gguf --shots 0
Run (few-shot):
  python -m indosmish.models.slm_prompt --gguf models/qwen2.5-1.5b-q4_k_m.gguf --shots 8
Output: results/slm_{mode}_{arm}.json
"""
import argparse
import re

from ..config import load_config
from ..data.schema import LABELS
from ..eval.metrics import compute_metrics, print_summary, save_result
from .data_utils import filter_arm, load_split

SYSTEM = (
    "You are an SMS security classifier for code-mixed Malayalam-English messages. "
    "Classify each message as exactly one of: ham, spam, smishing. "
    "ham = normal/personal/transactional; spam = bulk promotional; "
    "smishing = fraud that tries to make the user click a link, call a number, or "
    "share credentials/OTP/money. Answer with ONE word only."
)


def build_fewshot(k: int) -> str:
    if k <= 0:
        return ""
    train = load_split("train")
    parts = []
    per = max(1, k // len(LABELS))
    for lbl in LABELS:
        rows = train[train["label"] == lbl]["text"].head(per).tolist()
        for t in rows:
            parts.append(f"Message: {t}\nLabel: {lbl}")
    return "\n\n".join(parts) + "\n\n"


def parse_label(raw: str) -> str:
    raw = raw.strip().lower()
    for lbl in LABELS:
        if re.search(rf"\b{lbl}\b", raw):
            return lbl
    return "ham"  # conservative default


def make_gguf_predictor(path: str, fewshot: str):
    from llama_cpp import Llama

    llm = Llama(model_path=path, n_ctx=2048, verbose=False)

    def predict(text: str) -> str:
        prompt = (f"{SYSTEM}\n\n{fewshot}Message: {text}\nLabel:")
        out = llm(prompt, max_tokens=5, temperature=0.0, stop=["\n"])
        return parse_label(out["choices"][0]["text"])

    return predict


def make_hf_predictor(model_id: str, fewshot: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    def predict(text: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{fewshot}Message: {text}\nLabel:"}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=5, do_sample=False)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_label(gen)

    return predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=str)
    ap.add_argument("--hf", type=str)
    ap.add_argument("--shots", type=int, default=0)
    args = ap.parse_args()

    fewshot = build_fewshot(args.shots)
    if args.gguf:
        predict = make_gguf_predictor(args.gguf, fewshot)
        prec = "q4_k_m"
    elif args.hf:
        predict = make_hf_predictor(args.hf, fewshot)
        prec = "fp16"
    else:
        raise SystemExit("Pass --gguf or --hf")

    mode = f"{args.shots}shot_{prec}"
    test = load_split("test")
    for arm in ("roman", "native"):
        part = filter_arm(test, arm)
        if len(part) < 5:
            print(f"[skip] arm={arm}: {len(part)} rows")
            continue
        preds = [predict(t) for t in part["text"]]
        m = compute_metrics(part["label"].tolist(), preds)
        m["model"], m["arm"], m["shots"] = "qwen2.5-1.5b", arm, args.shots
        print_summary(f"Qwen2.5-1.5B {mode} [{arm}]", m)
        save_result(m, f"results/slm_{mode}_{arm}.json")


if __name__ == "__main__":
    main()
