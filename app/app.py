"""Gradio demo — deploys to Hugging Face Spaces (free CPU tier).

Paste an SMS -> tri-class verdict + confidence + persuasion-cue rationale + live
latency readout. The free tier is CPU-only, which forces the quantized model and so
demonstrates the paper's on-device efficiency thesis by construction.

Config via env:
  MODEL_KIND = "encoder" (default) | "gguf"
  MODEL_PATH = HF repo id / local dir for encoder, or .gguf path
For Spaces, set these in the Space's Variables, and put the model in the repo or
load an encoder straight from the Hub.
"""
import os
import time

import gradio as gr

LABELS = ["ham", "spam", "smishing"]
MODEL_KIND = os.environ.get("MODEL_KIND", "encoder")
MODEL_PATH = os.environ.get("MODEL_PATH", "results/best/muril")

CUES = {
    "urgency": ["urgent", "immediately", "expire", "block", "suspend", "24 hour", "last"],
    "authority": ["bank", "kyc", "govt", "income tax", "police", "rbi", "account"],
    "reward/scarcity": ["win", "prize", "lottery", "free", "reward", "offer", "gift"],
    "action vector": ["click", "link", "http", "bit.ly", "call", "verify", "otp", "upi"],
}


def find_cues(text: str) -> list[str]:
    t = text.lower()
    hits = []
    for cue, kws in CUES.items():
        if any(k in t for k in kws):
            hits.append(cue)
    return hits


_predict = None


def _load_encoder(path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def predict(text):
        enc = tok(text, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            probs = model(**enc).logits.softmax(-1)[0].tolist()
        return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

    return predict


def _load_gguf(path):
    from llama_cpp import Llama

    llm = Llama(model_path=path, n_ctx=1024, verbose=False)
    sys = ("Classify the SMS as ham, spam, or smishing. Answer one word.\n")

    def predict(text):
        out = llm(f"{sys}Message: {text}\nLabel:", max_tokens=5, temperature=0.0)
        raw = out["choices"][0]["text"].strip().lower()
        label = next((l for l in LABELS if l in raw), "ham")
        return {l: (1.0 if l == label else 0.0) for l in LABELS}

    return predict


def get_predictor():
    global _predict
    if _predict is None:
        _predict = _load_gguf(MODEL_PATH) if MODEL_KIND == "gguf" else _load_encoder(MODEL_PATH)
    return _predict


def classify(text):
    if not text or not text.strip():
        return {}, "Enter a message.", ""
    predict = get_predictor()
    t0 = time.perf_counter()
    scores = predict(text)
    latency = (time.perf_counter() - t0) * 1000

    top = max(scores, key=scores.get)
    cues = find_cues(text)
    verdict = {"🔴 SMISHING": "smishing", "🟡 SPAM": "spam", "🟢 HAM": "ham"}
    emoji_label = next(k for k, v in verdict.items() if v == top)

    rationale = f"**Verdict: {emoji_label}**\n\n"
    if top == "smishing":
        rationale += "Detected persuasion / attack cues: " + (
            ", ".join(f"`{c}`" for c in cues) if cues else "boundary case — review manually"
        )
    elif cues:
        rationale += "Surface cues present: " + ", ".join(f"`{c}`" for c in cues)
    else:
        rationale += "No strong fraud cues detected."

    return scores, rationale, f"⚡ {latency:.1f} ms on {MODEL_KIND} ({os.cpu_count()} CPU cores)"


EXAMPLES = [
    "Hey, reaching in 10 mins, order some chai",
    "Congratulations! Ningalude number won 25 lakh. Claim cheyyan http://bit.ly/x3k call cheyyu",
    "FLAT 50% OFF on all items this weekend only! Visit our store now",
    "Your SBI account KYC expired, ee link click cheythu verify cheyyu illenkil block aakum: http://sbi-verify.xyz",
]

with gr.Blocks(title="IndoSmish — Code-Mixed Smishing Detector") as demo:
    gr.Markdown(
        "# 📱 IndoSmish\n"
        "Tri-class **ham / spam / smishing** detection for **code-mixed Malayalam–English** SMS.\n\n"
        "Implements the on-device benchmark from *From SMS Spam Filters to Quantized LLMs* "
        "(IEEE Access 2026). Runs quantized on a free CPU — no message leaves the server "
        "beyond this demo."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="SMS message", lines=3,
                             placeholder="Paste a message (English or Romanized Malayalam-English)...")
            btn = gr.Button("Classify", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp)
        with gr.Column():
            out_scores = gr.Label(label="Class probabilities", num_top_classes=3)
            out_rationale = gr.Markdown()
            out_latency = gr.Markdown()

    btn.click(classify, inputs=inp, outputs=[out_scores, out_rationale, out_latency])
    inp.submit(classify, inputs=inp, outputs=[out_scores, out_rationale, out_latency])

if __name__ == "__main__":
    demo.launch()
