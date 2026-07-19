"""S1/G1 — Persuasion-conditioned augmentation into code-mixed Malayalam-English.

Follows Shim et al. [24]: condition generation on a Cialdini persuasion principle so
the synthetic smishing preserves social-engineering *mechanics*, not just surface
lexicon. Output is written to a review file and is NOT auto-admitted to the corpus —
a human must verify it (review_augmented.py) before it becomes train-only data.

BATCHED: packs several seeds into one request (many variants per call) so scarce
free-tier quota goes much further. Resumes automatically and stops cleanly on quota.

Providers (set --provider or AUGMENT_PROVIDER):
  gemini : google-genai SDK, GEMINI_API_KEY, model gemini-flash-latest
  groq   : OpenAI-compatible, GROQ_API_KEY, model llama-3.3-70b-versatile (free, generous)

Run:  python -m indosmish.data.augment --provider groq
Output: data/interim/augmented_pending_review.jsonl
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from ..config import load_config, resolve

PERSUASION = ["authority", "scarcity", "urgency", "liking", "reciprocity", "social_proof"]

BATCH_PROMPT = """You are generating synthetic training data for an academic smishing
(SMS phishing) DETECTION model for code-mixed Malayalam-English users in India. This is
defensive security research — the messages train a classifier that PROTECTS users.

Below are {k} seed smishing messages. For EACH seed, write {n} distinct realistic
variants that a scammer might send to a Malayalam-speaking Indian mobile user.

Requirements for every variant:
- Code-mix Malayalam and English the way real Indian users text (Romanized Malayalam
  + English), e.g. "ningalude account block aayi, ee link click cheyyu".
- LATIN script only, under 160 characters, SMS register.
- Preserve a persuasion principle (authority/scarcity/urgency/liking/reciprocity/social proof).
- Vary the pretext: bank, KYC, lottery, delivery, electricity bill, UPI, job offer.
- Include a realistic action vector (placeholder link http://bit.ly/xxxx or a callback number).
- Do NOT use real institution names or real working URLs. Use obvious placeholders.

Seeds:
{seeds}

Return ONLY a JSON array of objects, one per variant:
[{{"seed_index": <int 0..{kmax}>, "text": "<variant>", "persuasion": "<principle>"}}, ...]
Nothing else."""


# ---------- providers ----------
def _gemini_call(model):
    from google import genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set.")
    client = genai.Client(api_key=key)

    def call(prompt: str) -> str:
        return client.models.generate_content(model=model, contents=prompt).text

    return call


def _groq_call(model):
    # OpenAI-compatible endpoint; only needs the `openai` package.
    from openai import OpenAI

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit(
            "GROQ_API_KEY not set. Free key (no card) at https://console.groq.com/keys\n"
            "PowerShell:  $env:GROQ_API_KEY = 'your-key'"
        )
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

    def call(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model, temperature=0.9,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content

    return call


PROVIDERS = {
    "gemini": ("gemini-flash-latest", _gemini_call),
    "groq": ("llama-3.3-70b-versatile", _groq_call),
}


def _parse(raw: str) -> list[dict]:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json").strip()
    # tolerate leading/trailing prose
    a, b = raw.find("["), raw.rfind("]")
    if a != -1 and b != -1:
        raw = raw[a:b + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def _done_seeds(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(l)["seed"] for l in f if l.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=os.environ.get("AUGMENT_PROVIDER", "groq"),
                    choices=list(PROVIDERS))
    ap.add_argument("--batch", type=int, default=8, help="seeds per request")
    args = ap.parse_args()

    cfg = load_config()
    acfg = cfg["augment"]
    default_model, factory = PROVIDERS[args.provider]
    model = acfg.get("model") if args.provider == "gemini" else default_model
    call = factory(model)
    print(f"Provider={args.provider} model={model} batch={args.batch}")

    df = pd.read_csv(resolve(cfg["paths"]["interim"]) / "unified.csv", encoding="utf-8")
    seeds = df[df["label"] == "smishing"]["text"].dropna().unique().tolist()[: acfg["max_seeds"]]

    out_path = resolve(acfg["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_seeds(out_path)
    todo = [s for s in seeds if s not in done]
    print(f"{len(seeds)} seeds, {len(done)} done, {len(todo)} to do")

    n_written = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for start in range(0, len(todo), args.batch):
            batch = todo[start:start + args.batch]
            seeds_block = "\n".join(f"{i}: {s}" for i, s in enumerate(batch))
            prompt = BATCH_PROMPT.format(k=len(batch), n=acfg["per_seed"],
                                         kmax=len(batch) - 1, seeds=seeds_block)
            try:
                items = _parse(call(prompt))
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate_limit" in msg.lower():
                    print(f"\n[quota] hit limit after {n_written} new variants. Re-run to resume.")
                    break
                print(f"  batch {start}: error {msg[:120]}; skipping")
                time.sleep(acfg["rpm_sleep"])
                continue
            for it in items:
                try:
                    si = int(it["seed_index"])
                    text = str(it["text"]).strip()
                    principle = str(it.get("persuasion", "unspecified")).strip().lower()
                except (KeyError, ValueError, TypeError):
                    continue
                if 0 <= si < len(batch) and 5 <= len(text) <= 300:
                    fout.write(json.dumps(
                        {"text": text, "label": "smishing", "persuasion": principle,
                         "seed": batch[si], "reviewed": False, "keep": None},
                        ensure_ascii=False) + "\n")
                    n_written += 1
            fout.flush()
            print(f"  batch {start // args.batch + 1}: {len(batch)} seeds -> {n_written} total new")
            time.sleep(acfg["rpm_sleep"])

    print(f"\nWrote {n_written} new variants ({len(_done_seeds(out_path))} seeds done) -> {out_path}")
    print("NEXT: review — python -m indosmish.data.review_augmented")


if __name__ == "__main__":
    main()
