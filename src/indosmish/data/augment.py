"""S1/G1 — Persuasion-conditioned augmentation into code-mixed Malayalam-English.

Follows Shim et al. [24]: condition generation on a Cialdini persuasion principle so
the synthetic smishing preserves social-engineering *mechanics*, not just surface
lexicon. Output is written to a review file and is NOT auto-admitted to the corpus —
a human must verify it (see review_augmented.py) before it becomes train-only data.

Requires: GEMINI_API_KEY env var (free tier). Uses gemini-2.0-flash.

Run:  python -m indosmish.data.augment
Output: data/interim/augmented_pending_review.jsonl
"""
import json
import os
import time
from pathlib import Path

import pandas as pd

from ..config import load_config, resolve

PERSUASION = ["authority", "scarcity", "urgency", "liking", "reciprocity", "social_proof"]

PROMPT = """You are generating synthetic training data for an academic smishing
(SMS phishing) DETECTION model for code-mixed Malayalam-English users in India.
This is defensive security research — the messages train a classifier that PROTECTS users.

Rewrite the seed smishing SMS below as {n} DISTINCT realistic variants that a scammer
might send to a Malayalam-speaking Indian mobile user.

Requirements:
- Code-mix Malayalam and English the way real Indian users text (Romanized Malayalam
  words mixed with English), e.g. "ningalude account block aayi, ee link click cheyyu".
- Keep it in LATIN script (Romanized), under 160 characters, SMS register.
- Preserve the persuasion principle: {principle}.
- Vary the pretext (bank, KYC, lottery, delivery, electricity bill, UPI, job offer).
- Include a realistic action vector (fake link placeholder http://bit.ly/xxxx, or a
  callback number) as real smishing does.
- Do NOT include real institution names or real working URLs. Use obvious placeholders.

Seed: "{seed}"

Return ONLY a JSON array of {n} strings, nothing else."""


def _configure():
    import google.generativeai as genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey\n"
            "PowerShell:  $env:GEMINI_API_KEY = 'your-key'"
        )
    genai.configure(api_key=key)
    return genai


def _parse_array(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json").strip()
    try:
        arr = json.loads(raw)
        return [str(x).strip() for x in arr if str(x).strip()]
    except json.JSONDecodeError:
        return []


def main() -> None:
    cfg = load_config()
    acfg = cfg["augment"]
    genai = _configure()
    model = genai.GenerativeModel(acfg["model"])

    unified_path = resolve(cfg["paths"]["interim"]) / "unified.csv"
    if not unified_path.exists():
        raise SystemExit("Run indosmish.data.build_corpus first.")
    df = pd.read_csv(unified_path, encoding="utf-8")
    seeds = df[df["label"] == "smishing"]["text"].dropna().unique().tolist()
    seeds = seeds[: acfg["max_seeds"]]
    print(f"{len(seeds)} smishing seeds -> {acfg['per_seed']} variants each")

    out_path = resolve(acfg["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, seed in enumerate(seeds):
            principle = PERSUASION[i % len(PERSUASION)]
            prompt = PROMPT.format(n=acfg["per_seed"], principle=principle, seed=seed)
            try:
                resp = model.generate_content(prompt)
                variants = _parse_array(resp.text)
            except Exception as e:  # noqa: BLE001 — free tier: rate limits / safety blocks
                print(f"  [{i}] error: {e}; backing off 5s")
                time.sleep(5)
                continue
            for v in variants:
                if 5 <= len(v) <= 300:
                    fout.write(
                        json.dumps(
                            {
                                "text": v,
                                "label": "smishing",
                                "persuasion": principle,
                                "seed": seed,
                                "reviewed": False,
                                "keep": None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_written += 1
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(seeds)} seeds, {n_written} variants")
            time.sleep(1.5)  # stay under free-tier RPM

    print(f"\nWrote {n_written} variants -> {out_path}")
    print("NEXT: review them — python -m indosmish.data.review_augmented")


if __name__ == "__main__":
    main()
