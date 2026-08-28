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


HINDI_SMISH_PROMPT = """You are generating synthetic training data for an academic smishing
(SMS phishing) DETECTION model for code-mixed Hindi-English users in India. This is
defensive security research — the messages train a classifier that PROTECTS users.

Below are {k} seed smishing messages (in English). For EACH seed, write {n} distinct realistic
variants that a scammer might send to a Hindi-speaking Indian mobile user.

Requirements for every variant:
- Code-mix Hindi and English the way real Indian users text (Romanized Hindi / Hinglish
  + English), e.g. "aapka account block ho gaya hai, is link pe click karein".
- LATIN script (Hinglish), under 160 characters, SMS register.
- Preserve a persuasion principle (authority/scarcity/urgency/liking/reciprocity/social proof).
- Vary the pretext: bank, KYC, lottery, delivery, electricity bill, UPI, job offer, PAN/Aadhaar.
- Include a realistic action vector (placeholder link http://bit.ly/xxxx or a callback number).
- Do NOT use real institution names or real working URLs. Use obvious placeholders.

Seeds:
{seeds}

Return ONLY a JSON array of objects, one per variant:
[{{"seed_index": <int 0..{kmax}>, "text": "<variant>", "persuasion": "<principle>"}}, ...]
Nothing else."""


HINDI_SMISH_DEVA_PROMPT = """You are generating synthetic training data for an academic smishing
(SMS phishing) DETECTION model for Hindi users in India. This is defensive security
research — the messages train a classifier that PROTECTS users.

Below are {k} seed smishing messages (in English). For EACH seed, write {n} distinct realistic
variants that a scammer might send to a Hindi-speaking Indian mobile user.

Requirements for every variant:
- Write PRIMARILY in DEVANAGARI script (Hindi), the way real Hindi SMS scams are written,
  e.g. "आपका खाता ब्लॉक हो गया है, तुरंत KYC अपडेट करें". A few English/Latin tokens
  (links, OTP, KYC, bank names) mixed in are natural and fine.
- Under 160 characters, SMS register.
- Preserve a persuasion principle (authority/scarcity/urgency/liking/reciprocity/social proof).
- Vary the pretext: bank, KYC, lottery, delivery, electricity bill, UPI, job offer, PAN/Aadhaar.
- Include a realistic action vector (placeholder link http://bit.ly/xxxx or a callback number).
- Do NOT use real institution names or real working URLs. Use obvious placeholders.

Seeds:
{seeds}

Return ONLY a JSON array of objects, one per variant:
[{{"seed_index": <int 0..{kmax}>, "text": "<variant>", "persuasion": "<principle>"}}, ...]
Nothing else."""


HINDI_HAM_PROMPT = """You are generating synthetic training data for an academic smishing
DETECTION model for code-mixed Hindi-English users in India. This is defensive security
research. We need realistic LEGITIMATE (ham) messages that look superficially like the
alerts scammers imitate, so the classifier does not over-flag genuine texts (hard negatives).

Write {n} distinct realistic LEGITIMATE messages a Hindi-speaking Indian user would
genuinely receive. Requirements:
- Code-mix Hindi and English the way real users text (Hinglish + English), LATIN script.
- Under 160 characters, SMS register.
- Genuine, non-fraudulent: OTP codes, bank balance/credit alerts, delivery updates,
  recharge/bill confirmations, appointment reminders, family/friend chatter.
- NO links to click, NO callback-and-share-details, NO urgency-to-act-or-lose. These are
  the SAFE counterparts of scam pretexts.
- Do NOT use real working URLs.

Return ONLY a JSON array of objects, one per message:
[{{"text": "<message>", "persuasion": "none"}}, ...]
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
    import re as _re

    from openai import OpenAI, RateLimitError

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit(
            "GROQ_API_KEY not set. Free key (no card) at https://console.groq.com/keys\n"
            "PowerShell:  $env:GROQ_API_KEY = 'your-key'"
        )
    # max_retries=0: we do our own rate-limit backoff so a 429 never hangs the client.
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1",
                    timeout=60, max_retries=0)
    # gpt-oss models reason by default, which inflates token usage against a tight
    # free-tier TPM budget; ask for minimal reasoning to keep calls cheap.
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}

    def call(prompt: str) -> str:
        for attempt in range(6):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=0.9,
                    messages=[{"role": "user", "content": prompt}], **extra,
                )
                return r.choices[0].message.content
            except RateLimitError as e:
                m = _re.search(r"try again in ([\d.]+)s", str(e))
                wait = float(m.group(1)) + 1.5 if m else 12.0
                print(f"    [rate-limit] sleeping {wait:.0f}s (attempt {attempt+1})")
                time.sleep(wait)
        raise RuntimeError("rate limit: exhausted retries")

    return call


PROVIDERS = {
    "gemini": ("gemini-flash-latest", _gemini_call),
    # llama-3.3-70b was retired on Groq; gpt-oss-120b is the strongest current chat
    # model on the free tier and handles Hinglish/Malayalam code-mix well. Override
    # with GROQ_MODEL if the catalog changes again.
    "groq": (os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"), _groq_call),
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


def _write_batches(fout, seeds, prompt_tpl, call, acfg, batch_size, label):
    """Batched smishing generation from seeds; returns (count, stopped_on_quota).
    Resume is handled by the caller (already-done seeds are filtered out upstream)."""
    n_written = 0
    for start in range(0, len(seeds), batch_size):
        batch = seeds[start:start + batch_size]
        seeds_block = "\n".join(f"{i}: {s}" for i, s in enumerate(batch))
        prompt = prompt_tpl.format(k=len(batch), n=acfg["per_seed"],
                                   kmax=len(batch) - 1, seeds=seeds_block)
        try:
            items = _parse(call(prompt))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate_limit" in msg.lower():
                print(f"\n[quota] hit limit after {n_written} new variants. Re-run to resume.")
                return n_written, True
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
                    {"text": text, "label": label, "persuasion": principle,
                     "seed": batch[si], "reviewed": False, "keep": None},
                    ensure_ascii=False) + "\n")
                n_written += 1
        fout.flush()
        print(f"  batch {start // batch_size + 1}: {len(batch)} seeds -> {n_written} total new")
        time.sleep(acfg["rpm_sleep"])
    return n_written, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=os.environ.get("AUGMENT_PROVIDER", "groq"),
                    choices=list(PROVIDERS))
    ap.add_argument("--lang", default="ml", choices=["ml", "hi"],
                    help="target language of generated code-mix")
    ap.add_argument("--script", default="roman", choices=["roman", "deva"],
                    help="hi only: roman=Hinglish (Latin), deva=Devanagari script")
    ap.add_argument("--batch", type=int, default=8, help="seeds per request")
    args = ap.parse_args()

    cfg = load_config()
    acfg = cfg["augment"]
    default_model, factory = PROVIDERS[args.provider]
    model = acfg.get("model") if args.provider == "gemini" else default_model
    call = factory(model)
    print(f"Provider={args.provider} model={model} lang={args.lang} batch={args.batch}")

    df = pd.read_csv(resolve(cfg["paths"]["interim"]) / "unified.csv", encoding="utf-8")
    # Seed smishing generation from REAL smishing pretexts (English + any native).
    seeds = df[df["label"] == "smishing"]["text"].dropna().unique().tolist()[: acfg["max_seeds"]]

    if args.lang == "ml":
        out_path = resolve(acfg["out"])
        smish_tpl = BATCH_PROMPT
    else:
        out_path = resolve(acfg["out_hi"])
        if args.script == "deva":
            # separate file so the shared English seeds aren't marked "done" by the roman run
            out_path = out_path.with_name(out_path.stem + "_deva" + out_path.suffix)
        smish_tpl = HINDI_SMISH_DEVA_PROMPT if args.script == "deva" else HINDI_SMISH_PROMPT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = _done_seeds(out_path)
    todo = [s for s in seeds if s not in done]
    print(f"{len(seeds)} smishing seeds, {len(done)} done, {len(todo)} to do  (script={args.script})")

    total = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        n, stopped = _write_batches(fout, todo, smish_tpl, call, acfg, args.batch, "smishing")
        total += n

        # Hindi: also generate ham hard-negatives (legit Hinglish look-alikes) so the
        # model does not learn "Hinglish == scam". Gated on ham_ratio_hi and quota.
        if args.lang == "hi" and args.script == "roman" and not stopped:
            ratio = acfg.get("ham_ratio_hi", 0.0)
            n_ham_target = int((n / max(acfg["per_seed"], 1)) * ratio) if n else 0
            n_ham_target = max(n_ham_target, 8) if ratio else 0
            print(f"Generating ~{n_ham_target} Hindi ham hard-negatives...")
            got = 0
            while got < n_ham_target and not stopped:
                prompt = HINDI_HAM_PROMPT.format(n=min(12, n_ham_target - got))
                try:
                    items = _parse(call(prompt))
                except Exception as e:  # noqa: BLE001
                    if any(t in str(e).lower() for t in ("429", "rate_limit", "resource_exhausted")):
                        print(f"[quota] ham generation stopped after {got}."); break
                    time.sleep(acfg["rpm_sleep"]); continue
                for it in items:
                    text = str(it.get("text", "")).strip()
                    if 5 <= len(text) <= 300:
                        fout.write(json.dumps(
                            {"text": text, "label": "ham", "persuasion": "none",
                             "seed": "ham_hard_negative", "reviewed": False, "keep": None},
                            ensure_ascii=False) + "\n")
                        got += 1
                fout.flush()
                print(f"  ham: {got}/{n_ham_target}")
                time.sleep(acfg["rpm_sleep"])
            total += got

    print(f"\nWrote {total} new variants -> {out_path}")
    print("NEXT: review — python -m indosmish.data.review_augmented "
          f"--path {out_path.relative_to(resolve('.')) if out_path.is_absolute() else out_path}")


if __name__ == "__main__":
    main()
