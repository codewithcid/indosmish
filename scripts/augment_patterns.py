"""Augment the smishing class with a broad taxonomy of scam PATTERNS — especially the
link-free / conversational / callback social-engineering scams the link-based seed corpus
under-represents. Generates code-mixed Malayalam-English via Groq, per archetype.

Appends accepted examples straight into the augmented pool (reviewed=True) after a light
quality filter (length + code-mix signal + within-run dedup).

Run:  python scripts/augment_patterns.py
Output: appends to data/interim/augmented_pending_review.jsonl
"""
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from indosmish.config import load_config, resolve  # noqa: E402
from indosmish.data.build_corpus import _hash_id, detect_script  # noqa: E402
from indosmish.data.schema import dedup_key, normalize_text  # noqa: E402

# Scam archetypes — each a (tag, description). Emphasis on LINK-FREE social engineering.
ARCHETYPES = [
    ("impersonation", "A friend/relative claiming they lost their phone or changed number, asking to send money urgently to a NEW number. No link. e.g. 'Da ente phone poyi, ee number aanu puthiya, urgent aayi 5000 ee number-il ayakku'"),
    ("family_emergency", "A fake family emergency (accident, hospital) demanding immediate money transfer to a number/UPI. No link."),
    ("callback_bank", "A message telling the user their account/card/SIM will be blocked and to CALL a phone number to fix it. No link, phone number only."),
    ("reply_bait", "A message asking the user to REPLY with YES/1/OTP/details to confirm, claim, or stop something. No link."),
    ("job_scam", "A work-from-home / part-time job or task scam promising daily earnings, asking to message on WhatsApp or a number. Mostly link-free."),
    ("refund_overpayment", "A fake refund or 'we accidentally sent you money, please return it' scam, asking to transfer to a UPI/number. No link."),
    ("lottery_callback", "A prize/lottery win where the user must CALL a number or message to claim (no link)."),
    ("loan_approval", "A pre-approved instant loan offer asking the user to call or share Aadhaar/PAN/OTP to process. Link-free variant."),
    ("utility_disconnect", "Electricity/gas/DTH disconnection tonight unless the user calls a number to update/pay. No link."),
    ("delivery_callback", "A courier/parcel held due to address/customs, asking the user to CALL a number to release it. No link."),
    ("investment", "A crypto/stock/investment 'double your money' scam asking to message or join on WhatsApp/Telegram. Mostly link-free."),
    ("otp_social", "A social-engineering message asking the user to share the OTP they just received to 'verify/cancel' a transaction. No link."),
    ("romance", "A romance/relationship opener building trust and eventually hinting at money or moving to WhatsApp. Link-free."),
    ("fake_reward_upi", "A cashback/reward that requires the user to accept a UPI 'collect request' or send a small amount first. Link optional."),
]

# LEGITIMATE messages that superficially resemble scams — hard negatives so the model
# learns that numbers/actions/money alone are not fraud; intent is.
HAM_ARCHETYPES = [
    ("legit_otp", "A genuine OTP/verification SMS from a service: gives a code and says do NOT share it. e.g. '1234 is your OTP for login. Do not share it with anyone.'"),
    ("legit_delivery", "A genuine delivery notification: order shipped / out for delivery / arriving, maybe a tracking id or 'call driver'. No fraud."),
    ("legit_bank_txn", "A genuine bank transaction alert: Rs X debited/credited to a/c, available balance, UPI ref. Purely informational."),
    ("legit_bill", "A genuine bill/recharge reminder: your bill of Rs X is due on a date, pay to avoid disruption. From a normal biller, no scam."),
    ("legit_personal_money", "A genuine casual money request between friends/family: 'da send me 200 for lunch, will return', 'transfer the rent to my account', ending with da/bro."),
    ("legit_appointment", "A genuine appointment/booking confirmation: your booking/appointment is confirmed for a date/time, reply to reschedule."),
    ("legit_family", "A normal family/friend message that mentions money or a number casually but is clearly legitimate and warm."),
    ("legit_work", "A normal work/college message: meeting at 5, share the doc, call me when free — mundane, no fraud."),
]

N_PER = 16
PROMPT = """You are generating synthetic training data for an academic smishing (SMS
phishing) DETECTION model that PROTECTS code-mixed Malayalam-English users in India.
This is defensive security research.

Generate {n} DISTINCT, realistic SMS messages of this type:
{desc}

Requirements for EVERY message:
- {register}
- Vary wording, names, amounts, and phone numbers heavily. Use fake placeholder numbers.
- {constraint}
- Do NOT use real working URLs or real institution account numbers.

Return ONLY a JSON array of {n} strings, nothing else."""

CONSTRAINT_SCAM = ("Match the scam type above. Unless the type says otherwise, do NOT include a "
                   "URL/link — these are link-free social-engineering scams relying on calls, "
                   "replies, UPI, or money transfer.")
CONSTRAINT_HAM = ("These are LEGITIMATE, non-fraudulent messages — the kind a real service, "
                  "bank, delivery company, or a genuine friend/family member actually sends. "
                  "They may mention numbers, OTPs, money or deliveries, but they are NOT scams.")

REGISTER_MIX = ("Code-mix Malayalam and English the way real Indian users text (Romanized "
                "Malayalam + English), Latin script only, under 160 characters, SMS register.")
REGISTER_EN = ("Plain casual English SMS as Indian users text, sometimes ending with 'da'/'bro'/"
               "'machan', under 160 characters. Keep it conversational and link-free.")

# Looser signal for English link-free scams (no code-mix token required).
SCAM_SIGNAL = re.compile(
    r"(\b\d{5,}\b|call|reply|send|sent|transfer|pay|paid|money|cash|otp|upi|whatsapp|"
    r"urgent|number|account|click|link|verify|claim|prize|refund|gift|bank|kyc)", re.I)

CODEMIX = re.compile(
    r"\b(ningal|ningalude|ningalkk|cheyy|aay|aayi|aaya|und|aanu|aan|illa|udane|udan|ee|"
    r"ente|ende|naale|innu|inn|vilikk|labhich|nedaam|adach|ulla|kittum|kitti|cheyth|aakk|"
    r"akk|thanne|ipol|ippol|mathram|venam|ayakk|ayakku|poyi|number|paisa|rupa|help)\w*", re.I)


def _client():
    from openai import OpenAI
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("Set GROQ_API_KEY.")
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")


def _parse(raw: str) -> list[str]:
    raw = (raw or "").strip()
    a, b = raw.find("["), raw.rfind("]")
    if a != -1 and b != -1:
        raw = raw[a:b + 1]
    try:
        return [str(x).strip() for x in json.loads(raw) if str(x).strip()]
    except json.JSONDecodeError:
        return []


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--english", action="store_true", help="English (not code-mixed) register")
    ap.add_argument("--ham", action="store_true", help="generate legitimate ham hard-negatives")
    args = ap.parse_args()
    register = REGISTER_EN if args.english else REGISTER_MIX
    signal = SCAM_SIGNAL if args.english else CODEMIX
    archetypes = HAM_ARCHETYPES if args.ham else ARCHETYPES
    label = "ham" if args.ham else "smishing"
    constraint = CONSTRAINT_HAM if args.ham else CONSTRAINT_SCAM

    cfg = load_config()
    client = _client()
    out_path = resolve(cfg["augment"]["out"])

    # existing signatures to avoid dupes
    seen = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seen.add(dedup_key(json.loads(line)["text"]))
    print(f"{len(seen)} existing augmented signatures")

    n_new = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for tag, desc in archetypes:
            try:
                r = client.chat.completions.create(
                    model="llama-3.3-70b-versatile", temperature=1.0,
                    messages=[{"role": "user", "content": PROMPT.format(
                        n=N_PER, desc=desc, register=register, constraint=constraint)}])
                variants = _parse(r.choices[0].message.content)
            except Exception as e:  # noqa: BLE001
                if "rate" in str(e).lower() or "429" in str(e):
                    print(f"[quota] stopped at '{tag}' after {n_new} new. Re-run to continue."); break
                print(f"  {tag}: error {str(e)[:80]}"); time.sleep(4); continue
            kept = 0
            for v in variants:
                v = normalize_text(v)
                sig = dedup_key(v)
                if not (10 <= len(v) <= 200) or sig in seen or not signal.search(v):
                    continue
                seen.add(sig)
                fout.write(json.dumps({
                    "text": v, "label": label, "persuasion": tag, "seed": f"pattern:{tag}",
                    "reviewed": True, "keep": True}, ensure_ascii=False) + "\n")
                kept += 1; n_new += 1
            fout.flush()
            print(f"  {tag:20s} +{kept}")
            time.sleep(2)

    print(f"\nAdded {n_new} link-free/pattern smishing -> {out_path}")
    print("NEXT: dedup_split -> retrain")


if __name__ == "__main__":
    main()
