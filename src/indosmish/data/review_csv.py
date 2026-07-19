"""S1/G1 — Fast spreadsheet-based review of augmented smishing.

Reviewing 750 messages one-by-one is impractical. This exports them to a CSV with an
auto-filter pre-pass (dedup + validity + real-brand flagging), so you scan in Excel and
only flip the borderline rows, then import the decisions back.

export:  python -m indosmish.data.review_csv export
         -> data/interim/review.csv  (edit `keep` column: 1=keep, 0=drop)
import:  python -m indosmish.data.review_csv import
         -> writes keep flags back into augmented_pending_review.jsonl

Auto-filter pre-fills `keep`:
  0  if near-duplicate, too short/long, no action vector, or no code-mix signal
  1  otherwise
`flag` column warns about real institution names you may want to soften/drop.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ..config import load_config, resolve
from .schema import dedup_key

# Real brands/institutions we told the model NOT to use — flag if they slipped through.
REAL_BRANDS = re.compile(
    r"\b(sbi|hdfc|icici|axis|paytm|phonepe|gpay|google pay|airtel|jio|vodafone|"
    r"amazon|flipkart|irctc|aadhaar|aadhar|pan card|income tax|rbi)\b", re.I)

# Action vector: a link, a phone number, or an imperative verb smishing relies on.
ACTION = re.compile(r"(https?://|bit\.ly|\b\d{5,}\b|click|cheyy|vilikk|call|claim|verify|update|link)", re.I)

# Code-mix signal: common Romanized Malayalam tokens (broadened to reduce false drops).
CODEMIX = re.compile(
    r"\b(ningal|ningalude|ningalkk|cheyy|aay|aayi|aaya|und|aanu|aan|illa|udane|udan|"
    r"ee|ente|ende|naale|innu|inn|vilikk|labhich|nedaam|adach|ulla|kittum|kitti|"
    r"cheyth|aakk|akk|thanne|ipol|ippol|mathram|venam|cheyyu|cheyyuka|nnu)\w*", re.I)


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _save_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def do_export(cfg: dict) -> None:
    jsonl = resolve(cfg["augment"]["out"])
    rows = _load_jsonl(jsonl)

    seen: set[str] = set()
    out = []
    n_dup = n_novector = n_nocodemix = 0
    for i, r in enumerate(rows):
        text = str(r["text"]).strip()
        sig = dedup_key(text)
        keep, reasons = 1, []
        if sig in seen:
            keep, _ = 0, reasons.append("dup")
            n_dup += 1
        seen.add(sig)
        if not (5 <= len(text) <= 200):
            keep = 0; reasons.append("len")
        if not ACTION.search(text):
            keep = 0; reasons.append("no-action"); n_novector += 1
        if not CODEMIX.search(text):
            keep = 0; reasons.append("no-codemix"); n_nocodemix += 1
        flag = "REAL-BRAND" if REAL_BRANDS.search(text) else ""
        out.append({
            "row": i, "keep": keep, "auto_reason": ";".join(reasons),
            "flag": flag, "persuasion": r.get("persuasion", ""), "text": text,
        })

    df = pd.DataFrame(out)
    csv_path = resolve(cfg["paths"]["interim"]) / "review.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")  # BOM so Excel shows Malayalam
    print(f"Exported {len(df)} rows -> {csv_path}")
    print(f"  auto-kept: {int(df['keep'].sum())}   auto-dropped: {int((df['keep']==0).sum())}")
    print(f"  (dup={n_dup}, no-action={n_novector}, no-codemix={n_nocodemix}, "
          f"real-brand flagged={int((df['flag']!='').sum())})")
    print("\nNEXT: open review.csv in Excel, adjust the `keep` column (1=keep, 0=drop),")
    print("      save, then: python -m indosmish.data.review_csv import")


def do_import(cfg: dict) -> None:
    csv_path = resolve(cfg["paths"]["interim"]) / "review.csv"
    if not csv_path.exists():
        raise SystemExit("review.csv not found — run export first.")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    jsonl = resolve(cfg["augment"]["out"])
    rows = _load_jsonl(jsonl)

    keep_by_row = dict(zip(df["row"], df["keep"]))
    edited_text = dict(zip(df["row"], df["text"]))
    n_keep = 0
    for i, r in enumerate(rows):
        k = int(keep_by_row.get(i, 0))
        r["reviewed"] = True
        r["keep"] = bool(k)
        if i in edited_text and isinstance(edited_text[i], str):
            r["text"] = str(edited_text[i]).strip()  # honor in-place edits
        n_keep += k
    _save_jsonl(jsonl, rows)
    print(f"Imported decisions: {n_keep} kept / {len(rows)} total -> {jsonl}")
    print("NEXT: python -m indosmish.data.dedup_split")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "import"])
    args = ap.parse_args()
    cfg = load_config()
    (do_export if args.mode == "export" else do_import)(cfg)


if __name__ == "__main__":
    main()
