"""S1/G1 — Review tool for augmented data (verification pass before it enters the corpus).

The protocol requires verification before synthetic data enters the corpus. Two modes:
  interactive (default): shows each variant; press k/d/e/s/q.
  --auto: applies deterministic quality filters (dedup, length, action-vector presence
          for smishing) and marks the survivors keep=True, so large batches can be
          verified by rule + spot-check instead of thousands of keystrokes.

Run:  python -m indosmish.data.review_augmented [--path FILE] [--auto]
Output: updates the same jsonl in place; kept rows feed dedup_split.
"""
import argparse
import json
import re
from pathlib import Path

from ..config import load_config, resolve

# An SMS-phishing message almost always carries an action vector: a link or a callback.
_ACTION = re.compile(r"https?://|bit\.ly|tinyurl|\blink\b|click|call\b|\b\d{6,}\b|whatsapp|www\.", re.I)


def _load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def auto_review(path: Path) -> None:
    """Deterministic quality gate: dedup, length bounds, smishing must have an action
    vector. Marks survivors reviewed+keep; drops the rest. Idempotent."""
    rows = _load(path)
    seen: set[str] = set()
    kept = dropped = 0
    for r in rows:
        text = str(r.get("text", "")).strip()
        key = re.sub(r"[^a-z0-9]", "", text.lower())
        ok = 15 <= len(text) <= 220 and key and key not in seen
        if ok and r.get("label") == "smishing" and not _ACTION.search(text):
            ok = False  # a "smishing" example with no link/number/callback is a weak signal
        r["reviewed"] = True
        r["keep"] = bool(ok)
        if ok:
            seen.add(key); kept += 1
        else:
            dropped += 1
    _save(path, rows)
    print(f"[auto] {path.name}: kept {kept}, dropped {dropped} (of {len(rows)}).")
    print("Spot-check a sample before trusting these:")
    sample = [r for r in rows if r.get("keep")][:8]
    for r in sample:
        print(f"   [{r.get('label')}] {r['text']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None, help="review file (default: augment.out)")
    ap.add_argument("--auto", action="store_true", help="deterministic filter instead of prompts")
    args = ap.parse_args()

    cfg = load_config()
    path = resolve(args.path) if args.path else resolve(cfg["augment"]["out"])
    if not path.exists():
        raise SystemExit("Run indosmish.data.augment first.")
    if args.auto:
        auto_review(path)
        return
    rows = _load(path)
    pending = [r for r in rows if not r.get("reviewed")]
    print(f"{len(rows)} total, {len(pending)} pending review.")
    print("Keys: [k]eep  [d]rop  [e]dit  [s]kip  [q]uit&save\n")

    done = 0
    for r in rows:
        if r.get("reviewed"):
            continue
        print(f"  persuasion={r['persuasion']}")
        print(f"  TEXT: {r['text']}")
        choice = input("  > ").strip().lower()
        if choice == "q":
            break
        elif choice == "k":
            r["reviewed"], r["keep"] = True, True
        elif choice == "d":
            r["reviewed"], r["keep"] = True, False
        elif choice == "e":
            new = input("  edit: ").strip()
            if new:
                r["text"] = new
            r["reviewed"], r["keep"] = True, True
        elif choice == "s":
            pass
        done += 1
        _save(path, rows)  # checkpoint after each decision
        print()

    kept = sum(1 for r in rows if r.get("keep"))
    print(f"Reviewed {done} this session. Total kept: {kept}. Saved -> {path}")


if __name__ == "__main__":
    main()
