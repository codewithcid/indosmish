"""S1/G1 — Terminal review tool for augmented smishing (manual verification pass).

The protocol requires human verification before synthetic data enters the corpus.
This shows each pending variant; you press k (keep) / d (drop) / e (edit) / q (quit).
Progress is saved after every decision, so you can stop and resume.

Run:  python -m indosmish.data.review_augmented
Output: updates the same jsonl in place; kept rows feed dedup_split.
"""
import json
from pathlib import Path

from ..config import load_config, resolve


def _load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    cfg = load_config()
    path = resolve(cfg["augment"]["out"])
    if not path.exists():
        raise SystemExit("Run indosmish.data.augment first.")
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
