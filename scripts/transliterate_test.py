"""G2 — Build the native-script test arm by transliterating the Roman test set to
Malayalam script via LLM transliteration (the paper's strategy 4), then this enables
the transliteration-deficit test (evaluate the same messages in Roman vs native script).

Uses Groq (GROQ_API_KEY). Batched + resumable. Keeps English words in Latin, converts
Romanized Malayalam to Malayalam script.

Run:  python scripts/transliterate_test.py
Output: data/processed/splits/test_xlit.csv
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from indosmish.config import load_config, resolve  # noqa: E402

BATCH = 12
PROMPT = """Transliterate each Romanized Malayalam-English SMS below into Malayalam script.
Rules:
- Convert Romanized Malayalam words to Malayalam (native) script.
- Keep English words, URLs, numbers, and symbols exactly as-is (do NOT translate).
- Preserve meaning and order; this is transliteration, not translation.

Return ONLY a JSON array of objects: [{{"i": <index>, "t": "<transliterated>"}}, ...]

Messages:
{block}"""


def _client():
    from openai import OpenAI
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("Set GROQ_API_KEY.")
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")


def _parse(raw: str) -> list[dict]:
    raw = raw.strip()
    a, b = raw.find("["), raw.rfind("]")
    if a != -1 and b != -1:
        raw = raw[a:b + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def main() -> None:
    cfg = load_config()
    client = _client()
    test = pd.read_csv(resolve(cfg["paths"]["splits"]) / "test.csv", encoding="utf-8")
    # Transliterate the roman-script rows (the smishing/spam/English ham live here).
    roman = test[test["script"] == "roman"].reset_index(drop=True)
    print(f"Transliterating {len(roman)} roman test messages -> native script")

    out_path = resolve(cfg["paths"]["splits"]) / "test_xlit.csv"
    done = {}
    if out_path.exists():
        prev = pd.read_csv(out_path, encoding="utf-8")
        done = dict(zip(prev["id"], prev["text"]))
        print(f"  resuming: {len(done)} already done")

    rows = []
    todo_idx = [i for i in range(len(roman)) if roman.iloc[i]["id"] not in done]
    for start in range(0, len(todo_idx), BATCH):
        idxs = todo_idx[start:start + BATCH]
        block = "\n".join(f'{j}: {roman.iloc[i]["text"]}' for j, i in enumerate(idxs))
        try:
            r = client.chat.completions.create(
                model="llama-3.3-70b-versatile", temperature=0,
                messages=[{"role": "user", "content": PROMPT.format(block=block)}])
            items = _parse(r.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            if "rate" in str(e).lower() or "429" in str(e):
                print(f"  rate limited after {len(rows)}; re-run to resume."); break
            print(f"  batch {start} error: {str(e)[:80]}"); time.sleep(4); continue
        by_i = {int(it["i"]): str(it["t"]).strip() for it in items if "i" in it and "t" in it}
        for j, i in enumerate(idxs):
            row = roman.iloc[i]
            done[row["id"]] = by_i.get(j, row["text"])
        # checkpoint each batch
        _write(roman, done, out_path)
        print(f"  {min(start + BATCH, len(todo_idx))}/{len(todo_idx)}")
        time.sleep(2)

    print(f"\nWrote {out_path}. Next: evaluate with scripts/eval_xlit.py")


def _write(roman: pd.DataFrame, done: dict, path: Path) -> None:
    out = []
    for _, row in roman.iterrows():
        if row["id"] in done:
            out.append({"id": row["id"], "text": done[row["id"]], "label": row["label"],
                        "script": "native", "source": row["source"], "synthetic": False})
    pd.DataFrame(out).to_csv(path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
