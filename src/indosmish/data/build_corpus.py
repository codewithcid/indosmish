"""S1/G1 — Merge source corpora into the unified schema.

Reads the raw source CSVs named in configs/default.yaml, maps them onto the unified
schema (schema.COLUMNS), tags provenance and script, and writes a single interim file.

Run:  python -m indosmish.data.build_corpus
Output: data/interim/unified.csv
"""
import hashlib
from pathlib import Path

import pandas as pd

from ..config import load_config, resolve
from .schema import COLUMNS, LABELS, detect_lang, detect_script, normalize_text


def _hash_id(text: str, source: str) -> str:
    return hashlib.sha1(f"{source}:{text}".encode("utf-8")).hexdigest()[:16]


def _norm_label(raw: str) -> str | None:
    s = str(raw).strip().lower()
    if s in ("ham", "0", "legitimate", "legit", "normal"):
        return "ham"
    if s in ("spam", "1"):
        return "spam"
    if s in ("smishing", "smish", "phishing", "2"):
        return "smishing"
    return None


# DravidianCodeMix labels that denote normal (non-offensive) text -> ham candidates.
_NOT_OFFENSIVE = {"not_offensive", "not-malayalam"}


def _mk_row(text: str, label: str, source: str) -> dict:
    return {
        "id": _hash_id(text, source),
        "text": text,
        "label": label,
        "script": detect_script(text),
        "lang": detect_lang(text, source),
        "source": source,
        "synthetic": False,
    }


def load_source(name: str, spec: dict) -> pd.DataFrame:
    path = resolve(spec["file"])
    if not path.exists():
        print(f"  [skip] {name}: {path} not found (see data/DATA.md)")
        return pd.DataFrame(columns=COLUMNS)

    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    text_col, label_col = spec["text_col"], spec["label_col"]
    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(
            f"{name}: expected columns '{text_col}'/'{label_col}', "
            f"found {list(df.columns)}. Fix configs/default.yaml."
        )

    kind = spec.get("kind", "tri_class")
    rows = []
    if kind == "codemix_ham":
        # Keep only non-offensive, code-mixed (native-script present) rows as ham.
        for _, r in df.iterrows():
            text = normalize_text(r[text_col])
            raw_label = str(r[label_col]).strip().lower()
            if not text or raw_label not in _NOT_OFFENSIVE:
                continue
            if detect_script(text) != "native":  # want genuine Malayalam-script code-mix
                continue
            rows.append(_mk_row(text, "ham", name))
        cap = spec.get("max_ham")
        if cap and len(rows) > cap:
            rows = rows[:cap]
    else:  # tri_class
        for _, r in df.iterrows():
            text = normalize_text(r[text_col])
            label = _norm_label(r[label_col])
            if not text or label is None:
                continue
            rows.append(_mk_row(text, label, name))

    out = pd.DataFrame(rows, columns=COLUMNS)
    print(f"  [ok]   {name} ({kind}): {len(out)} rows  {out['label'].value_counts().to_dict()}")
    return out


def _balance_ham(df: pd.DataFrame, max_ham: int, seed: int) -> pd.DataFrame:
    """Cap ham, keeping ALL native-script (code-mixed) ham; sample roman ham to fill."""
    ham = df[df["label"] == "ham"]
    rest = df[df["label"] != "ham"]
    if len(ham) <= max_ham:
        return df
    native_ham = ham[ham["script"] == "native"]
    roman_ham = ham[ham["script"] == "roman"]
    keep_roman = max(0, max_ham - len(native_ham))
    roman_sample = roman_ham.sample(n=min(keep_roman, len(roman_ham)), random_state=seed)
    balanced = pd.concat([rest, native_ham, roman_sample], ignore_index=True)
    print(f"  balanced ham: kept {len(native_ham)} native + {len(roman_sample)} roman "
          f"(capped at {max_ham})")
    return balanced


def _cap_ham(df: pd.DataFrame, max_ham: int, seed: int) -> pd.DataFrame:
    """Uniformly sample ham down to max_ham. Used where native ham is abundant (Hindi),
    unlike _balance_ham which preserves scarce native code-mix (Malayalam)."""
    ham = df[df["label"] == "ham"]
    rest = df[df["label"] != "ham"]
    if len(ham) <= max_ham:
        return df
    ham = ham.sample(n=max_ham, random_state=seed)
    print(f"  capped ham: sampled {len(ham)} of the pool (cap {max_ham})")
    return pd.concat([rest, ham], ignore_index=True)


def main() -> None:
    cfg = load_config()
    frames = [load_source(name, spec) for name, spec in cfg["sources"].items()]
    unified = pd.concat(frames, ignore_index=True)
    unified = unified[unified["label"].isin(LABELS)].drop_duplicates(subset=["id"])

    # Language-partitioned balancing. The ml/en partition is balanced with the exact
    # same logic and seed as before Hindi existed, so the frozen ml/en splits remain
    # byte-identical (paper numbers stay comparable). Hindi is capped independently.
    bal = cfg.get("balance", {})
    max_ham = bal.get("max_ham")
    max_ham_hi = bal.get("max_ham_hi")
    non_hi = unified[unified["lang"] != "hi"].copy()
    hi = unified[unified["lang"] == "hi"].copy()
    if max_ham:
        non_hi = _balance_ham(non_hi, max_ham, cfg["seed"])
    if max_ham_hi:
        hi = _cap_ham(hi, max_ham_hi, cfg["seed"])
    unified = pd.concat([non_hi, hi], ignore_index=True)

    out_dir = resolve(cfg["paths"]["interim"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "unified.csv"
    unified.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\nWrote {len(unified)} rows -> {out_path}")
    print("By label:", unified["label"].value_counts().to_dict())
    print("By script:", unified["script"].value_counts().to_dict())
    print("By lang:", unified["lang"].value_counts().to_dict())
    if (unified["label"] == "smishing").sum() < 50:
        print(
            "\n[warn] <50 smishing seeds. Augmentation (S1) is essential — "
            "run indosmish.data.augment next."
        )


if __name__ == "__main__":
    main()
