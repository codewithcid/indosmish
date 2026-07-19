"""Auto-download the free, public datasets (no account, no IEEE).

Pulls DravidianCodeMix Malayalam-English and the UCI SMS Spam Collection via the
HuggingFace `datasets` library and writes them into data/raw/ in a simple text/label
CSV shape. The Mishra & Soni tri-class set is a 1-click Mendeley download (see DATA.md).

Run:  python -m indosmish.data.fetch_free_data
"""
from pathlib import Path

import pandas as pd

from ..config import load_config, resolve

# DravidianCodeMix offensive-language labels considered "normal" text -> ham candidates.
NOT_OFFENSIVE = {"Not_offensive", "not-Malayalam", "not-malayalam"}


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  wrote {len(df)} rows -> {path}")


def fetch_dravidian(raw: Path) -> None:
    from datasets import load_dataset

    print("DravidianCodeMix Malayalam-English (offenseval_dravidian, config=malayalam)...")
    try:
        ds = load_dataset("community-datasets/offenseval_dravidian", "malayalam")
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] could not load: {e}\n  Fallback: GitHub bharathichezhiyan/DravidianCodeMix-Dataset")
        return
    # Concatenate available splits; keep text + raw label.
    frames = []
    for split in ds:
        d = ds[split]
        cols = d.column_names
        text_col = "text" if "text" in cols else cols[0]
        label_col = "label" if "label" in cols else cols[-1]
        df = pd.DataFrame({"text": d[text_col], "label": d[label_col]})
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # HF may encode labels as ints; map via feature names if so.
    if pd.api.types.is_integer_dtype(out["label"]):
        try:
            names = ds[list(ds.keys())[0]].features["label"].names
            out["label"] = out["label"].map(lambda i: names[i])
        except Exception:  # noqa: BLE001
            pass
    _save(out, raw / "dravidian_codemix_ml.csv")


def fetch_uci(raw: Path) -> None:
    from datasets import load_dataset

    print("UCI SMS Spam Collection (sms_spam)...")
    try:
        ds = load_dataset("ucirvine/sms_spam", split="train")
    except Exception:  # noqa: BLE001
        try:
            ds = load_dataset("sms_spam", split="train")
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] could not load: {e}")
            return
    # sms_spam: {sms: str, label: 0 ham / 1 spam}
    text = ds["sms"] if "sms" in ds.column_names else ds[ds.column_names[0]]
    lab = ds["label"]
    lab = ["spam" if int(x) == 1 else "ham" for x in lab]
    _save(pd.DataFrame({"text": text, "label": lab}), raw / "uci_sms.csv")


def main() -> None:
    cfg = load_config()
    raw = resolve(cfg["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    fetch_dravidian(raw)
    fetch_uci(raw)
    print("\nDone. Still needed (1-click, free, no account):")
    print("  Mishra & Soni tri-class -> data/raw/sms_phishing.csv")
    print("  https://data.mendeley.com/datasets/f45bkkt8pr/1")


if __name__ == "__main__":
    main()
