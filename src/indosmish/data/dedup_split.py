"""S1/S2 — Dedup-before-split, produce frozen stratified splits.

Enforces the protocol:
 - deduplicate on an aggressive signature BEFORE splitting (no leakage),
 - synthetic (augmented, human-kept) messages are TRAIN-ONLY,
 - val/test contain only human (non-synthetic) messages,
 - no val/test message may share a dedup signature with a train message,
 - seed 42, stratified by label.

Run:  python -m indosmish.data.dedup_split
Output: data/processed/corpus.csv + data/processed/splits/{train,val,test}.csv
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import load_config, resolve
from .build_corpus import detect_script, _hash_id
from .schema import COLUMNS, dedup_key, normalize_text


def _load_kept_augmented(cfg: dict) -> pd.DataFrame:
    path = resolve(cfg["augment"]["out"])
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("keep"):
                continue
            text = normalize_text(r["text"])
            rows.append(
                {
                    "id": _hash_id(text, "augmented"),
                    "text": text,
                    "label": "smishing",
                    "script": detect_script(text),
                    "source": "augmented",
                    "synthetic": True,
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]

    unified = pd.read_csv(resolve(cfg["paths"]["interim"]) / "unified.csv", encoding="utf-8")
    # CSV round-trips booleans as strings; coerce back.
    unified["synthetic"] = unified["synthetic"].astype(str).str.lower().isin(("true", "1"))
    aug = _load_kept_augmented(cfg)
    print(f"Real: {len(unified)}  |  Augmented(kept): {len(aug)}")

    df = pd.concat([unified, aug], ignore_index=True)
    df["synthetic"] = df["synthetic"].astype(str).str.lower().isin(("true", "1"))
    df["text"] = df["text"].map(normalize_text)
    df = df[df["text"].str.len() > 0]
    df["sig"] = df["text"].map(dedup_key)

    # Collapse exact-signature duplicates, preferring a real (non-synthetic) copy.
    df = df.sort_values("synthetic").drop_duplicates(subset=["sig"], keep="first")
    print(f"After dedup: {len(df)}  {df['label'].value_counts().to_dict()}")

    real = df[~df["synthetic"]].copy()
    synth = df[df["synthetic"]].copy()

    # Split REAL data 70/15/15 stratified; synthetic is appended to train only.
    train_r, tmp = train_test_split(
        real, test_size=cfg["split"]["val"] + cfg["split"]["test"],
        stratify=real["label"], random_state=seed,
    )
    rel_test = cfg["split"]["test"] / (cfg["split"]["val"] + cfg["split"]["test"])
    val_r, test_r = train_test_split(
        tmp, test_size=rel_test, stratify=tmp["label"], random_state=seed,
    )

    # Guard: drop any synthetic row whose signature collides with val/test.
    heldout_sigs = set(val_r["sig"]) | set(test_r["sig"])
    before = len(synth)
    synth = synth[~synth["sig"].isin(heldout_sigs)]
    if before != len(synth):
        print(f"  dropped {before - len(synth)} synthetic rows colliding with val/test")

    train = pd.concat([train_r, synth], ignore_index=True)

    out_dir = resolve(cfg["paths"]["splits"])
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = COLUMNS  # drop the transient 'sig' column
    for name, part in [("train", train), ("val", val_r), ("test", test_r)]:
        part[keep].to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")

    df[keep].to_csv(resolve(cfg["paths"]["processed"]) / "corpus.csv", index=False, encoding="utf-8")

    def dist(p):
        return p["label"].value_counts().to_dict()

    print(f"\nFrozen splits -> {out_dir}")
    print(f"  train {len(train)}  {dist(train)}  (synthetic in train: {int(train['synthetic'].sum())})")
    print(f"  val   {len(val_r)}  {dist(val_r)}")
    print(f"  test  {len(test_r)}  {dist(test_r)}")
    print("  val/test contain NO synthetic rows:",
          not val_r["synthetic"].any() and not test_r["synthetic"].any())


if __name__ == "__main__":
    main()
