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
from .build_corpus import _hash_id
from .schema import COLUMNS, detect_lang, detect_script, dedup_key, normalize_text


def _load_one_augmented(path: Path, source: str) -> list[dict]:
    if not path.exists():
        return []
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
                    "id": _hash_id(text, source),
                    "text": text,
                    "label": r.get("label", "smishing"),  # respect ham hard-negatives
                    "script": detect_script(text),
                    "lang": detect_lang(text, source),
                    "source": source,
                    "synthetic": True,
                }
            )
    return rows


def _load_kept_augmented(cfg: dict) -> pd.DataFrame:
    """Kept augmented rows from every language's review file (Malayalam + Hindi)."""
    acfg = cfg["augment"]
    rows = _load_one_augmented(resolve(acfg["out"]), "augmented_ml")
    if acfg.get("out_hi"):
        hi = resolve(acfg["out_hi"])
        rows += _load_one_augmented(hi, "augmented_hi")
        rows += _load_one_augmented(hi.with_name(hi.stem + "_deva" + hi.suffix), "augmented_hi")
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

    rel_test = cfg["split"]["test"] / (cfg["split"]["val"] + cfg["split"]["test"])

    def _split3(part):
        """70/15/15 stratified by label, tolerant of tiny classes."""
        strat = part["label"] if part["label"].value_counts().min() >= 2 else None
        tr, tmp = train_test_split(
            part, test_size=cfg["split"]["val"] + cfg["split"]["test"],
            stratify=strat, random_state=seed,
        )
        strat2 = tmp["label"] if tmp["label"].value_counts().min() >= 2 else None
        va, te = train_test_split(tmp, test_size=rel_test, stratify=strat2, random_state=seed)
        return tr, va, te

    # Partition real data so the ml/en split is reproduced EXACTLY as before Hindi
    # existed (frozen — paper numbers stay comparable), and Hindi is split on its own.
    real_nonhi = real[real["lang"] != "hi"]
    real_hi = real[real["lang"] == "hi"]
    gold_hi = real_hi[real_hi["source"] == "gold_hi_smishing"]     # TEST-ONLY held-out
    real_hi_rest = real_hi[real_hi["source"] != "gold_hi_smishing"]

    train_n, val_n, test_n = _split3(real_nonhi)   # identical to the frozen ml/en splits
    parts_train, parts_val, parts_test = [train_n], [val_n], [test_n]
    if len(real_hi_rest):
        train_h, val_h, test_h = _split3(real_hi_rest)
        parts_train += [train_h]; parts_val += [val_h]; parts_test += [test_h]
    if len(gold_hi):
        parts_test += [gold_hi]                    # hand-authored Hindi smishing -> test only

    train_r = pd.concat(parts_train, ignore_index=True)
    val_r = pd.concat(parts_val, ignore_index=True)
    test_r = pd.concat(parts_test, ignore_index=True)

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

    def lang_dist(p):
        return p["lang"].value_counts().to_dict()

    print(f"\nFrozen splits -> {out_dir}")
    print(f"  train {len(train)}  {dist(train)}  lang={lang_dist(train)}  (synthetic in train: {int(train['synthetic'].sum())})")
    print(f"  val   {len(val_r)}  {dist(val_r)}  lang={lang_dist(val_r)}")
    print(f"  test  {len(test_r)}  {dist(test_r)}  lang={lang_dist(test_r)}")
    print("  Hindi test smishing (gold):",
          int(((test_r['lang'] == 'hi') & (test_r['label'] == 'smishing')).sum()))
    print("  val/test contain NO synthetic rows:",
          not val_r["synthetic"].any() and not test_r["synthetic"].any())


if __name__ == "__main__":
    main()
