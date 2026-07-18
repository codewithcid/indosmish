"""Offline smoke test: fabricate tiny raw CSVs, run the no-network pipeline stages,
assert protocol invariants. Not a substitute for real data — just plumbing validation.

Run:  python scripts/smoke_test.py
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def fabricate():
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    ham = [f"see you at {h} pm bro" for h in range(1, 40)]
    spam = [f"MEGA SALE {n}% off visit store now" for n in range(1, 40)]
    smish = [f"account blocked click http://bit.ly/a{n} to verify now" for n in range(1, 40)]
    ml = ["ningal 25 lakh won link click cheyyu http://x.co/p",
          "bank kyc expired verify cheyyu illenkil block"]

    rows = ([{"Message": t, "Label": "ham"} for t in ham]
            + [{"Message": t, "Label": "spam"} for t in spam])
    pd.DataFrame(rows).to_csv(raw / "dravidian_spam.csv", index=False)

    rows2 = ([{"TEXT": t, "LABEL": "ham"} for t in ham[:20]]
             + [{"TEXT": t, "LABEL": "spam"} for t in spam[:20]]
             + [{"TEXT": t, "LABEL": "smishing"} for t in smish]
             + [{"TEXT": t, "LABEL": "smishing"} for t in ml])
    pd.DataFrame(rows2).to_csv(raw / "sms_phishing.csv", index=False)


def run(mod):
    print(f"\n$ python -m {mod}")
    r = subprocess.run([sys.executable, "-m", mod], cwd=ROOT,
                       capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT / "src"), **_env()})
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
        raise SystemExit(f"{mod} failed")


def _env():
    import os
    return os.environ.copy()


def assert_invariants():
    splits = ROOT / "data" / "processed" / "splits"
    tr = pd.read_csv(splits / "train.csv")
    va = pd.read_csv(splits / "val.csv")
    te = pd.read_csv(splits / "test.csv")

    # Protocol: no synthetic in val/test (none here, but structure must hold).
    assert not va["synthetic"].any(), "synthetic leaked into val"
    assert not te["synthetic"].any(), "synthetic leaked into test"

    # Protocol: dedup-before-split — no exact text shared across train/test.
    from indosmish.data.schema import dedup_key
    tr_sig = set(tr["text"].map(dedup_key))
    te_sig = set(te["text"].map(dedup_key))
    assert not (tr_sig & te_sig), "duplicate signatures across train/test"

    # All three classes present in test.
    assert set(te["label"]) == {"ham", "spam", "smishing"}, f"missing class: {set(te['label'])}"
    print("\n[OK] protocol invariants hold: no synth in val/test, no train/test leakage, 3 classes.")


if __name__ == "__main__":
    fabricate()
    run("indosmish.data.build_corpus")
    run("indosmish.data.dedup_split")
    run("indosmish.models.classical")
    run("indosmish.eval.build_tables")
    assert_invariants()
    print("\nSMOKE TEST PASSED")
