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

    # Mishra & Soni tri-class (TEXT/LABEL)
    rows = ([{"TEXT": t, "LABEL": "ham"} for t in ham]
            + [{"TEXT": t, "LABEL": "spam"} for t in spam]
            + [{"TEXT": t, "LABEL": "smishing"} for t in smish])
    pd.DataFrame(rows).to_csv(raw / "sms_phishing.csv", index=False)

    # UCI SMS (text/label, English ham/spam)
    pd.DataFrame(
        [{"text": t, "label": "ham"} for t in ham[:15]]
        + [{"text": t, "label": "spam"} for t in spam[:15]]
    ).to_csv(raw / "uci_sms.csv", index=False)

    # DravidianCodeMix codemix_ham (text/label) — native-script Not_offensive -> ham
    ml_native = [f"ente veedu {n} manikku adakkam aanu, vaa" for n in range(1, 20)]
    # add Malayalam-script char so detect_script() sees 'native'
    ml_native = [t + " ✅ മലയാളം" for t in ml_native]
    pd.DataFrame(
        [{"text": t, "label": "Not_offensive"} for t in ml_native]
        + [{"text": "off topic english only", "label": "Offensive_Untargeted"}]
    ).to_csv(raw / "dravidian_codemix_ml.csv", index=False)


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
