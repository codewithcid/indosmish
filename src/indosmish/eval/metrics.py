"""Protocol metrics (S2): macro-F1 primary, per-class breakdown, smishing recall.

All model scripts import compute_metrics so every result is produced identically.
"""
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from ..data.schema import LABELS


def compute_metrics(y_true, y_pred) -> dict:
    """y_true / y_pred are label strings or ints aligned to schema.LABELS order."""
    labels = LABELS
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    smishing_recall = recall_score(
        y_true, y_pred, labels=["smishing"], average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "smishing_recall": smishing_recall,
        "per_class": {
            lbl: {
                "precision": report[lbl]["precision"],
                "recall": report[lbl]["recall"],
                "f1": report[lbl]["f1-score"],
                "support": report[lbl]["support"],
            }
            for lbl in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_result(result: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def print_summary(name: str, m: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"  macro-F1        : {m['macro_f1']:.4f}")
    print(f"  accuracy        : {m['accuracy']:.4f}")
    print(f"  smishing recall : {m['smishing_recall']:.4f}")
    for lbl, v in m["per_class"].items():
        print(f"    {lbl:9s} P={v['precision']:.3f} R={v['recall']:.3f} "
              f"F1={v['f1']:.3f} n={int(v['support'])}")
