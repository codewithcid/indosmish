"""S3 — Classical floor: TF-IDF + Random Forest (paper's recurring strong baseline).

Trains on all script arms combined, evaluates per arm. CPU-only, seconds to run.

Run:  python -m indosmish.models.classical
Output: results/classical_{arm}.json
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from ..config import load_config
from ..eval.metrics import compute_metrics, print_summary, save_result
from .data_utils import filter_arm, load_split


def main() -> None:
    cfg = load_config()
    train, test = load_split("train"), load_split("test")

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=50000,
            )),
            ("rf", RandomForestClassifier(
                n_estimators=400, class_weight="balanced",
                random_state=cfg["seed"], n_jobs=-1,
            )),
        ]
    )
    pipe.fit(train["text"], train["label"])

    for arm in ("roman", "native"):
        part = filter_arm(test, arm)
        if len(part) < 5:
            print(f"[skip] arm={arm}: only {len(part)} test rows")
            continue
        pred = pipe.predict(part["text"])
        m = compute_metrics(part["label"].tolist(), list(pred))
        m["model"], m["arm"] = "tfidf_rf", arm
        print_summary(f"TF-IDF+RF [{arm}]", m)
        save_result(m, f"results/classical_{arm}.json")


if __name__ == "__main__":
    main()
