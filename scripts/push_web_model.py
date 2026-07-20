"""Push the ONNX model to a free HF model repo in the layout transformers.js expects
(model.onnx under onnx/, tokenizer+config at root), for in-browser inference.

Model repos are free (only Spaces need PRO). Requires HF_TOKEN (write).
Run:  python scripts/push_web_model.py --repo cidcodes/indosmish-indicbert-onnx
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (write).")

    from huggingface_hub import HfApi, create_repo

    src = ROOT / "app" / "backend" / "model" / "onnx"
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        (stage / "onnx").mkdir()
        shutil.copy(src / "model.onnx", stage / "onnx" / "model.onnx")
        for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "spiece.model"):
            if (src / f).exists():
                shutil.copy(src / f, stage / f)
        # A tiny README so the repo card isn't empty.
        (stage / "README.md").write_text(
            "---\nlibrary_name: transformers.js\npipeline_tag: text-classification\n"
            "license: mit\n---\n\n# IndoSmish IndicBERT (ONNX)\n\n"
            "Tri-class ham/spam/smishing detector for code-mixed Malayalam-English SMS, "
            "fine-tuned IndicBERT exported to ONNX for in-browser inference via transformers.js.\n",
            encoding="utf-8")

        create_repo(args.repo, token=token, repo_type="model", exist_ok=True, private=False)
        print(f"Uploading (~{sum(f.stat().st_size for f in stage.rglob('*') if f.is_file())/1e6:.0f} MB)...")
        HfApi().upload_folder(folder_path=str(stage), repo_id=args.repo, repo_type="model",
                              token=token, commit_message="ONNX model for transformers.js")
    print(f"Done -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
