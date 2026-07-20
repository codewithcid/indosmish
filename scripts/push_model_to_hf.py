"""Upload the exported fp32 ONNX encoder to a public HuggingFace model repo, so the
Render backend can download it at startup (keeps the git repo small; sidesteps GitHub's
100MB file limit).

Prereq: HF_TOKEN env var (write-scope token) and a HF account.
Run:  python scripts/push_model_to_hf.py --repo <your-username>/indosmish-indicbert-onnx
Then set MODEL_REPO to that id on Render.
"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. codewithcid/indosmish-indicbert-onnx")
    ap.add_argument("--src", default="app/backend/model/onnx", help="ONNX dir to upload")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (write scope): $env:HF_TOKEN = 'hf_...'")

    from huggingface_hub import HfApi, create_repo

    src = ROOT / args.src
    if not (src / "model.onnx").exists():
        raise SystemExit(f"No model.onnx in {src}. Run export_onnx first.")

    create_repo(args.repo, token=token, repo_type="model", exist_ok=True, private=False)
    HfApi().upload_folder(
        folder_path=str(src), repo_id=args.repo, repo_type="model", token=token,
        commit_message="IndoSmish IndicBERT ONNX (fp32) for Render deployment",
    )
    print(f"Uploaded {src} -> https://huggingface.co/{args.repo}")
    print(f"Set MODEL_REPO={args.repo} on Render.")


if __name__ == "__main__":
    main()
