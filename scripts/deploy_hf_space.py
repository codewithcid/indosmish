"""Deploy the IndoSmish FastAPI+ONNX backend to a HuggingFace Docker Space.

Assembles the Space files (Dockerfile + README from app/space, main.py + requirements
from app/backend, and the ONNX model) into a staging dir and pushes it. HF repos handle
large files natively, so the 128MB model uploads without GitHub's limits.

Prereq: HF_TOKEN with WRITE scope.
Run:  python scripts/deploy_hf_space.py --repo <username>/indosmish-api
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF Space id, e.g. codewithcid/indosmish-api")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (WRITE scope): $env:HF_TOKEN = 'hf_...'")

    from huggingface_hub import HfApi, create_repo

    backend = ROOT / "app" / "backend"
    space = ROOT / "app" / "space"
    model = backend / "model" / "onnx"
    if not (model / "model.onnx").exists():
        raise SystemExit(f"Model not found at {model}. Run export_onnx first.")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copy(space / "Dockerfile", stage / "Dockerfile")
        shutil.copy(space / "README.md", stage / "README.md")
        shutil.copy(backend / "main.py", stage / "main.py")
        shutil.copy(backend / "requirements.txt", stage / "requirements.txt")
        shutil.copytree(model, stage / "model" / "onnx")

        create_repo(args.repo, token=token, repo_type="space", space_sdk="docker",
                    exist_ok=True)
        print(f"Uploading Space files ({sum(f.stat().st_size for f in stage.rglob('*') if f.is_file())/1e6:.0f} MB)...")
        HfApi().upload_folder(
            folder_path=str(stage), repo_id=args.repo, repo_type="space", token=token,
            commit_message="Deploy IndoSmish FastAPI+ONNX backend",
        )

    print(f"\nDeployed -> https://huggingface.co/spaces/{args.repo}")
    user, name = args.repo.split("/")
    print(f"API base -> https://{user}-{name}.hf.space")
    print("Build takes ~3-5 min. Test: <API base>/health")


if __name__ == "__main__":
    main()
