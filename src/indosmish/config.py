"""Config loading and repo-root path resolution."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(rel: str) -> Path:
    """Resolve a repo-relative path from the config into an absolute one."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p
