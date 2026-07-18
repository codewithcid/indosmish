"""Shared split loading + per-script-arm filtering for model scripts."""
import pandas as pd

from ..config import load_config, resolve


def load_split(name: str) -> pd.DataFrame:
    cfg = load_config()
    return pd.read_csv(resolve(cfg["paths"]["splits"]) / f"{name}.csv", encoding="utf-8")


def filter_arm(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Select the script arm for evaluation (protocol section 4).

    roman  -> roman-script rows only
    native -> native-script rows only
    all    -> everything (used for training a single model across arms)
    xlit-norm is handled at inference time by normalizing roman input; for training
    it is treated like 'all'.
    """
    if arm in ("all", "xlit-norm"):
        return df
    return df[df["script"] == arm]
