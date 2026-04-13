"""
Data loading: knowledge base + transformed files.
"""

import pandas as pd
from .config import (
    KB_FILE, DATA_DIR, QI_N1, QI_N2, QI_N3, LEVELS, PERCENTAGES, MODES,
)


def load_kb() -> pd.DataFrame:
    """Load the external knowledge base (ground truth)."""
    return pd.read_csv(KB_FILE, sep="\t")


def build_kb_views(df_kb: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build the three views (N1, N2, N3) of the knowledge base.
    Returns a dict {"n1": df_n1, "n2": df_n2, "n3": df_n3}.
    """
    cols_n1 = ["id_sejour"] + QI_N1
    cols_n2 = ["id_sejour"] + QI_N2
    cols_n3 = ["id_sejour"] + QI_N3

    available = set(df_kb.columns)
    return {
        "n1": df_kb[[c for c in cols_n1 if c in available]].copy(),
        "n2": df_kb[[c for c in cols_n2 if c in available]].copy(),
        "n3": df_kb[[c for c in cols_n3 if c in available]].copy(),
    }


def load_transformed(mode: str, pct: int) -> pd.DataFrame:
    """Load a transformed file out_{mode}_{pct}.txt."""
    path = f"{DATA_DIR}/out_{mode}_{pct}.txt"
    return pd.read_csv(path, sep="\t")


def iter_scenarios(percentages=None, modes=None):
    """
    Generator yielding (pct, mode, df_anon) over requested scenarios.
    Silently skips missing files.
    """
    percentages = percentages or PERCENTAGES
    modes = modes or MODES
    for pct in percentages:
        for mode in modes:
            try:
                df = load_transformed(mode, pct)
                yield pct, mode, df
            except FileNotFoundError:
                continue


def build_configs(kb_views: dict[str, pd.DataFrame]) -> list[tuple]:
    """
    Returns a list of (name, qi_list, df_kb) tuples for use in loops.
    """
    return [
        ("n1", LEVELS["n1"]["qi"], kb_views["n1"]),
        ("n2", LEVELS["n2"]["qi"], kb_views["n2"]),
        ("n3", LEVELS["n3"]["qi"], kb_views["n3"]),
    ]
