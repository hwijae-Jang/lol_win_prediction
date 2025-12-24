from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    meta: dict


def load_player_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a single player's match-level stats CSV.

    Expected columns:
    - numeric features
    - 'P value' (used to derive Win/Loss label, following original notebooks)
    """
    df = pd.read_csv(csv_path)

    # Drop unnamed index-like columns
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if "P value" not in df.columns:
        raise ValueError(f"Missing required column 'P value' in: {csv_path}")

    return df


def add_win_loss_label(df: pd.DataFrame, p_value_col: str = "P value", label_col: str = "Win/Loss") -> pd.DataFrame:
    """Create Win/Loss label from 'P value'.

    Original notebook logic:
    - 0 => Loss (0)
    - non-zero => Win (1)
    """
    out = df.copy()
    out[label_col] = (out[p_value_col].fillna(0) != 0).astype(int)
    return out


def prepare_dataset(df: pd.DataFrame, *, label_col: str = "Win/Loss", p_value_col: str = "P value") -> Dataset:
    df = add_win_loss_label(df, p_value_col=p_value_col, label_col=label_col)

    X = df.drop(columns=[c for c in [p_value_col, label_col] if c in df.columns])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Coerce non-numeric to numeric (rare here)
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    y = df[label_col].astype(int)
    meta = {"n_rows": int(len(df)), "n_features": int(X.shape[1]), "win_rate": float(y.mean())}
    return Dataset(X=X, y=y, meta=meta)
