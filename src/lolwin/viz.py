from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def save_feature_importance(feature_names, importances, out_path: str | Path, *, top_n: int = 15, title: str = "Top Feature Importances") -> pd.DataFrame:
    df = pd.DataFrame({"feature": list(feature_names), "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(df["feature"][::-1], df["importance"][::-1])
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

    return df


def save_correlation_heatmap(X: pd.DataFrame, out_path: str | Path, *, title: str = "Feature Correlation Heatmap", max_features: Optional[int] = None) -> None:
    if max_features is not None and X.shape[1] > max_features:
        X = X.iloc[:, :max_features]

    corr = X.corr(numeric_only=True)

    plt.figure(figsize=(12, 10))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=6)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
