from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lolwin.data import load_player_csv, prepare_dataset
from lolwin.modeling import train_random_forest_grid
from lolwin.viz import save_feature_importance, save_correlation_heatmap
from lolwin.reporting import write_json, write_markdown_report


def main():
    ap = argparse.ArgumentParser(description="LoL win/loss prediction (player stats) - train & eval")
    ap.add_argument("--csv", required=True, help="Path to a player CSV (e.g., data/Player_data/T1_Faker.csv)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=3, help="GridSearchCV folds")
    ap.add_argument("--out", default="reports", help="Output directory")
    ap.add_argument("--max_features_heatmap", type=int, default=None, help="Limit features shown in heatmap")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_player_csv(csv_path)
    ds = prepare_dataset(df)

    result = train_random_forest_grid(ds.X, ds.y, seed=args.seed, cv_folds=args.cv)

    metrics = {
        "csv": str(csv_path),
        "seed": args.seed,
        "cv_folds": args.cv,
        "dataset": ds.meta,
        "cv_best_score": result.cv_best_score,
        "test_accuracy": result.test_accuracy,
        "best_params": result.best_params,
    }
    write_json(out_dir / "metrics.json", metrics)

    fi_df = save_feature_importance(
        ds.X.columns,
        result.best_estimator.feature_importances_,
        out_dir / "feature_importance.png",
        top_n=15,
        title=f"Top Feature Importances ({csv_path.stem})",
    )
    fi_df.to_csv(out_dir / "feature_importance_top15.csv", index=False)

    save_correlation_heatmap(
        ds.X,
        out_dir / "corr_heatmap.png",
        title=f"Correlation Heatmap ({csv_path.stem})",
        max_features=args.max_features_heatmap,
    )

    write_markdown_report(
        out_dir / "report.md",
        title=f"LoL Win/Loss Prediction Report ({csv_path.stem})",
        dataset_info=ds.meta,
        train_info={
            "cv_best_score": result.cv_best_score,
            "test_accuracy": result.test_accuracy,
            "best_params": result.best_params,
        },
        classification_report=result.report,
        confusion_matrix=str(result.confusion),
        notes="\n**Label rule**: Win/Loss = 1 if `P value != 0` else 0 (from original notebook).\n",
    )

    print("✅ Done")
    print(f"- report: {out_dir/'report.md'}")


if __name__ == "__main__":
    main()
