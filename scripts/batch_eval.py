from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lolwin.data import load_player_csv, prepare_dataset
from lolwin.modeling import train_random_forest_grid


def main():
    ap = argparse.ArgumentParser(description="Batch evaluate multiple player CSVs")
    ap.add_argument("--dir", default="data/Player_data", help="Directory containing player CSVs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--out", default="reports/batch_summary.csv")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of CSVs (debug)")
    args = ap.parse_args()

    d = Path(args.dir)
    csvs = sorted(d.glob("*.csv"))
    if args.limit:
        csvs = csvs[: args.limit]

    rows = []
    for csv_path in csvs:
        df = load_player_csv(csv_path)
        ds = prepare_dataset(df)
        res = train_random_forest_grid(ds.X, ds.y, seed=args.seed, cv_folds=args.cv)

        rows.append({
            "file": csv_path.name,
            "rows": ds.meta["n_rows"],
            "features": ds.meta["n_features"],
            "win_rate": ds.meta["win_rate"],
            "cv_best_score": res.cv_best_score,
            "test_accuracy": res.test_accuracy,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("test_accuracy", ascending=False).to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
