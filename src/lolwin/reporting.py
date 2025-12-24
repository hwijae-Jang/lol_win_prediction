from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json
import datetime as dt


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_report(
    path: str | Path,
    *,
    title: str,
    dataset_info: Dict[str, Any],
    train_info: Dict[str, Any],
    classification_report: str,
    confusion_matrix: str,
    notes: str = "",
) -> None:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    md = f"""# {title}

Generated: {now}

## Dataset
- rows: {dataset_info.get('n_rows')}
- features: {dataset_info.get('n_features')}
- win_rate: {dataset_info.get('win_rate'):.3f}

## Training
- model: RandomForestClassifier + GridSearchCV
- cv_best_score: {train_info.get('cv_best_score'):.4f}
- test_accuracy: {train_info.get('test_accuracy'):.4f}
- best_params: `{train_info.get('best_params')}`

## Classification report
```text
{classification_report}
```

## Confusion matrix
```text
{confusion_matrix}
```

{notes}
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(md, encoding="utf-8")
