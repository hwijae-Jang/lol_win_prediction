from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass(frozen=True)
class TrainResult:
    best_estimator: RandomForestClassifier
    best_params: Dict[str, Any]
    cv_best_score: float
    test_accuracy: float
    y_test: np.ndarray
    y_pred: np.ndarray
    report: str
    confusion: np.ndarray


def train_random_forest_grid(
    X,
    y,
    *,
    seed: int = 42,
    test_size: float = 0.2,
    cv_folds: int = 3,
    n_jobs: int = -1,
    param_grid: Optional[Dict[str, Any]] = None,
) -> TrainResult:
    """RandomForest + GridSearchCV, then evaluate on held-out test set.

    Defaults are reduced for speed (still matches the 'CV + tuning' story).
    """
    if param_grid is None:
        param_grid = {
            "max_depth": [6, 8, 10],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [1, 2],
            "n_estimators": [300],
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    base = RandomForestClassifier(random_state=seed, n_jobs=n_jobs)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(base, param_grid=param_grid, cv=cv, n_jobs=n_jobs, scoring="accuracy", refit=True)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    rep = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    return TrainResult(
        best_estimator=best,
        best_params=dict(grid.best_params_),
        cv_best_score=float(grid.best_score_),
        test_accuracy=acc,
        y_test=np.array(y_test),
        y_pred=np.array(y_pred),
        report=rep,
        confusion=cm,
    )
