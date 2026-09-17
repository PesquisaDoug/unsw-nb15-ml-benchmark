from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

CV_SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "mcc": make_scorer(matthews_corrcoef),
}


def positive_scores(estimator: Any, X_input: Any) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X_input)
        return np.asarray(probabilities[:, 1])
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X_input))
    return np.asarray(estimator.predict(X_input), dtype=float)


def false_positive_rate(y_true: Any, y_pred: Any) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    normal = y_true == 0
    denominator = normal.sum()
    if denominator == 0:
        return float("nan")
    return float(((y_pred == 1) & normal).sum() / denominator)


def classification_metrics(y_true: Any, y_pred: Any, y_score: Any) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_detection_rate": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "false_positive_rate": false_positive_rate(y_true, y_pred),
    }


def timed_inference(estimator: Any, X_input: Any, repeats: int = 10) -> dict[str, float]:
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        estimator.predict(X_input)
        elapsed.append(time.perf_counter() - start)
    median_seconds = float(np.median(elapsed))
    return {
        "inference_ms_total_median": median_seconds * 1000,
        "inference_us_per_record_median": median_seconds * 1_000_000 / len(X_input),
    }


def save_model_and_size(estimator: Any, name: str, models_dir: Path) -> tuple[Path, float]:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(estimator, path, compress=3)
    return path, path.stat().st_size / (1024**2)


def bootstrap_confidence_intervals(
    y_true: Any,
    y_pred: Any,
    y_score: Any,
    iterations: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    collectors: dict[str, list[float]] = {
        "f1": [],
        "roc_auc": [],
        "pr_auc": [],
        "mcc": [],
        "recall_detection_rate": [],
        "false_positive_rate": [],
    }
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if np.unique(yt).size < 2:
            continue
        yp = y_pred[idx]
        ys = y_score[idx]
        collectors["f1"].append(f1_score(yt, yp, zero_division=0))
        collectors["roc_auc"].append(roc_auc_score(yt, ys))
        collectors["pr_auc"].append(average_precision_score(yt, ys))
        collectors["mcc"].append(matthews_corrcoef(yt, yp))
        collectors["recall_detection_rate"].append(recall_score(yt, yp, zero_division=0))
        collectors["false_positive_rate"].append(false_positive_rate(yt, yp))

    rows = []
    for metric, values in collectors.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        rows.append(
            {
                "metric": metric,
                "bootstrap_mean": float(arr.mean()) if len(arr) else float("nan"),
                "ci_lower": float(np.quantile(arr, alpha / 2)) if len(arr) else float("nan"),
                "ci_upper": float(np.quantile(arr, 1 - alpha / 2)) if len(arr) else float("nan"),
                "valid_bootstrap_samples": int(len(arr)),
            }
        )
    return pd.DataFrame(rows)


def attackwise_detection_metrics(
    prediction_frame: pd.DataFrame,
    model_names: list[str],
) -> pd.DataFrame:
    attack_only = prediction_frame.loc[prediction_frame["y_true"] == 1].copy()
    rows = []
    for name in model_names:
        for attack_cat, group in attack_only.groupby("attack_cat"):
            n = len(group)
            detected = int((group[f"{name}_prediction"] == 1).sum())
            missed = n - detected
            rows.append(
                {
                    "model": name,
                    "attack_cat": attack_cat,
                    "n": n,
                    "detected": detected,
                    "missed": missed,
                    "recall": detected / n if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def validate_inference_schema(frame: pd.DataFrame, feature_names: list[str]) -> list[str]:
    return [feature for feature in feature_names if feature not in frame.columns]
