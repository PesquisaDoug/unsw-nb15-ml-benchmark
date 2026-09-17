from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.evaluation import (
    attackwise_detection_metrics,
    bootstrap_confidence_intervals,
    classification_metrics,
    false_positive_rate,
    positive_scores,
)


def test_false_positive_rate() -> None:
    assert false_positive_rate([0, 0, 1, 1], [0, 1, 1, 0]) == 0.5


def test_classification_metrics_and_bootstrap() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_score = np.array([0.1, 0.8, 0.9, 0.7, 0.2, 0.3])
    metrics = classification_metrics(y_true, y_pred, y_score)
    assert "false_positive_rate" in metrics
    ci = bootstrap_confidence_intervals(y_true, y_pred, y_score, iterations=10, seed=1)
    assert set(ci["metric"]).issuperset({"f1", "roc_auc", "false_positive_rate"})


def test_positive_scores_predict_proba_and_decision_function() -> None:
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    lr = LogisticRegression().fit(X, y)
    svc = LinearSVC(dual="auto").fit(X, y)
    assert positive_scores(lr, X).shape == (4,)
    assert positive_scores(svc, X).shape == (4,)


def test_attackwise_detection_metrics() -> None:
    frame = pd.DataFrame(
        {
            "y_true": [1, 1, 1, 0],
            "attack_cat": ["A", "A", "B", "Normal"],
            "M_prediction": [1, 0, 1, 0],
        }
    )
    result = attackwise_detection_metrics(frame, ["M"])
    assert set(result["attack_cat"]) == {"A", "B"}
    assert result.loc[result["attack_cat"] == "A", "recall"].iloc[0] == 0.5
