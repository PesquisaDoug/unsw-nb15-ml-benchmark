from __future__ import annotations

from pathlib import Path
from typing import Any

from scipy.stats import loguniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import xgboost as xgb

from .preprocessing import build_scaled_preprocessor, build_tree_preprocessor
from .reproducibility import load_yaml


def _convert_search_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get("distribution") == "loguniform":
        return loguniform(float(value["low"]), float(value["high"]))
    return value


def load_model_config(path: Path) -> dict[str, Any]:
    return load_yaml(path)


def build_models(
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    random_seed: int,
    model_config: dict[str, Any],
) -> dict[str, Pipeline]:
    tree_preprocessor = build_tree_preprocessor(numeric_columns, categorical_columns)
    scaled_preprocessor = build_scaled_preprocessor(numeric_columns, categorical_columns)

    return {
        "Dummy": Pipeline(
            [
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    DummyClassifier(
                        strategy="prior",
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=random_seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    xgb.XGBClassifier(
                        random_state=random_seed,
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("preprocess", scaled_preprocessor),
                (
                    "model",
                    LinearSVC(
                        random_state=random_seed,
                        dual="auto",
                        max_iter=5000,
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("preprocess", scaled_preprocessor),
                (
                    "model",
                    MLPClassifier(
                        random_state=random_seed,
                        early_stopping=True,
                        max_iter=300,
                    ),
                ),
            ]
        ),
    }


def build_search_spaces(model_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    spaces: dict[str, dict[str, Any]] = {}
    for model_name, params in model_config["search_spaces"].items():
        spaces[model_name] = {
            name: _convert_search_value(value) for name, value in params.items()
        }
    return spaces
