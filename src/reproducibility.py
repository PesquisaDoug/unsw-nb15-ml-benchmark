from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import requests
import scipy
import sklearn
import yaml

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - imported during environment setup
    xgb = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def ensure_directories(project_root: Path) -> dict[str, Path]:
    paths = {
        "data": project_root / "data",
        "raw_data": project_root / "data" / "raw",
        "results": project_root / "results",
        "figures": project_root / "figures",
        "models": project_root / "artifacts" / "models",
        "app": project_root / "app",
        "paper": project_root / "paper",
        "paper_figures": project_root / "paper" / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def get_git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def package_versions() -> dict[str, str | None]:
    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgb.__version__ if xgb is not None else None,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
        "requests": requests.__version__,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, str)) else False:
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2),
        encoding="utf-8",
    )


def build_experiment_manifest(
    *,
    project_root: Path,
    config: dict[str, Any],
    profile: str,
    data_manifest: dict[str, Any],
    feature_schema: dict[str, Any],
    started_at_utc: str,
    best_model_name: str | None = None,
    best_model_metrics: dict[str, Any] | None = None,
    selected_dbscan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_config = config["profiles"][profile]
    dataset = data_manifest["dataset"]
    return {
        "experiment_name": "unsw_nb15_classical_ml_benchmark",
        "profile": profile,
        "seed": config["random_seed"],
        "quick_mode": profile == "quick",
        "training_sha256": data_manifest["training_file"]["sha256"],
        "testing_sha256": data_manifest["testing_file"]["sha256"],
        "train_rows": data_manifest["training_file"]["rows"],
        "test_rows": data_manifest["testing_file"]["rows"],
        "input_feature_count": feature_schema["feature_count"],
        "numeric_feature_count": len(feature_schema["numeric_features"]),
        "categorical_features": feature_schema["categorical_features"],
        "excluded_columns": feature_schema["excluded_columns"],
        "target": feature_schema["target"],
        "tuning_cv_folds": config["cross_validation"]["tuning_folds"],
        "evaluation_cv_folds": profile_config["evaluation_folds"],
        "tuning_iterations": profile_config["tuning_iterations"],
        "bootstrap_iterations": profile_config["bootstrap_iterations"],
        "cluster_sample_size": profile_config["cluster_sample_size"],
        "importance_sample_size": profile_config["importance_sample_size"],
        "importance_repeats": profile_config["importance_repeats"],
        "git_commit": get_git_commit(project_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "dataset": dataset,
        "data_manifest": data_manifest,
        "best_model_for_demo": best_model_name,
        "best_model_test_metrics": best_model_metrics or {},
        "selected_dbscan": selected_dbscan or {},
        "started_at_utc": started_at_utc,
        "completed_at_utc": utc_now(),
    }


def build_paper_summary(
    *,
    data_manifest: dict[str, Any],
    feature_schema: dict[str, Any],
    best_model_name: str | None,
) -> dict[str, Any]:
    return {
        "dataset": "UNSW-NB15",
        "canonical_source": data_manifest["canonical_source"],
        "train_rows": data_manifest["training_file"]["rows"],
        "test_rows": data_manifest["testing_file"]["rows"],
        "input_features": feature_schema["feature_count"],
        "supervised_models": ["Dummy", "RandomForest", "XGBoost", "LinearSVM", "MLP"],
        "unsupervised_models": ["KMeans", "DBSCAN"],
        "best_demo_model": best_model_name,
        "main_results": {
            "supervised": "results/metrics.csv",
            "cross_validation": "results/cv_metrics.csv",
            "bootstrap_ci": "results/bootstrap_ci.csv",
            "attackwise": "results/attackwise_detection.csv",
            "clustering": "results/clustering_metrics.csv",
        },
        "figures_directory": "figures/",
    }
