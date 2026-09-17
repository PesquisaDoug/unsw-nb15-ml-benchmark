from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.data import prepare_dataset
from src.reproducibility import (
    build_paper_summary,
    ensure_directories,
    load_yaml,
    utc_now,
    write_json,
)

REQUIRED_OUTPUTS = [
    "data/data_manifest.json",
    "results/data_quality_summary.csv",
    "results/schema_summary.csv",
    "results/tuning_summary.csv",
    "results/cv_metrics.csv",
    "results/metrics.csv",
    "results/predictions.csv",
    "results/bootstrap_ci.csv",
    "results/attackwise_detection.csv",
    "results/permutation_importance.csv",
    "results/kmeans_sensitivity.csv",
    "results/dbscan_parameter_search.csv",
    "results/clustering_metrics.csv",
    "results/feature_schema.json",
    "results/experiment_manifest.json",
]

REQUIRED_MODELS = [
    "artifacts/models/Dummy.joblib",
    "artifacts/models/RandomForest.joblib",
    "artifacts/models/XGBoost.joblib",
    "artifacts/models/SVM.joblib",
    "artifacts/models/MLP.joblib",
    "artifacts/models/best_model.joblib",
    "artifacts/models/clustering_preprocessor.joblib",
    "artifacts/models/clustering_pca.joblib",
    "artifacts/models/kmeans_k2.joblib",
    "artifacts/models/dbscan.joblib",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and consolidate benchmark outputs.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    paths = ensure_directories(project_root)
    config = load_yaml(project_root / args.config)
    prepared = prepare_dataset(
        project_root,
        config,
        retrieved_or_validated_at_utc=utc_now(),
    )

    missing = [
        relative for relative in [*REQUIRED_OUTPUTS, *REQUIRED_MODELS]
        if not (project_root / relative).exists()
    ]
    if missing:
        raise FileNotFoundError("Expected artifacts were not created:\n" + "\n".join(missing))

    manifest_path = paths["results"] / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clustering_summary = paths["results"] / "clustering_summary.json"
    if clustering_summary.exists():
        cluster_payload = json.loads(clustering_summary.read_text(encoding="utf-8"))
        manifest.update(cluster_payload)
    manifest["completed_at_utc"] = utc_now()
    write_json(manifest_path, manifest)

    paper_summary = build_paper_summary(
        data_manifest=prepared["data_manifest"],
        feature_schema=prepared["feature_schema"],
        best_model_name=manifest.get("best_model_for_demo"),
    )
    write_json(paths["results"] / "paper_summary.json", paper_summary)

    for figure in paths["figures"].glob("*.png"):
        shutil.copy2(figure, paths["paper_figures"] / figure.name)

    print("Result validation completed successfully.")


if __name__ == "__main__":
    main()
