from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

from .data import prepare_dataset
from .evaluation import (
    CV_SCORING,
    attackwise_detection_metrics,
    bootstrap_confidence_intervals,
    classification_metrics,
    positive_scores,
    save_model_and_size,
    timed_inference,
)
from .models import build_models, build_search_spaces, load_model_config
from .reproducibility import (
    build_experiment_manifest,
    build_paper_summary,
    ensure_directories,
    utc_now,
    write_json,
)
from .visualization import (
    plot_attack_categories,
    plot_attackwise_recall,
    plot_binary_class_distribution,
    plot_confusion_matrices,
    plot_permutation_importance,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_supervised_benchmark,
)


def run_supervised_benchmark(
    *,
    project_root: Path,
    config: dict[str, Any],
    model_config_path: Path,
    profile: str,
) -> dict[str, Any]:
    started = utc_now()
    paths = ensure_directories(project_root)
    profile_config = config["profiles"][profile]
    prepared = prepare_dataset(
        project_root,
        config,
        retrieved_or_validated_at_utc=started,
    )
    train_df = prepared["train_df"]
    test_df = prepared["test_df"]
    feature_schema = prepared["feature_schema"]
    data_manifest = prepared["data_manifest"]

    feature_columns = feature_schema["feature_names"]
    numeric_columns = feature_schema["numeric_features"]
    categorical_columns = feature_schema["categorical_features"]
    target_column = config["dataset"]["target"]
    attack_column = config["dataset"]["attack_category"]

    X_train = train_df[feature_columns].copy()
    y_train = train_df[target_column].astype(int).copy()
    X_test = test_df[feature_columns].copy()
    y_test = test_df[target_column].astype(int).copy()
    test_attack_categories = test_df[attack_column].fillna("Normal").astype(str).copy()

    plot_binary_class_distribution(train_df, test_df, paths["figures"])
    plot_attack_categories(test_df, paths["figures"])

    model_config = load_model_config(model_config_path)
    models = build_models(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        random_seed=config["random_seed"],
        model_config=model_config,
    )
    search_spaces = build_search_spaces(model_config)

    tuning_cv = StratifiedKFold(
        n_splits=config["cross_validation"]["tuning_folds"],
        shuffle=True,
        random_state=config["random_seed"],
    )
    best_estimators = {}
    tuning_rows = []

    start = time.perf_counter()
    models["Dummy"].fit(X_train, y_train)
    dummy_fit_time = time.perf_counter() - start
    best_estimators["Dummy"] = models["Dummy"]
    tuning_rows.append(
        {
            "model": "Dummy",
            "best_cv_f1": np.nan,
            "refit_time_seconds": dummy_fit_time,
            "best_params": "{}",
        }
    )

    for name in ["RandomForest", "XGBoost", "SVM", "MLP"]:
        print(f"\n=== Tuning {name} ===")
        search = RandomizedSearchCV(
            estimator=models[name],
            param_distributions=search_spaces[name],
            n_iter=profile_config["tuning_iterations"],
            scoring="f1",
            cv=tuning_cv,
            random_state=config["random_seed"],
            n_jobs=config["resources"]["search_n_jobs"],
            refit=True,
            verbose=1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        best_estimators[name] = search.best_estimator_
        tuning_rows.append(
            {
                "model": name,
                "best_cv_f1": search.best_score_,
                "refit_time_seconds": search.refit_time_,
                "best_params": json.dumps(search.best_params_, default=str),
            }
        )

    tuning_summary = pd.DataFrame(tuning_rows)
    tuning_summary.to_csv(paths["results"] / "tuning_summary.csv", index=False)

    evaluation_cv = StratifiedKFold(
        n_splits=profile_config["evaluation_folds"],
        shuffle=True,
        random_state=config["random_seed"] + 1,
    )
    cv_rows = []
    for name, estimator in best_estimators.items():
        print(f"Cross-validating {name}...")
        result = cross_validate(
            clone(estimator),
            X_train,
            y_train,
            cv=evaluation_cv,
            scoring=CV_SCORING,
            n_jobs=config["resources"]["evaluation_n_jobs"],
            return_train_score=False,
        )
        row = {
            "model": name,
            "fit_time_mean_s": float(np.mean(result["fit_time"])),
            "fit_time_std_s": float(np.std(result["fit_time"], ddof=1)),
        }
        for metric in CV_SCORING:
            values = result[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        cv_rows.append(row)
    cv_metrics = (
        pd.DataFrame(cv_rows)
        .sort_values("f1_mean", ascending=False)
        .reset_index(drop=True)
    )
    cv_metrics.to_csv(paths["results"] / "cv_metrics.csv", index=False)

    prediction_frame = pd.DataFrame(
        {
            "row_index": test_df.index,
            "y_true": y_test.to_numpy(),
            "attack_cat": test_attack_categories.to_numpy(),
        }
    )
    test_rows = []
    for name, estimator in best_estimators.items():
        print(f"Evaluating {name}...")
        final_model = clone(estimator)
        start = time.perf_counter()
        final_model.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        y_pred = final_model.predict(X_test)
        y_score = positive_scores(final_model, X_test)
        row = classification_metrics(y_test, y_pred, y_score)
        row.update(timed_inference(final_model, X_test, repeats=profile_config["inference_repeats"]))
        _, model_size_mb = save_model_and_size(final_model, name, paths["models"])
        row.update(
            {
                "model": name,
                "final_fit_time_seconds": fit_time,
                "model_size_mb": model_size_mb,
            }
        )
        test_rows.append(row)
        prediction_frame[f"{name}_prediction"] = y_pred
        prediction_frame[f"{name}_score"] = y_score
        best_estimators[name] = final_model

    test_metrics = (
        pd.DataFrame(test_rows)
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
    test_metrics.to_csv(paths["results"] / "metrics.csv", index=False)
    prediction_frame.to_csv(paths["results"] / "predictions.csv", index=False)

    best_model_name = (
        test_metrics.loc[test_metrics["model"] != "Dummy"]
        .sort_values("f1", ascending=False)
        .iloc[0]["model"]
    )
    best_model = best_estimators[best_model_name]
    joblib.dump(best_model, paths["models"] / "best_model.joblib", compress=3)

    bootstrap_frames = []
    model_names = list(best_estimators)
    for name in model_names:
        ci = bootstrap_confidence_intervals(
            prediction_frame["y_true"].to_numpy(),
            prediction_frame[f"{name}_prediction"].to_numpy(),
            prediction_frame[f"{name}_score"].to_numpy(),
            iterations=profile_config["bootstrap_iterations"],
            seed=config["random_seed"],
        )
        ci.insert(0, "model", name)
        bootstrap_frames.append(ci)
    bootstrap_ci = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap_ci.to_csv(paths["results"] / "bootstrap_ci.csv", index=False)

    attackwise_metrics = attackwise_detection_metrics(prediction_frame, model_names)
    attackwise_metrics.to_csv(paths["results"] / "attackwise_detection.csv", index=False)

    importance_n = min(profile_config["importance_sample_size"], len(X_test))
    importance_idx = X_test.sample(n=importance_n, random_state=config["random_seed"]).index
    perm = permutation_importance(
        best_model,
        X_test.loc[importance_idx],
        y_test.loc[importance_idx],
        scoring="f1",
        n_repeats=profile_config["importance_repeats"],
        random_state=config["random_seed"],
        n_jobs=-1,
    )
    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(paths["results"] / "permutation_importance.csv", index=False)

    plot_roc_curves(prediction_frame, model_names, paths["figures"])
    plot_precision_recall_curves(prediction_frame, model_names, paths["figures"])
    plot_supervised_benchmark(test_metrics, paths["figures"])
    plot_confusion_matrices(prediction_frame, model_names, paths["figures"])
    plot_attackwise_recall(attackwise_metrics, best_model_name, paths["figures"])
    plot_permutation_importance(importance_df, best_model_name, paths["figures"])

    best_test_row = (
        test_metrics.loc[test_metrics["model"] == best_model_name].iloc[0].to_dict()
    )
    best_metrics = {key: value for key, value in best_test_row.items() if key != "model"}
    manifest = build_experiment_manifest(
        project_root=project_root,
        config=config,
        profile=profile,
        data_manifest=data_manifest,
        feature_schema=feature_schema,
        started_at_utc=started,
        best_model_name=best_model_name,
        best_model_metrics=best_metrics,
    )
    write_json(paths["results"] / "experiment_manifest.json", manifest)
    write_json(
        paths["results"] / "paper_summary.json",
        build_paper_summary(
            data_manifest=data_manifest,
            feature_schema=feature_schema,
            best_model_name=best_model_name,
        ),
    )

    return {
        "best_model_name": best_model_name,
        "manifest": manifest,
        "test_metrics": test_metrics,
    }
