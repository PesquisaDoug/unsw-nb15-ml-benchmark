from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .reproducibility import sha256_file

OFFICIAL_DATASET_PAGE = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
EXPECTED_TRAIN_ROWS = 175341
EXPECTED_TEST_ROWS = 82332
EXPECTED_REQUIRED_COLUMNS = {"label", "attack_cat"}
EXPECTED_CATEGORICAL_COLUMNS = {"proto", "service", "state"}

MIRROR_FILES = {
    "UNSW_NB15_training-set.csv": {
        "url": (
            "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/"
            "resolve/main/UNSW_NB15_training-set.csv?download=true"
        ),
        "sha256": "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
    },
    "UNSW_NB15_testing-set.csv": {
        "url": (
            "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/"
            "resolve/main/UNSW_NB15_testing-set.csv?download=true"
        ),
        "sha256": "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
    },
}


def download_with_sha256(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    observed = sha256_file(tmp)
    if observed.lower() != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        raise ValueError(
            "Downloaded file failed SHA-256 validation. "
            f"Expected {expected_sha256}, observed {observed}."
        )
    tmp.replace(destination)


def resolve_dataset_files(
    project_root: Path,
    config: dict[str, Any],
) -> tuple[Path, Path, dict[str, str]]:
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset = config["dataset"]
    train_path = raw_dir / dataset["train_file"]
    test_path = raw_dir / dataset["test_file"]
    auto_download = bool(dataset.get("auto_download_mirror", True))
    origins: dict[str, str] = {}
    previous_origins: dict[str, str] = {}
    previous_manifest_path = project_root / "data" / "data_manifest.json"
    if previous_manifest_path.exists():
        previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        for key in ["training_file", "testing_file"]:
            file_info = previous_manifest.get(key, {})
            name = file_info.get("name")
            origin = file_info.get("origin_in_this_run")
            if name and origin:
                previous_origins[name] = origin

    for path in [train_path, test_path]:
        if path.exists():
            origins[path.name] = previous_origins.get(path.name, "existing local file")
            continue
        if not auto_download:
            raise FileNotFoundError(
                f"{path} not found. Download the official UNSW-NB15 predefined "
                "training/testing CSV files and place them in data/raw/."
            )
        mirror = MIRROR_FILES[path.name]
        print(f"Downloading verified mirror: {path.name}")
        download_with_sha256(mirror["url"], path, mirror["sha256"])
        origins[path.name] = "verified public mirror"

    return train_path, test_path, origins


def load_official_partition(
    project_root: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, dict[str, str]]:
    train_path, test_path, origins = resolve_dataset_files(project_root, config)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    validate_dataset_schema(train_df, test_df)
    return train_df, test_df, train_path, test_path, origins


def validate_dataset_schema(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for name, frame in [("train", train_df), ("test", test_df)]:
        missing = EXPECTED_REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{name} set missing required columns: {missing}")
        if not EXPECTED_CATEGORICAL_COLUMNS.issubset(frame.columns):
            raise ValueError(
                f"{name} set does not contain expected categorical fields "
                f"{EXPECTED_CATEGORICAL_COLUMNS}."
            )
        invalid_labels = set(pd.unique(frame["label"].dropna())) - {0, 1}
        if invalid_labels:
            raise ValueError(
                f"{name} set contains unexpected binary labels: {invalid_labels}"
            )

    if list(train_df.columns) != list(test_df.columns):
        raise ValueError("Training and testing schemas differ.")
    if len(train_df) != EXPECTED_TRAIN_ROWS:
        warnings.warn(
            f"Expected {EXPECTED_TRAIN_ROWS:,} official training rows; "
            f"observed {len(train_df):,}.",
            stacklevel=2,
        )
    if len(test_df) != EXPECTED_TEST_ROWS:
        warnings.warn(
            f"Expected {EXPECTED_TEST_ROWS:,} official testing rows; "
            f"observed {len(test_df):,}.",
            stacklevel=2,
        )


def build_feature_schema(
    train_df: pd.DataFrame,
    *,
    target_column: str = "label",
    attack_category_column: str = "attack_cat",
) -> dict[str, Any]:
    non_feature_columns = [target_column, attack_category_column]
    if "id" in train_df.columns:
        non_feature_columns.append("id")

    feature_columns = [c for c in train_df.columns if c not in non_feature_columns]
    categorical_columns = [c for c in ["proto", "service", "state"] if c in feature_columns]
    for column in feature_columns:
        if (
            str(train_df[column].dtype) in {"object", "category"}
            and column not in categorical_columns
        ):
            categorical_columns.append(column)

    numeric_columns = [c for c in feature_columns if c not in categorical_columns]

    assert target_column not in feature_columns
    assert attack_category_column not in feature_columns
    assert "id" not in feature_columns if "id" in train_df.columns else True

    return {
        "feature_names": feature_columns,
        "feature_count": len(feature_columns),
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "excluded_columns": non_feature_columns,
        "target": {
            "column": target_column,
            "0": "normal",
            "1": "attack",
        },
    }


def write_feature_schema(path: Path, feature_schema: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")


def create_data_quality_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results_dir: Path,
) -> pd.DataFrame:
    rows = []
    for partition, frame in [("train", train_df), ("test", test_df)]:
        rows.append(
            {
                "partition": partition,
                "rows": len(frame),
                "columns": frame.shape[1],
                "missing_cells": int(frame.isna().sum().sum()),
                "duplicate_rows": int(frame.duplicated().sum()),
                "attack_rate": float(frame["label"].mean()),
                "attack_categories": int(frame["attack_cat"].nunique(dropna=True)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "data_quality_summary.csv", index=False)
    return summary


def create_schema_summary(train_df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "column": train_df.columns,
            "dtype": train_df.dtypes.astype(str).values,
            "n_unique_train": [train_df[c].nunique(dropna=False) for c in train_df.columns],
            "missing_train": [train_df[c].isna().sum() for c in train_df.columns],
        }
    )
    summary.to_csv(results_dir / "schema_summary.csv", index=False)
    return summary


def create_data_manifest(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path: Path,
    test_path: Path,
    origins: dict[str, str],
    retrieved_or_validated_at_utc: str,
) -> dict[str, Any]:
    return {
        "dataset": "UNSW-NB15",
        "canonical_source": OFFICIAL_DATASET_PAGE,
        "official_train_rows_reported": EXPECTED_TRAIN_ROWS,
        "official_test_rows_reported": EXPECTED_TEST_ROWS,
        "training_file": {
            "name": train_path.name,
            "origin_in_this_run": origins[train_path.name],
            "rows": int(len(train_df)),
            "columns": int(train_df.shape[1]),
            "sha256": sha256_file(train_path),
        },
        "testing_file": {
            "name": test_path.name,
            "origin_in_this_run": origins[test_path.name],
            "rows": int(len(test_df)),
            "columns": int(test_df.shape[1]),
            "sha256": sha256_file(test_path),
        },
        "retrieved_or_validated_at_utc": retrieved_or_validated_at_utc,
        "target": {
            "column": "label",
            "0": "normal",
            "1": "attack",
        },
        "posthoc_metadata": "attack_cat",
        "note": (
            "attack_cat is retained only for post-hoc attack-wise analysis "
            "and is excluded from all supervised/unsupervised model inputs."
        ),
    }


def prepare_dataset(
    project_root: Path,
    config: dict[str, Any],
    *,
    retrieved_or_validated_at_utc: str,
) -> dict[str, Any]:
    train_df, test_df, train_path, test_path, origins = load_official_partition(
        project_root, config
    )
    paths = {
        "results": project_root / "results",
        "data": project_root / "data",
    }
    paths["results"].mkdir(parents=True, exist_ok=True)
    paths["data"].mkdir(parents=True, exist_ok=True)

    feature_schema = build_feature_schema(
        train_df,
        target_column=config["dataset"]["target"],
        attack_category_column=config["dataset"]["attack_category"],
    )
    write_feature_schema(paths["results"] / "feature_schema.json", feature_schema)
    create_data_quality_summary(train_df, test_df, paths["results"])
    create_schema_summary(train_df, paths["results"])

    data_manifest = create_data_manifest(
        train_df=train_df,
        test_df=test_df,
        train_path=train_path,
        test_path=test_path,
        origins=origins,
        retrieved_or_validated_at_utc=retrieved_or_validated_at_utc,
    )
    (paths["data"] / "data_manifest.json").write_text(
        json.dumps(data_manifest, indent=2),
        encoding="utf-8",
    )

    return {
        "train_df": train_df,
        "test_df": test_df,
        "feature_schema": feature_schema,
        "data_manifest": data_manifest,
    }
