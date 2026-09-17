from __future__ import annotations

import pandas as pd

from src.data import build_feature_schema, validate_dataset_schema
from src.reproducibility import sha256_file


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2],
            "dur": [0.1, 0.2],
            "proto": ["tcp", "udp"],
            "service": ["http", "-"],
            "state": ["FIN", "CON"],
            "attack_cat": ["Normal", "Generic"],
            "label": [0, 1],
        }
    )


def test_schema_validation_and_feature_exclusion() -> None:
    train = sample_frame()
    test = sample_frame()
    validate_dataset_schema(train, test)
    schema = build_feature_schema(train)
    assert "label" not in schema["feature_names"]
    assert "attack_cat" not in schema["feature_names"]
    assert "id" not in schema["feature_names"]
    assert schema["categorical_features"] == ["proto", "service", "state"]
    assert schema["numeric_features"] == ["dur"]


def test_invalid_label_rejected() -> None:
    train = sample_frame()
    test = sample_frame()
    train.loc[0, "label"] = 2
    try:
        validate_dataset_schema(train, test)
    except ValueError as exc:
        assert "unexpected binary labels" in str(exc)
    else:
        raise AssertionError("invalid label should raise ValueError")


def test_sha256_file(tmp_path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("abc", encoding="utf-8")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
