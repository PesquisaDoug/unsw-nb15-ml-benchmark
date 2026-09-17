from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.evaluation import validate_inference_schema
from src.preprocessing import build_tree_preprocessor


def test_small_model_predicts_and_schema_order_is_preserved() -> None:
    feature_names = ["dur", "proto"]
    frame = pd.DataFrame({"dur": [1.0, 2.0], "proto": ["tcp", "udp"], "extra": [9, 9]})
    y = [0, 1]
    model = Pipeline(
        [
            ("preprocess", build_tree_preprocessor(["dur"], ["proto"])),
            ("model", DummyClassifier(strategy="prior")),
        ]
    ).fit(frame[feature_names], y)
    pred = model.predict(frame[feature_names])
    assert set(pred).issubset({0, 1})
    assert validate_inference_schema(frame, feature_names) == []
    assert validate_inference_schema(frame.drop(columns=["dur"]), feature_names) == ["dur"]
