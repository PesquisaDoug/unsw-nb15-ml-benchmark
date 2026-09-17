from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import build_scaled_preprocessor, build_tree_preprocessor


def test_tree_pipeline_uses_passthrough_and_unknown_categories() -> None:
    frame = pd.DataFrame({"dur": [1.0, 2.0], "proto": ["tcp", "udp"]})
    pre = build_tree_preprocessor(["dur"], ["proto"])
    pre.fit(frame)
    assert pre.transformers[0][1] == "passthrough"
    encoder = pre.named_transformers_["cat"]
    assert isinstance(encoder, OneHotEncoder)
    assert encoder.handle_unknown == "ignore"
    transformed = pre.transform(pd.DataFrame({"dur": [3.0], "proto": ["icmp"]}))
    assert transformed.shape[0] == 1


def test_scaled_pipeline_uses_standard_scaler_without_nan() -> None:
    frame = pd.DataFrame({"dur": [1.0, 2.0], "proto": ["tcp", "udp"]})
    pre = build_scaled_preprocessor(["dur"], ["proto"])
    transformed = pre.fit_transform(frame)
    assert isinstance(pre.named_transformers_["num"], StandardScaler)
    arr = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert not np.isnan(arr).any()
