from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.evaluation import validate_inference_schema

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "artifacts" / "models"

st.set_page_config(page_title="UNSW-NB15 ML Benchmark", layout="wide")
st.title("UNSW-NB15 - Network Intrusion Detection ML Benchmark")
st.caption(
    "Research demonstrator using processed UNSW-NB15 flow features. "
    "This application does not parse PCAP files or monitor live traffic."
)

required = [
    RESULTS / "metrics.csv",
    RESULTS / "attackwise_detection.csv",
    RESULTS / "clustering_metrics.csv",
    RESULTS / "feature_schema.json",
    RESULTS / "experiment_manifest.json",
    MODELS / "best_model.joblib",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    st.error(
        "Required experiment artifacts were not found. Run the QUICK or FULL "
        "pipeline first.\n\n" + "\n".join(missing)
    )
    st.stop()

metrics = pd.read_csv(RESULTS / "metrics.csv")
attackwise = pd.read_csv(RESULTS / "attackwise_detection.csv")
clustering = pd.read_csv(RESULTS / "clustering_metrics.csv")
schema = json.loads((RESULTS / "feature_schema.json").read_text(encoding="utf-8"))
manifest = json.loads((RESULTS / "experiment_manifest.json").read_text(encoding="utf-8"))
model = joblib.load(MODELS / "best_model.joblib")
feature_names = schema["feature_names"]

tabs = st.tabs(
    [
        "Dataset",
        "Supervised Benchmark",
        "Attack-wise Detection",
        "Feature Analysis",
        "Clustering",
        "Inference",
        "Reproducibility",
    ]
)

with tabs[0]:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training records", manifest["train_rows"])
    c2.metric("Testing records", manifest["test_rows"])
    c3.metric("Input features", manifest["input_feature_count"])
    st.write("Canonical dataset: UNSW-NB15, UNSW Canberra. Binary target: 0=normal, 1=attack.")
    for filename in ["01_binary_class_distribution.png", "02_test_attack_categories.png"]:
        path = FIGURES / filename
        if path.exists():
            st.image(str(path))

with tabs[1]:
    st.subheader("Supervised benchmark")
    st.dataframe(metrics, use_container_width=True)
    for filename in ["05_supervised_benchmark.png", "03_roc_curves.png", "04_precision_recall_curves.png"]:
        path = FIGURES / filename
        if path.exists():
            st.image(str(path))

with tabs[2]:
    st.subheader("Attack-wise binary detection")
    model_options = sorted(attackwise["model"].unique())
    default_model = manifest.get("best_model_for_demo")
    selected_model = st.selectbox(
        "Model",
        model_options,
        index=model_options.index(default_model) if default_model in model_options else 0,
    )
    subset = attackwise.loc[attackwise["model"] == selected_model].sort_values("recall")
    st.dataframe(subset, use_container_width=True)
    st.bar_chart(subset.set_index("attack_cat")["recall"])

with tabs[3]:
    st.subheader("Feature analysis")
    path = RESULTS / "permutation_importance.csv"
    if path.exists():
        importance = pd.read_csv(path)
        st.dataframe(importance.head(20), use_container_width=True)
    image = FIGURES / "08_permutation_importance.png"
    if image.exists():
        st.image(str(image))

with tabs[4]:
    st.subheader("Unsupervised analysis")
    st.dataframe(clustering, use_container_width=True)
    for filename in [
        "09_pca_true_labels.png",
        "10_kmeans_sensitivity.png",
        "11_pca_kmeans.png",
        "12_dbscan_k_distance.png",
        "13_pca_dbscan.png",
    ]:
        path = FIGURES / filename
        if path.exists():
            st.image(str(path))

with tabs[5]:
    st.subheader("Inference on processed flow features")
    st.warning(
        "This application expects processed UNSW-NB15 feature columns. "
        "It does not parse PCAP files or monitor live network traffic."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        missing_features = validate_inference_schema(frame, feature_names)
        if missing_features:
            st.error("Missing required features: " + ", ".join(missing_features))
        else:
            X_upload = frame[feature_names].copy()
            predictions = model.predict(X_upload)
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X_upload)[:, 1]
                score_type = "attack_probability"
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_upload)
                score_type = "attack_decision_score"
            else:
                scores = predictions.astype(float)
                score_type = "attack_score"

            output = frame.copy()
            output["prediction"] = predictions
            output["prediction_label"] = output["prediction"].map({0: "normal", 1: "attack"})
            output[score_type] = scores
            st.success(f"Processed {len(output)} records.")
            st.dataframe(output, use_container_width=True)
            st.download_button(
                "Download predictions",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="unsw_nb15_predictions.csv",
                mime="text/csv",
            )

with tabs[6]:
    st.subheader("Reproducibility")
    st.json(manifest)
