from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from .data import prepare_dataset
from .preprocessing import build_unsupervised_preprocessor
from .reproducibility import ensure_directories, utc_now, write_json
from .visualization import (
    plot_dbscan_k_distance,
    plot_dbscan_pca,
    plot_kmeans_pca,
    plot_kmeans_sensitivity,
    plot_pca_true_labels,
)


def evaluate_dbscan_candidate(
    X_space: np.ndarray,
    labels: np.ndarray,
    y_external: np.ndarray,
    *,
    silhouette_sample_size: int,
    random_seed: int,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    non_noise = labels != -1
    unique_clusters = set(labels[non_noise])
    n_clusters = len(unique_clusters)
    noise_fraction = float(np.mean(labels == -1))
    non_noise_n = int(non_noise.sum())
    row = {
        "n_clusters": n_clusters,
        "noise_fraction": noise_fraction,
        "non_noise_n": non_noise_n,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "ari": adjusted_rand_score(y_external, labels),
        "nmi": normalized_mutual_info_score(y_external, labels),
    }
    if n_clusters >= 2 and non_noise_n >= 100:
        X_valid = X_space[non_noise]
        labels_valid = labels[non_noise]
        if len(np.unique(labels_valid)) >= 2:
            row["silhouette"] = silhouette_score(
                X_valid,
                labels_valid,
                sample_size=min(silhouette_sample_size, len(labels_valid)),
                random_state=random_seed,
            )
            row["davies_bouldin"] = davies_bouldin_score(X_valid, labels_valid)
            row["calinski_harabasz"] = calinski_harabasz_score(X_valid, labels_valid)
    return row


def run_clustering_analysis(
    *,
    project_root: Path,
    config: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    paths = ensure_directories(project_root)
    profile_config = config["profiles"][profile]
    prepared = prepare_dataset(
        project_root,
        config,
        retrieved_or_validated_at_utc=utc_now(),
    )
    train_df = prepared["train_df"]
    feature_schema = prepared["feature_schema"]
    feature_columns = feature_schema["feature_names"]
    numeric_columns = feature_schema["numeric_features"]
    categorical_columns = feature_schema["categorical_features"]
    target_column = config["dataset"]["target"]

    X_train = train_df[feature_columns].copy()
    y_train = train_df[target_column].astype(int).copy()
    cluster_n = min(profile_config["cluster_sample_size"], len(X_train))
    cluster_idx = X_train.sample(n=cluster_n, random_state=config["random_seed"]).index
    X_cluster = X_train.loc[cluster_idx].copy()
    y_cluster = y_train.loc[cluster_idx].copy()

    unsupervised_preprocessor = build_unsupervised_preprocessor(
        numeric_columns,
        categorical_columns,
    )
    X_cluster_encoded = unsupervised_preprocessor.fit_transform(X_cluster)
    pca_cluster = PCA(n_components=0.95, random_state=config["random_seed"])
    X_cluster_pca = pca_cluster.fit_transform(X_cluster_encoded)
    pca_2d = PCA(n_components=2, random_state=config["random_seed"])
    X_cluster_2d = pca_2d.fit_transform(X_cluster_encoded)

    silhouette_sample_size = config["clustering"]["silhouette_sample_size"]
    kmeans_rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=20, random_state=config["random_seed"])
        labels = km.fit_predict(X_cluster_pca)
        kmeans_rows.append(
            {
                "algorithm": "KMeans",
                "configuration": f"k={k}",
                "n_clusters": k,
                "noise_fraction": 0.0,
                "silhouette": silhouette_score(
                    X_cluster_pca,
                    labels,
                    sample_size=min(silhouette_sample_size, len(labels)),
                    random_state=config["random_seed"],
                ),
                "davies_bouldin": davies_bouldin_score(X_cluster_pca, labels),
                "calinski_harabasz": calinski_harabasz_score(X_cluster_pca, labels),
                "ari": adjusted_rand_score(y_cluster, labels),
                "nmi": normalized_mutual_info_score(y_cluster, labels),
            }
        )
    kmeans_metrics = pd.DataFrame(kmeans_rows)
    kmeans = KMeans(n_clusters=2, n_init=20, random_state=config["random_seed"])
    kmeans_labels = kmeans.fit_predict(X_cluster_pca)

    dbscan_candidates = []
    kth_distances_by_min_samples: dict[int, np.ndarray] = {}
    for min_samples in [5, 10, 20]:
        neighbors = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1)
        neighbors.fit(X_cluster_pca)
        distances, _ = neighbors.kneighbors(X_cluster_pca)
        kth_distances = np.sort(distances[:, -1])
        kth_distances_by_min_samples[min_samples] = kth_distances
        for quantile in [0.80, 0.85, 0.90, 0.95]:
            eps = float(np.quantile(kth_distances, quantile))
            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = db.fit_predict(X_cluster_pca)
            row = evaluate_dbscan_candidate(
                X_cluster_pca,
                labels,
                y_cluster.to_numpy(),
                silhouette_sample_size=silhouette_sample_size,
                random_seed=config["random_seed"],
            )
            row.update(
                {
                    "algorithm": "DBSCAN",
                    "configuration": f"eps={eps:.6f}, min_samples={min_samples}",
                    "eps": eps,
                    "eps_quantile": quantile,
                    "min_samples": min_samples,
                }
            )
            dbscan_candidates.append(row)
    dbscan_metrics = pd.DataFrame(dbscan_candidates)
    valid_dbscan = dbscan_metrics.dropna(subset=["silhouette"]).copy()
    if valid_dbscan.empty:
        warnings.warn(
            "No candidate produced a valid non-noise silhouette. "
            "A fallback configuration will be used only for visualization.",
            stacklevel=2,
        )
        best_dbscan_row = (
            dbscan_metrics.sort_values(
                ["n_clusters", "noise_fraction"],
                ascending=[False, True],
            ).iloc[0]
        )
    else:
        best_dbscan_row = (
            valid_dbscan.sort_values(
                ["silhouette", "noise_fraction"],
                ascending=[False, True],
            ).iloc[0]
        )

    best_dbscan = DBSCAN(
        eps=float(best_dbscan_row["eps"]),
        min_samples=int(best_dbscan_row["min_samples"]),
        n_jobs=-1,
    )
    dbscan_labels = best_dbscan.fit_predict(X_cluster_pca)
    common_columns = [
        "algorithm",
        "configuration",
        "n_clusters",
        "noise_fraction",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "ari",
        "nmi",
    ]
    clustering_metrics = pd.concat(
        [
            kmeans_metrics.loc[kmeans_metrics["configuration"] == "k=2", common_columns],
            best_dbscan_row.to_frame().T[common_columns],
        ],
        ignore_index=True,
    )

    kmeans_metrics.to_csv(paths["results"] / "kmeans_sensitivity.csv", index=False)
    dbscan_metrics.to_csv(paths["results"] / "dbscan_parameter_search.csv", index=False)
    clustering_metrics.to_csv(paths["results"] / "clustering_metrics.csv", index=False)

    joblib.dump(unsupervised_preprocessor, paths["models"] / "clustering_preprocessor.joblib", compress=3)
    joblib.dump(pca_cluster, paths["models"] / "clustering_pca.joblib", compress=3)
    joblib.dump(kmeans, paths["models"] / "kmeans_k2.joblib", compress=3)
    joblib.dump(best_dbscan, paths["models"] / "dbscan.joblib", compress=3)

    selected_min_samples = int(best_dbscan_row["min_samples"])
    plot_pca_true_labels(X_cluster_2d, y_cluster, paths["figures"])
    plot_kmeans_sensitivity(kmeans_metrics, paths["figures"])
    plot_kmeans_pca(X_cluster_2d, kmeans_labels, paths["figures"])
    plot_dbscan_k_distance(
        kth_distances_by_min_samples[selected_min_samples],
        float(best_dbscan_row["eps"]),
        selected_min_samples,
        paths["figures"],
    )
    plot_dbscan_pca(X_cluster_2d, dbscan_labels, paths["figures"])

    selected_dbscan = {
        "eps": float(best_dbscan_row["eps"]),
        "min_samples": int(best_dbscan_row["min_samples"]),
        "n_clusters": int(best_dbscan_row["n_clusters"]),
        "noise_fraction": float(best_dbscan_row["noise_fraction"]),
    }
    clustering_summary = {
        "cluster_sample_size": int(cluster_n),
        "encoded_dimensions": int(X_cluster_encoded.shape[1]),
        "pca_components": int(X_cluster_pca.shape[1]),
        "pca_variance_retained": float(pca_cluster.explained_variance_ratio_.sum()),
        "selected_dbscan": selected_dbscan,
    }
    write_json(paths["results"] / "clustering_summary.json", clustering_summary)

    manifest_path = paths["results"] / "experiment_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["selected_dbscan"] = selected_dbscan
        manifest["cluster_sample_size_actual"] = int(cluster_n)
        manifest["pca_components"] = int(X_cluster_pca.shape[1])
        manifest["pca_variance_retained"] = float(pca_cluster.explained_variance_ratio_.sum())
        write_json(manifest_path, manifest)

    return clustering_summary
