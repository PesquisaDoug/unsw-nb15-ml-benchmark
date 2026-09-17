from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_binary_class_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (partition, frame) in zip(
        axes,
        [("Training", train_df), ("Testing", test_df)],
    ):
        counts = frame["label"].value_counts().sort_index()
        ax.bar(["Normal", "Attack"], counts.reindex([0, 1], fill_value=0).values)
        ax.set_title(f"{partition} class distribution")
        ax.set_ylabel("Records")
    fig.tight_layout()
    fig.savefig(figures_dir / "01_binary_class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_attack_categories(test_df: pd.DataFrame, figures_dir: Path) -> None:
    attack_counts = (
        test_df.loc[test_df["label"] == 1, "attack_cat"]
        .fillna("Unknown")
        .value_counts()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(attack_counts.index.astype(str), attack_counts.values)
    ax.set_xlabel("Attack records")
    ax.set_title("Attack categories - official testing set")
    fig.tight_layout()
    fig.savefig(figures_dir / "02_test_attack_categories.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    prediction_frame: pd.DataFrame,
    model_names: list[str],
    figures_dir: Path,
) -> None:
    y_test = prediction_frame["y_true"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in model_names:
        scores = prediction_frame[f"{name}_score"].to_numpy()
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_value = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves - official test set")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "03_roc_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curves(
    prediction_frame: pd.DataFrame,
    model_names: list[str],
    figures_dir: Path,
) -> None:
    y_test = prediction_frame["y_true"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in model_names:
        scores = prediction_frame[f"{name}_score"].to_numpy()
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves - official test set")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "04_precision_recall_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_supervised_benchmark(test_metrics: pd.DataFrame, figures_dir: Path) -> None:
    plot_metrics = test_metrics.set_index("model")[["f1", "roc_auc", "pr_auc", "mcc"]]
    ax = plot_metrics.plot(kind="bar", figsize=(9, 5))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Supervised benchmark - official test set")
    ax.legend(loc="lower right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "05_supervised_benchmark.png", dpi=180, bbox_inches="tight")
    plt.close(ax.figure)


def plot_confusion_matrices(
    prediction_frame: pd.DataFrame,
    model_names: list[str],
    figures_dir: Path,
) -> None:
    y_test = prediction_frame["y_true"].to_numpy()
    for name in model_names:
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            prediction_frame[f"{name}_prediction"],
            display_labels=["Normal", "Attack"],
            values_format="d",
            ax=ax,
        )
        ax.set_title(f"Confusion matrix - {name}")
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"06_confusion_{name.lower()}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_attackwise_recall(
    attackwise_metrics: pd.DataFrame,
    best_model_name: str,
    figures_dir: Path,
) -> None:
    demo_attackwise = (
        attackwise_metrics.loc[attackwise_metrics["model"] == best_model_name]
        .sort_values("recall")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(demo_attackwise["attack_cat"], demo_attackwise["recall"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Detection recall")
    ax.set_title(f"Attack-wise detection - {best_model_name}")
    fig.tight_layout()
    fig.savefig(figures_dir / "07_attackwise_recall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_permutation_importance(
    importance_df: pd.DataFrame,
    best_model_name: str,
    figures_dir: Path,
    top_n: int = 15,
) -> None:
    top = importance_df.head(top_n).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_xlabel("Mean decrease in F1 after permutation")
    ax.set_title(f"Top {top_n} permutation importances - {best_model_name}")
    fig.tight_layout()
    fig.savefig(figures_dir / "08_permutation_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pca_true_labels(
    X_cluster_2d: np.ndarray,
    y_cluster: pd.Series,
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, label in [(0, "Normal"), (1, "Attack")]:
        mask = y_cluster.to_numpy() == cls
        ax.scatter(X_cluster_2d[mask, 0], X_cluster_2d[mask, 1], s=10, alpha=0.4, label=label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA projection colored by hidden binary label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "09_pca_true_labels.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_kmeans_sensitivity(kmeans_metrics: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(kmeans_metrics["n_clusters"], kmeans_metrics["silhouette"], marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette score")
    ax.set_title("k-Means sensitivity analysis")
    ax.set_xticks(range(2, 11))
    fig.tight_layout()
    fig.savefig(figures_dir / "10_kmeans_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_kmeans_pca(
    X_cluster_2d: np.ndarray,
    kmeans_labels: np.ndarray,
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X_cluster_2d[:, 0], X_cluster_2d[:, 1], c=kmeans_labels, s=10, alpha=0.45)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("k-Means clustering (k=2)")
    fig.tight_layout()
    fig.savefig(figures_dir / "11_pca_kmeans.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_dbscan_k_distance(
    kth_distances: np.ndarray,
    selected_eps: float,
    selected_min_samples: int,
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(kth_distances)
    ax.axhline(selected_eps, linestyle="--", label="selected eps")
    ax.set_xlabel("Sorted observations")
    ax.set_ylabel(f"{selected_min_samples}-NN distance")
    ax.set_title("DBSCAN k-distance diagnostic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "12_dbscan_k_distance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_dbscan_pca(
    X_cluster_2d: np.ndarray,
    dbscan_labels: np.ndarray,
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for cluster_label in sorted(np.unique(dbscan_labels)):
        mask = dbscan_labels == cluster_label
        display_name = "Noise" if cluster_label == -1 else f"Cluster {cluster_label}"
        ax.scatter(
            X_cluster_2d[mask, 0],
            X_cluster_2d[mask, 1],
            s=10,
            alpha=0.45,
            label=display_name,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("DBSCAN clustering")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "13_pca_dbscan.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def regenerate_available_figures(project_root: Path) -> None:
    # Placeholder hook for scripts/generate_results.py. The heavy figures are
    # produced by the supervised and clustering runs using real predictions.
    (project_root / "figures").mkdir(parents=True, exist_ok=True)
