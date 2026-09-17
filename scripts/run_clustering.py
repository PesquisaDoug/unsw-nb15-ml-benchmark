from __future__ import annotations

import argparse
from pathlib import Path

from src.clustering import run_clustering_analysis
from src.reproducibility import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UNSW-NB15 clustering benchmark.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    config = load_yaml(project_root / args.config)
    summary = run_clustering_analysis(
        project_root=project_root,
        config=config,
        profile=args.profile,
    )
    print(f"Clustering complete. PCA components: {summary['pca_components']}")


if __name__ == "__main__":
    main()
