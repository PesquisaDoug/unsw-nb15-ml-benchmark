from __future__ import annotations

import argparse
from pathlib import Path

from src.reproducibility import load_yaml
from src.training import run_supervised_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervised UNSW-NB15 benchmark.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    config = load_yaml(project_root / args.config)
    result = run_supervised_benchmark(
        project_root=project_root,
        config=config,
        model_config_path=project_root / args.models,
        profile=args.profile,
    )
    print(f"Supervised benchmark complete. Best demo model: {result['best_model_name']}")


if __name__ == "__main__":
    main()
