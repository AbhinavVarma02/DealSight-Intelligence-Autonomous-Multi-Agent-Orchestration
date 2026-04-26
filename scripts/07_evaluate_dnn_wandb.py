"""CLI for evaluating a trained deep neural network pricer.

Loads the saved weights, runs inference on the test split, and reports
MAE / RMSLE / hit rate. Pass `--wandb` to also log to W&B.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dealsight_intelligence.pricing.train_deep_neural_network import evaluate_deep_neural_network


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the saved DNN with optional W&B tracking.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Path to test_lite.pkl or test_full.pkl")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to deep_neural_network.pth")
    parser.add_argument("--limit", type=int, default=None, help="Optional small subset for smoke testing")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "dealsight-intelligence"))
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    metrics = evaluate_deep_neural_network(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        limit=args.limit,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    print(metrics)
