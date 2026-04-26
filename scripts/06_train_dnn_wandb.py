"""CLI for training the local deep neural network pricer.

Pass `--wandb` to log metrics to Weights & Biases; otherwise metrics are
just printed. Use `--limit` to smoke-test on a small subset.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dealsight_intelligence.pricing.train_deep_neural_network import train_deep_neural_network


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the deep neural network pricer with optional W&B tracking.")
    parser.add_argument("--train-path", type=Path, default=None, help="Path to train_lite.pkl or train_full.pkl")
    parser.add_argument("--validation-path", type=Path, default=None, help="Path to validation_lite.pkl or validation_full.pkl")
    parser.add_argument("--model-path", type=Path, default=None, help="Output .pth path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Optional small subset for smoke testing")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "dealsight-intelligence"))
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    output = train_deep_neural_network(
        train_path=args.train_path,
        validation_path=args.validation_path,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        limit=args.limit,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    print(f"Wrote {output}")
