"""Stub entry point for training the ensemble blender.

The ensemble model expects per-item rows that include each pricer's
estimate plus the actual price. Build those rows in a notebook or
custom script first, then call `train_ensemble(rows)` directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dealsight_intelligence.pricing.train_ensemble import train_ensemble


if __name__ == "__main__":
    raise SystemExit(
        "Prepare calibration rows with Specialist, Frontier, NeuralNetwork, Min, Max, Actual, "
        "then call dealsight_intelligence.pricing.train_ensemble.train_ensemble(rows)."
    )
