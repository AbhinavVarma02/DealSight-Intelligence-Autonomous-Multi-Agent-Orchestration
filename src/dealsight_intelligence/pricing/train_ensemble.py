"""Train the linear-regression ensemble that blends the three pricers.

Expects a list of rows containing each pricer's estimate plus min/max
and the actual price, fits a `LinearRegression`, and saves the model to
`artifacts/models/ensemble_model.pkl`.
"""

from __future__ import annotations

from pathlib import Path

from dealsight_intelligence import config


def train_ensemble(rows, model_path: Path | None = None) -> Path:
    model_path = model_path or config.ENSEMBLE_MODEL
    try:
        import joblib
        import pandas as pd
        from sklearn.linear_model import LinearRegression
    except ImportError as exc:
        raise RuntimeError("Install ML dependencies with: python -m pip install -e '.[ml]'") from exc

    data = pd.DataFrame(rows)
    required = ["Specialist", "Frontier", "NeuralNetwork", "Min", "Max", "Actual"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing ensemble training columns: {missing}")
    model = LinearRegression()
    model.fit(data[required[:-1]], data["Actual"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path
