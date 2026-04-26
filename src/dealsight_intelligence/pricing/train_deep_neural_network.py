"""Training and evaluation loops for the local deep neural network pricer.

Both functions can optionally log metrics to Weights & Biases when run
with `--wandb`; otherwise they print metrics locally.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

from dealsight_intelligence import config
from dealsight_intelligence.data.datasets import validate_structured_items


def train_deep_neural_network(
    train_path: Path | None = None,
    validation_path: Path | None = None,
    model_path: Path | None = None,
    epochs: int = 5,
    batch_size: int = 64,
    limit: int | None = None,
    use_wandb: bool = False,
    wandb_project: str = "dealsight-intelligence",
    wandb_run_name: str | None = None,
) -> Path:
    """Train the deep neural network pricer and (optionally) log to W&B.

    You only need to run this if you want to retrain from scratch. When
    you already have a saved `deep_neural_network.pth` artifact the
    runtime app will load it directly — no training required.
    """

    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.feature_extraction.text import HashingVectorizer
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("Install DNN dependencies with: python -m pip install -e '.[dnn]'") from exc

    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_module

            wandb = wandb_module
        except ImportError as exc:
            raise RuntimeError("Install tracking dependencies with: python -m pip install -e '.[tracking]'") from exc

    prefix = config.dataset_prefix()
    train_path = train_path or config.path_env("DEALSIGHT_INTELLIGENCE_STRUCTURED_TRAIN_PATH", config.DATASETS_DIR / f"train_{prefix}.pkl")
    validation_path = validation_path or config.path_env(
        "DEALSIGHT_INTELLIGENCE_STRUCTURED_VALIDATION_PATH",
        config.DATASETS_DIR / f"validation_{prefix}.pkl",
    )
    model_path = model_path or config.DEEP_NEURAL_NETWORK_MODEL

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {validation_path}")

    train_items = pickle.loads(train_path.read_bytes())
    validation_items = pickle.loads(validation_path.read_bytes())
    validate_structured_items(train_items, train_path)
    validate_structured_items(validation_items, validation_path)
    if limit:
        train_items = train_items[:limit]
        validation_items = validation_items[: max(1, min(len(validation_items), limit // 5))]

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)
    x_train_np = vectorizer.fit_transform([item.text for item in train_items])
    x_validation_np = vectorizer.transform([item.text for item in validation_items])

    x_train = torch.FloatTensor(x_train_np.toarray())
    x_validation = torch.FloatTensor(x_validation_np.toarray())
    y_train = torch.FloatTensor([float(item.price) for item in train_items]).unsqueeze(1)
    y_validation = torch.FloatTensor([float(item.price) for item in validation_items]).unsqueeze(1)

    y_train_log = torch.log(y_train + 1)
    y_validation_log = torch.log(y_validation + 1)
    y_mean = y_train_log.mean()
    y_std = y_train_log.std()
    y_train_norm = (y_train_log - y_mean) / y_std
    y_validation_norm = (y_validation_log - y_mean) / y_std

    model = _build_model(torch, nn, input_size=x_train.shape[1])
    device = _device(torch)
    model.to(device)

    loss_function = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=0)
    train_loader = DataLoader(TensorDataset(x_train, y_train_norm), batch_size=batch_size, shuffle=True)

    run = None
    if wandb:
        run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "train_items": len(train_items),
                "validation_items": len(validation_items),
                "n_features": 5000,
                "hidden_size": 4096,
                "layers": 10,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
            },
        )

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            validation_outputs = model(x_validation.to(device))
            validation_loss = loss_function(validation_outputs, y_validation_norm.to(device))
            validation_predictions = torch.exp(validation_outputs * y_std.to(device) + y_mean.to(device)) - 1
            validation_mae = torch.abs(validation_predictions - y_validation.to(device)).mean()
            rmsle = _rmsle(validation_predictions.cpu(), y_validation)

        metrics = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / max(1, len(train_losses)),
            "validation_loss": float(validation_loss.item()),
            "validation_mae": float(validation_mae.item()),
            "validation_rmsle": rmsle,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        if wandb:
            wandb.log(metrics)
        print(metrics)
        scheduler.step()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    if wandb and run:
        wandb.save(str(model_path))
        wandb.finish()
    return model_path


def evaluate_deep_neural_network(
    dataset_path: Path | None = None,
    model_path: Path | None = None,
    limit: int | None = None,
    use_wandb: bool = False,
    wandb_project: str = "dealsight-intelligence",
    wandb_run_name: str | None = None,
) -> dict[str, float]:
    """Evaluate a saved DNN artifact and optionally log test metrics to W&B."""

    try:
        import torch
        from dealsight_intelligence.agents.deep_neural_network import DeepNeuralNetworkInference
    except ImportError as exc:
        raise RuntimeError("Install DNN dependencies with: python -m pip install -e '.[dnn]'") from exc

    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_module

            wandb = wandb_module
        except ImportError as exc:
            raise RuntimeError("Install tracking dependencies with: python -m pip install -e '.[tracking]'") from exc

    dataset_path = dataset_path or config.path_env(
        "DEALSIGHT_INTELLIGENCE_STRUCTURED_TEST_PATH",
        config.DATASETS_DIR / f"test_{config.dataset_prefix()}.pkl",
    )
    model_path = model_path or config.DEEP_NEURAL_NETWORK_MODEL
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"DNN weights not found: {model_path}")

    items = pickle.loads(dataset_path.read_bytes())
    validate_structured_items(items, dataset_path)
    if limit:
        items = items[:limit]
    runner = DeepNeuralNetworkInference()
    runner.setup()
    runner.load(model_path)

    predictions = []
    actuals = []
    for item in items:
        predictions.append(runner.inference(item.text))
        actuals.append(float(item.price))

    errors = [abs(pred - actual) for pred, actual in zip(predictions, actuals)]
    rmsle = math.sqrt(
        sum((math.log1p(max(0.0, pred)) - math.log1p(max(0.0, actual))) ** 2 for pred, actual in zip(predictions, actuals))
        / max(1, len(items))
    )
    hit_rate = sum(
        1 for error, actual in zip(errors, actuals) if error < 40 or (actual and error / actual < 0.2)
    ) / max(1, len(items))
    metrics = {
        "test_items": float(len(items)),
        "test_mae": sum(errors) / max(1, len(errors)),
        "test_rmsle": rmsle,
        "test_hit_rate": hit_rate,
    }
    if wandb:
        run = wandb.init(project=wandb_project, name=wandb_run_name, config={"model_path": str(model_path)})
        wandb.log(metrics)
        wandb.finish()
    return metrics


def _build_model(torch, nn, input_size: int):
    class ResidualBlock(nn.Module):
        def __init__(self, hidden_size, dropout_prob):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.block(x) + x)

    class DeepNeuralNetwork(nn.Module):
        def __init__(self, input_size, num_layers=10, hidden_size=4096, dropout_prob=0.2):
            super().__init__()
            self.input_layer = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
            )
            self.residual_blocks = nn.ModuleList(
                ResidualBlock(hidden_size, dropout_prob) for _ in range(num_layers - 2)
            )
            self.output_layer = nn.Linear(hidden_size, 1)

        def forward(self, x):
            x = self.input_layer(x)
            for block in self.residual_blocks:
                x = block(x)
            return self.output_layer(x)

    return DeepNeuralNetwork(input_size)


def _device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _rmsle(predictions, actuals) -> float:
    total = 0.0
    for pred, actual in zip(predictions, actuals):
        total += (math.log1p(max(0.0, float(pred))) - math.log1p(max(0.0, float(actual)))) ** 2
    return math.sqrt(total / max(1, len(actuals)))
