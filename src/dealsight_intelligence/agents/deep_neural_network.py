"""Local deep neural network pricer.

A residual MLP over hashed text features. This module owns both the model
definition (so the saved state dict loads correctly) and the inference
helper that turns a product description into a dollar estimate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

# Targets were trained on log1p(price) and then standardised. We invert
# that transform at inference time using these constants.
Y_STD = 1.0328539609909058
Y_MEAN = 4.434937953948975


class DeepNeuralNetworkInference:
    """Inference-only wrapper around the residual MLP price-prediction network.

    The architecture (10 layers, 4096 hidden units, residual blocks with
    layer norm and dropout) and the input/output normalisation constants
    must match the model used to train the saved weights.
    """

    def __init__(self) -> None:
        self.vectorizer = None
        self.model = None
        self.device = None
        self.torch = None

    def setup(self) -> None:
        try:
            import torch
            import torch.nn as nn
            from sklearn.feature_extraction.text import HashingVectorizer
        except ImportError as exc:
            raise RuntimeError("Install DNN dependencies with: python -m pip install -e '.[dnn]'") from exc

        self.torch = torch
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

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

        self.vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)
        self.model = DeepNeuralNetwork(5000)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logging.info("Neural Network is using %s", self.device)
        self.model.to(self.device)

    def load(self, path: str | Path) -> None:
        if self.model is None or self.device is None or self.torch is None:
            self.setup()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DNN weights not found: {path}")
        state = self.torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def inference(self, text: str) -> float:
        if self.model is None or self.vectorizer is None or self.device is None or self.torch is None:
            raise RuntimeError("DNN is not set up")
        self.model.eval()
        with self.torch.no_grad():
            vector = self.vectorizer.transform([text])
            tensor = self.torch.FloatTensor(vector.toarray()).to(self.device)
            pred = self.model(tensor)[0]
            result = self.torch.exp(pred * Y_STD + Y_MEAN) - 1
            return max(0.0, float(result.item()))
