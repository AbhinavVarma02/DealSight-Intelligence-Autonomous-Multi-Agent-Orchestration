"""Neural network pricing agent.

Wraps the local PyTorch deep neural network and exposes it as just another
agent that returns a price (or `None` when the weights or torch are not
installed).
"""

from __future__ import annotations

from pathlib import Path

from dealsight_intelligence import config
from dealsight_intelligence.agents.agent import Agent
from dealsight_intelligence.agents.deep_neural_network import DeepNeuralNetworkInference


class NeuralNetworkAgent(Agent):
    name = "Neural Network Agent"
    color = Agent.MAGENTA

    def __init__(self, model_path: Path | None = None, enabled: bool | None = None) -> None:
        self.model_path = model_path or config.DEEP_NEURAL_NETWORK_MODEL
        self.enabled = config.bool_env("DEALSIGHT_INTELLIGENCE_ENABLE_DNN", True)
        if enabled is not None:
            self.enabled = enabled
        self.neural_network = None
        self.available = False
        if self.enabled:
            self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            self.log("DNN weights not found; skipping neural network pricing")
            return
        try:
            self.neural_network = DeepNeuralNetworkInference()
            self.neural_network.setup()
            self.neural_network.load(self.model_path)
            self.available = True
            self.log("DNN weights loaded")
        except Exception as exc:
            self.neural_network = None
            self.available = False
            self.log(f"DNN unavailable: {exc}")

    def price(self, description: str) -> float | None:
        if not self.available or self.neural_network is None:
            return None
        try:
            result = self.neural_network.inference(description)
            self.log(f"predicted ${result:.2f}")
            return result
        except Exception as exc:
            self.log(f"DNN prediction failed: {exc}")
            return None
