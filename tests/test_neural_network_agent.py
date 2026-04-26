"""Tests that the NeuralNetworkAgent degrades gracefully when weights or
torch are missing — `price()` should return None, never crash.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dealsight_intelligence.agents.neural_network_agent import NeuralNetworkAgent


class NeuralNetworkAgentTests(unittest.TestCase):
    def test_missing_model_or_torch_does_not_crash(self):
        agent = NeuralNetworkAgent(model_path=ROOT / "artifacts" / "models" / "missing.pth", enabled=True)
        self.assertIsNone(agent.price("Shure MV7 microphone with USB-C and XLR outputs"))


if __name__ == "__main__":
    unittest.main()
