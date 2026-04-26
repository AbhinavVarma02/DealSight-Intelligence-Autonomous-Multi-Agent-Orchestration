"""Ensemble pricing agent.

Combines the three pricing signals (frontier LLM, fine-tuned specialist,
local DNN) into a single estimate. Frontier is weighted heaviest because
it has retrieval context; the others act as second opinions.
"""

from __future__ import annotations

from dealsight_intelligence.agents.agent import Agent
from dealsight_intelligence.agents.frontier_agent import FrontierAgent
from dealsight_intelligence.agents.neural_network_agent import NeuralNetworkAgent
from dealsight_intelligence.agents.specialist_agent import SpecialistAgent


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.GREEN

    def __init__(self, collection=None) -> None:
        self.frontier = FrontierAgent(collection)
        self.specialist = SpecialistAgent()
        self.neural_network = NeuralNetworkAgent()

    def price(self, description: str) -> float:
        # Pull all three estimates. If specialist or DNN are unavailable we
        # fall back to the frontier estimate so weights still sum to 1.
        frontier = self.frontier.price(description)
        specialist = self.specialist.price(description)
        neural_network = self.neural_network.price(description)
        if specialist is None:
            self.log("Specialist unavailable; substituting Frontier estimate")
            specialist = frontier
        if neural_network is None:
            self.log("Neural network unavailable; substituting Frontier estimate")
            neural_network = frontier
        # Frontier dominates because it has RAG context; the other two are
        # second-opinion smoothing terms.
        combined = frontier * 0.8 + specialist * 0.1 + neural_network * 0.1
        self.log(f"Ensemble complete - returning ${combined:.2f}")
        return round(max(0.0, combined), 2)
