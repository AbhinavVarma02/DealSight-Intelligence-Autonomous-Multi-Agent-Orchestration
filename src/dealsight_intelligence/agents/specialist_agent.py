"""Specialist pricing agent.

Calls the Modal-hosted QLoRA fine-tuned Llama adapter (`pricer-service` /
`Pricer`) over the network. Returns `None` if Modal is disabled or the
remote call fails, so the ensemble can substitute another estimate.
"""

from __future__ import annotations

from dealsight_intelligence import config
from dealsight_intelligence.agents.agent import Agent


class SpecialistAgent(Agent):
    name = "Specialist Agent"
    color = Agent.BLUE

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = config.bool_env("DEALSIGHT_INTELLIGENCE_ENABLE_MODAL", False)
        if enabled is not None:
            self.enabled = enabled
        self.pricer = None
        if self.enabled:
            self._load_modal_pricer()
        else:
            self.log("disabled; set DEALSIGHT_INTELLIGENCE_ENABLE_MODAL=true to use Modal")

    def _load_modal_pricer(self) -> None:
        try:
            import modal

            try:
                pricer_cls = modal.Cls.from_name("pricer-service", "Pricer")
            except AttributeError:
                pricer_cls = modal.Cls.lookup("pricer-service", "Pricer")
            self.pricer = pricer_cls()
            self.log("connected to Modal pricer")
        except Exception as exc:
            self.pricer = None
            self.log(f"Modal pricer unavailable: {exc}")

    def price(self, description: str) -> float | None:
        if not self.pricer:
            return None
        try:
            return max(0.0, float(self.pricer.price.remote(description)))
        except Exception as exc:
            self.log(f"Modal pricing failed: {exc}")
            return None
