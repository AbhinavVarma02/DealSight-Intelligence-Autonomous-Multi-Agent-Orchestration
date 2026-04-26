"""Planning agent.

The planner is the conductor: it asks the Scanner for new deals, runs each
one through the Ensemble pricer, picks the deal with the largest discount,
and asks the Messenger to alert when the discount clears the threshold.
"""

from __future__ import annotations

from dealsight_intelligence import config
from dealsight_intelligence.agents.agent import Agent
from dealsight_intelligence.agents.deals import Deal, Opportunity
from dealsight_intelligence.agents.ensemble_agent import EnsembleAgent
from dealsight_intelligence.agents.messaging_agent import MessagingAgent
from dealsight_intelligence.agents.scanner_agent import ScannerAgent


class PlanningAgent(Agent):
    name = "Planning Agent"
    color = Agent.GREEN

    def __init__(
        self,
        collection=None,
        scanner: ScannerAgent | None = None,
        pricer: EnsembleAgent | None = None,
        messenger: MessagingAgent | None = None,
    ) -> None:
        self.scanner = scanner or ScannerAgent()
        self.pricer = pricer or EnsembleAgent(collection)
        self.messenger = messenger or MessagingAgent()
        self.deal_threshold = config.float_env("DEALSIGHT_INTELLIGENCE_DISCOUNT_THRESHOLD", 50.0)
        self.log("ready")

    def run(self, deal: Deal) -> Opportunity:
        estimate = self.pricer.price(deal.product_description)
        discount = estimate - deal.price
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: list[Opportunity] | None = None) -> Opportunity | None:
        # One full cycle: scan, price, rank, alert if the best discount is
        # large enough. URLs already in memory are skipped to avoid duplicates.
        memory = memory or []
        seen_urls = [opportunity.deal.url for opportunity in memory]
        selection = self.scanner.scan(memory=seen_urls)
        if not selection or not selection.deals:
            self.log("no selection returned")
            return None
        opportunities = [self.run(deal) for deal in selection.deals[: self.scanner.max_deals]]
        opportunities.sort(key=lambda item: item.discount, reverse=True)
        best = opportunities[0]
        self.log(f"best discount is ${best.discount:.2f}")
        if best.discount > self.deal_threshold:
            self.messenger.alert(best)
            return best
        return None
