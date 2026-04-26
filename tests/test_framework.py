"""Tests for the planner + framework wiring.

Uses a fake scanner and a fake pricer so the test runs fully offline and
verifies: (1) the planner returns the best deal when discount clears the
threshold, (2) memory persists across framework instances, and (3) stale
or invalid memory entries are pruned on load.
"""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dealsight_intelligence.agents.deals import Deal, DealSelection, Opportunity
from dealsight_intelligence.agents.messaging_agent import MessagingAgent
from dealsight_intelligence.agents.planning_agent import PlanningAgent
from dealsight_intelligence.app.deal_agent_framework import DealAgentFramework


class FakeScanner:
    max_deals = 5

    def scan(self, memory=None):
        if memory and "https://x.test/deal" in memory:
            return None
        return DealSelection(deals=[Deal(product_description="4K OLED TV", price=199, url="https://x.test/deal")])


class FakePricer:
    def price(self, description):
        return 500


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.old_threshold = os.environ.get("dealsight_intelligence_DISCOUNT_THRESHOLD")
        os.environ["dealsight_intelligence_DISCOUNT_THRESHOLD"] = "50"

    def tearDown(self):
        if self.old_threshold is None:
            os.environ.pop("dealsight_intelligence_DISCOUNT_THRESHOLD", None)
        else:
            os.environ["dealsight_intelligence_DISCOUNT_THRESHOLD"] = self.old_threshold

    def test_planner_returns_best_deal_when_threshold_is_met(self):
        planner = PlanningAgent(scanner=FakeScanner(), pricer=FakePricer(), messenger=MessagingAgent(do_push=False))
        result = planner.plan(memory=[])
        self.assertIsNotNone(result)
        self.assertEqual(result.discount, 301)

    def test_framework_persists_memory(self):
        memory_file = ROOT / "artifacts" / "memory" / "test_memory.json"
        if memory_file.exists():
            memory_file.unlink()
        planner = PlanningAgent(scanner=FakeScanner(), pricer=FakePricer(), messenger=MessagingAgent(do_push=False))
        framework = DealAgentFramework(memory_file=memory_file, planner=planner)
        memory = framework.run()
        self.assertEqual(len(memory), 1)
        loaded = DealAgentFramework(memory_file=memory_file, planner=planner)
        self.assertEqual(loaded.memory[0].deal.url, "https://x.test/deal")
        memory_file.unlink()

    def test_framework_prunes_invalid_saved_memory(self):
        memory_file = ROOT / "artifacts" / "memory" / "test_memory_invalid.json"
        if memory_file.exists():
            memory_file.unlink()
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        stale = Opportunity(
            deal=Deal(
                product_description="Samsung Galaxy S26 Lineup at Mint Mobile for $500 off + $15/month",
                price=15,
                url="https://x.test/lineup",
            ),
            estimate=689,
            discount=674,
        )
        good = Opportunity(
            deal=Deal(
                product_description="Lafati 2-Piece Hydraulic Car Ramp for $170. DealNews is reader-supported. Nav.",
                price=170,
                url="https://x.test/ramp",
            ),
            estimate=300,
            discount=130,
        )
        memory_file.write_text(f"[{stale.model_dump_json()}, {good.model_dump_json()}]", encoding="utf-8")
        loaded = DealAgentFramework(memory_file=memory_file, planner=PlanningAgent(scanner=FakeScanner(), pricer=FakePricer(), messenger=MessagingAgent(do_push=False)))
        self.assertEqual(len(loaded.memory), 1)
        self.assertEqual(loaded.memory[0].deal.url, "https://x.test/ramp")
        self.assertNotIn("DealNews is reader-supported", loaded.memory[0].deal.product_description)
        memory_file.unlink()


if __name__ == "__main__":
    unittest.main()
