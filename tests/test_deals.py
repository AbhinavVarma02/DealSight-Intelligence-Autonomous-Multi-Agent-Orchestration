"""Tests for HTML cleaning, RSS parsing, and the Deal/Opportunity models."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dealsight_intelligence.agents.deals import Deal, Opportunity, ScrapedDeal, clean_html, rss_items


class DealTests(unittest.TestCase):
    def test_clean_html_removes_tags(self):
        self.assertEqual(clean_html("<div>Hello <b>world</b></div>"), "Hello world")

    def test_rss_items_parses_minimal_feed(self):
        xml = """<rss><channel><item><title>A</title><link>https://x.test/a</link><description>$10 item</description></item></channel></rss>"""
        items = list(rss_items(xml))
        self.assertEqual(items, [{"title": "A", "url": "https://x.test/a", "summary": "$10 item"}])

    def test_opportunity_model_round_trip(self):
        deal = Deal(product_description="Nice 4K TV", price=199.99, url="https://x.test/tv")
        opportunity = Opportunity(deal=deal, estimate=350, discount=150.01)
        self.assertEqual(Opportunity.model_validate(opportunity.model_dump()).deal.url, "https://x.test/tv")

    def test_scraped_deal_describe_has_core_fields(self):
        deal = ScrapedDeal("Title", "Summary", "https://x.test", "Details", "Features")
        text = deal.describe()
        self.assertIn("Title: Title", text)
        self.assertIn("URL: https://x.test", text)


if __name__ == "__main__":
    unittest.main()
