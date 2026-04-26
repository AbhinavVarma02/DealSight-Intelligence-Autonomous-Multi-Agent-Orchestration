"""Tests that the heuristic scanner picks real product prices and rejects
sale-event language (save $X, starts at $X, up to N% off, lineup pages,
DealNews boilerplate, etc.).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dealsight_intelligence.agents.deals import ScrapedDeal
from dealsight_intelligence.agents.scanner_agent import (
    ScannerAgent,
    clean_product_description,
    extract_actual_price,
    is_specific_product_deal,
)


class ScannerTests(unittest.TestCase):
    def test_extract_actual_price_ignores_discount_language(self):
        self.assertEqual(extract_actual_price("Save $50 on this laptop, now $499.99"), 499.99)

    def test_extract_actual_price_rejects_starts_at_language(self):
        self.assertIsNone(extract_actual_price("Open-box laptops start at just $116, appliances up to 60% off"))

    def test_specific_product_filter_rejects_outlet_events(self):
        self.assertFalse(is_specific_product_deal("Best Buy Outlet Event: up to 60% off, laptops start at $116"))

    def test_specific_product_filter_rejects_top_deals_sale_pages(self):
        self.assertFalse(is_specific_product_deal("Best Buy Upgrade Sale: Top 100 Deals + free shipping"))

    def test_extract_actual_price_ignores_free_shipping_thresholds(self):
        self.assertEqual(
            extract_actual_price('Free shipping over $35. Pioneer 50" 4K UHD Smart TV for $159.99'),
            159.99,
        )

    def test_extract_actual_price_ignores_monthly_plan_prices(self):
        self.assertEqual(
            extract_actual_price("Phone for $500 off plus Unlimited for $15/month. The phone is $400."),
            400.0,
        )

    def test_specific_product_filter_rejects_lineup_pages(self):
        self.assertFalse(is_specific_product_deal("Samsung Galaxy S26 Lineup at Mint Mobile for $500 off"))

    def test_product_description_removes_dealnews_boilerplate(self):
        description = (
            "Lafati 2-Piece Hydraulic Car Ramp for $170. "
            "DealNews is reader-supported. We may earn commissions. "
            "Where every day is Black Friday! Sign In Categories Clothing Computers"
        )
        self.assertEqual(clean_product_description(description), "Lafati 2-Piece Hydraulic Car Ramp for $170.")

    def test_heuristic_scan_selects_priced_deals(self):
        scraped = [
            ScrapedDeal(
                title="Noise-canceling headphones",
                summary="Bluetooth headphones with long battery life for $89.99",
                url="https://x.test/headphones",
                details="Includes carrying case and USB-C charging.",
            ),
            ScrapedDeal(
                title="Coupon only",
                summary="Save $20 on selected accessories",
                url="https://x.test/coupon",
            ),
            ScrapedDeal(
                title="Best Buy Outlet Event",
                summary="Open-box laptops start at just $116 and appliances are up to 60% off",
                url="https://x.test/outlet",
            ),
        ]
        scanner = ScannerAgent(fetcher=lambda: scraped, use_openai=False)
        selection = scanner.scan(memory=[])
        self.assertIsNotNone(selection)
        self.assertEqual(len(selection.deals), 1)
        self.assertEqual(selection.deals[0].price, 89.99)


if __name__ == "__main__":
    unittest.main()
