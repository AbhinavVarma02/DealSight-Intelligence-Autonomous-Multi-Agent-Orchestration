"""Scanner agent.

Pulls fresh deals off DealNews RSS feeds and turns them into validated
`Deal` objects. Two strategies are supported:

- OpenAI mode: ask GPT-4o-mini to extract up to 5 specific products and
  prices in JSON.
- Heuristic mode: a regex/text pipeline that picks an actual price out of
  the deal text and rejects sale-event language (e.g. "save $50",
  "starts at $99", "up to 60% off").

Heuristic mode is the dry-run default and the safety net when OpenAI fails.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable

from pydantic import ValidationError

from dealsight_intelligence import config
from dealsight_intelligence.agents.agent import Agent
from dealsight_intelligence.agents.deals import Deal, DealSelection, ScrapedDeal

PRICE_RE = re.compile(r"(?<![A-Za-z])\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)")
BAD_PRICE_CONTEXT = re.compile(
    r"\b(off|save|savings|discount|coupon|rebate|credit|starts?\s+at|starting\s+at|from\s+just|as\s+low\s+as|up\s+to|over\s*\$?)\b",
    re.I,
)
MONTHLY_PRICE_AFTER = re.compile(r"^\s*(?:/\s*month|per\s+month|monthly|/mo\b)", re.I)
BAD_DEAL_LANGUAGE = re.compile(
    r"\b(up\s+to\s+\d+%?\s+off|starts?\s+at|starting\s+at|from\s+just|as\s+low\s+as|outlet\s+event|clearance|open-box\s+\w+\s+start|top\s+\d+\s+deals|shop\s+deals\s+on|lineup)\b",
    re.I,
)
BOILERPLATE_MARKERS = [
    "DealNews is reader-supported.",
    "Where every day is Black Friday!",
    "Sign In Categories",
    "All Clothing Deals",
]


class ScannerAgent(Agent):
    MODEL = "gpt-4o-mini"
    name = "Scanner Agent"
    color = Agent.CYAN

    SYSTEM_PROMPT = """You identify and summarize up to 5 specific product deals from a list.
Respond strictly in JSON. Include only individual products with a clear actual price.
Reject sale events, outlet events, category pages, bundles with multiple possible prices, and values described as "$X off", "save $X", "starts at $X", "from $X", "as low as $X", rebates, credits, or discounts.
Return {"deals":[{"product_description":"...", "price":99.99, "url":"..."}]}."""

    USER_PROMPT_PREFIX = """Select up to 5 specific individual products with clear product descriptions and actual prices.
Rephrase each product description to describe the item, not the discount terms.
Reject sale events, outlet events, category pages, clearance roundups, and "starts at" or "up to" prices.

Deals:
"""

    def __init__(
        self,
        fetcher: Callable[[], list[ScrapedDeal]] | None = None,
        use_openai: bool | None = None,
    ) -> None:
        self.fetcher = fetcher
        dry_run = config.bool_env("DEALSIGHT_INTELLIGENCE_DRY_RUN", True)
        self.use_openai = bool(os.getenv("OPENAI_API_KEY")) and not dry_run
        if use_openai is not None:
            self.use_openai = use_openai
        self.max_deals = config.int_env("DEALSIGHT_INTELLIGENCE_MAX_DEALS", 5)
        self.feed_limit = config.int_env("DEALSIGHT_INTELLIGENCE_FEED_LIMIT", 10)
        self.log("ready")

    def fetch_deals(self, memory: list[str] | None = None) -> list[ScrapedDeal]:
        seen = set(memory or [])
        if self.fetcher:
            scraped = self.fetcher()
        else:
            scraped = ScrapedDeal.fetch(per_feed=self.feed_limit)
        return [deal for deal in scraped if deal.url not in seen]

    def make_user_prompt(self, scraped: list[ScrapedDeal]) -> str:
        body = "\n\n".join(deal.describe() for deal in scraped)
        return f"{self.USER_PROMPT_PREFIX}{body}\n\nRespond in JSON only."

    def scan(self, memory: list[str] | None = None) -> DealSelection | None:
        scraped = self.fetch_deals(memory)
        if not scraped:
            self.log("no new scraped deals found")
            return None
        if self.use_openai:
            selection = self._scan_with_openai(scraped)
        else:
            selection = self._scan_heuristically(scraped)
        valid = clean_valid_deals(selection.deals)
        self.log(f"selected {len(valid)} deal(s)")
        return DealSelection(deals=valid) if valid else None

    def _scan_with_openai(self, scraped: list[ScrapedDeal]) -> DealSelection:
        try:
            from openai import OpenAI

            client = OpenAI()
            result = client.beta.chat.completions.parse(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.make_user_prompt(scraped)},
                ],
                response_format=DealSelection,
            )
            parsed = result.choices[0].message.parsed
            if parsed:
                return parsed
        except Exception as exc:
            self.log(f"OpenAI scan failed; falling back to heuristic scan: {exc}")
        return self._scan_heuristically(scraped)

    def _scan_heuristically(self, scraped: list[ScrapedDeal]) -> DealSelection:
        candidates: list[Deal] = []
        for item in scraped:
            price = extract_actual_price(item.describe())
            if not price:
                continue
            description = summarize_product(item)
            if not is_specific_product_deal(description):
                continue
            try:
                candidates.append(Deal(product_description=description, price=price, url=item.url))
            except ValidationError:
                continue
        candidates.sort(key=lambda deal: len(deal.product_description), reverse=True)
        return DealSelection(deals=candidates[: self.max_deals])


def extract_actual_price(text: str) -> float | None:
    # Walk through every dollar amount in the text. Skip any that sit next
    # to discount/save/starts-at language or are tagged as monthly fees.
    for match in PRICE_RE.finditer(text):
        start = max(match.start() - 24, 0)
        end = min(match.end() + 24, len(text))
        context = text[start:end]
        if BAD_PRICE_CONTEXT.search(context):
            continue
        if MONTHLY_PRICE_AFTER.search(text[match.end() : match.end() + 16]):
            continue
        try:
            price = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if price > 0:
            return price
    return None


def summarize_product(item: ScrapedDeal) -> str:
    parts = [item.title, item.summary, item.details, item.features]
    text = " ".join(part for part in parts if part).strip()
    return clean_product_description(text or item.title)


def clean_product_description(description: str) -> str:
    text = description
    for marker in BOILERPLATE_MARKERS:
        index = text.find(marker)
        if index >= 0:
            text = text[:index]
    text = re.sub(r"\s+", " ", text)
    if len(text) > 900:
        text = text[:900].rsplit(" ", 1)[0]
    return text.strip()


def clean_valid_deals(deals: list[Deal]) -> list[Deal]:
    valid: list[Deal] = []
    for deal in deals:
        description = clean_product_description(deal.product_description)
        if not description or deal.price <= 0 or not is_specific_product_deal(description):
            continue
        try:
            valid.append(Deal(product_description=description, price=deal.price, url=deal.url))
        except ValidationError:
            continue
    return valid


def is_specific_product_deal(description: str) -> bool:
    # Reject sale events / category roundups so the pipeline only acts on
    # individual products with a single clear price.
    return not BAD_DEAL_LANGUAGE.search(description)


def selection_from_json(raw: str) -> DealSelection:
    data = json.loads(raw)
    return DealSelection.model_validate(data)
