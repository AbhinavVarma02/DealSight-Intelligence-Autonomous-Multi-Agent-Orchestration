"""Frontier pricing agent.

Prices a product by asking GPT-4o-mini for an estimate, with retrieval-
augmented context drawn from the Chroma vector store of similar priced
products. Falls back to a local heuristic when OpenAI is unavailable or
when dry-run mode is on.
"""

from __future__ import annotations

import os
import re

from dealsight_intelligence import config
from dealsight_intelligence.agents.agent import Agent


class FrontierAgent(Agent):
    MODEL = "gpt-4o-mini"
    name = "Frontier Agent"
    color = Agent.MAGENTA


    def __init__(self, collection=None, use_openai: bool | None = None) -> None:
        self.collection = collection
        dry_run = config.bool_env("DEALSIGHT_INTELLIGENCE_DRY_RUN", True)
        self.use_openai = bool(os.getenv("OPENAI_API_KEY")) and not dry_run
        if use_openai is not None:
            self.use_openai = use_openai

    def price(self, description: str) -> float:
        # Try the live model first; on any failure fall back to the local
        # keyword-based estimator so the pipeline keeps moving.
        if self.use_openai:
            try:
                return self._price_with_openai(description)
            except Exception as exc:
                self.log(f"OpenAI pricing failed; using local fallback: {exc}")
        return fallback_price(description)

    def _price_with_openai(self, description: str) -> float:
        from openai import OpenAI

        client = OpenAI()
        context = self._similar_context(description)
        prompt = (
            "Estimate the fair market price of this product. "
            "Reply only with a number.\n\n"
            f"Similar products:\n{context}\n\nProduct:\n{description}"
        )
        result = client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": "You estimate retail product prices."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = result.choices[0].message.content or ""
        return parse_price(content)

    def _similar_context(self, description: str) -> str:
        if not self.collection:
            return "No vector store is available."
        try:
            result = self.collection.query(query_texts=[description], n_results=5)
        except Exception:
            return "No vector store is available."
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        rows = []
        for doc, metadata in zip(docs, metadatas):
            price = metadata.get("price") if isinstance(metadata, dict) else None
            rows.append(f"- {doc} Price: {price}")
        return "\n".join(rows) or "No similar products found."


def parse_price(text: str) -> float:
    match = re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?)", text)
    if not match:
        return 0.0
    return max(0.0, float(match.group(1).replace(",", "")))


def fallback_price(description: str) -> float:
    # Last-resort offline pricer. Keyword matches push the base estimate up;
    # "pro" and high-end display tags add a multiplier.
    text = description.lower()
    base = 75.0
    signals = {
        "laptop": 650,
        "computer": 550,
        "phone": 450,
        "tablet": 350,
        "tv": 500,
        "camera": 300,
        "speaker": 140,
        "headphone": 120,
        "vacuum": 220,
        "tool": 110,
        "smart": 95,
        "appliance": 260,
        "car": 300,
        "automotive": 180,
    }
    for word, value in signals.items():
        if word in text:
            base = max(base, value)
    if "pro" in text:
        base *= 1.2
    if "4k" in text or "oled" in text:
        base *= 1.35
    return round(base, 2)
