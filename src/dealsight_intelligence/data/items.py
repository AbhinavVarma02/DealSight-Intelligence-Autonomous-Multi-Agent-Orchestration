"""The `Item` dataclass: one priced product as it flows through the pipeline.

`Item` exposes a `text` view (cleaned product description) and a `prompt`
view (the question/answer string used during fine-tuning and evaluation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Self

MIN_CHARS = 300
CEILING_CHARS = 160 * 7
PREFIX = "Price is $"
QUESTION = "How much does this cost to the nearest dollar?"


@dataclass
class Item:
    title: str
    price: float
    description: str = ""
    features: list[str] | None = None
    details: dict[str, str] | None = None
    category: str = "Unknown"
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    saved_prompt: Optional[str] = None
    id: Optional[int] = None

    @property
    def text(self) -> str:
        if self.summary:
            return clean_text(self.summary)
        if self.full:
            return clean_text(self.full)[:CEILING_CHARS].rsplit(" ", 1)[0]
        features = self.features or []
        details = self.details or {}
        pieces = [self.title, self.description, " ".join(features), " ".join(details.values())]
        cleaned = clean_text(" ".join(piece for piece in pieces if piece))
        return cleaned[:CEILING_CHARS].rsplit(" ", 1)[0]

    @property
    def prompt(self) -> str:
        if self.saved_prompt:
            return self.saved_prompt
        return f"{QUESTION}\n\n{self.text}\n\n{PREFIX}{round(self.price)}.00"

    @property
    def test_prompt(self) -> str:
        return self.prompt.split(PREFIX)[0] + PREFIX

    def is_valid(self) -> bool:
        return self.price > 0 and len(self.text) >= MIN_CHARS

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        from dealsight_intelligence.data.datasets import load_structured_items

        return load_structured_items(dataset_name)

    @classmethod
    def from_mapping(cls, row) -> Self:
        return cls(
            title=row.get("title") or "",
            description=row.get("description") or "",
            features=row.get("features") or [],
            details=row.get("details") or {},
            price=float(row.get("price") or 0),
            category=row.get("category") or "Unknown",
            full=row.get("full"),
            weight=row.get("weight"),
            summary=row.get("summary"),
            saved_prompt=row.get("prompt"),
            id=row.get("id"),
        )


def clean_text(value: str) -> str:
    value = re.sub(r"[\r\n\t]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\b[A-Z0-9]{8,}\b", " ", value)
    return value.strip()
