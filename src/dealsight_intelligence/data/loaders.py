"""Loader for the Amazon Reviews 2023 metadata dataset.

Fetches a single product category, filters out items outside the price
window, and returns clean `Item` objects ready for downstream pipelines.
"""

from __future__ import annotations

from dealsight_intelligence.data.items import Item


class ItemLoader:
    """Pulls Amazon Reviews metadata for one category.

    The Hugging Face `datasets` library is large, so it is imported inside
    `load()` rather than at the top of the module.
    """

    MIN_PRICE = 0.5
    MAX_PRICE = 999.49

    def __init__(self, name: str = "Appliances") -> None:
        self.name = name

    def load(self, limit: int | None = None) -> list[Item]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install ML dependencies with: python -m pip install -e '.[ml]'") from exc

        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{self.name}",
            split="full",
            trust_remote_code=True,
        )
        items: list[Item] = []
        for row in dataset:
            price = parse_price(row.get("price"))
            if price is None or not (self.MIN_PRICE <= price <= self.MAX_PRICE):
                continue
            item = Item(
                title=row.get("title") or "",
                description=row.get("description") or "",
                features=row.get("features") or [],
                details=row.get("details") or {},
                price=price,
                category=self.name,
            )
            if item.is_valid():
                items.append(item)
            if limit and len(items) >= limit:
                break
        return items


def parse_price(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("$", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None
