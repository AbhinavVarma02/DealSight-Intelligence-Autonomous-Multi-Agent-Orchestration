"""Models and scrapers for deal feeds.

This module owns three things:
- `ScrapedDeal`: the raw shape of a deal pulled from an RSS feed
- `Deal` / `DealSelection` / `Opportunity`: validated pydantic models the
  pipeline passes between agents
- Helpers to fetch RSS, strip HTML, and pull readable content from a deal page
"""

from __future__ import annotations

import html
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable

from pydantic import BaseModel, Field, field_validator

DEALNEWS_FEEDS = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/c238/Automotive/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
    "https://www.dealnews.com/c196/Home-Garden/?rss=1",
]

USER_AGENT = "dealsight-intelligence/0.1"


def clean_html(value: str) -> str:
    # Unescape entities, drop script/style blocks, then strip remaining tags.
    text = html.unescape(value or "")
    text = re.sub(r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>", " ", text, flags=re.I)
    text = re.sub(r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_text(url: str, timeout: int = 20) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


def rss_items(xml_text: str) -> Iterable[dict[str, str]]:
    root = ET.fromstring(xml_text)
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        summary = item.findtext("description") or item.findtext("summary") or ""
        yield {"title": clean_html(title), "url": link.strip(), "summary": clean_html(summary)}


def extract_page_content(html_text: str) -> tuple[str, str]:
    # Pull the main article body out of a deal page and split it into
    # (details, features) when a "Features" heading is present.
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        content = (
            soup.find("div", class_="content-section")
            or soup.find("article")
            or soup.find("main")
            or soup.body
            or soup
        )
        cleaned = clean_html(str(content))
    except Exception:
        cleaned = clean_html(html_text)
    if "Features" in cleaned:
        details, features = cleaned.split("Features", 1)
        return details.strip(), features.strip()
    return cleaned[:2500].strip(), ""


@dataclass
class ScrapedDeal:
    title: str
    summary: str
    url: str
    details: str = ""
    features: str = ""

    @classmethod
    def from_feed_item(cls, item: dict[str, str], fetch_pages: bool = True) -> "ScrapedDeal":
        details = item.get("summary", "")
        features = ""
        if fetch_pages and item.get("url"):
            try:
                page = fetch_text(item["url"])
                details, features = extract_page_content(page)
            except (OSError, TimeoutError, urllib.error.URLError, ValueError):
                details = item.get("summary", "")
        return cls(
            title=item.get("title", "").strip(),
            summary=item.get("summary", "").strip(),
            url=item.get("url", "").strip(),
            details=details.strip(),
            features=features.strip(),
        )

    @classmethod
    def fetch(
        cls,
        feeds: Iterable[str] | None = None,
        per_feed: int = 10,
        sleep_seconds: float = 0.2,
        fetch_pages: bool = True,
    ) -> list["ScrapedDeal"]:
        deals: list[ScrapedDeal] = []
        for feed_url in feeds or DEALNEWS_FEEDS:
            try:
                xml_text = fetch_text(feed_url)
                entries = list(rss_items(xml_text))[:per_feed]
            except (ET.ParseError, OSError, TimeoutError, urllib.error.URLError, ValueError):
                continue
            for item in entries:
                if item.get("url"):
                    deals.append(cls.from_feed_item(item, fetch_pages=fetch_pages))
                    if sleep_seconds:
                        time.sleep(sleep_seconds)
        return deals

    def describe(self) -> str:
        parts = [
            f"Title: {self.title}",
            f"Summary: {self.summary}",
            f"Details: {self.details}",
            f"Features: {self.features}",
            f"URL: {self.url}",
        ]
        return "\n".join(part for part in parts if not part.endswith(": "))


# Validated models that downstream agents rely on. Pydantic guarantees
# that bad data never reaches the planner or messenger.
class Deal(BaseModel):
    product_description: str = Field(min_length=3)
    price: float = Field(gt=0)
    url: str = Field(min_length=3)

    @field_validator("product_description")
    @classmethod
    def collapse_description(cls, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()


class DealSelection(BaseModel):
    deals: list[Deal]


class Opportunity(BaseModel):
    deal: Deal
    estimate: float = Field(ge=0)
    discount: float
