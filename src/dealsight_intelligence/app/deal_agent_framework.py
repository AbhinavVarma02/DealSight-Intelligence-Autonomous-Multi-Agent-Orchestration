"""Top-level orchestration for DealSight Intelligence.

`DealAgentFramework` ties everything together: it loads `.env`, sets up
logging, opens the vector store if it exists, restores deal memory from
disk, and runs one planning cycle when called. The Gradio app and the
`--once` CLI mode both go through this class.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

from dealsight_intelligence import config
from dealsight_intelligence.agents.deals import Opportunity
from dealsight_intelligence.agents.planning_agent import PlanningAgent
from dealsight_intelligence.agents.scanner_agent import clean_product_description, is_specific_product_deal

BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"


def init_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if any(getattr(handler, "_dealsight_intelligence", False) for handler in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler._dealsight_intelligence = True
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %z",
        )
    )
    root.addHandler(handler)


class DealAgentFramework:
    def __init__(self, memory_file: Path | None = None, planner: PlanningAgent | None = None) -> None:
        load_dotenv(override=True)
        config.ensure_artifact_dirs()
        init_logging()
        self.memory_file = memory_file or config.MEMORY_FILE
        self.memory = self.read_memory()
        self.collection = self._load_collection()
        self.planner = planner

    def _load_collection(self):
        if not config.PRODUCTS_VECTORSTORE.exists():
            return None
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(config.PRODUCTS_VECTORSTORE))
            return client.get_or_create_collection("products")
        except Exception as exc:
            self.log(f"vector store unavailable: {exc}")
            return None

    def init_agents_as_needed(self) -> None:
        if not self.planner:
            self.log("initializing agents")
            self.planner = PlanningAgent(self.collection)
            self.log("agents ready")

    def read_memory(self) -> list[Opportunity]:
        if not self.memory_file.exists():
            return []
        try:
            data = json.loads(self.memory_file.read_text(encoding="utf-8"))
            memory = [Opportunity.model_validate(item) for item in data]
            return self._clean_memory(memory)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            self.log(f"could not read memory; starting fresh: {exc}")
            return []

    def _clean_memory(self, memory: list[Opportunity]) -> list[Opportunity]:
        cleaned: list[Opportunity] = []
        changed = False
        for opportunity in memory:
            description = clean_product_description(opportunity.deal.product_description)
            if not description or not is_specific_product_deal(description):
                changed = True
                continue
            if description != opportunity.deal.product_description:
                opportunity.deal.product_description = description
                changed = True
            cleaned.append(opportunity)
        if changed:
            self.memory = cleaned
            self.write_memory()
        return cleaned

    def write_memory(self) -> None:
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        data = [opportunity.model_dump() for opportunity in self.memory]
        self.memory_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def run(self) -> list[Opportunity]:
        # One planning cycle. New opportunities get appended to memory and
        # persisted so the next run can skip URLs we've already alerted on.
        self.init_agents_as_needed()
        result = self.planner.plan(memory=self.memory)
        if result and result.deal.url not in {item.deal.url for item in self.memory}:
            self.memory.append(result)
            self.write_memory()
        return self.memory

    def log(self, message: str) -> None:
        logging.info("%s%s[Agent Framework] %s%s", BG_BLUE, WHITE, message, RESET)

    @classmethod
    def get_plot_data(cls, max_datapoints: int = 1000):
        # Reduces the vector store embeddings to 3D with t-SNE for the
        # optional product-space visualisation in the UI.
        if not config.PRODUCTS_VECTORSTORE.exists():
            return [], [], []
        try:
            import chromadb
            import numpy as np
            from sklearn.manifold import TSNE

            client = chromadb.PersistentClient(path=str(config.PRODUCTS_VECTORSTORE))
            collection = client.get_or_create_collection("products")
            result = collection.get(include=["embeddings", "documents", "metadatas"], limit=max_datapoints)
            vectors = np.array(result["embeddings"])
            if len(vectors) < 3:
                return result.get("documents", []), vectors, []
            reduced = TSNE(n_components=3, random_state=42, perplexity=min(30, len(vectors) - 1)).fit_transform(vectors)
            categories = [metadata.get("category", "unknown") for metadata in result.get("metadatas", [])]
            return result.get("documents", []), reduced, categories
        except Exception:
            return [], [], []
