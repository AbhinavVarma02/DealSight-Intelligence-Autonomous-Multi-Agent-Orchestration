"""Dataset curation helpers.

Three entry points:

- `curate_lite`: build a fresh lite dataset from raw Amazon metadata.
- `download_hub_dataset`: pull a structured `items_*` dataset and write
  the train/validation/test splits to disk as pickled `Item` lists.
- `export_prompt_dataset`: same idea for prompt/completion datasets used
  during fine-tuning.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

from dealsight_intelligence import config
from dealsight_intelligence.data.datasets import (
    load_prompt_examples,
    load_structured_items,
    validate_prompt_examples,
    validate_structured_items,
)
from dealsight_intelligence.data.items import Item
from dealsight_intelligence.data.loaders import ItemLoader


def curate_lite(category: str = "Appliances", train_size: int = 20000, test_size: int = 2000) -> tuple[Path, Path]:
    config.ensure_artifact_dirs()
    items = ItemLoader(category).load(limit=train_size + test_size)
    random.seed(42)
    random.shuffle(items)
    train = items[:train_size]
    test = items[train_size : train_size + test_size]
    train_path = config.DATASETS_DIR / "train_lite.pkl"
    test_path = config.DATASETS_DIR / "test_lite.pkl"
    train_path.write_bytes(pickle.dumps(train))
    test_path.write_bytes(pickle.dumps(test))
    return train_path, test_path


def download_hub_dataset(
    dataset_name: str | None = None,
    prefix: str = "lite",
) -> tuple[Path, Path, Path]:
    config.ensure_artifact_dirs()
    source = dataset_name or config.structured_dataset_source()
    train, validation, test = load_structured_items(source)
    validate_structured_items(train, source)
    train_path = config.DATASETS_DIR / f"train_{prefix}.pkl"
    validation_path = config.DATASETS_DIR / f"validation_{prefix}.pkl"
    test_path = config.DATASETS_DIR / f"test_{prefix}.pkl"
    train_path.write_bytes(pickle.dumps(train))
    validation_path.write_bytes(pickle.dumps(validation))
    test_path.write_bytes(pickle.dumps(test))
    return train_path, validation_path, test_path


def export_prompt_dataset(dataset_name: str | None = None, prefix: str = "prompts_lite") -> tuple[Path, Path, Path | None]:
    config.ensure_artifact_dirs()
    source = dataset_name or config.prompt_dataset_source()
    splits = load_prompt_examples(source)
    validate_prompt_examples(splits.get("train", []), source)
    train_path = config.DATASETS_DIR / f"train_{prefix}.pkl"
    validation_path = config.DATASETS_DIR / f"validation_{prefix}.pkl"
    test_path = config.DATASETS_DIR / f"test_{prefix}.pkl" if "test" in splits else None
    train_path.write_bytes(pickle.dumps(splits.get("train", [])))
    validation_path.write_bytes(pickle.dumps(splits.get("validation", [])))
    if test_path:
        test_path.write_bytes(pickle.dumps(splits.get("test", [])))
    return train_path, validation_path, test_path
