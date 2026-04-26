"""Dataset registry, loaders, and schema validators.

Three kinds of Hugging Face datasets are supported:

- structured (`items_lite`/`items_full`) — the app, vector store, and
  pricing pipelines all use this shape (title, category, price, summary).
- prompt (`items_prompts_lite`/`items_prompts_full`) — prompt/completion
  pairs for prompt-based fine-tuning and prompt evaluation only.
- raw (`items_raw_lite`/`items_raw_full`) — raw source rows; supported
  but not the default app input.

The validators here raise `DatasetSchemaError` when a dataset is fed into
the wrong pipeline so silent shape mismatches never reach the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from dealsight_intelligence import config
from dealsight_intelligence.data.items import Item

DatasetPurpose = Literal["structured", "prompt", "raw"]

STRUCTURED_DATASETS = {
    "lite": "abhinavvathadi/items_lite",
    "full": "abhinavvathadi/items_full",
    "ed_lite": "ed-donner/items_lite",
    "ed_full": "ed-donner/items_full",
}

PROMPT_DATASETS = {
    "lite": "abhinavvathadi/items_prompts_lite",
    "full": "abhinavvathadi/items_prompts_full",
}

RAW_DATASETS = {
    "lite": "abhinavvathadi/items_raw_lite",
    "full": "abhinavvathadi/items_raw_full",
}

STRUCTURED_REQUIRED_FIELDS = {"title", "category", "price", "summary"}
PROMPT_REQUIRED_FIELDS = {"prompt", "completion"}


class DatasetSchemaError(ValueError):
    """Raised when a dataset source is used for the wrong pipeline."""


@dataclass(frozen=True)
class PromptExample:
    prompt: str
    completion: str


def resolve_dataset_source(
    source: str | Path | None = None,
    purpose: DatasetPurpose = "structured",
    size: str = "lite",
) -> str:
    if source:
        return str(source)
    if purpose == "structured":
        return config.structured_dataset_source()
    if purpose == "prompt":
        return config.prompt_dataset_source()
    return config.raw_dataset_source()


def load_dataset_anywhere(source: str | Path):
    """Load a Hugging Face dataset by repo ID or a local exported folder.

    Tries `load_from_disk` first when the path exists, then falls back to
    `load_dataset` so users can pass either an HF ID or a local copy.
    """

    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError("Install ML dependencies with: python -m pip install -e '.[ml]'") from exc

    source_text = str(source)
    source_path = Path(source_text).expanduser()
    if source_path.exists():
        try:
            return load_from_disk(str(source_path))
        except Exception as exc:
            try:
                return load_dataset(str(source_path))
            except Exception as second_exc:
                raise RuntimeError(
                    f"Could not load local dataset folder {source_path}. "
                    f"Tried load_from_disk and load_dataset. Errors: {exc}; {second_exc}"
                ) from second_exc
    try:
        return load_dataset(source_text)
    except Exception as exc:
        raise RuntimeError(f"Could not load Hugging Face dataset source {source_text}: {exc}") from exc


def load_structured_items(source: str | Path | None = None) -> tuple[list[Item], list[Item], list[Item]]:
    dataset_source = resolve_dataset_source(source, purpose="structured")
    dataset = load_dataset_anywhere(dataset_source)
    _validate_dataset_dict(dataset, STRUCTURED_REQUIRED_FIELDS, dataset_source, "structured")
    return (
        [Item.from_mapping(row) for row in dataset["train"]],
        [Item.from_mapping(row) for row in _optional_split(dataset, "validation")],
        [Item.from_mapping(row) for row in dataset["test"]],
    )


def load_prompt_examples(source: str | Path | None = None) -> dict[str, list[PromptExample]]:
    dataset_source = resolve_dataset_source(source, purpose="prompt")
    dataset = load_dataset_anywhere(dataset_source)
    _validate_dataset_dict(dataset, PROMPT_REQUIRED_FIELDS, dataset_source, "prompt")
    return {
        split: [PromptExample(prompt=str(row["prompt"]), completion=str(row["completion"])) for row in dataset[split]]
        for split in dataset.keys()
    }


def validate_structured_items(items: Iterable[Item], source: str | Path) -> None:
    missing_rows = []
    for index, item in enumerate(items):
        if not item.title or not item.category or item.price <= 0 or not item.summary:
            missing_rows.append(index)
        if len(missing_rows) >= 5:
            break
    if missing_rows:
        raise DatasetSchemaError(
            f"Structured dataset {source} has rows missing title/category/positive price/summary. "
            f"Example bad row indexes: {missing_rows}. Use items_lite/items_full, not prompt or raw datasets."
        )


def validate_prompt_examples(examples: Iterable[PromptExample], source: str | Path) -> None:
    missing_rows = []
    for index, example in enumerate(examples):
        if not example.prompt or not example.completion:
            missing_rows.append(index)
        if len(missing_rows) >= 5:
            break
    if missing_rows:
        raise DatasetSchemaError(
            f"Prompt dataset {source} has rows missing prompt/completion. "
            f"Example bad row indexes: {missing_rows}. Use items_prompts_lite/items_prompts_full."
        )


def _validate_dataset_dict(dataset, required_fields: set[str], source: str, purpose: DatasetPurpose) -> None:
    if "train" not in dataset or "test" not in dataset:
        raise DatasetSchemaError(
            f"{purpose.title()} dataset {source} must include at least train and test splits. "
            f"Found splits: {list(dataset.keys())}."
        )
    fields = set(dataset["train"].column_names)
    missing = sorted(required_fields - fields)
    if missing:
        expected = ", ".join(sorted(required_fields))
        found = ", ".join(sorted(fields))
        if purpose == "structured":
            hint = "Use items_lite/items_full for vector stores and structured pricing/evaluation."
        elif purpose == "prompt":
            hint = "Use items_prompts_lite/items_prompts_full only for prompt training/eval."
        else:
            hint = "Raw datasets are not app-ready by default."
        raise DatasetSchemaError(
            f"{purpose.title()} dataset {source} has the wrong schema. "
            f"Missing fields: {missing}. Expected fields include: {expected}. Found fields: {found}. {hint}"
        )


def _optional_split(dataset, split: str):
    return dataset[split] if split in dataset else []
