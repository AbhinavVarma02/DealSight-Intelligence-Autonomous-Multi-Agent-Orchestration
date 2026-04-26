"""Tests that the dataset schema validators catch the most common mistake:
feeding a prompt-style dataset into a structured pipeline (or vice versa).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dealsight_intelligence.data.datasets import (
    DatasetSchemaError,
    PromptExample,
    validate_prompt_examples,
    validate_structured_items,
)
from dealsight_intelligence.data.items import Item


class DatasetValidationTests(unittest.TestCase):
    def test_structured_validation_rejects_prompt_like_items(self):
        items = [Item(title="Prompt row", price=1.0, category="Unknown", summary=None, saved_prompt="Q")]
        with self.assertRaises(DatasetSchemaError):
            validate_structured_items(items, "bad-source")

    def test_structured_validation_accepts_structured_items(self):
        items = [Item(title="Product", price=12.0, category="Tools", summary="Useful product summary")]
        validate_structured_items(items, "good-source")

    def test_prompt_validation_requires_prompt_and_completion(self):
        with self.assertRaises(DatasetSchemaError):
            validate_prompt_examples([PromptExample(prompt="Question", completion="")], "bad-prompts")


if __name__ == "__main__":
    unittest.main()
