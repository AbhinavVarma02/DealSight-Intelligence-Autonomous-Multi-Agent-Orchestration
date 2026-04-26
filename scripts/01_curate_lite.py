"""CLI entry point for dataset preparation.

Three modes:
- default: build a fresh lite dataset from raw Amazon metadata.
- `--from-source --purpose structured`: download a structured `items_*`
  dataset from Hugging Face (or a local folder) and write the splits.
- `--from-source --purpose prompt`: do the same for a prompt/completion
  fine-tuning dataset.
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dealsight_intelligence import config
from dealsight_intelligence.data.curate_lite import curate_lite, download_hub_dataset, export_prompt_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or download datasets for DealSight Intelligence.")
    parser.add_argument(
        "--from-hub",
        "--from-source",
        action="store_true",
        dest="from_source",
        help="Load a Hugging Face dataset repo ID or local exported dataset folder",
    )
    parser.add_argument(
        "--purpose",
        choices=["structured", "prompt"],
        default="structured",
        help="structured is for app/vectorstore/RF; prompt is for prompt training/eval only",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="HF dataset repo ID or local dataset folder. Defaults to dealsight_intelligence_*_DATASET_SOURCE.",
    )
    parser.add_argument("--prefix", default=None, help="Output filename prefix, for example lite, full, prompts_lite")
    parser.add_argument("--category", default="Appliances", help="Amazon metadata category to load")
    parser.add_argument("--train-size", type=int, default=20000, help="Number of training items")
    parser.add_argument("--test-size", type=int, default=2000, help="Number of test items")
    args = parser.parse_args()
    if args.from_source:
        if args.purpose == "prompt":
            prefix = args.prefix or "prompts_lite"
            train_path, validation_path, test_path = export_prompt_dataset(dataset_name=args.dataset, prefix=prefix)
            print(f"Wrote {train_path}")
            print(f"Wrote {validation_path}")
            if test_path:
                print(f"Wrote {test_path}")
            raise SystemExit(0)
        prefix = args.prefix or config.dataset_prefix("lite")
        train_path, validation_path, test_path = download_hub_dataset(dataset_name=args.dataset, prefix=prefix)
        print(f"Wrote {train_path}")
        print(f"Wrote {validation_path}")
        print(f"Wrote {test_path}")
        raise SystemExit(0)
    train_path, test_path = curate_lite(
        category=args.category,
        train_size=args.train_size,
        test_size=args.test_size,
    )
    print(f"Wrote {train_path}")
    print(f"Wrote {test_path}")
