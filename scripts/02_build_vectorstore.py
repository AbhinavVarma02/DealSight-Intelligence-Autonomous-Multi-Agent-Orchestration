"""CLI entry point for building the Chroma vector store from a curated dataset."""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dealsight_intelligence.pricing.vectorstore import build_vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Chroma vector store from a pickled Item dataset.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Path to train_lite.pkl or train_full.pkl")
    parser.add_argument("--batch-size", type=int, default=1000, help="Chroma insert/upsert batch size")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the products collection first")
    args = parser.parse_args()
    print(
        f"Built vector store at "
        f"{build_vectorstore(dataset_path=args.dataset_path, batch_size=args.batch_size, reset=args.reset)}"
    )
