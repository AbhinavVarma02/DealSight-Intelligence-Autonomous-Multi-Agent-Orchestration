"""Build the Chroma vector store of priced products.

Reads a pickled list of `Item`s, encodes their text with
`all-MiniLM-L6-v2`, and upserts them in batches into a persistent Chroma
collection. The Frontier agent then queries this collection at runtime
for similar-item context.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from dealsight_intelligence import config
from dealsight_intelligence.data.datasets import validate_structured_items


def build_vectorstore(
    dataset_path: Path | None = None,
    persist_path: Path | None = None,
    batch_size: int = 1000,
    reset: bool = False,
) -> Path:
    dataset_path = dataset_path or config.path_env(
        "DEALSIGHT_INTELLIGENCE_STRUCTURED_TRAIN_PATH",
        config.DATASETS_DIR / f"train_{config.dataset_prefix()}.pkl",
    )
    persist_path = persist_path or config.PRODUCTS_VECTORSTORE
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install ML dependencies with: python -m pip install -e '.[ml]'") from exc

    items = pickle.loads(dataset_path.read_bytes())
    validate_structured_items(items, dataset_path)
    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))
    collection_name = "products"
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(collection_name)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    documents = [item.text for item in items]
    metadatas = [{"price": float(item.price), "category": item.category} for item in items]
    total = len(documents)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_documents = documents[start:end]
        batch_embeddings = encoder.encode(batch_documents).tolist()
        batch_ids = [f"item-{index}" for index in range(start, end)]
        collection.upsert(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=batch_embeddings,
            metadatas=metadatas[start:end],
        )
        print(f"Added {end:,}/{total:,} items to vector store")
    return persist_path
