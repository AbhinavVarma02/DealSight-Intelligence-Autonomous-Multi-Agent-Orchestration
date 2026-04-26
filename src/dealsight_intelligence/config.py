"""Project paths, defaults, and helpers for reading typed environment variables.

Everything that touches disk or relies on configuration goes through this
module so paths and defaults stay consistent across agents and scripts.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    # Walks up from src/dealsight_intelligence/config.py to the repo root.
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
ARTIFACTS = ROOT / "artifacts"
DATASETS_DIR = ARTIFACTS / "datasets"
MEMORY_DIR = ARTIFACTS / "memory"
MODELS_DIR = ARTIFACTS / "models"
VECTORSTORES_DIR = ARTIFACTS / "vectorstores"

MEMORY_FILE = MEMORY_DIR / "memory.json"
PRODUCTS_VECTORSTORE = VECTORSTORES_DIR / "products_vectorstore"
ENSEMBLE_MODEL = MODELS_DIR / "ensemble_model.pkl"
DEEP_NEURAL_NETWORK_MODEL = MODELS_DIR / "deep_neural_network.pth"

DEFAULT_STRUCTURED_DATASET = "abhinavvathadi/items_lite"
DEFAULT_PROMPT_DATASET = "abhinavvathadi/items_prompts_lite"
DEFAULT_RAW_DATASET = "abhinavvathadi/items_raw_lite"


# Typed env-var readers. Each one falls back to the default when the variable
# is missing, blank, or cannot be parsed.
def bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def str_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value and value.strip() else default


def structured_dataset_source() -> str:
    return str_env("DEALSIGHT_INTELLIGENCE_STRUCTURED_DATASET_SOURCE", DEFAULT_STRUCTURED_DATASET)


def prompt_dataset_source() -> str:
    return str_env("DEALSIGHT_INTELLIGENCE_PROMPT_DATASET_SOURCE", DEFAULT_PROMPT_DATASET)


def raw_dataset_source() -> str:
    return str_env("DEALSIGHT_INTELLIGENCE_RAW_DATASET_SOURCE", DEFAULT_RAW_DATASET)


def dataset_prefix(default: str = "lite") -> str:
    return str_env("DEALSIGHT_INTELLIGENCE_DATASET_PREFIX", default)


def ensure_artifact_dirs() -> None:
    # Create the artifact subdirectories on first run so writes never fail
    # because of a missing folder.
    for path in (DATASETS_DIR, MEMORY_DIR, MODELS_DIR, VECTORSTORES_DIR):
        path.mkdir(parents=True, exist_ok=True)
