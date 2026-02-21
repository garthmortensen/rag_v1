"""System utilities for resource-aware model selection.

Uses psutil to check available RAM and selects the best
HuggingFace sentence-transformers embedding model that fits
within available memory (with a 30% overhead buffer).

Runtime memory estimates include the model weights, tokenizer,
PyTorch runtime, and a working buffer for inference.
"""

import logging

import psutil

logger = logging.getLogger(__name__)

# ── Model registry ──────────────────────────────────────────────────
# Ordered from best quality → smallest, so we pick the best one that
# fits.  Memory estimates are conservative runtime totals (weights +
# tokenizer + PyTorch overhead + inference buffer) measured empirically.
#
# Model                  | Params | Dims | Disk    | RAM (runtime)
# ─────────────────────────────────────────────────────────────────
# all-mpnet-base-v2      | 109 M  | 768  | ~420 MB | ~600 MB
# all-MiniLM-L12-v2      |  33 M  | 384  | ~130 MB | ~260 MB
# all-MiniLM-L6-v2       |  23 M  | 384  |  ~90 MB | ~200 MB

EMBEDDING_MODELS = [
    {
        "name": "all-mpnet-base-v2",
        "params_millions": 109,
        "dimensions": 768,
        "ram_mb": 600,
        "max_seq_length": 384,
        "description": "Best quality — highest accuracy on semantic similarity benchmarks",
    },
    {
        "name": "all-MiniLM-L12-v2",
        "params_millions": 33,
        "dimensions": 384,
        "ram_mb": 260,
        "max_seq_length": 256,
        "description": "Good balance — 12-layer MiniLM, better accuracy than L6",
    },
    {
        "name": "all-MiniLM-L6-v2",
        "params_millions": 23,
        "dimensions": 384,
        "ram_mb": 200,
        "max_seq_length": 256,
        "description": "Lightest — 6-layer MiniLM, fast and small",
    },
]

# 30% overhead buffer: only use 70% of available RAM for the model
OVERHEAD_FACTOR = 1.30


def get_available_ram_mb() -> float:
    """Return available system RAM in megabytes.

    Uses psutil.virtual_memory().available, which accounts for
    OS caches and buffers (i.e. memory the OS can reclaim).
    """
    mem = psutil.virtual_memory()
    available_mb = mem.available / (1024 * 1024)
    return available_mb


def select_best_model(available_ram_mb: float | None = None) -> dict:
    """Pick the best embedding model that fits in available RAM.

    Applies a 30% overhead factor — the model's estimated runtime
    RAM is multiplied by 1.3 to leave headroom for the OS, ChromaDB,
    document processing, and other concurrent work.

    Parameters
    ----------
    available_ram_mb : float | None
        Override for testing.  If None, reads from psutil.

    Returns
    -------
    dict
        The selected model entry from EMBEDDING_MODELS.

    Raises
    ------
    MemoryError
        If even the smallest model won't fit.
    """
    if available_ram_mb is None:
        available_ram_mb = get_available_ram_mb()

    logger.info(f"Available RAM: {available_ram_mb:,.0f} MB")

    for model in EMBEDDING_MODELS:
        required_mb = model["ram_mb"] * OVERHEAD_FACTOR
        if available_ram_mb >= required_mb:
            logger.info(
                f"Selected model: {model['name']}  "
                f"({model['params_millions']}M params, {model['dimensions']}d)  "
                f"needs {required_mb:,.0f} MB (with 30% overhead), "
                f"{available_ram_mb:,.0f} MB available"
            )
            return model
        else:
            logger.debug(
                f"Skipping {model['name']} — needs {required_mb:,.0f} MB, "
                f"only {available_ram_mb:,.0f} MB available"
            )

    smallest = EMBEDDING_MODELS[-1]
    required = smallest["ram_mb"] * OVERHEAD_FACTOR
    raise MemoryError(
        f"Not enough RAM to load even the smallest model "
        f"({smallest['name']}). "
        f"Need {required:,.0f} MB but only {available_ram_mb:,.0f} MB available."
    )


def log_system_info() -> None:
    """Log a summary of system memory and the selected model."""
    mem = psutil.virtual_memory()
    total_mb = mem.total / (1024 * 1024)
    available_mb = mem.available / (1024 * 1024)
    used_pct = mem.percent

    logger.info(
        f"System RAM: {total_mb:,.0f} MB total, "
        f"{available_mb:,.0f} MB available ({used_pct}% used)"
    )

    model = select_best_model(available_mb)
    logger.info(
        f"Recommended model: {model['name']} — {model['description']}"
    )
