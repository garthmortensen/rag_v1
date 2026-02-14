"""Load pipeline settings from config.txt.

Reads a simple ``key = value`` text file from the project root.
Blank lines and lines starting with ``#`` are ignored.
Integer-looking values are cast to ``int`` automatically.

Usage::

    from src.config import CFG

    chunk_size      = CFG["chunk_size"]       # int
    collection_name = CFG["collection_name"]  # str

If config.txt is missing, sensible defaults are used so the rest of
the pipeline still works.
"""

import logging
import os

logger = logging.getLogger(__name__)

# ── Defaults (match the original hard-coded values) ─────────────────
DEFAULTS: dict[str, str | int] = {
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "collection_name": "stress_test_docs_1k",
}

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "config.txt")


def load_config(path: str = CONFIG_PATH) -> dict[str, str | int]:
    """Parse *path* and return a merged dict of defaults + overrides.

    File format (one pair per line)::

        # comment
        chunk_size = 6000
        collection_name = stress_test_docs_6k  # or stress_test_docs_1k

    Returns
    -------
    dict[str, str | int]
        Merged configuration.  Keys not present in the file keep
        their default values.
    """
    cfg: dict[str, str | int] = dict(DEFAULTS)

    resolved = os.path.normpath(path)
    if not os.path.isfile(resolved):
        logger.warning(
            f"Config file not found at {resolved} — using defaults"
        )
        return cfg

    with open(resolved, encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                logger.warning(
                    f"config.txt:{lineno}: skipping malformed line: {line!r}"
                )
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Auto-cast purely numeric values to int
            if value.isdigit():
                value = int(value)

            cfg[key] = value

    logger.info(f"Loaded config from {resolved}: {cfg}")
    return cfg


# Module-level singleton — imported everywhere as ``from src.config import CFG``
CFG: dict[str, str | int] = load_config()
