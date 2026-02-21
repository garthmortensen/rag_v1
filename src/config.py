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

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

# ── Load .env (API keys, OLLAMA_HOST, etc.) ─────────────────────────
load_dotenv()  # reads .env from project root, if present

_console = Console()

# ── Defaults (match the original hard-coded values) ─────────────────
DEFAULTS: dict[str, str | int | bool] = {
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "collection_name": "rag_docs_1k",
    "beep_on_answer": True,
    "llm_provider": "ollama",
    "llm_model": "llama3.2:3b",
}

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "config.txt")


# Values recognised as boolean true / false (case-insensitive).
_BOOL_TRUE = frozenset({"true", "yes", "1", "on"})
_BOOL_FALSE = frozenset({"false", "no", "0", "off"})


def load_config(path: str = CONFIG_PATH) -> dict[str, str | int | bool]:
    """Parse *path* and return a merged dict of defaults + overrides.

    File format (one pair per line)::

        # comment
        chunk_size = 6000
        collection_name = rag_docs_6k  # or rag_docs_1k
        beep_on_answer = true

    Boolean values are recognised as true/yes/1/on and false/no/0/off
    (case-insensitive).

    Returns
    -------
    dict[str, str | int | bool]
        Merged configuration.  Keys not present in the file keep
        their default values.
    """
    cfg: dict[str, str | int | bool] = dict(DEFAULTS)

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

            # Auto-cast booleans (true/yes/1/on, false/no/0/off)
            if value.lower() in _BOOL_TRUE:
                cfg[key] = True
                continue
            if value.lower() in _BOOL_FALSE:
                cfg[key] = False
                continue

            # Auto-cast purely numeric values to int
            if value.isdigit():
                value = int(value)

            cfg[key] = value

    logger.info(f"Loaded config from {resolved}: {cfg}")
    return cfg


# Module-level singleton — imported everywhere as ``from src.config import CFG``
CFG: dict[str, str | int | bool] = load_config()


def print_config(cfg: dict[str, str | int | bool] | None = None) -> None:
    """Pretty-print the active configuration using a rich table.

    Called after each ASCII banner so the user always sees which
    chunk size and collection are in effect.
    """
    cfg = cfg if cfg is not None else CFG
    table = Table(
        title="config.txt",
        title_style="bold yellow",
        border_style="yellow",
        show_header=True,
        header_style="bold",
        padding=(0, 2),
    )
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white bold")
    for key, value in cfg.items():
        table.add_row(str(key), str(value))
    _console.print(table)


def config_as_text(cfg: dict[str, str | int | bool] | None = None) -> str:
    """Return the active configuration as a plain-text block for log files."""
    cfg = cfg if cfg is not None else CFG
    lines = [f"{k} = {v}" for k, v in cfg.items()]
    return "\n".join(lines)
