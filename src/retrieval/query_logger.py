"""Log every query session to a timestamped file in logs/.

Each run of the CLI (or programmatic call to ``log_query_session``)
produces a single ``.log`` file containing:

* Active config.txt settings
* Timestamp & collection name
* The user's query string
* All retrieved chunks (rank, id, distance, title, source, full text)
* The LLM-generated answer (if any)

Files are named ``YYYYMMDD_HHMMSS.log`` so they sort
chronologically and never collide.
"""

import logging
import os
from datetime import datetime

from src.config import config_as_text

logger = logging.getLogger(__name__)

LOGS_DIR = os.path.join("logs")


def _ensure_logs_dir(logs_dir: str = LOGS_DIR) -> None:
    """Create the logs directory if it doesn't exist."""
    os.makedirs(logs_dir, exist_ok=True)


def log_query_session(
    query: str,
    results: list[dict],
    answer: str | None = None,
    collection_name: str = "",
    logs_dir: str = LOGS_DIR,
    query_time: datetime | None = None,
    response_time: datetime | None = None,
    elapsed_seconds: float | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Write a complete query session to a log file.

    Parameters
    ----------
    query : str
        The user's natural-language query.
    results : list[dict]
        Output of ``retrieve_formatted()`` — each dict has keys
        ``rank``, ``id``, ``distance``, ``text``, ``metadata``.
    answer : str | None
        The LLM-generated answer, if any.
    collection_name : str
        The ChromaDB collection that was queried.
    logs_dir : str
        Directory to write log files into.
    query_time : datetime | None
        When the question was submitted.
    response_time : datetime | None
        When the answer was completed.
    elapsed_seconds : float | None
        Wall-clock seconds for generation.
    provider : str | None
        LLM provider used (ollama, openai, etc.).
    model : str | None
        LLM model name used.

    Returns
    -------
    str
        Path to the log file that was written.
    """
    _ensure_logs_dir(logs_dir)

    now = datetime.now()
    filename = now.strftime("%Y%m%d_%H%M%S.log")
    filepath = os.path.join(logs_dir, filename)

    separator = "─" * 72

    with open(filepath, "w", encoding="utf-8") as fh:
        # ── Config ──────────────────────────────────────────────────
        fh.write("CONFIG\n")
        fh.write(f"{separator}\n")
        fh.write(f"{config_as_text()}\n")
        fh.write(f"{separator}\n\n")

        # ── Header ──────────────────────────────────────────────────
        fh.write(f"Timestamp:  {now.isoformat()}\n")
        if collection_name:
            fh.write(f"Collection: {collection_name}\n")
        if provider:
            fh.write(f"Provider:   {provider}\n")
        if model:
            fh.write(f"Model:      {model}\n")
        fh.write(f"Query:      {query}\n")
        fh.write(f"Results:    {len(results)}\n")
        if query_time:
            fh.write(f"Asked at:   {query_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if response_time:
            fh.write(f"Answered:   {response_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if elapsed_seconds is not None:
            fh.write(f"Elapsed:    {elapsed_seconds:.1f}s\n")
        fh.write(f"{separator}\n\n")

        # ── Retrieved chunks ────────────────────────────────────────
        for r in results:
            title = r["metadata"].get("title", "—")
            source = r["metadata"].get("source", "—")
            fh.write(f"Rank {r['rank']}  │  distance: {r['distance']:.4f}\n")
            fh.write(f"  ID:     {r['id']}\n")
            fh.write(f"  Title:  {title}\n")
            fh.write(f"  Source: {source}\n")
            fh.write(f"  Text:\n{r['text']}\n")
            fh.write(f"\n{separator}\n\n")

        # ── LLM answer (optional) ──────────────────────────────────
        if answer is not None:
            fh.write("LLM ANSWER\n")
            fh.write(f"{separator}\n")
            fh.write(f"{answer}\n")
            fh.write(f"{separator}\n")

    logger.info(f"Query session logged to {filepath}")
    return filepath
