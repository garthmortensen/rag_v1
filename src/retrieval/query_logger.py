"""Log every query and rewrite session to a timestamped file in logs/.

Each run of the CLI (or programmatic call to ``log_query_session`` /
``log_rewrite_session``) produces a single ``.log`` file containing:

* Active config.txt settings
* Timestamp & collection name
* The user's query string
* All retrieved chunks (rank, id, distance, title, source, full text)
* The LLM-generated answer (if any)

For rewrite sessions the log records:

* PDF path, mode, provider, model, temperature, custom prompt
* Per-section input/output pairs (section, subsection, source chars, rewritten text)
* Output file path and timing

Files are named ``YYYYMMDD_HHMMSS.log`` so they sort
chronologically and never collide.
"""

import logging
import os
from dataclasses import dataclass
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


# ── Rewrite session logging ─────────────────────────────────────────


@dataclass
class RewriteSectionLog:
    """One section's worth of rewrite data for logging."""

    section: str
    subsection: str
    source_chars: int
    rewritten_chars: int
    rewritten_text: str


def log_rewrite_session(
    *,
    pdf_path: str,
    mode: str,
    provider: str,
    model: str,
    temperature: float,
    custom_prompt: str | None = None,
    sections: list[RewriteSectionLog] | None = None,
    output_path: str | None = None,
    elapsed_seconds: float | None = None,
    logs_dir: str = LOGS_DIR,
) -> str:
    """Write a complete rewrite session to a log file.

    Parameters
    ----------
    pdf_path : str
        Path to the source PDF.
    mode : str
        ``"rewrite"`` or ``"summarize"``.
    provider, model : str
        LLM provider and model name.
    temperature : float
        Sampling temperature used.
    custom_prompt : str | None
        Any user-supplied custom instructions.
    sections : list[RewriteSectionLog] | None
        Per-section input/output records.
    output_path : str | None
        Path to the written Markdown file.
    elapsed_seconds : float | None
        Wall-clock time for the full rewrite.
    logs_dir : str
        Directory for log files.

    Returns
    -------
    str
        Path to the log file.
    """
    _ensure_logs_dir(logs_dir)

    now = datetime.now()
    filename = now.strftime("%Y%m%d_%H%M%S_rewrite.log")
    filepath = os.path.join(logs_dir, filename)

    separator = "─" * 72

    with open(filepath, "w", encoding="utf-8") as fh:
        # ── Config ──────────────────────────────────────────────────
        fh.write("CONFIG\n")
        fh.write(f"{separator}\n")
        fh.write(f"{config_as_text()}\n")
        fh.write(f"{separator}\n\n")

        # ── Parameters ──────────────────────────────────────────────
        fh.write("REWRITE SESSION\n")
        fh.write(f"{separator}\n")
        fh.write(f"Timestamp:    {now.isoformat()}\n")
        fh.write(f"PDF:          {pdf_path}\n")
        fh.write(f"Mode:         {mode}\n")
        fh.write(f"Provider:     {provider}\n")
        fh.write(f"Model:        {model}\n")
        fh.write(f"Temperature:  {temperature}\n")
        if custom_prompt:
            fh.write(f"Custom Prompt:\n  {custom_prompt.strip()}\n")
        else:
            fh.write("Custom Prompt: (none)\n")
        if output_path:
            fh.write(f"Output:       {output_path}\n")
        if elapsed_seconds is not None:
            mins, secs = divmod(int(elapsed_seconds), 60)
            fh.write(f"Elapsed:      {elapsed_seconds:.1f}s ({mins:02d}:{secs:02d})\n")
        total = len(sections) if sections else 0
        fh.write(f"Sections:     {total}\n")
        fh.write(f"{separator}\n\n")

        # ── Per-section details ─────────────────────────────────────
        if sections:
            for i, sec in enumerate(sections, 1):
                fh.write(f"[{i}/{total}] {sec.section} › {sec.subsection}\n")
                fh.write(f"  Source chars:    {sec.source_chars:,}\n")
                fh.write(f"  Rewritten chars: {sec.rewritten_chars:,}\n")
                fh.write(f"  Rewritten text:\n{sec.rewritten_text}\n")
                fh.write(f"\n{separator}\n\n")

    logger.info(f"Rewrite session logged to {filepath}")
    return filepath
