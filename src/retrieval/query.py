"""Semantic search over the stress testing vector DB.

Embeds a natural-language query with the same HuggingFace model used
at ingestion time, then performs approximate nearest-neighbor (ANN)
search against the ChromaDB collection to return the most relevant
document chunks.

Usage (CLI):
    python -m src.retrieval.query "What is the peak unemployment rate?"
    python -m src.retrieval.query "CET1 capital ratio" --top-k 10
    python -m src.retrieval.query "credit risk" --filter source_type=pdf
"""

import argparse
import logging
import sys
import textwrap

from rich.console import Console
from rich.panel import Panel

from src.embedding.model import (
    get_embedding_function,
    get_or_create_collection,
    COLLECTION_NAME,
    VECTOR_DB_DIR,
)

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5


# ── Core ────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    n_results: int = DEFAULT_TOP_K,
    where: dict | None = None,
    persist_dir: str = VECTOR_DB_DIR,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """Embed a query and return the top-k most similar chunks.

    Parameters
    ----------
    query : str
        Natural-language question or keyword phrase.
    n_results : int
        Number of results to return.
    where : dict | None
        Optional ChromaDB metadata filter, e.g.
        ``{"source_type": "pdf"}`` or
        ``{"title": "Credit Risk Models"}``.
    persist_dir : str
        Path to the ChromaDB on-disk directory.
    collection_name : str
        Name of the ChromaDB collection.

    Returns
    -------
    dict
        Raw ChromaDB query results with keys:
        ``ids``, ``documents``, ``metadatas``, ``distances``.
        Each value is a list-of-lists (one inner list per query).
    """
    logger.info(f"Embedding query: '{query}'")
    embedding_fn = get_embedding_function()
    collection = get_or_create_collection(persist_dir, collection_name)

    query_embedding = embedding_fn.embed_query(query)
    logger.info(f"Query embedded to {len(query_embedding)}-dim vector")

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
        logger.info(f"Applying metadata filter: {where}")

    results = collection.query(**kwargs)

    n_returned = len(results["ids"][0])
    best_dist = results["distances"][0][0] if n_returned else float("inf")
    logger.info(
        f"Retrieved {n_returned} result(s)  "
        f"(best distance: {best_dist:.4f})"
    )
    return results


def retrieve_formatted(
    query: str,
    n_results: int = DEFAULT_TOP_K,
    where: dict | None = None,
    persist_dir: str = VECTOR_DB_DIR,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """Run a retrieval query and return results as a flat list of dicts.

    Each dict contains ``rank``, ``id``, ``distance``, ``text``,
    and ``metadata`` — a convenient shape for downstream consumers
    like an LLM prompt builder or evaluation harness.

    Parameters
    ----------
    query : str
        Natural-language question or keyword phrase.
    n_results : int
        Number of results to return.
    where : dict | None
        Optional ChromaDB metadata filter.
    persist_dir : str
        Path to the ChromaDB on-disk directory.
    collection_name : str
        Name of the ChromaDB collection.

    Returns
    -------
    list[dict]
        One dict per result, ranked by ascending distance (nearest
        first).  Keys: ``rank``, ``id``, ``distance``, ``text``,
        ``metadata``.
    """
    raw = retrieve(
        query,
        n_results=n_results,
        where=where,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    formatted = []
    for i, chunk_id in enumerate(raw["ids"][0]):
        formatted.append(
            {
                "rank": i + 1,
                "id": chunk_id,
                "distance": raw["distances"][0][i],
                "text": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],
            }
        )

    logger.info(
        f"Formatted {len(formatted)} result(s) for query: "
        f"'{query[:60]}{'…' if len(query) > 60 else ''}'"
    )
    return formatted

console = Console()

def print_ascii_banner():
    console.print(
        Panel.fit(
            """[bold medium_purple1]
            ,/   *
         _,'/_   |
         `(")' ,'/   QUERY
      _ _,-H-./ /    INTERFACE
      \_\_\.   /
        )" |  (
     __/   H   \__
     \    /|\    /
      `--'|||`--'
         ==^==
[/bold medium_purple1]
--------------------------------
""",
            border_style="grey39",
        )
    )


def _parse_filter(raw: str) -> dict:
    """Parse 'key=value' into a ChromaDB where-filter dict.

    Supports a single key=value pair.  Raises ValueError on bad
    syntax.

    >>> _parse_filter("source_type=pdf")
    {'source_type': 'pdf'}
    """
    if "=" not in raw:
        raise ValueError(
            f"Filter must be key=value (got '{raw}'). "
            "Example: --filter source_type=pdf"
        )
    key, value = raw.split("=", 1)
    return {key.strip(): value.strip()}


def _print_results(results: list[dict]) -> None:
    """Pretty-print formatted retrieval results to stdout."""
    if not results:
        print("No results found.")
        return

    for r in results:
        title = r["metadata"].get("title", "—")
        source = r["metadata"].get("source", "—")
        preview = textwrap.shorten(r["text"], width=120, placeholder=" …")

        print(f"\n{'─' * 72}")
        print(f"  Rank {r['rank']}  │  distance: {r['distance']:.4f}")
        print(f"  ID:     {r['id']}")
        print(f"  Title:  {title}")
        print(f"  Source: {source}")
        print(f"  Text:   {preview}")

    print(f"\n{'─' * 72}")
    print(f"  {len(results)} result(s) returned.\n")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for interactive retrieval queries."""
    parser = argparse.ArgumentParser(
        prog="python -m src.retrieval.query",
        description="Query the stress-testing vector DB.",
    )
    parser.add_argument("query", help="Natural-language query string")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Metadata filter as key=value (e.g. source_type=pdf)",
    )

    args = parser.parse_args(argv)

    print_ascii_banner()

    where = _parse_filter(args.filter) if args.filter else None
    results = retrieve_formatted(args.query, n_results=args.top_k, where=where)
    _print_results(results)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    main()
