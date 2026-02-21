"""Semantic search over a ChromaDB-backed vector store.

Embeds a natural-language query with the same HuggingFace model used
at ingestion time, then performs approximate nearest-neighbor (ANN)
search against the ChromaDB collection to return the most relevant
document chunks.

Usage (programmatic)::

    from src.retrieval.query import retrieve_formatted
    chunks = retrieve_formatted("What are the key points in this document?")
"""

import logging

from src.config import CFG
from src.embedding.model import (
    get_embedding_function,
    get_or_create_collection,
    VECTOR_DB_DIR,
)

# Derive from config so queries hit the same collection that was ingested
COLLECTION_NAME = str(CFG["collection_name"])

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
    logger.info(f"Retrieved {n_returned} result(s)  (best distance: {best_dist:.4f})")
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
