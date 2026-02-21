"""Tests for src/retrieval/query.py.

Two test classes:

1.  **TestRetrievalUnit** — fast, mocked unit tests that verify
    retrieve() and retrieve_formatted() call the right things
    with the right arguments, without touching ChromaDB or
    downloading an embedding model.

2.  **TestRetrievalIntegration** — slower integration tests that
    query the *live* vector DB at corpus/vector_db/ and verify
    that basic retrieval works against a populated collection.
    These are skipped automatically when the DB is missing.

Run with:
    pytest tests/test_retrieval.py -v
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.query import (
    retrieve,
    retrieve_formatted,
    DEFAULT_TOP_K,
)
from src.embedding.model import COLLECTION_NAME, VECTOR_DB_DIR


# ── Helpers ─────────────────────────────────────────────────────────

def _mock_query_results(n: int = 3) -> dict:
    """Build a fake ChromaDB query result dict with *n* hits."""
    return {
        "ids": [[f"doc_chunk_{i:04d}" for i in range(n)]],
        "documents": [[f"Sample text for chunk {i}" for i in range(n)]],
        "metadatas": [
            [
                {
                    "source": f"corpus/raw_data/file_{i}.pdf",
                    "title": f"Title {i}",
                    "doc_id": f"ID{i}",
                    "source_type": "pdf",
                }
                for i in range(n)
            ]
        ],
        "distances": [[0.1 * (i + 1) for i in range(n)]],
    }


# ── Unit Tests (mocked) ────────────────────────────────────────────

class TestRetrievalUnit(unittest.TestCase):
    """Fast, mocked tests for retrieve() and retrieve_formatted()."""

    @patch("src.retrieval.query.get_or_create_collection")
    @patch("src.retrieval.query.get_embedding_function")
    def test_retrieve_embeds_query_and_calls_collection(
        self, mock_get_emb, mock_get_col
    ):
        """retrieve() should embed the query and call collection.query()."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.0] * 384
        mock_get_emb.return_value = mock_embedding_fn

        mock_collection = MagicMock()
        mock_collection.query.return_value = _mock_query_results(3)
        mock_get_col.return_value = mock_collection

        results = retrieve("test query", n_results=5)

        mock_embedding_fn.embed_query.assert_called_once_with("test query")
        mock_collection.query.assert_called_once()

        # Verify the right kwargs were passed
        call_kwargs = mock_collection.query.call_args[1]
        self.assertEqual(call_kwargs["n_results"], 5)
        self.assertIn("documents", call_kwargs["include"])
        self.assertIn("metadatas", call_kwargs["include"])
        self.assertIn("distances", call_kwargs["include"])

        self.assertEqual(len(results["ids"][0]), 3)

    @patch("src.retrieval.query.get_or_create_collection")
    @patch("src.retrieval.query.get_embedding_function")
    def test_retrieve_applies_where_filter(self, mock_get_emb, mock_get_col):
        """When a where-filter is supplied, it should be passed to collection.query()."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.0] * 384
        mock_get_emb.return_value = mock_embedding_fn

        mock_collection = MagicMock()
        mock_collection.query.return_value = _mock_query_results(1)
        mock_get_col.return_value = mock_collection

        where = {"source_type": "pdf"}
        retrieve("filtered query", where=where)

        call_kwargs = mock_collection.query.call_args[1]
        self.assertEqual(call_kwargs["where"], {"source_type": "pdf"})

    @patch("src.retrieval.query.get_or_create_collection")
    @patch("src.retrieval.query.get_embedding_function")
    def test_retrieve_no_filter_by_default(self, mock_get_emb, mock_get_col):
        """Without a where-filter, 'where' should not appear in query kwargs."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.0] * 384
        mock_get_emb.return_value = mock_embedding_fn

        mock_collection = MagicMock()
        mock_collection.query.return_value = _mock_query_results(1)
        mock_get_col.return_value = mock_collection

        retrieve("no filter query")

        call_kwargs = mock_collection.query.call_args[1]
        self.assertNotIn("where", call_kwargs)

    @patch("src.retrieval.query.get_or_create_collection")
    @patch("src.retrieval.query.get_embedding_function")
    def test_retrieve_forwards_persist_dir_and_collection_name(
        self, mock_get_emb, mock_get_col
    ):
        """Custom persist_dir and collection_name should reach get_or_create_collection."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.0] * 384
        mock_get_emb.return_value = mock_embedding_fn

        mock_collection = MagicMock()
        mock_collection.query.return_value = _mock_query_results(0)
        mock_get_col.return_value = mock_collection

        retrieve("q", persist_dir="/tmp/db", collection_name="custom_col")

        mock_get_col.assert_called_once_with("/tmp/db", "custom_col")

    @patch("src.retrieval.query.get_or_create_collection")
    @patch("src.retrieval.query.get_embedding_function")
    def test_retrieve_formatted_returns_list_of_dicts(
        self, mock_get_emb, mock_get_col
    ):
        """retrieve_formatted() should return a list of dicts with expected keys."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.0] * 384
        mock_get_emb.return_value = mock_embedding_fn

        mock_collection = MagicMock()
        mock_collection.query.return_value = _mock_query_results(3)
        mock_get_col.return_value = mock_collection

        results = retrieve_formatted("test query")

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)

        for r in results:
            self.assertIn("rank", r)
            self.assertIn("id", r)
            self.assertIn("distance", r)
            self.assertIn("text", r)
            self.assertIn("metadata", r)

    @patch("src.retrieval.query.get_or_create_collection")
    @patch("src.retrieval.query.get_embedding_function")
    def test_retrieve_formatted_ranks_are_sequential(
        self, mock_get_emb, mock_get_col
    ):
        """Ranks should be 1, 2, 3, … matching the order returned by ChromaDB."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.0] * 384
        mock_get_emb.return_value = mock_embedding_fn

        mock_collection = MagicMock()
        mock_collection.query.return_value = _mock_query_results(4)
        mock_get_col.return_value = mock_collection

        results = retrieve_formatted("test query")
        ranks = [r["rank"] for r in results]
        self.assertEqual(ranks, [1, 2, 3, 4])

    @patch("src.retrieval.query.get_or_create_collection")
    @patch("src.retrieval.query.get_embedding_function")
    def test_retrieve_formatted_empty_results(self, mock_get_emb, mock_get_col):
        """An empty result set should return an empty list."""
        mock_embedding_fn = MagicMock()
        mock_embedding_fn.embed_query.return_value = [0.0] * 384
        mock_get_emb.return_value = mock_embedding_fn

        mock_collection = MagicMock()
        mock_collection.query.return_value = _mock_query_results(0)
        mock_get_col.return_value = mock_collection

        results = retrieve_formatted("empty query")
        self.assertEqual(results, [])


# ── Integration Tests (live vector DB) ──────────────────────────────

class TestRetrievalIntegration(unittest.TestCase):
    """Query the live vector DB and verify results are relevant.

    These tests are skipped when the vector DB is not populated.
    """

    @classmethod
    def setUpClass(cls):
        """Load the collection and embedding model once for all tests."""
        if not os.path.isdir(VECTOR_DB_DIR):
            raise unittest.SkipTest(
                f"Vector DB not found at {VECTOR_DB_DIR}. "
                "Run the ingestion pipeline first."
            )

        from src.retrieval.query import retrieve as _retrieve
        # Quick probe to make sure the collection is populated
        try:
            probe = _retrieve("test", n_results=1)
            if not probe["ids"][0]:
                raise unittest.SkipTest("Vector DB collection is empty.")
        except Exception as exc:
            raise unittest.SkipTest(f"Cannot query vector DB: {exc}")

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _sources(results: list[dict]) -> list[str]:
        return [r["metadata"].get("source", "") for r in results]

    @staticmethod
    def _any_source_contains(results: list[dict], sub: str) -> bool:
        return any(sub in s for s in TestRetrievalIntegration._sources(results))

    @staticmethod
    def _any_text_contains(results: list[dict], sub: str) -> bool:
        sub_lower = sub.lower()
        return any(sub_lower in r["text"].lower() for r in results)

    # ── Smoke Queries ───────────────────────────────────────────────

    def test_query_returns_results(self):
        """A simple query should return at least one result."""
        results = retrieve_formatted("test")
        self.assertGreater(len(results), 0)

    # ── Retrieval Quality / Sanity Checks ───────────────────────────

    def test_results_have_metadata(self):
        """Every result should carry source metadata."""
        results = retrieve_formatted("test")
        for r in results:
            self.assertIn("source", r["metadata"], "Chunk missing 'source' metadata")

    def test_distances_are_sorted(self):
        """Results should be sorted by ascending distance (nearest first)."""
        results = retrieve_formatted("test")
        distances = [r["distance"] for r in results]
        self.assertEqual(
            distances,
            sorted(distances),
            "Results should be sorted by ascending distance",
        )

    def test_top_result_distance_is_reasonable(self):
        """The closest match for a domain-specific query should have
        a relatively small distance (< 1.5 for L2 / cosine on MiniLM)."""
        results = retrieve_formatted(
            "severely adverse scenario unemployment rate peak"
        )
        self.assertLess(
            results[0]["distance"],
            1.5,
            f"Top-result distance {results[0]['distance']:.4f} seems too large.",
        )

    def test_no_empty_documents(self):
        """Returned documents should contain actual text, not empty strings."""
        results = retrieve_formatted("test")
        for r in results:
            self.assertTrue(
                len(r["text"].strip()) > 0,
                "Retrieved an empty document chunk",
            )

    def test_metadata_filter_narrows_results(self):
        """Filtering by source_type should only return matching docs."""
        results = retrieve_formatted(
            "test",
            where={"source_type": "html"},
        )
        for r in results:
            if "source_type" in r["metadata"]:
                self.assertEqual(
                    r["metadata"]["source_type"],
                    "html",
                    "Metadata filter did not restrict source_type",
                )


if __name__ == "__main__":
    unittest.main()
