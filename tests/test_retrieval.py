"""Tests for src/retrieval/query.py.

Two test classes:

1.  **TestRetrievalUnit** — fast, mocked unit tests that verify
    retrieve() and retrieve_formatted() call the right things
    with the right arguments, without touching ChromaDB or
    downloading an embedding model.

2.  **TestRetrievalIntegration** — slower integration tests that
    query the *live* vector DB at corpus/vector_db/ and verify
    that domain-specific queries surface the expected documents.
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

    # ── Scenario Data Queries ───────────────────────────────────────

    def test_severely_adverse_unemployment(self):
        """A query about severely adverse unemployment should surface
        the severely adverse domestic scenario data."""
        results = retrieve_formatted(
            "What is the peak unemployment rate in the severely adverse scenario?"
        )
        self.assertTrue(
            self._any_source_contains(results, "severely_adverse")
            or self._any_text_contains(results, "unemployment"),
            f"Expected severely adverse scenario data. "
            f"Got sources: {self._sources(results)}",
        )

    def test_baseline_gdp_growth(self):
        """A query about baseline GDP should surface baseline scenario data."""
        results = retrieve_formatted(
            "What is the real GDP growth rate in the baseline scenario?"
        )
        self.assertTrue(
            self._any_source_contains(results, "baseline")
            or self._any_text_contains(results, "real gdp growth"),
            f"Expected baseline scenario data. "
            f"Got sources: {self._sources(results)}",
        )

    def test_market_volatility_vix(self):
        """A query about VIX / market volatility should return relevant results."""
        results = retrieve_formatted(
            "Market Volatility Index VIX level under stress"
        )
        self.assertTrue(
            self._any_text_contains(results, "volatility")
            or self._any_text_contains(results, "vix"),
            f"Expected VIX/volatility content.",
        )

    def test_house_price_index(self):
        """A query about house prices should surface scenario data with HPI."""
        results = retrieve_formatted(
            "House Price Index decline in the severely adverse scenario"
        )
        self.assertTrue(
            self._any_text_contains(results, "house price")
            or self._any_source_contains(results, "severely_adverse"),
            f"Expected house price content. "
            f"Got sources: {self._sources(results)}",
        )

    def test_commercial_real_estate(self):
        """A query about CRE should surface relevant scenario content."""
        results = retrieve_formatted(
            "Commercial Real Estate Price Index stress scenario"
        )
        self.assertTrue(
            self._any_text_contains(results, "commercial real estate")
            or self._any_text_contains(results, "cre"),
        )

    # ── Model Documentation Queries ─────────────────────────────────

    def test_credit_risk_models(self):
        """A query about credit risk models should surface the credit risk PDF."""
        results = retrieve_formatted(
            "credit risk model loan loss estimation methodology"
        )
        self.assertTrue(
            self._any_source_contains(results, "credit_risk")
            or self._any_text_contains(results, "credit risk"),
        )

    def test_operational_risk(self):
        """A query about operational risk should surface that model doc."""
        results = retrieve_formatted("operational risk model losses")
        self.assertTrue(
            self._any_source_contains(results, "operational_risk")
            or self._any_text_contains(results, "operational risk"),
        )

    def test_ppnr_models(self):
        """A query about PPNR should surface pre-provision net revenue docs."""
        results = retrieve_formatted("pre-provision net revenue PPNR model")
        self.assertTrue(
            self._any_source_contains(results, "pre_provision")
            or self._any_text_contains(results, "ppnr")
            or self._any_text_contains(results, "pre-provision net revenue"),
        )

    def test_market_risk_models(self):
        """A query about market risk should surface market risk model docs."""
        results = retrieve_formatted("market risk trading losses models")
        self.assertTrue(
            self._any_source_contains(results, "market_risk")
            or self._any_text_contains(results, "market risk"),
        )

    def test_global_market_shock(self):
        """A query about the GMS component should surface relevant docs."""
        results = retrieve_formatted(
            "global market shock component trading counterparty"
        )
        self.assertTrue(
            self._any_text_contains(results, "global market shock")
            or self._any_text_contains(results, "gms")
            or self._any_source_contains(results, "gms"),
        )

    # ── Transparency / Policy Queries ───────────────────────────────

    def test_transparency_proposals(self):
        """A query about transparency should surface the Q&A or press release."""
        results = retrieve_formatted(
            "enhanced transparency public accountability stress test proposals"
        )
        self.assertTrue(
            self._any_source_contains(results, "transparency")
            or self._any_text_contains(results, "transparency"),
        )

    def test_dodd_frank_act(self):
        """A query about Dodd-Frank should surface DFA-related content."""
        results = retrieve_formatted("Dodd-Frank Act stress test requirements")
        self.assertTrue(
            self._any_text_contains(results, "dodd-frank")
            or self._any_source_contains(results, "dfa"),
        )

    # ── Capital / Financial Metrics Queries ─────────────────────────

    def test_cet1_capital_ratio(self):
        """A query about CET1 should surface capital-related content."""
        results = retrieve_formatted(
            "Common Equity Tier 1 CET1 capital ratio minimum"
        )
        self.assertTrue(
            self._any_text_contains(results, "cet1")
            or self._any_text_contains(results, "common equity tier 1")
            or self._any_text_contains(results, "capital ratio"),
        )

    def test_nine_quarter_paths(self):
        """A query about nine-quarter paths should surface that CSV data."""
        results = retrieve_formatted(
            "detailed nine quarter paths projected losses"
        )
        self.assertTrue(
            self._any_source_contains(results, "nine_quarter")
            or self._any_text_contains(results, "nine quarter")
            or self._any_text_contains(results, "nine-quarter"),
        )

    # ── Interest Rate Queries ───────────────────────────────────────

    def test_treasury_rates(self):
        """A query about Treasury rates should surface scenario data."""
        results = retrieve_formatted(
            "3-month Treasury rate 10-year Treasury yield scenario"
        )
        self.assertTrue(
            self._any_text_contains(results, "treasury")
            or self._any_text_contains(results, "interest rate"),
        )

    def test_bbb_corporate_yield_spread(self):
        """A query about BBB spreads should return relevant results."""
        results = retrieve_formatted(
            "BBB corporate bond yield spread stress scenario"
        )
        self.assertTrue(
            self._any_text_contains(results, "bbb")
            or self._any_text_contains(results, "corporate yield")
            or self._any_text_contains(results, "spread"),
        )

    # ── Retrieval Quality / Sanity Checks ───────────────────────────

    def test_results_have_metadata(self):
        """Every result should carry source metadata."""
        results = retrieve_formatted("stress test")
        for r in results:
            self.assertIn("source", r["metadata"], "Chunk missing 'source' metadata")

    def test_distances_are_sorted(self):
        """Results should be sorted by ascending distance (nearest first)."""
        results = retrieve_formatted("unemployment rate severely adverse")
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
        results = retrieve_formatted("stress test capital requirements")
        for r in results:
            self.assertTrue(
                len(r["text"].strip()) > 0,
                "Retrieved an empty document chunk",
            )

    def test_metadata_filter_narrows_results(self):
        """Filtering by source_type should only return matching docs."""
        results = retrieve_formatted(
            "stress test",
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
