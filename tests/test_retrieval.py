"""Tests for vector-DB retrieval quality.

Queries the live ChromaDB collection built by the ingestion pipeline
and checks that the top-k results are relevant — i.e. the correct
source documents surface for a battery of domain-specific questions.

These are *integration* tests: they require the vector DB at
corpus/vector_db/ to be populated (run ``python -m src.ingestion.processor``
first).  They embed test queries at runtime with the same model the
pipeline used, so they exercise the full embed → ANN-search path.

Run with:
    pytest tests/test_retrieval.py -v
"""

import os
import sys
import unittest

import chromadb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embedding.model import (
    get_embedding_function,
    COLLECTION_NAME,
    VECTOR_DB_DIR,
)

# ── Constants ───────────────────────────────────────────────────────
TOP_K = 5  # number of results to retrieve per query


# ── Helpers ─────────────────────────────────────────────────────────

def _get_collection():
    """Return the existing ChromaDB collection (read-only)."""
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    return client.get_collection(name=COLLECTION_NAME)


def _query(collection, embedding_fn, text: str, n_results: int = TOP_K):
    """Embed *text* and query the collection, returning full results."""
    query_embedding = embedding_fn.embed_query(text)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def _sources(results) -> list[str]:
    """Extract the source paths from query results."""
    return [m.get("source", "") for m in results["metadatas"][0]]


def _titles(results) -> list[str]:
    """Extract titles from query results (metadata-enriched chunks)."""
    return [m.get("title", "") for m in results["metadatas"][0]]


def _documents(results) -> list[str]:
    """Extract the text content from query results."""
    return results["documents"][0]


def _any_source_contains(results, substring: str) -> bool:
    """True if any top-k source path contains *substring*."""
    return any(substring in s for s in _sources(results))


def _any_doc_contains(results, substring: str) -> bool:
    """True if any top-k document text contains *substring* (case-insensitive)."""
    sub = substring.lower()
    return any(sub in d.lower() for d in _documents(results))


# ── Test Cases ──────────────────────────────────────────────────────

class TestRetrievalIntegration(unittest.TestCase):
    """Query the live vector DB and verify results are relevant."""

    @classmethod
    def setUpClass(cls):
        """Load the collection and embedding model once for all tests."""
        if not os.path.isdir(VECTOR_DB_DIR):
            raise unittest.SkipTest(
                f"Vector DB not found at {VECTOR_DB_DIR}. "
                "Run the ingestion pipeline first."
            )
        cls.collection = _get_collection()
        if cls.collection.count() == 0:
            raise unittest.SkipTest("Vector DB collection is empty.")
        cls.embedding_fn = get_embedding_function()

    # ── Scenario Data Queries ───────────────────────────────────────

    def test_severely_adverse_unemployment(self):
        """A query about severely adverse unemployment should surface
        the severely adverse domestic scenario data."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "What is the peak unemployment rate in the severely adverse scenario?",
        )
        self.assertTrue(
            _any_source_contains(results, "severely_adverse")
            or _any_doc_contains(results, "unemployment"),
            f"Expected severely adverse scenario data. Got sources: {_sources(results)}",
        )

    def test_baseline_gdp_growth(self):
        """A query about baseline GDP should surface baseline scenario data."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "What is the real GDP growth rate in the baseline scenario?",
        )
        self.assertTrue(
            _any_source_contains(results, "baseline")
            or _any_doc_contains(results, "real gdp growth"),
            f"Expected baseline scenario data. Got sources: {_sources(results)}",
        )

    def test_market_volatility_vix(self):
        """A query about VIX / market volatility should return relevant results."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "Market Volatility Index VIX level under stress",
        )
        self.assertTrue(
            _any_doc_contains(results, "volatility")
            or _any_doc_contains(results, "vix"),
            f"Expected VIX/volatility content. Got docs: {_documents(results)[:2]}",
        )

    def test_house_price_index(self):
        """A query about house prices should surface scenario data with HPI."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "House Price Index decline in the severely adverse scenario",
        )
        self.assertTrue(
            _any_doc_contains(results, "house price")
            or _any_source_contains(results, "severely_adverse"),
            f"Expected house price content. Got sources: {_sources(results)}",
        )

    def test_commercial_real_estate(self):
        """A query about CRE should surface relevant scenario content."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "Commercial Real Estate Price Index stress scenario",
        )
        self.assertTrue(
            _any_doc_contains(results, "commercial real estate")
            or _any_doc_contains(results, "cre"),
            f"Expected CRE content. Got docs: {_documents(results)[:2]}",
        )

    # ── Model Documentation Queries ─────────────────────────────────

    def test_credit_risk_models(self):
        """A query about credit risk models should surface the credit risk PDF."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "credit risk model loan loss estimation methodology",
        )
        self.assertTrue(
            _any_source_contains(results, "credit_risk")
            or _any_doc_contains(results, "credit risk"),
            f"Expected credit risk model docs. Got sources: {_sources(results)}",
        )

    def test_operational_risk(self):
        """A query about operational risk should surface that model doc."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "operational risk model losses",
        )
        self.assertTrue(
            _any_source_contains(results, "operational_risk")
            or _any_doc_contains(results, "operational risk"),
            f"Expected operational risk docs. Got sources: {_sources(results)}",
        )

    def test_ppnr_models(self):
        """A query about PPNR should surface pre-provision net revenue docs."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "pre-provision net revenue PPNR model",
        )
        self.assertTrue(
            _any_source_contains(results, "pre_provision")
            or _any_doc_contains(results, "ppnr")
            or _any_doc_contains(results, "pre-provision net revenue"),
            f"Expected PPNR docs. Got sources: {_sources(results)}",
        )

    def test_market_risk_models(self):
        """A query about market risk should surface market risk model docs."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "market risk trading losses models",
        )
        self.assertTrue(
            _any_source_contains(results, "market_risk")
            or _any_doc_contains(results, "market risk"),
            f"Expected market risk docs. Got sources: {_sources(results)}",
        )

    def test_global_market_shock(self):
        """A query about the GMS component should surface relevant docs."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "global market shock component trading counterparty",
        )
        self.assertTrue(
            _any_doc_contains(results, "global market shock")
            or _any_doc_contains(results, "gms")
            or _any_source_contains(results, "gms"),
            f"Expected GMS content. Got sources: {_sources(results)}",
        )

    # ── Transparency / Policy Queries ───────────────────────────────

    def test_transparency_proposals(self):
        """A query about transparency should surface the Q&A or press release."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "enhanced transparency public accountability stress test proposals",
        )
        self.assertTrue(
            _any_source_contains(results, "transparency")
            or _any_doc_contains(results, "transparency"),
            f"Expected transparency docs. Got sources: {_sources(results)}",
        )

    def test_dodd_frank_act(self):
        """A query about Dodd-Frank should surface DFA-related content."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "Dodd-Frank Act stress test requirements",
        )
        self.assertTrue(
            _any_doc_contains(results, "dodd-frank")
            or _any_source_contains(results, "dfa"),
            f"Expected Dodd-Frank content. Got sources: {_sources(results)}",
        )

    # ── Capital / Financial Metrics Queries ─────────────────────────

    def test_cet1_capital_ratio(self):
        """A query about CET1 should surface capital-related content."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "Common Equity Tier 1 CET1 capital ratio minimum",
        )
        self.assertTrue(
            _any_doc_contains(results, "cet1")
            or _any_doc_contains(results, "common equity tier 1")
            or _any_doc_contains(results, "capital ratio"),
            f"Expected CET1/capital ratio content. Got docs: {_documents(results)[:2]}",
        )

    def test_nine_quarter_paths(self):
        """A query about nine-quarter paths should surface that CSV data."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "detailed nine quarter paths projected losses",
        )
        self.assertTrue(
            _any_source_contains(results, "nine_quarter")
            or _any_doc_contains(results, "nine quarter")
            or _any_doc_contains(results, "nine-quarter"),
            f"Expected nine-quarter paths data. Got sources: {_sources(results)}",
        )

    # ── Interest Rate Queries ───────────────────────────────────────

    def test_treasury_rates(self):
        """A query about Treasury rates should surface scenario data."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "3-month Treasury rate 10-year Treasury yield scenario",
        )
        self.assertTrue(
            _any_doc_contains(results, "treasury")
            or _any_doc_contains(results, "interest rate"),
            f"Expected Treasury rate content. Got docs: {_documents(results)[:2]}",
        )

    def test_bbb_corporate_yield_spread(self):
        """A query about BBB spreads should return relevant results."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "BBB corporate bond yield spread stress scenario",
        )
        self.assertTrue(
            _any_doc_contains(results, "bbb")
            or _any_doc_contains(results, "corporate yield")
            or _any_doc_contains(results, "spread"),
            f"Expected BBB spread content. Got docs: {_documents(results)[:2]}",
        )

    # ── Retrieval Quality / Sanity Checks ───────────────────────────

    def test_results_have_metadata(self):
        """Every result should carry source metadata."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "stress test",
        )
        for meta in results["metadatas"][0]:
            self.assertIn("source", meta, "Chunk missing 'source' metadata")

    def test_distances_are_sorted(self):
        """ChromaDB should return results sorted by ascending distance."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "unemployment rate severely adverse",
        )
        distances = results["distances"][0]
        self.assertEqual(
            distances,
            sorted(distances),
            "Results should be sorted by ascending distance (nearest first)",
        )

    def test_top_result_distance_is_reasonable(self):
        """The closest match for a domain-specific query should have
        a relatively small distance (< 1.5 for L2 / cosine on MiniLM)."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "severely adverse scenario unemployment rate peak",
        )
        best_distance = results["distances"][0][0]
        self.assertLess(
            best_distance,
            1.5,
            f"Top-result distance {best_distance:.4f} seems too large — "
            "model may not be finding relevant content.",
        )

    def test_no_empty_documents(self):
        """Returned documents should contain actual text, not empty strings."""
        results = _query(
            self.collection,
            self.embedding_fn,
            "stress test capital requirements",
        )
        for doc in _documents(results):
            self.assertTrue(
                len(doc.strip()) > 0,
                "Retrieved an empty document chunk",
            )


if __name__ == "__main__":
    unittest.main()
