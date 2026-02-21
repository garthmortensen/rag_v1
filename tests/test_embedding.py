"""Tests for src/embedding/model.py.

Covers chunk ID generation (both from metadata CSV and hash fallback),
metadata enrichment, batch upsert logic, and edge cases.
"""

import hashlib
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embedding import model


# ── Helpers ─────────────────────────────────────────────────────────

def _make_chunk(page_content: str, source: str, **extra_meta):
    """Create a minimal LangChain-style Document mock."""
    doc = MagicMock()
    doc.page_content = page_content
    doc.metadata = {"source": source, **extra_meta}
    return doc


SAMPLE_CSV_ROWS = [
    {
        "doc_id": "1JA8WZFYSY0",
        "source_type": "pdf",
        "source_url": "https://example.com/credit_risk.pdf",
        "local_path": "corpus/raw_data/credit_risk_models.pdf",
        "title": "Credit Risk Models",
        "category": "Sample Category",
        "source_org": "Example Org",
        "author": "example.com",
    },
    {
        "doc_id": "1JA8WZ0TJT1",
        "source_type": "html",
        "source_url": "https://example.com/transparency.html",
        "local_path": "corpus/raw_data/transparency_qas.html",
        "title": "Transparency Q&As",
        "category": "General",
        "source_org": "Example Org",
        "author": "example.com",
    },
]


# ── Test Cases ──────────────────────────────────────────────────────

class TestLoadDocIdMap(unittest.TestCase):
    """Tests for load_doc_id_map()."""

    @patch("src.embedding.model.os.path.exists", return_value=False)
    def test_returns_empty_when_csv_missing(self, _mock_exists):
        result = model.load_doc_id_map("nonexistent.csv")
        self.assertEqual(result, {})

    @patch("src.embedding.model.os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.embedding.model.csv.DictReader")
    def test_loads_csv_rows_keyed_by_local_path(
        self, mock_reader, _mock_file, _mock_exists
    ):
        mock_reader.return_value = iter(SAMPLE_CSV_ROWS)
        result = model.load_doc_id_map("corpus/metadata.csv")

        self.assertEqual(len(result), 2)
        self.assertIn("corpus/raw_data/credit_risk_models.pdf", result)
        self.assertEqual(
            result["corpus/raw_data/credit_risk_models.pdf"]["doc_id"],
            "1JA8WZFYSY0",
        )


class TestFallbackDocId(unittest.TestCase):
    """Tests for _fallback_doc_id()."""

    def test_returns_11_char_hex_prefix(self):
        result = model._fallback_doc_id("corpus/raw_data/mystery.txt")
        self.assertEqual(len(result), 11)

    def test_deterministic(self):
        a = model._fallback_doc_id("same/path.pdf")
        b = model._fallback_doc_id("same/path.pdf")
        self.assertEqual(a, b)

    def test_matches_sha256_prefix(self):
        source = "corpus/raw_data/test.pdf"
        expected = hashlib.sha256(source.encode()).hexdigest()[:11]
        self.assertEqual(model._fallback_doc_id(source), expected)


class TestGetEmbeddingFunction(unittest.TestCase):
    """Tests for get_embedding_function()."""

    @patch("src.embedding.model.HuggingFaceEmbeddings")
    def test_creates_huggingface_embeddings(self, mock_hfe):
        model.get_embedding_function("test-model")
        mock_hfe.assert_called_once_with(model_name="test-model")

    @patch("src.embedding.model.HuggingFaceEmbeddings")
    @patch("src.embedding.model.select_best_model", return_value={"name": "all-MiniLM-L6-v2"})
    def test_default_model_auto_selects(self, mock_select, mock_hfe):
        model.get_embedding_function()
        mock_select.assert_called_once()
        mock_hfe.assert_called_once_with(model_name="all-MiniLM-L6-v2")


class TestGetOrCreateCollection(unittest.TestCase):
    """Tests for get_or_create_collection()."""

    @patch("src.embedding.model.chromadb.PersistentClient")
    def test_creates_persistent_client_and_collection(self, mock_client_cls):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        result = model.get_or_create_collection("/tmp/db", "test_col")

        mock_client_cls.assert_called_once_with(path="/tmp/db")
        mock_client.get_or_create_collection.assert_called_once_with(name="test_col")
        self.assertEqual(result, mock_collection)


class TestEmbedAndStore(unittest.TestCase):
    """Tests for embed_and_store()."""

    def _setup_mocks(self):
        """Create common mock objects for embed_and_store tests."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        mock_embedding_fn = MagicMock()
        # Return 384-dim zero vectors (one per chunk)
        mock_embedding_fn.embed_documents.return_value = [[0.0] * 384]

        return mock_collection, mock_embedding_fn

    @patch("src.embedding.model.get_or_create_collection")
    @patch("src.embedding.model.get_embedding_function")
    @patch("src.embedding.model.load_doc_id_map")
    def test_chunk_ids_use_doc_id_from_csv(
        self, mock_load_map, mock_get_emb, mock_get_col
    ):
        """Chunks whose source matches metadata.csv get {doc_id}_chunk_{n} IDs."""
        mock_collection, mock_embedding_fn = self._setup_mocks()
        mock_get_col.return_value = mock_collection
        mock_get_emb.return_value = mock_embedding_fn

        mock_load_map.return_value = {
            "corpus/raw_data/credit_risk_models.pdf": SAMPLE_CSV_ROWS[0],
        }

        chunks = [
            _make_chunk("text A", "corpus/raw_data/credit_risk_models.pdf", page=0),
        ]

        model.embed_and_store(chunks, batch_size=500)

        call_args = mock_collection.upsert.call_args
        ids = call_args[1]["ids"] if "ids" in call_args[1] else call_args[0][0]
        self.assertEqual(ids, ["1JA8WZFYSY0_chunk_0000"])

    @patch("src.embedding.model.get_or_create_collection")
    @patch("src.embedding.model.get_embedding_function")
    @patch("src.embedding.model.load_doc_id_map")
    def test_chunk_ids_use_hash_fallback(
        self, mock_load_map, mock_get_emb, mock_get_col
    ):
        """Chunks with no CSV match get a hash-based fallback ID."""
        mock_collection, mock_embedding_fn = self._setup_mocks()
        mock_get_col.return_value = mock_collection
        mock_get_emb.return_value = mock_embedding_fn
        mock_load_map.return_value = {}  # no CSV entries

        source = "corpus/raw_data/mystery.txt"
        chunks = [_make_chunk("text X", source)]

        model.embed_and_store(chunks, batch_size=500)

        expected_prefix = hashlib.sha256(source.encode()).hexdigest()[:11]
        call_args = mock_collection.upsert.call_args
        ids = call_args[1]["ids"] if "ids" in call_args[1] else call_args[0][0]
        self.assertEqual(ids, [f"{expected_prefix}_chunk_0000"])

    @patch("src.embedding.model.get_or_create_collection")
    @patch("src.embedding.model.get_embedding_function")
    @patch("src.embedding.model.load_doc_id_map")
    def test_sequential_chunk_numbering_per_source(
        self, mock_load_map, mock_get_emb, mock_get_col
    ):
        """Chunk numbers increment per source file, zero-padded to 4 digits."""
        mock_collection, mock_embedding_fn = self._setup_mocks()
        mock_embedding_fn.embed_documents.return_value = [[0.0] * 384] * 3
        mock_get_col.return_value = mock_collection
        mock_get_emb.return_value = mock_embedding_fn

        mock_load_map.return_value = {
            "corpus/raw_data/a.pdf": {"doc_id": "AAA", "title": "", "category": "", "source_org": "", "author": "", "source_url": "", "source_type": "pdf"},
            "corpus/raw_data/b.html": {"doc_id": "BBB", "title": "", "category": "", "source_org": "", "author": "", "source_url": "", "source_type": "html"},
        }

        chunks = [
            _make_chunk("chunk 1", "corpus/raw_data/a.pdf"),
            _make_chunk("chunk 2", "corpus/raw_data/a.pdf"),
            _make_chunk("chunk 3", "corpus/raw_data/b.html"),
        ]

        model.embed_and_store(chunks, batch_size=500)

        call_args = mock_collection.upsert.call_args
        ids = call_args[1]["ids"] if "ids" in call_args[1] else call_args[0][0]
        self.assertEqual(ids, ["AAA_chunk_0000", "AAA_chunk_0001", "BBB_chunk_0000"])

    @patch("src.embedding.model.get_or_create_collection")
    @patch("src.embedding.model.get_embedding_function")
    @patch("src.embedding.model.load_doc_id_map")
    def test_metadata_enriched_from_csv(
        self, mock_load_map, mock_get_emb, mock_get_col
    ):
        """Chunk metadata includes doc_id, title, category, author, source_url, source_type from CSV."""
        mock_collection, mock_embedding_fn = self._setup_mocks()
        mock_get_col.return_value = mock_collection
        mock_get_emb.return_value = mock_embedding_fn

        mock_load_map.return_value = {
            "corpus/raw_data/credit_risk_models.pdf": SAMPLE_CSV_ROWS[0],
        }

        chunks = [
            _make_chunk("text", "corpus/raw_data/credit_risk_models.pdf", page=5),
        ]

        model.embed_and_store(chunks, batch_size=500)

        call_args = mock_collection.upsert.call_args
        metadatas = call_args[1].get("metadatas") or call_args[0][3]
        meta = metadatas[0]

        # Original chunk metadata preserved
        self.assertEqual(meta["source"], "corpus/raw_data/credit_risk_models.pdf")
        self.assertEqual(meta["page"], 5)

        # Enriched fields from CSV
        self.assertEqual(meta["doc_id"], "1JA8WZFYSY0")
        self.assertEqual(meta["title"], "Credit Risk Models")
        self.assertEqual(meta["category"], "Sample Category")
        self.assertEqual(meta["source_org"], "Example Org")
        self.assertEqual(meta["author"], "example.com")
        self.assertEqual(meta["source_url"], "https://example.com/credit_risk.pdf")
        self.assertEqual(meta["source_type"], "pdf")

    @patch("src.embedding.model.get_or_create_collection")
    @patch("src.embedding.model.get_embedding_function")
    @patch("src.embedding.model.load_doc_id_map")
    def test_metadata_not_enriched_for_unknown_source(
        self, mock_load_map, mock_get_emb, mock_get_col
    ):
        """Chunks with no CSV match keep only their original metadata."""
        mock_collection, mock_embedding_fn = self._setup_mocks()
        mock_get_col.return_value = mock_collection
        mock_get_emb.return_value = mock_embedding_fn
        mock_load_map.return_value = {}

        chunks = [_make_chunk("text", "corpus/raw_data/mystery.txt")]

        model.embed_and_store(chunks, batch_size=500)

        call_args = mock_collection.upsert.call_args
        metadatas = call_args[1].get("metadatas") or call_args[0][3]
        meta = metadatas[0]

        self.assertEqual(meta["source"], "corpus/raw_data/mystery.txt")
        self.assertNotIn("doc_id", meta)
        self.assertNotIn("title", meta)

    @patch("src.embedding.model.get_or_create_collection")
    @patch("src.embedding.model.get_embedding_function")
    @patch("src.embedding.model.load_doc_id_map")
    def test_batching_splits_upserts(
        self, mock_load_map, mock_get_emb, mock_get_col
    ):
        """With batch_size=2 and 5 chunks, upsert is called 3 times (2+2+1)."""
        mock_collection, mock_embedding_fn = self._setup_mocks()
        mock_embedding_fn.embed_documents.return_value = [[0.0] * 384] * 5
        mock_get_col.return_value = mock_collection
        mock_get_emb.return_value = mock_embedding_fn
        mock_load_map.return_value = {}

        chunks = [_make_chunk(f"text {i}", "corpus/raw_data/a.pdf") for i in range(5)]

        model.embed_and_store(chunks, batch_size=2)

        self.assertEqual(mock_collection.upsert.call_count, 3)

        # Verify batch sizes: 2, 2, 1
        batch_sizes = [
            len(call.kwargs.get("ids") or call.args[0])
            for call in mock_collection.upsert.call_args_list
        ]
        self.assertEqual(batch_sizes, [2, 2, 1])

    @patch("src.embedding.model.get_or_create_collection")
    @patch("src.embedding.model.get_embedding_function")
    @patch("src.embedding.model.load_doc_id_map")
    def test_empty_chunks_returns_zero(
        self, mock_load_map, mock_get_emb, mock_get_col
    ):
        """embed_and_store([]) returns 0 and doesn't touch ChromaDB."""
        result = model.embed_and_store([])
        self.assertEqual(result, 0)
        mock_get_col.assert_not_called()

    @patch("src.embedding.model.get_or_create_collection")
    @patch("src.embedding.model.get_embedding_function")
    @patch("src.embedding.model.load_doc_id_map")
    def test_returns_total_chunk_count(
        self, mock_load_map, mock_get_emb, mock_get_col
    ):
        """embed_and_store returns the total number of chunks upserted."""
        mock_collection, mock_embedding_fn = self._setup_mocks()
        mock_embedding_fn.embed_documents.return_value = [[0.0] * 384] * 3
        mock_get_col.return_value = mock_collection
        mock_get_emb.return_value = mock_embedding_fn
        mock_load_map.return_value = {}

        chunks = [_make_chunk(f"text {i}", "corpus/raw_data/a.pdf") for i in range(3)]

        result = model.embed_and_store(chunks)
        self.assertEqual(result, 3)


if __name__ == "__main__":
    unittest.main()
