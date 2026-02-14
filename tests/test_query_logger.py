"""Tests for src/retrieval/query_logger.py.

Covers log file creation, content structure, and edge cases.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.query_logger import log_query_session


def _sample_results(n: int = 2) -> list[dict]:
    """Build fake retrieve_formatted() output."""
    return [
        {
            "rank": i + 1,
            "id": f"doc_chunk_{i:04d}",
            "distance": 0.1 * (i + 1),
            "text": f"Full text of chunk {i}.",
            "metadata": {
                "source": f"corpus/raw_data/file_{i}.pdf",
                "title": f"Title {i}",
                "doc_id": f"ID{i}",
                "source_type": "pdf",
            },
        }
        for i in range(n)
    ]


class TestLogQuerySession(unittest.TestCase):
    """Tests for log_query_session()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_creates_log_file(self):
        path = log_query_session(
            query="test query",
            results=_sample_results(),
            logs_dir=self.tmpdir,
        )
        self.assertTrue(os.path.isfile(path))

    def test_filename_format(self):
        path = log_query_session(
            query="test",
            results=_sample_results(),
            logs_dir=self.tmpdir,
        )
        basename = os.path.basename(path)
        # Format: YYYYMMDD_HHMMSS.log
        self.assertRegex(basename, r"^\d{8}_\d{6}\.log$")

    def test_log_contains_query(self):
        path = log_query_session(
            query="What is CCAR?",
            results=_sample_results(),
            logs_dir=self.tmpdir,
        )
        content = open(path, encoding="utf-8").read()
        self.assertIn("What is CCAR?", content)

    def test_log_contains_chunk_text(self):
        results = _sample_results(3)
        path = log_query_session(
            query="test",
            results=results,
            logs_dir=self.tmpdir,
        )
        content = open(path, encoding="utf-8").read()
        for r in results:
            self.assertIn(r["text"], content)
            self.assertIn(r["id"], content)

    def test_log_contains_metadata(self):
        path = log_query_session(
            query="test",
            results=_sample_results(),
            logs_dir=self.tmpdir,
        )
        content = open(path, encoding="utf-8").read()
        self.assertIn("Title 0", content)
        self.assertIn("corpus/raw_data/file_0.pdf", content)

    def test_log_contains_collection_name(self):
        path = log_query_session(
            query="test",
            results=_sample_results(),
            collection_name="stress_test_docs_6k",
            logs_dir=self.tmpdir,
        )
        content = open(path, encoding="utf-8").read()
        self.assertIn("stress_test_docs_6k", content)

    def test_log_without_answer(self):
        path = log_query_session(
            query="test",
            results=_sample_results(),
            answer=None,
            logs_dir=self.tmpdir,
        )
        content = open(path, encoding="utf-8").read()
        self.assertNotIn("LLM ANSWER", content)

    def test_log_with_answer(self):
        path = log_query_session(
            query="test",
            results=_sample_results(),
            answer="The answer is 42.",
            logs_dir=self.tmpdir,
        )
        content = open(path, encoding="utf-8").read()
        self.assertIn("LLM ANSWER", content)
        self.assertIn("The answer is 42.", content)

    def test_empty_results(self):
        path = log_query_session(
            query="nothing",
            results=[],
            logs_dir=self.tmpdir,
        )
        content = open(path, encoding="utf-8").read()
        self.assertIn("Results:    0", content)

    def test_creates_logs_dir_if_missing(self):
        nested = os.path.join(self.tmpdir, "sub", "logs")
        path = log_query_session(
            query="test",
            results=_sample_results(),
            logs_dir=nested,
        )
        self.assertTrue(os.path.isfile(path))

    def test_log_contains_config(self):
        path = log_query_session(
            query="test",
            results=_sample_results(),
            logs_dir=self.tmpdir,
        )
        content = open(path, encoding="utf-8").read()
        self.assertIn("CONFIG", content)
        self.assertIn("chunk_size", content)
        self.assertIn("collection_name", content)


if __name__ == "__main__":
    unittest.main()
