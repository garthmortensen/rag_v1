"""Tests for src/retrieval/query_logger.py.

Covers log file creation, content structure, and edge cases.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.query_logger import log_query_session, log_rewrite_session, RewriteSectionLog


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


# ── log_rewrite_session tests ──────────────────────────────────────


def _sample_section_logs(n: int = 2) -> list[RewriteSectionLog]:
    """Build fake rewrite section logs."""
    return [
        RewriteSectionLog(
            section=f"Section {i}",
            subsection=f"Subsection {i}",
            source_chars=1000 * (i + 1),
            rewritten_chars=800 * (i + 1),
            rewritten_text=f"Rewritten text for section {i}.",
        )
        for i in range(n)
    ]


class TestLogRewriteSession(unittest.TestCase):
    """Tests for log_rewrite_session()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_creates_log_file(self):
        path = log_rewrite_session(
            pdf_path="test.pdf",
            mode="rewrite",
            provider="ollama",
            model="llama3.2:3b",
            temperature=0.3,
            logs_dir=self.tmpdir,
        )
        self.assertTrue(os.path.isfile(path))

    def test_filename_has_rewrite_suffix(self):
        path = log_rewrite_session(
            pdf_path="test.pdf",
            mode="rewrite",
            provider="ollama",
            model="llama3.2:3b",
            temperature=0.3,
            logs_dir=self.tmpdir,
        )
        basename = os.path.basename(path)
        self.assertRegex(basename, r"^\d{8}_\d{6}_rewrite\.log$")

    def test_log_contains_parameters(self):
        path = log_rewrite_session(
            pdf_path="corpus/raw_data/credit_risk.pdf",
            mode="summarize",
            provider="openai",
            model="gpt-5.2",
            temperature=0.7,
            logs_dir=self.tmpdir,
        )
        content = open(path, encoding="utf-8").read()
        self.assertIn("credit_risk.pdf", content)
        self.assertIn("summarize", content)
        self.assertIn("openai", content)
        self.assertIn("gpt-5.2", content)
        self.assertIn("0.7", content)

    def test_log_contains_custom_prompt(self):
        path = log_rewrite_session(
            pdf_path="test.pdf",
            mode="rewrite",
            provider="ollama",
            model="test",
            temperature=0.3,
            custom_prompt="Write at an elementary-school reading level.",
            logs_dir=self.tmpdir,
        )
        content = open(path, encoding="utf-8").read()
        self.assertIn("elementary-school", content)
        self.assertIn("Custom Prompt", content)

    def test_log_no_custom_prompt(self):
        path = log_rewrite_session(
            pdf_path="test.pdf",
            mode="rewrite",
            provider="ollama",
            model="test",
            temperature=0.3,
            logs_dir=self.tmpdir,
        )
        content = open(path, encoding="utf-8").read()
        self.assertIn("Custom Prompt: (none)", content)

    def test_log_contains_section_details(self):
        sections = _sample_section_logs(3)
        path = log_rewrite_session(
            pdf_path="test.pdf",
            mode="rewrite",
            provider="ollama",
            model="test",
            temperature=0.3,
            sections=sections,
            logs_dir=self.tmpdir,
        )
        content = open(path, encoding="utf-8").read()
        self.assertIn("[1/3]", content)
        self.assertIn("[3/3]", content)
        self.assertIn("Section 0", content)
        self.assertIn("Subsection 1", content)
        self.assertIn("Rewritten text for section 0.", content)

    def test_log_contains_output_path(self):
        path = log_rewrite_session(
            pdf_path="test.pdf",
            mode="rewrite",
            provider="ollama",
            model="test",
            temperature=0.3,
            output_path="/tmp/output/test_rewrite_20260215.md",
            logs_dir=self.tmpdir,
        )
        content = open(path, encoding="utf-8").read()
        self.assertIn("test_rewrite_20260215.md", content)

    def test_log_contains_elapsed_time(self):
        path = log_rewrite_session(
            pdf_path="test.pdf",
            mode="rewrite",
            provider="ollama",
            model="test",
            temperature=0.3,
            elapsed_seconds=125.7,
            logs_dir=self.tmpdir,
        )
        content = open(path, encoding="utf-8").read()
        self.assertIn("125.7s", content)
        self.assertIn("02:05", content)

    def test_log_contains_config(self):
        path = log_rewrite_session(
            pdf_path="test.pdf",
            mode="rewrite",
            provider="ollama",
            model="test",
            temperature=0.3,
            logs_dir=self.tmpdir,
        )
        content = open(path, encoding="utf-8").read()
        self.assertIn("CONFIG", content)
        self.assertIn("chunk_size", content)


if __name__ == "__main__":
    unittest.main()
