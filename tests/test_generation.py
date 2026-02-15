"""Tests for src/generation/llm.py.

All tests are fast, mocked unit tests — no Ollama server or
embedding model required.

Run with:
    pytest tests/test_generation.py -v
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generation.llm import (
    build_prompt,
    format_context,
    generate_answer,
    get_llm,
    ask,
    RAG_PROMPT,
    SYSTEM_PROMPT,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    PROVIDER_DEFAULTS,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _fake_chunks(n: int = 3) -> list[dict]:
    """Build a list of fake retrieve_formatted() results."""
    return [
        {
            "rank": i + 1,
            "id": f"doc_chunk_{i:04d}",
            "distance": 0.3 + i * 0.05,
            "text": f"Sample text about stress testing topic {i}.",
            "metadata": {
                "source": f"corpus/raw_data/file_{i}.pdf",
                "title": f"Stress Test Document {i}",
                "doc_id": f"ID{i}",
                "source_type": "pdf",
            },
        }
        for i in range(n)
    ]


# ── build_prompt tests ─────────────────────────────────────────────


class TestFormatContext(unittest.TestCase):
    """Tests for the context formatter."""

    def test_formats_chunks_with_source(self):
        """Each chunk is prefixed with [Source: title | path]."""
        result = format_context(_fake_chunks(2))
        self.assertIn("[Source: Stress Test Document 0", result)
        self.assertIn("[Source: Stress Test Document 1", result)

    def test_separates_chunks_with_divider(self):
        """Chunks are joined by --- dividers."""
        result = format_context(_fake_chunks(2))
        self.assertIn("---", result)

    def test_empty_chunks(self):
        """Returns empty string for no chunks."""
        result = format_context([])
        self.assertEqual(result, "")

    def test_missing_metadata_keys(self):
        """Falls back to 'Unknown' when title/source are missing."""
        chunks = [{"rank": 1, "id": "x", "distance": 0.1,
                    "text": "txt", "metadata": {}}]
        result = format_context(chunks)
        self.assertIn("Unknown", result)


class TestRAGPrompt(unittest.TestCase):
    """Tests for the ChatPromptTemplate."""

    def test_format_messages_returns_two(self):
        """RAG_PROMPT produces exactly a system + human message."""
        messages = RAG_PROMPT.format_messages(
            context="some context", question="some question",
        )
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].type, "system")
        self.assertEqual(messages[1].type, "human")

    def test_context_and_question_injected(self):
        """The human message contains the supplied context and question."""
        messages = RAG_PROMPT.format_messages(
            context="CTX_VALUE", question="Q_VALUE",
        )
        self.assertIn("CTX_VALUE", messages[1].content)
        self.assertIn("Q_VALUE", messages[1].content)


class TestBuildPrompt(unittest.TestCase):
    """Tests for the prompt builder."""

    def test_contains_query(self):
        """The prompt includes the user's question."""
        prompt = build_prompt("What is CET1?", _fake_chunks())
        self.assertIn("What is CET1?", prompt)

    def test_contains_chunk_text(self):
        """Each chunk's text appears in the prompt."""
        chunks = _fake_chunks(3)
        prompt = build_prompt("query", chunks)
        for chunk in chunks:
            self.assertIn(chunk["text"], prompt)

    def test_contains_source_metadata(self):
        """Each chunk's title and source appear in the prompt."""
        chunks = _fake_chunks(2)
        prompt = build_prompt("query", chunks)
        for chunk in chunks:
            self.assertIn(chunk["metadata"]["title"], prompt)
            self.assertIn(chunk["metadata"]["source"], prompt)

    def test_has_context_and_question_sections(self):
        """The prompt has CONTEXT, QUESTION, and ANSWER labels."""
        prompt = build_prompt("query", _fake_chunks(1))
        self.assertIn("CONTEXT:", prompt)
        self.assertIn("QUESTION:", prompt)
        self.assertIn("ANSWER:", prompt)

    def test_empty_chunks(self):
        """Prompt still works with zero chunks."""
        prompt = build_prompt("something", [])
        self.assertIn("CONTEXT:", prompt)
        self.assertIn("something", prompt)

    def test_missing_metadata_keys(self):
        """Handles chunks whose metadata lacks title/source."""
        chunks = [
            {
                "rank": 1,
                "id": "x",
                "distance": 0.1,
                "text": "some text",
                "metadata": {},
            }
        ]
        prompt = build_prompt("q", chunks)
        self.assertIn("Unknown", prompt)
        self.assertIn("some text", prompt)


# ── get_llm tests ──────────────────────────────────────────────────


class TestGetLLM(unittest.TestCase):
    """Tests for LLM instantiation."""

    @patch("src.generation.llm.ChatOllama")
    def test_default_params(self, mock_cls):
        """get_llm() passes default model and temperature for ollama."""
        get_llm()
        mock_cls.assert_called_once_with(
            model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE
        )

    @patch("src.generation.llm.ChatOllama")
    def test_custom_params(self, mock_cls):
        """get_llm() forwards custom model and temperature."""
        get_llm(model="phi3", temperature=0.5)
        mock_cls.assert_called_once_with(model="phi3", temperature=0.5)

    def test_unknown_provider_raises(self):
        """get_llm() raises ValueError for unknown providers."""
        with self.assertRaises(ValueError):
            get_llm(provider="nonexistent")


# ── generate_answer tests ──────────────────────────────────────────


class TestGenerateAnswer(unittest.TestCase):
    """Tests for the LLM generation wrapper."""

    @patch("src.generation.llm.get_llm")
    def test_returns_answer_text(self, mock_get_llm):
        """generate_answer() returns the model's response content."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="The answer is 42.")
        mock_get_llm.return_value = mock_llm

        answer = generate_answer("question", _fake_chunks())
        self.assertEqual(answer, "The answer is 42.")

    @patch("src.generation.llm.get_llm")
    def test_passes_system_and_human_messages(self, mock_get_llm):
        """The LLM receives a system message and a human message
        via ChatPromptTemplate.format_messages()."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="ok")
        mock_get_llm.return_value = mock_llm

        generate_answer("question", _fake_chunks())
        messages = mock_llm.invoke.call_args[0][0]
        self.assertEqual(len(messages), 2)
        # ChatPromptTemplate produces BaseMessage objects, not tuples
        self.assertEqual(messages[0].type, "system")
        self.assertIn("precise research assistant", messages[0].content)
        self.assertEqual(messages[1].type, "human")
        self.assertIn("CONTEXT:", messages[1].content)
        self.assertIn("QUESTION:", messages[1].content)

    @patch("src.generation.llm.get_llm")
    def test_connection_error_raised(self, mock_get_llm):
        """A ConnectionError is raised when the LLM server is unreachable."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Connection refused")
        mock_get_llm.return_value = mock_llm

        with self.assertRaises(ConnectionError) as ctx:
            generate_answer("question", _fake_chunks())
        self.assertIn("provider", str(ctx.exception).lower())

    @patch("src.generation.llm.get_llm")
    def test_non_connection_error_propagated(self, mock_get_llm):
        """Non-connection errors propagate unchanged."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ValueError("bad input")
        mock_get_llm.return_value = mock_llm

        with self.assertRaises(ValueError):
            generate_answer("question", _fake_chunks())

    @patch("src.generation.llm.get_llm")
    def test_custom_model_forwarded(self, mock_get_llm):
        """generate_answer() passes the model name to get_llm()."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="ok")
        mock_get_llm.return_value = mock_llm

        generate_answer("q", _fake_chunks(), model="phi3")
        mock_get_llm.assert_called_once_with(
            model="phi3",
            temperature=DEFAULT_TEMPERATURE,
            provider=DEFAULT_PROVIDER,
        )


# ── ask() tests ─────────────────────────────────────────────────────


class TestAsk(unittest.TestCase):
    """Tests for the end-to-end ask() convenience wrapper."""

    @patch("src.generation.llm.generate_answer")
    @patch("src.generation.llm.retrieve_formatted", create=True)
    def test_returns_dict_shape(self, mock_retrieve, mock_generate):
        """ask() returns a dict with query, answer, chunks, model."""
        # Patch the lazy imports inside ask()
        with patch(
            "src.retrieval.query.retrieve_formatted",
            return_value=_fake_chunks(2),
        ), patch(
            "src.generation.llm.generate_answer",
            return_value="The answer.",
        ):
            result = ask("test query")

        self.assertIn("query", result)
        self.assertIn("answer", result)
        self.assertIn("chunks", result)
        self.assertIn("model", result)
        self.assertIn("provider", result)
        self.assertEqual(result["query"], "test query")
        self.assertEqual(result["answer"], "The answer.")
        self.assertEqual(result["model"], DEFAULT_MODEL)
        self.assertEqual(result["provider"], DEFAULT_PROVIDER)

    @patch("src.generation.llm.generate_answer", return_value="ans")
    def test_passes_n_results(self, mock_generate):
        """ask() forwards n_results to retrieve_formatted."""
        with patch(
            "src.retrieval.query.retrieve_formatted",
            return_value=_fake_chunks(3),
        ) as mock_retrieve:
            ask("q", n_results=10)
        mock_retrieve.assert_called_once()
        _, kwargs = mock_retrieve.call_args
        self.assertEqual(kwargs["n_results"], 10)

    @patch("src.generation.llm.generate_answer", return_value="ans")
    def test_passes_where_filter(self, mock_generate):
        """ask() forwards the where filter to retrieve_formatted."""
        where = {"source_type": "pdf"}
        with patch(
            "src.retrieval.query.retrieve_formatted",
            return_value=_fake_chunks(1),
        ) as mock_retrieve:
            ask("q", where=where)
        _, kwargs = mock_retrieve.call_args
        self.assertEqual(kwargs["where"], where)


# ── CLI --answer flag tests ─────────────────────────────────────────


class TestCLIAnswerFlag(unittest.TestCase):
    """Tests for the --answer flag in query.py's CLI."""

    @patch("src.retrieval.query._generate_and_print_answer")
    @patch("src.retrieval.query.retrieve_formatted", return_value=[])
    @patch("src.retrieval.query.print_ascii_banner")
    def test_answer_flag_calls_generate(
        self, mock_banner, mock_retrieve, mock_gen
    ):
        """--answer triggers _generate_and_print_answer()."""
        from src.retrieval.query import main

        main(["test query", "--answer"])
        mock_gen.assert_called_once()

    @patch("src.retrieval.query._generate_and_print_answer")
    @patch("src.retrieval.query.retrieve_formatted", return_value=[])
    @patch("src.retrieval.query.print_ascii_banner")
    def test_no_answer_flag_skips_generate(
        self, mock_banner, mock_retrieve, mock_gen
    ):
        """Without --answer, _generate_and_print_answer() is not called."""
        from src.retrieval.query import main

        main(["test query"])
        mock_gen.assert_not_called()

    @patch("src.retrieval.query._generate_and_print_answer")
    @patch("src.retrieval.query.retrieve_formatted", return_value=[])
    @patch("src.retrieval.query.print_ascii_banner")
    def test_model_flag_forwarded(
        self, mock_banner, mock_retrieve, mock_gen
    ):
        """--model is forwarded to _generate_and_print_answer()."""
        from src.retrieval.query import main

        main(["test query", "--answer", "--model", "phi3"])
        mock_gen.assert_called_once()
        _, kwargs = mock_gen.call_args
        self.assertEqual(kwargs.get("model") or mock_gen.call_args[0][2], "phi3")


if __name__ == "__main__":
    unittest.main()
