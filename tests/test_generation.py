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
    format_context,
    generate_answer,
    get_llm,
    rag_chain,
    stream_answer,
    RAG_PROMPT,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _fake_chunks(n: int = 3) -> list[dict]:
    """Build a list of fake retrieve_formatted() results."""
    return [
        {
            "rank": i + 1,
            "id": f"doc_chunk_{i:04d}",
            "distance": 0.3 + i * 0.05,
            "text": f"Sample text about topic {i}.",
            "metadata": {
                "source": f"corpus/raw_data/file_{i}.pdf",
                "title": f"Sample Document {i}",
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
        self.assertIn("[Source: Sample Document 0", result)
        self.assertIn("[Source: Sample Document 1", result)

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
        chunks = [
            {"rank": 1, "id": "x", "distance": 0.1, "text": "txt", "metadata": {}}
        ]
        result = format_context(chunks)
        self.assertIn("Unknown", result)


class TestRAGPrompt(unittest.TestCase):
    """Tests for the ChatPromptTemplate."""

    def test_format_messages_returns_two(self):
        """RAG_PROMPT produces exactly a system + human message."""
        messages = RAG_PROMPT.format_messages(
            context="some context",
            question="some question",
        )
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].type, "system")
        self.assertEqual(messages[1].type, "human")

    def test_context_and_question_injected(self):
        """The human message contains the supplied context and question."""
        messages = RAG_PROMPT.format_messages(
            context="CTX_VALUE",
            question="Q_VALUE",
        )
        self.assertIn("CTX_VALUE", messages[1].content)
        self.assertIn("Q_VALUE", messages[1].content)


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
    """Tests for the LLM generation wrapper (uses LCEL chain)."""

    @patch("src.generation.llm.rag_chain")
    def test_returns_answer_text(self, mock_rag_chain):
        """generate_answer() returns the chain's output."""
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = "The answer is 42."
        mock_rag_chain.return_value = mock_runnable

        answer = generate_answer("question", _fake_chunks())
        self.assertEqual(answer, "The answer is 42.")

    def test_chain_receives_system_and_human_messages(self):
        """The RAG_PROMPT formats system + human messages correctly."""
        # Test the prompt template directly (no LLM needed)
        messages = RAG_PROMPT.format_messages(
            context="some context",
            question="some question",
        )
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].type, "system")
        self.assertIn("precise research assistant", messages[0].content)
        self.assertEqual(messages[1].type, "human")
        self.assertIn("CONTEXT:", messages[1].content)
        self.assertIn("QUESTION:", messages[1].content)

    @patch("src.generation.llm.rag_chain")
    def test_connection_error_raised(self, mock_rag_chain):
        """A ConnectionError is raised when the LLM server is unreachable."""
        mock_runnable = MagicMock()
        mock_runnable.invoke.side_effect = Exception("Connection refused")
        mock_rag_chain.return_value = mock_runnable

        with self.assertRaises(ConnectionError) as ctx:
            generate_answer("question", _fake_chunks())
        self.assertIn("provider", str(ctx.exception).lower())

    @patch("src.generation.llm.rag_chain")
    def test_non_connection_error_propagated(self, mock_rag_chain):
        """Non-connection errors propagate unchanged."""
        mock_runnable = MagicMock()
        mock_runnable.invoke.side_effect = ValueError("bad input")
        mock_rag_chain.return_value = mock_runnable

        with self.assertRaises(ValueError):
            generate_answer("question", _fake_chunks())

    @patch("src.generation.llm.rag_chain")
    def test_custom_model_forwarded(self, mock_rag_chain):
        """generate_answer() passes the model name to rag_chain()."""
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = "ok"
        mock_rag_chain.return_value = mock_runnable

        generate_answer("q", _fake_chunks(), model="phi3")
        mock_rag_chain.assert_called_once_with(
            model="phi3",
            temperature=DEFAULT_TEMPERATURE,
            provider=DEFAULT_PROVIDER,
        )

    @patch("src.generation.llm.rag_chain")
    def test_invoke_receives_context_and_question(self, mock_rag_chain):
        """The chain is invoked with context and question keys."""
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = "answer"
        mock_rag_chain.return_value = mock_runnable

        generate_answer("What is CET1?", _fake_chunks())
        call_args = mock_runnable.invoke.call_args[0][0]
        self.assertIn("context", call_args)
        self.assertIn("question", call_args)
        self.assertEqual(call_args["question"], "What is CET1?")


# ── ask() tests ─────────────────────────────────────────────────────


# ── CLI --answer flag tests ─────────────────────────────────────────


# ── format_docs tests ──────────────────────────────────────────────


# ── rag_chain tests ─────────────────────────────────────────────────


class TestRagChain(unittest.TestCase):
    """Tests for the LCEL chain builder."""

    @patch("src.generation.llm.get_llm")
    def test_chain_is_runnable(self, mock_get_llm):
        """rag_chain() returns an object with .invoke() and .stream() methods."""
        from langchain_core.messages import AIMessage

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="test")
        # The | operator needs the mock to behave like a Runnable
        mock_get_llm.return_value = mock_llm

        chain = rag_chain()
        self.assertTrue(hasattr(chain, "invoke"))
        self.assertTrue(hasattr(chain, "stream"))

    @patch("src.generation.llm.rag_chain")
    def test_chain_invoke_returns_string(self, mock_rag_chain_fn):
        """generate_answer() returns a string via the LCEL chain."""
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = "The answer."
        mock_rag_chain_fn.return_value = mock_runnable

        result = generate_answer("q?", _fake_chunks())
        self.assertIsInstance(result, str)
        self.assertEqual(result, "The answer.")

    @patch("src.generation.llm.get_llm")
    def test_chain_passes_provider(self, mock_get_llm):
        """rag_chain() forwards provider to get_llm()."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        rag_chain(provider="openai", model="gpt-4o")
        mock_get_llm.assert_called_once_with(
            model="gpt-4o",
            temperature=DEFAULT_TEMPERATURE,
            provider="openai",
        )


# ── stream_answer tests ────────────────────────────────────────────


class TestStreamAnswer(unittest.TestCase):
    """Tests for the streaming answer generator."""

    @patch("src.generation.llm.rag_chain")
    def test_yields_tokens(self, mock_rag_chain):
        """stream_answer() yields string tokens from the chain."""
        mock_runnable = MagicMock()
        mock_runnable.stream.return_value = iter(["Hello", " world"])
        mock_rag_chain.return_value = mock_runnable

        tokens = list(stream_answer("question", _fake_chunks()))
        self.assertEqual(tokens, ["Hello", " world"])

    @patch("src.generation.llm.rag_chain")
    def test_stream_is_iterator(self, mock_rag_chain):
        """stream_answer() returns an iterator."""
        mock_runnable = MagicMock()
        mock_runnable.stream.return_value = iter([])
        mock_rag_chain.return_value = mock_runnable

        result = stream_answer("q", _fake_chunks())
        self.assertTrue(hasattr(result, "__iter__"))
        self.assertTrue(hasattr(result, "__next__"))

    @patch("src.generation.llm.rag_chain")
    def test_stream_passes_context_and_question(self, mock_rag_chain):
        """stream_answer() passes context and question to chain.stream()."""
        mock_runnable = MagicMock()
        mock_runnable.stream.return_value = iter(["ans"])
        mock_rag_chain.return_value = mock_runnable

        list(stream_answer("What is CET1?", _fake_chunks()))
        call_args = mock_runnable.stream.call_args[0][0]
        self.assertIn("context", call_args)
        self.assertIn("question", call_args)
        self.assertEqual(call_args["question"], "What is CET1?")


if __name__ == "__main__":
    unittest.main()
