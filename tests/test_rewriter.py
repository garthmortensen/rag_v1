"""Tests for src/generation/rewriter.py.

All tests are fast, mocked unit tests — no LLM, no PDF files,
no network access required.

Run with:
    pytest tests/test_rewriter.py -v
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document

from src.generation.rewriter import (
    rewrite_pdf,
    _rewrite_chunk,
    _invoke_llm,
    _split_on_paragraphs,
    _build_prompt,
    _split_markdown_sections,
    discover_rewrite_outputs,
    refine_markdown_iter,
    MAX_CHUNK_CHARS,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _mock_llm(response_text: str = "Rewritten text.") -> MagicMock:
    """Create a mock LLM that returns a fixed response."""
    llm = MagicMock()
    msg = MagicMock()
    msg.content = response_text
    llm.invoke.return_value = msg
    return llm


def _fake_docs(n: int = 3) -> list[Document]:
    """Build a list of fake section-split Documents."""
    sections = ["Corporate Model", "Corporate Model", "CRE Model"]
    subsections = ["Statement of Purpose", "Model Overview", "Statement of Purpose"]
    return [
        Document(
            page_content=f"Content for section {i}. " * 20,
            metadata={
                "source": "test.pdf",
                "section": sections[i] if i < len(sections) else f"Section {i}",
                "subsection": subsections[i] if i < len(subsections) else f"Sub {i}",
                "start_page": i * 10 + 1,
                "end_page": (i + 1) * 10,
            },
        )
        for i in range(n)
    ]


# ── _split_on_paragraphs tests ─────────────────────────────────────


class TestSplitOnParagraphs(unittest.TestCase):
    """Tests for paragraph-boundary splitting."""

    def test_short_text_single_block(self):
        """Text shorter than max_chars → one block."""
        result = _split_on_paragraphs("Hello world", max_chars=100)
        self.assertEqual(result, ["Hello world"])

    def test_splits_on_paragraph_boundary(self):
        """Two paragraphs that together exceed max → two blocks."""
        para1 = "A" * 60
        para2 = "B" * 60
        text = f"{para1}\n\n{para2}"
        result = _split_on_paragraphs(text, max_chars=80)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], para1)
        self.assertEqual(result[1], para2)

    def test_oversized_single_paragraph_kept_intact(self):
        """A single paragraph bigger than max_chars is not split."""
        big = "X" * 200
        result = _split_on_paragraphs(big, max_chars=50)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], big)

    def test_empty_paragraphs_filtered(self):
        """Blank paragraphs are excluded from output."""
        text = "Hello\n\n\n\nWorld"
        result = _split_on_paragraphs(text, max_chars=1000)
        # "\n\n\n\n" splits into ["Hello", "", "World"]
        for block in result:
            self.assertTrue(block.strip())

    def test_multiple_paragraphs_grouped(self):
        """Multiple small paragraphs that fit together are grouped."""
        paras = ["Short." for _ in range(10)]
        text = "\n\n".join(paras)
        result = _split_on_paragraphs(text, max_chars=1000)
        self.assertEqual(len(result), 1)


# ── _invoke_llm tests ──────────────────────────────────────────────


class TestInvokeLlm(unittest.TestCase):
    """Tests for the single-call LLM wrapper."""

    def test_extracts_content_from_ai_message(self):
        """Should extract .content from an AIMessage-like response."""
        llm = _mock_llm("  Plain English version.  ")
        result = _invoke_llm(llm, "Rewrite: {text}", "original")
        self.assertEqual(result, "Plain English version.")
        llm.invoke.assert_called_once_with("Rewrite: original")

    def test_handles_plain_string_response(self):
        """Should handle LLMs that return plain strings."""
        llm = MagicMock()
        llm.invoke.return_value = "Just a string"
        result = _invoke_llm(llm, "{text}", "input")
        self.assertEqual(result, "Just a string")

    def test_handles_list_content_blocks(self):
        """Should handle Anthropic-style list-of-dicts content blocks."""
        llm = MagicMock()
        msg = MagicMock()
        msg.content = [{"type": "text", "text": "Block one."}, {"type": "text", "text": "Block two."}]
        llm.invoke.return_value = msg
        result = _invoke_llm(llm, "{text}", "input")
        self.assertIn("Block one.", result)
        self.assertIn("Block two.", result)

    def test_handles_list_of_plain_strings(self):
        """Should handle content returned as a list of strings."""
        llm = MagicMock()
        msg = MagicMock()
        msg.content = ["First part.", "Second part."]
        llm.invoke.return_value = msg
        result = _invoke_llm(llm, "{text}", "input")
        self.assertIn("First part.", result)
        self.assertIn("Second part.", result)


# ── _rewrite_chunk tests ───────────────────────────────────────────


class TestRewriteChunk(unittest.TestCase):
    """Tests for the chunk rewriter (handles splitting)."""

    def test_short_text_single_call(self):
        """Text within MAX_CHUNK_CHARS → single LLM call."""
        llm = _mock_llm("Simplified.")
        result = _rewrite_chunk(llm, "{text}", "Short input.")
        self.assertEqual(result, "Simplified.")
        self.assertEqual(llm.invoke.call_count, 1)

    def test_long_text_splits_and_joins(self):
        """Text exceeding MAX_CHUNK_CHARS → multiple LLM calls."""
        # Build text that's > MAX_CHUNK_CHARS with clear paragraph breaks
        para = "A" * (MAX_CHUNK_CHARS // 2 + 100)
        text = f"{para}\n\n{para}"
        self.assertGreater(len(text), MAX_CHUNK_CHARS)

        llm = _mock_llm("Part.")
        result = _rewrite_chunk(llm, "{text}", text)

        # Should have called LLM twice (once per paragraph block)
        self.assertEqual(llm.invoke.call_count, 2)
        self.assertEqual(result, "Part.\n\nPart.")


# ── _build_prompt tests ────────────────────────────────────────────


class TestBuildPrompt(unittest.TestCase):
    """Tests for _build_prompt() custom instruction injection."""

    def test_rewrite_mode_no_custom(self):
        """Default rewrite prompt with no custom instructions."""
        prompt = _build_prompt("rewrite")
        self.assertIn("plain-language editor", prompt)
        self.assertIn("{text}", prompt)
        self.assertNotIn("ADDITIONAL INSTRUCTIONS", prompt)

    def test_summarize_mode_no_custom(self):
        """Default summarize prompt with no custom instructions."""
        prompt = _build_prompt("summarize")
        self.assertIn("Summarize", prompt)
        self.assertIn("{text}", prompt)
        self.assertNotIn("ADDITIONAL INSTRUCTIONS", prompt)

    def test_rewrite_with_custom_prompt(self):
        """Custom instructions injected into rewrite prompt."""
        prompt = _build_prompt("rewrite", "Write at a 5th-grade level.")
        self.assertIn("ADDITIONAL INSTRUCTIONS", prompt)
        self.assertIn("5th-grade level", prompt)
        self.assertIn("{text}", prompt)

    def test_summarize_with_custom_prompt(self):
        """Custom instructions injected into summarize prompt."""
        prompt = _build_prompt("summarize", "Focus on risk metrics only.")
        self.assertIn("ADDITIONAL INSTRUCTIONS", prompt)
        self.assertIn("risk metrics", prompt)

    def test_none_custom_prompt_same_as_omitted(self):
        """Passing None is equivalent to no custom_prompt."""
        prompt_none = _build_prompt("rewrite", None)
        prompt_omitted = _build_prompt("rewrite")
        self.assertEqual(prompt_none, prompt_omitted)


# ── rewrite_pdf tests ──────────────────────────────────────────────


class TestRewritePdf(unittest.TestCase):
    """Tests for the top-level rewrite_pdf() function."""

    @patch("src.generation.rewriter.load_pdf_by_section")
    @patch("src.generation.rewriter.has_section_headers")
    @patch("src.generation.rewriter.get_llm")
    @patch("os.path.isfile", return_value=True)
    def test_produces_markdown_file(
        self, mock_isfile, mock_get_llm, mock_has_headers, mock_load
    ):
        """Should write a .md file and return its path."""
        mock_has_headers.return_value = True
        mock_load.return_value = _fake_docs(2)
        mock_get_llm.return_value = _mock_llm("Rewritten.")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = rewrite_pdf(
                "test.pdf", output_dir=tmpdir, provider="ollama", model="test"
            )
            self.assertTrue(result.endswith(".md"))
            self.assertTrue(os.path.isfile(result))

            with open(result) as f:
                content = f.read()

            # Check markdown structure (metadata is in config.txt)
            self.assertIn("# Test", content)
            self.assertIn("## Corporate Model", content)
            self.assertIn("### Statement of Purpose", content)
            self.assertIn("Rewritten.", content)

            # Config.txt should exist alongside the markdown
            config_path = os.path.join(os.path.dirname(result), "config.txt")
            self.assertTrue(os.path.isfile(config_path))
            with open(config_path) as f:
                cfg = f.read()
            self.assertIn("rewrite", cfg)
            self.assertIn("ollama", cfg)
            self.assertIn("test", cfg)

    @patch("src.generation.rewriter.load_pdf_by_section")
    @patch("src.generation.rewriter.has_section_headers")
    @patch("src.generation.rewriter.get_llm")
    @patch("os.path.isfile", return_value=True)
    def test_summarize_mode(
        self, mock_isfile, mock_get_llm, mock_has_headers, mock_load
    ):
        """Summarize mode should use the summarize prompt and tag output."""
        mock_has_headers.return_value = True
        mock_load.return_value = _fake_docs(1)
        mock_get_llm.return_value = _mock_llm("Summary.")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = rewrite_pdf(
                "test.pdf", output_dir=tmpdir, mode="summarize",
                provider="ollama", model="test",
            )
            with open(result) as f:
                content = f.read()

            self.assertIn("Summary.", content)
            self.assertIn("_summarize.md", result)

            # Mode recorded in config.txt
            config_path = os.path.join(os.path.dirname(result), "config.txt")
            with open(config_path) as f:
                cfg = f.read()
            self.assertIn("summarize", cfg)

    @patch("src.generation.rewriter.has_section_headers")
    @patch("os.path.isfile", return_value=True)
    def test_rejects_unstructured_pdf(self, mock_isfile, mock_has_headers):
        """Should raise ValueError for PDFs without section headers."""
        mock_has_headers.return_value = False
        with self.assertRaises(ValueError) as ctx:
            rewrite_pdf("flat.pdf")
        self.assertIn("section headers", str(ctx.exception))

    def test_rejects_missing_file(self):
        """Should raise FileNotFoundError for non-existent files."""
        with self.assertRaises(FileNotFoundError):
            rewrite_pdf("/nonexistent/path/missing.pdf")

    @patch("src.generation.rewriter.load_pdf_by_section")
    @patch("src.generation.rewriter.has_section_headers")
    @patch("src.generation.rewriter.get_llm")
    @patch("os.path.isfile", return_value=True)
    def test_section_headers_in_output(
        self, mock_isfile, mock_get_llm, mock_has_headers, mock_load
    ):
        """Section changes should produce new ## headings in the output."""
        mock_has_headers.return_value = True
        mock_load.return_value = _fake_docs(3)  # 2 sections
        mock_get_llm.return_value = _mock_llm("Text.")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = rewrite_pdf(
                "test.pdf", output_dir=tmpdir, provider="ollama", model="test"
            )
            with open(result) as f:
                content = f.read()

            self.assertIn("## Corporate Model", content)
            self.assertIn("## CRE Model", content)
            # Corporate Model heading should appear only once
            self.assertEqual(content.count("## Corporate Model"), 1)

    @patch("src.generation.rewriter.load_pdf_by_section")
    @patch("src.generation.rewriter.has_section_headers")
    @patch("src.generation.rewriter.get_llm")
    @patch("os.path.isfile", return_value=True)
    def test_llm_called_per_section(
        self, mock_isfile, mock_get_llm, mock_has_headers, mock_load
    ):
        """LLM invoked twice per section in rewrite mode (rewrite + summary)."""
        mock_has_headers.return_value = True
        docs = _fake_docs(3)
        mock_load.return_value = docs
        mock_llm = _mock_llm("Done.")
        mock_get_llm.return_value = mock_llm

        with tempfile.TemporaryDirectory() as tmpdir:
            rewrite_pdf(
                "test.pdf", output_dir=tmpdir, provider="ollama", model="test"
            )

        # 3 sections × 2 LLM calls each (rewrite + summary) = 6
        self.assertEqual(mock_llm.invoke.call_count, 6)

    @patch("src.generation.rewriter.load_pdf_by_section")
    @patch("src.generation.rewriter.has_section_headers")
    @patch("src.generation.rewriter.get_llm")
    @patch("os.path.isfile", return_value=True)
    def test_page_range_in_output(
        self, mock_isfile, mock_get_llm, mock_has_headers, mock_load
    ):
        """Output should include page ranges from metadata."""
        mock_has_headers.return_value = True
        mock_load.return_value = _fake_docs(1)
        mock_get_llm.return_value = _mock_llm("Text.")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = rewrite_pdf(
                "test.pdf", output_dir=tmpdir, provider="ollama", model="test"
            )
            with open(result) as f:
                content = f.read()

            self.assertIn("Pages 1–10", content)

    @patch("src.generation.rewriter.load_pdf_by_section")
    @patch("src.generation.rewriter.has_section_headers")
    @patch("src.generation.rewriter.get_llm")
    @patch("os.path.isfile", return_value=True)
    def test_custom_prompt_in_config_file(
        self, mock_isfile, mock_get_llm, mock_has_headers, mock_load
    ):
        """Custom prompt should appear in config.txt, not the markdown."""
        mock_has_headers.return_value = True
        mock_load.return_value = _fake_docs(1)
        mock_get_llm.return_value = _mock_llm("Simple text.")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = rewrite_pdf(
                "test.pdf",
                output_dir=tmpdir,
                provider="ollama",
                model="test",
                custom_prompt="Write at an elementary-school reading level.",
            )

            # Custom prompt should NOT be in the markdown
            with open(result) as f:
                md = f.read()
            self.assertNotIn("Custom instructions:", md)

            # Custom prompt should be in config.txt
            config_path = os.path.join(os.path.dirname(result), "config.txt")
            with open(config_path) as f:
                cfg = f.read()
            self.assertIn("elementary-school", cfg)
            self.assertIn("custom_instructions:", cfg)

    @patch("src.generation.rewriter.load_pdf_by_section")
    @patch("src.generation.rewriter.has_section_headers")
    @patch("src.generation.rewriter.get_llm")
    @patch("os.path.isfile", return_value=True)
    def test_custom_prompt_injected_into_llm_call(
        self, mock_isfile, mock_get_llm, mock_has_headers, mock_load
    ):
        """Custom prompt text should appear in the prompt sent to the LLM."""
        mock_has_headers.return_value = True
        mock_load.return_value = _fake_docs(1)
        mock_llm = _mock_llm("Rewritten.")
        mock_get_llm.return_value = mock_llm

        with tempfile.TemporaryDirectory() as tmpdir:
            rewrite_pdf(
                "test.pdf",
                output_dir=tmpdir,
                provider="ollama",
                model="test",
                custom_prompt="Use only one-syllable words.",
            )

        # The first prompt sent to LLM (rewrite) should contain the custom
        # instructions; the second call is the summary prompt.
        actual_prompt = mock_llm.invoke.call_args_list[0][0][0]
        self.assertIn("one-syllable words", actual_prompt)
        self.assertIn("ADDITIONAL INSTRUCTIONS", actual_prompt)

    @patch("src.generation.rewriter.load_pdf_by_section")
    @patch("src.generation.rewriter.has_section_headers")
    @patch("src.generation.rewriter.get_llm")
    @patch("os.path.isfile", return_value=True)
    def test_rewrite_mode_appends_summary(
        self, mock_isfile, mock_get_llm, mock_has_headers, mock_load
    ):
        """Rewrite mode should append a summary blockquote after each section."""
        mock_has_headers.return_value = True
        mock_load.return_value = _fake_docs(1)
        mock_get_llm.return_value = _mock_llm("Rewritten text.")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = rewrite_pdf(
                "test.pdf", output_dir=tmpdir, provider="ollama", model="test"
            )
            with open(result) as f:
                content = f.read()

            self.assertIn("> **Summary:**", content)

    @patch("src.generation.rewriter.load_pdf_by_section")
    @patch("src.generation.rewriter.has_section_headers")
    @patch("src.generation.rewriter.get_llm")
    @patch("os.path.isfile", return_value=True)
    def test_summarize_mode_no_extra_summary(
        self, mock_isfile, mock_get_llm, mock_has_headers, mock_load
    ):
        """Summarize mode should NOT append an extra summary blockquote."""
        mock_has_headers.return_value = True
        mock_load.return_value = _fake_docs(1)
        mock_llm = _mock_llm("Summary text.")
        mock_get_llm.return_value = mock_llm

        with tempfile.TemporaryDirectory() as tmpdir:
            result = rewrite_pdf(
                "test.pdf", output_dir=tmpdir, provider="ollama", model="test",
                mode="summarize",
            )
            with open(result) as f:
                content = f.read()

            self.assertNotIn("> **Summary:**", content)
            # Summarize mode: 1 LLM call per section (no extra summary)
            self.assertEqual(mock_llm.invoke.call_count, 1)

    @patch("src.generation.rewriter.load_pdf_by_section")
    @patch("src.generation.rewriter.has_section_headers")
    @patch("src.generation.rewriter.get_llm")
    @patch("os.path.isfile", return_value=True)
    def test_selected_sections_filters_docs(
        self, mock_isfile, mock_get_llm, mock_has_headers, mock_load
    ):
        """Only selected section/subsection documents should be processed."""
        mock_has_headers.return_value = True
        mock_load.return_value = _fake_docs(3)
        mock_llm = _mock_llm("Selected only.")
        mock_get_llm.return_value = mock_llm

        with tempfile.TemporaryDirectory() as tmpdir:
            result = rewrite_pdf(
                "test.pdf",
                output_dir=tmpdir,
                provider="ollama",
                model="test",
                selected_sections={
                    "Corporate Model": ["Statement of Purpose"],
                },
            )

            with open(result) as f:
                content = f.read()

        self.assertIn("### Statement of Purpose", content)
        self.assertNotIn("### Model Overview", content)
        self.assertNotIn("## CRE Model", content)
        self.assertEqual(mock_llm.invoke.call_count, 2)

    @patch("src.generation.rewriter.load_pdf_by_section")
    @patch("src.generation.rewriter.has_section_headers")
    @patch("src.generation.rewriter.get_llm")
    @patch("os.path.isfile", return_value=True)
    def test_selected_sections_no_match_raises_value_error(
        self, mock_isfile, mock_get_llm, mock_has_headers, mock_load
    ):
        """An empty filter result should raise a clear ValueError."""
        mock_has_headers.return_value = True
        mock_load.return_value = _fake_docs(2)
        mock_get_llm.return_value = _mock_llm("Ignored.")

        with self.assertRaises(ValueError) as ctx:
            rewrite_pdf(
                "test.pdf",
                provider="ollama",
                model="test",
                selected_sections={"Nonexistent Section": ["Nope"]},
            )
        self.assertIn("No matching sections were selected", str(ctx.exception))


# ── CLI tests ──────────────────────────────────────────────────────


class TestCLI(unittest.TestCase):
    """Tests for the argparse CLI entry point."""

    @patch("src.generation.rewriter.rewrite_pdf")
    def test_main_calls_rewrite_pdf(self, mock_rewrite):
        """main() should parse args and call rewrite_pdf."""
        mock_rewrite.return_value = "/tmp/out.md"

        from src.generation.rewriter import main

        with patch(
            "sys.argv",
            ["rewriter", "test.pdf", "--mode", "summarize", "--provider", "openai"],
        ):
            main()

        mock_rewrite.assert_called_once()
        call_kwargs = mock_rewrite.call_args
        self.assertEqual(call_kwargs[0][0], "test.pdf")
        self.assertEqual(call_kwargs[1]["mode"], "summarize")
        self.assertEqual(call_kwargs[1]["provider"], "openai")

    @patch("src.generation.rewriter.rewrite_pdf")
    def test_main_passes_custom_prompt(self, mock_rewrite):
        """main() should forward --custom-prompt to rewrite_pdf."""
        mock_rewrite.return_value = "/tmp/out.md"

        from src.generation.rewriter import main

        with patch(
            "sys.argv",
            [
                "rewriter", "test.pdf",
                "--custom-prompt", "Write at an elementary-school level.",
            ],
        ):
            main()

        mock_rewrite.assert_called_once()
        call_kwargs = mock_rewrite.call_args
        self.assertEqual(
            call_kwargs[1]["custom_prompt"],
            "Write at an elementary-school level.",
        )


# ── _split_markdown_sections tests ─────────────────────────────────


class TestSplitMarkdownSections(unittest.TestCase):
    """Tests for Markdown section splitting."""

    def test_splits_on_h2_headings(self):
        md = "# Title\n\n---\n\n## Section A\n\nBody A\n\n## Section B\n\nBody B\n"
        result = _split_markdown_sections(md)
        sections = [s["section"] for s in result]
        self.assertIn("Section A", sections)
        self.assertIn("Section B", sections)

    def test_splits_on_h3_subsections(self):
        md = (
            "## Section A\n\n### Sub 1\n\nBody 1\n\n"
            "### Sub 2\n\nBody 2\n"
        )
        result = _split_markdown_sections(md)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["subsection"], "Sub 1")
        self.assertEqual(result[1]["subsection"], "Sub 2")

    def test_preamble_captured(self):
        md = "Some intro text\n\n## Section A\n\nBody A\n"
        result = _split_markdown_sections(md)
        preamble = [s for s in result if s["section"] == "(preamble)"]
        self.assertTrue(len(preamble) >= 1)

    def test_empty_sections_filtered(self):
        md = "## Empty\n\n---\n\n## Real\n\nSome body\n"
        result = _split_markdown_sections(md)
        # The empty section (only "---") should be filtered out
        bodies = [s["body"] for s in result]
        for b in bodies:
            self.assertTrue(b.replace("-", "").strip())

    def test_no_headings_returns_preamble(self):
        md = "Just some plain text with no headings.\n\nAnother paragraph.\n"
        result = _split_markdown_sections(md)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["section"], "(preamble)")


# ── discover_rewrite_outputs tests ─────────────────────────────────


class TestDiscoverRewriteOutputs(unittest.TestCase):
    """Tests for the output discovery helper."""

    def test_finds_md_in_subdirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "20260215_1519")
            os.makedirs(subdir)
            md_path = os.path.join(subdir, "test_rewrite.md")
            with open(md_path, "w") as f:
                f.write("# Test\n")
            result = discover_rewrite_outputs(tmpdir)
            self.assertEqual(len(result), 1)
            self.assertIn("20260215_1519 / test_rewrite.md", list(result.keys())[0])

    def test_finds_top_level_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = os.path.join(tmpdir, "standalone.md")
            with open(md_path, "w") as f:
                f.write("# Standalone\n")
            result = discover_rewrite_outputs(tmpdir)
            self.assertIn("standalone.md", result)

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = discover_rewrite_outputs(tmpdir)
            self.assertEqual(len(result), 0)

    def test_nonexistent_dir(self):
        result = discover_rewrite_outputs("/nonexistent/path")
        self.assertEqual(len(result), 0)


# ── _build_prompt refine tests ─────────────────────────────────────


class TestBuildPromptRefine(unittest.TestCase):
    """Tests for _build_prompt in refine mode."""

    def test_refine_uses_custom_prompt(self):
        prompt = _build_prompt("refine", "Add more diagrams.")
        self.assertIn("Add more diagrams.", prompt)
        self.assertIn("{text}", prompt)
        self.assertNotIn("ADDITIONAL INSTRUCTIONS", prompt)

    def test_refine_fallback_when_no_custom(self):
        prompt = _build_prompt("refine")
        self.assertIn("Improve clarity", prompt)
        self.assertIn("{text}", prompt)

    def test_refine_prompt_is_different_from_rewrite(self):
        refine = _build_prompt("refine", "Make it shorter.")
        rewrite = _build_prompt("rewrite", "Make it shorter.")
        self.assertNotEqual(refine, rewrite)


# ── refine_markdown_iter tests ─────────────────────────────────────


class TestRefineMarkdownIter(unittest.TestCase):
    """Tests for refine_markdown_iter()."""

    _SAMPLE_MD = (
        "# Test Doc\n\n---\n\n"
        "## Section A\n\n### Sub 1\n\nOriginal body 1.\n\n"
        "### Sub 2\n\nOriginal body 2.\n\n"
        "## Section B\n\n### Sub 3\n\nOriginal body 3.\n"
    )

    @patch("src.generation.rewriter.get_llm")
    def test_produces_refined_output(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("Refined.")

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "source_rewrite.md")
            with open(src_path, "w") as f:
                f.write(self._SAMPLE_MD)

            events = list(refine_markdown_iter(
                src_path,
                provider="ollama",
                model="test",
                custom_prompt="Make it shorter.",
                output_dir=tmpdir,
            ))

            # Should end with a "done" event
            done = [e for e in events if e.phase == "done"]
            self.assertEqual(len(done), 1)
            out_path = done[0].output_path
            self.assertTrue(os.path.isfile(out_path))
            self.assertIn("_refine.md", out_path)

            with open(out_path) as f:
                content = f.read()
            self.assertIn("Refined.", content)

    @patch("src.generation.rewriter.get_llm")
    def test_yields_processing_and_rewriting(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("Done.")

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "test_rewrite.md")
            with open(src_path, "w") as f:
                f.write(self._SAMPLE_MD)

            events = list(refine_markdown_iter(
                src_path,
                provider="ollama",
                model="test",
                custom_prompt="Fix typos.",
                output_dir=tmpdir,
            ))

            phases = [e.phase for e in events]
            self.assertIn("processing", phases)
            self.assertIn("rewriting", phases)
            self.assertIn("done", phases)

    def test_rejects_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            list(refine_markdown_iter(
                "/nonexistent/path.md",
                provider="ollama",
                model="test",
            ))

    @patch("src.generation.rewriter.get_llm")
    def test_config_file_records_refine_mode(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("Refined.")

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "input_rewrite.md")
            with open(src_path, "w") as f:
                f.write(self._SAMPLE_MD)

            events = list(refine_markdown_iter(
                src_path,
                provider="ollama",
                model="test",
                custom_prompt="Add glossary.",
                output_dir=tmpdir,
            ))

            done = [e for e in events if e.phase == "done"]
            config_path = os.path.join(
                os.path.dirname(done[0].output_path), "config.txt"
            )
            with open(config_path) as f:
                cfg = f.read()
            self.assertIn("refine", cfg)
            self.assertIn("Add glossary.", cfg)

    @patch("src.generation.rewriter.get_llm")
    def test_strips_rewrite_suffix_from_basename(self, mock_get_llm):
        """Output filename strips trailing _rewrite from the source name."""
        mock_get_llm.return_value = _mock_llm("Refined.")

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "my_doc_rewrite.md")
            with open(src_path, "w") as f:
                f.write(self._SAMPLE_MD)

            events = list(refine_markdown_iter(
                src_path,
                provider="ollama",
                model="test",
                custom_prompt="Fix.",
                output_dir=tmpdir,
            ))
            done = [e for e in events if e.phase == "done"]
            basename = os.path.basename(done[0].output_path)
            self.assertIn("my_doc_refine.md", basename)
            self.assertNotIn("_rewrite_refine", basename)


if __name__ == "__main__":
    unittest.main()
