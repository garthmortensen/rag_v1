"""Tests for the section-aware PDF splitter.

Tests use mock PdfReader objects to avoid depending on real PDF files
in the corpus.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.pdf_section_splitter import (
    has_section_headers,
    load_pdf_by_section,
    _strip_page_header,
    _PAGE_HEADER_RE,
    _SUBSECTION_HEADING_RE,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_page(text: str) -> MagicMock:
    """Create a mock PdfReader page that returns *text*."""
    page = MagicMock()
    page.extract_text.return_value = text
    return page


def _make_reader(page_texts: list[str]) -> MagicMock:
    """Create a mock PdfReader with the given page texts."""
    reader = MagicMock()
    pages = [_make_page(t) for t in page_texts]
    reader.pages = pages
    reader.__len__ = lambda self: len(pages)
    return reader


# ── Regex tests ─────────────────────────────────────────────────────


class TestPageHeaderRegex(unittest.TestCase):
    """Verify the _PAGE_HEADER_RE pattern matches real page headers."""

    def test_matches_standard_header(self):
        text = "7 Model Documentation: Corporate Model\nSome body text"
        m = _PAGE_HEADER_RE.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "Corporate Model")

    def test_matches_long_section_name(self):
        text = "118 Model Documentation: First Lien Mortgage Model\nBody"
        m = _PAGE_HEADER_RE.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "First Lien Mortgage Model")

    def test_matches_three_digit_page_number(self):
        text = "502 Model Documentation: Auto Model\nBody"
        m = _PAGE_HEADER_RE.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "Auto Model")

    def test_no_match_on_plain_text(self):
        text = "This is just regular paragraph text."
        m = _PAGE_HEADER_RE.search(text)
        self.assertIsNone(m)


class TestSubsectionHeadingRegex(unittest.TestCase):
    """Verify the _SUBSECTION_HEADING_RE pattern."""

    def test_matches_roman_i(self):
        m = _SUBSECTION_HEADING_RE.search("i. Statement of Purpose")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "i")
        self.assertEqual(m.group(2), "Statement of Purpose")

    def test_matches_roman_ii(self):
        m = _SUBSECTION_HEADING_RE.search("ii. Model Overview")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), "Model Overview")

    def test_matches_roman_iii(self):
        m = _SUBSECTION_HEADING_RE.search("iii. Key Assumptions for the Credit Card Model")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), "Key Assumptions for the Credit Card Model")

    def test_matches_roman_iv(self):
        m = _SUBSECTION_HEADING_RE.search("iv. Alternatives to the Home Equity Model")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), "Alternatives to the Home Equity Model")

    def test_matches_roman_viii(self):
        m = _SUBSECTION_HEADING_RE.search("viii. Question")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), "Question")

    def test_strips_toc_dotted_leader(self):
        m = _SUBSECTION_HEADING_RE.search(
            "i. Statement of Purpose .......... 7"
        )
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), "Statement of Purpose")

    def test_no_match_on_plain_sentence(self):
        m = _SUBSECTION_HEADING_RE.search(
            "The model is described in detail below."
        )
        self.assertIsNone(m)


# ── _strip_page_header tests ───────────────────────────────────────


class TestStripPageHeader(unittest.TestCase):
    """Verify header stripping removes the Model Documentation line."""

    def test_removes_header(self):
        text = "7 Model Documentation: Corporate Model\nBody text here."
        result = _strip_page_header(text)
        self.assertNotIn("Model Documentation", result)
        self.assertIn("Body text here.", result)

    def test_preserves_text_without_header(self):
        text = "Just normal text with no header."
        result = _strip_page_header(text)
        self.assertEqual(result, text)


# ── has_section_headers tests ──────────────────────────────────────


class TestHasSectionHeaders(unittest.TestCase):
    """Verify the quick probe function."""

    @patch("src.ingestion.pdf_section_splitter.PdfReader")
    def test_returns_true_when_headers_present(self, mock_reader_cls):
        mock_reader_cls.return_value = _make_reader([
            "Title page",
            "Table of Contents",
            "7 Model Documentation: Corporate Model\nBody",
        ])
        self.assertTrue(has_section_headers("fake.pdf"))

    @patch("src.ingestion.pdf_section_splitter.PdfReader")
    def test_returns_false_when_no_headers(self, mock_reader_cls):
        mock_reader_cls.return_value = _make_reader([
            "Title page",
            "Some general content without structure",
        ])
        self.assertFalse(has_section_headers("fake.pdf"))

    @patch("src.ingestion.pdf_section_splitter.PdfReader")
    def test_only_checks_sample_pages(self, mock_reader_cls):
        """If the header is on page 15 and sample_pages=10, should miss it."""
        pages = ["No header"] * 14 + [
            "15 Model Documentation: Late Section\nBody"
        ]
        mock_reader_cls.return_value = _make_reader(pages)
        self.assertFalse(has_section_headers("fake.pdf", sample_pages=10))
        self.assertTrue(has_section_headers("fake.pdf", sample_pages=15))


# ── load_pdf_by_section tests ─────────────────────────────────────


class TestLoadPdfBySection(unittest.TestCase):
    """Integration-style tests for the full section splitter."""

    @patch("src.ingestion.pdf_section_splitter.PdfReader")
    def test_simple_two_section_pdf(self, mock_reader_cls):
        """Two sections, each with one subsection."""
        mock_reader_cls.return_value = _make_reader([
            "Title page content",                                          # p1 preamble
            "2 Model Documentation: Section A\ni. Overview\nBody of A",    # p2
            "3 Model Documentation: Section B\ni. Purpose\nBody of B",     # p3
        ])
        docs = load_pdf_by_section("test.pdf")

        # Should produce: preamble, Section A/Overview, Section B/Purpose
        self.assertEqual(len(docs), 3)

        # Preamble
        self.assertEqual(docs[0].metadata["section"], "(preamble)")
        self.assertEqual(docs[0].metadata["subsection"], "(intro)")

        # Section A
        self.assertEqual(docs[1].metadata["section"], "Section A")
        self.assertEqual(docs[1].metadata["subsection"], "Overview")
        self.assertEqual(docs[1].metadata["start_page"], 2)
        self.assertEqual(docs[1].metadata["end_page"], 2)
        self.assertEqual(docs[1].metadata["source"], "test.pdf")

        # Section B
        self.assertEqual(docs[2].metadata["section"], "Section B")
        self.assertEqual(docs[2].metadata["subsection"], "Purpose")

    @patch("src.ingestion.pdf_section_splitter.PdfReader")
    def test_multiple_subsections(self, mock_reader_cls):
        """One section with two subsections across multiple pages."""
        mock_reader_cls.return_value = _make_reader([
            "1 Model Documentation: Corp Model\ni. Statement of Purpose\nPurpose body",   # p1
            "2 Model Documentation: Corp Model\nMore purpose text",                         # p2
            "3 Model Documentation: Corp Model\nii. Model Overview\nOverview body",         # p3
            "4 Model Documentation: Corp Model\nMore overview",                              # p4
        ])
        docs = load_pdf_by_section("test.pdf")

        # Should produce 2 docs: Statement of Purpose (pp 1-2), Model Overview (pp 3-4)
        self.assertEqual(len(docs), 2)

        self.assertEqual(docs[0].metadata["subsection"], "Statement of Purpose")
        self.assertEqual(docs[0].metadata["start_page"], 1)
        self.assertEqual(docs[0].metadata["end_page"], 2)

        self.assertEqual(docs[1].metadata["subsection"], "Model Overview")
        self.assertEqual(docs[1].metadata["start_page"], 3)
        self.assertEqual(docs[1].metadata["end_page"], 4)

    @patch("src.ingestion.pdf_section_splitter.PdfReader")
    def test_section_transition(self, mock_reader_cls):
        """Section boundary detected from page header change."""
        mock_reader_cls.return_value = _make_reader([
            "1 Model Documentation: Alpha\ni. Intro\nAlpha content",       # p1
            "2 Model Documentation: Alpha\nMore alpha",                     # p2
            "3 Model Documentation: Beta\ni. Intro\nBeta content",         # p3
        ])
        docs = load_pdf_by_section("test.pdf")

        # Alpha/Intro (pp 1-2), Beta/Intro (p3)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["section"], "Alpha")
        self.assertEqual(docs[0].metadata["end_page"], 2)
        self.assertEqual(docs[1].metadata["section"], "Beta")
        self.assertEqual(docs[1].metadata["start_page"], 3)

    @patch("src.ingestion.pdf_section_splitter.PdfReader")
    def test_preamble_pages_not_split_by_toc_headings(self, mock_reader_cls):
        """TOC entries on preamble pages should NOT create subsection splits."""
        mock_reader_cls.return_value = _make_reader([
            "Title page",
            "Table of Contents\ni. Statement of Purpose ... 7\nii. Model Overview ... 10",
            "3 Model Documentation: Section A\ni. Statement of Purpose\nBody",
        ])
        docs = load_pdf_by_section("test.pdf")

        # Preamble should be one doc (pages 1-2), Section A one doc
        preamble_docs = [d for d in docs if d.metadata["section"] == "(preamble)"]
        self.assertEqual(len(preamble_docs), 1)
        self.assertEqual(preamble_docs[0].metadata["start_page"], 1)
        self.assertEqual(preamble_docs[0].metadata["end_page"], 2)

    @patch("src.ingestion.pdf_section_splitter.PdfReader")
    def test_page_headers_stripped_from_content(self, mock_reader_cls):
        """The 'Model Documentation: ...' header line should not appear in page_content."""
        mock_reader_cls.return_value = _make_reader([
            "1 Model Documentation: Test\ni. Intro\nActual body text",
        ])
        docs = load_pdf_by_section("test.pdf")
        self.assertEqual(len(docs), 1)
        self.assertNotIn("Model Documentation", docs[0].page_content)
        self.assertIn("Actual body text", docs[0].page_content)

    @patch("src.ingestion.pdf_section_splitter.PdfReader")
    def test_empty_pages_skipped(self, mock_reader_cls):
        """Pages with no text should not produce empty documents."""
        mock_reader_cls.return_value = _make_reader([
            "1 Model Documentation: Test\ni. Intro\nBody",
            "",  # empty page
            "3 Model Documentation: Test\nMore body",
        ])
        docs = load_pdf_by_section("test.pdf")
        for doc in docs:
            self.assertTrue(len(doc.page_content.strip()) > 0)

    @patch("src.ingestion.pdf_section_splitter.PdfReader")
    def test_metadata_contains_all_required_fields(self, mock_reader_cls):
        """Every Document must have source, section, subsection, start_page, end_page."""
        mock_reader_cls.return_value = _make_reader([
            "1 Model Documentation: MySection\ni. MySubsection\nBody text",
        ])
        docs = load_pdf_by_section("test.pdf")
        self.assertEqual(len(docs), 1)
        meta = docs[0].metadata
        self.assertIn("source", meta)
        self.assertIn("section", meta)
        self.assertIn("subsection", meta)
        self.assertIn("start_page", meta)
        self.assertIn("end_page", meta)
        self.assertEqual(meta["source"], "test.pdf")
        self.assertEqual(meta["section"], "MySection")
        self.assertEqual(meta["subsection"], "MySubsection")
        self.assertIsInstance(meta["start_page"], int)
        self.assertIsInstance(meta["end_page"], int)

    @patch("src.ingestion.pdf_section_splitter.PdfReader")
    def test_intro_subsection_for_pages_before_first_heading(self, mock_reader_cls):
        """Pages in a section before any subsection heading get '(intro)' label."""
        mock_reader_cls.return_value = _make_reader([
            "1 Model Documentation: Revisions\nSome revision notes",
            "2 Model Documentation: Revisions\nMore revision notes",
        ])
        docs = load_pdf_by_section("test.pdf")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["section"], "Revisions")
        self.assertEqual(docs[0].metadata["subsection"], "(intro)")

    @patch("src.ingestion.pdf_section_splitter.PdfReader")
    def test_returns_langchain_documents(self, mock_reader_cls):
        """Verify return type is list of LangChain Document objects."""
        from langchain_core.documents import Document

        mock_reader_cls.return_value = _make_reader([
            "1 Model Documentation: Test\ni. Intro\nBody",
        ])
        docs = load_pdf_by_section("test.pdf")
        for doc in docs:
            self.assertIsInstance(doc, Document)
            self.assertIsInstance(doc.page_content, str)
            self.assertIsInstance(doc.metadata, dict)


# ── Loader integration test ────────────────────────────────────────


class TestLoaderIntegration(unittest.TestCase):
    """Verify that loaders.py routes structured PDFs to the section splitter."""

    @patch("src.ingestion.loaders.load_pdf_by_section")
    @patch("src.ingestion.loaders.has_section_headers", return_value=True)
    def test_load_file_uses_section_splitter_for_structured_pdfs(
        self, mock_has_headers, mock_load_sections
    ):
        from langchain_core.documents import Document
        from src.ingestion.loaders import load_file

        mock_load_sections.return_value = [
            Document(
                page_content="Section body",
                metadata={
                    "source": "fake.pdf",
                    "section": "Corporate Model",
                    "subsection": "Statement of Purpose",
                    "start_page": 7,
                    "end_page": 9,
                },
            )
        ]
        docs = load_file("fake.pdf")
        mock_has_headers.assert_called_once_with("fake.pdf")
        mock_load_sections.assert_called_once_with("fake.pdf")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["section"], "Corporate Model")

    @patch("src.ingestion.loaders.has_section_headers", return_value=False)
    def test_load_file_falls_back_to_pypdf_for_unstructured_pdfs(
        self, mock_has_headers
    ):
        from langchain_core.documents import Document
        from src.ingestion.loaders import load_file

        mock_loader_cls = MagicMock()
        mock_loader_cls.__name__ = "PyPDFLoader"
        mock_instance = MagicMock()
        mock_instance.load.return_value = [
            Document(page_content="Page 1 content", metadata={"source": "flat.pdf"})
        ]
        mock_loader_cls.return_value = mock_instance

        with patch.dict("src.ingestion.loaders.LOADER_MAP", {".pdf": mock_loader_cls}):
            docs = load_file("flat.pdf")

        mock_has_headers.assert_called_once_with("flat.pdf")
        mock_loader_cls.assert_called_once_with("flat.pdf")
        self.assertEqual(len(docs), 1)


if __name__ == "__main__":
    unittest.main()
