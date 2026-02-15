"""Section-aware PDF splitter for structured model documentation.

The Federal Reserve's stress test model documentation PDFs (credit risk,
market risk, PPNR, etc.) follow a consistent structure:

  - Every page has a header like ``Model Documentation: Corporate Model``
  - Top-level sections are lettered (A, B, C, …)
  - Subsections use Roman numerals (i, ii, iii, …)

This module detects those boundaries and splits PDF text into
LangChain Document objects tagged with rich metadata:

  - ``section``     – e.g. "Corporate Model"
  - ``subsection``  – e.g. "Model Overview"
  - ``page``        – starting page number of the chunk

The resulting Documents can then be passed through the normal
RecursiveCharacterTextSplitter for final chunking, inheriting
section/subsection metadata into every chunk.
"""

import logging
import os
import re

from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# ── Header / heading patterns ───────────────────────────────────────

# Page header stamped on every page: "7 Model Documentation: Corporate Model"
_PAGE_HEADER_RE = re.compile(
    r"^\d+\s+Model Documentation:\s+(.+)",
    re.MULTILINE,
)

# Lettered top-level heading inside the body text:
#   "A. Corporate Model" or "B. Commercial Real Estate Model"
_SECTION_HEADING_RE = re.compile(
    r"^([A-Z])\.\s+(.+?)(?:\s*\.{3,}\s*\d+)?$",
    re.MULTILINE,
)

# Roman-numeral subsection heading:
#   "i. Statement of Purpose" or "viii. Question"
_SUBSECTION_HEADING_RE = re.compile(
    r"^((?:x{0,3})(?:ix|iv|v?i{1,3}))\.\s+(.+?)(?:\s*\.{3,}\s*\d+)?$",
    re.MULTILINE,
)


# ── Public API ──────────────────────────────────────────────────────


def has_section_headers(filepath: str, sample_pages: int = 10) -> bool:
    """Return True if the PDF uses ``Model Documentation:`` page headers.

    Checks the first *sample_pages* pages for the header pattern.
    This is a cheap probe to decide whether to use the section-aware
    splitter vs. the generic page-per-document loader.
    """
    reader = PdfReader(filepath)
    for i in range(min(sample_pages, len(reader.pages))):
        text = reader.pages[i].extract_text() or ""
        if _PAGE_HEADER_RE.search(text):
            return True
    return False


def load_pdf_by_section(filepath: str) -> list[Document]:
    """Load a structured PDF and return one Document per subsection.

    Each Document's ``page_content`` contains the full text of one
    subsection (e.g. "Statement of Purpose" under "Corporate Model").
    Metadata includes:

    - ``source``     – the original file path
    - ``section``    – top-level section name (from page header)
    - ``subsection`` – subsection name (from Roman-numeral heading)
    - ``start_page`` – 1-based page number where the subsection begins
    - ``end_page``   – 1-based page number where the subsection ends

    Pages that don't belong to any detected subsection are grouped
    under the subsection name ``"(intro)"`` within their section.

    Returns
    -------
    list[Document]
        One Document per subsection, ordered by page number.
    """
    reader = PdfReader(filepath)
    n_pages = len(reader.pages)

    # ── Step 1: Extract text and detect the section for each page ───
    page_texts: list[str] = []
    page_sections: list[str] = []

    current_section = "(preamble)"
    for i in range(n_pages):
        text = reader.pages[i].extract_text() or ""
        page_texts.append(text)

        m = _PAGE_HEADER_RE.search(text)
        if m:
            current_section = m.group(1).strip()
        page_sections[i:i] = []  # placeholder — set below
        page_sections.append(current_section)

    # ── Step 2: Detect subsection headings within each page ─────────
    #
    # Build a list of (page_index, char_offset, subsection_name) for
    # every Roman-numeral heading found.
    SubsectionMark = tuple  # (page_idx, char_offset, name)
    marks: list[SubsectionMark] = []

    for i, text in enumerate(page_texts):
        # Skip preamble pages (title, TOC) — they contain TOC entries
        # that look like subsection headings but aren't.
        if page_sections[i] == "(preamble)":
            continue
        for m in _SUBSECTION_HEADING_RE.finditer(text):
            numeral = m.group(1).strip().lower()
            name = m.group(2).strip()
            # Filter false positives: Roman numeral must be valid
            # and the name must look like a real heading (not a TOC entry)
            if numeral and name and len(name) > 2:
                marks.append((i, m.start(), name))

    # ── Step 3: Segment pages into (section, subsection) blocks ─────
    #
    # We merge consecutive pages that share the same (section, subsection).
    # When a new subsection heading is detected on a page, we start a
    # new block from that page onward.

    # First, assign each page a subsection label.
    page_subsections: list[str] = ["(intro)"] * n_pages

    # Sort marks by page then offset
    marks.sort(key=lambda t: (t[0], t[1]))

    # Walk marks: from each mark's page onward (within same section),
    # label pages with that subsection until the next mark.
    for idx, (page_i, _offset, sub_name) in enumerate(marks):
        section_of_mark = page_sections[page_i]
        # Label from this page to either the next mark's page or end
        if idx + 1 < len(marks):
            end_page = marks[idx + 1][0]
        else:
            end_page = n_pages

        for p in range(page_i, end_page):
            # Stop if we've moved into a different section
            if page_sections[p] != section_of_mark:
                break
            page_subsections[p] = sub_name

    # ── Step 4: Build Documents by grouping consecutive same-label pages
    documents: list[Document] = []
    block_section = page_sections[0]
    block_subsection = page_subsections[0]
    block_start = 0
    block_texts: list[str] = [_strip_page_header(page_texts[0])]

    for i in range(1, n_pages):
        sec = page_sections[i]
        sub = page_subsections[i]

        if sec == block_section and sub == block_subsection:
            # Same block — accumulate text
            block_texts.append(_strip_page_header(page_texts[i]))
        else:
            # Flush the previous block
            _flush_block(
                documents, block_texts,
                filepath, block_section, block_subsection,
                block_start, i - 1,
            )
            # Start a new block
            block_section = sec
            block_subsection = sub
            block_start = i
            block_texts = [_strip_page_header(page_texts[i])]

    # Flush the last block
    _flush_block(
        documents, block_texts,
        filepath, block_section, block_subsection,
        block_start, n_pages - 1,
    )

    logger.info(
        f"Section-split '{os.path.basename(filepath)}': "
        f"{n_pages} pages → {len(documents)} section document(s)"
    )
    for doc in documents:
        logger.debug(
            f"  [{doc.metadata['section']}] {doc.metadata['subsection']}  "
            f"pp. {doc.metadata['start_page']}–{doc.metadata['end_page']}  "
            f"({len(doc.page_content)} chars)"
        )

    return documents


# ── Private helpers ─────────────────────────────────────────────────


def _strip_page_header(text: str) -> str:
    """Remove the ``Model Documentation: …`` header line from page text.

    Also strips the leading page number that typically precedes the
    header.  This avoids duplicating the header text inside every
    chunk.
    """
    return _PAGE_HEADER_RE.sub("", text, count=1).lstrip("\n")


def _flush_block(
    documents: list[Document],
    block_texts: list[str],
    source: str,
    section: str,
    subsection: str,
    start_idx: int,
    end_idx: int,
) -> None:
    """Join accumulated page texts and append a Document."""
    content = "\n".join(block_texts).strip()
    if not content:
        return
    documents.append(
        Document(
            page_content=content,
            metadata={
                "source": source,
                "section": section,
                "subsection": subsection,
                "start_page": start_idx + 1,   # 1-based
                "end_page": end_idx + 1,        # 1-based
            },
        )
    )
