#!/usr/bin/env python3
"""Parse academic citations from PDF documents in the corpus.

Extracts structured citation data and resolves downloadable URLs
for known sources (NBER, Federal Reserve, SSRN, arXiv, DOI).

Usage:
    python corpus/citations_parser.py
    python corpus/citations_parser.py --output corpus/citations.csv
    python corpus/citations_parser.py --pdf corpus/raw_data/some_file.pdf
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-19s  %(levelname)-9s %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)
console = Console()

# ── Data class ──────────────────────────────────────────────────────


@dataclass
class Citation:
    """A single parsed citation."""

    authors: str
    year: str
    title: str
    venue: str = ""
    source_pdf: str = ""
    category: str = "Unknown"
    resolved_url: str = ""
    download_url: str = ""
    identifier: str = ""


# ── Citation regex patterns ─────────────────────────────────────────
# These patterns cover the most common formats found in Fed stress
# test documentation.  They are applied in order; first match wins.

# Pattern 1:  Author, A. and B. Author (2025). "Title," Venue.
# Pattern 2:  Author, A., B. Author, and C. Author (2025). "Title." Venue.
# Pattern 3:  Author, A. (2025). Title. Venue.  (no quotes)

CITATION_PATTERNS = [
    # Quoted title with parenthetical year
    re.compile(
        r"(?P<authors>"
        r"[A-Z][a-zà-ÿA-Z\-']+"  # first author surname
        r"(?:[,;]?\s+(?:and\s+)?"  # separator
        r"[A-Z][\.\w]*\.?"  # initials or first name
        r"(?:\s+[A-Z][a-zà-ÿA-Z\-']+)?"  # optional additional surname
        r")*"  # repeat for co-authors
        r")"
        r"\s*\((?P<year>\d{4}[a-z]?)\)"  # (2025) or (2025a)
        r"[\.,;]?\s*"
        r'["\u201c](?P<title>[^"\u201d]+)["\u201d]'  # "Title"
        r"[,.\s]*"
        r"(?P<venue>[^.]*(?:\.\s|$))?"  # Venue.
    ),
    # Non-quoted title (books, reports)
    re.compile(
        r"(?P<authors>"
        r"[A-Z][a-zà-ÿA-Z\-']+"
        r"(?:[,;]?\s+(?:and\s+)?"
        r"[A-Z][\.\w]*\.?"
        r"(?:\s+[A-Z][a-zà-ÿA-Z\-']+)?"
        r")*"
        r")"
        r"\s*\((?P<year>\d{4}[a-z]?)\)"
        r"[\.,;]?\s*"
        r"(?P<title>[^.]+\.)"
        r"\s*(?P<venue>[^.]*(?:\.\s|$))?"
    ),
]

# ── Identifier extraction (NBER, DOI, arXiv, FEDS, etc.) ───────────

IDENTIFIER_PATTERNS = {
    "NBER": re.compile(
        r"(?:NBER\s+[Ww]orking\s+[Pp]aper|"
        r"National\s+Bureau\s+of\s+Economic\s+Research)"
        r"[,\s]*(?:[Nn]o\.?\s*)?(\d{3,6})",
    ),
    "NBER_w": re.compile(r"nber\.org/papers/w(\d+)"),
    "DOI": re.compile(r"(?:doi(?:\.org)?[:/]\s*)(10\.\d{4,}/[^\s,;]+)"),
    "arXiv": re.compile(r"arXiv[:\s]*(\d{4}\.\d{4,5})"),
    "SSRN": re.compile(r"(?:SSRN|ssrn\.com/abstract=)(\d+)"),
    "FEDS": re.compile(
        r"(?:FEDS|Finance\s+and\s+Economics\s+Discussion\s+Series)"
        r"[,\s]*(?:[Nn]o\.?\s*)?(\d{4}[-–]\d{2,4})"
    ),
    "Staff_Report": re.compile(r"(?:Staff\s+Report|SR)[,\s]*(?:[Nn]o\.?\s*)?(\d{2,4})"),
    "Federal_Reserve": re.compile(
        r"(?:Federal\s+Reserve\s+(?:Board|Bank|Bulletin|System))"
    ),
}

# ── URL resolvers ───────────────────────────────────────────────────


def _resolve_url(category: str, identifier: str) -> str:
    """Return a landing-page URL for known source types."""
    resolvers = {
        "NBER": lambda id_: f"https://www.nber.org/papers/w{id_}",
        "NBER_w": lambda id_: f"https://www.nber.org/papers/w{id_}",
        "DOI": lambda id_: f"https://doi.org/{id_}",
        "arXiv": lambda id_: f"https://arxiv.org/abs/{id_}",
        "SSRN": lambda id_: (
            f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={id_}"
        ),
        "FEDS": lambda id_: (
            f"https://www.federalreserve.gov/econres/feds/"
            f"files/{id_.replace('-', '')}.pdf"
        ),
    }
    resolver = resolvers.get(category)
    if resolver and identifier:
        return resolver(identifier)
    return ""


def _resolve_download_url(category: str, identifier: str) -> str:
    """Return a direct-download PDF/file URL for known source types."""
    resolvers = {
        "NBER": lambda id_: (
            f"https://www.nber.org/system/files/working_papers/w{id_}/w{id_}.pdf"
        ),
        "NBER_w": lambda id_: (
            f"https://www.nber.org/system/files/working_papers/w{id_}/w{id_}.pdf"
        ),
        "arXiv": lambda id_: f"https://arxiv.org/pdf/{id_}",
        "FEDS": lambda id_: (
            f"https://www.federalreserve.gov/econres/feds/"
            f"files/{id_.replace('-', '')}.pdf"
        ),
    }
    resolver = resolvers.get(category)
    if resolver and identifier:
        return resolver(identifier)
    return ""


# ── Text extraction ─────────────────────────────────────────────────


def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from a PDF file."""
    try:
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        return "\n".join(p.page_content for p in pages)
    except ImportError:
        logger.warning("PyPDFLoader not available; skipping %s", pdf_path.name)
        return ""


def _extract_references_section(text: str) -> str:
    """Try to isolate the References/Bibliography section.

    Falls back to the full text if no section header is found.
    """
    # Common section headers in these documents
    patterns = [
        re.compile(
            r"\n\s*"
            r"(?:\d+[\. \t]+|[IVX]+[\. \t]+)?"
            r"(?:References|Bibliography|Works\s+Cited|Literature\s+Cited)"
            r"\s*\n",
            re.IGNORECASE,
        ),
    ]
    for pat in patterns:
        match = pat.search(text)
        if match:
            # Take everything after the header
            section = text[match.end() :]
            # Stop at the next major section header (if any)
            next_section = re.search(
                r"\n\s*(?:Appendix|Annex|Tables?|Figures?|Endnotes?)\s*\n",
                section,
                re.IGNORECASE,
            )
            if next_section:
                section = section[: next_section.start()]
            return section
    return text


# ── Core parser ─────────────────────────────────────────────────────


def parse_citations(text: str, source_pdf: str = "") -> list[Citation]:
    """Parse citation strings from text and resolve identifiers.

    Parameters
    ----------
    text : str
        Raw text (ideally from the References section).
    source_pdf : str
        Filename of the source PDF for provenance tracking.

    Returns
    -------
    list[Citation]
        De-duplicated list of parsed citations.
    """
    citations: list[Citation] = []
    seen: set[tuple[str, str, str]] = set()

    for pattern in CITATION_PATTERNS:
        for match in pattern.finditer(text):
            authors = match.group("authors").strip().rstrip(",")
            year = match.group("year").strip()
            title = match.group("title").strip().strip('"').strip("\u201c\u201d")
            venue = (match.group("venue") or "").strip().rstrip(".")

            # De-duplicate by (authors_lower, year, first 40 chars of title)
            dedup_key = (authors.lower(), year, title[:40].lower())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Skip very short or obviously non-citation matches
            if len(title) < 5 or len(authors) < 3:
                continue

            # Skip matches where the title is a URL fragment
            if title.startswith(("http", "www.", "//")):
                continue

            # Skip if authors look like a sentence fragment
            if authors.lower().startswith(("see ", "the ", "for ", "in ", "as ")):
                continue

            # Try to identify source category
            context = f"{title} {venue}"
            category = "Unknown"
            identifier = ""

            for cat, id_pat in IDENTIFIER_PATTERNS.items():
                id_match = id_pat.search(context)
                if id_match:
                    category = cat
                    identifier = id_match.group(1) if id_match.lastindex else ""
                    break

            # If category is still unknown, try the full match span + surrounding
            if category == "Unknown":
                span_start = max(0, match.start() - 20)
                span_end = min(len(text), match.end() + 200)
                surrounding = text[span_start:span_end]
                for cat, id_pat in IDENTIFIER_PATTERNS.items():
                    id_match = id_pat.search(surrounding)
                    if id_match:
                        category = cat
                        identifier = id_match.group(1) if id_match.lastindex else ""
                        break

            url = _resolve_url(category, identifier)
            download = _resolve_download_url(category, identifier)

            citations.append(
                Citation(
                    authors=authors,
                    year=year,
                    title=title,
                    venue=venue,
                    source_pdf=source_pdf,
                    category=category,
                    resolved_url=url,
                    download_url=download,
                    identifier=identifier,
                )
            )

    return citations


def parse_pdf(pdf_path: Path) -> list[Citation]:
    """Extract and parse citations from a single PDF.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.

    Returns
    -------
    list[Citation]
        Parsed citations.
    """
    logger.info("Parsing citations from %s", pdf_path.name)
    text = _extract_text_from_pdf(pdf_path)
    if not text:
        return []

    ref_text = _extract_references_section(text)
    logger.info(
        "References section: %d chars (full text: %d chars)",
        len(ref_text),
        len(text),
    )
    return parse_citations(ref_text, source_pdf=pdf_path.name)


def parse_all_pdfs(raw_dir: Path | None = None) -> list[Citation]:
    """Parse citations from every PDF in the corpus.

    Parameters
    ----------
    raw_dir : Path | None
        Directory containing downloaded PDFs.  Defaults to
        ``corpus/raw_data``.

    Returns
    -------
    list[Citation]
        Combined, de-duplicated citation list.
    """
    if raw_dir is None:
        raw_dir = Path(__file__).parent / "raw_data"

    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in %s", raw_dir)
        return []

    logger.info("Found %d PDF(s) in %s", len(pdfs), raw_dir)

    all_citations: list[Citation] = []
    seen: set[str] = set()

    for pdf_path in pdfs:
        for cit in parse_pdf(pdf_path):
            dedup_key = (cit.authors.lower(), cit.year, cit.title[:40].lower())
            key_str = str(dedup_key)
            if key_str not in seen:
                seen.add(key_str)
                all_citations.append(cit)

    return all_citations


# ── CSV export ──────────────────────────────────────────────────────


def export_csv(citations: list[Citation], output_path: Path) -> None:
    """Write citations to a CSV file.

    Parameters
    ----------
    citations : list[Citation]
        Parsed citations.
    output_path : Path
        Destination CSV path.
    """
    if not citations:
        logger.warning("No citations to export")
        return

    fieldnames = list(asdict(citations[0]).keys())
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for cit in citations:
            writer.writerow(asdict(cit))

    logger.info("Wrote %d citations to %s", len(citations), output_path)


# ── Rich display ────────────────────────────────────────────────────


def display_summary(citations: list[Citation]) -> None:
    """Print a rich summary table to the console."""
    banner = r"""[bold cyan]
   ____ _ _        _   _
  / ___(_) |_ __ _| |_(_) ___  _ __  ___
 | |   | | __/ _` | __| |/ _ \| '_ \/ __|
 | |___| | || (_| | |_| | (_) | | | \__ \
  \____|_|\__\__,_|\__|_|\___/|_| |_|___/
[/bold cyan]"""
    console.print(Panel.fit(banner, border_style="grey39"))

    # Category breakdown
    categories: dict[str, int] = {}
    downloadable = 0
    with_direct = 0
    for cit in citations:
        categories[cit.category] = categories.get(cit.category, 0) + 1
        if cit.resolved_url:
            downloadable += 1
        if cit.download_url:
            with_direct += 1

    summary_table = Table(
        title="Citation Summary",
        show_header=True,
        header_style="bold",
    )
    summary_table.add_column("Category", style="cyan")
    summary_table.add_column("Count", justify="right")
    summary_table.add_column("Landing URL", justify="right", style="green")
    summary_table.add_column("Direct PDF", justify="right", style="blue")

    for cat in sorted(categories):
        with_url = sum(1 for c in citations if c.category == cat and c.resolved_url)
        with_pdf = sum(1 for c in citations if c.category == cat and c.download_url)
        summary_table.add_row(cat, str(categories[cat]), str(with_url), str(with_pdf))

    summary_table.add_section()
    summary_table.add_row(
        "TOTAL",
        str(len(citations)),
        str(downloadable),
        str(with_direct),
        style="bold",
    )
    console.print(summary_table)
    console.print()

    # Detailed listing
    detail_table = Table(
        title="Parsed Citations",
        show_header=True,
        header_style="bold",
        show_lines=True,
    )
    detail_table.add_column("#", justify="right", style="dim", width=4)
    detail_table.add_column("Authors", max_width=25)
    detail_table.add_column("Year", width=6)
    detail_table.add_column("Title", max_width=40)
    detail_table.add_column("Category", style="cyan", width=16)
    detail_table.add_column("Landing URL", style="green", max_width=40)
    detail_table.add_column("Download URL", style="blue", max_width=45)

    for i, cit in enumerate(citations, 1):
        url_display = cit.resolved_url if cit.resolved_url else "[dim]—[/dim]"
        dl_display = cit.download_url if cit.download_url else "[dim]—[/dim]"
        detail_table.add_row(
            str(i),
            cit.authors,
            cit.year,
            cit.title,
            cit.category,
            url_display,
            dl_display,
        )

    console.print(detail_table)


# ── CLI ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse academic citations from corpus PDFs",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Parse a single PDF instead of all PDFs in corpus/raw_data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "citations.csv",
        help="Output CSV path (default: corpus/citations.csv)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV export, just display results",
    )
    args = parser.parse_args()

    if args.pdf:
        citations = parse_pdf(args.pdf)
    else:
        citations = parse_all_pdfs()

    if not citations:
        console.print("[yellow]No citations found.[/yellow]")
        sys.exit(0)

    display_summary(citations)

    if not args.no_csv:
        export_csv(citations, args.output)
        console.print(f"\n[green]✓ Exported to {args.output}[/green]")


if __name__ == "__main__":
    main()
