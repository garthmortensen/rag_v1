"""Document loaders for raw corpus files.

Supports HTML, CSV, PDF, Excel, text, Markdown, JSON, Word, and PowerPoint.
Each loader wraps a LangChain Community loader and returns a list of
LangChain Document objects with text + metadata.

load_directory() is the entry point, and it calls load_file() for each supported file in the target directory.

"""

import os
import glob
import logging
import re
import codecs

# langchain-community provides pre-built file readers for common formats
from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from rich.console import Console
from rich.panel import Panel

from src.ingestion.pdf_section_splitter import has_section_headers, load_pdf_by_section

logger = logging.getLogger(__name__)


_META_CHARSET_RE = re.compile(
    r"charset\s*=\s*['\"]?\s*([a-zA-Z0-9_\-:]+)",
    re.IGNORECASE,
)


def _decode_html_bytes(data: bytes, default_encoding: str = "utf-8") -> tuple[str, str]:
    """Decode HTML bytes into text.

    Strategy:
    1) Honor BOMs when present (utf-8-sig/utf-16/utf-32)
    2) Try strict UTF-8
    3) Look for a <meta charset=...> declaration in the first ~4KB
    4) Fall back to common legacy encodings (cp1252, latin-1)

    Returns (text, encoding_used).
    """
    if data.startswith(codecs.BOM_UTF8):
        return data.decode("utf-8-sig"), "utf-8-sig"
    if data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
        return data.decode("utf-16"), "utf-16"
    if data.startswith(codecs.BOM_UTF32_LE) or data.startswith(codecs.BOM_UTF32_BE):
        return data.decode("utf-32"), "utf-32"

    try:
        return data.decode(default_encoding), default_encoding
    except UnicodeDecodeError:
        pass

    head = data[:4096].decode("ascii", errors="ignore")
    m = _META_CHARSET_RE.search(head)
    if m:
        enc = m.group(1).strip().strip("\"'")
        try:
            codecs.lookup(enc)
            return data.decode(enc, errors="strict"), enc
        except Exception:
            pass

    for enc in ("cp1252", "latin-1"):
        try:
            return data.decode(enc), enc
        except Exception:
            continue

    return data.decode(default_encoding, errors="replace"), f"{default_encoding}-replace"


class RobustHTMLLoader:
    """HTML loader tolerant to non-UTF8 source files.

    Mirrors the behavior of LangChain's BSHTMLLoader (1 document per file),
    but avoids crashing on legacy encodings.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        with open(self.file_path, "rb") as f:
            data = f.read()

        html_text, encoding_used = _decode_html_bytes(data)

        try:
            soup = BeautifulSoup(html_text, "lxml")
            text = soup.get_text(separator="\n")
        except Exception:
            text = html_text

        text = text.strip()
        return [
            Document(
                page_content=text,
                metadata={"source": self.file_path, "encoding": encoding_used},
            )
        ]

# Map file extensions to their LangChain loader class
LOADER_MAP = {
    ".html": RobustHTMLLoader,  # 1 document per file
    ".csv": CSVLoader,  # 1 document per row
    ".pdf": PyPDFLoader,  # 1 document per page
    ".xlsx": UnstructuredExcelLoader,  # 1 document per sheet
    ".xls": UnstructuredExcelLoader,  # 1 document per sheet
    ".txt": TextLoader,  # 1 document per file
    ".md": UnstructuredMarkdownLoader,  # 1 document per file
    ".json": JSONLoader,  # 1 document per file
    ".docx": UnstructuredWordDocumentLoader,  # 1 document per file
    ".pptx": UnstructuredPowerPointLoader,  # 1 document per file
}

RAW_DATA_DIR = os.path.join("corpus", "raw_data")

console = Console()


def print_ascii_banner():
    console.print(
        Panel.fit(
            """[bold green1]
               .__  .__  __    __  .__                     .___                    
  ____________ |  | |__|/  |__/  |_|__| ____    ____     __| _/____   ____   ______
 /  ___/\____ \|  | |  \   __\   __\  |/    \  / ___\   / __ |/  _ \_/ ___\ /  ___/
 \___ \ |  |_> >  |_|  ||  |  |  | |  |   |  \/ /_/  > / /_/ (  <_> )  \___ \___ \ 
/____  >|   __/|____/__||__|  |__| |__|___|  /\___  /  \____ |\____/ \___  >____  >
     \/ |__|                               \//_____/        \/           \/     \/ 
[/bold green1]
 --------------------------------
""",
            border_style="grey39",
        )
    )


def load_file(filepath: str) -> list:
    """Load a single file and return a list of Documents.

    Selects the appropriate loader based on the file extension.
    Raises ValueError for unsupported file types.
    """
    ext = os.path.splitext(filepath)[1].lower()
    # given file extension, get the corresponding loader class
    document_loader = LOADER_MAP.get(ext)

    if document_loader is None:
        raise ValueError(f"Unsupported file type: {ext} ({filepath})")

    # For PDFs with structured "Model Documentation:" headers, use the
    # section-aware splitter so each subsection becomes its own Document
    # with section/subsection metadata.  This produces much better RAG
    # chunks than the default one-Document-per-page approach.
    if ext == ".pdf" and has_section_headers(filepath):
        logger.info(f"Loading {filepath} with section-aware PDF splitter")
        docs = load_pdf_by_section(filepath)
        return docs

    logger.info(f"Loading {filepath} with {document_loader.__name__}")
    # e.g. PyPDFLoader(filepath).load() returns a list of Documents, one per page
    docs = document_loader(filepath).load()

    # Ensure every document carries the source filepath in metadata
    for doc in docs:
        # Use setdefault to avoid overwriting existing 'source' metadata if present
        doc.metadata.setdefault("source", filepath)

    return docs


def load_directory(directory: str = RAW_DATA_DIR) -> list:
    """Load all supported files from a directory.

    Returns a flat list of Documents from every supported file found.
    Unsupported file types are skipped with a warning.
    """
    print_ascii_banner()

    all_docs = []
    files = sorted(glob.glob(os.path.join(directory, "**", "*"), recursive=True))

    for filepath in files:
        if os.path.isdir(filepath):
            continue
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in LOADER_MAP:
            logger.warning(f"Skipping unsupported file type: {filepath}")
            continue

        try:
            # Read this file into a list of documents
            docs = load_file(filepath)
            # Add them to our running list
            # append will create a list of lists, e.g. [[doc1, doc2], [doc3, doc4], ...]
            # extend will add elements to the single list, e.g. [doc1, doc2, doc3, ...]
            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} document(s) from {filepath}")
        except Exception as e:
            # If anything goes wrong, log it and move on to the next file
            logger.error(f"Failed to load {filepath}: {e}")

    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs
