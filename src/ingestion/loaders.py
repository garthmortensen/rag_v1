"""Document loaders for raw corpus files.

Supports HTML, CSV, PDF, Excel, text, Markdown, JSON, Word, and PowerPoint.
Each loader wraps a LangChain Community loader and returns a list of
LangChain Document objects with text + metadata.

load_directory() is the entry point, and it calls load_file() for each supported file in the target directory.

"""

import os
import glob
import logging

# langchain-community provides pre-built file readers for common formats
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

logger = logging.getLogger(__name__)

# Map file extensions to their LangChain loader class
LOADER_MAP = {
    ".html": BSHTMLLoader,  # 1 document per file
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
    all_docs = []
    files = sorted(glob.glob(os.path.join(directory, "*")))

    for filepath in files:
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
