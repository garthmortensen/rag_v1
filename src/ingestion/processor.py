"""Main ingestion pipeline.

Downloads raw files from data_sources.csv, loads them into
Documents, splits into chunks using RecursiveCharacterTextSplitter,
embeds chunks via HuggingFace, and upserts them into a persistent
ChromaDB collection at corpus/vector_db/.
"""

import logging
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CFG
from src.embedding.model import embed_and_store
from src.ingestion.downloader import download_files
from src.ingestion.loaders import load_directory, LOADER_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Chunking defaults (driven by config.txt)
DEFAULT_CHUNK_SIZE_IN_CHARS = int(CFG["chunk_size"])
DEFAULT_CHUNK_OVERLAP_IN_CHARS = int(CFG["chunk_overlap"])


def chunk_documents(
    docs: list,
    chunk_size: int = DEFAULT_CHUNK_SIZE_IN_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_IN_CHARS,
) -> list:
    """Split documents into smaller chunks for embedding.

    Uses LangChain's RecursiveCharacterTextSplitter, which tries to
    split on paragraph → sentence → word boundaries in order to keep
    semantically related text together.

    Each resulting chunk inherits the metadata of its parent document.
    """

    # the magic
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # count characters (default)
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} document(s) into {len(chunks)} chunk(s)  (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def run(
    chunk_size: int = DEFAULT_CHUNK_SIZE_IN_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_IN_CHARS,
) -> list:
    """Run the full ingestion pipeline.

    0. Download raw files from corpus/data_sources.csv → corpus/raw_data/.
    1. Load all supported files from corpus/raw_data/.
    2. Split loaded documents into chunks.
    3. Embed chunks and upsert into ChromaDB.
    4. Return the list of chunks.
    """
    logger.info("Starting ingestion pipeline")

    # Step 0: Download
    download_files()

    # Step 1: Load
    docs = load_directory()

    if not docs:
        logger.warning("No documents were loaded. Check corpus/raw_data/ for supported files.")
        return docs

    # Log a breakdown by source file
    # counts numbers of docs per source file, e.g.:
    # {"corpus/raw_data/credit_risk_models.pdf": 641,     # 641 pages
    #  "corpus/raw_data/market_risk_models.pdf": 296,      # 296 pages
    #  "corpus/raw_data/baseline_domestic_final.csv": 13,  # 13 rows
    #  "corpus/raw_data/transparency_qas.html": 1}         # 1 whole page
    sources = {}
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1  # +1 for every new doc in the file

    logger.info(f"Loaded {len(docs)} document(s) from {len(sources)} file(s):")
    for source, count in sorted(sources.items()):
        logger.info(f"  {source}  →  {count} doc(s)")

    supported = ", ".join(LOADER_MAP.keys())
    logger.info(f"Supported file types: {supported}")

    # Step 2: Chunk
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Step 3: Embed and store (collection driven by config.txt)
    embed_and_store(chunks, collection_name=str(CFG["collection_name"]))

    return chunks


if __name__ == "__main__":
    run()
