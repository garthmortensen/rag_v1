"""Main ingestion pipeline.

Loads raw corpus files, logs a summary, and prepares documents
for downstream embedding and storage.
"""

import logging
import sys

from src.ingestion.loaders import load_directory, LOADER_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def run() -> list:
    """Run the ingestion pipeline: load all supported files from raw_data/."""
    logger.info("Starting ingestion pipeline")

    docs = load_directory()

    if not docs:
        logger.warning("No documents were loaded. Check corpus/raw_data/ for supported files.")
        return docs

    # Log a breakdown by source file
    # counts numbers of docs per source file, e.g.:
    # {}"corpus/raw_data/credit_risk_models.pdf": 641,     # 641 pages
    # "corpus/raw_data/market_risk_models.pdf": 296,       # 296 pages
    # "corpus/raw_data/baseline_domestic_final.csv": 13,   # 13 rows
    # "corpus/raw_data/transparency_qas.html": 1,}         # 1 whole page
    sources = {}
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1  # =1 for every new doc in the file

    logger.info("Loaded %d document(s) from %d file(s):", len(docs), len(sources))
    for source, count in sorted(sources.items()):
        logger.info("  %s  →  %d chunk(s)", source, count)

    supported = ", ".join(LOADER_MAP.keys())
    logger.info("Supported file types: %s", supported)

    return docs


if __name__ == "__main__":
    run()
