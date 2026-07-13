"""
One-shot build script:
  1. Runs the ingestion pipeline (clean + validate + sentiment)
  2. Saves the processed CSV
  3. Builds and persists the Chroma dense index
  4. Pickles the document list for the BM25 sparse index

Run this once locally before deploying / running the Streamlit app:
    python scripts/build_index.py
"""
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion import (  # noqa: E402
    dataframe_to_documents,
    run_ingestion_pipeline,
    save_processed_data,
)
from src.vectorstore import build_chroma_index, save_bm25_corpus  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Running ingestion pipeline...")
    df = run_ingestion_pipeline()
    save_processed_data(df)

    logger.info("Converting %d rows to documents...", len(df))
    documents = dataframe_to_documents(df)

    logger.info("Building Chroma dense index (this embeds every document once)...")
    build_chroma_index(documents)

    logger.info("Saving BM25 corpus...")
    save_bm25_corpus(documents)

    logger.info("Done. Indexes are ready - you can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
