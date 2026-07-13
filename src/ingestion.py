"""
End-to-end ingestion: raw CSV -> cleaned + validated + sentiment-scored
DataFrame -> LangChain Documents ready for embedding.
"""
from __future__ import annotations

import logging

import pandas as pd
from langchain_core.documents import Document

from config import settings
from src.preprocessing import clean_dataframe, load_raw_data, validate_raw_rows
from src.sentiment import add_sentiment_columns

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(
    raw_csv_path: str | None = None,
    sample_size: int | None = None,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: load -> validate raw rows -> clean -> sentiment score ->
    (optional) sample -> return a DataFrame ready to persist / index.
    """
    raw_csv_path = raw_csv_path or str(settings.raw_data_path)
    sample_size = settings.sample_size if sample_size is None else sample_size
    random_seed = random_seed or settings.random_seed

    df = load_raw_data(raw_csv_path)

    df, num_dropped = validate_raw_rows(df)
    logger.info("Raw validation dropped %d malformed rows", num_dropped)

    df = clean_dataframe(df)
    df = add_sentiment_columns(df)

    if sample_size is not None and len(df) > sample_size:
        # Stratify by Branch so all three parks stay represented in the demo index.
        # Deliberately avoids groupby(...).apply(...): recent pandas versions exclude
        # the grouping column from what's passed into the applied function, which
        # silently dropped 'Branch' from the result. Plain iteration avoids that.
        sampled_parts = []
        for _, group in df.groupby("Branch"):
            n = min(len(group), max(1, int(sample_size * len(group) / len(df))))
            sampled_parts.append(group.sample(n=n, random_state=random_seed))
        df = pd.concat(sampled_parts).reset_index(drop=True)
        logger.info("Sampled dataframe down to %d rows for indexing", len(df))

    return df


def save_processed_data(df: pd.DataFrame, output_path: str | None = None) -> None:
    output_path = output_path or str(settings.processed_data_path)
    df.to_csv(output_path, index=False)
    logger.info("Saved processed data to %s", output_path)


def load_processed_data(path: str | None = None) -> pd.DataFrame:
    path = path or str(settings.processed_data_path)
    return pd.read_csv(path)


def dataframe_to_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convert each review row into a LangChain Document. Reviews are short
    (a few sentences to a couple paragraphs) so we index them whole rather
    than chunking - chunking would fragment sentiment/rating context.
    """
    documents = []
    for _, row in df.iterrows():
        metadata = {
            "review_id": int(row["Review_ID"]),
            "rating": int(row["Rating"]),
            "branch": str(row["Branch"]),
            "reviewer_location": str(row["Reviewer_Location"]),
            "year": int(row["Year"]),
            "month": int(row["Month"]),
            "sentiment": float(row["Sentiment"]),
            "sentiment_label": str(row["Sentiment_Label"]),
        }
        documents.append(Document(page_content=str(row["Review_Text"]), metadata=metadata))
    return documents
