"""
Data preprocessing for the Disneyland Reviews dataset.

  1. Drop duplicate reviews
  2. Parse missing Year_Month values out of the review text itself
  3. Split Year_Month into integer Year / Month columns
  4. Cast dtypes

On top of the notebook logic, this module adds a Pydantic validation pass so
malformed rows are caught and dropped (with a report) before they ever reach
the vector store.
"""
from __future__ import annotations

import logging
import re

import pandas as pd
from pydantic import ValidationError

from src.schemas import RawReview

logger = logging.getLogger(__name__)

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

MONTH_PATTERN = r"\b(?:" + "|".join(MONTH_MAP.keys()) + r")\b"
YEAR_PATTERN = r"\b(20\d{2}|195[6-9]|19[6-9]\d|200\d)\b"


def parse_year_month(row: pd.Series) -> str | None:
    """Recover a missing Year_Month by scanning the free-text review body."""
    if row["Year_Month"] != "missing":
        return row["Year_Month"]

    month_match = re.search(MONTH_PATTERN, row["Review_Text"], re.IGNORECASE)
    year_match = re.search(YEAR_PATTERN, row["Review_Text"])

    if not (month_match and year_match):
        return None

    month_num = MONTH_MAP[month_match.group(0).lower()]
    return f"{year_match.group(0)}-{month_num}"


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load the raw Kaggle CSV (latin1 encoded, per the source dataset)."""
    df = pd.read_csv(csv_path, encoding="latin1")
    logger.info("Loaded %d raw rows from %s", len(df), csv_path)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's cleaning steps: dedupe, recover dates, split, cast."""
    df = df.copy()

    df = df.drop_duplicates(subset="Review_Text", keep="first")

    df["Year_Month"] = df.apply(parse_year_month, axis=1)
    df = df.dropna(subset=["Year_Month"])

    year_month = df["Year_Month"].str.split("-", expand=True, n=1)
    df["Year"] = year_month[0]
    df["Month"] = year_month[1]
    df = df.drop(columns="Year_Month")

    df["Year"] = df["Year"].astype("int64")
    df["Month"] = df["Month"].astype("int64")

    df = df.reset_index(drop=True)
    logger.info("Cleaned dataframe: %d rows remain", len(df))
    return df


def validate_raw_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Validate rows against RawReview *before* cleaning (checks the raw schema:
    Rating range, Branch is one of the 3 known parks, non-blank review text).

    Returns (valid_subset_of_df, num_dropped).
    """
    valid_mask = []
    for _, row in df.iterrows():
        try:
            RawReview(**row.to_dict())
            valid_mask.append(True)
        except ValidationError:
            valid_mask.append(False)

    valid_df = df[pd.Series(valid_mask, index=df.index)]
    num_dropped = len(df) - len(valid_df)
    if num_dropped:
        logger.warning("Dropped %d rows failing raw schema validation", num_dropped)
    return valid_df, num_dropped
