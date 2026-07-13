"""
Sentiment analysis using NLTK's VADER, following DisneylandReviewAnalysis.ipynb.

VADER is used (rather than a transformer model) because it's fast, free,
needs no GPU, and is exactly what the reference notebook used - keeping the
RAG app's sentiment metadata consistent with the EDA notebook.
"""
from __future__ import annotations

import logging
from collections import Counter
from functools import lru_cache

import ssl

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

from config import settings

logger = logging.getLogger(__name__)

# Extra domain stopwords, same list used in the notebook's word-frequency EDA
EXTRA_STOPWORDS = {
    "disney", "park", "disneyland", "get", "go", "one", "would", "place",
    "went", "even", "us", "day", "really", "see", "also", "like", "much",
    "visit", "could", "back", "parks", "great", "good", "many",
}


def _allow_unverified_ssl_for_nltk() -> None:
    """
    Work around 'CERTIFICATE_VERIFY_FAILED' errors some local Python installs hit
    when downloading NLTK corpora (common on macOS installs missing a cert bundle).
    Safe here since NLTK's corpora are public, static, non-sensitive data.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context


def ensure_nltk_data() -> None:
    """Download the NLTK corpora needed for sentiment + tokenization (idempotent)."""
    for package in ("vader_lexicon", "stopwords", "punkt", "punkt_tab"):
        try:
            nltk.download(package, quiet=True)
        except Exception as exc:
            logger.warning("Could not download nltk package %s (%s); retrying with unverified SSL", package, exc)
            try:
                _allow_unverified_ssl_for_nltk()
                nltk.download(package, quiet=True)
            except Exception as retry_exc:  # pragma: no cover - genuine network outage
                logger.warning("Still could not download nltk package %s: %s", package, retry_exc)


@lru_cache(maxsize=1)
def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    ensure_nltk_data()
    return SentimentIntensityAnalyzer()


def compute_sentiment(text: str) -> float:
    """Return VADER's compound sentiment score in [-1, 1]."""
    analyzer = get_sentiment_analyzer()
    return analyzer.polarity_scores(text)["compound"]


def label_sentiment(score: float) -> str:
    if score >= settings.sentiment_positive_threshold:
        return "Positive"
    if score <= settings.sentiment_negative_threshold:
        return "Negative"
    return "Neutral"


def add_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add Sentiment (float) and Sentiment_Label (str) columns to the dataframe."""
    df = df.copy()
    df["Sentiment"] = df["Review_Text"].apply(compute_sentiment)
    df["Sentiment_Label"] = df["Sentiment"].apply(label_sentiment)
    return df


def top_words_by_sentiment(df: pd.DataFrame, threshold: float = 0.5, top_n: int = 10) -> pd.DataFrame:
    """
    Reproduces the notebook's "most common words in high vs low sentiment
    reviews" table, used in the Streamlit dataset-insights tab.
    """
    ensure_nltk_data()
    stop_words = set(stopwords.words("english")) | EXTRA_STOPWORDS

    def clean_and_tokenize(texts: pd.Series) -> list[str]:
        words: list[str] = []
        for text in texts:
            tokens = word_tokenize(text.lower())
            words += [w for w in tokens if w.isalpha() and w not in stop_words]
        return words

    low_reviews = df[df["Sentiment"] < threshold]["Review_Text"]
    high_reviews = df[df["Sentiment"] >= threshold]["Review_Text"]

    low_common = Counter(clean_and_tokenize(low_reviews)).most_common(top_n)
    high_common = Counter(clean_and_tokenize(high_reviews)).most_common(top_n)

    high_df = pd.DataFrame(high_common, columns=["Word", "Count"])
    high_df["Sentiment"] = "High"
    low_df = pd.DataFrame(low_common, columns=["Word", "Count"])
    low_df["Sentiment"] = "Low"

    return pd.concat([high_df, low_df], ignore_index=True)
