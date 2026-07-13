"""
Pydantic models used across the pipeline:
- Raw/processed review validation (data quality gate before indexing)
- API-shaped request/response models for the chatbot
- Structured LLM-judge output for evaluation

Keeping these in one module makes every boundary in the app (CSV -> DataFrame,
DataFrame -> Documents, question -> answer, eval -> scores) explicitly typed.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Branch(str, Enum):
    CALIFORNIA = "Disneyland_California"
    PARIS = "Disneyland_Paris"
    HONG_KONG = "Disneyland_HongKong"


class SentimentLabel(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"


class RawReview(BaseModel):
    """Validates a single row straight out of the Kaggle CSV."""

    Review_ID: int
    Rating: int = Field(ge=1, le=5)
    Year_Month: str
    Reviewer_Location: str
    Review_Text: str = Field(min_length=1)
    Branch: Branch

    @field_validator("Review_Text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Review_Text cannot be blank")
        return v.strip()


class ProcessedReview(BaseModel):
    """Row shape after cleaning + sentiment scoring, right before indexing."""

    Review_ID: int
    Rating: int = Field(ge=1, le=5)
    Reviewer_Location: str
    Review_Text: str
    Branch: Branch
    Year: int = Field(ge=1990, le=datetime.now().year)
    Month: int = Field(ge=1, le=12)
    Sentiment: float = Field(ge=-1.0, le=1.0)
    Sentiment_Label: SentimentLabel


class RetrievedDocument(BaseModel):
    """A single retrieved review, normalized for display in the UI."""

    review_id: int
    branch: str
    rating: int
    sentiment_label: str
    snippet: str
    score: Optional[float] = None


class ChatQuery(BaseModel):
    """Validated shape of a user question hitting the RAG pipeline."""

    question: str = Field(min_length=3, max_length=1000)
    pipeline: Literal["simple", "hybrid_rerank"] = "hybrid_rerank"
    top_k: int = Field(default=4, ge=1, le=10)
    branch_filter: Optional[list[Branch]] = None


class ChatResponse(BaseModel):
    """Validated shape of what the RAG pipeline returns to the UI."""

    answer: str
    pipeline: Literal["simple", "hybrid_rerank"]
    sources: list[RetrievedDocument]
    latency_seconds: float


class EvalJudgeScore(BaseModel):
    """Structured output we force the LLM-judge to return (parsed from JSON)."""

    faithfulness: int = Field(ge=1, le=5, description="Is the answer grounded in the retrieved reviews?")
    relevance: int = Field(ge=1, le=5, description="Does the answer address the question?")
    justification: str = Field(max_length=400)


class EvalResult(BaseModel):
    """One row of the evaluation comparison table."""

    question: str
    pipeline: Literal["simple", "hybrid_rerank"]
    answer: str
    latency_seconds: float
    keyword_recall: float
    faithfulness: int
    relevance: int
