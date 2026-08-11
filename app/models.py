"""Pydantic schemas for API responses."""
from pydantic import BaseModel
from typing import Optional, List

class SummarizeRequest(BaseModel):
    text: str
    target_words: Optional[int] = None
    keywords: Optional[List[str]] = None

class SummarizeResponse(BaseModel):
    summary: str
    num_sentences: int
    num_words: int
    original_word_count: int
    reduction: float

class AnalyticsResponse(BaseModel):
    sentiment_label: str
    sentiment_score: float
    keywords: List[str]
    topics: List[dict]  # e.g., [{"topic": 0, "words": "..."}]