import pytest
from app.hybrid_summarizer import HybridSummarizer
from app.utils import count_words

@pytest.fixture
def summarizer():
    return HybridSummarizer()

def test_summarizer_basic(summarizer):
    text = "This is a test text. It has multiple sentences. We want to see if it summarizes properly. The model should produce a concise summary."
    summary, num_sent, num_words = summarizer.summarize(text, target_words=10)
    assert num_words <= 15  # approximate
    assert len(summary) > 0

def test_summarizer_with_keywords(summarizer):
    text = "Python is a programming language. It is widely used in data science. Machine learning is popular."
    summary, _, _ = summarizer.summarize(text, target_words=8, keywords=["Python", "data"])
    assert "Python" in summary or "data" in summary