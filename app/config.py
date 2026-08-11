from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    summarizer_model: str = "facebook/bart-large-cnn"
    extractive_model: str = "bert-base-uncased"  # not used now
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    whisper_model: str = "tiny"
    ocr_languages: list = ["en"]
    
    default_summary_ratio: float = 0.2
    max_chunk_words: int = 1000          # <-- add
    overlap_ratio: float = 0.1           # <-- add
    
    cache_ttl_seconds: int = 3600
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings():
    return Settings()