from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import tempfile
from app.hybrid_summarizer import HybridSummarizer
from app.extractors import youtube, file, text
from app.analytics import Analytics
from app.models import SummarizeRequest, SummarizeResponse, AnalyticsResponse
from app.config import get_settings
import logging

app = FastAPI(title="ByteBrief API", version="2.0")
logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize summarizer and analytics
summarizer = HybridSummarizer()
analytics = Analytics()

class SummarizeTextRequest(BaseModel):
    text: str
    target_words: Optional[int] = None
    keywords: Optional[List[str]] = None

@app.post("/api/summarize/text", response_model=SummarizeResponse)
async def summarize_text(req: SummarizeTextRequest):
    try:
        summary, num_sentences, num_words = summarizer.summarize(
            req.text, req.target_words, req.keywords
        )
        # Also compute analytics if needed
        word_count = len(req.text.split())
        return SummarizeResponse(
            summary=summary,
            num_sentences=num_sentences,
            num_words=num_words,
            original_word_count=word_count,
            reduction=round((1 - num_words/word_count)*100, 2) if word_count else 0
        )
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize/upload")
async def summarize_file(file: UploadFile = File(...), target_words: Optional[int] = Form(None)):
    # Use file extractor
    try:
        content = await file.read()
        extracted = file.extract_from_bytes(content, file.filename)
        if not extracted:
            raise HTTPException(400, "Could not extract text from file.")
        summary, num_sentences, num_words = summarizer.summarize(extracted, target_words)
        return SummarizeResponse(
            summary=summary,
            num_sentences=num_sentences,
            num_words=num_words,
            original_word_count=len(extracted.split()),
            reduction=...
        )
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/summarize/youtube")
async def summarize_youtube(url: str = Form(...), target_words: Optional[int] = Form(None)):
    video_id = youtube.extract_video_id(url)
    if not video_id:
        raise HTTPException(400, "Invalid YouTube URL")
    transcript = youtube.get_transcript(video_id)
    if not transcript:
        raise HTTPException(404, "No transcript could be extracted.")
    summary, num_sentences, num_words = summarizer.summarize(transcript, target_words)
    return SummarizeResponse(...)