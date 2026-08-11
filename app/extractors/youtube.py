import re
import tempfile
import os
import logging
from typing import Optional
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube
from app.config import get_settings
from app.utils import count_words

logger = logging.getLogger(__name__)
settings = get_settings()

_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            # Monkey-patch ctypes.util.find_library to return a dummy path if None
            import ctypes.util
            original_find_library = ctypes.util.find_library
            def patched_find_library(name):
                result = original_find_library(name)
                if result is None and name == 'c':
                    # Fallback to a common path on Windows
                    return 'msvcrt.dll'
                return result
            ctypes.util.find_library = patched_find_library

            import whisper
            _whisper_model = whisper.load_model(settings.whisper_model)
        except Exception as e:
            logger.warning(f"Whisper model failed to load: {e}")
            _whisper_model = None
            # Restore original function (optional)
            ctypes.util.find_library = original_find_library
    return _whisper_model

def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/live\/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id: str, prefer_languages: list = ['en', 'hi']) -> Optional[str]:
    # Method 1: YouTubeTranscriptApi
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        for lang in prefer_languages:
            try:
                transcript = transcript_list.find_transcript([lang])
                break
            except:
                continue
        if not transcript:
            transcript = transcript_list.find_transcript([t.language_code for t in transcript_list])
        data = transcript.fetch()
        text = " ".join([entry['text'] for entry in data])
        if count_words(text) >= 30:
            return text
        else:
            logger.warning(f"Transcript too short ({count_words(text)} words), trying fallbacks.")
    except (TranscriptsDisabled, NoTranscriptFound, Exception) as e:
        logger.info(f"Transcript API failed: {e}, falling back to pytube.")

    # Method 2: pytube captions
    try:
        yt = YouTube(video_id)
        caption = None
        for code in prefer_languages:
            if code in yt.captions:
                caption = yt.captions[code]
                break
        if not caption and yt.captions:
            caption = next(iter(yt.captions.values()))
        if caption:
            srt = caption.generate_srt_captions()
            text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', srt)
            text = re.sub(r'\n+', ' ', text).strip()
            if count_words(text) >= 30:
                return text
            else:
                logger.warning(f"Pytube captions too short ({count_words(text)} words).")
    except Exception as e:
        logger.info(f"Pytube captions failed: {e}")

    # Method 3: Whisper audio transcription (last resort)
    whisper_model = get_whisper_model()
    if whisper_model:
        try:
            yt = YouTube(video_id)
            audio_stream = yt.streams.filter(only_audio=True).order_by('abr').last()
            if audio_stream:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    audio_file = tmp.name
                try:
                    audio_stream.download(filename=audio_file)
                    result = whisper_model.transcribe(audio_file)
                    text = result["text"].strip()
                    if count_words(text) >= 30:
                        return text
                    else:
                        logger.warning(f"Whisper transcription too short.")
                finally:
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
        except Exception as e:
            logger.error(f"Whisper fallback failed: {e}")

    return None