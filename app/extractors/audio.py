import logging
import tempfile
import os
from app.config import get_settings
from app.utils import count_words

logger = logging.getLogger(__name__)
settings = get_settings()

_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            _whisper_model = whisper.load_model(settings.whisper_model)
        except Exception as e:
            logger.warning(f"Whisper model failed to load: {e}")
            _whisper_model = None
    return _whisper_model

def transcribe_audio(file_bytes: bytes, file_ext: str = ".mp3") -> str:
    model = get_whisper_model()
    if model is None:
        logger.error("Whisper model not available.")
        return ""
    try:
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        result = model.transcribe(tmp_path)
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)