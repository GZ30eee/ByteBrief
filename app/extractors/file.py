"""Text extraction from uploaded files: TXT, PDF, DOCX, PPT, images (OCR)."""
import os
import logging
from io import BytesIO
import PyPDF2
import docx
from pptx import Presentation
from PIL import Image
import numpy as np
import easyocr
from app.utils import count_words, get_ocr_reader

logger = logging.getLogger(__name__)

def extract_from_bytes(file_bytes: bytes, filename: str) -> str:
    """
    Determine file type by extension and extract text.
    Returns extracted text or empty string on failure.
    """
    ext = filename.split('.')[-1].lower()
    text = ""
    try:
        if ext == 'txt':
            text = file_bytes.decode('utf-8', errors='ignore')
        elif ext == 'pdf':
            reader = PyPDF2.PdfReader(BytesIO(file_bytes))
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        elif ext == 'docx':
            doc = docx.Document(BytesIO(file_bytes))
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext in ['ppt', 'pptx']:
            prs = Presentation(BytesIO(file_bytes))
            text = "\n".join([shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, 'text')])
        elif ext in ['jpg', 'jpeg', 'png']:
            reader = get_ocr_reader()
            if reader is None:
                logger.error("OCR reader not available.")
                return ""
            image = Image.open(BytesIO(file_bytes))
            result = reader.readtext(np.array(image), detail=0)
            text = " ".join(result)
        else:
            logger.warning(f"Unsupported file type: {ext}")
        return text.strip()
    except Exception as e:
        logger.error(f"File extraction error for {filename}: {e}")
        return ""