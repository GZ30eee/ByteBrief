"""Helper functions: word count, chunking, token truncation, export, caching."""
import nltk
import re
from io import BytesIO
import docx
import fitz  # PyMuPDF
import easyocr
from app.config import get_settings

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

settings = get_settings()

def count_words(text: str) -> int:
    return len(text.split())

def chunk_text(text: str, max_words: int = 1000, overlap: float = 0.1) -> list:
    """
    Split text into chunks by sentence, each <= max_words.
    Overlap between chunks is overlap * max_words (in words).
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0
    overlap_words = int(max_words * overlap)
    
    for sent in sentences:
        sent_len = count_words(sent)
        if current_len + sent_len <= max_words:
            current_chunk.append(sent)
            current_len += sent_len
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep last few sentences for overlap
                keep_sentences = []
                temp_len = 0
                for s in reversed(current_chunk):
                    wc = count_words(s)
                    if temp_len + wc <= overlap_words:
                        keep_sentences.insert(0, s)
                        temp_len += wc
                    else:
                        break
                current_chunk = keep_sentences
                current_len = temp_len
            # start new chunk with current sentence
            current_chunk.append(sent)
            current_len += sent_len
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def truncate_to_token_limit(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate text to fit within tokenizer's max length."""
    tokens = tokenizer(text, truncation=False, return_tensors="pt")["input_ids"][0]
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return tokenizer.decode(tokens, skip_special_tokens=True)
    return text

def convert_to_bullets(summary: str) -> str:
    sentences = nltk.sent_tokenize(summary)
    return "\n\n".join([f"• {s}" for s in sentences])

def export_summary(summary: str, format_type: str):
    if format_type == "DOCX":
        doc = docx.Document()
        doc.add_paragraph(summary)
        bio = BytesIO()
        doc.save(bio)
        bio.seek(0)
        return bio, "summary.docx"
    elif format_type == "PDF":
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((50, 50), summary)
        bio = BytesIO()
        pdf.save(bio)
        bio.seek(0)
        return bio, "summary.pdf"
    else:  # TXT
        bio = BytesIO()
        bio.write(summary.encode('utf-8'))
        bio.seek(0)
        return bio, "summary.txt"

# OCR reader singleton
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            _ocr_reader = easyocr.Reader(settings.ocr_languages)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to init OCR: {e}")
            _ocr_reader = None
    return _ocr_reader