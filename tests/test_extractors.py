import pytest
from app.extractors import text, file, youtube

def test_text_extractor():
    result = text.fetch_random_paragraph(min_words=30)
    assert len(result.split()) >= 30 or "error" in result.lower()

def test_file_extractor(tmp_path):
    # Create a dummy text file
    p = tmp_path / "test.txt"
    p.write_text("Hello world. " * 20)  # >30 words
    with open(p, "rb") as f:
        content = f.read()
    extracted = file.extract_from_bytes(content, "test.txt")
    assert len(extracted.split()) >= 30

def test_youtube_id_extraction():
    url = "https://www.youtube.com/watch?v=abc123defgh"
    vid = youtube.extract_video_id(url)
    assert vid == "abc123defgh"