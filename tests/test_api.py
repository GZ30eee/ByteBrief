import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_summarize_text():
    response = client.post("/api/summarize/text", json={
        "text": "This is a test. It has enough words. We want to see if it works. The model should return a summary."
    })
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert data["num_words"] > 0

def test_invalid_youtube():
    response = client.post("/api/summarize/youtube", data={"url": "invalid"})
    assert response.status_code == 400