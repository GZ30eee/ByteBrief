"""Plain text extraction and random Wikipedia fallback."""
import logging
import requests
import time
from app.utils import count_words

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "ByteBrief/1.0 (https://github.com/your-repo; contact@example.com)"
}

# A static fallback in case Wikipedia is unavailable
FALLBACK_TEXT = (
    "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, "
    "especially computer systems. These processes include learning, reasoning, and self‑correction. "
    "Applications of AI include expert systems, speech recognition, and machine vision. AI research "
    "is divided into subfields such as machine learning, deep learning, natural language processing, "
    "and robotics. While AI has grown rapidly, it raises ethical concerns about job displacement, bias, "
    "and privacy. Many experts believe AI will transform every sector of the economy."
)

def request_with_retry(url, params=None, max_retries=3, initial_delay=2, backoff=2):
    """
    Make a GET request with exponential backoff retry on 429 or 5xx errors.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                logger.warning(f"Rate limited (429). Retry {attempt+1}/{max_retries} after {delay}s")
                time.sleep(delay)
                delay *= backoff
            else:
                logger.warning(f"HTTP {resp.status_code} on attempt {attempt+1}")
                # Non-429 errors: retry with shorter delay
                time.sleep(1)
        except Exception as e:
            logger.warning(f"Request exception: {e}. Retry {attempt+1}/{max_retries}")
            time.sleep(delay)
            delay *= backoff
    return None

def fetch_random_paragraph(min_words: int = 30, max_attempts: int = 3) -> str:
    """
    Fetch a random Wikipedia article's first section with at least `min_words`.
    Falls back to a static text if the API is unavailable.
    """
    for attempt in range(max_attempts):
        try:
            # Step 1: Get a random page title
            title_resp = request_with_retry(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "random",
                    "rnnamespace": 0,
                    "rnlimit": 1,
                    "format": "json"
                },
                max_retries=2,
                initial_delay=3  # start with a longer delay
            )
            if title_resp is None:
                logger.warning("Title request failed after retries, falling back to summary endpoint")
                return _fetch_random_summary(min_words)

            title_data = title_resp.json()
            random_pages = title_data.get("query", {}).get("random", [])
            if not random_pages:
                continue
            page_title = random_pages[0]["title"]

            # Step 2: Fetch the full extract
            extract_resp = request_with_retry(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "titles": page_title,
                    "format": "json"
                },
                max_retries=2,
                initial_delay=2
            )
            if extract_resp is None:
                continue

            extract_data = extract_resp.json()
            pages = extract_data.get("query", {}).get("pages", {})
            for page_id, page_info in pages.items():
                if page_id != "-1":
                    text = page_info.get("extract", "")
                    if count_words(text) >= min_words:
                        return text
                    else:
                        logger.info(f"Attempt {attempt+1}: '{page_title}' has only {count_words(text)} words.")
                        break
            # If we get here, article was too short; wait before next attempt
            time.sleep(2)

        except Exception as e:
            logger.warning(f"Random fetch attempt {attempt+1} failed: {e}")
            time.sleep(2)

    # Fallback to the random summary endpoint
    summary_result = _fetch_random_summary(min_words)
    if summary_result.startswith("Could not"):
        # If even that fails, return the static fallback
        return FALLBACK_TEXT
    return summary_result

def _fetch_random_summary(min_words: int = 30) -> str:
    """
    Fallback: fetch a random summary via /random/summary endpoint.
    """
    resp = request_with_retry(
        "https://en.wikipedia.org/api/rest_v1/page/random/summary",
        max_retries=3,
        initial_delay=3,
        backoff=2
    )
    if resp and resp.status_code == 200:
        text = resp.json().get("extract", "")
        if count_words(text) >= min_words:
            return text
        else:
            logger.warning(f"Random summary too short: {count_words(text)} words.")
    else:
        logger.warning("Random summary endpoint unavailable.")
    return "Could not fetch a paragraph with at least 30 words. Please try again."