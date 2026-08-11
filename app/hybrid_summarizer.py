import logging
from typing import List, Optional, Tuple
from functools import lru_cache
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from app.config import get_settings
from app.utils import chunk_text, count_words, truncate_to_token_limit

logger = logging.getLogger(__name__)
settings = get_settings()

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class HybridSummarizer:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.device_id = 0
        else:
            self.device = torch.device("cpu")
            self.device_id = -1

        logger.info(f"Loading BART summarization model on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.summarizer_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(settings.summarizer_model)
        self.model.to(self.device)
        self.model.eval()

    @lru_cache(maxsize=128)
    def summarize(self, text: str, target_words: Optional[int] = None,
                  keywords: Optional[Tuple[str]] = None) -> Tuple[str, int, int]:
        """
        Summarize text using BART with chunking.
        keywords are not used in pure abstractive mode, but kept for compatibility.
        """
        original_word_count = count_words(text)
        if target_words is None:
            target_words = max(30, int(original_word_count * settings.default_summary_ratio))

        # Chunk text into manageable pieces (max_words from config)
        # chunks = chunk_text(text, max_words=settings.max_chunk_words, overlap=settings.overlap_ratio)
        chunks = chunk_text(text, max_words=settings.max_chunk_words, overlap=settings.overlap_ratio)

        if len(chunks) == 1:
            summary = self._abstractive_summarize(chunks[0], target_words)
        else:
            chunk_summaries = []
            per_chunk_target = max(30, target_words // len(chunks))
            for chunk in chunks:
                s = self._abstractive_summarize(chunk, per_chunk_target)
                chunk_summaries.append(s)
            combined = " ".join(chunk_summaries)
            # Refine if combined is still too long
            if count_words(combined) > target_words * 1.5:
                summary = self._abstractive_summarize(combined, target_words)
            else:
                summary = combined

        # Trim to exact target words by sentences
        sentences = nltk.sent_tokenize(summary)
        final_summary = []
        current_len = 0
        for sent in sentences:
            wc = count_words(sent)
            if current_len + wc <= target_words:
                final_summary.append(sent)
                current_len += wc
            else:
                break
        final_text = " ".join(final_summary)
        return final_text, len(final_summary), count_words(final_text)

    def _abstractive_summarize(self, text: str, target_words: int) -> str:
        """Summarize a single text chunk using BART's generate()"""
        # Truncate to model's max length
        truncated = truncate_to_token_limit(text, self.tokenizer, 1024 - 50)
        inputs = self.tokenizer(truncated, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        max_len = min(target_words + 20, 1024)
        min_len = max(target_words // 2, 10)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                num_beams=4,
                early_stopping=True
            )
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary