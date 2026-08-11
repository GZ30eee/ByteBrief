import logging
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import pandas as pd
from transformers import pipeline
from keybert import KeyBERT
from gensim import corpora, models
import spacy
from functools import lru_cache
from app.config import get_settings
from app.utils import count_words

logger = logging.getLogger(__name__)
settings = get_settings()

@lru_cache(maxsize=1)
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Downloading...")
        try:
            spacy.cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Failed to download spaCy model: {e}")
            return None

@lru_cache(maxsize=1)
def load_sentiment():
    try:
        return pipeline("sentiment-analysis", model=settings.sentiment_model)
    except Exception as e:
        logger.error(f"Failed to load sentiment model: {e}")
        return None

@lru_cache(maxsize=1)
def load_keybert():
    try:
        return KeyBERT(model='paraphrase-MiniLM-L6-v2')
    except Exception as e:
        logger.error(f"Failed to load KeyBERT: {e}")
        return None

_nlp = load_nlp()
_sentiment = load_sentiment()
_kw_model = load_keybert()

class Analytics:
    @staticmethod
    def get_sentiment(text: str) -> dict:
        if _sentiment is None:
            return {"label": "N/A", "score": 0.0}
        try:
            tokens = _sentiment.tokenizer(text, truncation=True, max_length=512, return_tensors="pt")["input_ids"][0]
            truncated = _sentiment.tokenizer.decode(tokens, skip_special_tokens=True)
            result = _sentiment(truncated)[0]
            return {"label": result['label'], "score": result['score']}
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"label": "Error", "score": 0.0}
    
    @staticmethod
    def get_keywords(text: str, top_n: int = 10) -> list:
        if _kw_model is None:
            return []
        try:
            return _kw_model.extract_keywords(text, top_n=top_n)
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    @staticmethod
    def get_topics(text: str, num_topics: int = 3) -> list:
        if _nlp is None:
            return []
        try:
            doc = _nlp(text)
            tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
            if len(tokens) < 10:
                return []
            dictionary = corpora.Dictionary([tokens])
            corpus = [dictionary.doc2bow(tokens)]
            lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
            return lda.print_topics()
        except Exception as e:
            logger.error(f"Topic modeling failed: {e}")
            return []
    
    @staticmethod
    def generate_wordcloud(text: str, width=800, height=400, background='white'):
        try:
            wc = WordCloud(width=width, height=height, background_color=background).generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            return fig
        except Exception as e:
            logger.error(f"Word cloud generation failed: {e}")
            return None
    
    @staticmethod
    def plot_topic_distribution(topics: list):
        if not topics:
            return None
        try:
            data = [{"Topic": f"Topic {i}", "Words": topic[1]} for i, topic in enumerate(topics)]
            df = pd.DataFrame(data)
            return px.bar(df, x="Topic", y="Words", text="Words")
        except Exception as e:
            logger.error(f"Topic plot failed: {e}")
            return None