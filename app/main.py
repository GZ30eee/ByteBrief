"""ByteBrief - Streamlit UI entry point."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from app.config import get_settings
from app.hybrid_summarizer import HybridSummarizer
from app.extractors import text as text_extractor, file as file_extractor, youtube as youtube_extractor, audio as audio_extractor
from app.analytics import Analytics
from app.utils import count_words, convert_to_bullets, export_summary
import logging

st.set_page_config(page_title="ByteBrief - Smart Summarizer", layout="wide")
settings = get_settings()

# Initialize summarizer (cached)
@st.cache_resource
def load_summarizer():
    return HybridSummarizer()

summarizer = load_summarizer()

# UI
st.title("📄 ByteBrief - Multi-Modal Summarizer")
st.markdown("Summarize text from various sources with hybrid AI (BERT + BART).")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Input", "📎 File Upload", "🎬 YouTube", "📊 Analytics"])

# --- Tab 1: Text Input ---
with tab1:
    st.subheader("Enter or paste text")
    if "text" not in st.session_state:
        st.session_state.text = ""
    text_area = st.text_area("Content", height=250, value=st.session_state.text)
    if st.button("📥 Get Random Wikipedia Article"):
        with st.spinner("Fetching..."):
            rand_text = text_extractor.fetch_random_paragraph()
            st.session_state.text = rand_text
            st.rerun()
    if text_area and count_words(text_area) < 30:
        st.warning(f"Text has only {count_words(text_area)} words. Need at least 30.")
    st.session_state.text = text_area

# --- Tab 2: File Upload ---
with tab2:
    st.subheader("Upload a file")
    uploaded = st.file_uploader("Choose file", type=["txt","pdf","docx","ppt","pptx","jpg","jpeg","png","mp3","wav"])
    if uploaded:
        with st.spinner("Extracting text..."):
            file_bytes = uploaded.read()
            extracted = file_extractor.extract_from_bytes(file_bytes, uploaded.name)
            if extracted:
                st.session_state.text = extracted
                st.success(f"Extracted {count_words(extracted)} words.")
                st.text_area("Extracted content", extracted, height=200)
            else:
                st.error("Could not extract text from file.")

# --- Tab 3: YouTube ---
with tab3:
    st.subheader("YouTube Video Summarizer")
    url = st.text_input("Enter YouTube URL")
    if st.button("Get Transcript & Summarize"):
        if url:
            video_id = youtube_extractor.extract_video_id(url)
            if not video_id:
                st.error("Invalid YouTube URL")
            else:
                with st.spinner("Extracting transcript..."):
                    transcript = youtube_extractor.get_transcript(video_id)
                if transcript:
                    st.session_state.text = transcript
                    st.success(f"Transcript extracted ({count_words(transcript)} words).")
                    with st.expander("View transcript"):
                        st.write(transcript)
                else:
                    st.error("Could not extract transcript from this video. Reasons:\n"
                            "- No captions available (YouTubeTranscriptApi & pytube).\n"
                            "- Whisper (audio fallback) is not available on your system.\n"
                            "Try a different video with captions, or install Whisper properly.")

# --- Tab 4: Analytics ---
with tab4:
    st.subheader("AI/ML Insights")
    st.markdown("""
    - **Hybrid Summarization**: Extractive (BERT) + Abstractive (BART)
    - **Chunking**: Handles long documents via hierarchical summarization
    - **Topic Modeling**: LDA (Gensim)
    - **Keyword Extraction**: KeyBERT
    - **Sentiment Analysis**: DistilBERT
    - **OCR**: EasyOCR (English)
    - **Speech-to-Text**: Whisper
    """)
    if st.session_state.text:
        with st.expander("Run analytics on current text"):
            if st.button("Analyze"):
                ana = Analytics()
                sent = ana.get_sentiment(st.session_state.text)
                st.write(f"Sentiment: {sent['label']} (score: {sent['score']:.2f})")
                keywords = ana.get_keywords(st.session_state.text, top_n=5)
                st.write("Keywords:", ", ".join([kw[0] for kw in keywords]))
                topics = ana.get_topics(st.session_state.text)
                if topics:
                    fig = ana.plot_topic_distribution(topics)
                    if fig:
                        st.plotly_chart(fig)
                fig_wc = ana.generate_wordcloud(st.session_state.text)
                st.pyplot(fig_wc)
    else:
        st.info("No text loaded yet.")

# --- Common summarization section ---
if st.session_state.text and count_words(st.session_state.text) >= 30:
    st.markdown("---")
    st.subheader("⚙️ Summarize")
    col1, col2 = st.columns(2)
    with col1:
        target_words = st.slider("Target summary length (words)", min_value=30, max_value=count_words(st.session_state.text), value=min(150, count_words(st.session_state.text)))
    with col2:
        output_format = st.selectbox("Format", ["Paragraph", "Bullet Points"])
    
    # Optional keywords
    ana = Analytics()
    keywords = ana.get_keywords(st.session_state.text, top_n=10)
    selected_keywords = st.multiselect("Focus keywords (optional)", [kw[0] for kw in keywords], default=[])

    if st.button("🚀 Generate Summary"):
        with st.spinner("Summarizing..."):
            summary, num_sent, num_words = summarizer.summarize(st.session_state.text, target_words, tuple(selected_keywords))
        if output_format == "Bullet Points":
            summary = convert_to_bullets(summary)
        st.subheader("📌 Summary")
        st.write(summary)
        st.caption(f"{num_sent} sentences, {num_words} words")
        
        # Export
        export_format = st.selectbox("Export as", ["TXT", "DOCX", "PDF"])
        if st.button("Download"):
            file_bytes, filename = export_summary(summary, export_format)
            st.download_button(label=f"Download {export_format}", data=file_bytes, file_name=filename)

else:
    st.info("Please provide at least 30 words to enable summarization.")

# Clear button
if st.button("🗑️ Clear all"):
    st.session_state.text = ""
    st.rerun()