import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import requests
from PIL import Image
from io import BytesIO
import time
import re
import textwrap

# Initialize summarization pipeline with error handling
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    st.error(f"Failed to load summarization model: {str(e)}")
    st.stop()

def extract_video_id(url):
    """Extract YouTube video ID from URL with robust pattern matching"""
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

def get_video_info(video_id):
    """Get video thumbnail, title, channel with better error handling"""
    try:
        response = requests.get(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            thumbnail_url = data['thumbnail_url'].replace('hqdefault', 'maxresdefault')
            
            # Try to get higher resolution thumbnail first
            try:
                thumbnail_response = requests.get(thumbnail_url, timeout=5)
                thumbnail_response.raise_for_status()
                thumbnail = Image.open(BytesIO(thumbnail_response.content))
            except:
                # Fallback to default thumbnail
                thumbnail_response = requests.get(data['thumbnail_url'], timeout=5)
                thumbnail = Image.open(BytesIO(thumbnail_response.content))
            
            return {
                'thumbnail': thumbnail,
                'title': data['title'],
                'channel': data['author_name'],
                'views': "N/A (API key required)"
            }
        return None
    except Exception as e:
        st.error(f"Couldn't fetch video info: {str(e)}")
        return None

def get_transcript(video_id):
    """Get transcript with proper error handling and progress feedback"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Finding video subtitles...")
        progress_bar.progress(20)
        
        # Check if transcript exists first
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        if not any(t.is_generated for t in transcript_list):
            raise Exception("No human-generated subtitles available")
        
        status_text.text("📝 Extracting transcript...")
        progress_bar.progress(50)
        
        # Get transcript with fallback to generated if needed
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        
        text = " ".join([entry['text'] for entry in transcript])
        
        if len(text.split()) < 10:
            raise Exception("Transcript too short (min 10 words required)")
        
        status_text.text("✅ Transcript extracted!")
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return text
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error: {str(e)}. This video may not have accessible subtitles.")
        return None

def chunk_text(text, max_chunk_size=1024):
    """Split text into chunks that won't exceed model's maximum input size"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def summarize_text(text, max_length=150, min_length=30):
    """Summarize text with chunking and error handling"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🧠 Analyzing video content...")
        progress_bar.progress(20)
        
        # Chunk the text if too long
        chunks = chunk_text(text)
        summaries = []
        
        status_text.text("✂️ Generating summary (this may take a while)...")
        
        for i, chunk in enumerate(chunks):
            progress = 20 + (i * 60 // len(chunks))
            progress_bar.progress(progress)
            
            try:
                summary = summarizer(
                    chunk,
                    max_length=max_length//len(chunks),
                    min_length=min_length//len(chunks),
                    do_sample=False
                )
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                st.warning(f"Couldn't summarize one section: {str(e)}")
                summaries.append(chunk[:max_length//len(chunks)] + "...")
        
        full_summary = " ".join(summaries)
        
        # Final refinement if we have multiple chunks
        if len(chunks) > 1:
            try:
                refined = summarizer(
                    full_summary,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                full_summary = refined[0]['summary_text']
            except:
                pass  # Keep the concatenated version if refinement fails
        
        status_text.text("🎉 Summary ready!")
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return full_summary
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Summarization failed: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="YouTube Summarizer", layout="wide")
    st.title("🎬 YouTube Video Summarizer Pro")
    st.markdown("""
    <style>
    .small-font { font-size:14px !important; }
    .highlight { background-color: #f0f2f6; border-radius: 5px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
        1. Paste a YouTube video URL (must have subtitles/CC)
        2. Click "Get Transcript & Summarize"
        3. Adjust summary length
        4. Generate and download your summary
        """)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
    with col2:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        if st.button("Clear All"):
            st.session_state.clear()
            st.rerun()
    
    if url:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL. Please check the URL and try again.")
            return
        
        video_info = get_video_info(video_id)
        if not video_info:
            st.error("Couldn't fetch video information. The video might be private or unavailable.")
            return
        
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(video_info['thumbnail'], use_container_width=True, caption="Video Thumbnail")
            with col2:
                st.subheader(video_info['title'])
                st.markdown(f"""
                **Channel:** {video_info['channel']}  
                **Views:** {video_info['views']}
                """)
                
                if st.button("Get Transcript & Summarize", key="get_transcript"):
                    with st.spinner('Processing video...'):
                        transcript = get_transcript(video_id)
                    
                    if transcript:
                        st.session_state.transcript = transcript
                        word_count = len(transcript.split())
                        st.session_state.word_count = word_count
                        
                        with st.expander("View Full Transcript", expanded=False):
                            st.text_area("Transcript", transcript, height=200, label_visibility="collapsed")
                        
                        st.success(f"✅ Extracted {word_count} words from video transcript")
        
        if 'transcript' in st.session_state:
            st.markdown("---")
            st.subheader("📝 Summary Options")
            
            # Calculate reasonable length limits
            word_count = st.session_state.word_count
            max_summary = min(300, max(50, word_count // 3))
            default_summary = min(150, max_summary)
            
            summary_length = st.slider(
                "Summary length (words)",
                min_value=30,
                max_value=max_summary,
                value=default_summary,
                help="Longer summaries preserve more details"
            )
            
            if st.button("Generate Summary", type="primary"):
                if 'transcript' not in st.session_state:
                    st.warning("Please extract transcript first")
                    return
                
                with st.spinner('Generating summary...'):
                    summary = summarize_text(
                        st.session_state.transcript,
                        max_length=summary_length,
                        min_length=max(30, summary_length//2)
                    )
                
                if summary:
                    st.markdown("---")
                    st.subheader("📌 Your Summary")
                    st.write(summary)
                    
                    # Download options
                    st.download_button(
                        "📥 Download Summary",
                        data=summary,
                        file_name=f"summary_{video_id}.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()