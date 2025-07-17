import streamlit as st
import easyocr
from PIL import Image
import numpy as np
import time
from summarizer import Summarizer  # Using BERT extractive summarization
import os

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize summarizer
model = Summarizer()

def extract_text_from_image(image):
    """Extract text from image using EasyOCR"""
    img = Image.open(image)
    img_array = np.array(img)
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Processing image...")
    progress_bar.progress(20)
    time.sleep(0.5)
    
    status_text.text("Extracting text...")
    progress_bar.progress(50)
    
    # Perform OCR
    result = reader.readtext(img_array)
    extracted_text = " ".join([res[1] for res in result])
    
    status_text.text("Text extraction complete!")
    progress_bar.progress(100)
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    return extracted_text

def summarize_text(text, ratio=0.2):
    """Summarize text using BERT extractive summarization"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Analyzing your precious text...")
    progress_bar.progress(30)
    time.sleep(0.5)
    
    status_text.text("Generating summary...")
    progress_bar.progress(70)
    
    # Generate summary
    summary = model(text, ratio=ratio)
    
    status_text.text("Summary ready!")
    progress_bar.progress(100)
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    return summary

def main():
    st.title("📝 Image Text Extractor & Summarizer")
    st.markdown("Upload an image containing text, extract the text, and get a summary!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        # Extract text button
        if st.button("Extract Text"):
            with st.spinner('Extracting text from image...'):
                extracted_text = extract_text_from_image(uploaded_file)
            
            st.session_state.extracted_text = extracted_text
            st.success("Text extracted successfully!")
            
            # Display extracted text
            st.subheader("Extracted Text")
            st.text_area("", extracted_text, height=200)
            
            # Count words
            word_count = len(extracted_text.split())
            st.write(f"Word count: {word_count}")
            
            if word_count < 20:
                st.warning("Please upload an image with at least 20 words for better summarization.")
            else:
                st.session_state.word_count = word_count
    
    # Summarization section
    if 'extracted_text' in st.session_state and st.session_state.word_count >= 20:
        st.markdown("---")
        st.subheader("📌 Summarization Options")
        
        # Slider for summary length
        max_words = min(100, st.session_state.word_count)  # Cap at 100 words max
        summary_length = st.slider(
            "Select summary length (in words)", 
            min_value=10, 
            max_value=max_words,
            value=min(20, max_words)
        )
        
        if st.button("Generate Summary"):
            ratio = summary_length / st.session_state.word_count
            summary = summarize_text(st.session_state.extracted_text, ratio=ratio)
            
            st.subheader("Summary")
            st.write(summary)

if __name__ == "__main__":
    main()