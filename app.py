import os
import streamlit as st
import nltk

# ✅ Set NLTK data directory for Streamlit Cloud compatibility
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

@st.cache_resource
def download_nltk_punkt():
    nltk.download("punkt", download_dir=NLTK_DATA_DIR)

download_nltk_punkt()

from summarizer import (
    extract_text_from_file,
    load_model,
    sentence_chunking,
    apply_style_prompt,
    summarize_all_chunks,
    post_process,
    evaluate_summary,
    save_summary_as_pdf
)

st.set_page_config(page_title="Smart Summarizer", layout="wide")
st.title("🧠 AI Text Summarizer")

# Upload or Paste Text
uploaded_file = st.file_uploader("📂 Upload your document (.txt, .pdf, .docx, .pptx)", type=["txt", "pdf", "docx", "pptx"])
text_input = st.text_area("📝 Or paste your text here:", height=200)

col1, col2, col3 = st.columns(3)

with col1:
    model_choice = st.selectbox("Choose Model", ["BART", "T5", "Pegasus"])

with col2:
    length_mode = st.radio("Summary Length", ["Auto", "Manual"])

with col3:
    style = st.selectbox("Summary Style", ["Default", "Bullet Points", "Headlines"])

# Manual length controls
if length_mode == "Manual":
    max_len = st.slider("Max Length", 50, 512, 130)
    min_len = st.slider("Min Length", 10, max_len, 30)
else:
    max_len = min_len = None

# Summarize button
if st.button("✨ Generate Summary"):

    # Input text from file or textarea
    if uploaded_file:
        os.makedirs("temp", exist_ok=True)
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        input_text = extract_text_from_file(file_path)
    elif text_input.strip():
        input_text = text_input
    else:
        st.warning("Please upload a file or paste text.")
        st.stop()

    word_count = len(input_text.split())
    st.info(f"Detected ~{word_count} words.")

    # Load model
    model_map = {"BART": "1", "T5": "2", "Pegasus": "3"}
    tokenizer, model, model_name = load_model(model_map[model_choice])

    # Auto length
    if length_mode == "Auto":
        max_len = max(int(word_count * 0.3), 30)
        min_len = max(int(word_count * 0.1), 10)

    # Summary style
    style_map = {"Default": "1", "Bullet Points": "2", "Headlines": "3"}

    # Chunking and summarization
    with st.spinner("⏳ Summarizing..."):
        chunks = sentence_chunking(input_text, max_words=500)
        styled_chunks = [apply_style_prompt(chunk, style_map[style], model_name) for chunk in chunks]
        summaries = summarize_all_chunks(styled_chunks, tokenizer, model, max_len, min_len)
        final_summary = post_process("\n".join(summaries), style_map[style])

    st.subheader("📝 Final Summary:")
    st.text_area("Summary Output", final_summary, height=300)

    # Downloads
    col4, col5 = st.columns(2)
    with col4:
        st.download_button("💾 Download TXT", final_summary, file_name="summary.txt")

    with col5:
        save_summary_as_pdf(final_summary, "summary.pdf")
        with open("summary.pdf", "rb") as f:
            st.download_button("📄 Download PDF", f, file_name="summary.pdf")

    # ROUGE Evaluation
    if st.checkbox("📊 Show ROUGE Evaluation"):
        with st.spinner("Calculating ROUGE scores..."):
            scores = evaluate_summary(final_summary, input_text)
            for k, v in scores.items():
                st.markdown(f"**{k.upper()}**: `{v:.4f}`")
