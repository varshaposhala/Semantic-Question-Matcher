# app.py (Users must provide their own API key)

import os
import streamlit as st
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List

st.cache_data.clear()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Semantic Question Matcher",
    page_icon="🔑",
    layout="wide"
)

# --- API KEY HANDLING ---
st.sidebar.title("Configuration")
st.sidebar.markdown("This tool requires your own Google Gemini API key to function.")
api_key_input = st.sidebar.text_input(
    "Enter your Google Gemini API Key here",
    type="password",
    help="You can get your API key from Google AI Studio."
)

api_configured = False
if api_key_input:
    try:
        genai.configure(api_key=api_key_input)
        api_configured = True
    except Exception as e:
        st.sidebar.error(f"Failed to configure API: {e}")

# --- CORE FUNCTIONS (with Caching) ---
@st.cache_data(show_spinner=False)
def get_embeddings(texts: List[str]) -> np.ndarray:
    if not texts or not any(text.strip() for text in texts):
        return np.array([])
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL, content=texts)
        return np.array(result['embedding'])
    except Exception as e:
        st.error(f"An error occurred while getting embeddings. Please check your API key and network connection. Details: {e}")
        return np.array([])

# --- UI LAYOUT ---
st.title("🔑 Semantic Question Matcher")
st.write("Enter your master questions and their subtopics. Then, enter the questions you want to match. **This tool uses your own API key for all calculations.**")

if not api_configured:
    st.info("Please enter a valid Google Gemini API key in the sidebar to begin.")
    st.stop()

EMBEDDING_MODEL = 'models/embedding-001'

col1, col2 = st.columns(2)

with col1:
    st.header("Master Questions")
    master_questions_text = st.text_area(
        "Enter one entry per line using your custom format",
        height=300,
        placeholder="Format 1: Why is my animation lagging?<br>\tSUB_TOPIC_JS_PERFORMANCE\nFormat 2: How do I center a div?\tSUB_TOPIC_CSS_LAYOUT"
    )

with col2:
    st.header("Questions to Match")
    generated_questions_text = st.text_area(
        "Enter the new questions you want to find matches for.",
        height=300,
        placeholder="e.g., My animation isn't smooth.\nWhat's the best way to center an element with CSS?"
    )

st.markdown("---")

similarity_threshold = st.slider(
    "Similarity Score Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.01,
    help="Only show matches with a score equal to or higher than this value."
)

if st.button("Find Similar Questions", type="primary", use_container_width=True):
    master_questions = []
    master_subtopics = []
    delimiter1 = "<br>\tSUB_TOPIC_"
    delimiter2 = "\tSUB_TOPIC_"

    # Allow multiline or ----separated master questions
    if "---" in master_questions_text:
        master_lines = [q.strip() for q in master_questions_text.split('---') if q.strip()]
    else:
        master_lines = [line.strip() for line in master_questions_text.splitlines() if line.strip()]

    for line in master_lines:
        question, subtopic = "", "N/A"
        if delimiter1 in line:
            parts = line.split(delimiter1, 1)
            question, subtopic = parts[0].strip(), parts[1].strip()
        elif delimiter2 in line:
            parts = line.split(delimiter2, 1)
            question, subtopic = parts[0].strip(), parts[1].strip()
        else:
            question = line.strip()
        master_questions.append(question)
        master_subtopics.append(subtopic)

    # Accept both line-by-line and --- separated format
    if "---" in generated_questions_text:
        generated_questions = [q.strip() for q in generated_questions_text.split('---') if q.strip()]
    else:
        generated_questions = [q.strip() for q in generated_questions_text.splitlines() if q.strip()]

    if not master_questions or not generated_questions:
        st.warning("Please enter questions in both text areas.")
    else:
        with st.spinner("Calculating embeddings and similarities using your API key..."):
            master_embeddings = get_embeddings(master_questions)
            generated_embeddings = get_embeddings(generated_questions)

            if master_embeddings.size > 0 and generated_embeddings.size > 0:
                similarity_matrix = cosine_similarity(generated_embeddings, master_embeddings)
                results = []
                for i, gen_q in enumerate(generated_questions):
                    best_match_index = np.argmax(similarity_matrix[i])
                    best_score = similarity_matrix[i][best_match_index]

                    if best_score >= similarity_threshold:
                        results.append({
                            "Your Question": gen_q,
                            "Best Match in Master List": master_questions[best_match_index],
                            "Subtopic": master_subtopics[best_match_index],
                            "Similarity Score": f"{best_score:.3f}"
                        })

                if results:
                    st.success(f"Found {len(results)} matches above the threshold!")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                else:
                    st.info("No matches found above the specified similarity threshold.")
