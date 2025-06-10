# app.py

import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Semantic Question Matcher",
    page_icon="🤖",
    layout="wide"
)

# --- GEMINI API CONFIGURATION ---
# Allow user to enter API key in the sidebar, or use .env file
st.sidebar.title("Configuration")
api_key_input = st.sidebar.text_input(
    "Enter your Google Gemini API Key",
    type="password",
    help="You can get your API key from Google AI Studio."
)

# Load from .env if the user doesn't provide a key
if not api_key_input:
    load_dotenv()
    api_key_from_env = os.getenv('GEMINI_API_KEY')
    if api_key_from_env:
        genai.configure(api_key=api_key_from_env)
    else:
        st.sidebar.warning("Please enter your API Key or set it in a .env file.")
else:
    genai.configure(api_key=api_key_input)


EMBEDDING_MODEL = 'models/embedding-001'

# --- CORE FUNCTIONS (with Caching) ---
@st.cache_data(show_spinner=False) # Caching to avoid re-calling API for same text
def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Get embeddings for a list of texts.
    
    Returns a NumPy array of embeddings.
    """
    if not texts or not any(text.strip() for text in texts):
        return np.array([])
    
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL, content=texts)
        return np.array(result['embedding'])
    except Exception as e:
        st.error(f"Error getting embeddings: {e}")
        return np.array([])

# --- UI LAYOUT ---
st.title("🤖 Semantic Question Matcher")
st.write("Enter your master questions and the questions you want to match. The app will find the most similar master question for each of your new questions based on semantic meaning.")

col1, col2 = st.columns(2)

with col1:
    st.header("Master Questions")
    master_questions_text = st.text_area(
        "Enter one master question per line. This is your knowledge base.",
        height=300,
        placeholder="e.g., What is the capital of France?\nHow does photosynthesis work?"
    )

with col2:
    st.header("Questions to Match")
    generated_questions_text = st.text_area(
        "Enter the new questions you want to find matches for.",
        height=300,
        placeholder="e.g., Which city is the capital of France?\nCan you explain the process of photosynthesis?"
    )

st.markdown("---")

# --- CONTROLS AND ACTION BUTTON ---
similarity_threshold = st.slider(
    "Similarity Score Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.8,  # Default to 0.8
    step=0.05,
    help="Only show matches with a score equal to or higher than this value."
)

if st.button("Find Similar Questions", type="primary", use_container_width=True):
    # 1. Clean and validate inputs
    master_questions = [q.strip() for q in master_questions_text.split('\n') if q.strip()]
    generated_questions = [q.strip() for q in generated_questions_text.split('\n') if q.strip()]

    if not master_questions or not generated_questions:
        st.warning("Please enter questions in both text areas.")
    else:
        with st.spinner("Calculating embeddings and similarities... This may take a moment."):
            # 2. Get embeddings for both lists
            master_embeddings = get_embeddings(master_questions)
            generated_embeddings = get_embeddings(generated_questions)

            if master_embeddings.size > 0 and generated_embeddings.size > 0:
                # 3. Calculate similarity matrix
                similarity_matrix = cosine_similarity(generated_embeddings, master_embeddings)
                
                # 4. Find the best match for each generated question
                results = []
                for i, gen_q in enumerate(generated_questions):
                    best_match_index = np.argmax(similarity_matrix[i])
                    best_score = similarity_matrix[i][best_match_index]
                    
                    if best_score >= similarity_threshold:
                        results.append({
                            "Your Question": gen_q,
                            "Best Match in Master List": master_questions[best_match_index],
                            "Similarity Score": f"{best_score:.2f}" # Format to 2 decimal places
                        })

                # 5. Display results
                if results:
                    st.success(f"Found {len(results)} matches above the threshold!")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                else:
                    st.info("No matches found above the specified similarity threshold.")