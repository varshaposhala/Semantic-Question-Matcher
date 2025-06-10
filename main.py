# main.py (with batch processing and threshold filtering)

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv
import sqlalchemy as db
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# --- INITIAL SETUP (No changes here) ---
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in the .env file.")

genai.configure(api_key=API_KEY)
EMBEDDING_MODEL = 'models/embedding-001'
DATABASE_FILE = 'questions.db'

# --- IN-MEMORY DATA STORE AND DB LOADING (No changes here) ---
master_data = {}
engine = db.create_engine(f'sqlite:///{DATABASE_FILE}')

def load_master_data_from_db():
    global master_data
    master_data = {}
    try:
        with engine.connect() as connection:
            result = connection.execute(db.text("SELECT question_text, embedding FROM master_questions"))
            all_embeddings = []
            all_texts = []
            for row in result:
                all_texts.append(row.question_text)
                parsed_embedding = json.loads(row.embedding)
                all_embeddings.append(parsed_embedding)
            
            master_data = {
                "texts": all_texts,
                "embeddings": np.array(all_embeddings)
            }

        if not master_data.get("texts"):
            print("WARNING: Master data is empty. Did you run precompute_embeddings.py?")
        else:
            print(f"Successfully loaded {len(master_data['texts'])} master questions into memory.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load data from database. {e}")
        print("Please ensure 'questions.db' exists and you have run 'precompute_embeddings.py'.")

# --- FASTAPI APPLICATION ---
app = FastAPI(title="Semantic Match API")

@app.on_event("startup")
def startup_event():
    load_master_data_from_db()

# --- Request and Response Models for Batching ---

# <<< --- CHANGE 1: ADD AN OPTIONAL THRESHOLD TO THE REQUEST --- >>>
class BatchMatchRequest(BaseModel):
    questions: List[str]
    similarity_threshold: float = 0.9 # Default to 0.9 if not provided

class MatchResult(BaseModel):
    original_question: str
    best_match_question: str
    similarity_score: float

class BatchMatchResponse(BaseModel):
    matches: List[MatchResult]

@app.post("/find-batch-matches", response_model=BatchMatchResponse)
def find_batch_matches(request: BatchMatchRequest):
    if not master_data or not master_data.get("texts"):
        raise HTTPException(status_code=503, detail="Server is not ready. Master data not loaded.")

    try:
        result = genai.embed_content(model=EMBEDDING_MODEL, content=request.questions)
        new_embeddings = np.array(result['embedding'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get embeddings from AI provider: {e}")

    similarity_matrix = cosine_similarity(new_embeddings, master_data['embeddings'])

    batch_results = []
    for i, original_question in enumerate(request.questions):
        best_match_index = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i][best_match_index]
        best_match_text = master_data['texts'][best_match_index]
        
        # <<< --- CHANGE 2: ONLY APPEND THE RESULT IF THE SCORE IS HIGH ENOUGH --- >>>
        if best_score >= request.similarity_threshold:
            batch_results.append(MatchResult(
                original_question=original_question,
                best_match_question=best_match_text,
                similarity_score=float(best_score)
            ))

    return BatchMatchResponse(matches=batch_results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)