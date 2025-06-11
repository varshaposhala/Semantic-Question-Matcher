# main.py (COMPLETE - with batch processing, threshold, and subtopics)

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import sqlalchemy as db
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# --- INITIAL SETUP ---
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in the .env file.")

genai.configure(api_key=API_KEY)
EMBEDDING_MODEL = 'models/embedding-001'
DATABASE_FILE = 'questions.db' # This was the missing variable

# --- IN-MEMORY DATA STORE AND DB LOADING ---
master_data = {}
engine = db.create_engine(f'sqlite:///{DATABASE_FILE}') # Now this line will work

def load_master_data_from_db():
    global master_data
    master_data = {}
    try:
        with engine.connect() as connection:
            # Select the question_text, subtopic, and embedding columns
            query = db.text("SELECT question_text, subtopic, embedding FROM master_questions")
            result = connection.execute(query)
            
            all_embeddings = []
            all_texts = []
            all_subtopics = [] # New list to hold subtopics
            
            for row in result:
                all_texts.append(row.question_text)
                all_subtopics.append(row.subtopic) # Store the subtopic
                # Parse the embedding string from the DB back into a Python list
                parsed_embedding = json.loads(row.embedding)
                all_embeddings.append(parsed_embedding)
            
            # Add all data to the in-memory store
            master_data = {
                "texts": all_texts,
                "subtopics": all_subtopics,
                "embeddings": np.array(all_embeddings)
            }

        if not master_data.get("texts"):
            print("WARNING: Master data is empty. Did you run precompute_embeddings.py?")
        else:
            print(f"Successfully loaded {len(master_data['texts'])} master questions (with subtopics) into memory.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load data from database. {e}")
        print("Please ensure 'questions.db' exists and was created with the 'subtopic' column.")

# --- FASTAPI APPLICATION ---
app = FastAPI(title="Semantic Match API")

@app.on_event("startup")
def startup_event():
    load_master_data_from_db()

# --- Request and Response Models ---
class BatchMatchRequest(BaseModel):
    questions: List[str]
    similarity_threshold: float = 0.9

class MatchResult(BaseModel):
    original_question: str
    best_match_question: str
    matched_subtopic: Optional[str] = None # The subtopic of the matched question
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
        
        if best_score >= request.similarity_threshold:
            best_match_text = master_data['texts'][best_match_index]
            # Retrieve the subtopic using the same index
            best_match_subtopic = master_data['subtopics'][best_match_index]
            
            batch_results.append(MatchResult(
                original_question=original_question,
                best_match_question=best_match_text,
                matched_subtopic=best_match_subtopic, # Add it to the result object
                similarity_score=float(best_score)
            ))

    return BatchMatchResponse(matches=batch_results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)