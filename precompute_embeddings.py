# precompute_embeddings.py (with subtopics)

import os
import csv
import time
import google.generativeai as genai
from dotenv import load_dotenv
import sqlalchemy as db

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in the .env file.")

genai.configure(api_key=API_KEY)
MODEL = 'models/embedding-001'
CSV_FILE = 'master_questions.csv'
DATABASE_FILE = 'questions.db'

# --- DATABASE SETUP ---
engine = db.create_engine(f'sqlite:///{DATABASE_FILE}')
metadata = db.MetaData()

# <<< CHANGE 1: ADD THE 'subtopic' COLUMN TO THE TABLE DEFINITION >>>
master_questions_table = db.Table('master_questions', metadata,
    db.Column('id', db.Integer, primary_key=True, autoincrement=True),
    db.Column('question_text', db.String, nullable=False, unique=True),
    db.Column('subtopic', db.String, nullable=True), # Can be null if some questions don't have one
    db.Column('embedding', db.JSON, nullable=False)
)

metadata.create_all(engine)
print(f"Database setup complete with 'subtopic' column. Using '{DATABASE_FILE}'.")

def get_embedding(text, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return genai.embed_content(model=MODEL, content=text)['embedding']
        except Exception as e:
            print(f"API Error (Attempt {attempt + 1}/{retries}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
    print(f"Failed to get embedding for: {text}")
    return None

def main():
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            # <<< CHANGE 2: READ BOTH QUESTION AND SUBTOPIC >>>
            all_entries = [(row[0].strip(), row[1].strip()) for row in reader if row and row[0].strip()]
            # Use a dictionary to get unique questions, keeping the first subtopic found
            unique_entries = list(dict(all_entries).items())

    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE}' was not found.")
        return

    print(f"Found {len(unique_entries)} unique questions in '{CSV_FILE}'. Processing...")
    
    with engine.connect() as connection:
        connection.execute(master_questions_table.delete())
        connection.commit()
        print("Cleared old data from the table.")

        # <<< CHANGE 3: LOOP THROUGH (QUESTION, SUBTOPIC) PAIRS >>>
        for i, (question, subtopic) in enumerate(unique_entries):
            print(f"Processing question {i+1}/{len(unique_entries)}: '{question[:50]}...' (Subtopic: {subtopic})")
            embedding = get_embedding(question)
            
            if embedding:
                # <<< CHANGE 4: INSERT THE SUBTOPIC INTO THE DATABASE >>>
                insert_query = master_questions_table.insert().values(
                    question_text=question,
                    subtopic=subtopic, # Add subtopic here
                    embedding=embedding
                )
                connection.execute(insert_query)
        
        connection.commit()

    print("\nPre-computation complete. Your database 'questions.db' is ready with subtopics.")

if __name__ == "__main__":
    main()