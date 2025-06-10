# precompute_embeddings.py

import os
import csv
import time
import google.generativeai as genai
from dotenv import load_dotenv
import sqlalchemy as db

# --- CONFIGURATION (GLOBAL VARIABLES) ---
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in the .env file.")

genai.configure(api_key=API_KEY)
MODEL = 'models/embedding-001'
CSV_FILE = 'master_questions.csv'
DATABASE_FILE = 'questions.db' # SQLite database file

# --- DATABASE SETUP ---
engine = db.create_engine(f'sqlite:///{DATABASE_FILE}')
metadata = db.MetaData()

master_questions_table = db.Table('master_questions', metadata,
    db.Column('id', db.Integer, primary_key=True, autoincrement=True),
    db.Column('question_text', db.String, nullable=False, unique=True),
    db.Column('embedding', db.JSON, nullable=False)
)

metadata.create_all(engine)
print(f"Database setup complete. Using '{DATABASE_FILE}'.")

def get_embedding(text, retries=3, delay=5):
    """Gets embedding for a given text with retry logic."""
    for attempt in range(retries):
        try:
            return genai.embed_content(model=MODEL, content=text)['embedding']
        except Exception as e:
            print(f"API Error (Attempt {attempt + 1}/{retries}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
    print(f"Failed to get embedding for: {text}")
    return None

def main():
    """Reads questions from CSV, gets embeddings, and stores them in the SQLite DB."""
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip header row
            next(reader)
            # Read all questions, stripping whitespace to catch subtle duplicates
            all_questions = [row[0].strip() for row in reader if row and row[0].strip()]
            # Use a set to automatically get unique questions, preserving order
            unique_questions = list(dict.fromkeys(all_questions))

    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE}' was not found.")
        return

    if len(all_questions) != len(unique_questions):
        print(f"Found {len(all_questions)} total questions, but only {len(unique_questions)} are unique. Processing unique questions.")
    else:
        print(f"Found {len(unique_questions)} unique questions in '{CSV_FILE}'. Processing...")

    with engine.connect() as connection:
        # Clear existing data to avoid duplicates if the script is run again
        connection.execute(master_questions_table.delete())
        connection.commit() # Commit the deletion
        print("Cleared old data from the table.")

        # Loop through the list of UNIQUE questions
        for i, question in enumerate(unique_questions):
            print(f"Processing question {i+1}/{len(unique_questions)}: '{question[:60]}...'")
            embedding = get_embedding(question)
            # print(embedding)
            if embedding:
                insert_query = master_questions_table.insert().values(
                    question_text=question,
                    embedding=embedding
                )
                connection.execute(insert_query)

        connection.commit() # Commit all the insertions

    print("\nPre-computation complete. Your database 'questions.db' is ready.")

if __name__ == "__main__":
    main()