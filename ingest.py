import os
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- 1. Load the document ---
with open("data/drug1.txt", "r", encoding="utf-8") as file:
    text = file.read()

# --- 2. Split into overlapping chunks (by characters, simple v1) ---
def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """Splits the text into chunks with specified size and overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

chunks = chunk_text(text)
print(f"Split into {len(chunks)} chunks")

# --- 3. Embed each chunk ---
model = SentenceTransformer("intfloat/multilingual-e5-small")
embeddings = model.encode(chunks)
print(f"Created {len(embeddings)} embeddings")

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    dbname="medical_rag",
    user="postgres",
    password="devpassword",
)
cur = conn.cursor()

for chunk, embedding in zip(chunks, embeddings):
    cur.execute(
        "INSERT INTO chunks (source, chunk_text, embedding) VALUES (%s, %s, %s)",
        ("paracetamol_rlsnet", chunk, embedding.tolist()),
    )

conn.commit()
cur.close()
conn.close()
print("Done — chunks stored in database")
