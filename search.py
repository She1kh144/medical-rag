import os
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# The question we're asking the knowledge base
query = "Какая дозировка парацетамола для взрослых?"

# --- 1. Embed the question (same model as ingestion) ---
model = SentenceTransformer("intfloat/multilingual-e5-small")
query_embedding = model.encode(f"query: {query}").tolist()

# --- 2. Find the nearest chunks in the database ---
conn = psycopg2.connect(
    host="localhost", port=5433, dbname="medical_rag",
    user="postgres", password="devpassword",
)
cur = conn.cursor()

cur.execute(
    """
    SELECT chunk_text, embedding <=> %s::vector AS distance
    FROM chunks
    ORDER BY distance
    LIMIT 3
    """,
    (query_embedding,),
)

results = cur.fetchall()
cur.close()
conn.close()

# --- 3. Show what came back ---
print(f"Question: {query}\n")
for i, (chunk_text, distance) in enumerate(results, 1):
    print(f"--- Result {i} (distance: {distance:.4f}) ---")
    print(chunk_text)
    print()