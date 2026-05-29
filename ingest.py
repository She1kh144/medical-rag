import os
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- Define the documents to ingest: (filepath, source_label) ---
documents = [
    ("data/drug1.txt", "paracetamol_rlsnet"),
    ("data/drug2.txt", "ibuprofen_rlsnet"),
    ("data/drug3.txt", "acetylsalicylic_acid_rlsnet"),
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

model = SentenceTransformer("intfloat/multilingual-e5-small")

conn = psycopg2.connect(
    host=os.environ.get("DB_HOST", "localhost"),
    port=int(os.environ.get("DB_PORT", "5433")),
    dbname=os.environ.get("DB_NAME", "medical_rag"),
    user=os.environ.get("DB_USER", "postgres"),
    password=os.environ.get("DB_PASSWORD", "devpassword"),
)
cur = conn.cursor()

# Create the chunks table with a vector column for embeddings if not exists
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id SERIAL PRIMARY KEY,
        source TEXT NOT NULL,
        chunk_text TEXT NOT NULL,
        embedding vector(384)
    );
""")
conn.commit()

total_chunks = 0
for filepath, source_label in documents:
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = splitter.split_text(text)
    embeddings = model.encode([f"context: {chunk}" for chunk in chunks])

    for chunk, embedding in zip(chunks, embeddings):
        cur.execute(
            "INSERT INTO chunks (source, chunk_text, embedding) VALUES (%s, %s, %s)",
            (source_label, chunk, embedding.tolist()),
        )

    print(f"Ingested {len(chunks)} chunks from {source_label}")
    total_chunks += len(chunks)

conn.commit()
cur.close()
conn.close()
print(f"Done — {total_chunks} total chunks stored")
