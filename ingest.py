import os
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- 1. Load the document ---
with open("data/drug1.txt", "r", encoding="utf-8") as file:
    text = file.read()

# change comment below
# --- 2. Split into overlapping chunks with RecursiveCharacterTextSplitter (by separators, v2) ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(text)
print(f"Split into {len(chunks)} chunks")

# --- 3. Embed each chunk ---
model = SentenceTransformer("intfloat/multilingual-e5-small")
# We prepend "passage: " to each chunk to give the model a hint that these are passages to be used for retrieval
embeddings = model.encode([f"passage: {chunk}" for chunk in chunks])
print(f"Created {len(embeddings)} embeddings")

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    dbname="medical_rag",
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
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
