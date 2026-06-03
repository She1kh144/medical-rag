import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

# --- Load model and client ONCE at startup, not per request ---
embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
_rerank_model = None  # Lazy-loaded when needed
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

app = FastAPI(title="Medical RAG")

class Question(BaseModel):
    query: str
    k: int = 10
    rerank: bool = False
    hybrid: bool = False

class Answer(BaseModel):
    answer: str
    sources: list[dict]


def get_rerank_model():
    """Lazy-load the reranker model only if needed."""
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _rerank_model

def rerank(query: str, chunks: list, top_k: int = 10):
    """Rerank chunks by cross-encoder relevance to the query."""
    rerank_model = get_rerank_model()
    pairs = [[query, chunk_text] for chunk_text, _, _ in chunks]
    scores = rerank_model.predict(pairs)
    
    # Pair each chunk with its new score and sort descending
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored[:top_k]]

def retrieve(query: str, k: int):
    """Retrieve top-k relevant chunks from the database using vector similarity."""
    query_embedding = embed_model.encode(f"query: {query}").tolist()
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "5433")),
        dbname=os.environ.get("DB_NAME", "medical_rag"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "devpassword"),
    )
    cur = conn.cursor()
    cur.execute(
        """
        SELECT chunk_text, source, embedding <=> %s::vector AS distance
        FROM chunks
        ORDER BY distance
        LIMIT %s
        """,
        (query_embedding, k),
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def retrieve_bm25(query: str, k: int = 50):
    """Keyword retrieval via Postgres full-text search."""
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "5433")),
        dbname=os.environ.get("DB_NAME", "medical_rag"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "devpassword"),
    )
    cur = conn.cursor()
    cur.execute(
        """
        SELECT chunk_text, source, ts_rank_cd(tsv, query) AS rank
        FROM chunks, websearch_to_tsquery('russian', %s) query
        WHERE tsv @@ query
        ORDER BY rank DESC
        LIMIT %s
        """,
        (query, k),
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def hybrid_retrieve(query: str, k: int = 50):
    """Combine vector and BM25 retrieval using RRF (Reciprocal Rank Fusion)."""
    vector_results = retrieve(query, k=k)
    bm25_results = retrieve_bm25(query, k=k)

    # Build rank dicts keyed by chunk_text (since that's what's unique per chunk)
    # Rank starts at 1 for the top result
    vector_ranks = {chunk[0]: rank for rank, chunk in enumerate(vector_results, start=1)}
    bm25_ranks = {chunk[0]: rank for rank, chunk in enumerate(bm25_results, start=1)}

    # Collect all unique chunks from both lists, preserving the full tuple
    all_chunks = {}
    for chunk in vector_results:
        all_chunks[chunk[0]] = chunk
    for chunk in bm25_results:
        if chunk[0] not in all_chunks:
            all_chunks[chunk[0]] = chunk

    # Compute RRF scores
    C = 60  # RRF smoothing constant
    rrf_scores = {}
    for chunk_text in all_chunks:
        score = 0
        if chunk_text in vector_ranks:
            score += 1 / (C + vector_ranks[chunk_text])
        if chunk_text in bm25_ranks:
            score += 1 / (C + bm25_ranks[chunk_text])
        rrf_scores[chunk_text] = score

    # Sort by RRF score descending, return top-k
    sorted_chunks = sorted(all_chunks.values(), key=lambda chunk: rrf_scores[chunk[0]], reverse=True)
    return sorted_chunks[:k]

def generate_answer(query: str, chunks):
    """Generate an answer using the retrieved chunks as context."""
    context = "\n\n".join(
        f"[Источник: {source}]\n{text}"
        for text, source, distance in chunks
    )
    system_prompt = (
        "Ты — помощник, отвечающий на вопросы о лекарствах СТРОГО на основе "
        "предоставленного контекста. Используй ТОЛЬКО информацию из контекста ниже. "
        "Если в контексте нет ответа, скажи: «В предоставленных документах нет ответа на этот вопрос». "
        "Не придумывай информацию. В конце ответа укажи источник. "
        "Всегда добавляй: «Это не медицинская консультация, обратитесь к врачу.»"
    )
    user_prompt = f"Контекст:\n{context}\n\nВопрос: {query}"
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=Answer)
def ask(question: Question):
    try:
        if question.hybrid:
            candidates = hybrid_retrieve(question.query, k=50 if question.rerank else question.k)
        else:
            # Bi-encoder retrieval with optional reranking step
            candidates = retrieve(question.query, k=50 if question.rerank else question.k)
        
        if not candidates:
            raise HTTPException(status_code=404, detail="No chunks found")
        
        # Optional intermediate reranking before generation (if enabled)
        if question.rerank:
            chunks = rerank(question.query, candidates, top_k=10)
        else:
            chunks = candidates[:question.k]
        
        answer_text = generate_answer(question.query, chunks)
        sources = [
            {"source": src, "distance": float(dist)}
            for _, src, dist in chunks
        ]
        return Answer(answer=answer_text or "", sources=sources)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))