import os
import json
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
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

def generate_answer(query: str, chunks: list):
    """Generate an answer using the retrieved chunks as context."""
    context = "\n\n".join(
        f"[Источник: {source}]\n{text}"
        for text, source, distance in chunks
    )
    system_prompt = (
        "Ты — помощник, отвечающий на вопросы о лекарствах СТРОГО на основе "
        "предоставленного контекста. Используй ТОЛЬКО информацию из контекста ниже. "
        "Отвечай 'В предоставленных документах нет ответа на этот вопрос' ТОЛЬКО если "
        "ты внимательно прочитал весь контекст и убедился, что общего ответа в нём нет. "
        "Не придумывай информацию. В конце ответа укажи источник. "
        "Всегда добавляй: 'Это не медицинская консультация, обратитесь к врачу.'"
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

def generate_answer_stream(query: str, chunks: list):
    """Generator that yields LLM response chunks as they arrive."""
    context = "\n\n".join(
        f"[Источник: {source}]\n{text}"
        for text, source, distance in chunks
    )
    system_prompt = (
        "Ты — помощник, отвечающий на вопросы о лекарствах СТРОГО на основе "
        "предоставленного контекста. Используй ТОЛЬКО информацию из контекста ниже. "
        "Отвечай 'В предоставленных документах нет ответа на этот вопрос' ТОЛЬКО если "
        "ты внимательно прочитал весь контекст и убедился, что общего ответа в нём нет. "
        "Не придумывай информацию. В конце ответа укажи источник. "
        "Всегда добавляй: 'Это не медицинская консультация, обратитесь к врачу.'"
    )
    user_prompt = f"Контекст:\n{context}\n\nВопрос: {query}"

    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        stream=True,
    )

    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=Answer)
def ask(question: Question):
    try:
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
    
@app.post("/ask/stream")
def ask_stream(question: Question):
    candidates = retrieve(question.query, k=50 if question.rerank else question.k)

    if not candidates:
        raise HTTPException(status_code=404, detail="No chunks found")

    if question.rerank:
        chunks = rerank(question.query, candidates, top_k=10)
    else:
        chunks = candidates[:question.k]

    def event_stream():
        # First, send the sources as a structured event
        sources = [{"source": src, "distance": float(dist)} for _, src, dist in chunks]
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

        # Then stream the answer text
        for token in generate_answer_stream(question.query, chunks):
            # SSE format: data: <content>\n\n
            # Replace newlines in content to avoid breaking the SSE framing
            safe_token = token.replace("\n", "\\n")
            yield f"event: token\ndata: {safe_token}\n\n"

        yield "event: done\ndata: \n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")