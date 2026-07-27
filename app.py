import os
import json
import time
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from collections import defaultdict, deque
from fastapi import Depends, FastAPI, HTTPException, Request
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

system_prompt = (
    "Ты — помощник, отвечающий на вопросы о лекарствах СТРОГО на основе "
    "предоставленного контекста. Используй ТОЛЬКО информацию из контекста ниже.\n\n"

    "ВАЖНЫЕ ПРАВИЛА ЧТЕНИЯ КОНТЕКСТА:\n"
    "1. Если в контексте есть информация, релевантная вопросу — отвечай на основе этой "
    "информации, даже если она неполная или относится к частному случаю. Лучше дать частичный "
    "ответ с указанием того, что именно есть в контексте, чем отказать.\n"
    "2. Если вопрос касается общего случая (например, 'доза для взрослых'), "
    "ищи в контексте основные/обычные рекомендации, которые обычно идут в начале "
    "соответствующих разделов. Не отвечай 'не указано' только потому, что в контексте "
    "есть много специальных случаев.\n"
    "3. Если вопрос про класс препаратов или сравнение (например, 'какой антибиотик группы X' "
    "или 'какой НПВС лучше'), и в контексте есть препарат(ы) из этого класса — "
    "назови его(их) и опиши на основе контекста.\n"
    "4. Отвечай 'В предоставленных документах нет ответа на этот вопрос' ТОЛЬКО если "
    "в контексте действительно НЕТ информации, имеющей отношение к вопросу. "
    "Не отказывай только потому, что ответ требует синтеза или интерпретации.\n\n"

    "Не придумывай информацию, отсутствующую в контексте. "
    "В конце ответа укажи источник из предоставленного контекста (не придумывай источники). "
    "Всегда добавляй: 'Это не медицинская консультация, обратитесь к врачу.'"
)

app = FastAPI(title="Medical RAG")

PER_IP_LIMIT = 20
PER_IP_WINDOW = 3600        # seconds -> one hour

GLOBAL_LIMIT = 400
GLOBAL_WINDOW = 86400       # seconds -> one day

ip_hits = defaultdict(deque)
global_hits = deque()

class Question(BaseModel):
    query: str = Field(max_length=500)
    k: int = 10
    rerank: bool = False

class Answer(BaseModel):
    answer: str
    sources: list[dict]
    chunks: list[dict] 

def client_ip(request: Request):
    """Real visitor IP, accounting for the reverse proxy in front of us."""
    forwarded = request.headers.get("X-Forwarded-For")

    if forwarded:
        return forwarded.split(",")[0].strip()

    return request.client.host if request.client else "unknown"

def rate_limit(request: Request):
    """Rejects a request if the per-IP or global limit is exceeded."""
    now = time.time()
    ip = client_ip(request)

    hits = ip_hits[ip]

    while hits and now - hits[0] > PER_IP_WINDOW:
        hits.popleft()

    while global_hits and now - global_hits[0] > GLOBAL_WINDOW:
        global_hits.popleft()

    if len(hits) >= PER_IP_LIMIT:
        raise HTTPException(429, "Слишком много запросов. Попробуйте через час.")

    if len(global_hits) >= GLOBAL_LIMIT:
        raise HTTPException(429, "Демо временно недоступно: исчерпан дневной лимит запросов.")

    hits.append(now)
    global_hits.append(now)

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
    user_prompt = f"Контекст:\n{context}\n\nВопрос: {query}"

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
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
    user_prompt = f"Контекст:\n{context}\n\nВопрос: {query}"

    stream = client.chat.completions.create(
        model="deepseek-v4-flash",
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

@app.get("/search")
def search(query: str, k: int = 10):
    try:
        chunks = retrieve(query, k)

        chunks_data = [
            {"text": text, "source": src, "distance": float(distance)}
            for text, src, distance in chunks
        ]

        return {"chunks": chunks_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ask", response_model=Answer)
def ask(question: Question, _: None = Depends(rate_limit)):
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

        chunks_data = [
            {"text": text, "source": src, "distance": float(distance)}
            for text, src, distance in chunks
        ]
        sources = [
            {"source": src, "distance": float(dist)}
            for _, src, dist in chunks
        ]
        
        return Answer(answer=answer_text or "", sources=sources, chunks=chunks_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ask/stream")
def ask_stream(question: Question, _: None = Depends(rate_limit)):
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))