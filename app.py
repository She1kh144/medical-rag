import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Load model and client ONCE at startup, not per request ---
embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

app = FastAPI(title="Medical RAG")

class Question(BaseModel):
    query: str
    k: int = 10

class Answer(BaseModel):
    answer: str
    sources: list[dict]

def retrieve(query: str, k: int):
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

def generate_answer(query: str, chunks):
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
        chunks = retrieve(question.query, question.k)
        if not chunks:
            raise HTTPException(status_code=404, detail="No chunks found")
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