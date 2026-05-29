import os
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Setup ---
embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

def retrieve(query, k=10):
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

def generate_answer(query, chunks):
    # Build the context block from retrieved chunks
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

# --- Run the full pipeline ---
if __name__ == "__main__":
    query = "Какая дозировка парацетамола для взрослых?"

    chunks = retrieve(query)
    answer = generate_answer(query, chunks)

    print(f"Вопрос: {query}\n")
    print(f"Ответ:\n{answer}\n")
    print("--- Использованные источники ---")
    for text, source, distance in chunks:
        print(f"[{source}] (distance: {distance:.4f})")