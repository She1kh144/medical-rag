import os
import json
import hashlib
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

CACHE_FILE = "hypothetical_questions.json"

def _load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def generate_hypothetical_questions(chunk: str, drug_name: str, cache: dict) -> list[str]:
    """
    Generate 3 hypothetical user questions that this chunk answers.
    Cached by hash of (chunk + drug_name) so re-ingest only re-generates changed chunks.
    """
    key = _hash(f"{drug_name}::{chunk}")
    if key in cache:
        return cache[key]

    prompt = (
        f"Ниже приведён фрагмент инструкции к препарату «{drug_name}».\n\n"
        f"Фрагмент:\n{chunk}\n\n"
        f"Сгенерируй ровно 3 коротких вопроса на русском, которые мог бы задать "
        f"обычный пользователь и на которые этот фрагмент содержит ответ. "
        f"Вопросы должны быть в естественном разговорном стиле — так, как их задал бы "
        f"человек, ищущий информацию (например: «С какого возраста можно принимать X?», "
        f"«Какая доза X для взрослых?», «У какого препарата период полувыведения N часов?»). "
        f"Используй конкретные значения из фрагмента, если они там есть. "
        f"Каждый вопрос — на отдельной строке, без нумерации, без пояснений."
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content or ""
    # Parse questions from the response — one per line, strip empty lines
    questions = [line.strip() for line in content.strip().split("\n") if line.strip()]

    # If the model returned a different count than 3, just keep what we got
    # (occasionally returns 2 or 4; not worth retrying)
    cache[key] = questions
    return questions