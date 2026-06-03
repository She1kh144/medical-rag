import os
import json
import hashlib
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

CACHE_FILE = "chunk_contexts.json"

def _load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def _hash(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def generate_context(chunk: str, drug_name: str, cache: dict):
    """
    Generate context from chunk + drug name only (no full doc).
    Cached by hash of (chunk + drug_name) so re-ingest only re-generates changed chunks.
    """
    key = _hash(f"{drug_name}::{chunk}")
    if key in cache:
        return cache[key]

    prompt = (
        f"Ниже приведён фрагмент инструкции к препарату «{drug_name}».\n\n"
        f"Фрагмент:\n{chunk}\n\n"
        f"Напиши одно короткое предложение (15-25 слов) на русском, "
        f"которое объясняет, о чём этот фрагмент и какую информацию он содержит. "
        f"Не пересказывай содержимое подробно — просто опиши тему. "
        f"Не используй вводных слов вроде «Этот фрагмент». Начни сразу с сути."
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content
    context = (content or "").strip()
    cache[key] = context
    return context