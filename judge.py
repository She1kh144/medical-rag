import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


JUDGE_SYSTEM_PROMPT = """Ты — строгий судья качества ответов медицинской справочной системы.

Ты получишь:
1. Вопрос пользователя
2. Контекст, который был предоставлен системе (фрагменты документов из корпуса)
3. Сгенерированный системой ответ
4. Ожидаемые источники
5. Ожидаемые ключевые факты

ВАЖНО: Оценивай ответ ТОЛЬКО на основе предоставленного контекста, а не на основе своих общих знаний.
Если в контексте нет какой-то информации — система не могла её знать, и это не её вина.

Твоя задача — оценить ответ по трём критериям:

1. **correct** (bool): Содержит ли ответ информацию, которая согласуется с предоставленным контекстом? Если контекст содержит ответ, а система его правильно воспроизвела — true. Учитывай эквивалентные формулировки (например, "8-9 ч" и "8–9 часов" — одно и то же). Не требуй полноты сверх того, что разумно ответить на конкретный вопрос.

2. **faithful** (bool): Опирается ли ответ на правильные фрагменты из контекста? Если ответ называет препарат, информации о котором нет в контексте, или придумывает источник, faithful = false. Если источник назван правильно, но не является "основным" ожидаемым — это всё равно faithful, если информация действительно есть в контексте.

3. **refusal_appropriate** (bool):
   - Если ожидаемые источники пусты — ответ должен корректно отказать.
   - Если ожидаемые источники НЕ пусты И в контексте есть релевантная информация — система должна была ответить, а не отказать.
   - Если ожидаемые источники не пусты, но в предоставленном контексте релевантной информации нет — отказ уместен.

Верни ТОЛЬКО валидный JSON в формате:
{"correct": true/false, "faithful": true/false, "refusal_appropriate": true/false, "reasoning": "одно-два предложения почему"}

Без markdown, без ```json, без других пояснений. Только JSON."""


def judge_answer(question: str, answer: str, retrieved_context: str, expected_sources: list, expected_keywords: list) -> dict:
    """Judge a single answer using LLM. Returns dict with correct/faithful/refusal_appropriate/reasoning."""

    user_prompt = (
        f"Вопрос: {question}\n\n"
        f"Контекст, который был предоставлен системе для ответа:\n{retrieved_context}\n\n"
        f"Сгенерированный ответ:\n{answer}\n\n"
        f"Ожидаемые источники: {expected_sources}\n"
        f"Ожидаемые ключевые факты: {expected_keywords}\n\n"
        f"Оцени ответ."
    )

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content or ""
    raw = content.strip()

    # Sometimes the model wraps JSON in code fences despite instructions
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "correct": False,
            "faithful": False,
            "refusal_appropriate": False,
            "reasoning": f"JSON parse failed. Raw: {raw[:200]}"
        }

    return result