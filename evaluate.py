import json
import requests

API_URL = "http://127.0.0.1:8000/ask"

def normalize_expected_source(expected):
    """Accept either a string or list; always return a list."""
    if isinstance(expected, str):
        return [expected]
    return expected

def normalize(s):
    return (
        s.lower()
        .replace("ё", "е")
        .replace("–", "-")  # en-dash
        .replace("—", "-")  # em-dash
        .replace("−", "-")  # minus sign
    )

def evaluate_one(item, top_n=3):
    response = requests.post(
        API_URL,
        json={"query": item["question"], "rerank": False, "hybrid": False, "k": 10},
        #timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    answer = data["answer"]
    sources = data["sources"]

    # --- Retrieval check: is an expected source in the top N? ---
    expected_sources = set(normalize_expected_source(item["expected_source"]))
    top_sources = {s["source"] for s in sources[:top_n]}
    retrieval_hit = bool(expected_sources & top_sources) if expected_sources else True

    # --- Answer check: keyword present and right source cited ---
    answer_lower = answer.lower()

    # --- Keyword check (existing) ---
    keyword_hit = any(
        normalize(kw) in normalize(answer) for kw in item["expected_keywords"]
    )

    # --- Source check: the expected source must appear in the answer text ---
    source_in_answer = any(
        src.lower() in answer_lower for src in expected_sources
    ) if expected_sources else True

    answer_hit = keyword_hit and source_in_answer

    return {
        "question": item["question"],
        "retrieval_hit": retrieval_hit,
        "answer_hit": answer_hit,
        "answer": answer,
        "top_sources": [s["source"] for s in sources[:top_n]],
    }

def main():
    with open("data/drug_questions.json", "r", encoding="utf-8") as file:
        questions = json.load(file)

    results = []
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {item['question'][:60]}...")
        result = evaluate_one(item)
        results.append(result)
        r_mark = "✓" if result["retrieval_hit"] else "✗"
        a_mark = "✓" if result["answer_hit"] else "✗"
        print(f"   retrieval {r_mark}   answer {a_mark}")

    # --- Aggregate ---
    n = len(results)
    retrieval_acc = sum(r["retrieval_hit"] for r in results) / n
    answer_acc = sum(r["answer_hit"] for r in results) / n

    print(f"\n{'='*50}")
    print(f"Retrieval accuracy: {retrieval_acc:.0%} ({sum(r['retrieval_hit'] for r in results)}/{n})")
    print(f"Answer accuracy:    {answer_acc:.0%} ({sum(r['answer_hit'] for r in results)}/{n})")
    print(f"{'='*50}")

    # --- Save detailed results for inspection ---
    with open("data/eval_result.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print("\nDetailed results saved to data/eval_result.json")

if __name__ == "__main__":
    main()

