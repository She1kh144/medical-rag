import os
import re
import json
import hashlib
import psycopg2
from dotenv import load_dotenv

load_dotenv()

CACHE_FILE = "data/hypothetical_questions.json"

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Connect to DB
conn = psycopg2.connect(
    host=os.environ.get("DB_HOST", "localhost"),
    port=int(os.environ.get("DB_PORT", "5433")),
    dbname=os.environ.get("DB_NAME", "medical_rag"),
    user=os.environ.get("DB_USER", "postgres"),
    password=os.environ.get("DB_PASSWORD", "devpassword"),
)
cur = conn.cursor()
cur.execute("SELECT chunk_text FROM chunks ORDER BY id;")
rows = cur.fetchall()
cur.close()
conn.close()

print(f"Loaded {len(rows)} chunks from DB")

# Regex for the prefix line: [<name_prefix> — <Section>]
prefix_re = re.compile(r"^\[([^\]]+?)\s+—\s+([^\]]+)\]$")

cache = {}
parse_failures = []

for (chunk_text,) in rows:
    lines = chunk_text.split("\n")

    # Line 0 must be the prefix
    m = prefix_re.match(lines[0])
    if not m:
        parse_failures.append(("bad_prefix", lines[0][:80]))
        continue
    drug_name = m.group(1).strip()

    # Line 1 must be "Возможные вопросы:"
    if lines[1].strip() != "Возможные вопросы:":
        parse_failures.append(("missing_header", lines[0][:80]))
        continue

    # Lines 2+ are questions: lines starting with "- " AND ending with "?"
    questions = []
    i = 2
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("- ") and stripped.endswith("?"):
            questions.append(stripped[2:].strip())
            i += 1
        else:
            break

    if not questions:
        parse_failures.append(("no_questions", lines[0][:80]))
        continue

    # Everything after the questions block is the original chunk text
    original_chunk = "\n".join(lines[i:])

    key = _hash(f"{drug_name}::{original_chunk}")
    cache[key] = questions

print(f"Reconstructed {len(cache)} cache entries")
print(f"Parse failures: {len(parse_failures)}")
for kind, sample in parse_failures[:5]:
    print(f"  {kind}: {sample}")

with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

print(f"Written to {CACHE_FILE}")