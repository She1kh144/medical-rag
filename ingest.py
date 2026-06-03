import os
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from contextualize import generate_context, _load_cache, _save_cache

load_dotenv()

# Drug filename -> Russian display name and active ingredient(s)
DRUG_NAMES = {
    "paracetamol": ("Парацетамол", "парацетамол"),
    "ibuprofen": ("Ибупрофен", "ибупрофен"),
    "aspirin": ("Аспирин", "ацетилсалициловая кислота"),
    "amoxicillin": ("Амоксициллин", "амоксициллин"),
    "amoksiklav": ("Амоксиклав", "амоксициллин + клавулановая кислота"),
    "doxycycline": ("Доксициклин", "доксициклин"),
    "ciprofloxacin": ("Ципрофлоксацин", "ципрофлоксацин"),
    "sumamed": ("Сумамед", "азитромицин"),
    "nise": ("Найз", "нимесулид"),
    "ketorol": ("Кеторол", "кеторолак"),
    "suprastin": ("Супрастин", "хлоропирамин"),
    "clemastine": ("Клемастин", "клемастин"),
    "claritin": ("Кларитин", "лоратадин"),
    "aerius": ("Эриус", "дезлоратадин"),
    "zyrtec": ("Зиртек", "цетиризин"),
}

# Section header normalization
SECTION_MAP = {
    "Действующее вещество": "Состав",
    "Состав": "Состав",
    "Состав на одну таблетку": "Состав",
    "Описание лекарственной формы": "Описание лекарственной формы",
    "Фармакокинетика": "Фармакокинетика",
    "Фармакодинамика": "Фармакодинамика",
    "Фармакологическое действие": "Фармакодинамика",
    "Механизм действия": "Фармакодинамика",
    "Показания": "Показания к применению",
    "Показания к применению": "Показания к применению",
    "Противопоказания": "Противопоказания",
    "С осторожностью": "Противопоказания",
    "Применение при беременности и кормлении грудью": "Беременность и лактация",
    "Беременность": "Беременность и лактация",
    "Лактация": "Беременность и лактация",
    "Кормление грудью": "Беременность и лактация",
    "Фертильность": "Беременность и лактация",
    "Способ применения и дозы": "Способ применения и дозы",
    "Режим дозирования": "Способ применения и дозы",
    "Побочные действия": "Побочные действия",
    "Взаимодействие": "Взаимодействие",
    "Передозировка": "Передозировка",
    "Особые указания": "Особые указания",
    "Меры предосторожности": "Особые указания",
    "Влияние на способность управлять транспортными средствами и механизмами": "Особые указания",
    "Форма выпуска": "Форма выпуска",
    "Условия хранения": "Условия хранения",
    "Срок годности": "Срок годности",
    "Условия отпуска из аптек": "Условия отпуска",
    "Производитель": "Производитель",
}

def parse_sections(text: str):
    """Split a document into (section_name, section_body) pairs."""
    sections = []
    current_section = "Описание"
    current_body = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped in SECTION_MAP:
            # Flush the previous section
            if current_body:
                sections.append((current_section, "\n".join(current_body).strip()))
            current_section = SECTION_MAP[stripped]
            current_body = []
        else:
            current_body.append(line)

    # Flush the final section
    if current_body:
        sections.append((current_section, "\n".join(current_body).strip()))

    # Drop empty sections (some files may have empty 'Описание' if first line is a header)
    return [(name, body) for name, body in sections if body]

# Append all drugs from data/drugs to the list
documents = []
for filename in sorted(os.listdir("data/drugs")):
    if filename.endswith(".txt"):
        filepath = os.path.join("data/drugs", filename)
        source_label = filename.replace(".txt", "") + "_rlsnet"
        documents.append((filepath, source_label))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

model = SentenceTransformer("intfloat/multilingual-e5-small")

conn = psycopg2.connect(
    host=os.environ.get("DB_HOST", "localhost"),
    port=int(os.environ.get("DB_PORT", "5433")),
    dbname=os.environ.get("DB_NAME", "medical_rag"),
    user=os.environ.get("DB_USER", "postgres"),
    password=os.environ.get("DB_PASSWORD", "devpassword"),
)
cur = conn.cursor()

# Create the chunks table with a vector column for embeddings if not exists
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id SERIAL PRIMARY KEY,
        source TEXT NOT NULL,
        chunk_text TEXT NOT NULL,
        embedding vector(384)
    );
""")
conn.commit()

total_chunks = 0
for filepath, source_label in documents:
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()

    # Derive drug display name from filename
    drug_key = os.path.basename(filepath).replace(".txt", "")
    brand, active_ingredient = DRUG_NAMES.get(drug_key, (drug_key, drug_key))

    if brand == active_ingredient:
        name_prefix = brand
    else:
        name_prefix = f"{brand} ({active_ingredient})"

    # Parse into sections
    sections = parse_sections(text)

    cache = _load_cache() # Save and load cache around each document to avoid re-generating contexts for unchanged chunks across documents

    # Chunk each section, prefix every chunk with [Drug — Section]
    prefixed_chunks = []
    for section_name, section_body in sections:
        section_chunks = splitter.split_text(section_body)
        for chunk in section_chunks:
            context = generate_context(chunk, name_prefix, cache)
            prefixed = f"[{name_prefix} — {section_name}]\nКонтекст: {context}\n{chunk}"
            prefixed_chunks.append(prefixed)

    _save_cache(cache) # Save cache after processing each document to persist any new contexts generated

    embeddings = model.encode([f"context: {chunk}" for chunk in prefixed_chunks])

    for chunk, embedding in zip(prefixed_chunks, embeddings):
        cur.execute(
            "INSERT INTO chunks (source, chunk_text, embedding) VALUES (%s, %s, %s)",
            (source_label, chunk, embedding.tolist()),
        )

    print(f"Ingested {len(prefixed_chunks)} chunks from {source_label} ({len(sections)} sections)")
    total_chunks += len(prefixed_chunks)

conn.commit()
cur.close()
conn.close()
print(f"Done — {total_chunks} total chunks stored")
