import os
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hypotheticals import generate_hypothetical_questions, _load_cache, _save_cache

load_dotenv()

# Drug filename -> Russian display name and active ingredient(s)
DRUG_NAMES = {
    "paracetamol": ("Парацетамол", "парацетамол"),
    "ibuprofen": ("Ибупрофен", "ибупрофен"),
    "aspirin": ("Аспирин", "ацетилсалициловая кислота"),
    "nise": ("Найз", "нимесулид"),
    "ketorol": ("Кеторол", "кеторолак"),
    "claritin": ("Кларитин", "лоратадин"),
    "zyrtec": ("Зиртек", "цетиризин"),
    "aerius": ("Эриус", "дезлоратадин"),
    "suprastin": ("Супрастин", "хлоропирамин"),
    "clemastine": ("Клемастин", "клемастин"),
    "amoxicillin": ("Амоксициллин", "амоксициллин"),
    "sumamed": ("Сумамед", "азитромицин"),
    "ciprofloxacin": ("Ципрофлоксацин", "ципрофлоксацин"),
    "amoksiklav": ("Амоксиклав", "амоксициллин + клавулановая кислота"),
    "doxycycline": ("Доксициклин", "доксициклин"),
    "enalapril": ("Эналаприл", "эналаприл"),
    "bisoprolol": ("Бисопролол", "бисопролол"),
    "amlodipine": ("Амлодипин", "амлодипин"),
    "losartan": ("Лозартан", "лозартан"),
    "atorvastatin": ("Аторвастатин", "аторвастатин"),
    "omeprazole": ("Омепразол", "омепразол"),
    "zantac": ("Зантак", "ранитидин"),
    "motilium": ("Мотилиум", "домперидон"),
    "smecta": ("Смекта", "диосмектит"),
    "aphobazolum": ("Афобазол", "Фабомотизол"),
    "glycine": ("Глицин", "глицин"),
    "melatonin": ("Мелатонин", "мелатонин"),
    "theraflu": ("Терафлю", "парацетамол + фенилэфрин + фенирамин"),
    "acc": ("АЦЦ", "ацетилцистеин"),
    "ambrobene": ("Амбробене", "амброксол"),
}

# Section header normalization
SECTION_MAP = {
    "Действующее вещество": "Состав",
    "Состав": "Состав",
    "Состав на одну таблетку": "Состав",
    "Состав на одну капсулу": "Состав",
    "Состав на одну таблетку:": "Состав",
    "Активное вещество:": "Состав",
    "Вспомогательные вещества": "Состав",
    "Вспомогательные вещества:": "Состав",
    "Описание лекарственной формы": "Описание лекарственной формы",
    "Лекарственная форма": "Описание лекарственной формы",
    "Характеристика": "Описание лекарственной формы",
    "Фармакокинетика": "Фармакокинетика",
    "Абсорбция": "Фармакокинетика",
    "Всасывание": "Фармакокинетика",
    "Распределение": "Фармакокинетика",
    "Биотрансформация": "Фармакокинетика",
    "Метаболизм": "Фармакокинетика",
    "Выведение": "Фармакокинетика",
    "Элиминация": "Фармакокинетика",
    "Генетический полиморфизм": "Фармакокинетика",
    "Особые группы пациентов": "Фармакокинетика",
    "Пациенты с нарушением функции почек": "Фармакокинетика",
    "Пациенты с нарушением функции печени": "Фармакокинетика",
    "Пациенты с нарушением функции почек или печени": "Фармакокинетика",
    "Фармакодинамика": "Фармакодинамика",
    "Фармакологическое действие": "Фармакодинамика",
    "Механизм действия": "Фармакодинамика",
    "Фармако-терапевтическая группа": "Фармакодинамика",
    "Фармакологическая группа": "Фармакодинамика",
    "Фармакология": "Фармакодинамика",
    "Доклинические данные по безопасности": "Доклинические данные по безопасности",
    "Токсичность": "Доклинические данные по безопасности",
    "Острая токсичность": "Доклинические данные по безопасности",
    "Подострая токсичность": "Доклинические данные по безопасности",
    "Хроническая токсичность": "Доклинические данные по безопасности",
    "Влияние на репродуктивную систему и тератогенность": "Доклинические данные по безопасности",
    "Исследования на животных": "Доклинические данные по безопасности",
    "Доклинические исследования": "Доклинические данные по безопасности",
    "Клинические исследования": "Клинические исследования",
    "Клиническая эффективность и безопасность": "Клинические исследования",
    "Клиническая эффективность": "Клинические исследования",
    "Эффективность и безопасность": "Клинические исследования",
    "Показания": "Показания к применению",
    "Показания к применению": "Показания к применению",
    "Гипертоническая болезнь": "Показания к применению",
    "Противопоказания": "Противопоказания",
    "С осторожностью": "Противопоказания",
    "Применение при беременности и кормлении грудью": "Применение при беременности и кормлении грудью",
    "Беременность": "Применение при беременности и кормлении грудью",
    "Лактация": "Применение при беременности и кормлении грудью",
    "Кормление грудью": "Применение при беременности и кормлении грудью",
    "Фертильность": "Применение при беременности и кормлении грудью",
    "Применение при беременности и в период грудного вскармливания": "Применение при беременности и кормлении грудью",
    "Способ применения и дозы": "Способ применения и дозы",
    "Режим дозирования": "Способ применения и дозы",
    "Дозировка при одонтогенных инфекциях": "Способ применения и дозы",
    "Рекомендуемая доза и корректировка дозы": "Способ применения и дозы",
    "Доза при почечной недостаточности": "Способ применения и дозы",
    "Побочные действия": "Побочные действия",
    "Побочные реакции": "Побочные действия",
    "Побочные реакции в ходе клинических испытаний": "Побочные действия",
    "Побочные реакции в пострегистрационном периоде": "Побочные действия",
    "Резюме профиля безопасности": "Побочные действия",
    "Резюме нежелательных реакций": "Побочные действия",
    "Нежелательные реакции": "Побочные действия",
    "Взаимодействие": "Взаимодействие",
    "Лекарственное взаимодействие": "Взаимодействие",
    "Влияние других препаратов на фармакокинетику": "Взаимодействие",
    "Влияние аторвастатина на фармакокинетику других препаратов": "Взаимодействие",
    "Передозировка": "Передозировка",
    "Симптомы": "Передозировка",
    "Лечение": "Передозировка",
    "Особые указания": "Особые указания",
    "Меры предосторожности": "Особые указания",
    "Влияние на способность управлять транспортными средствами и механизмами": "Особые указания",
    "Влияние на способность к управлению транспортными средствами и механизмами": "Особые указания",
    "Влияние на способность управлять транспортными средствами, механизмами": "Особые указания",
    "Влияние на способность вождения транспорта и на управление машинами и механизмами": "Особые указания",
    "Гиперкалиемия": "Особые указания",
    "Гипогликемия": "Особые указания",
    "Анафилактоидные реакции во время проведения десенсибилизации": "Особые указания",
    "Форма выпуска": "Форма выпуска",
    "Гранулы для приготовления раствора": "Форма выпуска",
    "Таблетки шипучие": "Форма выпуска",
    "Дозирующие устройства:": "Форма выпуска",
    "Условия хранения": "Условия хранения",
    "Срок годности": "Срок годности",
    "Производитель": "Производитель",
    "Владелец регистрационного удостоверения": "Производитель",
    "Владелец РУ": "Производитель",
    "Условия отпуска из аптек": "Производитель",
    "Условия отпуска": "Производитель",
    "Источники информации": "Производитель"
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

    if brand.lower() == active_ingredient.lower():
        name_prefix = brand
    else:
        name_prefix = f"{brand} ({active_ingredient})"

    # Parse into sections
    sections = parse_sections(text)

    cache = _load_cache() # Load cache for this document to avoid re-generating unchanged chunks

    # Chunk each section, prefix every chunk with [Drug — Section]
    prefixed_chunks = []
    for section_name, section_body in sections:
        section_chunks = splitter.split_text(section_body)
        for chunk in section_chunks:
            questions = generate_hypothetical_questions(chunk, name_prefix, cache)
            questions_block = "Возможные вопросы:\n" + "\n".join(f"- {q}" for q in questions)
            prefixed = f"[{name_prefix} — {section_name}]\n{questions_block}\n{chunk}"
            prefixed_chunks.append(prefixed)

    _save_cache(cache)  # Save cache after processing each document, so we don't lose progress if interrupted

    # Reingesting a document replaces its chunks instead of duplicating them
    cur.execute("DELETE FROM chunks WHERE source = %s", (source_label,))

    embeddings = model.encode([f"passage: {chunk}" for chunk in prefixed_chunks])
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
