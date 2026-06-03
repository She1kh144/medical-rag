# Medical RAG (Russian)

A retrieval-augmented question-answering system over Russian-language drug instructions, with strict source citation and no-medical-advice guardrails.

## What it does

The user asks a question in Russian about a medication. The system retrieves the most semantically relevant chunks from a corpus of drug instructions, then generates an answer grounded strictly in those chunks, with the source cited and a mandatory disclaimer attached. When the retrieved context does not contain the answer, the system reports that honestly instead of hallucinating.

It is an information-retrieval tool, not a medical advisor.

## Architecture

- **FastAPI** service exposing `/ask` and `/health` endpoints
- **pgvector** (PostgreSQL with vector extension) as the vector store and citation metadata store
- **multilingual-e5-small** sentence-transformer running locally for embeddings (384-dimensional)
- **bge-reranker-v2-m3** cross-encoder, lazy-loaded, used as a second-stage reranker
- **DeepSeek API** for generation (OpenAI-compatible)
- The full stack runs via **docker-compose** with one command

Request flow:

```
user query
   │
   ▼
FastAPI /ask
   │
   ▼
embed query (e5, "query:" prefix)
   │
   ▼
pgvector cosine similarity search → top-k candidates
   │
   ▼  (optional, enabled via "rerank": true)
cross-encoder rerank → top-10 by relevance
   │
   ▼
DeepSeek generation, grounded in retrieved chunks
   │
   ▼
cited answer + disclaimer
```

## Design decisions

**DeepSeek for generation.** OpenAI and several other Western LLM APIs are geoblocked in Russia. DeepSeek is accessible, OpenAI-compatible (so the standard `openai` Python client works), and cost-effective at this scale.

**pgvector over a dedicated vector database.** The corpus is small (~700 chunks across 15 documents) and every chunk needs to carry citation metadata (source document) alongside its embedding. Keeping vectors and relational metadata in a single Postgres table simplifies queries and removes a moving part. A dedicated vector DB like Qdrant would be the right migration target if the corpus grew significantly or required advanced payload filtering.

**Local embeddings instead of an embedding API.** `multilingual-e5-small` is ~470MB and runs on CPU. No paid embedding service is needed, there are no rate limits, and document text doesn't leave the local environment.

**`query:` / `passage:` prefixes on e5 inputs.** The multilingual-e5 family was trained with these prefixes; using them consistently at ingest and query time improves retrieval quality measurably. Discovered by reading the model card after a first pass without prefixes underperformed.

**RecursiveCharacterTextSplitter (structure-aware chunking).** An initial hand-written character-count chunker produced visible mid-word and mid-sentence cuts on Russian text, which degraded retrieval on factual passages. Switching to LangChain's RecursiveCharacterTextSplitter — which prefers paragraph and sentence boundaries before falling back to character cuts — improved chunk coherence and retrieval distance scores on the same queries.

**Section-aware chunking with drug-name prefixes.** Documents are parsed into their standard sections (Показания, Противопоказания, Способ применения и дозы, etc.) and each chunk is prefixed with [Brand (active ingredient) — Section]. This embeds drug identity and section context directly into every chunk vector, so a query like "противопоказания ибупрофена" or "с какого возраста принимать лоратадин" aligns with the prefix and pulls the right chunk.

**Cross-encoder reranking in second stage.** After scaling the corpus from 3 to 15 documents, baseline answer accuracy dropped because semantic neighbors crowd the right chunks at top-k. A cross-encoder reranker (bge-reranker-v2-m3) re-scores the top-50 bi-encoder candidates by reading query + chunk together, surfacing the actually-relevant passage. Toggled per-request via a `rerank` flag; default off because latency on CPU is ~50s/request. See Evaluation for the measured impact.

**Reranker lazy-loaded; HuggingFace cache mounted from host.** The cross-encoder model is ~2.27GB. Rather than baking it into the Docker image (tripling image size), it's loaded on first use and read from a host-mounted HuggingFace cache. The Docker image stays small; the model downloads once on the host and is reused across container restarts.

**`temperature=0` for generation.** This is a factual-retrieval tool, not a creative one. Determinism is preferable to variety.

**Auto-discovered ingest.** `ingest.py` reads every `.txt` file from `data/drugs` and derives the source label from the filename. Adding a document is one drag-and-drop plus a re-ingest; no code change.

## Safety properties

- Answers are constructed only from retrieved context, never from the model's training knowledge. This is enforced by the system prompt.
- Every answer cites its source document.
- Every answer ends with: «Это не медицинская консультация, обратитесь к врачу.»
- When the retrieved context does not contain the answer, the system says so explicitly. This behavior is verified by the evaluation set.

## Evaluation

The evaluation set (`drug_questions.json`) contains 29 Russian questions across the 15-drug corpus. Each question specifies an expected source document and a list of keyword stems that should appear in a correct answer. The eval script (`evaluate.py`) hits the running `/ask` endpoint twice — once with reranking off, once on — and produces two result files for direct comparison.

The scorer checks two things:

- **Retrieval accuracy (top-3):** does at least one chunk from the expected source appear in the top three retrieved chunks?
- **Answer accuracy:** does the generated answer contain at least one expected keyword stem **and** cite the expected source label?

Keyword matching uses stem fragments rather than full inflected forms, to accommodate Russian morphology (e.g. `печеноч` matches `печеночная`, `печеночной`, `печеночному`).

### Results

| Configuration | Retrieval (top-3) | Answer accuracy |
|---|---|---|
| Old chunking, no reranker | 96% (28/29) | 86% (25/29) |
| Old chunking, with reranker | 93% (27/29) | 96% (28/29) |
| Section-aware chunking, no reranker | 96% (28/29) | **96% (28/29)** |
| Section-aware chunking, with reranker | 96% (28/29) | 96% (28/29) |

Reranking lifted answer accuracy from 86% to 96% on the old chunking — (one question was evaluated wrongly in the previous tests, now, the table is correct), replicated on the expanded 29-question eval as +10%. Section-aware chunking achieved the same 96% on its own, at zero runtime cost. Adding the reranker on top of section-aware chunking provided no further gain (96% → 96%), suggesting the two techniques addressed the same underlying failure (right chunk not surfacing among semantic neighbors) and that the chunking version is the cleaner solution.

## Known limitations / Future work

- **Reranker latency.** ~50s/request on CPU. Acceptable for evaluation, not for interactive use. Production deployment would require GPU inference, a lighter reranker, or a different approach entirely.
- **Hybrid retrieval (BM25 + vectors)** would likely address the same recall-ceiling problem at lower runtime cost than reranking — keyword matching catches exact terms (drug names, specific dosages) that semantic search misses. Postgres has full-text search built in, so this could be implemented in the same database without new infrastructure.
- **Eval set size.** 29 questions is enough to surface system properties but produces coarse percentages. Growing the set to 50-100 would tighten the numbers and exercise more cross-drug discrimination cases.
- **Substring-based eval.** Stem matching handles Russian inflection but not deeper paraphrasing. An LLM-as-judge eval would be more robust at the cost of additional API calls.
- **General-purpose generation model.** `deepseek-chat` is not medically tuned. Real medical use would require a domain-tuned model and clinical review.
- **Retrieval is based on textual similarity.** queries phrased in terms not used by the source document (e.g., asking by age when the source dosing is by weight) may fail to retrieve relevant chunks. Query rewriting via the LLM could mitigate this.
- **Numeric-fact retrieval is still weak.** The current eval set contains one question that fails across all four configurations: "У какого препарата период полувыведения около 27 часов?" — the answer exists in the Эриус document, but "27 часов" carries weak semantic signal (most pharmacokinetics chunks discuss half-lives in hours), so pure semantic retrieval can't distinguish it. This is the canonical case for hybrid retrieval (BM25 keyword + vector), where exact-token matching would surface the right chunk instantly. This is the next planned improvement.
- **Hybrid retrieval was evaluated and did not improve accuracy.** A BM25 keyword-search stage (Postgres tsvector + RRF merge with vector search) was implemented as a candidate fix for queries on specific numeric facts. On the 64-question eval, hybrid produced no measurable lift over section-aware chunking alone. The diagnosis: natural-language Russian queries ("период полувыведения около 27 часов") rarely share enough exact tokens with stemmed chunks ("T1/2 — 20–30 ч") for BM25 to surface the right answer, while RRF still pulls ranking toward low-quality BM25 matches. Implementation preserved on the hybrid-retrieval branch for reference. The principled fix for this failure mode is query rewriting (HyDE), which translates user phrasing toward document phrasing before retrieval.

## Running it

### Prerequisites

- Docker Desktop
- A DeepSeek API key (https://platform.deepseek.com)

### Setup

1. Clone the repository.
2. Copy `.env.example` to `.env` and fill in the values.
3. Bring up the stack:

   ```bash
   docker compose up --build
   ```

4. In a separate terminal, ingest the sample documents:

   ```bash
   docker compose exec app python ingest.py
   ```

5. Open the interactive API docs:

   ```
   http://localhost:8000/docs
   ```

### Running the evaluation

With the stack running, from the host:

```bash
python evaluate.py
```

The script runs with rerank off (you can turn it on by setting rerank true) and writes `eval_results.json`. Aggregate scores print to the terminal.

Note: the first rerank-enabled run on a clean host will download the cross-encoder model (~2.27GB) into the local HuggingFace cache. Subsequent runs reuse it.

## Project structure

```
.
├── app.py                    # FastAPI service: /ask and /health
├── ingest.py                 # Auto-discovers data/*.txt, chunks, embeds, stores in pgvector
├── evaluate.py               # Runs the eval set against /ask
├── scripts/
│   └── search.py             # Script for inspecting retrieval
├── data/         
│   ├── drugs/                # 15 Russian drug instructions (.txt)
│   ├── drug_questions.json   # Evaluation set
│   └── sources.md            # Names and sources of the drugs
├── Dockerfile                # App container definition
├── docker-compose.yml        # Full stack: app + pgvector
├── requirements.txt          # Direct Python dependencies
├── .env.example              # Template for environment variables
└── README.md
```

## Tech stack

Python 3.12, FastAPI, Uvicorn, pgvector, sentence-transformers, LangChain text splitters, psycopg2, Docker, DeepSeek API.
