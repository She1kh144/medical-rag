# Medical RAG (Russian)

A retrieval-augmented question-answering system over Russian-language drug instructions, with strict source citation and no-medical-advice guardrails.

## What it does

The user asks a question in Russian about a medication. The system retrieves the most semantically relevant chunks from a corpus of drug instructions, then generates an answer grounded strictly in those chunks, with the source cited and a mandatory disclaimer attached. When the retrieved context does not contain the answer, the system reports that honestly instead of hallucinating.

It is an information-retrieval tool, not a medical advisor.

## Live demo

Deployment runbook in [`DEPLOY.md`](DEPLOY.md). Tested production deployment on Ubuntu 22.04/24.04 via Docker Compose, with Caddy reverse proxy and automatic Let's Encrypt HTTPS. The instance is not currently live to keep hosting costs at zero; it can be redeployed in ~45 minutes.

## Architecture

- **FastAPI** service exposing `/ask` and `/health` endpoints
- **pgvector** (PostgreSQL with vector extension) as the vector store and citation metadata store
- **multilingual-e5-small** sentence-transformer running locally for embeddings (384-dimensional)
- **bge-reranker-v2-m3** cross-encoder, lazy-loaded, used as a second-stage reranker
- **DeepSeek API** for generation (OpenAI-compatible)
- The full stack runs via **docker-compose** with one command
- A static **single-page** frontend (static/index.html) consumes the streaming endpoint via Server-Sent Events. Markdown rendering, keyboard shortcuts, example queries, and responsive mobile layout. No build step — vanilla HTML/CSS/JS.

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

**Cross-encoder reranking** After scaling the corpus from 3 to 15 documents, baseline answer accuracy dropped because semantic neighbors crowd the right chunks at top-k. A cross-encoder reranker (bge-reranker-v2-m3) re-scores the top-50 bi-encoder candidates by reading query + chunk together, surfacing the actually-relevant passage. Toggled per-request via a `rerank` flag; default off because latency on CPU is ~50s/request.

**Reranker lazy-loaded; HuggingFace cache mounted from host.** The cross-encoder model is ~2.27GB. Rather than baking it into the Docker image (tripling image size), it's loaded on first use and read from a host-mounted HuggingFace cache. The Docker image stays small; the model downloads once on the host and is reused across container restarts.

**`temperature=0` for generation.** This is a factual-retrieval tool, not a creative one. Determinism is preferable to variety.

**Auto-discovered ingest.** `ingest.py` reads every `.txt` file from `data/drugs` and derives the source label from the filename. Adding a document is one drag-and-drop plus a re-ingest; no code change.

**Hypothetical questions** at ingest time directly addresses the user-phrasing-to-document-phrasing gap that section-aware prefixing and contextual chunks could not. By generating 3 synthetic user-style questions per chunk and embedding them alongside the chunk text, retrieval no longer relies on the user phrasing their question the way the document is written. Particularly effective on numeric-fact lookups, where the chunk now carries 'У какого препарата период полувыведения 27 часов?' as embedded text, sitting much closer to real user queries than the document text 'T1/2 — 20–30 ч' does.

## Safety properties

- Answers are constructed only from retrieved context, never from the model's training knowledge. This is enforced by the system prompt.
- Every answer cites its source document.
- Every answer ends with: «Это не медицинская консультация, обратитесь к врачу.»
- When the retrieved context does not contain the answer, the system says so explicitly. This behavior is verified by the evaluation set.

## Evaluation

The evaluation set (`data/drug_questions.json`) contains 64 Russian questions across the 15-drug corpus. Categories include single-drug factual lookup, symptom-based ambiguity ("что от боли в суставах"), active-substance queries (using ingredient name not brand), numeric-fact lookup (T1/2 values, percentages, age cutoffs), drug interactions, pregnancy/lactation, and refusal probes for out-of-scope questions.

Each question specifies an expected source document and a list of keyword stems that should appear in a correct answer. The eval script (`evaluate.py`) hits the running `/ask` endpoint and writes `eval_result.json`.

The scorer checks two things:

- **Retrieval accuracy (top-3):** does at least one chunk from the expected source appear in the top three retrieved chunks?
- **Answer accuracy:** does the generated answer contain at least one expected keyword stem **and** cite the expected source label?

Keyword matching uses stem fragments rather than full inflected forms to accommodate Russian morphology (e.g. `печеноч` matches `печеночная`, `печеночной`, `печеночному`). Dash and `ё`/`е` variants are normalized.

### Results (no reranker)

| Configuration | Retrieval (top-3) | Answer accuracy |
|---|---|---|
| **Hypothetical questions (production)** | **89% (57/64)** | **95% (61/64)** |
| Section-aware chunking | 87% (56/64) | 92% (59/64) |
| Section-aware + contextual chunks | 87% (56/64) | 92% (59/64) |
| Section-aware + hybrid retrieval | 87% (56/64) | 92% (59/64) |
| Old chunking | 89% (57/64) | 76% (49/64) |

Hypothetical questions at ingest is the production retrieval strategy: 95% answer accuracy on the 64-question eval, +19% over the original baseline and +3% over section-aware chunking alone. Section-aware chunking was a substantial intermediate win (+16% over old chunking) and remains in the production pipeline; hypothetical questions builds on top of it. Hybrid retrieval and contextual chunks were evaluated and did not outperform section-aware alone — see Known limitations for diagnosis.

## Known limitations / Future work

- **Reranker latency.** ~50s/request on CPU. Acceptable for evaluation, not for interactive use. Production deployment would require GPU inference or a lighter reranker.

- **Eval set size.** 64 questions is enough to surface system properties but produces coarse percentages. Growing the set to 100+ would tighten the numbers and exercise more cross-drug discrimination cases.

- **Substring-based eval.** Stem matching handles Russian inflection but not deeper paraphrasing. An LLM-as-judge eval would be more robust at the cost of additional API calls.

- **General-purpose generation model.** `deepseek-chat` is not medically tuned. Real medical use would require a domain-tuned model and clinical review.

- **Retrieval is based on textual similarity.** Queries phrased in terms not used by the source document (e.g., asking by age when the source dosing is by weight) may fail to retrieve relevant chunks.

- **Hybrid retrieval was evaluated and did not improve accuracy.** A BM25 keyword-search stage (Postgres tsvector + RRF merge with vector search) was implemented as a candidate fix for queries on specific numeric facts. On the 64-question eval, hybrid produced no measurable lift over section-aware chunking alone. The diagnosis: natural-language Russian queries ("период полувыведения около 27 часов") rarely share enough exact tokens with stemmed chunks ("T1/2 — 20–30 ч") for BM25 to surface the right answer, while RRF still pulls ranking toward low-quality BM25 matches. Implementation preserved on the `hybrid-retrieval` branch.

- **HyDE evaluated by simulation, not implemented.** Hypothetical Document Embeddings were tested by hand-simulating LLM-rewritten queries against the corpus. The hypothetical answers failed to surface correct chunks for reverse-lookup numeric queries, indicating the failure mode is not a phrasing gap (which HyDE addresses) but a rare-fact problem — specific numbers in a single chunk cannot compete in embedding space against the broader semantic neighborhood. The principled fix would be structured metadata extraction at ingest time (parsing "T1/2 — 27 ч" into a queryable numeric field), out of scope for this prototype.

- **Contextual chunks (Anthropic-style) evaluated, no lift.** Generating per-chunk LLM context and embedding it alongside chunk text matched section-aware chunking but did not exceed it. Diagnosis: section-aware prefixes already encode most of what cheap contextualization would add. Implementation preserved on the `contextual-chunks` branch.

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
├── app.py                       # FastAPI service: /ask, /ask/stream, /health
├── ingest.py                    # Chunks, embeds, stores chunks + hypothetical questions
├── hypotheticals.py             # Generates synthetic questions per chunk (used by ingest)
├── evaluate.py                  # Runs the eval set against /ask
├── hypothetical_questions.json  # Cached LLM-generated questions (committed for reproducibility)
├── scripts/
│   └── search.py                # Script for inspecting retrieval
├── data/
│   ├── drugs/                   # 15 Russian drug instructions (.txt)
│   ├── drug_questions.json      # Evaluation set
│   └── sources.md               # Names and sources of the drugs
├── static/
│   └── index.html               # Streaming frontend (vanilla HTML/CSS/JS)
├── DEPLOY.md                    # Deployment runbook
├── Dockerfile                   # App container definition
├── docker-compose.yml           # Full stack: app + pgvector (+ Caddy in deployment)
├── requirements.txt             # Direct Python dependencies
├── .env.example                 # Template for environment variables
└── README.md
```

## Branches

- `main` — production state (section-aware chunking + hypothetical questions)
- `hybrid-retrieval` — BM25 hybrid retrieval, evaluated and rejected (see Known limitations)
- `contextual-chunks` — Anthropic-style contextual chunks, evaluated and not adopted

## Tech stack

Python 3.12, FastAPI, Uvicorn, pgvector, sentence-transformers, LangChain text splitters, psycopg2, Docker, DeepSeek API.
