"Used port 5433 because existing Postgres occupies 5432."
"Embedding dim is 384, column must match."
"DeepSeek for generation because OpenAI is geoblocked in Russia."

"I'm building a Russian medical RAG. Stack: DeepSeek generation, multilingual-e5-small embeddings (384-dim), pgvector in Docker on port 5433, FastAPI next. Working so far: ingest.py, search.py, rag.py — full pipeline runs. Here's where I'm stuck:"

-------------------------------- 24th may -------------------------------- 

I've done a lot of things today.
I created virtual environment for my medical-rag project, topped up my deepseek balance and created api key. Moreover, I called deepseek_api through openai chat completions endpoint, then ran small embedding model locally to create a 384 dimesional vector out of russian sentence. 
And also I ran  posgresql server in docker, then connected to it for pgvector extension. Overall, everything worked just fine. Of course with the help of my teacher =).

-------------------------------- 25th may -------------------------------- 

So, tonight, I added ingestion and searching features on only one drug, called paracetamol.
The ingest.py had basic character chunk splitting, but now it's done recursively with several
seperators ("\n\n", "\n", etc.). 
The search.py retrieves 3 the most relevant chunks out of database with cosine similarity. Retrieving happens like this "embedding <=> %s::vector" in the query.
The rag.py is the mix of the two previous files. Retrieval-Augmented-Generation

-------------------------------- 28th may -------------------------------- 

Same RAG pipeline, but it's running as an HTTP server with a documented, validated /ask endpoint, a health check, JSON in and JSON out, and the model loaded once at startup the way a real backend does it. Anyone — a frontend, another service, a curl command, an interviewer poking at /docs — can talk to the system now.

## PROJECT ORIENTATION (paste into a new chat to get up to speed)

I'm building a Russian-language medical RAG system as a portfolio project to
transition into AI/backend engineering. I'm in Russia, so Western APIs that
geoblock Russia (OpenAI, Groq) are off the table — this matters for every
tooling choice.

Stack:
- Generation: DeepSeek API via the openai client library (OpenAI-compatible,
  works from Russia). Key in .env as DEEPSEEK_API_KEY. temperature=0 for factual output.
- Embeddings: intfloat/multilingual-e5-small, run LOCALLY via sentence-transformers.
  Output dimension = 384 (the pgvector column MUST match this exactly).
- Vector store: pgvector, running in Docker (image pgvector/pgvector:pg15),
  container name "medical-rag-db", on port 5433 (5432 is taken by an older
  local Postgres install). DB name: medical_rag. Password: devpassword (local dev only).
- Chunking: LangChain RecursiveCharacterTextSplitter, chunk_size=500, overlap=100.
  (Started with a hand-written character chunker, saw it cut words mid-token,
  switched to the structure-aware splitter — visible in git history.)

Project identity / non-negotiables:
- Answers must be GROUNDED: built only from retrieved chunks, never the model's
  own training knowledge.
- Must CITE its source on every answer.
- Must say "нет ответа в документах" when the context doesn't contain the answer,
  rather than hallucinating. (This already works — verified it honestly reported
  a missing dosage instead of inventing one.)
- Always appends "Это не медицинская консультация, обратитесь к врачу."
- It's an information-retrieval tool, NOT a medical advice-giver.

Files that work so far:
- test_generation.py — proves DeepSeek call works
- test_embedding.py — proves local embedding works (prints 384)
- ingest.py — loads data/drug1.txt, chunks, embeds, stores in pgvector
- search.py — embeds a query, retrieves nearest chunks via cosine distance (<=>)
- rag.py — full pipeline: retrieve + grounded/cited generation

Current corpus: one document, data/drug1.txt (paracetamol instruction from rlsnet.ru).

To restart the database after a reboot: docker start medical-rag-db

Next planned steps: diagnose whether the precise adult dosage exists in the doc
but didn't retrieve (search drug1.txt for "масса тела более 40"); then wrap the
pipeline in a FastAPI endpoint; then build a small eval set (30-50 Q&A pairs
where I know the correct source) to measure retrieval accuracy.

## HOW I WANT MY AI TEACHER TO ACT

- I write the code myself. Don't write whole solutions for me to paste blindly —
  guide me, explain the WHY, and let me implement. The struggle is the learning.
- Explain every line/command before I run it. I don't want to run things I can't read.
- Be honest and direct, including about my mistakes — but warm, not harsh.
  (I asked for honesty and sometimes it stung; the goal is honest AND kind.)
- One step at a time. Don't dump the whole roadmap; give me the next concrete move.
- Push me to SHIP, not to collect more courses/plans. My weakness is over-planning
  and re-asking for permission instead of starting. Call it out gently if I do it.
- Stop me on a win sometimes. Ending a session on momentum matters more than
  squeezing in one more thing.
- Make me verify factual claims rather than trusting them — Western-API geoblocks
  bit me repeatedly because I assumed instead of checking.
- I'm a real person doing a hard career change under money pressure. A tool is for
  debugging and thinking; the real support comes from people and communities.

