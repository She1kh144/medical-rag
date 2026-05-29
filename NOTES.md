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

Same RAG pipeline, but it's running as an HTTP server with a documented, validated /ask endpoint, a health check, JSON in and JSON out, and the model loaded once at startup the way a real backend does it. Anyone - a frontend, another service, a curl command, an interviewer poking at /docs can talk to the system now.

-------------------------------- 29th may -------------------------------- 

Eval v1: 100% retrieval top-3, 90% answer accuracy (9/10) on 10 Russian medical questions. Single failure is a known recall ceiling on factual-list chunks - same pattern observed in paracetamol dose diagnostic. Likely addressable with a reranker or section-aware chunking.

Q7 deep-dive: correct GI contraindications chunk exists in corpus, ranks in positions 11-40 (verified surfaces at k=40). Likely cause: semantic crowding with other contraindication chunks. Production k=10 stays; reranker would be the natural fix. Documented as known limit, not patched.

