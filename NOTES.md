"Used port 5433 because existing Postgres occupies 5432."
"Embedding dim is 384, column must match."
"DeepSeek for generation because OpenAI is geoblocked in Russia."

"I'm building a Russian medical RAG. Stack: DeepSeek generation, multilingual-e5-small embeddings (384-dim), pgvector in Docker on port 5433, FastAPI next. Working so far: ingest.py, search.py, rag.py — full pipeline runs."

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

I containerized the whole project today. Wrote a dockerfile for the app, then a docker-compose.yml that runs the app and pgvector together with one command. Made ingest.py create the pgvector extension and chunks table on its own with IF NOT EXISTS, so the database sets itself up. Moved DB connection settings to env vars so the same code works locally and inside docker.
Biggest things I actually learned: dockerfile layer order matters (put rarely-changing stuff before COPY . . or you re-download everything on every code change), "localhost" inside a container means the container itself not my machine, services find each other by name on the docker network, and pip pulls GPU torch by default which dragged in 5GB of CUDA I don't use until I switched to the CPU index.
Hit two real snags. pip timed out downloading CUDA, fixed by using torch's CPU index. Then after compose-up the chunks table didn't exist in the new database because I'd created it manually in the old container, fixed by making ingest.py bootstrap the schema itself. Lesson: anything infrastructure should self-create from code.
The whole stack runs with `docker compose up --build` + `docker compose exec app python ingest.py`.

-------------------------------- 30th may -------------------------------- 
Today I reorganized the structure of the project files and added 12 more drugs to the corpus. 
Moreover, I created sources.md containing names of the drugs and links to them. Changed file names to their actual drug names. 
Then modified ingest.py file to load all the txt files from data/drugs directory which contains all the drugs.
And I'm super tired rn!!! But, it was worth it.