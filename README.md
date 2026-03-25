## Setup

Copy [.env.example](/home/nguyenquocdung/PAM/.env.example) to `.env` and choose providers separately for:

- `EMBEDDING_PROVIDER`: `openai`, `gemini`, `hf_inference`, `ollama`
- `EXTRACTION_PROVIDER`: `openai`, `gemini`, `hf_inference`, `ollama`
- `GRAPH_EMBEDDING_PROVIDER`: `openai`, `gemini`, `hf_inference`, `ollama`

Common combinations:

- Remote + local:
  `EMBEDDING_PROVIDER=hf_inference`, `EXTRACTION_PROVIDER=ollama`
- Fully OpenAI:
  `EMBEDDING_PROVIDER=openai`, `EXTRACTION_PROVIDER=openai`, `GRAPH_EMBEDDING_PROVIDER=openai`
- Mostly local:
  `EMBEDDING_PROVIDER=ollama`, `EXTRACTION_PROVIDER=ollama`, `GRAPH_EMBEDDING_PROVIDER=ollama`

## Benchmark

Use the diagnostic script before a full ingest:

```bash
python3 scripts/benchmark_ingest.py data/test_docs/SYSTEM_DESIGN.md
python3 scripts/benchmark_ingest.py data/test_docs/SYSTEM_DESIGN.md --embed
python3 scripts/benchmark_graph.py data/test_docs/SYSTEM_DESIGN.md --max-chunks 5
```

This reports chunk counts, optional embedding latency, and graph extraction timing with the current `.env`.

The app and benchmark scripts now validate provider settings at startup. If a selected provider is missing its token, base URL, or model, startup fails immediately with a direct error.

If `EMBEDDING_PROVIDER=ollama` and you see `unsupported value: NaN`, the selected Ollama embedding model is unstable in your runtime. Use a dedicated embedding model such as `nomic-embed-text`, `nomic-embed-text-v2-moe`, or `mxbai-embed-large`.

## Reset

```bash
docker compose down -v --remove-orphans
rm -rf data/neo4j_data data/es_data data/postgres_data
rm -rf migrations/versions/*
docker compose up -d
alembic revision --autogenerate -m "initial"
alembic upgrade head
```
