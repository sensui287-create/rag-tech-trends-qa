# 🚀 TechPulse-RAG: A RAG-Powered Intelligent Q&A System for Tech Trends

[中文文档](README.md) · [Quick start](#quick-start) · [Verification](#verification)

A research prototype for agentic retrieval-augmented generation, built with LangGraph and FastAPI. It routes simple questions to a retrieval path and complex questions to planning, candidate retrieval, reranking, distillation and reflection. Responses and source metadata are streamed to a browser through SSE.

The default retrieval ensemble combines **Chinese** Chroma vector retrieval and Chinese BM25. English indexes and bilingual query rewriting are available internally, but English retrieval is not fused into the default graph. The main workflow searches a local corpus rather than the live web. This project is independent of the official LangChain project.

## Quick start

Use Python 3.11, a running [Ollama](https://ollama.com/) service and a DashScope credential with access to the configured models.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
# Copy .env.example to .env, then fill in DASHSCOPE_API_KEY.
ollama pull qwen3-embedding:4b
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). API documentation is at `/docs`, with status at `/health`. Initialization builds indexes and loads a reranker before the server accepts requests. The frontend uses CDN-hosted libraries.

The template config uses synthetic bilingual demo CSV files and isolated indexes under `data/demo/`. The reranker defaults to `BAAI/bge-reranker-large`, downloaded on first use; an explicit local path can be supplied instead. Model availability, downloads and API usage depend on your environment and provider account. Both language corpora or their existing valid indexes are currently required at initialization.

Data/index paths resolve from the repository root. Shell environment variables take precedence over the root `.env`. When changing corpora, embedding models or splitting settings, use new Chroma and BM25 paths: existing nonempty indexes are reused. Load pickle caches only from trusted environments.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m pyflakes backend experiments tests
```

These tests cover configuration, experiment input and output, frontend SSE decoding, and HTTP/SSE behavior with a fake backend. They do not validate live models or benchmark results. Runtime requirements use version ranges rather than a fully tested inference lockfile. Experiments run from the repository root with `python -m experiments.<module> --help`. They require `requirements.txt` plus `openai>=1.50,<3`, `requests>=2.32,<3`, `tqdm>=4.66,<5`, `tabulate>=0.9,<1` and `httpx>=0.27,<1`.

Original corpora, private configuration, model weights, retrieval databases, caches and experimental results are excluded from this repository. Synthetic sample data is included for a basic workflow check. It cannot reproduce the original experiment results.

## Limitations and contributing

This is a local research demo without authentication or request quotas. Experimental commands need additional dependencies, data and model services. Structured model output can fail validation, and generated citations do not guarantee factual correctness.

Contributions are welcome through issues and pull requests. Do not post credentials or private data in public issues. Original project code and synthetic samples are available under the [MIT License](LICENSE). Third-party models, libraries and original corpora retain their own terms.
