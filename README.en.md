# LongChain

[中文文档](README.md) · [API](docs/API.md) · [Experiments](docs/EXPERIMENTS.md) · [Archive inventory](docs/ARCHIVE.md) · [Publishing](docs/PUBLISHING.md)

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
python scripts/check_repository.py
python -m unittest discover -s tests -v
python scripts/prepare_release.py
```

These checks cover publication files, reviewed-code hashes, credential patterns, Python syntax, configuration and HTTP/SSE behavior with a fake backend. They do not validate live models or benchmark results. Runtime requirements use version ranges rather than a fully tested inference lockfile. Experiments live in `experiments/` and run from the repository root using `python -m experiments.<module>`; consult the Chinese [experiment notes](docs/EXPERIMENTS.md). Only `tests/` contains the offline test suite. See the [validation record](docs/VALIDATION.md) for the tested scope.

The archive is written to `dist/LongChain-github.zip`, with a per-file manifest and a ZIP SHA-256 sidecar. Original corpora, private configuration, model weights, retrieval databases, caches and experimental results are excluded; original local resources remain on disk. Extract the archive and upload its source files, rather than uploading the ZIP as the repository contents. Historical interfaces and analysis scripts are separated under `archive/`; its contents are reference material rather than supported service entry points.

## Limitations and contributing

This is a local research demo without authentication or request quotas. Some historical experiment scripts need adaptation, and the original experiments cannot be reproduced from the included synthetic data. Structured model output can fail validation, and generated citations do not guarantee factual correctness.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). Original project code, documentation and synthetic samples are available under the [MIT License](LICENSE). Third-party models, libraries and original corpora retain their own terms; see [THIRD_PARTY.md](docs/THIRD_PARTY.md).
