"""Runtime configuration shared by the API and retrieval components."""

import os
from pathlib import Path

import dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"

# Resolve the same .env regardless of the shell's current working directory.
# Variables explicitly set by the caller take precedence over this file.
dotenv.load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_path(name: str, default: Path) -> str:
    """Resolve configurable local paths relative to the repository root."""
    value = os.getenv(name, "").strip()
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


CSV_FILE_PATH_ZH = _env_path(
    "CSV_FILE_PATH_ZH", BACKEND_DIR / "cleaned_tech_news_rag_ready.csv"
)
CSV_FILE_PATH_EN = _env_path("CSV_FILE_PATH_EN", BACKEND_DIR / "hacker_news.csv")
CHROMA_DB_DIR_ZH = _env_path("CHROMA_DB_DIR_ZH", BACKEND_DIR / "chroma_db_ollama_qwen3_zh")
CHROMA_DB_DIR_EN = _env_path("CHROMA_DB_DIR_EN", BACKEND_DIR / "chroma_db_ollama_qwen3_en")
BM25_INDEX_PATH_ZH = _env_path("BM25_INDEX_PATH_ZH", BACKEND_DIR / "bm25_index_zh.pkl")
BM25_INDEX_PATH_EN = _env_path("BM25_INDEX_PATH_EN", BACKEND_DIR / "bm25_index_en.pkl")

_local_reranker = BACKEND_DIR / "models" / "bge-reranker-large"
_reranker_override = os.getenv("RERANKER_MODEL_PATH", "").strip()
if _reranker_override:
    # Explicit path prefixes (including Windows drives), or an existing path,
    # select local files. Other values are Hugging Face model identifiers.
    _is_local_reranker = (
        _reranker_override.startswith(("./", "../", ".\\", "..\\", "/", "\\", "~"))
        or (len(_reranker_override) > 1 and _reranker_override[1] == ":")
        or (PROJECT_ROOT / _reranker_override).exists()
    )
    RERANKER_MODEL_PATH = (
        _env_path("RERANKER_MODEL_PATH", _local_reranker)
        if _is_local_reranker
        else _reranker_override
    )
else:
    RERANKER_MODEL_PATH = (
        str(_local_reranker) if _local_reranker.is_dir() else "BAAI/bge-reranker-large"
    )

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-max")
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "qwen3.5-flash")

try:
    MAX_RESEARCH_ITERATIONS = int(os.getenv("MAX_RESEARCH_ITERATIONS", "5"))
except ValueError as exc:
    raise ValueError("MAX_RESEARCH_ITERATIONS must be a positive integer") from exc
if MAX_RESEARCH_ITERATIONS < 1:
    raise ValueError("MAX_RESEARCH_ITERATIONS must be a positive integer")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
