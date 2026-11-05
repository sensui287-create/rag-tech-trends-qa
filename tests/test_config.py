"""Offline configuration checks; no RAG dependencies or real .env are loaded."""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import dotenv


ROOT = Path(__file__).resolve().parents[1]
CONFIG_SOURCE = ROOT / "backend" / "rag_core" / "config.py"


@contextmanager
def temporary_cwd(path: Path):
    original = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original)


def load_config():
    """Load just config.py without importing the LangChain application."""
    spec = importlib.util.spec_from_file_location("isolated_rag_config", CONFIG_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        dotenv_patch = mock.patch("dotenv.load_dotenv")
        self.load_dotenv = dotenv_patch.start()
        self.addCleanup(dotenv_patch.stop)

    def test_defaults_and_dotenv_path_are_independent_of_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            with temporary_cwd(Path(directory)):
                config = load_config()

        self.assertEqual(config.PROJECT_ROOT, ROOT)
        self.assertEqual(config.BACKEND_DIR, ROOT / "backend")
        self.assertEqual(
            Path(config.CSV_FILE_PATH_ZH),
            ROOT / "backend" / "cleaned_tech_news_rag_ready.csv",
        )
        self.assertEqual(Path(config.CSV_FILE_PATH_EN), ROOT / "backend" / "hacker_news.csv")
        self.assertEqual(
            Path(config.CHROMA_DB_DIR_ZH), ROOT / "backend" / "chroma_db_ollama_qwen3_zh"
        )
        self.assertEqual(Path(config.BM25_INDEX_PATH_EN), ROOT / "backend" / "bm25_index_en.pkl")
        self.load_dotenv.assert_called_once_with(ROOT / ".env", override=False)

    def test_environment_and_relative_path_overrides(self):
        os.environ.update({
            "CSV_FILE_PATH_ZH": "examples/data/tech_news_zh.csv",
            "CSV_FILE_PATH_EN": "examples/data/tech_news_en.csv",
            "CHROMA_DB_DIR_ZH": "data/chroma_zh",
            "CHROMA_DB_DIR_EN": "data/chroma_en",
            "BM25_INDEX_PATH_ZH": "data/cache/zh.pkl",
            "BM25_INDEX_PATH_EN": "data/cache/en.pkl",
            "OLLAMA_BASE_URL": "http://localhost:12345",
            "EMBEDDING_MODEL": "example-embedding",
            "LLM_MODEL": "example-llm",
            "FAST_LLM_MODEL": "example-fast-llm",
            "MAX_RESEARCH_ITERATIONS": "3",
        })
        with tempfile.TemporaryDirectory() as directory:
            with temporary_cwd(Path(directory)):
                config = load_config()

        for name in (
            "CSV_FILE_PATH_ZH", "CSV_FILE_PATH_EN", "CHROMA_DB_DIR_ZH",
            "CHROMA_DB_DIR_EN", "BM25_INDEX_PATH_ZH", "BM25_INDEX_PATH_EN",
        ):
            with self.subTest(name=name):
                self.assertEqual(Path(getattr(config, name)), (ROOT / os.environ[name]).resolve())
        self.assertEqual(config.OLLAMA_BASE_URL, "http://localhost:12345")
        self.assertEqual(config.EMBEDDING_MODEL, "example-embedding")
        self.assertEqual(config.LLM_MODEL, "example-llm")
        self.assertEqual(config.FAST_LLM_MODEL, "example-fast-llm")
        self.assertEqual(config.MAX_RESEARCH_ITERATIONS, 3)

    def test_explicit_environment_takes_precedence_over_dotenv(self):
        # Use the real parser with an in-memory stream, so the repository's .env
        # never contributes credentials or machine-specific settings to this test.
        actual_loader = dotenv.main.load_dotenv

        def load_controlled_dotenv(path, *, override):
            return actual_loader(
                stream=io.StringIO("LLM_MODEL=from-dotenv\nFAST_LLM_MODEL=fast-from-dotenv\n"),
                override=override,
            )

        self.load_dotenv.side_effect = load_controlled_dotenv
        os.environ["LLM_MODEL"] = "from-environment"
        config = load_config()
        self.assertEqual(config.LLM_MODEL, "from-environment")
        self.assertEqual(config.FAST_LLM_MODEL, "fast-from-dotenv")
        self.load_dotenv.assert_called_once_with(ROOT / ".env", override=False)

    def test_hugging_face_model_id_is_preserved(self):
        os.environ["RERANKER_MODEL_PATH"] = "BAAI/bge-reranker-large"
        config = load_config()
        self.assertEqual(config.RERANKER_MODEL_PATH, "BAAI/bge-reranker-large")

    def test_explicit_relative_model_path_resolves_from_root(self):
        os.environ["RERANKER_MODEL_PATH"] = "./custom-model"
        config = load_config()
        self.assertEqual(Path(config.RERANKER_MODEL_PATH), ROOT / "custom-model")

    def test_absolute_model_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            os.environ["RERANKER_MODEL_PATH"] = str(model_path)
            config = load_config()
        self.assertEqual(Path(config.RERANKER_MODEL_PATH), model_path.resolve())

    def test_invalid_iteration_counts_are_rejected(self):
        for value in ("0", "-1", "not-an-integer", "1.5", ""):
            with self.subTest(value=value):
                os.environ["MAX_RESEARCH_ITERATIONS"] = value
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    load_config()


if __name__ == "__main__":
    unittest.main()
