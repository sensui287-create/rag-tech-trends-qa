"""Shared, opt-in API configuration for evaluation and dataset generation.

Importing this module only loads the repository's .env file. A client is created
when create_eval_client() is called; no API request is made by this module.
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")

@lru_cache(maxsize=1)
def create_eval_client():
    """Create the separate OpenAI-compatible evaluation client on first use."""
    api_key = os.getenv("EVAL_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "请在项目根目录 .env 或环境变量中设置 EVAL_API_KEY。"
            "评测/数据生成密钥独立于主系统 DASHSCOPE_API_KEY。"
        )

    from openai import OpenAI

    options = {
        "api_key": api_key,
        "base_url": os.getenv("EVAL_BASE_URL", "").strip() or "https://api.openai.com/v1",
        "timeout": 60.0,
    }
    proxy_url = os.getenv("EVAL_PROXY_URL", "").strip()
    if proxy_url:
        import httpx

        options["http_client"] = httpx.Client(proxy=proxy_url)

    return OpenAI(**options)
