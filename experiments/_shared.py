"""Standard-library utilities shared by the optional experiment commands."""

import csv
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_DIR = PROJECT_ROOT / "backend"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

def save_json(path, payload):
    """Replace a complete checkpoint atomically, including after partial failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

def normalize_id(value):
    """Normalize integer IDs read as floats without damaging dotted string IDs."""
    text = str(value).strip()
    if re.fullmatch(r"(?:zh_|en_)?\d+\.0", text):
        return text[:-2]
    return text

def load_records(path, required_keys=()):
    records = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a nonempty JSON array")
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} must be an object")
        for key in required_keys:
            value = record.get(key)
            if value is None or not str(value).strip():
                raise ValueError(f"{path}: record {index} has no {key}")
    return records

def load_news_csv(path, language):
    """Read the five-column local news format, with or without a header.

    Header aliases match the application corpus: id/序号, title/新闻标题,
    content/新闻内容, date/新闻日期, URL/链接. IDs retain zh_/en_ prefixes.
    """
    aliases = {
        "id": ("id", "序号"),
        "title": ("title", "新闻标题"),
        "content": ("content", "新闻内容"),
        "date": ("date", "新闻日期", "时间"),
        "url": ("url", "链接", "新闻链接"),
    }
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        return []
    header = [value.strip().lower() for value in rows[0]]
    has_header = any(name in header for name in aliases["title"]) and any(
        name in header for name in aliases["content"]
    )
    positions = {key: index for index, key in enumerate(aliases)}
    if has_header:
        positions = {key: next((header.index(a) for a in names if a in header), None)
                     for key, names in aliases.items()}
        rows = rows[1:]
    documents = []
    for index, row in enumerate(rows):
        if not any(value.strip() for value in row):
            continue
        values = {key: row[pos].strip() if pos is not None and pos < len(row) else ""
                  for key, pos in positions.items()}
        if not values["content"]:
            continue
        identifier = normalize_id(values["id"] or index)
        prefix = language + "_"
        values["id"] = identifier if identifier.startswith(prefix) else prefix + identifier
        values["lang"] = language
        values["text"] = (f"标题: {values['title']}\n内容: {values['content']}" if language == "zh"
                          else f"Title: {values['title']}\nContent: {values['content']}")
        documents.append(values)
    return documents
