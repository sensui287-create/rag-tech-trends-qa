"""Check the exact Git publication set without importing or running RAG code."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_SIZE = 10 * 1024 * 1024
CODE_SUFFIXES = {".py", ".html", ".js", ".mjs", ".cjs", ".ts", ".ps1", ".sh", ".yml", ".yaml"}
TOKEN_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"tvly-[A-Za-z0-9_-]{16,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
FORBIDDEN_SUFFIXES = {
    ".pkl", ".pickle", ".sqlite3", ".safetensors", ".pt", ".pth", ".onnx",
    ".zip", ".pyc", ".pem", ".key", ".bin",
}
REQUIRED_FILES = (
    "README.md", "LICENSE", ".gitignore", ".env.example", "requirements.txt",
    "CONTRIBUTING.md", "SECURITY.md", "backend/server.py",
    "examples/data/tech_news_zh.csv", "examples/data/tech_news_en.csv",
    "docs/ARCHIVE.md", "docs/code_inventory.json", "experiments/agent_experiment.py",
    "experiments/agent_results.py",
)


def publication_files(root: Path = ROOT) -> list[Path]:
    """Use Git's actual ignore rules, including tracked files when present.

    For an uninitialized folder, a disposable Git directory outside the folder
    evaluates .gitignore without creating or changing the user's repository.
    Global excludes and the system Git configuration are disabled for this step.
    """
    root = root.resolve()
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("GIT_"):
            del env[name]
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    with tempfile.TemporaryDirectory(prefix="longchain-check-") as temp:
        if (root / ".git").exists():
            command = ["git", "-C", str(root)]
        else:
            subprocess.run(["git", "init", "--quiet", temp], env=env, check=True,
                           capture_output=True)
            command = ["git", f"--git-dir={Path(temp) / '.git'}", f"--work-tree={root}"]
        result = subprocess.run(
            command + ["-c", f"core.excludesFile={os.devnull}", "ls-files", "-z",
                       "--cached", "--others", "--exclude-standard"],
            cwd=root, env=env, check=True, capture_output=True,
        )
    return [root / name for name in sorted(set(result.stdout.decode("utf-8").split("\0"))) if name]


def local_secret_values(root: Path) -> list[str]:
    """Read local .env values only for matching; never display their contents."""
    path = root / ".env"
    if not path.is_file():
        return []
    values = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*?)\s*$", line)
        if match and re.search(r"KEY|TOKEN|SECRET|PASSWORD", match[1], re.I):
            value = match[2].strip("\"'")
            if len(value) >= 12 and not re.search(r"replace|example|your-|placeholder", value, re.I):
                values.append(value)
    return values


def check_inventory(root: Path, files: list[Path]) -> list[str]:
    """Require an up-to-date review entry for every published code file."""
    inventory_path = root / "docs" / "code_inventory.json"
    if not inventory_path.is_file():
        return ["Missing code review inventory: docs/code_inventory.json"]
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        entries = inventory["files"]
        if not isinstance(entries, list):
            raise ValueError("Invalid inventory entries")
        indexed = {entry["path"]: entry for entry in entries}
        if len(indexed) != len(entries):
            raise ValueError("Duplicate inventory paths")
    except (OSError, ValueError, TypeError, KeyError):
        return ["Invalid code review inventory: docs/code_inventory.json"]
    actual = {path.relative_to(root).as_posix(): path for path in files
              if path.suffix.lower() in CODE_SUFFIXES and path.is_file()
              and not path.is_symlink() and path.resolve().is_relative_to(root)}
    errors = []
    for name, path in actual.items():
        entry = indexed.get(name)
        if entry is None:
            errors.append(f"Code file missing from review inventory: {name}")
        elif entry.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            errors.append(f"Code changed since review inventory was recorded: {name}")
    for name in sorted(set(indexed) - set(actual)):
        errors.append(f"Stale path in review inventory: {name}")
    return errors


def check_files(root: Path, files: list[Path]) -> list[str]:
    """Return location-only diagnostics, never source lines or token values."""
    root = root.resolve()
    errors = []
    names = {path.relative_to(root).as_posix() for path in files}
    for name in REQUIRED_FILES:
        if name not in names:
            errors.append(f"Missing publication file: {name}")
    secrets = local_secret_values(root)
    for path in files:
        name = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            errors.append(f"Symlink or external path: {name}")
            continue
        if not path.is_file():
            errors.append(f"Missing file or unsupported submodule: {name}")
            continue
        normalized_name = name.lower()
        parts = Path(normalized_name).parts
        private = (
            any(part in {".git", ".idea", ".venv", "__pycache__"} for part in parts)
            or (path.name.lower().startswith(".env") and path.name.lower() != ".env.example")
            or normalized_name.startswith(("backend/models/", "backend/chroma_db", "data/", "dist/"))
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
            or (len(parts) == 2 and parts[0] == "backend"
                and path.suffix.lower() in {".csv", ".json", ".pdf", ".svg", ".txt"})
        )
        if private:
            errors.append(f"Local-only resource in publication set: {name}")
            continue
        if path.stat().st_size > MAX_FILE_SIZE:
            errors.append(f"File exceeds publication limit (10 MiB): {name}")
            continue
        try:
            source = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            errors.append(f"Unreviewed binary or non-UTF-8 file: {name}")
            continue
        if path.suffix.lower() == ".py":
            try:
                ast.parse(source, filename=name)
            except SyntaxError as exc:
                errors.append(f"Python syntax error: {name}:{exc.lineno}")
        for number, line in enumerate(source.splitlines(), 1):
            if any(pattern.search(line) for pattern in TOKEN_PATTERNS) or any(value in line for value in secrets):
                errors.append(f"Possible credential: {name}:{number} (value hidden)")
    errors.extend(check_inventory(root, files))
    return errors


def main() -> int:
    try:
        files = publication_files()
        errors = check_files(ROOT, files)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Repository check could not run: {type(exc).__name__}. Ensure Git is installed.")
        return 1
    if errors:
        for error in errors:
            print(error)
        return 1
    size = sum(path.stat().st_size for path in files)
    print(f"PASS: {len(files)} publication files, {size / 1024:.1f} KiB; syntax and credential checks passed.")
    print("This is an offline publication check; it does not validate model APIs or experimental results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
