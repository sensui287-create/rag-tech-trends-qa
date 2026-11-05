"""Build a source ZIP, its SHA-256 sidecar and a per-file integrity manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from check_repository import ROOT, check_files, publication_files

MANIFEST_NAME = "ARCHIVE_MANIFEST.json"


def artifact_paths(target: Path) -> tuple[Path, Path, Path]:
    return target, target.with_suffix(".manifest.json"), target.with_suffix(".zip.sha256")


def validate_target(root: Path, target: Path, sources: list[Path]) -> Path:
    """Protect existing source/private files and custom output artifacts."""
    if target.suffix.lower() != ".zip":
        raise ValueError("Output must use a .zip filename.")
    if any(path.is_symlink() for path in (target, *target.parents)):
        raise ValueError("Output must not use a symbolic link.")
    target = target.resolve()
    source_paths = {path.resolve() for path in sources}
    defaults = set(artifact_paths((root / "dist" / "LongChain-github.zip").resolve()))
    for path in artifact_paths(target):
        if path in source_paths:
            raise ValueError("Output must not overwrite a source file.")
        if path.exists() and path not in defaults:
            raise ValueError("Refusing to overwrite an existing custom output. Choose a new filename.")
    return target


def create_archive(root: Path, sources: list[Path], target: Path) -> dict:
    """Write a deterministic snapshot after the caller validates its contents."""
    root = root.resolve()
    target = validate_target(root, target, sources)
    snapshots = []
    for path in sorted(sources):
        name = path.relative_to(root).as_posix()
        if name == MANIFEST_NAME:
            raise ValueError(f"{MANIFEST_NAME} is reserved for the generated manifest.")
        snapshots.append((name, path.read_bytes()))
    manifest = {
        "schema_version": 1,
        "project": "LongChain",
        "algorithm": "sha256",
        "scope": "Published source snapshot; excludes this generated manifest itself.",
        "file_count": len(snapshots),
        "files": [
            {"path": name, "size_bytes": len(content),
             "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in snapshots
        ],
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".longchain-", suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in snapshots + [(MANIFEST_NAME, manifest_bytes)]:
                info = zipfile.ZipInfo("LongChain/" + name)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content)
        with zipfile.ZipFile(temp_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive integrity check failed.")
        os.replace(temp_path, target)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    _, manifest_path, hash_path = artifact_paths(target)
    manifest_path.write_bytes(manifest_bytes)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    hash_path.write_text(f"{digest}  {target.name}\n", encoding="ascii", newline="\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "LongChain-github.zip")
    args = parser.parse_args()
    try:
        files = publication_files()
        errors = check_files(ROOT, files)
        if errors:
            for error in errors:
                print(error)
            return 1
        manifest = create_archive(ROOT, files, args.output)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        if isinstance(exc, ValueError):
            print(f"Archive preparation failed: {exc}")
        else:
            print(f"Archive preparation failed: {type(exc).__name__}. Check Git and output permissions.")
        return 1
    target = args.output.resolve()
    print(f"Created {target} ({manifest['file_count']} source files plus integrity manifest; "
          f"{target.stat().st_size / 1024:.1f} KiB).")
    print("Wrote per-file manifest and ZIP SHA-256 sidecar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
