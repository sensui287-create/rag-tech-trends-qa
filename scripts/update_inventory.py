"""Refresh reviewed-code hashes after inspecting changes; require descriptions."""

import argparse
import hashlib
import json
import sys

from check_repository import CODE_SUFFIXES, ROOT, publication_files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-date", help="Optional ISO date for the completed review")
    args = parser.parse_args(argv)
    path = ROOT / "docs" / "code_inventory.json"
    inventory = json.loads(path.read_text(encoding="utf-8"))
    indexed = {entry["path"]: entry for entry in inventory["files"]}
    code = [source for source in publication_files() if source.suffix.lower() in CODE_SUFFIXES]
    missing = [source.relative_to(ROOT).as_posix() for source in code
               if source.relative_to(ROOT).as_posix() not in indexed]
    if missing:
        for name in missing:
            print(f"Add a reviewed inventory entry with purpose/category first: {name}")
        return 1
    updated = []
    for source in code:
        name = source.relative_to(ROOT).as_posix()
        if source.is_symlink() or not source.resolve().is_relative_to(ROOT):
            raise ValueError(f"Refusing external path: {name}")
        entry = indexed[name]
        if not entry.get("purpose") or not entry.get("category"):
            raise ValueError(f"Missing review description: {name}")
        entry["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        updated.append(entry)
    inventory["files"] = updated
    if args.review_date:
        from datetime import date
        inventory["review_date"] = date.fromisoformat(args.review_date).isoformat()
    path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated hashes for {len(updated)} described code files. This does not run or review model behavior.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
