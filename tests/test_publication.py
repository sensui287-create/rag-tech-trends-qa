"""Publication safety checks; no network, models or experiment execution."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_repository import check_files, check_inventory, publication_files
from prepare_release import MANIFEST_NAME, artifact_paths, create_archive, validate_target


class PublicationTests(unittest.TestCase):
    def test_gitignore_excludes_private_resources_and_keeps_examples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".gitignore").write_text(".env\nbackend/models/\nbackend/*.csv\ndist/\n", encoding="utf-8")
            for name in (".env", "backend/models/weights.bin", "backend/corpus.csv",
                         "examples/data/demo.csv", "dist/upload.zip", ".env.example", "backend/server.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("demo", encoding="utf-8")
            names = {path.relative_to(root).as_posix() for path in publication_files(root)}
            self.assertEqual(names, {".gitignore", ".env.example", "backend/server.py", "examples/data/demo.csv"})

    def test_credentials_are_detected_without_disclosing_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Construct the fixture so the repository's own scanner stays clean.
            token = "sk-" + "a" * 24
            path = root / "demo.py"
            path.write_text(f"API_KEY = '{token}'\n", encoding="utf-8")
            errors = check_files(root, [path])
            self.assertTrue(any("Possible credential: demo.py:1" in error for error in errors))
            self.assertNotIn(token, "\n".join(errors))

    def test_local_env_secret_is_detected_even_without_known_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            token = "local-" + "x" * 24
            (root / ".env").write_text(f"CUSTOM_API_KEY={token}\n", encoding="utf-8")
            path = root / "notes.md"
            path.write_text(f"Accidental credential: {token}\n", encoding="utf-8")
            self.assertTrue(any("Possible credential: notes.md:1" in error for error in check_files(root, [path])))

    def test_forced_private_files_and_invalid_python_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            secret = root / ".env"
            secret.write_text("", encoding="utf-8")
            source = root / "invalid.py"
            source.write_text("def broken(\n", encoding="utf-8")
            errors = check_files(root, [secret, source])
            self.assertIn("Local-only resource in publication set: .env", errors)
            self.assertTrue(any("Python syntax error: invalid.py" in error for error in errors))

    def test_ignored_but_tracked_credentials_are_still_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True, capture_output=True)
            (root / ".gitignore").write_text(".env\n", encoding="utf-8")
            private = root / ".env"
            private.write_text("", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "-f", ".env"], check=True, capture_output=True)
            files = publication_files(root)
            self.assertIn(private, files)
            self.assertIn("Local-only resource in publication set: .env", check_files(root, files))

    def test_inventory_detects_new_and_modified_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "demo.py"
            source.write_text("answer = 42\n", encoding="utf-8")
            (root / "docs").mkdir()
            inventory = root / "docs" / "code_inventory.json"
            inventory.write_text(json.dumps({"files": [{
                "path": "demo.py", "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }]}), encoding="utf-8")
            self.assertEqual(check_inventory(root, [source]), [])
            source.write_text("answer = 43\n", encoding="utf-8")
            self.assertTrue(any("changed since review" in error for error in check_inventory(root, [source])))
            additional = root / "extra.py"
            additional.write_text("pass\n", encoding="utf-8")
            self.assertTrue(any("missing from review" in error for error in check_inventory(root, [source, additional])))

    def test_archive_manifest_hashes_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.py"
            source.write_text("answer = 42\n", encoding="utf-8")
            first = root / "first.zip"
            second = root / "second.zip"
            create_archive(root, [source], first)
            create_archive(root, [source], second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                manifest = json.loads(archive.read("LongChain/" + MANIFEST_NAME))
                self.assertEqual(manifest["file_count"], 1)
                record = manifest["files"][0]
                self.assertEqual(record["path"], "source.py")
                self.assertEqual(record["sha256"], hashlib.sha256(archive.read("LongChain/source.py")).hexdigest())
            _, manifest_path, hash_path = artifact_paths(first)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), manifest)
            self.assertEqual(hash_path.read_text(encoding="ascii").split()[0], hashlib.sha256(first.read_bytes()).hexdigest())

    def test_archive_output_cannot_overwrite_private_files_or_existing_sidecars(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private = root / ".env"
            private.write_text("local configuration", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "zip filename"):
                validate_target(root, private, [])
            target = root / "custom.zip"
            target.with_suffix(".manifest.json").write_text("keep me", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "existing custom output"):
                validate_target(root, target, [])
            self.assertEqual(private.read_text(encoding="utf-8"), "local configuration")


if __name__ == "__main__":
    unittest.main()
