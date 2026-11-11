"""Tests for read-only research release integrity verification."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "scripts/check_released_artifacts.py"
spec = importlib.util.spec_from_file_location("release_integrity", MODULE)
integrity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integrity)


class ReleaseIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "report.txt").write_bytes(b"frozen artifact")
        self.manifest = self.root / "sha256.json"

    def write_manifest(self, entries):
        self.manifest.write_text(json.dumps(entries), encoding="utf-8")

    def test_valid_manifest(self):
        sha = hashlib.sha256(b"frozen artifact").hexdigest()
        self.write_manifest({"report.txt": sha})
        self.assertEqual(integrity.check(self.manifest, self.root), (1, []))

    def test_changed_artifact_is_rejected(self):
        self.write_manifest({"report.txt": hashlib.sha256(b"original").hexdigest()})
        count, failures = integrity.check(self.manifest, self.root)
        self.assertEqual(count, 1)
        self.assertIn("mismatch", failures[0])

    def test_missing_and_invalid_entries_are_rejected(self):
        self.write_manifest({"missing.txt": "0" * 64, "report.txt": "not a SHA"})
        count, failures = integrity.check(self.manifest, self.root)
        self.assertEqual(count, 2)
        self.assertEqual(len(failures), 2)

    def test_parent_traversal_is_rejected(self):
        self.write_manifest({"../outside.txt": "0" * 64})
        _, failures = integrity.check(self.manifest, self.root)
        self.assertIn("outside repository", failures[0])

    def test_empty_manifest_is_rejected(self):
        self.write_manifest({})
        with self.assertRaises(ValueError):
            integrity.check(self.manifest, self.root)


if __name__ == "__main__":
    unittest.main()
