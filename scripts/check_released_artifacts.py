"""Read-only SHA-256 integrity check for the frozen v1.0.0 research release."""

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/releases/v1.0.0/artifacts.sha256.json"
SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def check(manifest_path=MANIFEST, root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Expected a non-empty path-to-SHA256 manifest")
    failures = []
    for name, expected in sorted(manifest.items()):
        if not isinstance(name, str) or not isinstance(expected, str) or not SHA256.fullmatch(expected):
            failures.append(f"{name!r}: invalid manifest entry")
            continue
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(f"{name}: missing or outside repository")
            continue
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            failures.append(f"{name}: SHA-256 mismatch ({actual})")
    return len(manifest), failures


def main():
    count, failures = check()
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        print(f"FAILED: {len(failures)}/{count} frozen artifacts", file=sys.stderr)
        return 1
    print(f"PASS: {count} frozen v1.0.0 artifacts match their SHA-256 manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
