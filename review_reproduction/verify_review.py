"""Verify the review artifact, optionally comparing a fresh numerical replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, replayed=None):
    root = Path(root).resolve()
    manifest_path = root / "MANIFEST.json"
    if sha(manifest_path) != (root / "MANIFEST.sha256").read_text(encoding="ascii").strip():
        raise ValueError("Review manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = root.joinpath(*row["path_parts"]).resolve()
        if not path.is_relative_to(root) or sha(path) != row["sha256"]:
            raise ValueError("Review artifact file changed: " + str(path))
    count = 0
    if replayed is not None:
        replayed = Path(replayed)
        for path in (root / "results").iterdir():
            if path.is_file():
                if sha(path) != sha(replayed / path.name):
                    raise ValueError("Numerical replay differs: " + path.name)
                count += 1
    print(f"PASS: {len(manifest['files'])} manifest-bound review files; {count} replay outputs compared")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replayed", type=Path)
    args = parser.parse_args()
    verify(Path(__file__).resolve().parent, args.replayed)
