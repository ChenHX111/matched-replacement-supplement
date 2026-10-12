"""Verify the far-control artifact: manifest hashes, then a byte-identical replay in a temporary copy.

Usage (from anywhere):  python -B verify_e01.py
Re-runs the frozen analysis, the post-hoc script and the table generator in a temporary copy and
requires every result file and generated table to be byte-identical to the shipped ones.
"""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
OUTPUTS = ["results/E01_ANALYSIS_RESULT.json", "results/POSTHOC_E01.json", "generated/e01_endpoints.tex",
           "generated/e01_descriptive.tex", "generated/e01_posthoc.tex", "generated/DISPLAY_PROVENANCE.json"]


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def run(root, *args):
    subprocess.run([sys.executable, "-B", *args], cwd=root, check=True, capture_output=True)


def main():
    manifest_bytes = (ROOT / "MANIFEST.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != (ROOT / "MANIFEST.sha256").read_text(encoding="ascii").split()[0]:
        raise SystemExit("MANIFEST.json does not match MANIFEST.sha256")
    manifest = json.loads(manifest_bytes)
    for row in manifest["files"]:
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise SystemExit("file changed: " + row["path"])
    with tempfile.TemporaryDirectory() as tmp:
        copy = pathlib.Path(tmp) / "e01_reproduction"
        shutil.copytree(ROOT, copy)
        for rel in OUTPUTS:
            (copy / rel).unlink()
        run(copy, "code/analysis_e01.py", "--near", "inputs/NEAR_INPUT_BUNDLE.json", "--far", "inputs/E01_RETURN_ALL.json",
            "--output", "results/E01_ANALYSIS_RESULT.json", "--rule", "A_complete_grid")
        run(copy, "code/posthoc_e01.py", "results/POSTHOC_E01.json")
        run(copy, "code/make_e01_tables.py")
        for rel in OUTPUTS:
            if sha(copy / rel) != sha(ROOT / rel):
                raise SystemExit("replay differs: " + rel)
    print(f"PASS: {len(manifest['files'])} manifest-bound files; {len(OUTPUTS)} outputs replayed byte-identically")


if __name__ == "__main__":
    main()
