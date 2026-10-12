"""Verify the exposure-and-gradient replay artifact: manifest hashes, then a byte-identical replay in a temporary copy.

Usage (from anywhere):  python -B verify_e06.py
Re-runs the frozen analysis, the post-hoc script and the table generator in a temporary copy and requires every
result file and generated table to be byte-identical to the shipped ones. The near bundle, the follow-up contexts
and the far-control return are read (read-only) from the other artifacts' committed inputs.
"""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parent
NEAR = REPO / "e01_reproduction" / "inputs" / "NEAR_INPUT_BUNDLE.json"
FOLLOWUP = REPO / "followup_reproduction" / "inputs" / "CONTEXTS.json"
FAR = REPO / "e01_reproduction" / "inputs" / "E01_RETURN_ALL.json"
EXTERNAL = {NEAR: "e3e5a9cf750eaca3e894a91d6dbab2ece054cc1cdc878a6b8a3ff009eab1a12c",
            FOLLOWUP: "1be362477773f1a735bf7640e730593f18199541c12dfabe8360e57f1f1c8e0a",
            FAR: "83434b3cb7cfd731fd1b5c475392c509fb7ebd2caa0a1993cdcbad804aa76370"}
OUTPUTS = ["results/E06_ANALYSIS_RESULT.json", "results/POSTHOC_E06.json", "generated/e06_main.tex",
           "generated/e06_endpoints.tex", "generated/e06_exposure.tex", "generated/e06_gradient.tex",
           "generated/e06_posthoc.tex", "generated/DISPLAY_PROVENANCE.json"]


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def run(root, *args):
    subprocess.run([sys.executable, "-B", *map(str, args)], cwd=root, check=True, capture_output=True)


def main():
    manifest_bytes = (ROOT / "MANIFEST.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != (ROOT / "MANIFEST.sha256").read_text(encoding="ascii").split()[0]:
        raise SystemExit("MANIFEST.json does not match MANIFEST.sha256")
    manifest = json.loads(manifest_bytes)
    for row in manifest["files"]:
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise SystemExit("file changed: " + row["path"])
    for path, digest in EXTERNAL.items():
        if sha(path) != digest:
            raise SystemExit("external input changed: " + str(path.relative_to(REPO)))
    with tempfile.TemporaryDirectory() as tmp:
        copy = pathlib.Path(tmp) / "e06_reproduction"
        shutil.copytree(ROOT, copy)
        for rel in OUTPUTS:
            (copy / rel).unlink()
        run(copy, "code/analysis_e06.py", "--return", "inputs/E06_RETURN_ALL.json", "--near", NEAR,
            "--followup", FOLLOWUP, "--far", FAR, "--e01-in-epochs", "inputs/E01_IN_EPOCH_READOUTS.json",
            "--output", "results/E06_ANALYSIS_RESULT.json")
        run(copy, "code/posthoc_e06.py", "--analysis", "code/analysis_e06.py", "--return", "inputs/E06_RETURN_ALL.json",
            "--near", NEAR, "--far", FAR, "--e01-in-epochs", "inputs/E01_IN_EPOCH_READOUTS.json",
            "--result", "results/E06_ANALYSIS_RESULT.json", "--output", "results/POSTHOC_E06.json")
        run(copy, "code/make_e06_tables.py")
        for rel in OUTPUTS:
            if sha(copy / rel) != sha(ROOT / rel):
                raise SystemExit("replay differs: " + rel)
    print(f"PASS: {len(manifest['files'])} manifest-bound files; {len(OUTPUTS)} outputs replayed byte-identically")


if __name__ == "__main__":
    main()
