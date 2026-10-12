"""Verify the post-hoc saved-data artifact: manifest hashes, then a byte-identical replay in a temporary copy.

Usage (from anywhere):  python -B verify_panel.py
The analysis scripts read the other artifacts' committed inputs (read-only) from the repository root
and write into a temporary copy of this folder; the table generator then runs in that copy. Every result
file and generated table must be byte-identical to the shipped ones.
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
RESULTS = {"code/posthoc_gradient_comparator.py": "results/POSTHOC_GRADIENT_COMPARATOR.json",
           "code/saved_data_checks.py": "results/SAVED_DATA_CHECKS.json",
           "code/saved_data_checks_2.py": "results/SAVED_DATA_CHECKS_2.json",
           "code/background_overlap.py": "results/BACKGROUND_OVERLAP.json",
           "code/saved_data_checks_3.py": "results/SAVED_DATA_CHECKS_3.json"}
GENERATED = ["generated/panel_head_to_head.tex", "generated/panel_gradient.tex", "generated/panel_robustness.tex",
             "generated/panel_selection.tex", "generated/panel_reliability.tex", "generated/panel_support.tex",
             "generated/panel_decomposition.tex",
             "generated/DISPLAY_PROVENANCE.json"]


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    manifest_bytes = (ROOT / "MANIFEST.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != (ROOT / "MANIFEST.sha256").read_text(encoding="ascii").split()[0]:
        raise SystemExit("MANIFEST.json does not match MANIFEST.sha256")
    manifest = json.loads(manifest_bytes)
    for row in manifest["files"]:
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise SystemExit("file changed: " + row["path"])
    with tempfile.TemporaryDirectory() as tmp:
        copy = pathlib.Path(tmp) / "panel_reproduction"
        shutil.copytree(ROOT, copy)
        for rel in list(RESULTS.values()) + GENERATED:
            (copy / rel).unlink()
        for script, output in RESULTS.items():
            subprocess.run([sys.executable, "-B", str(copy / script), str(REPO), str(copy / output)],
                           cwd=copy, check=True, capture_output=True)
        subprocess.run([sys.executable, "-B", str(copy / "code" / "make_panel_tables.py")], cwd=copy, check=True,
                       capture_output=True)
        for rel in list(RESULTS.values()) + GENERATED:
            if sha(copy / rel) != sha(ROOT / rel):
                raise SystemExit("replay differs: " + rel)
    print(f"PASS: {len(manifest['files'])} manifest-bound files; "
          f"{len(RESULTS) + len(GENERATED)} outputs replayed byte-identically")


if __name__ == "__main__":
    main()
