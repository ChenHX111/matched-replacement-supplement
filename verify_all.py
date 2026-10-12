"""Run every module's own checks in a temporary copy of this repository; the checkout is never modified.

Usage:  python -B verify_all.py          (about two minutes)
        python -B verify_all.py --full   (also replays the 100,000-draw sensitivity in reproduction/; slow)

Requires Python 3.11 and NumPy 1.26.4 (NumPy is used by reproduction/, review_reproduction/ and
panel_reproduction/; the other modules use the standard library only).
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent
MODULES = ["reproduction", "followup_reproduction", "review_reproduction", "e01_reproduction",
           "panel_reproduction", "e06_reproduction"]


def steps(full):
    quick = [] if full else ["--quick"]
    return [
        ("reproduction: unit tests", "reproduction", ["code/test_core.py"]),
        ("reproduction: replay" + ("" if full else " (quick)"), "reproduction", ["code/reproduce.py", *quick]),
        ("followup_reproduction: verify", "followup_reproduction", ["replay.py", "verify"]),
        ("review_reproduction: unit tests", "review_reproduction", ["test_saved_diagnostics.py"]),
        ("review_reproduction: replay", "review_reproduction", ["saved_diagnostics.py", "--output", "replayed"]),
        ("review_reproduction: verify", "review_reproduction", ["verify_review.py", "--replayed", "replayed"]),
        ("e01_reproduction: verify", ".", ["e01_reproduction/verify_e01.py"]),
        ("panel_reproduction: verify", ".", ["panel_reproduction/verify_panel.py"]),
        ("e06_reproduction: verify", ".", ["e06_reproduction/verify_e06.py"]),
    ]


def main():
    full = "--full" in sys.argv[1:]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        copy = pathlib.Path(tmp) / "supplement"
        copy.mkdir()
        for name in MODULES:
            shutil.copytree(ROOT / name, copy / name)
        for label, cwd, args in steps(full):
            started = time.monotonic()
            done = subprocess.run([sys.executable, "-B", *args], cwd=copy / cwd, env=env,
                                  capture_output=True, text=True)
            seconds = time.monotonic() - started
            status = "PASS" if done.returncode == 0 else "FAIL"
            failures += status == "FAIL"
            last = (done.stdout.strip().splitlines() or [""])[-1]
            print(f"{status}  {label:45s} {seconds:6.1f}s  {last[:90]}")
            if status == "FAIL":
                print(done.stdout[-2000:], done.stderr[-2000:], sep="\n")
    print("ALL CHECKS PASSED" if not failures else f"{failures} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
