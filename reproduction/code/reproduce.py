"""Reproduce the anonymous numerical artifact: saved scalars only, no model or data download."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "tensorflow", "jax"}:
            raise RuntimeError("This artifact performs scalar reproduction only")
        return None


sys.meta_path.insert(0, NoModels())
import statistics_core as core
import asset_generator
import verify_paper_values
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inventory(root):
    if sha(root / "MANIFEST.json") != (root / "MANIFEST.sha256").read_text(encoding="ascii").strip():
        raise RuntimeError("Artifact manifest digest differs")
    manifest = read(root / "MANIFEST.json")
    for record in manifest["files"]:
        path = root.joinpath(*record["path_parts"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != record["sha256"]:
            raise RuntimeError("Artifact inventory mismatch")
    return manifest


def compare_summary(actual, expected, context):
    if actual["estimate"] != expected["estimate"] or actual["interval"] != expected["interval"]:
        raise RuntimeError("Exact saved-scalar reproduction failed: " + context)


def reproduce(root, out, *, quick=False, resume=False):
    manifest = verify_inventory(root)
    protocol = read(root / "protocol.json")
    inputs = read(root / "data" / "cpu_inputs.json")
    counts = read(root / "reference" / "COUNT_CHECK.json")
    sensitivity = read(root / "reference" / "BOOTSTRAP_SENSITIVITY.json")
    values = read(root / "data" / "paper_values.json")
    asset_inputs = read(root / "data" / "asset_scalars.json")
    if inputs["protocol_sha256"] != protocol["binding_protocol_sha256"]:
        raise RuntimeError("Input/protocol binding differs")
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    for spec in protocol["bootstrap_check"]["comparisons"]:
        data, subtract, kind, strata = core.primary_cells(inputs, spec)
        summary = core.bootstrap(
            data, subtract=subtract, kind=kind, strata=strata,
            confidence=1-.05/11, seed=spec["bootstrap_seed"], replicates=5000,
        )
        compare_summary(summary, {"estimate": spec["original_estimate"], "interval": spec["original_interval"]}, spec["id"])
        assert core.directional(summary, spec["direction"]) == spec["original_direction_supported"]
        checks.append({"id": spec["id"], "original_5000_exact": True})
    print("ORIGINAL_5000_EXACT", len(checks), flush=True)
    count_copy = copy.deepcopy(counts)
    metric_cache = {}
    for name, setting in inputs["p5"].items():
        setting["scores"]["unweighted_k4"] = (np.asarray(setting["uncovered_count_k4"], dtype=float)/255).tolist()
        for method in ("unweighted_k4", "uncovered_base_nll_k4", "base_difficulty_nll"):
            metric_cache[(name, method)] = core.p5_metric_cells(setting, method)
    for row in count_copy["comparisons"]:
        left = metric_cache[(row["setting"], "unweighted_k4")][row["endpoint"]]
        right = metric_cache[(row["setting"], row["right"])][row["endpoint"]]
        data = left-right if row["endpoint"] == "spearman" else right-left
        summary = core.bootstrap(data, confidence=.95, seed=row["bootstrap_seed"], replicates=5000)
        compare_summary(summary, {"estimate": row["estimate"], "interval": [row["ci_low"], row["ci_high"]]}, row["id"])
        checks.append({"id": row["id"], "count_comparison_exact": True})
    for row in count_copy["absolute_unweighted_metrics"]:
        data = metric_cache[(row["setting"], "unweighted_k4")][row["endpoint"]]
        summary = core.bootstrap(data, confidence=.95, seed=row["bootstrap_seed"], replicates=5000)
        compare_summary(summary, {"estimate": row["estimate"], "interval": [row["ci_low"], row["ci_high"]]}, row["id"])
        checks.append({"id": row["id"], "count_absolute_exact": True})
    write(out / "COUNT_CHECK.json", count_copy)
    print("COUNT_CHECK_EXACT", 18, flush=True)
    if not quick:
        for row in sensitivity["rows"]:
            spec = row["specification"]
            saved_path = out / "sensitivity" / (row["id"] + ".json")
            checkpoint_path = out / "checkpoints" / (row["id"] + ".json")
            if resume and saved_path.exists():
                summary = read(saved_path)
            else:
                data, subtract, kind, strata = core.primary_cells(inputs, spec)
                state = read(checkpoint_path) if resume and checkpoint_path.exists() else None
                def checkpoint(state):
                    write(checkpoint_path, state)
                summary = core.bootstrap(
                    data, subtract=subtract, kind=kind, strata=strata, confidence=1-.05/11,
                    seed=spec["bootstrap_seed"], replicates=100000, checkpoint=checkpoint, resume=state,
                )
                write(saved_path, summary)
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
            compare_summary(summary, row["sensitivity"], row["id"] + ":100000")
            row["sensitivity"] = summary
            checks.append({"id": row["id"], "same_100000_sequence_exact": True})
            print("FIXED_100000_REPLAY_EXACT", row["id"], flush=True)
    write(out / "BOOTSTRAP_SENSITIVITY.json", sensitivity)
    replay = verify_paper_values.verify(values, asset_inputs)
    write(out / "paper_value_replay.json", replay)
    if not replay["all_pass"]:
        raise RuntimeError("Existing source-summary replay failed")
    asset_generator.generate(values, count_copy, sensitivity, out / "generated_assets")
    assets = read(root / "reference" / "ASSET_HASHES.json")
    for name, expected_hash in assets["files"].items():
        if sha(out / "generated_assets" / name) != expected_hash:
            raise RuntimeError("Paper asset differs: " + name)
    report = {
        "schema": "anonymous-numerical-reproduction-report-v1",
        "status": "PASS", "mode": "quick_original_and_count" if quick else "full_fixed_numerical_replay",
        "checks": checks, "source_summary_checks": replay["checks_run"],
        "source_summary_max_absolute_difference": replay["maximum_absolute_difference"],
        "byte_identical_paper_assets": len(assets["files"]),
        "original_5000_tests": 11, "count_checks": 18,
        "same_100000_sensitivity_tests": 0 if quick else 11,
        "new_model_calls": 0, "new_fits": 0, "new_random_sequences": 0,
        "python": sys.version.split()[0], "numpy": np.__version__,
        "manifest_inputs_verified": len(manifest["files"]),
        "original_decisions_retained": True,
    }
    write(out / "REPRODUCTION_REPORT.json", report)
    print("ANONYMOUS_REPRODUCTION_PASS", report["byte_identical_paper_assets"], "identical paper assets", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reproduce(root, args.output if args.output else root / "reproduced",
              quick=args.quick, resume=args.resume)


if __name__ == "__main__":
    main()
