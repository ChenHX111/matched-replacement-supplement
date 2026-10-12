"""Replay the fixed review diagnostics from de-identified saved inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.abc
import itertools
import json
import math
import os
from pathlib import Path
import sys

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"


class NoNeuralImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "tensorflow", "jax"}:
            raise RuntimeError("Review diagnostics use only retained scalar/input descriptors")
        return None


sys.meta_path.insert(0, NoNeuralImports())
import numpy as np

SETTINGS = ("p5_main", "p5_corpus", "p5_model")
METHODS = ("loss_drop", "difficulty", "final_nll", "count", "U4", "G20", "one_seed")
COMPARATORS = ("loss_drop", "difficulty", "count", "U4", "G20")
ROOT_SEED = 20261010


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def write(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")


def load_core(root):
    sys.path.insert(0, str(Path(root) / "code"))
    import statistics_core
    return statistics_core


def other_seed_target(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 3 or theta.shape[1:] != (6, 8):
        raise ValueError("Expected complete block x two-background/three-seed x eight-focal grid")
    target = np.empty_like(theta)
    for background, seed in itertools.product(range(2), range(3)):
        indices = [background * 3 + other for other in range(3) if other != seed]
        target[:, background * 3 + seed] = np.mean(theta[:, indices], axis=1)
    return target


def rank_cells(left, right, core):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 3 or left.shape[-1] != 8:
        raise ValueError("Rank operands must have identical complete eight-focal cells")
    return np.asarray([[core.rank_correlation(a.tolist(), b.tolist())
                        for a, b in zip(la, ra)]
                       for la, ra in zip(left, right)], dtype=float)


def metric_cells(prediction, target, core):
    ranks = rank_cells(prediction, target, core)
    regrets = np.asarray([[core.top_two_regret(a.tolist(), b.tolist())
                           for a, b in zip(pa, ta)]
                          for pa, ta in zip(prediction, target)], dtype=float)
    return {"rank": ranks, "regret": regrets}


def different_seed_components(nonfocal, theta, core):
    if np.shape(nonfocal) != np.shape(theta):
        raise ValueError("Component grid mismatch")
    other_seed_target(theta)
    pairs = [(background * 3 + a, background * 3 + b)
             for background in range(2) for a in range(3) for b in range(3) if a != b]
    return rank_cells(np.asarray(nonfocal)[:, [a for a, _ in pairs]],
                      np.asarray(theta)[:, [b for _, b in pairs]], core)


def control_coverage(member_content, control_content):
    if len(member_content) != 254 or len(control_content) != 254:
        raise ValueError("Control coverage requires the original 254-token content windows")
    control_grams = {tuple(control_content[i:i + 4]) for i in range(251)}
    return sum(tuple(member_content[i:i + 4]) in control_grams for i in range(251)) / 255


def fixed_halves(values, blocks):
    values = np.asarray(values, dtype=float)
    if values.shape != (len(blocks),) or len(blocks) % 2 or not np.isfinite(values).all():
        raise ValueError("Strata need one finite input descriptor per block and an even population")
    order = sorted(range(len(blocks)), key=lambda index: (values[index], index))
    middle = len(order) // 2
    return {"lower": sorted(order[:middle]), "upper": sorted(order[middle:])}


def selected_two(values):
    if len(values) != 8 or not np.isfinite(values).all():
        return None
    return frozenset(sorted(range(8), key=lambda i: (-values[i], i))[:2])


def selection_stability(prediction):
    overlaps, identical = [], []
    for block in prediction:
        overlap, equal = [], []
        for background in range(2):
            sets = [selected_two(block[background * 3 + seed]) for seed in range(3)]
            for a, b in itertools.combinations(range(3), 2):
                valid = sets[a] is not None and sets[b] is not None
                overlap.append(len(sets[a] & sets[b]) / 2 if valid else np.nan)
                equal.append(float(sets[a] == sets[b]) if valid else np.nan)
        overlaps.append(overlap)
        identical.append(equal)
    return {"top_two_overlap_fraction": np.asarray(overlaps),
            "identical_top_two_fraction": np.asarray(identical)}


def point(values, core):
    values = np.asarray(values, dtype=float)
    return core.statistic(values, "mean") if np.isfinite(values).all() else None


def stability(values, blocks, core):
    values = np.asarray(values, dtype=float)
    block_means = [point(values[i:i + 1], core) for i in range(len(blocks))]
    leave_block = [point(np.delete(values, i, axis=0), core) for i in range(len(blocks))]
    return {
        "block_means": dict(zip(blocks, block_means)),
        "by_seed": {str(seed): point(values[:, [seed, 3 + seed]], core) for seed in range(3)},
        "by_background": {str(bg): point(values[:, bg * 3:bg * 3 + 3], core) for bg in range(2)},
        "leave_one_scoring_seed_out": {
            str(seed): point(values[:, [i for i in range(6) if i % 3 != seed]], core)
            for seed in range(3)},
        "leave_one_block_out": dict(zip(blocks, leave_block)),
        "leave_one_block_out_range": ([min(leave_block), max(leave_block)]
                                      if all(v is not None for v in leave_block) else None),
        "original_other_two_seed_target_unchanged": True,
    }


def load_inputs(reproduction, review_inputs):
    reproduction = Path(reproduction)
    extra = read(review_inputs)
    for name, expected in extra["reproduction_input_hashes"].items():
        if sha(reproduction / "data" / name) != expected:
            raise ValueError("Original scalar input changed: " + name)
    if sha(reproduction / "code" / "statistics_core.py") != extra["statistics_core_sha256"]:
        raise ValueError("Original statistics helper changed")
    data = read(reproduction / "data" / "cpu_inputs.json")["p5"]
    for setting, expected_blocks in zip(SETTINGS, (24, 8, 8)):
        current = data[setting]
        if len(current["blocks"]) != expected_blocks:
            raise ValueError("Original block count changed")
        if current["context_order"] != [[b, s] for b in range(2) for s in range(3)]:
            raise ValueError("Context ordering changed")
        shape = (expected_blocks, 6, 8)
        arrays = [current["oracle"], current["uncovered_count_k4"], *current["scores"].values()]
        if any(np.shape(value) != shape for value in arrays):
            raise ValueError("Incomplete or differently ordered scalar grid")
        if extra["p5"][setting]["blocks"] != current["blocks"]:
            raise ValueError("Input descriptor block join changed")
        if np.shape(extra["p5"][setting]["control_coverage_k4"]) != (expected_blocks, 8):
            raise ValueError("Incomplete matched-control descriptor")
    return data, extra


def analyze(reproduction, review_inputs, output):
    core = load_core(reproduction)
    data, extra = load_inputs(reproduction, review_inputs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results, cells, stability_rows, stratum_inputs, replay_checks = [], {}, {}, {}, []

    for setting in SETTINGS:
        current = data[setting]
        blocks = current["blocks"]
        theta = np.asarray(current["oracle"], dtype=float)
        target = other_seed_target(theta)
        scores = current["scores"]
        original = np.asarray(scores["audit_loss_drop"], dtype=float)
        difficulty = np.asarray(scores["base_difficulty_nll"], dtype=float)
        nonfocal = original - theta
        counts = np.asarray(current["uncovered_count_k4"], dtype=float) / 255
        predictors = {
            "loss_drop": original, "difficulty": difficulty, "final_nll": difficulty - original,
            "count": counts, "U4": np.asarray(scores["uncovered_base_nll_k4"], dtype=float),
            "G20": np.asarray(scores["G20"], dtype=float), "one_seed": theta,
        }
        _, saved_target = core.p5_arrays(current, "G20")
        if not np.array_equal(target, saved_target, equal_nan=True):
            raise ValueError("Original cross-seed target changed")
        cells[setting] = {"blocks": blocks, "target": target, "predictors": predictors,
                          "nonfocal": nonfocal, "endpoints": {}}
        stability_rows[setting] = {}
        metric_cache = {method: metric_cells(predictors[method], target, core) for method in METHODS}

        def add(name, values, *, units, kind="mean", indices=None, retained=None):
            values = np.asarray(values, dtype=float).reshape(len(blocks) if indices is None else len(indices), -1)
            old_name = retained
            seed = (core.stable_seed(2026100205, "exploratory", setting, old_name)
                    if old_name else core.stable_seed(ROOT_SEED, "review-v1", setting, name))
            result = core.bootstrap(values, kind=kind, confidence=.95, seed=seed,
                                    replicates=5000 if old_name else 20000)
            result.update({"id": name, "setting": setting, "units": units,
                           "role": "retained_diagnostic_replay" if old_name else "new_post_hoc",
                           "interval_status": "nominal_95_percent_not_multiplicity_adjusted",
                           "block_ids": blocks if indices is None else [blocks[i] for i in indices]})
            if old_name:
                saved = extra["p5"][setting]["retained_diagnostics"][old_name]
                difference = max(abs(result["estimate"] - saved["estimate"]),
                                 *(abs(a - b) for a, b in zip(result["interval"], saved["interval"])))
                if difference > 1e-10:
                    raise ValueError(f"Retained diagnostic replay differs: {setting}/{old_name}: {difference}")
                replay_checks.append({"setting": setting, "diagnostic": old_name,
                                      "max_absolute_difference": difference,
                                      "original_estimate_and_interval_preserved": True})
            results.append(result)
            cells[setting]["endpoints"][name] = values
            return result

        for method in METHODS:
            for endpoint, values in metric_cache[method].items():
                old = ("individual_oracle_as_cross_seed_predictor_" +
                       ("spearman" if endpoint == "rank" else "top_two_regret")) if method == "one_seed" else None
                add(f"absolute.{method}.{endpoint}", values,
                    units="Spearman" if endpoint == "rank" else "nats/token", retained=old)
                stability_rows[setting][f"{method}.{endpoint}"] = stability(values, blocks, core)
            for endpoint, values in selection_stability(predictors[method]).items():
                add(f"selection_stability.{method}.{endpoint}", values, units="fraction")

        for comparator in COMPARATORS:
            for endpoint in ("rank", "regret"):
                left, right = metric_cache["final_nll"][endpoint], metric_cache[comparator][endpoint]
                values = left - right if endpoint == "rank" else right - left
                add(f"final_nll_vs.{comparator}.{endpoint}", values,
                    units="Spearman_difference" if endpoint == "rank" else "nats/token",
                    )

        for name, values, old in (
            ("mean_A", original, "mean_original_A"),
            ("mean_theta", theta, "mean_individual_oracle"),
            ("mean_N", nonfocal, "mean_nonfocal_learning_base_minus_FOCAL_OUT"),
        ):
            add("component." + name, values, units="nats/token", retained=old)
        add("component.same_seed_rank", rank_cells(nonfocal, theta, core), units="Spearman",
            retained="rank_nonfocal_vs_same_realization_theta")
        add("component.other_two_seed_rank", rank_cells(nonfocal, target, core), units="Spearman")
        add("component.all_ordered_different_seed_rank",
            different_seed_components(nonfocal, theta, core), units="Spearman")
        add("one_seed.median_absolute_error", np.abs(theta - target), kind="median", units="nats/token",
            retained="individual_oracle_as_cross_seed_predictor_median_absolute_error")

        control = np.asarray(extra["p5"][setting]["control_coverage_k4"], dtype=float)
        control_context = np.repeat(control[:, None, :], 6, axis=1)
        add("association.U4_difficulty", rank_cells(predictors["U4"], difficulty, core), units="Spearman")
        add("association.control_coverage_target", rank_cells(control_context, target, core), units="Spearman")
        add("association.control_coverage_N", rank_cells(control_context, nonfocal, core), units="Spearman")
        descriptors = {"control_coverage": np.mean(control, axis=1),
                       "corpus_uncovered": np.mean(counts, axis=(1, 2))}
        stratum_inputs[setting] = {}
        for descriptor, block_values in descriptors.items():
            halves = fixed_halves(block_values, blocks)
            stratum_inputs[setting][descriptor] = {
                "block_values": dict(zip(blocks, block_values)), "strata": {}}
            for half, indices in halves.items():
                stratum_inputs[setting][descriptor]["strata"][half] = {
                    "blocks": [blocks[i] for i in indices],
                    "input_range": [float(min(block_values[indices])), float(max(block_values[indices]))],
                    "input_mean": float(np.mean(block_values[indices])),
                    "mean_absolute_log_source_length_ratio": float(np.mean(
                        np.asarray(extra["p5"][setting]["absolute_log_source_length_ratio"])[indices])),
                }
                strata_metrics = {
                    "mean_theta": (theta[indices], "mean", "nats/token"),
                    "mean_N": (nonfocal[indices], "mean", "nats/token"),
                    "G20_median_absolute_error": (np.abs(predictors["G20"] - target)[indices], "median", "nats/token"),
                    "U4_minus_difficulty_rank": ((metric_cache["U4"]["rank"] -
                                                 metric_cache["difficulty"]["rank"])[indices], "mean", "Spearman_difference"),
                    "difficulty_minus_U4_regret": ((metric_cache["difficulty"]["regret"] -
                                                  metric_cache["U4"]["regret"])[indices], "mean", "nats/token"),
                }
                for endpoint, (values, kind, units) in strata_metrics.items():
                    add(f"stratum.{descriptor}.{half}.{endpoint}", values,
                        kind=kind, units=units, indices=indices)
        print(f"Completed {setting}: {len(blocks)} blocks, no outcome-based exclusions", flush=True)

    report = {
        "schema": "review-existing-evidence-v1",
        "analysis_spec_sha256": extra["analysis_spec_sha256"],
        "review_inputs_sha256": sha(review_inputs),
        "statistics_core_sha256": extra["statistics_core_sha256"],
        "analysis_code_sha256": sha(__file__),
        "root_seed": ROOT_SEED, "new_bootstrap_draws": 20000,
        "target": "other two original seeds in the same background",
        "new_hypothesis_family": False, "original_hypothesis_families_modified": False,
        "summaries": results,
        "retained_diagnostic_replay_checks": replay_checks,
        "unavailable_diagnostics": extra["unavailable_diagnostics"],
    }
    write(output / "SUMMARY.json", report)
    write(output / "CELLS.json", cells)
    write(output / "STABILITY.json", stability_rows)
    write(output / "INPUT_STRATA.json", stratum_inputs)
    with (output / "SUMMARY.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("setting", "id", "role", "estimate", "lower", "upper", "units",
                  "clusters", "total_cells", "undefined_cells_count", "replicates", "cluster_seed")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            record = {key: row[key] for key in fields if key in row}
            record["lower"], record["upper"] = row["interval"] or (None, None)
            writer.writerow(record)
    print(f"Wrote {len(results)} complete-grid summaries; replayed {len(replay_checks)} retained diagnostics")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduction", type=Path, default=Path(__file__).resolve().parent.parent / "reproduction")
    parser.add_argument("--inputs", type=Path, default=Path(__file__).with_name("review_inputs.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("replayed"))
    args = parser.parse_args()
    analyze(args.reproduction, args.inputs, args.output)
