"""Replay already reported paper summaries from de-identified saved scalars."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import statistics_core as core
import numpy as np


def record_cells(record, inputs):
    kind = record["kind"]
    setting_name = record.get("setting")
    if kind == "comparison":
        return core.primary_cells(inputs, record["specification"])
    if kind == "p5_metric":
        values = core.p5_metric_cells(inputs["p5"][setting_name], record["method"])[record["endpoint"]]
        statistic = "median" if record["endpoint"] == "median_absolute_error" else "mean"
        return values, None, statistic, None
    if kind == "repeat_error":
        setting = inputs["p5"][setting_name]
        _, reference = core.p5_arrays(setting, "G20")
        values = np.abs(np.asarray(setting["oracle"])-reference).reshape(len(setting["blocks"]), -1)
        return values, None, "median", None
    if kind == "reliability":
        oracle = np.asarray(inputs["p5"][setting_name]["oracle"])
        if record["endpoint"] == "oracle_cross_seed_stability":
            values = [[
                core.rank_correlation(block[bg*3+a].tolist(), block[bg*3+b].tolist())
                for bg in (0, 1) for a, b in itertools.combinations(range(3), 2)
            ] for block in oracle]
        else:
            values = [[core.rank_correlation(
                [core.mean(block[:3, i].tolist()) for i in range(8)],
                [core.mean(block[3:, i].tolist()) for i in range(8)],
            )] for block in oracle]
        return np.asarray(values, dtype=float), None, "mean", None
    p4 = inputs["p4"][setting_name]
    if kind == "p4_effect":
        values = p4["endpoints"][record["endpoint"]]
    elif kind == "p4_panel":
        values = p4["panels"][record["endpoint"]]
    elif kind == "p4_paired_panel":
        values = p4["panel_effects"][record["endpoint"]]
    else:
        raise ValueError("Unknown frozen summary recipe: " + kind)
    return np.asarray(values), None, "mean", p4["strata"]


def verify(values, inputs):
    checks = []
    for rid, record in values["records"].items():
        data, subtract, kind, strata = record_cells(record, inputs)
        result = core.bootstrap(
            data, subtract=subtract, kind=kind, strata=strata, confidence=record["confidence"],
            seed=record["seed"], replicates=5000,
        )
        differences = [abs(result["estimate"]-record["estimate"])]
        differences.extend(abs(a-b) for a, b in zip(result["interval"], record["interval"]))
        checks.append({
            "id": rid, "max_absolute_difference": max(differences),
            "within_saved_scalar_rounding_tolerance": max(differences) <= 1e-10,
            "source": record["source"],
        })
    for tail in values["g20_tails"]:
        prediction, reference = core.p5_arrays(inputs["p5"][tail["setting"]], "G20")
        error = prediction-reference
        calculated = {
            "signed_bias_mean": float(np.mean(error)),
            "median_absolute_error": float(np.median(np.abs(error))),
            "p95_absolute_error": float(np.quantile(np.abs(error), .95)),
        }
        for key, actual in calculated.items():
            difference = abs(actual-tail[key])
            checks.append({
                "id": "existing_tail:" + tail["setting"] + ":" + key,
                "max_absolute_difference": difference,
                "within_saved_scalar_rounding_tolerance": difference <= 1e-10,
                "source": tail["source"],
            })
    for cost in values["costs"]:
        for key in ("total", "training", "scoring", "diagnostics"):
            difference = abs(cost["seconds_" + key]/3600 - cost["summed_gpu_process_hours_" + key])
            checks.append({
                "id": "cost_units:" + cost["setting"] + ":" + cost["variant"] + ":" + key,
                "max_absolute_difference": difference,
                "within_saved_scalar_rounding_tolerance": difference <= 1e-10,
                "source": values["cost_source"],
            })
    return {
        "schema": "r03c-existing-paper-value-replay-v1",
        "scope": "Replay of existing saved summaries, not a new scientific analysis",
        "checks": checks, "checks_run": len(checks),
        "maximum_absolute_difference": max(row["max_absolute_difference"] for row in checks),
        "all_pass": all(row["within_saved_scalar_rounding_tolerance"] for row in checks),
        "tolerance": 1e-10,
        "original_paper_values_retained": True,
        "new_model_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", type=Path, required=True)
    parser.add_argument("--scalars", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = json.loads(args.values.read_text(encoding="utf-8"))
    scalars = json.loads(args.scalars.read_text(encoding="utf-8"))
    result = verify(values, scalars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("EXISTING_PAPER_VALUE_REPLAY", result["checks_run"], "MAX_ERROR", result["maximum_absolute_difference"])
    if not result["all_pass"]:
        raise SystemExit("Source summary replay differs; see preserved report")


if __name__ == "__main__":
    main()
