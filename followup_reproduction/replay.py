"""Standard-library replay of the fixed follow-up scalar evidence, not training."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import random
import struct
import sys

BLOCK_COUNTS = {"p5_main": 24, "p5_corpus": 8, "p5_model": 8}
DEVELOPMENT = ["p5_main-block001", "p5_main-block005", "p5_main-block021", "p5_main-block019"]
TIE_ORDER = ["raw_0.001", "raw_0.01", "raw_0.0001",
             "normalized_0.001", "normalized_0.01", "normalized_0.0001"]
ENDPOINT_IDS = [
    "E01_companion_error", "E01_selection", "E02_rank", "E02_regret",
    "E03_Q1", "E03_Q2", "E03_Q3", "E04_M1_difficulty", "E04_G20_M1",
    "E04_M2_difficulty", "E04_G20_M2",
]
REFERENCE_ORDER = ["difficulty", "G20", "G5_coherent", "G5_dispersed",
                   "R_half", "R_disj", "R_disj_next_block_reuse", "E02_plain_epoch"]
METRICS = ["cross_seed_spearman", "cross_seed_top_two_regret",
           "same_realization_spearman", "same_realization_mean_absolute_error"]
SETTING_NAMES = {"p5_main": "MultiEURLEX / Pythia (main)",
                 "p5_corpus": "IMDb / Pythia (exploratory)",
                 "p5_model": "MultiEURLEX / GPT-2 (exploratory)"}
ENDPOINT_NAMES = {
    "E01_companion_error": "Companion MAE: coherent minus dispersed",
    "E01_selection": "Selection regret: group of twenty minus dispersed five",
    "E02_rank": "Rank: shared reference minus LoGra",
    "E02_regret": "Regret: LoGra minus shared reference",
    "E03_Q1": "Rank: matched reference minus role-disjoint reference",
    "E03_Q2": "Rank: disjoint minus difficulty",
    "E03_Q3": "Within-context score-difference/support association",
    "E04_M1_difficulty": "Rank: MLP EK-FAC minus difficulty",
    "E04_G20_M1": "Rank: group of twenty minus MLP EK-FAC",
    "E04_M2_difficulty": "Rank: Adam-checkpoint approximation minus difficulty",
    "E04_G20_M2": "Rank: shared reference minus Adam-checkpoint approximation",
}
METHOD_NAMES = {
    "difficulty": "Difficulty",
    "G20": "Matched reference (group of twenty)",
    "G5_coherent": "Coherent group-of-five reference",
    "G5_dispersed": "Dispersed group-of-five reference",
    "R_half": "Half-background replacement reference",
    "R_disj": "Role-disjoint reference",
    "R_disj_next_block_reuse": "Fixed next-original-block reference reuse",
    "E02_plain_epoch": "Independent plain epoch-gradient diagnostic",
    "m1_mlp_ekfac": "MLP EK-FAC (primary)",
    "m1_mlp_ekfac_damping_0p1": "MLP EK-FAC, damping factor 0.1",
    "m1_mlp_ekfac_damping_10p0": "MLP EK-FAC, damping factor 10",
    "m1_block_linear_ekfac": "Block-linear EK-FAC (optional)",
    "m2_epoch_preconditioned": "Adam-checkpoint approximation (primary)",
    "m2_epoch_plain": "Epoch plain checkpoint (sensitivity)",
    "m2_epoch_less_style_cosine": "Epoch LESS-style cosine (sensitivity)",
    "m2_actual_step_preconditioned": "Actual-step preconditioned checkpoint (sensitivity)",
    "m2_actual_step_plain": "Actual-step plain checkpoint (sensitivity)",
    "m2_actual_step_less_style_cosine": "Actual-step LESS-style cosine (sensitivity)",
}


class ReplayError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReplayError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def nonfinite(value):
        raise ReplayError("Nonfinite JSON number: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=nonfinite)


def mean(values):
    return math.fsum(values) / len(values)


def stable_seed(*parts):
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31 - 1)


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result, start = [0.0] * len(values), 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            result[i] = (start + end - 1) / 2
        start = end
    return result


def correlation(left, right):
    a, b = ranks(left), ranks(right)
    a, b = [v - mean(a) for v in a], [v - mean(b) for v in b]
    denominator = math.sqrt(math.fsum(v*v for v in a) * math.fsum(v*v for v in b))
    return math.fsum(x*y for x, y in zip(a, b)) / denominator if denominator else None


def regret(prediction, target):
    require(len(prediction) == len(target) == 8, "Exactly eight ordered focals required")
    chosen = sorted(range(8), key=lambda i: (-prediction[i], i))[:2]
    best = sorted(range(8), key=lambda i: (-target[i], i))[:2]
    return mean([target[i] for i in best]) - mean([target[i] for i in chosen])


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap(cells, confidence, seed, replicates=20000):
    block_means = [mean(row) for row in cells]
    count = len(block_means)
    rng = random.Random(seed)
    stream = hashlib.sha256(canonical({"n_blocks": count, "seed": seed, "draws": replicates}))
    values = []
    for _ in range(replicates):
        draw = [rng.randrange(count) for _ in range(count)]
        stream.update(struct.pack(">" + "I" * count, *draw))
        values.append(mean([block_means[i] for i in draw]))
    tail = (1 - confidence) / 2
    return mean(block_means), [percentile(values, tail), percentile(values, 1-tail)], stream.hexdigest()


def vector(value, name):
    require(isinstance(value, list) and len(value) == 8
            and all(type(v) in (int, float) and math.isfinite(v) for v in value),
            "Missing/nonfinite/incorrect eight-focal vector: " + name)
    return value


def covered_fraction(row):
    counts = row["original_ALL_OUT_uncovered_count"]
    require(len(counts) == 8 and all(type(n) is int and 0 <= n <= 251 for n in counts),
            "Original ALL_OUT k4 uncovered counts invalid")
    return [1 - n / 255 for n in counts]


def block_ids(setting, experiment=None):
    values = [f"{setting}-block{n:03d}" for n in range(BLOCK_COUNTS[setting])]
    return [b for b in values if b not in DEVELOPMENT] if setting == "p5_main" and experiment == "E02" else values


def validate_inputs(protocol, data, costs):
    require(protocol["backgrounds"] == [0] and protocol["seeds"] == [0, 1, 2], "Frozen background/seed scope changed")
    require(protocol["family_denominator"] == 11 and protocol["alpha"] == 0.05
            and protocol["bootstrap_replicates"] == 20000 and protocol["root_seed"] == 202610051754,
            "Frozen inference family changed")
    require(protocol["main_blocks"] == 24 and protocol["transport_blocks"] == 8
            and protocol["focals_per_context"] == 8 and protocol["fixed_training_rows"] == 768
            and protocol["shifted_target_count_including_EOS"] == 255, "Frozen populations/budgets changed")
    require(protocol["development_blocks_in_frozen_order"] == DEVELOPMENT
            and protocol["E02_tie_order"] == TIE_ORDER, "Frozen original development split/tie order changed")
    require([r["id"] for r in protocol["endpoints"]] == ENDPOINT_IDS, "Missing/extra/reordered endpoints")
    require(protocol["E02_selection"]["status"] == "blocked_development"
            and protocol["E02_selection"]["selected_configuration"] is None,
            "The accepted denied development lock cannot be replaced")
    require(protocol["availability"] == {"E02": "unavailable_qualification", "E04": "unavailable_qualification"},
            "Failed qualifications cannot become available methods")
    require(len(protocol["E04_variant_order"]) == len(set(protocol["E04_variant_order"])) == 20,
            "All twenty frozen E04 variants must be retained")
    index = {r["context_id"]: r for r in data["contexts"]}
    expected = {f"{b}-b0-s{s}" for setting in BLOCK_COUNTS for b in block_ids(setting) for s in range(3)}
    require(len(data["contexts"]) == len(index) == data["context_count"] == 120 and set(index) == expected,
            "Missing/duplicate context or altered full block population; no complete cases")
    for cid, row in index.items():
        setting, bid, seed = row["setting"], row["block_id"], row["seed"]
        require(setting in BLOCK_COUNTS and row["background"] == 0 and type(seed) is int and seed in range(3)
                and cid == f"{bid}-b0-s{seed}" and bid in block_ids(setting), "Context/seed/background identity drift")
        require(row["focal_order"] == list(range(8)), "Frozen focal tie order changed")
        positions = row["focal_member_positions"]
        require(len(positions) == len(set(positions)) == 8
                and all(type(v) is int and v in range(20) for v in positions), "Focal member positions invalid")
        for key in ("original_individual_effect", "original_G20_score", "difficulty_nll", "all_in_nll",
                    "half_reference_nll", "disjoint_reference_nll"):
            vector(row[key], key)
        covered_fraction(row)
        order = block_ids(setting)
        n = order.index(bid)
        following = f"{order[(n+1) % len(order)]}-b0-s{seed}"
        previous = f"{order[(n-1) % len(order)]}-b0-s{seed}"
        require(row["fixed_next_block_donor"] == following, "Reuse is next ORIGINAL block, same seed/background")
        readout = row["disjoint_supplemental_readout"]
        require(readout["recipient_context_id"] == previous and readout["recipient_focal_order"] == list(range(8)),
                "Donor supplemental readout belongs to the wrong recipient/focal order")
        vector(readout["recipient_nll"], "donor supplemental NLL")
        if setting == "p5_model":
            require(row["G5_partitions"] == {} and row["independent_plain_epoch_score"] is None,
                    "Unscheduled E01/E02 model transport cannot be added")
        else:
            vector(row["independent_plain_epoch_score"], "qualified independent plain score")
            require(set(row["G5_partitions"]) == {"G5_coherent", "G5_dispersed"}, "Missing G5 policy")
            for parts in row["G5_partitions"].values():
                require(len(parts) == 4 and [p["group_index"] for p in parts] == list(range(4)),
                        "All four group reference fits required")
                require(sorted(p for part in parts for p in part["member_positions"]) == list(range(20)),
                        "G5 must partition all twenty members")
                for part in parts:
                    expected_indices = [i for i, pos in enumerate(positions) if pos in part["member_positions"]]
                    require(len(part["member_positions"]) == 5 and len(expected_indices) == 2
                            and part["assigned_focal_indices"] == expected_indices
                            and len(part["assigned_focal_nll"]) == 2
                            and all(type(x) in (int, float) and math.isfinite(x)
                                    for x in part["assigned_focal_nll"]), "G5 requires exactly two assigned focals per five")
        for other_seed in range(3):
            peer = index[f"{bid}-b0-s{other_seed}"]
            require(peer["focal_member_positions"] == positions
                    and peer["original_ALL_OUT_uncovered_count"] == row["original_ALL_OUT_uncovered_count"],
                    "Cross-seed focal/support identity drift")
            if setting != "p5_model":
                for policy, parts in row["G5_partitions"].items():
                    require([p["member_positions"] for p in peer["G5_partitions"][policy]]
                            == [p["member_positions"] for p in parts], "G5 assignment cannot depend on optimizer seed")
    require(costs["timing_issue"] == "UNRESOLVED_SHARED_PLAIN_FINALIZER_OVERLAP"
            and costs["accepted_scientific_job_hours"] is None and costs["campaign_elapsed_hours"] is None,
            "Overlapping science time or unverified synchronized elapsed cannot be repaired")
    require(costs["E02_failed_qualifications"] == 6 and costs["E04_failed_qualifications"] == 10,
            "Native qualification failures changed")
    require(costs["actions"] == {"accepted": 1300, "active": 0, "blocked": 216, "failed": 16,
                                 "interrupted": 0, "planned": 1532, "unstarted": 0},
            "Accepted action population changed")
    for key, count in (("science_fits", 1128), ("qualification_fits", 60)):
        require(costs[key]["accepted"] == costs[key]["planned"] == count
                and all(costs[key][state] == 0 for state in
                        ("active", "blocked", "failed", "interrupted", "unstarted")),
                "Accepted fit population changed")
    return index


def target_other_two(index, cid):
    row = index[cid]
    peers = [index[f"{row['block_id']}-b0-s{s}"]["original_individual_effect"]
             for s in range(3) if s != row["seed"]]
    return [mean([peers[0][i], peers[1][i]]) for i in range(8)]


def scores(row, index):
    baseline = row["all_in_nll"]
    subtract = lambda values: [x-y for x, y in zip(values, baseline)]
    donor = index[row["fixed_next_block_donor"]]["disjoint_supplemental_readout"]
    require(donor["recipient_context_id"] == row["context_id"], "Incorrect fixed donor readout")
    result = {"G20": row["original_G20_score"], "difficulty": row["difficulty_nll"],
              "R_half": subtract(row["half_reference_nll"]),
              "R_disj": subtract(row["disjoint_reference_nll"]),
              "R_disj_next_block_reuse": subtract(donor["recipient_nll"])}
    for policy, parts in row["G5_partitions"].items():
        values = [None] * 8
        require(len(parts) == 4, "Missing mandatory G5 reference")
        for part in parts:
            for i, loss in zip(part["assigned_focal_indices"], part["assigned_focal_nll"]):
                require(values[i] is None, "Repeated G5 focal assignment")
                values[i] = loss - baseline[i]
        result[policy] = vector(values, policy)
    if row["independent_plain_epoch_score"] is not None:
        result["E02_plain_epoch"] = row["independent_plain_epoch_score"]
    return result


def context_endpoint(endpoint_id, row, predictions, target):
    if endpoint_id.startswith(("E02_", "E04_")):
        return None
    if endpoint_id == "E01_companion_error":
        same = row["original_individual_effect"]
        return (mean([abs(x-y) for x, y in zip(predictions["G5_coherent"], same)])
                - mean([abs(x-y) for x, y in zip(predictions["G5_dispersed"], same)]))
    if endpoint_id == "E01_selection":
        return regret(predictions["G20"], target) - regret(predictions["G5_dispersed"], target)
    if endpoint_id == "E03_Q3":
        return correlation([d-g for d, g in zip(predictions["R_disj"], predictions["G20"])],
                           covered_fraction(row))
    left, right = ("G20", "R_disj") if endpoint_id == "E03_Q1" else ("R_disj", "difficulty")
    a, b = correlation(predictions[left], target), correlation(predictions[right], target)
    return None if a is None or b is None else a-b


def endpoint_record(test, setting, index, predictions, targets):
    bids = block_ids(setting, test["experiment"])
    confidence = 1 - .05 / 11 if setting == "p5_main" else .95
    seed = stable_seed(202610051754, test["id"])
    cells, blocks = [], []
    for bid in bids:
        values = [context_endpoint(test["id"], index[f"{bid}-b0-s{s}"],
                                   predictions[f"{bid}-b0-s{s}"], targets[f"{bid}-b0-s{s}"])
                  for s in range(3)]
        cells.append(values)
        blocks.append({"block_id": bid, "seed_values": values,
                       "mean": mean(values) if all(x is not None for x in values) else None})
    missing = any(x is None for row in cells for x in row)
    result = {"id": test["id"], "setting": setting,
              "status": ("blocked_incomplete" if test["experiment"] in ("E02", "E04") else "blocked_undefined")
                        if missing else "available",
              "estimate": None, "interval": None, "confidence": confidence,
              "family_denominator": 11 if setting == "p5_main" else None,
              "population": {"block_ids": bids, "blocks_expected": len(bids), "contexts_expected": 3*len(bids),
                             "backgrounds": [0], "seeds": [0, 1, 2], "focals_per_context": 8,
                             "inference_unit": "original block", "weighting": "equal seeds within block, equal blocks"},
              "block_values": None, "direction_resolved": None, "desired_direction_resolved": None,
              "bootstrap": {"root_seed": 202610051754, "endpoint_seed": seed, "replicates": 20000,
                            "block_order_sha256": digest(bids), "input_cells_sha256": digest(cells),
                            "index_stream_sha256": None}}
    if not missing:
        estimate, interval, stream = bootstrap(cells, confidence, seed)
        direction = "greater" if interval[0] > 0 else "less" if interval[1] < 0 else "unresolved"
        result.update(estimate=estimate, interval=interval, block_values=blocks,
                      direction_resolved=direction,
                      desired_direction_resolved=direction != "unresolved" if test["desired_direction"] == "either"
                                                else direction == test["desired_direction"])
        result["bootstrap"]["index_stream_sha256"] = stream
    return result


def sign_counts(values):
    return {"positive": sum(v > 0 for v in values), "negative": sum(v < 0 for v in values),
            "zero": sum(v == 0 for v in values)}


def method_diagnostics(protocol, index, predictions, targets):
    e02 = ["E02_" + key for key in TIE_ORDER]
    scopes = []
    for setting in BLOCK_COUNTS:
        ids = sorted(cid for cid, row in index.items() if row["setting"] == setting)
        refs = [m for m in REFERENCE_ORDER if m in predictions[ids[0]]]
        scopes.append((setting + "_all_blocks_descriptive", setting, ids,
                       refs + (e02 if setting == "p5_corpus" else []) + protocol["E04_variant_order"]))
    evaluation = [f"{bid}-b0-s{s}" for bid in block_ids("p5_main", "E02") for s in range(3)]
    scopes.append(("E02_frozen_main_evaluation20_descriptive", "p5_main", evaluation,
                   ["G20", "E02_plain_epoch"] + e02))
    summaries, block_rows, context_rows = [], [], []
    for scope, setting, ids, methods in scopes:
        bids = sorted({index[cid]["block_id"] for cid in ids})
        for order, method in enumerate(methods):
            missing = [cid for cid in ids if method not in predictions[cid]]
            summary = {"scope": scope, "setting": setting, "method": method, "frozen_display_order": order,
                       "status": "blocked_qualification" if missing else "available_descriptive",
                       "blocks_expected": len(bids), "contexts_expected": len(ids),
                       "unavailable_context_count": len(missing),
                       "primary_E04_variant": method in ("m1_mlp_ekfac", "m2_epoch_preconditioned")}
            if missing:
                require(len(missing) == len(ids), "Partial method coverage: never use complete cases")
                summary.update({metric: None for metric in METRICS})
                summaries.append(summary)
                continue
            cells = []
            for cid in ids:
                row, prediction = index[cid], predictions[cid][method]
                same = row["original_individual_effect"]
                cell = {"scope": scope, "setting": setting, "method": method, "context_id": cid,
                        "block_id": row["block_id"], "background": 0, "seed": row["seed"],
                        "cross_seed_spearman": correlation(prediction, targets[cid]),
                        "cross_seed_top_two_regret": regret(prediction, targets[cid]),
                        "same_realization_spearman": correlation(prediction, same),
                        "same_realization_mean_absolute_error":
                            mean([abs(x-y) for x, y in zip(prediction, same)])
                            if method in ("G20", "G5_coherent", "G5_dispersed", "R_half", "R_disj",
                                          "R_disj_next_block_reuse") else None}
                cells.append(cell)
                context_rows.append(cell)
            blocks = []
            for bid in bids:
                selected = sorted((r for r in cells if r["block_id"] == bid), key=lambda r: r["seed"])
                require([r["seed"] for r in selected] == [0, 1, 2], "Missing seed in descriptive population")
                block = {"scope": scope, "setting": setting, "method": method, "block_id": bid}
                for metric in METRICS:
                    values = [r[metric] for r in selected]
                    block[metric] = mean(values) if all(v is not None for v in values) else None
                blocks.append(block)
                block_rows.append(block)
            for metric in METRICS:
                values = [r[metric] for r in blocks]
                summary[metric] = mean(values) if all(v is not None for v in values) else None
            if summary["cross_seed_spearman"] is None or summary["same_realization_spearman"] is None:
                summary["status"] = "blocked_undefined_rank_no_complete_case"
            summaries.append(summary)
    return summaries, block_rows, context_rows


def stability_records(endpoints):
    result = []
    for record in endpoints:
        if record["status"] != "available":
            continue
        blocks = record["block_values"]
        values = [b["mean"] for b in blocks]
        loo = [mean(values[:n] + values[n+1:]) for n in range(len(values))]
        result.append({"setting": record["setting"], "endpoint_id": record["id"],
                       "block_count": len(values), "estimate": record["estimate"],
                       "block_sign_counts": sign_counts(values),
                       "context_sign_counts": sign_counts([x for b in blocks for x in b["seed_values"]]),
                       "seed_means": {str(s): mean([b["seed_values"][s] for b in blocks]) for s in range(3)},
                       "block_minimum": min(values), "block_maximum": max(values),
                       "leave_one_block_out_mean_minimum": min(loo),
                       "leave_one_block_out_mean_maximum": max(loo)})
    return result


def ladder_records(index, block_rows, context_rows):
    lookup = {(r["scope"], r["block_id"], r["method"]): r for r in block_rows}
    contexts = {(r["scope"], r["context_id"], r["method"]): r for r in context_rows}
    result, details = [], []
    for setting in BLOCK_COUNTS:
        scope = setting + "_all_blocks_descriptive"
        ids = sorted(cid for cid, r in index.items() if r["setting"] == setting)
        bids = block_ids(setting)
        context_monotone, undefined = 0, 0
        for cid in ids:
            values = [contexts[(scope, cid, m)]["cross_seed_spearman"] for m in ("G20", "R_half", "R_disj")]
            if any(x is None for x in values):
                undefined += 1
            else:
                context_monotone += values[0] >= values[1] >= values[2]
        local = []
        for bid in bids:
            g20, half, disjoint, reuse = [lookup[(scope, bid, m)]["cross_seed_spearman"]
                                         for m in ("G20", "R_half", "R_disj", "R_disj_next_block_reuse")]
            diff = lambda x, y: None if x is None or y is None else x-y
            row = {"setting": setting, "block_id": bid, "G20_rank": g20, "half_rank": half,
                   "disjoint_rank": disjoint, "fixed_next_block_reuse_rank": reuse,
                   "G20_minus_half": diff(g20, half), "half_minus_disjoint": diff(half, disjoint),
                   "disjoint_minus_fixed_reuse": diff(disjoint, reuse),
                   "monotone_G20_ge_half_ge_disjoint":
                       g20 >= half >= disjoint if all(v is not None for v in (g20, half, disjoint)) else None}
            local.append(row)
            details.append(row)
        result.append({"setting": setting, "blocks": len(bids), "contexts": len(ids),
                       "monotone_block_count": sum(r["monotone_G20_ge_half_ge_disjoint"] for r in local)
                           if all(r["monotone_G20_ge_half_ge_disjoint"] is not None for r in local) else None,
                       "monotone_context_count": context_monotone if not undefined else None,
                       "undefined_context_frontiers": undefined,
                       "descriptive_mean_rank_differences": {
                           key: mean([r[key] for r in local]) if all(r[key] is not None for r in local) else None
                           for key in ("G20_minus_half", "half_minus_disjoint", "disjoint_minus_fixed_reuse")},
                       "no_new_significance_tests": True})
    return result, details


def feasibility_diagnostics(source):
    indexed = {(r["setting"], r["block_id"], r["background"]): r
               for r in source["assignment_objectives"]}
    expected = {(setting, bid, background) for setting in ("p5_main", "p5_corpus")
                for bid in block_ids(setting) for background in (0, 1)}
    require(len(indexed) == len(source["assignment_objectives"]) == 64 and set(indexed) == expected,
            "Input-only feasibility requires both original backgrounds, not a guessed projection")
    for row in indexed.values():
        require(row["denominator"] == 2040
                and type(row["coherent_support_numerator"]) is int
                and type(row["dispersed_support_numerator"]) is int
                and row["coherent_support_numerator"] >= row["dispersed_support_numerator"],
                "Frozen composition objective is not an eight-query support fraction")
    gate = source["original_gate"]
    require(gate["threshold"] == .02 and gate["required_main_blocks"] == 12
            and gate["total_main_blocks"] == gate["passing_main_blocks"] == 24
            and gate["IMDb_was_gated"] is False and gate["outcomes_used"] is False
            and gate["gate_recomputed_for_amendment"] is False, "Original input-only gate changed")
    blocks, ranges = [], []
    for scope in ("original_both_background_feasibility", "executed_background0_assignment"):
        for setting in ("p5_main", "p5_corpus"):
            values = []
            for bid in block_ids(setting):
                selected = [indexed[(setting, bid, b)] for b in ((0, 1) if scope.startswith("original") else (0,))]
                numerator = sum(r["coherent_support_numerator"] - r["dispersed_support_numerator"] for r in selected)
                denominator = len(selected) * 2040
                contrast = numerator / denominator
                values.append(contrast)
                blocks.append({"scope": scope, "setting": setting, "block_id": bid,
                               "background_count": len(selected), "contrast_numerator": numerator,
                               "contrast_denominator": denominator, "net_support_fraction_contrast": contrast,
                               "original_gate_applies": scope.startswith("original") and setting == "p5_main"})
            ranges.append({"scope": scope, "setting": setting, "blocks": len(values),
                           "minimum_net_support_fraction_contrast": min(values),
                           "maximum_net_support_fraction_contrast": max(values),
                           "original_gate_applies": scope.startswith("original") and setting == "p5_main",
                           "unit": "net companion k4 support fraction; not outcome loss"})
    main_gate_rows = [r for r in blocks if r["original_gate_applies"]]
    require(sum(r["contrast_numerator"] * 100 >= 2 * r["contrast_denominator"] for r in main_gate_rows) == 24,
            "Original main-only gate does not reproduce")
    changes = source["Q1_reference_training_data_change"]["checked_contexts"]
    require(len(changes) == len({r["context_id"] for r in changes}) == 120
            and all(r["changed_background_slots"] == 748 and r["changed_matched_control_slots"] == 20
                    and r["changed_total_slots"] == 768 for r in changes),
            "Q1 must disclose joint background and matched-control changes")
    return {"ranges": ranges, "block_values": blocks, "new_gate_or_outcome_test": False}


def replay(protocol, data, costs, feasibility):
    index = validate_inputs(protocol, data, costs)
    predictions = {cid: scores(row, index) for cid, row in sorted(index.items())}
    targets = {cid: target_other_two(index, cid) for cid in sorted(index)}
    endpoints = [endpoint_record(test, setting, index, predictions, targets)
                 for setting in BLOCK_COUNTS for test in protocol["endpoints"]
                 if setting != "p5_model" or test["experiment"] in ("E03", "E04")]
    methods, blocks, contexts = method_diagnostics(protocol, index, predictions, targets)
    ladder, ladder_blocks = ladder_records(index, blocks, contexts)
    damping = {"0.001": "1e-3", "0.01": "1e-2", "0.0001": "1e-4"}
    configurations = []
    for setting in ("p5_main", "p5_corpus"):
        for key in TIE_ORDER:
            form, amount = key.split("_")
            configurations.append({
                "setting": setting, "protocol_key": key, "worker_score_key": form + "-" + damping[amount],
                "status": "blocked_selection", "selected_by_development": None, "target": "original_cross_seed",
                "scope": "all six retained; descriptive evaluation points, no extra tests",
                "block_ids": block_ids(setting, "E02"), "mean_spearman": None, "mean_top_two_regret": None,
                "block_values": None})
    return {"schema": "anonymous-followup-replayed-results-v1", "endpoints": endpoints,
            "method_summaries": methods, "method_block_metrics": blocks, "method_context_metrics": contexts,
            "endpoint_stability": stability_records(endpoints), "reference_ladder": ladder,
            "reference_ladder_blocks": ladder_blocks, "E02_configuration_summaries": configurations,
            "costs": costs, "input_feasibility": feasibility_diagnostics(feasibility),
            "new_neural_calls": 0, "new_significance_family": False}


def compare_expected(result, expected):
    errors = []
    strict_bindings = 0
    def numeric(actual, wanted, name):
        require((actual is None) == (wanted is None), "Missingness changed: " + name)
        if actual is not None:
            delta = abs(float(actual) - float(wanted))
            require(delta <= 1e-14, "Numerical mismatch: " + name)
            errors.append(delta)
    require(len(result["endpoints"]) == len(expected["endpoints"]) == 29, "Endpoint denominator changed")
    for actual, wanted in zip(result["endpoints"], expected["endpoints"]):
        for key in ("id", "setting", "status", "confidence", "family_denominator", "population",
                    "direction_resolved", "desired_direction_resolved"):
            require(actual[key] == wanted[key], "Endpoint decision/identity mismatch: " + key)
        for key in ("root_seed", "endpoint_seed", "replicates", "block_order_sha256",
                    "input_cells_sha256", "index_stream_sha256"):
            require(actual["bootstrap"][key] == wanted["bootstrap"][key], "Frozen stream/cell binding differs: " + key)
            strict_bindings += 1
        numeric(actual["estimate"], wanted["estimate"], actual["id"])
        require((actual["interval"] is None) == (wanted["interval"] is None), "Interval availability changed")
        if actual["interval"] is not None:
            for x, y in zip(actual["interval"], wanted["interval"]):
                numeric(x, y, actual["id"] + " interval")
            for a, b in zip(actual["block_values"], wanted["block_values"]):
                require(a["block_id"] == b["block_id"] and a["seed_values"] == b["seed_values"],
                        "Complete ordered context values differ")
                numeric(a["mean"], b["mean"], "block mean")
    require(len(result["method_summaries"]) == len(expected["method_summaries"]) == 95, "Method menu changed")
    for actual, wanted in zip(result["method_summaries"], expected["method_summaries"]):
        for key in ("scope", "setting", "method", "status"):
            require(actual[key] == wanted[key], "Method scope/availability changed")
        for key in ("blocks_expected", "contexts_expected", "unavailable_context_count", "frozen_display_order"):
            require(actual[key] == int(wanted[key]), "Method denominator/order changed")
        for key in METRICS:
            numeric(actual[key], None if wanted[key] == "" else float(wanted[key]), key)
    require(result["E02_configuration_summaries"] == expected["E02_configuration_summaries"],
            "Denied all-configuration records changed")
    for key in ("reference_ladder", "endpoint_stability"):
        require(len(result[key]) == len(expected[key]), "Descriptive population changed")
        for actual, wanted in zip(result[key], expected[key]):
            for field, value in actual.items():
                require(value == wanted[field], "Descriptive stability/ladder mismatch: " + field)
    return {"status": "pass", "endpoint_records": 29, "available_intervals": 13,
            "exact_stream_and_cell_binding_checks": strict_bindings,
            "numerical_checks": len(errors), "maximum_absolute_numeric_difference": max(errors, default=0),
            "numeric_tolerance": 1e-14,
            "tolerance_scope": "Only floating aggregation; context-cell hashes, streams, populations, "
                               "availability, directions and descriptive decisions must match exactly."}


def method_name(key):
    if key.startswith("E02_") and key != "E02_plain_epoch":
        return "LoGra, " + key[4:].replace("_", ", damping ")
    if key.endswith("_removal_only"):
        return METHOD_NAMES[key[:-13]] + ", removal-only sensitivity"
    return METHOD_NAMES[key]


def csv_bytes(rows, columns=None):
    stream = io.StringIO(newline="")
    fields = columns or list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def tex_escape(value):
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def tex_table(headers, rows):
    lines = ["% Generated numerical follow-up table. Do not edit numbers by hand.",
             r"\begin{tabular}{@{}" + "l" * len(headers) + r"@{}}", r"\hline",
             " & ".join(tex_escape(v) for v in headers) + r" \\", r"\hline"]
    lines.extend(" & ".join(tex_escape(v) for v in row) + r" \\" for row in rows)
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines).encode("utf-8")


def display(value):
    return "Unavailable" if value is None else format(value, ".6f")


def render_outputs(result, protocol):
    outputs = {"followup_replayed.json": json.dumps(result, indent=2, sort_keys=True,
                                                    allow_nan=False, ensure_ascii=False).encode() + b"\n"}
    tests = {r["id"]: r for r in protocol["endpoints"]}
    all_csv = []
    for setting, suffix in (("p5_main", "main"), ("p5_corpus", "imdb"), ("p5_model", "gpt2")):
        selected = [r for r in result["endpoints"] if r["setting"] == setting]
        rows = []
        for r in selected:
            interval = r["interval"]
            rows.append({"setting": setting, "setting_name": SETTING_NAMES[setting],
                         "endpoint_id": r["id"], "contrast": ENDPOINT_NAMES[r["id"]],
                         "status": r["status"], "blocks": r["population"]["blocks_expected"],
                         "family_denominator": r["family_denominator"], "confidence": r["confidence"],
                         "estimate": r["estimate"], "ci_low": interval[0] if interval else None,
                         "ci_high": interval[1] if interval else None, "unit": tests[r["id"]]["unit"],
                         "direction_resolved": r["direction_resolved"]})
        outputs[f"followup_{suffix}_endpoints.csv"] = csv_bytes(rows)
        outputs[f"followup_{suffix}_endpoints.tex"] = tex_table(
            ["Contrast", "Estimate", "Interval", "Unit"],
            [[r["contrast"], display(r["estimate"]),
              f'[{display(r["ci_low"])}, {display(r["ci_high"])}]' if r["estimate"] is not None else "Unavailable",
              "nats/token" if r["unit"] == "nats/token" else "Spearman"]
             for r in rows])
        all_csv.extend(rows)
    outputs["followup_endpoints.csv"] = csv_bytes(all_csv)
    short_names = {
        "E01_companion_error": "Coherent minus dispersed error",
        "E01_selection": "Shared minus dispersed regret",
        "E03_Q1": "Matched minus disjoint rank",
        "E03_Q2": "Disjoint minus difficulty rank",
        "E03_Q3": "Score-shift/support association",
    }
    outputs["followup_main_summary.tex"] = tex_table(
        ["Contrast", "Unit", "Estimate", "Adjusted interval"],
        [[short_names[r["endpoint_id"]],
          "nats/token" if r["unit"] == "nats/token" else "Spearman",
          format(r["estimate"], ".4f"),
          f'[{r["ci_low"]:.4f}, {r["ci_high"]:.4f}]']
         for r in all_csv if r["setting"] == "p5_main" and r["status"] == "available"])
    methods = [{**r, "method_name": method_name(r["method"])} for r in result["method_summaries"]]
    outputs["followup_methods.csv"] = csv_bytes(methods)
    for setting, suffix in (("p5_main", "main"), ("p5_corpus", "imdb"), ("p5_model", "gpt2")):
        available = [r for r in methods if r["scope"] == setting + "_all_blocks_descriptive"
                     and r["status"] == "available_descriptive"]
        outputs[f"followup_{suffix}_reference_methods.tex"] = tex_table(
            ["Method", "Cross-seed rank", "Regret (nats/token)", "Same-realization MAE"],
            [[r["method_name"], display(r["cross_seed_spearman"]), display(r["cross_seed_top_two_regret"]),
              display(r["same_realization_mean_absolute_error"])] for r in available])
    outputs["followup_method_blocks.csv"] = csv_bytes(result["method_block_metrics"])
    outputs["followup_method_contexts.csv"] = csv_bytes(result["method_context_metrics"])
    unavailable = [r for r in methods if r["status"] == "blocked_qualification"]
    outputs["followup_unavailable_methods.tex"] = tex_table(
        ["Scope", "Method", "Status"],
        [["Main evaluation (20 blocks)" if r["scope"].startswith("E02_frozen") else SETTING_NAMES[r["setting"]],
          r["method_name"], "Unavailable: qualification"] for r in unavailable])
    configs = [{**{k: v for k, v in r.items() if k not in ("block_ids", "block_values")},
                "blocks": len(r["block_ids"])} for r in result["E02_configuration_summaries"]]
    outputs["followup_all_projected_configurations.csv"] = csv_bytes(configs)
    outputs["followup_all_projected_configurations.tex"] = tex_table(
        ["Setting", "Configuration", "Blocks", "Rank / regret"],
        [[SETTING_NAMES[r["setting"]], method_name("E02_" + r["protocol_key"]), r["blocks"],
          "Unavailable: development denied"] for r in configs])
    stability = []
    for r in result["endpoint_stability"]:
        stability.append({"setting": r["setting"], "endpoint_id": r["endpoint_id"],
                          "contrast": ENDPOINT_NAMES[r["endpoint_id"]], "blocks": r["block_count"],
                          "positive_blocks": r["block_sign_counts"]["positive"],
                          "negative_blocks": r["block_sign_counts"]["negative"],
                          "zero_blocks": r["block_sign_counts"]["zero"],
                          "seed0_mean": r["seed_means"]["0"], "seed1_mean": r["seed_means"]["1"],
                          "seed2_mean": r["seed_means"]["2"],
                          "leave_one_out_mean_minimum": r["leave_one_block_out_mean_minimum"],
                          "leave_one_out_mean_maximum": r["leave_one_block_out_mean_maximum"]})
    outputs["followup_stability.csv"] = csv_bytes(stability)
    outputs["followup_stability.tex"] = tex_table(
        ["Setting / contrast", "Blocks (+/-/0)", "Seed 0", "Seed 1", "Seed 2"],
        [[SETTING_NAMES[r["setting"]] + ": " + r["contrast"],
          f'{r["positive_blocks"]}/{r["negative_blocks"]}/{r["zero_blocks"]}',
          display(r["seed0_mean"]), display(r["seed1_mean"]), display(r["seed2_mean"])] for r in stability])
    outputs["followup_reference_ladder_blocks.csv"] = csv_bytes(result["reference_ladder_blocks"])
    ladder = [{**{k: v for k, v in r.items() if k != "descriptive_mean_rank_differences"},
               **r["descriptive_mean_rank_differences"]} for r in result["reference_ladder"]]
    outputs["followup_reference_ladder.csv"] = csv_bytes(ladder)
    outputs["followup_reference_ladder.tex"] = tex_table(
        ["Setting", "Monotone blocks", "Matched-half rank", "Half-disjoint rank", "Disjoint-reuse rank"],
        [[SETTING_NAMES[r["setting"]], f'{r["monotone_block_count"]}/{r["blocks"]}',
          display(r["G20_minus_half"]), display(r["half_minus_disjoint"]),
          display(r["disjoint_minus_fixed_reuse"])] for r in ladder])
    outputs["followup_cost_categories.csv"] = csv_bytes(result["costs"]["accepted_categories"])
    cost_rows = [
        ["Known exclusive science subtotal", result["costs"]["known_exclusive_scientific_job_hours_subtotal"], "job-hours"],
        ["Accepted qualification (separate)", result["costs"]["accepted_qualification_job_hours"], "job-hours"],
        ["Raw reported science sum (overlaps)", result["costs"]["reported_scientific_seconds"], "seconds"],
        ["Failed work (separate)", result["costs"]["failed_observed_seconds"], "seconds"],
        ["Exact exclusive-science total", None, "job-hours"],
        ["Synchronized campaign elapsed", None, "hours"],
        ["Observed six-worker journal window", result["costs"]["observed_six_worker_journal_window_seconds"], "seconds"],
    ]
    outputs["followup_costs.tex"] = tex_table(["Quantity", "Value", "Unit"],
                                           [[name, display(value), unit] for name, value, unit in cost_rows])
    outputs["followup_costs.csv"] = csv_bytes([dict(quantity=n, value=v, unit=u) for n, v, u in cost_rows])
    costs = result["costs"]
    outputs["followup_qualification.csv"] = csv_bytes([
        {"quantity": "Planned actions", "count": costs["actions"]["planned"]},
        {"quantity": "Accepted actions", "count": costs["actions"]["accepted"]},
        {"quantity": "Failed qualifications", "count": costs["actions"]["failed"]},
        {"quantity": "Blocked scientific actions", "count": costs["actions"]["blocked"]},
        {"quantity": "Accepted science fits", "count": costs["science_fits"]["accepted"]},
        {"quantity": "Accepted qualification fits", "count": costs["qualification_fits"]["accepted"]},
        {"quantity": "Failed LoGra qualifications", "count": costs["E02_failed_qualifications"]},
        {"quantity": "Failed EK-FAC/checkpoint qualifications", "count": costs["E04_failed_qualifications"]},
    ])
    outputs["followup_reference_policy_costs.csv"] = csv_bytes(result["costs"]["reference_policy_costs"])
    feasibility = result["input_feasibility"]
    outputs["followup_input_feasibility.csv"] = csv_bytes(feasibility["ranges"])
    outputs["followup_input_feasibility_blocks.csv"] = csv_bytes(feasibility["block_values"])
    outputs["followup_input_feasibility.tex"] = tex_table(
        ["Input-only scope", "Setting", "Support-contrast range", "Original gate"],
        [["Original backgrounds 0 and 1" if r["scope"].startswith("original") else "Executed background 0",
          SETTING_NAMES[r["setting"]],
          "[" + display(r["minimum_net_support_fraction_contrast"]) + ", "
          + display(r["maximum_net_support_fraction_contrast"]) + "]",
          "Main-only gate" if r["original_gate_applies"] else "Not gated"]
         for r in feasibility["ranges"]])
    return outputs


def safe_relative(root, name):
    path = PurePosixPath(name)
    require(bool(path.parts) and not path.is_absolute() and "\\" not in name and ":" not in name
            and name == path.as_posix() and all(p not in ("..", ".") for p in path.parts),
            "Manifest requires safe relative paths")
    return root.joinpath(*path.parts)


def verify_manifest(root, include_generated=True):
    path = root / "MANIFEST.json"
    require(path.is_file(), "MANIFEST.json is required for verification")
    checksum = (root / "MANIFEST.sha256").read_text(encoding="ascii").strip()
    require(file_hash(path) == checksum, "Manifest checksum changed")
    manifest = strict_json(path)
    actual_names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    allowed = set(manifest["files"]) | {"MANIFEST.json", "MANIFEST.sha256"}
    require(actual_names <= allowed, "Untracked files are not part of the anonymous artifact")
    for name, record in manifest["files"].items():
        if not include_generated and name.startswith("generated/"):
            continue
        source = safe_relative(root, name)
        require(source.is_file() and source.stat().st_size == record["bytes"]
                and file_hash(source) == record["sha256"], "Artifact bytes differ: " + name)
    return manifest


def load_inputs(root):
    values = [strict_json(root / "inputs" / name) for name in
              ("PROTOCOL.json", "CONTEXTS.json", "QUALIFICATION_COSTS.json", "EXPECTED.json", "INPUT_FEASIBILITY.json")]
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "verify"))
    parser.add_argument("--output", default="generated")
    args = parser.parse_args(argv)
    root = Path(__file__).absolute().parent
    if args.command == "verify" or (root / "MANIFEST.json").exists():
        verify_manifest(root, include_generated=args.command == "verify")
    protocol, data, costs, expected, feasibility = load_inputs(root)
    result = replay(protocol, data, costs, feasibility)
    checked = compare_expected(result, expected)
    outputs = render_outputs(result, protocol)
    destination = safe_relative(root, args.output)
    if args.command == "generate":
        destination.mkdir(parents=True, exist_ok=True)
        require({p.name for p in destination.iterdir() if p.is_file()} <= set(outputs), "Unexpected output files")
        for name, content in outputs.items():
            (destination / name).write_bytes(content)
    else:
        require({p.name for p in destination.iterdir() if p.is_file()} == set(outputs), "Generated inventory differs")
        for name, content in outputs.items():
            require((destination / name).read_bytes() == content, "Generated bytes do not reproduce: " + name)
    print(json.dumps({"status": "pass", "command": args.command, "generated_files": len(outputs),
                      **checked}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReplayError as error:
        print("Replay rejected: " + str(error), file=sys.stderr)
        raise SystemExit(2)
