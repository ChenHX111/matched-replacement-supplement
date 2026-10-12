"""E01 far-control analysis: a frozen-law mirror of the original P5 rank/regret analysis.

DRAFT for coordinator review.  Specification: ``ANALYSIS_SPEC_DRAFT.md`` (same folder).

Scope and guarantees
--------------------
* No training, inference, tokenization, network access, GPU or neural import.
  Standard library only.  numpy is not needed for the exact mirror, and pandas is not
  installed in the project PYTHON (checked 10 Oct 2026); nothing was installed.
* Reads only the JSON inputs named on the command line and writes exactly one *new* output
  file (``open(..., "x")``); it never overwrites.
* The statistics mirror, operation for operation, the sealed main-study analysis
  (``approved_main_20261002/package/lm_support/statistics.py`` and ``p5_analysis.py``):
  average-rank Spearman with undefined constant-rank vectors, top-two-of-eight regret with
  frozen focal-vector tie order, the same-background other-two-seed target, equal context
  weights inside a block, equal block weights, and a Python ``random.Random`` whole-block
  percentile bootstrap with linear-interpolated quantiles and SHA-256 ``stable_seed``
  endpoint seeds.  ``NEAR_REPLAY_REPORT.json`` shows bit-exact reproduction of the published
  near-control values with this code.
* No E01 outcome exists.  Numbers produced from synthetic fixtures are not evidence.

Availability rules (see the DECISION REQUEST in the specification)
------------------------------------------------------------------
``A_complete_grid`` (default, recommended): an endpoint is estimated only from a complete,
defined 24-block x 6-context grid (the original rule).  ``B_available_context`` and
``C_complete_grid_plus_descriptive`` are implemented only so the coordinator can choose
between the drafted options without new code; they are labelled in the output.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import struct  # noqa: E402
from pathlib import Path  # noqa: E402

SCHEMA_RETURN = "e01-far-control-return-v1"
SCHEMA_NEAR = "e01-near-baseline-bundle-v1"
SCHEMA_OUTPUT = "e01-far-control-analysis-draft-v1"

# ----------------------------------------------------------------------------------------
# Frozen population (identical to the original p5_main setting)
# ----------------------------------------------------------------------------------------
BLOCK_IDS = tuple(f"p5_main-block{index:03d}" for index in range(24))
BACKGROUNDS = (0, 1)
SEEDS = (0, 1, 2)
FOCALS = 8
SCORED_TARGETS = 255
TRAIN_ROWS = 768
OPTIMIZER_STEPS = 384
CONTEXT_KEYS = tuple((block, background, seed) for block in BLOCK_IDS
                     for background in BACKGROUNDS for seed in SEEDS)
FIT_ROLES = ("IN", "OUT_FAR", "G20_FAR")
FIT_STATUSES = ("completed", "failed", "not_run")
IDENTITY_STATUSES = ("exact_match", "mismatch", "not_evaluable")
READOUT_QUALIFICATION = ("passed", "failed_cannot_emit_frozen_255_target_nll")

# Proposal method-failure threshold: rank undefined in MORE than 8 of 144 contexts.
UNDEFINED_CONTEXT_LIMIT = 8

FAMILY = {
    "name": "E01_far_control_primary_family",
    "alpha": 0.05,
    "test_count": 5,
    "adjustment": "Bonferroni: each two-sided percentile interval at 1 - alpha/5 (0.99)",
    "root_seed": 202610101747,
    "bootstrap_replicates": 20000,
    "endpoint_seed_rule": ("stable_seed(root_seed, endpoint_id): SHA-256 of the pipe-joined decimal/string "
                           "parts, first 8 bytes big-endian, modulo 2**31-1 (original helper)."),
    "law": ("Python random.Random(endpoint_seed); per replicate draw 24 block indices with replacement by "
            "24 successive randrange(24) calls over the sorted block order; statistic = equal-weight mean "
            "of the drawn block values; fixed backgrounds, seeds and focals are retained inside blocks."),
    "percentile": "Linear interpolation at index (n-1)*p with p = (1-confidence)/2 and 1-(1-confidence)/2.",
    "unavailable_endpoints_stay_in_denominator": True,
    "separate_from": [
        "LM_primary: original 11-test family (5,000 draws, root 2026100205, 1-0.05/11)",
        "new_joint_E01_E02_E03_E04_main_family: later 11-endpoint follow-up family "
        "(20,000 draws, root 202610051754, 1-0.05/11); its 'E01' is a different (G5 reference) study",
    ],
}

ENDPOINTS = (
    {"number": 1, "id": "E01FAR_far_G20far_vs_A_rank", "kind": "far_rank_difference",
     "left": "G20_far", "right": "audit_loss_drop", "unit": "Spearman difference",
     "context_value": "Spearman(G20_far, T_far) - Spearman(audit_loss_drop, T_far)",
     "near_analog": "p5_main_G_vs_A_spearman", "anticipated_direction": "either",
     "positive_means": "G20_far ranks far cross-seed benefit better than total loss drop",
     "negative_means": "total loss drop ranks far cross-seed benefit better than G20_far"},
    {"number": 2, "id": "E01FAR_far_U4_vs_difficulty_rank", "kind": "far_rank_difference",
     "left": "uncovered_base_nll_k4", "right": "base_difficulty_nll", "unit": "Spearman difference",
     "context_value": "Spearman(U4, T_far) - Spearman(base_difficulty, T_far)",
     "near_analog": "p5_main_corpus_vs_difficulty_rank", "anticipated_direction": "either",
     "positive_means": "U4 ranks far cross-seed benefit better than base difficulty",
     "negative_means": "base difficulty ranks far cross-seed benefit better than U4"},
    {"number": 3, "id": "E01FAR_far_G20far_vs_A_regret", "kind": "far_regret_improvement",
     "left": "G20_far", "right": "audit_loss_drop", "unit": "nats/token",
     "context_value": "regret(audit_loss_drop, T_far) - regret(G20_far, T_far)",
     "near_analog": "p5_main_G_vs_A_top_two_regret", "anticipated_direction": "either",
     "positive_means": "G20_far has lower far top-two regret than total loss drop",
     "negative_means": "total loss drop has lower far top-two regret than G20_far"},
    {"number": 4, "id": "E01FAR_far_minus_near_U4_vs_difficulty_rank", "kind": "policy_change",
     "left": "uncovered_base_nll_k4", "right": "base_difficulty_nll", "unit": "Spearman difference",
     "context_value": "far: Spearman(U4,T_far)-Spearman(D,T_far); near: Spearman(U4,T_near)-Spearman(D,T_near)",
     "block_value": "mean of the block's 6 far context values minus mean of its 6 near context values",
     "near_analog": "p5_main_corpus_vs_difficulty_rank (its per-block cluster_means are the near component)",
     "anticipated_direction": "less",
     "positive_means": "U4's rank advantage over difficulty is larger under far controls (contrary)",
     "negative_means": "U4's rank advantage over difficulty is smaller under far controls (supports policy sensitivity)"},
    {"number": 5, "id": "E01FAR_far_minus_near_G20_vs_A_rank", "kind": "policy_change",
     "left": "G20_far (far arm) / G20 (near arm)", "right": "audit_loss_drop", "unit": "Spearman difference",
     "context_value": "far: Spearman(G20_far,T_far)-Spearman(A,T_far); near: Spearman(G20,T_near)-Spearman(A,T_near)",
     "block_value": "mean of the block's 6 far context values minus mean of its 6 near context values",
     "near_analog": "p5_main_G_vs_A_spearman (its per-block cluster_means are the near component)",
     "anticipated_direction": "either",
     "positive_means": "the G20-over-loss-drop rank advantage is larger under far controls",
     "negative_means": "the G20-over-loss-drop rank advantage is smaller under far controls"},
)
ENDPOINT_IDS = tuple(row["id"] for row in ENDPOINTS)
AVAILABILITY_RULES = ("A_complete_grid", "B_available_context", "C_complete_grid_plus_descriptive")
FAR_SCORES = ("G20_far", "audit_loss_drop", "uncovered_base_nll_k4", "base_difficulty_nll")
NEAR_SCORE_KEYS = ("audit_loss_drop", "base_difficulty_nll", "uncovered_base_nll_k4", "G20")


class ContractError(ValueError):
    """Structurally invalid, duplicated or identity-inconsistent input. Never an outcome."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


# ----------------------------------------------------------------------------------------
# Identities (byte-compatible with the original helpers)
# ----------------------------------------------------------------------------------------
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def stable_seed(*parts):
    """lm_support/common.py:96-98 and analysis_bootstrap.py:22-24, same law."""
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2 ** 31 - 1)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _no_constant(name):
    raise ContractError("nonfinite JSON constant: " + name)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.loads(stream.read(), object_pairs_hook=_unique_object, parse_constant=_no_constant)


# ----------------------------------------------------------------------------------------
# Core statistics: lm_support/statistics.py:11-72 (same arithmetic, same order)
# ----------------------------------------------------------------------------------------
def mean(values):
    values = list(values)
    require(bool(values), "mean of empty cell")
    return math.fsum(values) / len(values)


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            result[index] = (start + end - 1) / 2
        start = end
    return result


def is_constant(values):
    """True when every value ties exactly (float64 equality): its rank vector is constant."""
    return all(value == values[0] for value in values)


def spearman(left, right):
    """Return (value, reason); value is None when undefined and reason names why."""
    left, right = list(left), list(right)
    require(len(left) == len(right) and bool(left), "unpaired correlation vectors")
    if any(value is None for value in left + right):
        return None, "unavailable_value"
    require(all(finite(value) for value in left + right), "nonfinite correlation input")
    if len(left) < 2:
        return None, "fewer_than_two_items"
    a, b = ranks(left), ranks(right)
    a, b = [x - mean(a) for x in a], [x - mean(b) for x in b]
    denominator = math.sqrt(math.fsum(x * x for x in a) * math.fsum(x * x for x in b))
    if not denominator:
        return None, "constant_rank_vector"
    return math.fsum(x * y for x, y in zip(a, b)) / denominator, None


def top_two_regret(predicted, oracle):
    """Mean oracle of the best two minus mean oracle of the two selected by ``predicted``.

    Ties in either vector are broken by ascending index in the frozen focal vector, so
    constant vectors are *defined* (a constant oracle gives regret 0).
    """
    predicted, oracle = list(predicted), list(oracle)
    require(len(predicted) == len(oracle) == FOCALS, "fixed top-two-of-eight needs exactly eight focals")
    if any(value is None for value in predicted + oracle):
        return None
    require(all(finite(value) for value in predicted + oracle), "nonfinite regret input")
    chosen = sorted(range(FOCALS), key=lambda index: (-predicted[index], index))[:2]
    best = sorted(range(FOCALS), key=lambda index: (-oracle[index], index))[:2]
    return mean([oracle[index] for index in best]) - mean([oracle[index] for index in chosen])


def quantile(values, probability):
    require(bool(values) and 0 <= probability <= 1, "invalid quantile")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def bootstrap_interval(block_values, *, seed, replicates, confidence):
    """Whole-block percentile interval: cluster_summary (statistics.py:75-134) with unit weights.

    ``block_values`` must be in the frozen sorted block order.  Also returns the SHA-256 of the
    drawn index stream (follow-up encoding: canonical header, then big-endian uint32 indices).
    """
    require(len(block_values) >= 2 and all(finite(value) for value in block_values),
            "bootstrap needs at least two finite block values")
    require(type(replicates) is int and replicates >= 20 and 0 < confidence < 1, "invalid bootstrap design")
    count = len(block_values)
    rng = random.Random(seed)
    stream = hashlib.sha256(canonical({"n_blocks": count, "seed": seed, "draws": replicates}))
    packer = struct.Struct(">" + "I" * count)
    boot = []
    for _ in range(replicates):
        sample = [rng.randrange(count) for _ in range(count)]
        stream.update(packer.pack(*sample))
        boot.append(math.fsum(block_values[index] for index in sample) / count)
    tail = (1 - confidence) / 2
    return [quantile(boot, tail), quantile(boot, 1 - tail)], stream.hexdigest()


def direction_of(interval):
    if interval is None:
        return None
    if interval[0] > 0:
        return "greater"
    if interval[1] < 0:
        return "less"
    return "unresolved"


def other_two_seed_target(theta_by_seed, seed):
    """Same-background other-two-seed mean (p5_analysis.py:181-186 via _average:83-86)."""
    others = [theta_by_seed[other] for other in SEEDS if other != seed]
    return [None if any(row is None or row[index] is None for row in others)
            else mean([row[index] for row in others]) for index in range(FOCALS)]


def context_id(block, background, seed):
    return f"{block}-b{background}-s{seed}"


def _present(vector):
    return vector is not None and all(value is not None for value in vector)


def _is_hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _vector8(values, label):
    require(isinstance(values, list) and len(values) == FOCALS, f"{label}: exactly eight values required")
    require(all(finite(value) for value in values), f"{label}: non-finite or missing value")
    return list(values)


# ----------------------------------------------------------------------------------------
# Near baseline bundle (historical, locally built from sealed records; never refit)
# ----------------------------------------------------------------------------------------
def validate_near_bundle(bundle):
    require(isinstance(bundle, dict) and bundle.get("schema") == SCHEMA_NEAR, "unknown near bundle schema")
    rows = bundle.get("contexts")
    require(isinstance(rows, list) and len(rows) == len(CONTEXT_KEYS), "near bundle must hold all 144 contexts")
    allowed, index = set(CONTEXT_KEYS), {}
    for row in rows:
        key = (row.get("block_id"), row.get("background"), row.get("seed_label"))
        require(key in allowed, "near context outside the frozen population")
        require(key not in index, "duplicate near context")
        require(row.get("context_id") == context_id(*key), "near context_id mismatch")
        require(type(row.get("fit_seed")) is int, "near fit_seed must be an integer")
        for name in ("focal_query_ids", "focal_member_ids", "near_control_ids"):
            ids = row.get(name)
            require(isinstance(ids, list) and len(ids) == FOCALS and len(set(ids)) == FOCALS
                    and all(isinstance(value, str) for value in ids), f"near {name} must be 8 distinct strings")
        _vector8(row.get("theta_near"), "theta_near")
        _vector8(row.get("all_in_focal_nll"), "all_in_focal_nll")
        scores = row.get("scores")
        require(isinstance(scores, dict) and set(NEAR_SCORE_KEYS) <= set(scores), "near scores incomplete")
        for name in NEAR_SCORE_KEYS:
            _vector8(scores[name], "near score " + name)
        index[key] = row
    for block in BLOCK_IDS:
        orders = {tuple(index[(block, g, s)]["focal_query_ids"]) for g in BACKGROUNDS for s in SEEDS}
        require(len(orders) == 1, "near focal order differs inside a block: " + block)
    return index


# ----------------------------------------------------------------------------------------
# Server return (ANALYSIS_INPUT_SCHEMA.json); hand-written validator (jsonschema not installed)
# ----------------------------------------------------------------------------------------
def _readout(fit, query_id):
    """Return (nll or None, problem code or None) for one required focal readout of a completed fit."""
    nll_map, count_map = fit.get("focal_nll"), fit.get("focal_scored_tokens")
    if not isinstance(nll_map, dict) or not isinstance(count_map, dict):
        return None, "READOUT_MAP_ABSENT"
    if query_id not in nll_map or query_id not in count_map:
        return None, "READOUT_MISSING_QUERY"
    if count_map[query_id] != SCORED_TARGETS:
        return None, "READOUT_NOT_255_TARGETS"
    value = nll_map[query_id]
    if not finite(value):
        return None, "READOUT_NONFINITE"
    return value, None


def _check_fit(fit, row, far_controls):
    role = fit.get("role")
    require(role in FIT_ROLES, "unknown fit role")
    focal_index = fit.get("focal_index")
    if role == "OUT_FAR":
        require(type(focal_index) is int and 0 <= focal_index < FOCALS, "OUT_FAR needs focal_index 0..7")
        require(fit.get("replaced_member_id") == row["focal_member_ids"][focal_index]
                and fit.get("far_control_id") == far_controls[focal_index],
                "OUT_FAR replacement identity differs from the context focal/far-control vectors")
    else:
        require(focal_index is None, role + " must carry focal_index null")
    if role == "G20_FAR":
        require(fit.get("far_map_sha256") == row["far_map_sha256"], "G20_FAR uses a different far map")
    require(fit.get("status") in FIT_STATUSES, "fit status")
    require(isinstance(fit.get("fit_id"), str) and bool(fit["fit_id"]), "fit_id required")
    require(fit.get("fit_seed") == row["fit_seed"], "fit seed differs inside a context")
    if fit["status"] == "completed":
        require(fit.get("optimizer_steps") == OPTIMIZER_STEPS and fit.get("train_rows") == TRAIN_ROWS,
                "completed fit violates 768 rows / 384 optimizer steps")
        require(all(_is_hex64(fit.get(name)) for name in
                    ("train_ids_sha256", "data_order_sha256", "runtime_identity_sha256")),
                "completed fit lacks training/runtime identity hashes")
    return role, focal_index


def _check_far_map(row, frozen_far_map):
    """Optional binding to the frozen far-control map (recon_inputs), keyed by block or block-background."""
    if frozen_far_map is None:
        return
    entry = frozen_far_map.get(f"{row['block_id']}-b{row['background']}", frozen_far_map.get(row["block_id"]))
    require(isinstance(entry, dict), "frozen far map has no entry for " + row["context_id"])
    require(entry.get("far_map_sha256") == row["far_map_sha256"]
            and entry.get("focal_far_control_ids") == row["far_control_ids"],
            "context far controls differ from the frozen far map: " + row["context_id"])


def validate_far_return(document, near_index, frozen_far_map=None):
    """Validate identities and collect per-context readouts; missing data is retained, never imputed.

    Returns (contexts, workers, release).  ``contexts`` maps (block, background, seed) to a record
    with L_IN, L_OUT, L_G20 (lists with None where unavailable) and the per-fit problems.
    Structural contradictions (duplicates, focal-order or seed drift, unknown roles) raise.
    """
    require(isinstance(document, dict) and document.get("schema") == SCHEMA_RETURN, "unknown return schema")
    release = document.get("release")
    require(isinstance(release, dict) and isinstance(release.get("release_id"), str)
            and _is_hex64(release.get("package_sha256")), "release identity incomplete")
    workers = {}
    for worker in document.get("workers", []):
        require(type(worker.get("worker_id")) is int and worker["worker_id"] not in workers, "worker id")
        require(worker.get("readout_qualification") in READOUT_QUALIFICATION, "worker readout qualification")
        workers[worker["worker_id"]] = worker
    contexts = {}
    for row in document.get("contexts", []):
        key = (row.get("block_id"), row.get("background"), row.get("seed_label"))
        require(key in near_index, "returned context outside the frozen population: " + repr(key))
        require(key not in contexts, "duplicate returned context: " + repr(key))
        require(row.get("context_id") == context_id(*key), "context_id mismatch")
        near = near_index[key]
        require(row.get("fit_seed") == near["fit_seed"], "fit_seed differs from the historical context")
        require(row.get("focal_query_ids") == near["focal_query_ids"],
                "focal query order differs from the frozen historical focal vector")
        require(row.get("focal_member_ids") == near["focal_member_ids"], "focal member ids differ")
        far_controls = row.get("far_control_ids")
        require(isinstance(far_controls, list) and len(far_controls) == FOCALS
                and all(isinstance(value, str) for value in far_controls), "far_control_ids must list 8 ids")
        require(all(far != nearest for far, nearest in zip(far_controls, near["near_control_ids"])),
                "a focal's far control equals its recorded nearest control")
        require(_is_hex64(row.get("far_map_sha256")), "far_map_sha256 required")
        _check_far_map(row, frozen_far_map)
        require(row.get("worker_id") in workers, "context names an undeclared worker")
        identity = row.get("in_identity")
        require(isinstance(identity, dict) and identity.get("status") in IDENTITY_STATUSES,
                "in_identity.status required")
        historical_in = near.get("historical_job_ids", {}).get("ALL_IN")
        require(isinstance(identity.get("historical_all_in_job_id"), str)
                and (historical_in is None or identity["historical_all_in_job_id"] == historical_in),
                "in_identity.historical_all_in_job_id must name the context's historical ALL_IN job")
        fits = row.get("fits")
        require(isinstance(fits, list), "fits list required")
        seen = {}
        for fit in fits:
            slot = _check_fit(fit, row, far_controls)
            require(slot not in seen, "duplicate fit record: " + repr(slot))
            seen[slot] = fit
        record = {"row": row, "L_IN": [None] * FOCALS, "L_OUT": [None] * FOCALS,
                  "L_G20": [None] * FOCALS, "problems": []}
        queries = row["focal_query_ids"]

        def collect(slot, positions, target):
            fit = seen.get(slot)
            if fit is None:
                record["problems"].append({"fit": list(slot), "code": "FIT_RECORD_ABSENT"})
                return
            if fit["status"] != "completed":
                record["problems"].append({"fit": list(slot), "code": "FIT_" + fit["status"].upper(),
                                           "detail": fit.get("failure_code")})
                return
            for position in positions:
                value, problem = _readout(fit, queries[position])
                if problem:
                    record["problems"].append({"fit": list(slot), "focal_index": position, "code": problem})
                target[position] = value

        collect(("IN", None), range(FOCALS), record["L_IN"])
        for position in range(FOCALS):
            collect(("OUT_FAR", position), [position], record["L_OUT"])
        collect(("G20_FAR", None), range(FOCALS), record["L_G20"])
        contexts[key] = record
    return contexts, workers, release


# ----------------------------------------------------------------------------------------
# Per-context quantities
# ----------------------------------------------------------------------------------------
def _difference(left, right):
    return None if left is None or right is None else left - right


def far_vectors(far_contexts):
    """theta_far = L(OUT_far) - L(IN) and G20_far = L(G20_FAR) - L(IN) (p5_analysis.py:133, :139)."""
    theta, g20 = {}, {}
    for key in CONTEXT_KEYS:
        record = far_contexts.get(key)
        if record is None:
            theta[key], g20[key] = [None] * FOCALS, [None] * FOCALS
            continue
        theta[key] = [_difference(o, i) for o, i in zip(record["L_OUT"], record["L_IN"])]
        g20[key] = [_difference(g, i) for g, i in zip(record["L_G20"], record["L_IN"])]
    return theta, g20


def targets(theta):
    result = {}
    for block in BLOCK_IDS:
        for background in BACKGROUNDS:
            by_seed = {seed: theta[(block, background, seed)] for seed in SEEDS}
            for seed in SEEDS:
                result[(block, background, seed)] = other_two_seed_target(by_seed, seed)
    return result


def context_table(near_index, far_contexts):
    """Every per-context rank/regret used by the family, with reasons for undefined cells."""
    theta_far, g20_far = far_vectors(far_contexts)
    t_far = targets(theta_far)
    t_near = targets({key: near_index[key]["theta_near"] for key in CONTEXT_KEYS})
    table = {}
    for key in CONTEXT_KEYS:
        near = near_index[key]["scores"]
        far_scores = {"G20_far": g20_far[key], "audit_loss_drop": near["audit_loss_drop"],
                      "uncovered_base_nll_k4": near["uncovered_base_nll_k4"],
                      "base_difficulty_nll": near["base_difficulty_nll"]}
        near_scores = {"G20": near["G20"], "audit_loss_drop": near["audit_loss_drop"],
                       "uncovered_base_nll_k4": near["uncovered_base_nll_k4"],
                       "base_difficulty_nll": near["base_difficulty_nll"]}
        row = {"T_far": t_far[key], "T_near": t_near[key], "theta_far": theta_far[key],
               "G20_far": g20_far[key], "near_G20": near["G20"], "far_rank": {}, "far_reason": {},
               "near_rank": {}, "near_reason": {}, "far_regret": {}, "near_regret": {}}
        for name, vector in far_scores.items():
            row["far_rank"][name], row["far_reason"][name] = spearman(vector, t_far[key])
            row["far_regret"][name] = top_two_regret(vector, t_far[key])
        for name, vector in near_scores.items():
            row["near_rank"][name], row["near_reason"][name] = spearman(vector, t_near[key])
            row["near_regret"][name] = top_two_regret(vector, t_near[key])
        constant = []
        if _present(t_far[key]) and is_constant(t_far[key]):
            constant.append("T_far")
        for name, vector in far_scores.items():
            if _present(vector) and is_constant(vector):
                constant.append(name)
        row["constant_vectors"] = constant
        table[key] = row
    return table


def endpoint_context_values(table, number):
    """Per-context {far, near, reasons} for one endpoint (near is None for endpoints 1-3)."""
    values = {}
    for key, row in table.items():
        far, near_value, reasons = None, None, []
        if number in (1, 5):
            names = ("G20_far", "audit_loss_drop")
        elif number in (2, 4):
            names = ("uncovered_base_nll_k4", "base_difficulty_nll")
        else:
            names = ()
        if names:
            a, b = (row["far_rank"][name] for name in names)
            far = None if a is None or b is None else a - b
            reasons += [f"far:{name}:{row['far_reason'][name]}" for name in names if row["far_reason"][name]]
        else:  # endpoint 3; paired_summary (statistics.py:144-150) uses direction -1 for regret
            left, right = row["far_regret"]["G20_far"], row["far_regret"]["audit_loss_drop"]
            far = None if left is None or right is None else -1 * (left - right)
            if far is None:
                reasons.append("far:top_two_regret:unavailable_value")
        if number in (4, 5):
            names = ("uncovered_base_nll_k4", "base_difficulty_nll") if number == 4 else ("G20", "audit_loss_drop")
            a, b = (row["near_rank"][name] for name in names)
            near_value = None if a is None or b is None else a - b
            reasons += [f"near:{name}:{row['near_reason'][name]}" for name in names if row["near_reason"][name]]
        values[key] = {"far": far, "near": near_value, "reasons": reasons}
    return values


def _cell_defined(cell, number):
    return cell["far"] is not None and (number not in (4, 5) or cell["near"] is not None)


def _constant_only(cell):
    return bool(cell["reasons"]) and all(reason.endswith(":constant_rank_vector") for reason in cell["reasons"])


# ----------------------------------------------------------------------------------------
# Endpoint inference
# ----------------------------------------------------------------------------------------
def _block_value(cells, number, keys):
    far = mean([cells[key]["far"] for key in keys])
    if number in (4, 5):
        return far - mean([cells[key]["near"] for key in keys])
    return far


def endpoint_branch(spec, direction):
    if direction == "unresolved":
        return "UNRESOLVED"
    if spec["anticipated_direction"] == "either":
        return "RESOLVED_POSITIVE" if direction == "greater" else "RESOLVED_NEGATIVE"
    return "SUPPORT" if direction == spec["anticipated_direction"] else "CONTRARY"


def _available_context_block_values(cells, number):
    values, used = [], []
    for block in BLOCK_IDS:
        keys = [(block, g, s) for g in BACKGROUNDS for s in SEEDS if _cell_defined(cells[(block, g, s)], number)]
        if not keys:
            return None, None
        values.append(_block_value(cells, number, keys))
        used.append(len(keys))
    return values, used


def infer_endpoint(spec, cells, *, identity_failures, rule, replicates):
    number = spec["number"]
    confidence = 1 - FAMILY["alpha"] / FAMILY["test_count"]
    seed = stable_seed(FAMILY["root_seed"], spec["id"])
    record = {**spec, "family": FAMILY["name"], "family_denominator": FAMILY["test_count"],
              "confidence": confidence, "endpoint_seed": seed, "replicates": replicates,
              "availability_rule": rule, "status": None, "estimate": None, "interval": None,
              "direction_resolved": None, "branch": None, "block_values": None,
              "undefined_or_missing_cells": [], "blocking_reasons": [], "index_stream_sha256": None,
              "input_cells_sha256": None}
    undefined = [key for key in CONTEXT_KEYS if not _cell_defined(cells[key], number)]
    record["undefined_or_missing_cells"] = [{"context_id": context_id(*key),
                                             "reasons": cells[key]["reasons"] or ["undefined"]}
                                            for key in undefined]
    if number in (4, 5) and identity_failures:
        record["blocking_reasons"].append({
            "code": "IN_REPLAY_IDENTITY_NOT_ESTABLISHED", "contexts": identity_failures,
            "rule": "Far-minus-historical-near contrasts are unavailable unless every fresh IN replay is "
                    "identical to its historical ALL_IN (handoff B0a item 4)."})
    tolerated = (rule == "B_available_context" and 0 < len(undefined) <= UNDEFINED_CONTEXT_LIMIT
                 and all(_constant_only(cells[key]) for key in undefined))
    if undefined and not tolerated:
        record["blocking_reasons"].append({
            "code": "INCOMPLETE_OR_UNDEFINED_GRID", "count": len(undefined),
            "rule": "Undefined or missing required cells preclude the complete metric; no cell is dropped, "
                    "zero-filled or imputed."})
    if record["blocking_reasons"]:
        record.update(status="unavailable", branch="UNAVAILABLE")
        if rule == "C_complete_grid_plus_descriptive" and not (number in (4, 5) and identity_failures):
            values, used = _available_context_block_values(cells, number)
            record["descriptive_available_context"] = (
                {"status": "not_computable_block_without_defined_context", "label": "descriptive_only"}
                if values is None else
                {"status": "descriptive_only_not_primary", "estimate": mean(values), "interval": None,
                 "contexts_used_per_block": used,
                 "label": "Available-context mean; not a complete-grid primary; no interval or branch."})
        return record
    values, used = _available_context_block_values(cells, number)
    if values is None:
        record["blocking_reasons"].append({"code": "BLOCK_WITHOUT_DEFINED_CONTEXT"})
        record.update(status="unavailable", branch="UNAVAILABLE")
        return record
    record["block_values"] = [
        {"block_id": block, "value": value, "contexts_used": count,
         "far_context_values": [cells[(block, g, s)]["far"] for g in BACKGROUNDS for s in SEEDS],
         "near_context_values": ([cells[(block, g, s)]["near"] for g in BACKGROUNDS for s in SEEDS]
                                 if number in (4, 5) else None)}
        for block, value, count in zip(BLOCK_IDS, values, used)]
    record["input_cells_sha256"] = digest([[[cells[(b, g, s)]["far"], cells[(b, g, s)]["near"]]
                                            for g in BACKGROUNDS for s in SEEDS] for b in BLOCK_IDS])
    record["estimate"] = mean(values)
    record["interval"], record["index_stream_sha256"] = bootstrap_interval(
        values, seed=seed, replicates=replicates, confidence=confidence)
    record["direction_resolved"] = direction_of(record["interval"])
    record["status"] = "available" if not undefined else "available_context_estimate_not_complete_grid"
    record["branch"] = endpoint_branch(spec, record["direction_resolved"])
    return record


def identity_failures(near_index, far_contexts):
    failures = []
    for key in CONTEXT_KEYS:
        record = far_contexts.get(key)
        cid = context_id(*key)
        if record is None:
            failures.append({"context_id": cid, "code": "CONTEXT_ABSENT"})
            continue
        status = record["row"]["in_identity"]["status"]
        values, historical = record["L_IN"], near_index[key]["all_in_focal_nll"]
        exact = _present(values) and all(a == b for a, b in zip(values, historical))
        if status != "exact_match" or not exact:
            failures.append({"context_id": cid, "declared_status": status,
                             "focal_nll_exactly_equal": bool(exact)})
    return failures


def analyze(far_return, near_bundle, *, rule="A_complete_grid", replicates=None, source_hashes=None,
            include_descriptive=True, frozen_far_map=None):
    """Full E01 primary-family analysis.  ``replicates`` may be lowered only in synthetic tests."""
    require(rule in AVAILABILITY_RULES, "unknown availability rule")
    replicates = FAMILY["bootstrap_replicates"] if replicates is None else replicates
    near_index = validate_near_bundle(near_bundle)
    far_contexts, workers, release = validate_far_return(far_return, near_index, frozen_far_map)
    table = context_table(near_index, far_contexts)
    absent = [context_id(*key) for key in CONTEXT_KEYS if key not in far_contexts]
    fit_problems = [{"context_id": context_id(*key), **problem}
                    for key, record in sorted(far_contexts.items()) for problem in record["problems"]]
    constant_contexts = {context_id(*key): row["constant_vectors"] for key, row in table.items()
                         if row["constant_vectors"]}
    failed_readout_workers = sorted(worker for worker, value in workers.items()
                                    if value["readout_qualification"] != "passed")
    abort_reasons = []
    if len(constant_contexts) > UNDEFINED_CONTEXT_LIMIT:
        abort_reasons.append({"code": "RANK_UNDEFINED_IN_MORE_THAN_8_OF_144_CONTEXTS",
                              "count": len(constant_contexts), "contexts": sorted(constant_contexts)})
    if failed_readout_workers:
        abort_reasons.append({"code": "WORKER_CANNOT_EMIT_FROZEN_255_TARGET_NLL",
                              "workers": failed_readout_workers})
    failures = identity_failures(near_index, far_contexts)
    endpoints = []
    for spec in ENDPOINTS:
        if abort_reasons:
            endpoints.append({**spec, "family": FAMILY["name"], "family_denominator": FAMILY["test_count"],
                              "status": "not_evaluated_method_failure", "estimate": None, "interval": None,
                              "branch": "ABORT", "endpoint_seed": stable_seed(FAMILY["root_seed"], spec["id"])})
            continue
        cells = endpoint_context_values(table, spec["number"])
        endpoints.append(infer_endpoint(spec, cells, identity_failures=failures, rule=rule,
                                        replicates=replicates))
    by_number = {row["number"]: row for row in endpoints}
    result = {
        "schema": SCHEMA_OUTPUT,
        "status": ("draft_analysis_output" if replicates == FAMILY["bootstrap_replicates"]
                   else "NONFROZEN_REPLICATE_COUNT_TEST_ONLY_NOT_THE_E01_LAW"),
        "family": FAMILY,
        "availability_rule": rule,
        "population": {"blocks": list(BLOCK_IDS), "backgrounds": list(BACKGROUNDS), "seed_labels": list(SEEDS),
                       "focals_per_context": FOCALS, "contexts_expected": len(CONTEXT_KEYS),
                       "inference_unit": "original block",
                       "weighting": "equal contexts within block, equal blocks"},
        "release": release,
        "source_hashes": source_hashes or {},
        "experiment_branch": {
            "U4_policy_contrast_endpoint_4": "ABORT_METHOD_FAILURE" if abort_reasons else by_number[4]["branch"],
            "G20_policy_contrast_endpoint_5": "ABORT_METHOD_FAILURE" if abort_reasons else by_number[5]["branch"],
            "abort_reasons": abort_reasons,
            "interpretation_limits": [
                "An unresolved interval does not establish invariance or equivalence.",
                "Unavailability and method failure are not scientific nulls.",
                "No outcome demonstrates practical data-selection utility or supplies missing attribution "
                "comparisons.",
                "Far-minus-near contrasts test replacement-policy dependence, not lexical distance as a sole cause.",
            ],
        },
        "accounting": {
            "contexts_returned": len(far_contexts), "contexts_absent": absent,
            "fit_problems": fit_problems,
            "constant_rank_contexts": constant_contexts,
            "constant_rank_context_count": len(constant_contexts),
            "undefined_context_limit": UNDEFINED_CONTEXT_LIMIT,
            "in_identity_failures": failures,
            "planned_science_fits": 10 * len(CONTEXT_KEYS),
        },
        "endpoints": endpoints,
    }
    if include_descriptive and not abort_reasons:
        result["descriptive_outside_family"] = descriptive(table, replicates=replicates)
    return result


# ----------------------------------------------------------------------------------------
# Descriptive diagnostics (outside the family; nominal 95%; never used for branches)
# ----------------------------------------------------------------------------------------
def _descriptive_mean(values, item_id, replicates):
    if any(value is None for value in values.values()):
        return {"status": "unavailable_undefined_or_missing_cells",
                "undefined": sum(value is None for value in values.values())}
    blocks = [mean([values[(b, g, s)] for g in BACKGROUNDS for s in SEEDS]) for b in BLOCK_IDS]
    seed = stable_seed(FAMILY["root_seed"], "e01-descriptive", item_id)
    interval, stream = bootstrap_interval(blocks, seed=seed, replicates=replicates, confidence=0.95)
    return {"status": "available_descriptive", "estimate": mean(blocks), "interval": interval,
            "confidence": 0.95, "seed": seed, "index_stream_sha256": stream}


def descriptive(table, *, replicates):
    out = {"label": "descriptive_outside_family_nominal_95_not_multiplicity_adjusted", "items": {}}
    for name in FAR_SCORES:
        for metric, source in (("spearman", "far_rank"), ("top_two_regret", "far_regret")):
            item_id = f"far.{name}.{metric}"
            values = {key: row[source][name] for key, row in table.items()}
            out["items"][item_id] = _descriptive_mean(values, item_id, replicates)
    # Mismatched-reference description (proposal sec. 3): published near G20 against T_far.
    item_id = "far.mismatched_near_G20.spearman"
    values = {key: spearman(row["near_G20"], row["T_far"])[0] for key, row in table.items()}
    out["items"][item_id] = _descriptive_mean(values, item_id, replicates)
    out["items"][item_id]["label"] = "mismatched-reference description; secondary"
    sd = {context_id(*key): (statistics.pstdev(row["theta_far"]) if _present(row["theta_far"]) else None)
          for key, row in table.items()}
    defined = [value for value in sd.values() if value is not None]
    out["theta_far_within_context_sd"] = {
        "statistic": "population SD of the eight same-seed theta_far values per context",
        "contexts_defined": len(defined), "zero_sd_contexts": sum(value == 0 for value in defined),
        "minimum": min(defined) if defined else None,
        "median": statistics.median(defined) if defined else None}
    errors = [abs(g - t) for row in table.values() if _present(row["G20_far"]) and _present(row["T_far"])
              for g, t in zip(row["G20_far"], row["T_far"])]
    out["G20_far_median_absolute_error_vs_T_far"] = {
        "value": statistics.median(errors) if len(errors) == FOCALS * len(CONTEXT_KEYS) else None,
        "cells": len(errors), "label": "pooled focal/context median of |G20_far - T_far|, nats/token"}
    return out


# ----------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="E01 far-control analysis (draft).")
    parser.add_argument("--near", required=True, help="e01-near-baseline-bundle-v1 JSON")
    parser.add_argument("--far", required=True, help="e01-far-control-return-v1 JSON (server return)")
    parser.add_argument("--output", required=True, help="new output path; never overwritten")
    parser.add_argument("--rule", default="A_complete_grid", choices=AVAILABILITY_RULES)
    parser.add_argument("--far-map", default=None,
                        help="optional frozen far map JSON {block_id or block_id-bN: {far_map_sha256, "
                             "focal_far_control_ids}}")
    args = parser.parse_args(argv)
    hashes = {"near": file_sha256(args.near), "far": file_sha256(args.far),
              "analysis_e01.py": file_sha256(__file__)}
    far_map = None
    if args.far_map:
        hashes["far_map"] = file_sha256(args.far_map)
        far_map = read_json(args.far_map)
    result = analyze(read_json(args.far), read_json(args.near), rule=args.rule, source_hashes=hashes,
                     frozen_far_map=far_map)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": args.output, "branch": result["experiment_branch"]}, sort_keys=True))


if __name__ == "__main__":
    main()
