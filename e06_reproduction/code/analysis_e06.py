"""E06 exposure-gradient analysis: the frozen analysis law of E06_PROSPECTIVE_SPEC.md sections 7-8.

Specification: ``E06_ANALYSIS_SPEC.md`` (same folder).  Written 2026-10-11, before any E06 data exist.

Scope and guarantees
--------------------
* No training, inference, tokenization, network access, GPU or neural import.  Standard library only.
* Reads only the JSON inputs named on the command line and writes exactly one *new* output file
  (``open(..., "x")``); it never overwrites.
* The statistics primitives are copied verbatim from the proven E01 analysis
  (``e01_reproduction/code/analysis_e01.py``, SHA-256 536fac37...26dd5): average-rank Spearman,
  undefined for constant rank vectors; top-two-of-eight regret with ties to the lower frozen focal
  index; the same-background other-two-seed target; fsum means; and the Python ``random.Random``
  whole-block percentile bootstrap with linear-interpolated quantiles and SHA-256 ``stable_seed``
  endpoint seeds.  ``test_analysis_e06.py`` checks each copy against that file, text for text.
* Numbers produced from ``make_synthetic_return.py`` fixtures are not E06 outcomes.

Usage
-----
    python -B analysis_e06.py --return E06_RETURN_ALL.json --near NEAR_INPUT_BUNDLE.json \
        --followup CONTEXTS.json --far E01_RETURN_ALL.json --output E06_ANALYSIS_RESULT.json
    python -B analysis_e06.py --merge-workers E06_RETURN_machine1.json ... E06_RETURN_machine6.json \
        [--absent-worker machineN] --near NEAR_INPUT_BUNDLE.json --output E06_RETURN_ALL.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import struct  # noqa: E402
from pathlib import Path  # noqa: E402

SCHEMA_RETURN = "e06-exposure-gradient-return-v1"
SCHEMA_NEAR = "e01-near-baseline-bundle-v1"
SCHEMA_FOLLOWUP = "anonymous-followup-primitive-scalar-projection-v1"
SCHEMA_FAR = "e01-far-control-return-v1"
SCHEMA_OUTPUT = "e06-exposure-gradient-analysis-v1"

# ----------------------------------------------------------------------------------------
# Frozen population (identical to the original p5_main setting)
# ----------------------------------------------------------------------------------------
BLOCK_IDS = tuple(f"p5_main-block{index:03d}" for index in range(24))
BACKGROUNDS = (0, 1)
SEEDS = (0, 1, 2)
FOCALS = 8
CONTEXT_KEYS = tuple((block, background, seed) for block in BLOCK_IDS
                     for background in BACKGROUNDS for seed in SEEDS)
EXPOSURE_KEYS = tuple(key for key in CONTEXT_KEYS if key[1] == 0)
NEAR_SCORE_KEYS = ("audit_loss_drop", "base_difficulty_nll", "uncovered_base_nll_k4", "G20")
EPOCHS = (1, 2, 3, 4)
EPOCH_KEYS = tuple(str(epoch) for epoch in EPOCHS)
STEPS_PER_EPOCH = 96
WORKER_IDS = tuple(f"machine{w}" for w in range(1, 7))
WORKER_BLOCKS = {f"machine{w}": BLOCK_IDS[4 * (w - 1):4 * w] for w in range(1, 7)}  # spec section 9
FILLED_RECORD = "merge_filled_absent_record"
FILLED_CONTEXT = "merge_filled_absent_context"

FIT_ROLES = ("IN", "OUTNEAR", "G20NEAR")
FIT_STATUSES = ("pass", "fail", "missing")
IDENTITY_STATUSES = ("exact_match", "mismatch", "unavailable")
IDENTITY_TIER = {"IN": "D", "OUTNEAR": "H", "G20NEAR": "H"}
GRADIENT_STATUSES = ("pass", "fail", "unavailable")
WORKER_STATUSES = ("returned", "absent")
COMPONENTS = ("S_qq", "S_qc", "S_qUq", "S_qUc", "Nu_q", "Nu_c", "Ng_q", "S_qPq", "S_qPc")
GRADIENT_SCORES = ("Adam", "Plain", "Precond", "LESS_cosine")
OBSERVER = {"eta": 5e-05, "betas": [0.9, 0.999], "eps": 1e-08, "parameter_count": 405334016}
PLANNED_SCIENCE_FITS = {"IN": 144, "OUTNEAR": 576, "G20NEAR": 72}
MIN_AVAILABLE_BLOCKS = 20
DESCRIPTIVE_CONFIDENCE = 0.95
METRICS = ("spearman", "top_two_regret")

FAMILY = {
    "name": "E06_primary_family",
    "alpha": 0.05,
    "test_count": 8,
    "adjustment": "Bonferroni: each two-sided percentile interval at 1 - alpha/8 (0.99375)",
    "root_seed": 202610112300,
    "bootstrap_replicates": 20000,
    "endpoint_seed_rule": ("stable_seed(root_seed, endpoint_id): SHA-256 of the pipe-joined decimal/string "
                           "parts, first 8 bytes big-endian, modulo 2**31-1 (E01 analysis convention)."),
    "law": ("Python random.Random(endpoint_seed); per replicate draw n block indices with replacement by n "
            "successive randrange(n) calls over the sorted order of the n available blocks; statistic = "
            "equal-weight fsum mean of the drawn block values; seeds and focals are retained inside blocks."),
    "percentile": "Linear interpolation at index (n-1)*p with p = (1-confidence)/2 and 1-(1-confidence)/2.",
    "availability": ("A context contributes only if every fit it needs passed and every score it needs is "
                     "defined. Block value = fsum mean of its available context values; estimate = fsum mean "
                     "over the blocks with at least one available context. UNAVAILABLE if fewer than 20 of "
                     "the 24 blocks have one."),
    "unavailable_endpoints_stay_in_denominator": True,
    "separate_from": [
        "LM_primary: original 11-test family (5,000 draws, root 2026100205, 1-0.05/11)",
        "new_joint_E01_E02_E03_E04_main_family (20,000 draws, root 202610051754, 1-0.05/11)",
        "E01_far_control_primary_family (20,000 draws, root 202610101747, 1-0.05/5)",
    ],
}

ENDPOINTS = (
    {"number": 1, "id": "E06_X1_G20e1_vs_Ae1_rank", "arm": "exposure", "metric": "spearman",
     "first": "G20", "second": "A", "target": "T", "epoch": 1, "unit": "Spearman difference",
     "context_value": "rho(G20^(1), T^(1)) - rho(A^(1), T^(1))", "contexts_expected": 72,
     "positive_means": "after one pass the shared reference G20^(1) ranks benefit better than loss drop A^(1)",
     "negative_means": "after one pass loss drop A^(1) ranks benefit better than G20^(1)"},
    {"number": 2, "id": "E06_X2_U4_vs_D_rank_e1", "arm": "exposure", "metric": "spearman",
     "first": "U4", "second": "D", "target": "T", "epoch": 1, "unit": "Spearman difference",
     "context_value": "rho(U4, T^(1)) - rho(D, T^(1))", "contexts_expected": 72,
     "positive_means": "after one pass the corpus-aware U4 ranks benefit better than difficulty D",
     "negative_means": "after one pass difficulty D ranks benefit better than U4"},
    {"number": 3, "id": "E06_X3_Ae1_minus_G20e1_regret", "arm": "exposure", "metric": "top_two_regret",
     "first": "A", "second": "G20", "target": "T", "epoch": 1, "unit": "nats/token",
     "context_value": "R(A^(1), T^(1)) - R(G20^(1), T^(1))", "contexts_expected": 72,
     "positive_means": "after one pass G20^(1) has lower top-two regret than loss drop A^(1)",
     "negative_means": "after one pass loss drop A^(1) has lower top-two regret than G20^(1)"},
    {"number": 4, "id": "E06_X4_D_minus_U4_regret_e1", "arm": "exposure", "metric": "top_two_regret",
     "first": "D", "second": "U4", "target": "T", "epoch": 1, "unit": "nats/token",
     "context_value": "R(D, T^(1)) - R(U4, T^(1))", "contexts_expected": 72,
     "positive_means": "after one pass U4 has lower top-two regret than difficulty D",
     "negative_means": "after one pass difficulty D has lower top-two regret than U4"},
    {"number": 5, "id": "E06_Y1_G20_vs_Adam_rank", "arm": "attribution", "metric": "spearman",
     "first": "G20", "second": "Adam", "target": "T", "epoch": None, "unit": "Spearman difference",
     "context_value": "rho(G20, T) - rho(Adam, T)", "contexts_expected": 144,
     "positive_means": "the access-aware G20 ranks the published benefit target better than the Adam score",
     "negative_means": "the Adam-preconditioned gradient score ranks the published target better than G20"},
    {"number": 6, "id": "E06_Y2_U4_vs_Adam_rank", "arm": "attribution", "metric": "spearman",
     "first": "U4", "second": "Adam", "target": "T", "epoch": None, "unit": "Spearman difference",
     "context_value": "rho(U4, T) - rho(Adam, T)", "contexts_expected": 144,
     "positive_means": "the access-aware U4 ranks the published benefit target better than the Adam score",
     "negative_means": "the Adam-preconditioned gradient score ranks the published target better than U4"},
    {"number": 7, "id": "E06_Y3_Adam_minus_G20_regret", "arm": "attribution", "metric": "top_two_regret",
     "first": "Adam", "second": "G20", "target": "T", "epoch": None, "unit": "nats/token",
     "context_value": "R(Adam, T) - R(G20, T)", "contexts_expected": 144,
     "positive_means": "G20 has lower top-two regret than the Adam score",
     "negative_means": "the Adam score has lower top-two regret than G20"},
    {"number": 8, "id": "E06_Y4_Adam_minus_U4_regret", "arm": "attribution", "metric": "top_two_regret",
     "first": "Adam", "second": "U4", "target": "T", "epoch": None, "unit": "nats/token",
     "context_value": "R(Adam, T) - R(U4, T)", "contexts_expected": 144,
     "positive_means": "U4 has lower top-two regret than the Adam score",
     "negative_means": "the Adam score has lower top-two regret than U4"},
)
ENDPOINT_IDS = tuple(row["id"] for row in ENDPOINTS)

BRANCHES = {
    "exposure": {"greater": "HOLDS_AFTER_ONE_PASS", "less": "REVERSED_AFTER_ONE_PASS",
                 "unresolved": "UNRESOLVED_AFTER_ONE_PASS"},
    "attribution": {"greater": "ACCESS_AWARE_OUTRANKS_GRADIENT", "less": "GRADIENT_OUTRANKS_ACCESS_AWARE",
                    "unresolved": "UNRESOLVED"},
}
INTERPRETATION = {
    "HOLDS_AFTER_ONE_PASS": "Interval above zero: the ordering holds after one pass.",
    "REVERSED_AFTER_ONE_PASS": "Interval below zero: the ordering is reversed after one pass.",
    "UNRESOLVED_AFTER_ONE_PASS": "Interval contains zero: unresolved after one pass (not 'fails').",
    "ACCESS_AWARE_OUTRANKS_GRADIENT": "Interval above zero: the access-aware score outranks the "
                                      "preconditioned gradient score.",
    "GRADIENT_OUTRANKS_ACCESS_AWARE": "Interval below zero: the preconditioned gradient score outranks the "
                                      "access-aware score; reported as a favourable gradient result.",
    "UNRESOLVED": "Interval contains zero: unresolved.",
    "UNAVAILABLE": "Fewer than 20 of 24 blocks have an available context. Not a null; stays in the denominator.",
}


class ContractError(ValueError):
    """Structurally invalid, duplicated or identity-inconsistent input. Never an outcome."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


# ----------------------------------------------------------------------------------------
# Identities (byte-compatible with the original helpers; verbatim from analysis_e01.py)
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


def read_return_json(path):
    """E06 return reader: as ``read_json`` but NaN/Infinity literals parse to floats.

    Spec section 6 makes a non-finite observer component a per-context outcome (gradient scores
    unavailable, fit kept), not a halt, so the validator must see the value.  Non-finite readouts in a
    passing fit, and anything non-finite outside the observer components, still raise ContractError.
    """
    with Path(path).open(encoding="utf-8") as stream:
        return json.loads(stream.read(), object_pairs_hook=_unique_object, parse_constant=float)


# ----------------------------------------------------------------------------------------
# Core statistics: lm_support/statistics.py:11-72 (same arithmetic, same order; verbatim E01)
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


def targets(theta):
    result = {}
    for block in BLOCK_IDS:
        for background in BACKGROUNDS:
            by_seed = {seed: theta[(block, background, seed)] for seed in SEEDS}
            for seed in SEEDS:
                result[(block, background, seed)] = other_two_seed_target(by_seed, seed)
    return result


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


def validate_near_bundle_e06(bundle):
    """E01 validation plus the fields E06 reads and the bitwise identities the published values rest on."""
    index = validate_near_bundle(bundle)
    for key, row in index.items():
        cid = context_id(*key)
        for name in ("base_focal_nll", "focal_out_focal_nll", "all_out_focal_nll"):
            _vector8(row.get(name), f"{cid} {name}")
        jobs = row.get("historical_job_ids")
        require(isinstance(jobs, dict) and jobs.get("ALL_IN") == cid + "-all_in"
                and jobs.get("ALL_OUT") == cid + "-all_out"
                and jobs.get("FOCAL_OUT") == [f"{cid}-focal_out_{k}" for k in range(FOCALS)],
                f"{cid}: historical job ids differ from the sealed job pattern")
        positions = row.get("focal_member_positions")
        require(isinstance(positions, list) and len(positions) == FOCALS
                and all(type(value) is int for value in positions), f"{cid}: focal_member_positions")
        all_in, scores = row["all_in_focal_nll"], row["scores"]
        require(row["theta_near"] == [o - i for o, i in zip(row["focal_out_focal_nll"], all_in)],
                f"{cid}: theta_near is not FOCAL_OUT - ALL_IN bitwise")
        require(scores["G20"] == [o - i for o, i in zip(row["all_out_focal_nll"], all_in)],
                f"{cid}: G20 is not ALL_OUT - ALL_IN bitwise")
        require(scores["audit_loss_drop"] == [b - i for b, i in zip(row["base_focal_nll"], all_in)],
                f"{cid}: loss drop is not base - ALL_IN bitwise")
        require(scores["base_difficulty_nll"] == row["base_focal_nll"], f"{cid}: difficulty is not base NLL")
    return index


# ----------------------------------------------------------------------------------------
# Follow-up plain scores (background 0) and the E01 far-control return (descriptive inputs)
# ----------------------------------------------------------------------------------------
def validate_followup(document, near_index):
    """Background-0 ``p5_main-`` rows; the plain vector is in frozen focal positions."""
    require(isinstance(document, dict) and document.get("schema") == SCHEMA_FOLLOWUP, "unknown follow-up schema")
    rows = document.get("contexts")
    require(isinstance(rows, list), "follow-up contexts list required")
    index = {}
    for row in rows:
        require(isinstance(row, dict), "follow-up context must be an object")
        cid = row.get("context_id")
        if not (isinstance(cid, str) and cid.startswith("p5_main-")):
            continue
        key = (row.get("block_id"), row.get("background"), row.get("seed"))
        require(key in near_index and key[1] == 0, "follow-up p5_main context outside background 0: " + cid)
        require(cid == context_id(*key) and key not in index, "follow-up context id mismatch or duplicate: " + cid)
        near = near_index[key]
        require(row.get("focal_order") == list(range(FOCALS))
                and row.get("focal_member_positions") == near["focal_member_positions"],
                cid + ": follow-up focal order differs from the frozen focal positions")
        require(row.get("all_in_nll") == near["all_in_focal_nll"]
                and row.get("original_individual_effect") == near["theta_near"]
                and row.get("original_G20_score") == near["scores"]["G20"]
                and row.get("difficulty_nll") == near["scores"]["base_difficulty_nll"],
                cid + ": follow-up historical vectors differ bitwise from the near bundle")
        index[key] = _vector8(row.get("independent_plain_epoch_score"), cid + " independent_plain_epoch_score")
    require(set(index) == set(EXPOSURE_KEYS), "follow-up must hold all 72 background-0 p5_main contexts")
    return index


def validate_far_return(document, near_index):
    """E01 far-control return: per-focal NLL of IN, OUT_FAR_k and G20_FAR (E01 readout rules)."""
    require(isinstance(document, dict) and document.get("schema") == SCHEMA_FAR, "unknown E01 return schema")
    rows = document.get("contexts")
    require(isinstance(rows, list), "E01 return contexts list required")
    result = {}
    for row in rows:
        require(isinstance(row, dict), "E01 context must be an object")
        key = (row.get("block_id"), row.get("background"), row.get("seed_label"))
        require(key in near_index and key not in result, "E01 context outside the population or duplicated")
        near = near_index[key]
        require(row.get("context_id") == context_id(*key) and row.get("fit_seed") == near["fit_seed"]
                and row.get("focal_query_ids") == near["focal_query_ids"]
                and row.get("focal_member_ids") == near["focal_member_ids"],
                context_id(*key) + ": E01 context identity or focal order differs from the near bundle")
        record = {"IN": [None] * FOCALS, "OUT_FAR": [None] * FOCALS, "G20_FAR": [None] * FOCALS, "problems": []}
        seen = set()
        queries = row["focal_query_ids"]
        fits = row.get("fits")
        require(isinstance(fits, list), "E01 fits list required")
        for fit in fits:
            role, focal_index = fit.get("role"), fit.get("focal_index")
            require(role in ("IN", "OUT_FAR", "G20_FAR"), "unknown E01 fit role")
            require((type(focal_index) is int and 0 <= focal_index < FOCALS) if role == "OUT_FAR"
                    else focal_index is None, "E01 focal_index")
            require((role, focal_index) not in seen, "duplicate E01 fit record")
            seen.add((role, focal_index))
            if fit.get("status") != "completed":
                record["problems"].append(f"{role}{'' if focal_index is None else '_' + str(focal_index)}:"
                                          f"{fit.get('status')}")
                continue
            positions = [focal_index] if role == "OUT_FAR" else range(FOCALS)
            nll, tokens = fit.get("focal_nll"), fit.get("focal_scored_tokens")
            for position in positions:
                query = queries[position]
                ok = (isinstance(nll, dict) and isinstance(tokens, dict) and tokens.get(query) == 255
                      and finite(nll.get(query)))
                if ok:
                    record[role][position] = nll[query]
                else:
                    record["problems"].append(f"{role}:{query}:readout_unusable")
        result[key] = record
    return result


# ----------------------------------------------------------------------------------------
# Worker merge (E06_RETURN_SCHEMA.md: the analysis merges the six worker files)
# ----------------------------------------------------------------------------------------
E01_IN_SCHEMA = "e06-e01-in-epoch-readouts-v1"


def validate_e01_in_epochs(document, near_index):
    """D-08 fallback input: E01 IN-replay epoch readouts (8 focal NLLs per epoch) for the 144 contexts.

    Only contexts whose E01 IN replay was an exact historical match are eligible; for them the epoch-4
    readout must equal the published ALL_IN focal NLLs bit for bit.
    """
    require(isinstance(document, dict) and document.get("schema") == E01_IN_SCHEMA,
            "E01 IN epoch readouts: unexpected schema")
    by_id = {context_id(*key): key for key in near_index}
    index = {}
    for row in document.get("contexts", []):
        cid = row.get("context_id")
        require(cid in by_id and by_id[cid] not in index, f"E01 IN epoch readouts: unknown or duplicate {cid}")
        key = by_id[cid]
        require(row.get("focal_query_ids") == near_index[key]["focal_query_ids"],
                f"{cid}: E01 IN focal order differs from the near bundle")
        epochs = row.get("epoch_nll")
        require(isinstance(epochs, dict) and sorted(epochs) == list(EPOCH_KEYS),
                f"{cid}: E01 IN readouts need epochs 1-4")
        values = {int(e): _vector8(epochs[e], f"{cid} E01 IN epoch {e}") for e in EPOCH_KEYS}
        exact = row.get("identity_status") == "exact_match"
        if exact:
            require(values[4] == near_index[key]["all_in_focal_nll"],
                    f"{cid}: exact-match E01 IN epoch-4 readout differs from the published ALL_IN NLLs")
        index[key] = {"epochs": values, "exact": exact}
    require(set(index) == set(near_index), "E01 IN epoch readouts must cover all 144 contexts")
    return index


def _spec_worker(block):
    return next(worker for worker, blocks in WORKER_BLOCKS.items() if block in blocks)


def _context_key(row):
    key = (row.get("block_id"), row.get("background"), row.get("seed_label"))
    require(all(isinstance(part, (str, int)) for part in key), "context key fields must be scalars: " + repr(key))
    return key


def _slot_of(fit):
    """(role, focal_index) of a fit record, or None when the record is too malformed to name a slot."""
    if not isinstance(fit, dict) or not isinstance(fit.get("role"), str):
        return None
    focal_index = fit.get("focal_index")
    return (fit["role"], focal_index) if focal_index is None or type(focal_index) is int else None


def _filled_fit(near, slot):
    """Terminal ``missing`` record for a planned fit with no record in any worker projection."""
    role, focal_index = slot
    return {"role": role, "focal_index": focal_index, "job_id": _expected_job(near, role, focal_index),
            "status": "missing", "identity": {"status": "unavailable", "tier": IDENTITY_TIER[role]},
            "epoch_nll": None, "final_nll": None, "record_origin": FILLED_RECORD}


def _filled_context(near, key):
    """A planned context with no record in any worker projection: every fit missing, observer unavailable."""
    return {"context_id": context_id(*key), "block_id": key[0], "background": key[1], "seed_label": key[2],
            "fit_seed": near["fit_seed"], "focal_query_ids": list(near["focal_query_ids"]),
            "near_control_query_ids": [value + "-q-prefix" for value in near["near_control_ids"]],
            "fits": [_filled_fit(near, slot) for slot in _ordered_slots(key[1])],
            "gradient": {"status": "unavailable", "failure": {"type": "RECORD_ABSENT_IN_WORKER_PROJECTIONS"}},
            "record_origin": FILLED_CONTEXT}


def merge_worker_returns(documents, near_bundle, *, sources=None, absent_workers=()):
    """Merge the per-worker projections into E06_RETURN_ALL; no existing field is altered.

    The roster is fixed: machine1..machine6 must each be returned or declared absent.  A planned fit with no
    record in any projection (never started, or its worker returned nothing) is materialized as a terminal
    ``missing`` record marked ``record_origin``; a planned context with no record is materialized whole.
    Nothing else is filled, and every filled record is counted in the worker provenance.
    """
    near_index = validate_near_bundle_e06(near_bundle)
    require(isinstance(documents, list), "worker returns must be a list")
    absent_workers = list(absent_workers)
    releases, ids = set(), []
    for document in documents:
        require(isinstance(document, dict) and document.get("schema") == SCHEMA_RETURN, "unknown worker schema")
        require(isinstance(document.get("worker_id"), str) and document["worker_id"], "worker_id required")
        require(isinstance(document.get("release_id"), str) and document["release_id"], "release_id required")
        require(isinstance(document.get("flags"), list)
                and all(isinstance(flag, str) for flag in document["flags"]), "worker flags must be strings")
        require(isinstance(document.get("contexts"), list), "worker contexts list required")
        releases.add(document["release_id"])
        ids.append(document["worker_id"])
    require(len(releases) <= 1, "worker returns disagree on release_id")
    require(len(ids) + len(absent_workers) == len(WORKER_IDS) and set(ids) | set(absent_workers) == set(WORKER_IDS),
            "the merge needs the full roster machine1..machine6, each returned exactly once or declared absent")
    rows, owner = {}, {}
    for document in documents:
        for row in document["contexts"]:
            require(isinstance(row, dict), "worker context must be an object")
            key = _context_key(row)
            require(key in near_index and key not in rows,
                    "worker context outside the population or returned twice: " + repr(key))
            rows[key], owner[key] = row, document["worker_id"]
    filled = {worker: {"contexts": [], "fits": 0} for worker in WORKER_IDS}
    contexts = []
    for key in CONTEXT_KEYS:
        near, row = near_index[key], rows.get(key)
        if row is None:
            contexts.append(_filled_context(near, key))
            filled[_spec_worker(key[0])]["contexts"].append(context_id(*key))
            continue
        fits = row.get("fits")
        require(isinstance(fits, list), context_id(*key) + ": fits list required")
        present = {_slot_of(fit) for fit in fits}
        absent_slots = [slot for slot in _ordered_slots(key[1]) if slot not in present]
        if absent_slots:
            row = {**row, "fits": fits + [_filled_fit(near, slot) for slot in absent_slots]}
            filled[owner[key]]["fits"] += len(absent_slots)
        contexts.append(row)
    workers = []
    for worker in WORKER_IDS:
        index = ids.index(worker) if worker in ids else None
        document = None if index is None else documents[index]
        workers.append({
            "worker_id": worker, "status": "absent" if document is None else "returned",
            "flags": [] if document is None else list(document["flags"]),
            "context_ids": sorted(context_id(*key) for key, name in owner.items() if name == worker),
            "filled_context_ids": sorted(filled[worker]["contexts"]), "filled_fit_records": filled[worker]["fits"],
            "source_sha256": None if document is None or sources is None else sources[index],
            "provenance": {} if document is None else {
                key: value for key, value in document.items()
                if key not in ("schema", "worker_id", "release_id", "flags", "contexts")}})
    return {"schema": SCHEMA_RETURN, "release_id": releases.pop() if releases else "NO_WORKER_RETURNED",
            "flags": sorted({flag for document in documents for flag in document["flags"]}),
            "workers": workers, "contexts": contexts,
            "merge_note": "analysis_e06 merge: worker projections concatenated without altering any field; "
                          "planned records absent from every projection are materialized as terminal 'missing' "
                          "records marked record_origin and counted per worker; nothing else is filled"}


# ----------------------------------------------------------------------------------------
# E06 return (E06_RETURN_SCHEMA.md); hand-written validator
# ----------------------------------------------------------------------------------------
def _slot_label(role, focal_index):
    return role if focal_index is None else f"{role}_{focal_index}"


def _ordered_slots(background):
    """Planned science fits of a context in spec section 9 order (IN, OUTNEAR_0..7, G20NEAR; bg1: IN)."""
    if background != 0:
        return [("IN", None)]
    return [("IN", None)] + [("OUTNEAR", k) for k in range(FOCALS)] + [("G20NEAR", None)]


def _planned_slots(background):
    return set(_ordered_slots(background))


def _expected_job(near, role, focal_index):
    jobs = near["historical_job_ids"]
    if role == "IN":
        return jobs["ALL_IN"]
    if role == "G20NEAR":
        return jobs["ALL_OUT"]
    return jobs["FOCAL_OUT"][focal_index]


def _readout_map(mapping, queries, label):
    require(isinstance(mapping, dict) and len(mapping) == FOCALS and set(mapping) == set(queries),
            label + ": readout map must hold exactly the 8 focal query ids")
    values = [mapping[query] for query in queries]
    require(all(finite(value) for value in values), label + ": non-finite or non-numeric readout")
    return values


def _check_fit(fit, row, near):
    require(isinstance(fit, dict), row["context_id"] + ": fit record must be an object")
    role, focal_index = fit.get("role"), fit.get("focal_index")
    require(role in FIT_ROLES, row["context_id"] + ": unknown fit role")
    if role == "OUTNEAR":
        require(type(focal_index) is int and 0 <= focal_index < FOCALS, "OUTNEAR needs focal_index 0..7")
    else:
        require(focal_index is None, role + " must carry focal_index null")
    label = f"{row['context_id']}:{_slot_label(role, focal_index)}"
    require(fit.get("job_id") == _expected_job(near, role, focal_index), label + ": job_id is not the sealed job")
    status = fit.get("status")
    require(status in FIT_STATUSES, label + ": fit status")
    identity = fit.get("identity")
    tiers = (IDENTITY_TIER[role],) if status == "pass" else (IDENTITY_TIER[role], None)  # tier moot without a fit
    require(isinstance(identity, dict) and identity.get("status") in IDENTITY_STATUSES
            and identity.get("tier") in tiers, label + ": identity status or tier")
    origin = fit.get("record_origin")
    require(origin is None or (origin == FILLED_RECORD and status == "missing"), label + ": record_origin")
    record = {"role": role, "focal_index": focal_index, "status": status, "identity": identity["status"],
              "filled_by_merge": origin == FILLED_RECORD, "epoch": None, "final": None}
    if status == "pass":
        epochs = fit.get("epoch_nll")
        require(isinstance(epochs, dict) and set(epochs) == set(EPOCH_KEYS),
                label + ": a passing fit needs epoch_nll for epochs 1-4")
        queries = row["focal_query_ids"]
        record["epoch"] = {e: _readout_map(epochs[str(e)], queries, f"{label}:epoch {e}") for e in EPOCHS}
        record["final"] = _readout_map(fit.get("final_nll"), queries, label + ":final")
    else:
        require(fit.get("epoch_nll") is None and fit.get("final_nll") is None,
                label + ": a non-pass fit must not carry readouts")
    return (role, focal_index), record


def _json_safe(value):
    try:
        canonical(value)
    except (TypeError, ValueError):
        return "<non-finite or non-JSON content omitted>"
    return value


def _check_gradient(gradient, label, in_status):
    """Structural checks raise; observer qualification failures (spec section 6) return reasons."""
    require(isinstance(gradient, dict), label + ": every context needs a gradient object")
    status = gradient.get("status")
    require(status in GRADIENT_STATUSES, label + ": gradient status")
    require("failure" in gradient, label + ": gradient.failure must be present (null when passing)")
    result = {"declared": status, "components": None, "reasons": [], "failure": _json_safe(gradient["failure"])}
    if status != "pass":
        result["reasons"] = ["GRADIENT_STATUS_" + status.upper()]
        return result
    require(gradient["failure"] is None, label + ": a passing gradient carries a failure")
    for name, expected in OBSERVER.items():
        value = gradient.get(name)
        require(type(value) is type(expected) and value == expected,
                f"{label}: gradient {name} differs from the frozen observer recipe")
    reasons = []
    if in_status != "pass":
        reasons.append("IN_FIT_NOT_PASS")
    epochs, components = gradient.get("epochs"), {}
    if not (isinstance(epochs, dict) and set(epochs) == set(EPOCH_KEYS)):
        reasons.append("OBSERVER_EPOCHS_INCOMPLETE")
    else:
        for e in EPOCHS:
            records, by_focal = epochs[str(e)], {}
            if not isinstance(records, list):
                reasons.append(f"OBSERVER_EPOCH_{e}_NOT_A_LIST")
                continue
            for item in records:
                focal = item.get("focal_index") if isinstance(item, dict) else None
                if type(focal) is not int or not 0 <= focal < FOCALS or focal in by_focal:
                    reasons.append(f"OBSERVER_EPOCH_{e}_FOCAL_RECORD_MALFORMED_OR_DUPLICATED")
                    continue
                by_focal[focal] = item
            if len(records) != FOCALS or set(by_focal) != set(range(FOCALS)):
                reasons.append(f"OBSERVER_EPOCH_{e}_FOCALS_INCOMPLETE")
                continue
            for k in range(FOCALS):
                item = by_focal[k]
                step = item.get("optimizer_step")
                if not (type(step) is int and step == STEPS_PER_EPOCH * e):
                    reasons.append(f"OBSERVER_EPOCH_{e}_OPTIMIZER_STEP_NOT_{STEPS_PER_EPOCH * e}")
                if not all(finite(item.get(name)) for name in COMPONENTS):
                    reasons.append(f"OBSERVER_EPOCH_{e}_COMPONENT_NONFINITE_OR_MISSING")
                elif not item["S_qq"] > 0:
                    reasons.append(f"OBSERVER_EPOCH_{e}_S_qq_NOT_POSITIVE")
            components[e] = [by_focal[k] for k in range(FOCALS)]
    preservation = gradient.get("state_preservation")
    if not (isinstance(preservation, dict) and preservation.get("epochs_verified") == list(EPOCHS)):
        reasons.append("STATE_PRESERVATION_NOT_VERIFIED_FOR_EPOCHS_1_TO_4")
    result["reasons"] = sorted(set(reasons))
    if not result["reasons"]:
        result["components"], result["eta"] = components, gradient["eta"]
    return result


def _check_workers(document, filled_contexts):
    """Merge provenance: the fixed roster, who returned which contexts, and what the merge filled."""
    workers = document.get("workers")
    require(isinstance(workers, list) and [w.get("worker_id") if isinstance(w, dict) else None
                                           for w in workers] == list(WORKER_IDS),
            "merged return needs the workers provenance list machine1..machine6 in order")
    planned = {context_id(*key) for key in CONTEXT_KEYS}
    returned, filled, summary = set(), set(), []
    for worker in workers:
        require(worker.get("status") in WORKER_STATUSES, "worker status must be returned or absent")
        require(isinstance(worker.get("flags"), list) and all(isinstance(f, str) for f in worker["flags"]),
                "worker flags must be strings")
        lists = [worker.get("context_ids"), worker.get("filled_context_ids")]
        require(all(isinstance(ids, list) and all(isinstance(cid, str) for cid in ids) and set(ids) <= planned
                    and len(set(ids)) == len(ids) for ids in lists), "worker context id lists")
        require(not set(lists[0]) & returned and not set(lists[1]) & filled, "a context is claimed by two workers")
        require(worker["status"] == "returned" or not lists[0], "an absent worker cannot hold returned contexts")
        count = worker.get("filled_fit_records")
        require(type(count) is int and count >= 0, "worker filled_fit_records")
        require(worker.get("source_sha256") is None or _is_hex64(worker["source_sha256"]), "worker source_sha256")
        returned.update(lists[0])
        filled.update(lists[1])
        blocks = sorted({cid.rsplit("-b", 1)[0] for cid in lists[0]})
        summary.append({"worker_id": worker["worker_id"], "status": worker["status"], "flags": worker["flags"],
                        "contexts_returned": len(lists[0]), "contexts_filled_by_merge": len(lists[1]),
                        "fit_records_filled_by_merge": count, "blocks_returned": blocks,
                        "source_sha256": worker.get("source_sha256"),
                        "blocks_match_spec_section_9": (blocks == list(WORKER_BLOCKS[worker["worker_id"]])
                                                        if worker["status"] == "returned" else None)})
    require(not returned & filled and returned | filled == planned and filled == filled_contexts,
            "provenance must partition the 144 contexts into returned and merge-filled, matching record_origin")
    return summary


def validate_return(document, near_index):
    """Strict structural validation of the merged E06 return.

    Returns (contexts, workers, release).  ``contexts`` maps (block, background, seed) to the parsed fits and
    gradient qualification for all 144 planned contexts.  Structural contradictions raise ContractError; fit
    failures and observer qualification failures are retained as outcomes of availability, never imputed.
    """
    require(isinstance(document, dict) and document.get("schema") == SCHEMA_RETURN, "unknown return schema")
    require(isinstance(document.get("release_id"), str) and document["release_id"], "release_id required")
    require(isinstance(document.get("flags"), list) and all(isinstance(f, str) for f in document["flags"]),
            "flags must be a list of strings")
    rows = document.get("contexts")
    require(isinstance(rows, list), "contexts list required")
    contexts = {}
    for row in rows:
        require(isinstance(row, dict), "context must be an object")
        key = _context_key(row)
        require(key in near_index, "returned context outside the frozen population: " + repr(key))
        require(key not in contexts, "duplicate returned context: " + repr(key))
        cid = context_id(*key)
        require(row.get("context_id") == cid, "context_id mismatch: " + cid)
        near = near_index[key]
        require(type(row.get("fit_seed")) is int and row["fit_seed"] == near["fit_seed"],
                cid + ": fit_seed differs from the historical context")
        require(row.get("focal_query_ids") == near["focal_query_ids"],
                cid + ": focal query order differs from the frozen historical focal vector")
        require(row.get("near_control_query_ids") == [value + "-q-prefix" for value in near["near_control_ids"]],
                cid + ": near control query ids differ from the frozen near controls")
        fits = row.get("fits")
        require(isinstance(fits, list), cid + ": fits list required")
        seen = {}
        for fit in fits:
            slot, record = _check_fit(fit, row, near)
            require(slot not in seen, cid + ": duplicate fit record " + repr(slot))
            seen[slot] = record
        require(set(seen) == _planned_slots(key[1]),
                cid + ": fit records must be exactly the planned set (bg0: IN, OUTNEAR 0-7, G20NEAR; bg1: IN)")
        gradient = _check_gradient(row.get("gradient"), cid, seen[("IN", None)]["status"])
        origin = row.get("record_origin")
        require(origin is None or (origin == FILLED_CONTEXT and all(f["filled_by_merge"] for f in seen.values())
                                   and gradient["declared"] == "unavailable"), cid + ": context record_origin")
        contexts[key] = {"fits": seen, "gradient": gradient, "filled_by_merge": origin == FILLED_CONTEXT}
    require(set(contexts) == set(CONTEXT_KEYS),
            "the merged return must hold all 144 planned contexts (the merge materializes absent ones as missing)")
    workers = _check_workers(document, {context_id(*key) for key, row in contexts.items() if row["filled_by_merge"]})
    release = {"release_id": document["release_id"], "flags": list(document["flags"])}
    return contexts, workers, release


# ----------------------------------------------------------------------------------------
# Per-context quantities
# ----------------------------------------------------------------------------------------
def _difference(left, right):
    return None if left is None or right is None else left - right


def _vector(values, reasons):
    return {"values": values, "reasons": [] if _present(values) else sorted(set(reasons)) or ["undefined"]}


def gradient_scores(components, eta):
    """Scores summed over epochs 1-4 (fsum of per-epoch eta*(...), the follow-up plain arithmetic)."""
    scores, reasons = {name: [None] * FOCALS for name in GRADIENT_SCORES}, {name: [] for name in GRADIENT_SCORES}
    for k in range(FOCALS):
        records = [components[e][k] for e in EPOCHS]
        terms = {
            "Adam": [eta * (r["S_qUq"] - r["S_qUc"]) for r in records],
            "Plain": [eta * (r["S_qq"] - r["S_qc"]) for r in records],
            "Precond": [eta * (r["S_qPq"] - r["S_qPc"]) for r in records],
        }
        if all(r["Ng_q"] > 0 and r["Nu_q"] > 0 and r["Nu_c"] > 0 for r in records):
            terms["LESS_cosine"] = [eta * (r["S_qUq"] / (math.sqrt(r["Ng_q"]) * math.sqrt(r["Nu_q"]))
                                           - r["S_qUc"] / (math.sqrt(r["Ng_q"]) * math.sqrt(r["Nu_c"])))
                                    for r in records]
        else:
            reasons["LESS_cosine"].append(f"focal_{k}:LESS_NORM_NOT_POSITIVE")
        for name, values in terms.items():
            try:
                total = math.fsum(values)
            except (OverflowError, ValueError):
                total = math.nan
            if math.isfinite(total):
                scores[name][k] = total
            else:
                reasons[name].append(f"focal_{k}:SCORE_NONFINITE")
    return scores, reasons


def build_table(near_index, contexts, far_contexts, e01_in=None):
    """Every per-context vector the family and the descriptives use, with reasons for missing ones.

    D-08 (pre-registered fallback): when a background-0 context has no passing E06 IN replay, its IN epoch
    readouts come from the exact-match E01 IN replay of the same historical fit (``e01_in``). The source
    is recorded per context in ``IN_source``; the fallback is never used when the E06 IN replay passed.
    """
    table = {key: {} for key in CONTEXT_KEYS}
    hist_theta = {key: near_index[key]["theta_near"] for key in CONTEXT_KEYS}
    hist_target = targets(hist_theta)
    for key in CONTEXT_KEYS:
        scores = near_index[key]["scores"]
        row = table[key]
        row["theta"] = _vector(hist_theta[key], [])
        row["T"] = _vector(hist_target[key], [])
        row["G20"] = _vector(scores["G20"], [])
        row["A"] = _vector(scores["audit_loss_drop"], [])
        row["U4"] = _vector(scores["uncovered_base_nll_k4"], [])
        row["D"] = _vector(scores["base_difficulty_nll"], [])

    def fit_of(key, slot):
        record = contexts.get(key)
        return None if record is None else record["fits"].get(slot)

    def problem(key, slot):
        fit = fit_of(key, slot)
        status = "no_record" if fit is None else fit["status"] + (":filled_by_merge" if fit["filled_by_merge"] else "")
        return [] if status == "pass" else [f"{context_id(*key)}:{_slot_label(*slot)}:{status}"]

    def readout(key, slot, epoch):
        fit = fit_of(key, slot)
        return [None] * FOCALS if fit is None or fit["status"] != "pass" else list(fit["epoch"][epoch])

    def in_source(key):
        fit = fit_of(key, ("IN", None))
        if fit is not None and fit["status"] == "pass":
            return "E06"
        if e01_in is not None and e01_in.get(key, {}).get("exact"):
            return "E01_fallback"
        return "unavailable"

    for key in EXPOSURE_KEYS:
        table[key]["IN_source"] = in_source(key)

    for e in EPOCHS:
        theta_e = {}
        for key in CONTEXT_KEYS:
            if key[1] != 0:
                theta_e[key] = None
                continue
            row, base = table[key], near_index[key]["base_focal_nll"]
            if row["IN_source"] == "E01_fallback":
                in_values, in_problems = list(e01_in[key]["epochs"][e]), []
            else:
                in_values, in_problems = readout(key, ("IN", None), e), problem(key, ("IN", None))
            out_values = [readout(key, ("OUTNEAR", k), e)[k] for k in range(FOCALS)]
            g20_values = readout(key, ("G20NEAR", None), e)
            out_problems = [p for k in range(FOCALS) for p in problem(key, ("OUTNEAR", k))]
            theta = [_difference(o, i) for o, i in zip(out_values, in_values)]
            a_values = [_difference(b, i) for b, i in zip(base, in_values)]
            row[f"IN@{e}"] = _vector(in_values, in_problems)
            row[f"theta@{e}"] = _vector(theta, in_problems + out_problems)
            row[f"G20@{e}"] = _vector([_difference(g, i) for g, i in zip(g20_values, in_values)],
                                      in_problems + problem(key, ("G20NEAR", None)))
            row[f"A@{e}"] = _vector(a_values, in_problems)
            row[f"N@{e}"] = _vector([_difference(a, t) for a, t in zip(a_values, theta)], in_problems + out_problems)
            theta_e[key] = theta
        target_e = targets(theta_e)
        for key in EXPOSURE_KEYS:
            siblings = [(key[0], 0, s) for s in SEEDS if s != key[2]]
            reasons = [reason for sibling in siblings for reason in table[sibling][f"theta@{e}"]["reasons"]]
            table[key][f"T@{e}"] = _vector(target_e[key], reasons)

    for key in CONTEXT_KEYS:
        row = table[key]
        record = contexts.get(key)
        if record is None:
            qualification = {"declared": None, "components": None, "reasons": ["NO_CONTEXT_RECORD"]}
        else:
            qualification = record["gradient"]
        if qualification["components"] is None:
            for name in GRADIENT_SCORES:
                row[name] = _vector([None] * FOCALS, [f"{context_id(*key)}:gradient:{reason}"
                                                      for reason in qualification["reasons"]])
            continue
        scores, reasons = gradient_scores(qualification["components"], qualification["eta"])  # eta from the return
        for name in GRADIENT_SCORES:
            row[name] = _vector(scores[name], [f"{context_id(*key)}:gradient:{reason}" for reason in reasons[name]])

    theta_far, g20_far = {}, {}
    for key in CONTEXT_KEYS:
        record = far_contexts.get(key)
        problems = ["E01:" + context_id(*key) + ":context_absent"] if record is None else \
            ["E01:" + context_id(*key) + ":" + p for p in record["problems"]]
        if record is None:
            theta_far[key] = g20_far[key] = [None] * FOCALS
        else:
            theta_far[key] = [_difference(o, i) for o, i in zip(record["OUT_FAR"], record["IN"])]
            g20_far[key] = [_difference(g, i) for g, i in zip(record["G20_FAR"], record["IN"])]
        table[key]["theta_far"] = _vector(theta_far[key], problems)
        table[key]["G20_far"] = _vector(g20_far[key], problems)
    far_target = targets(theta_far)
    for key in CONTEXT_KEYS:
        siblings = [(key[0], key[1], s) for s in SEEDS if s != key[2]]
        reasons = [reason for sibling in siblings for reason in table[sibling]["theta_far"]["reasons"]]
        table[key]["T_far"] = _vector(far_target[key], reasons)
    return table


# ----------------------------------------------------------------------------------------
# Context values, block aggregation and endpoint inference
# ----------------------------------------------------------------------------------------
def _metric(metric, vector, target):
    if metric == "spearman":
        return spearman(vector, target)
    value = top_two_regret(vector, target)
    return value, None if value is not None else "unavailable_value"


def context_cell(row, first, second, target, metric):
    """metric(first, target) - metric(second, target) with the reasons for an undefined cell."""
    reasons = []
    for role, name in (("first", first), ("second", second), ("target", target)):
        vector = row[name]
        if not _present(vector["values"]):
            reasons += [f"{role}:{name}:missing:{reason}" for reason in vector["reasons"]]
    if reasons:
        return {"value": None, "first": None, "second": None, "reasons": reasons}
    target_values = row[target]["values"]
    a, reason_a = _metric(metric, row[first]["values"], target_values)
    b, reason_b = _metric(metric, row[second]["values"], target_values)
    for role, name, reason in (("first", first, reason_a), ("second", second, reason_b)):
        if reason:
            constant = [label for label, values in ((name, row[name]["values"]), (target, target_values))
                        if is_constant(values)]
            reasons.append(f"{role}:{name}:{reason}" + (":constant=" + ",".join(constant) if constant else ""))
    return {"value": None if a is None or b is None else a - b, "first": a, "second": b, "reasons": reasons}


def resolve(name, spec, epoch):
    """Vector name inside the table: epoch-resolved for the exposure arm (G20, A, T), else historical."""
    if spec["arm"] == "exposure" and name in ("G20", "A", "T") and epoch is not None:
        return f"{name}@{epoch}"
    return name


def endpoint_cells(table, spec, epoch):
    keys = EXPOSURE_KEYS if spec["arm"] == "exposure" else CONTEXT_KEYS
    names = tuple(resolve(spec[side], spec, epoch) for side in ("first", "second", "target"))
    return {key: context_cell(table[key], *names, spec["metric"]) for key in keys}, names


def block_means(values, keys):
    """[(block_id, block mean, contexts used)] over blocks with at least one defined value, sorted order."""
    result = []
    for block in BLOCK_IDS:
        defined = [values[key] for key in keys if key[0] == block and values[key] is not None]
        if defined:
            result.append((block, mean(defined), len(defined)))
    return result


def endpoint_branch(spec, direction):
    return "UNAVAILABLE" if direction is None else BRANCHES[spec["arm"]][direction]


def infer_endpoint(spec, table, *, replicates):
    keys = EXPOSURE_KEYS if spec["arm"] == "exposure" else CONTEXT_KEYS
    confidence = 1 - FAMILY["alpha"] / FAMILY["test_count"]
    seed = stable_seed(FAMILY["root_seed"], spec["id"])
    cells, names = endpoint_cells(table, spec, spec["epoch"])
    values = {key: cells[key]["value"] for key in keys}
    blocks = block_means(values, keys)
    available = [key for key in keys if values[key] is not None]
    record = {**spec, "family": FAMILY["name"], "family_denominator": FAMILY["test_count"],
              "confidence": confidence, "endpoint_seed": seed, "replicates": replicates,
              "vectors": dict(zip(("first", "second", "target"), names)),
              "contexts_available": len(available), "blocks_available": len(blocks),
              "blocks_required": MIN_AVAILABLE_BLOCKS, "complete_grid": len(available) == len(keys),
              "status": None, "estimate": None, "interval": None, "direction_resolved": None, "branch": None,
              "block_values": None, "components": None, "index_stream_sha256": None, "input_cells_sha256": None,
              "undefined_cells": [{"context_id": context_id(*key), "reasons": cells[key]["reasons"]}
                                  for key in keys if values[key] is None]}
    if len(blocks) < MIN_AVAILABLE_BLOCKS:
        record.update(status="unavailable", branch="UNAVAILABLE", interpretation=INTERPRETATION["UNAVAILABLE"])
        return record
    block_values = [value for _, value, _ in blocks]
    record["block_values"] = [
        {"block_id": block, "value": value, "contexts_used": used,
         "context_values": {context_id(*key): values[key] for key in keys if key[0] == block}}
        for block, value, used in blocks]
    record["input_cells_sha256"] = digest([[values[key] for key in keys if key[0] == block] for block in BLOCK_IDS])
    record["estimate"] = mean(block_values)
    record["interval"], record["index_stream_sha256"] = bootstrap_interval(
        block_values, seed=seed, replicates=replicates, confidence=confidence)
    record["direction_resolved"] = direction_of(record["interval"])
    record["status"] = "available" if record["complete_grid"] else "available_context_estimate_not_complete_grid"
    record["branch"] = endpoint_branch(spec, record["direction_resolved"])
    record["interpretation"] = INTERPRETATION[record["branch"]]
    record["components"] = {
        side: {"vector": names[index], "metric": spec["metric"],
               "estimate": mean([value for _, value, _ in block_means(
                   {key: cells[key][side] if values[key] is not None else None for key in keys}, keys)])}
        for index, side in enumerate(("first", "second"))}
    return record


# ----------------------------------------------------------------------------------------
# Descriptive diagnostics (outside the family; nominal 95%; never used for branches)
# ----------------------------------------------------------------------------------------
def describe(values, keys, item_id, replicates, **extra):
    """Available-context block mean with a nominal 95% whole-block interval (same 20-block rule)."""
    blocks = block_means(values, keys)
    available = sum(values[key] is not None for key in keys)
    record = {"item": item_id, "units_expected": len(keys), "units_available": available,
              "blocks_available": len(blocks), "complete_grid": available == len(keys), **extra}
    if len(blocks) < MIN_AVAILABLE_BLOCKS:
        record.update(status="unavailable_fewer_than_20_blocks", estimate=None, interval=None)
        return record
    seed = stable_seed(FAMILY["root_seed"], "e06-descriptive", item_id)
    block_values = [value for _, value, _ in blocks]
    interval, stream = bootstrap_interval(block_values, seed=seed, replicates=replicates,
                                          confidence=DESCRIPTIVE_CONFIDENCE)
    record.update(status="available_descriptive", estimate=mean(block_values), interval=interval,
                  confidence=DESCRIPTIVE_CONFIDENCE, seed=seed, index_stream_sha256=stream)
    return record


def _point_estimate(values, keys):
    blocks = block_means(values, keys)
    return mean([value for _, value, _ in blocks]) if len(blocks) >= MIN_AVAILABLE_BLOCKS else None


def _absolute_values(table, keys, score, target, metric):
    result = {}
    for key in keys:
        row = table[key]
        if not (_present(row[score]["values"]) and _present(row[target]["values"])):
            result[key] = None
        else:
            result[key] = _metric(metric, row[score]["values"], row[target]["values"])[0]
    return result


def _context_mean(table, keys, name):
    return {key: mean(table[key][name]["values"]) if _present(table[key][name]["values"]) else None
            for key in keys}


def _repeatability(table, name, item_id, replicates):
    """Mean pairwise cross-seed Spearman of a background-0 vector inside each block (units are seed pairs)."""
    values, keys = {}, []
    for block in BLOCK_IDS:
        for left, right in ((0, 1), (0, 2), (1, 2)):
            unit = (block, left, right)
            a, b = table[(block, 0, left)][name]["values"], table[(block, 0, right)][name]["values"]
            values[unit] = spearman(a, b)[0] if _present(a) and _present(b) else None
            keys.append(unit)
    return describe(values, keys, item_id, replicates, unit="seed pair within block (3 per block)")


def descriptive(table, *, replicates):
    out = {"label": "descriptive_outside_family_nominal_95_not_multiplicity_adjusted",
           "confidence": DESCRIPTIVE_CONFIDENCE,
           "seed_rule": f"stable_seed({FAMILY['root_seed']}, 'e06-descriptive', item)",
           "availability": "same available-context rule and 20-block threshold as the family"}
    contrasts = {}
    for e in (2, 3, 4):
        contrasts[f"e{e}"] = {}
        for spec in ENDPOINTS[:4]:
            cells, names = endpoint_cells(table, spec, e)
            values = {key: cell["value"] for key, cell in cells.items()}
            item = describe(values, EXPOSURE_KEYS, f"exposure_contrast.X{spec['number']}.e{e}", replicates,
                            context_value=spec["context_value"].replace("(1)", f"({e})"),
                            vectors=dict(zip(("first", "second", "target"), names)))
            if e == 4:
                history, _ = endpoint_cells(table, spec, None)
                hist_values = {key: cell["value"] for key, cell in history.items()}
                item["historical_bg0_estimate"] = _point_estimate(hist_values, EXPOSURE_KEYS)
                item["replay_equals_history"] = (all(values[key] == hist_values[key] for key in EXPOSURE_KEYS)
                                                 and item["estimate"] == item["historical_bg0_estimate"])
            contrasts[f"e{e}"][f"X{spec['number']}"] = item
    out["exposure_contrasts"] = contrasts
    components = {}
    for e in EPOCHS:
        components[f"e{e}"] = {}
        for score in ("G20", "A", "U4", "D"):
            name = f"{score}@{e}" if score in ("G20", "A") else score
            components[f"e{e}"][score] = {
                metric: describe(_absolute_values(table, EXPOSURE_KEYS, name, f"T@{e}", metric), EXPOSURE_KEYS,
                                 f"exposure_component.{score}.{metric}.e{e}", replicates,
                                 vectors={"score": name, "target": f"T@{e}"})
                for metric in METRICS}
            if e == 4:
                for metric in METRICS:
                    hist = _absolute_values(table, EXPOSURE_KEYS, score, "T", metric)
                    components["e4"][score][metric]["historical_bg0_estimate"] = _point_estimate(hist, EXPOSURE_KEYS)
    out["exposure_components"] = components
    out["theta_and_nonfocal_means"] = {
        **{f"theta.e{e}": describe(_context_mean(table, EXPOSURE_KEYS, f"theta@{e}"), EXPOSURE_KEYS,
                                   f"theta_mean.e{e}", replicates, unit_of_value="nats/token, focal mean")
           for e in EPOCHS},
        **{f"N.e{e}": describe(_context_mean(table, EXPOSURE_KEYS, f"N@{e}"), EXPOSURE_KEYS,
                               f"nonfocal_mean.e{e}", replicates,
                               definition="N^(e) = A^(e) - theta^(e) = NLL(base) - NLL^(e)(OUTNEAR_k)")
           for e in EPOCHS},
        "theta.historical_bg0": describe(_context_mean(table, EXPOSURE_KEYS, "theta"), EXPOSURE_KEYS,
                                         "theta_mean.historical_bg0", replicates),
    }
    out["cross_seed_repeatability"] = {
        **{f"theta.e{e}": _repeatability(table, f"theta@{e}", f"repeatability.e{e}", replicates) for e in EPOCHS},
        "theta.historical_bg0": _repeatability(table, "theta", "repeatability.historical_bg0", replicates),
    }
    scopes = {"background_0": EXPOSURE_KEYS, "background_1": tuple(k for k in CONTEXT_KEYS if k[1] == 1),
              "pooled": CONTEXT_KEYS}
    attribution = {}
    for scope, keys in scopes.items():
        attribution[scope] = {}
        for score in GRADIENT_SCORES + ("G20", "U4", "A", "D"):
            attribution[scope][score] = {
                metric: describe(_absolute_values(table, keys, score, "T", metric), keys,
                                 f"attribution.{scope}.{score}.{metric}", replicates,
                                 role="gradient score" if score in GRADIENT_SCORES
                                 else "historical reference over all its contexts")
                for metric in METRICS}
    out["attribution_absolute"] = attribution
    out["far_targets"] = {
        score: {metric: describe(_absolute_values(table, CONTEXT_KEYS, score, "T_far", metric), CONTEXT_KEYS,
                                 f"far.{score}.{metric}", replicates,
                                 target="E01 T_far: other-two-seed mean of NLL(OUT_FAR_k) - NLL(IN), same background")
                for metric in METRICS}
        for score in GRADIENT_SCORES + ("G20_far", "U4")}
    return out


# ----------------------------------------------------------------------------------------
# Plain reproduction and identity summaries (reported; never gates)
# ----------------------------------------------------------------------------------------
def plain_reproduction(table, followup_index):
    compared, identical, rhos, rel_max, rel_at = 0, 0, [], None, None
    cells = equal = zero_reference = 0
    per_context = []
    for key in EXPOSURE_KEYS:
        mine, theirs = table[key]["Plain"]["values"], followup_index[key]
        if not _present(mine):
            per_context.append({"context_id": context_id(*key), "status": "e06_plain_unavailable"})
            continue
        compared += 1
        same_ranks = ranks(mine) == ranks(theirs)
        identical += same_ranks
        rho = spearman(mine, theirs)[0]
        rhos.append(rho)
        context_rel = []
        for k, (a, b) in enumerate(zip(mine, theirs)):
            cells += 1
            equal += a == b
            if b == 0:
                zero_reference += a != b
                continue
            rel = abs(a - b) / abs(b)
            context_rel.append(rel)
            if rel_max is None or rel > rel_max:
                rel_max, rel_at = rel, {"context_id": context_id(*key), "focal_index": k}
        per_context.append({"context_id": context_id(*key), "identical_ranks": same_ranks,
                            "within_context_spearman": rho, "max_relative_difference": max(context_rel, default=None),
                            "cells_bitwise_equal": sum(a == b for a, b in zip(mine, theirs))})
    defined = [rho for rho in rhos if rho is not None]
    status = ("no_context_compared" if not compared else "bitwise_identical" if equal == cells
              else "ranks_identical_values_differ" if identical == compared else "ranks_differ")
    return {"gate": False, "label": "reported, not a gate (spec section 6)",
            "definition": "E06 Plain = fsum_e eta*(S_qq - S_qc) versus follow-up independent_plain_epoch_score",
            "contexts_expected": len(EXPOSURE_KEYS), "contexts_compared": compared, "cells_compared": cells,
            "cells_bitwise_equal": equal, "max_relative_difference": rel_max, "max_relative_difference_at": rel_at,
            "zero_reference_cells_with_difference": zero_reference,
            "contexts_with_identical_within_context_ranks": identical,
            "fraction_contexts_identical_ranks": identical / compared if compared else None,
            "mean_within_context_spearman": mean(defined) if defined else None,
            "min_within_context_spearman": min(defined) if defined else None,
            "status": status, "per_context": per_context}


def identity_summary(near_index, contexts, table, document_flags, workers):
    fits = {role: {status: 0 for status in FIT_STATUSES} for role in FIT_ROLES}
    filled = {role: 0 for role in FIT_ROLES}
    declared = {role: {status: 0 for status in IDENTITY_STATUSES} for role in FIT_ROLES}
    checks = {name: {"checked": 0, "equal": 0, "differing": []} for name in (
        "IN_final_equals_historical_all_in", "IN_epoch4_equals_final",
        "OUTNEAR_final_equals_historical_focal_out", "OUTNEAR_epoch4_equals_final",
        "G20NEAR_final_equals_historical_all_out", "G20NEAR_epoch4_equals_final")}

    def check(name, label, equal_flag):
        checks[name]["checked"] += 1
        checks[name]["equal"] += bool(equal_flag)
        if not equal_flag:
            checks[name]["differing"].append(label)

    declared_exact_but_differs = []
    for key in CONTEXT_KEYS:
        record = contexts[key]
        for slot in _ordered_slots(key[1]):
            role, k = slot
            fit = record["fits"][slot]
            fits[role][fit["status"]] += 1
            filled[role] += fit["filled_by_merge"]
            declared[role][fit["identity"]] += 1
            if fit["status"] != "pass":
                continue
            near, label = near_index[key], f"{context_id(*key)}:{_slot_label(role, k)}"
            epoch4_equal = fit["epoch"][4] == fit["final"]
            if role == "IN":
                final_equal = fit["final"] == near["all_in_focal_nll"]
            elif role == "OUTNEAR":
                final_equal = fit["final"][k] == near["focal_out_focal_nll"][k]
            else:
                final_equal = fit["final"] == near["all_out_focal_nll"]
            check(f"{role}_final_equals_historical_" + {"IN": "all_in", "OUTNEAR": "focal_out",
                                                        "G20NEAR": "all_out"}[role], label, final_equal)
            check(f"{role}_epoch4_equals_final", label, epoch4_equal)
            if fit["identity"] == "exact_match" and not (final_equal and epoch4_equal):
                declared_exact_but_differs.append(label)
    replay = {}
    for name, historical in (("theta@4", "theta"), ("T@4", "T"), ("G20@4", "G20"), ("A@4", "A")):
        available = [key for key in EXPOSURE_KEYS if _present(table[key][name]["values"])]
        replay[f"{name}_equals_historical_{historical}"] = {
            "contexts_checked": len(available),
            "contexts_equal": sum(table[key][name]["values"] == table[key][historical]["values"] for key in available)}
    gradient = {"declared": {status: 0 for status in GRADIENT_STATUSES}, "qualified": 0,
                "unqualified": [], "declared_pass_failed_analysis_qualification": [],
                "S_qq_vs_Ng_q_max_relative_difference": None}
    for key in CONTEXT_KEYS:
        qualification = contexts[key]["gradient"]
        gradient["declared"][qualification["declared"]] += 1
        if qualification["components"] is None:
            gradient["unqualified"].append({"context_id": context_id(*key), "reasons": qualification["reasons"],
                                            "failure": qualification["failure"]})
            if qualification["declared"] == "pass":
                gradient["declared_pass_failed_analysis_qualification"].append(context_id(*key))
            continue
        gradient["qualified"] += 1
        for e in EPOCHS:
            for item in qualification["components"][e]:
                rel = abs(item["S_qq"] - item["Ng_q"]) / item["S_qq"]
                current = gradient["S_qq_vs_Ng_q_max_relative_difference"]
                gradient["S_qq_vs_Ng_q_max_relative_difference"] = rel if current is None else max(current, rel)
    return {"fits_by_role_and_status": fits, "planned_science_fits": PLANNED_SCIENCE_FITS,
            "missing_fit_records_filled_by_merge_by_role": filled,
            "contexts_filled_by_merge": sorted(context_id(*key) for key, row in contexts.items()
                                               if row["filled_by_merge"]),
            "declared_identity_by_role": declared, "return_flags": document_flags,
            "worker_flags": {row["worker_id"]: row["flags"] for row in workers},
            "analysis_checks_float64_equality": checks,
            "declared_exact_match_but_analysis_check_differs": declared_exact_but_differs,
            "epoch4_replay_against_history_background_0": replay, "gradient_observer": gradient}


def costs(document):
    """Spec section 7 'costs'.  Timing fields lie outside the frozen return schema: reported when present."""
    fit_seconds = {role: [] for role in FIT_ROLES}
    observer_seconds, without = [], {"fits": 0, "observers": 0}
    for row in document["contexts"]:
        for fit in row["fits"]:
            value = fit.get("exclusive_action_seconds")
            if finite(value) and value >= 0:
                fit_seconds[fit["role"]].append(value)
            elif fit.get("status") == "pass":
                without["fits"] += 1
        value = row["gradient"].get("seconds_total")
        if finite(value) and value >= 0:
            observer_seconds.append(value)
        elif row["gradient"].get("status") == "pass":
            without["observers"] += 1

    def summary(values):
        return {"count": len(values), "total_hours": math.fsum(values) / 3600 if values else None,
                "mean_minutes": math.fsum(values) / len(values) / 60 if values else None}

    return {"status": "reported_when_present_outside_the_return_schema",
            "fit_exclusive_seconds_by_role": {role: summary(values) for role, values in fit_seconds.items()},
            "observer_seconds_total": summary(observer_seconds),
            "passing_records_without_timing": without}


# ----------------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------------
def analyze(e06_return, near_bundle, followup, far_return, *, replicates=None, source_hashes=None,
            include_descriptive=True, e01_in_epochs=None):
    """Full E06 analysis.  ``replicates`` may be lowered only in synthetic tests.

    ``e01_in_epochs`` is the D-08 fallback input (required by the command line)."""
    replicates = FAMILY["bootstrap_replicates"] if replicates is None else replicates
    near_index = validate_near_bundle_e06(near_bundle)
    followup_index = validate_followup(followup, near_index)
    far_contexts = validate_far_return(far_return, near_index)
    contexts, workers, release = validate_return(e06_return, near_index)
    e01_in = None if e01_in_epochs is None else validate_e01_in_epochs(e01_in_epochs, near_index)
    table = build_table(near_index, contexts, far_contexts, e01_in)
    endpoints = [infer_endpoint(spec, table, replicates=replicates) for spec in ENDPOINTS]
    result = {
        "schema": SCHEMA_OUTPUT,
        "status": ("frozen_law_output" if replicates == FAMILY["bootstrap_replicates"]
                   else "NONFROZEN_REPLICATE_COUNT_TEST_ONLY_NOT_THE_E06_LAW"),
        "family": FAMILY,
        "parameters": {"root_seed": FAMILY["root_seed"], "replicates": replicates,
                       "family_confidence": 1 - FAMILY["alpha"] / FAMILY["test_count"],
                       "descriptive_confidence": DESCRIPTIVE_CONFIDENCE,
                       "min_available_blocks": MIN_AVAILABLE_BLOCKS, "epochs": list(EPOCHS),
                       "observer_recipe": OBSERVER,
                       "endpoint_seeds": {spec["id"]: stable_seed(FAMILY["root_seed"], spec["id"])
                                          for spec in ENDPOINTS}},
        "population": {"blocks": list(BLOCK_IDS), "backgrounds": list(BACKGROUNDS), "seed_labels": list(SEEDS),
                       "focals_per_context": FOCALS, "exposure_arm_contexts": len(EXPOSURE_KEYS),
                       "attribution_arm_contexts": len(CONTEXT_KEYS), "inference_unit": "original block",
                       "weighting": "equal available contexts within block, equal available blocks"},
        "release": release,
        "workers": workers,
        "source_hashes": source_hashes or {},
        "endpoints": endpoints,
        "branches": {row["id"]: row["branch"] for row in endpoints},
        "interpretation_limits": [
            "X1-X4: interval above zero = the ordering holds after one pass; containing zero = unresolved "
            "after one pass (not 'fails'); below zero = reversed after one pass.",
            "Y1-Y4 are two-sided; a favourable gradient result is reported as such.",
            "Unavailability is not a scientific null; unavailable endpoints stay in the denominator of 8.",
            "Nothing here is evidence about larger models, other data sizes, or downstream audit utility.",
        ],
        "accounting": {
            "contexts_returned_by_workers": sum(not row["filled_by_merge"] for row in contexts.values()),
            "d08_in_fallback": {"input_supplied": e01_in is not None,
                                "contexts_using_e01_in_readouts": sorted(
                                    context_id(*key) for key in EXPOSURE_KEYS
                                    if table[key]["IN_source"] == "E01_fallback"),
                                "contexts_without_in_readouts": sorted(
                                    context_id(*key) for key in EXPOSURE_KEYS
                                    if table[key]["IN_source"] == "unavailable")},
            "identity_summary": identity_summary(near_index, contexts, table, release["flags"], workers),
        },
    }
    if include_descriptive:
        result["descriptive_outside_family"] = descriptive(table, replicates=replicates)
        result["descriptive_outside_family"]["plain_reproduction"] = plain_reproduction(table, followup_index)
        result["descriptive_outside_family"]["costs"] = costs(e06_return)
    return result


# ----------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------
def _write_new(path, document, *, allow_nan=False):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=1, sort_keys=True, allow_nan=allow_nan) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="E06 exposure-gradient analysis (frozen law).")
    parser.add_argument("--return", dest="return_path", help="merged e06-exposure-gradient-return-v1 JSON")
    parser.add_argument("--near", help="e01-near-baseline-bundle-v1 JSON (NEAR_INPUT_BUNDLE.json)")
    parser.add_argument("--followup", help="follow-up CONTEXTS.json (independent_plain_epoch_score)")
    parser.add_argument("--far", help="E01 far-control return E01_RETURN_ALL.json")
    parser.add_argument("--e01-in-epochs", help="D-08 fallback: E01_IN_EPOCH_READOUTS.json")
    parser.add_argument("--output", required=True, help="new output path; never overwritten")
    parser.add_argument("--replicates", type=int, default=FAMILY["bootstrap_replicates"],
                        help="bootstrap draws; anything but 20000 is labelled test-only")
    parser.add_argument("--merge-workers", nargs="*", default=None,
                        help="merge mode: the returned per-worker E06_RETURN_machineN.json files")
    parser.add_argument("--absent-worker", action="append", default=[],
                        help="merge mode: a worker that returned no projection (repeatable)")
    args = parser.parse_args(argv)
    if args.merge_workers is not None:
        if any(value is not None for value in (args.return_path, args.followup, args.far,
                                                args.e01_in_epochs)) or args.near is None:
            parser.error("merge mode takes --merge-workers, --absent-worker, --near and --output only")
        documents = [read_return_json(path) for path in args.merge_workers]
        merged = merge_worker_returns(documents, read_json(args.near),
                                      sources=[file_sha256(path) for path in args.merge_workers],
                                      absent_workers=args.absent_worker)
        _write_new(args.output, merged, allow_nan=True)
        print(json.dumps({"output": args.output, "workers": [
            [row["worker_id"], row["status"], len(row["filled_context_ids"]), row["filled_fit_records"]]
            for row in merged["workers"]]}))
        return
    if args.absent_worker:
        parser.error("--absent-worker is a merge-mode option")
    if any(value is None for value in (args.return_path, args.near, args.followup, args.far, args.e01_in_epochs)):
        parser.error("analysis mode needs --return, --near, --followup, --far and --e01-in-epochs")
    hashes = {"return": file_sha256(args.return_path), "near": file_sha256(args.near),
              "followup": file_sha256(args.followup), "far": file_sha256(args.far),
              "e01_in_epochs": file_sha256(args.e01_in_epochs), "analysis_e06.py": file_sha256(__file__)}
    result = analyze(read_return_json(args.return_path), read_json(args.near), read_json(args.followup),
                     read_json(args.far), replicates=args.replicates, source_hashes=hashes,
                     e01_in_epochs=read_json(args.e01_in_epochs))
    _write_new(args.output, result)
    print(json.dumps({"output": args.output, "status": result["status"], "branches": result["branches"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
