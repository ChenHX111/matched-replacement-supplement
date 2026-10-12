"""Portable block-bootstrap arithmetic. No models, datasets, paths, or neural dependencies."""
from __future__ import annotations

import hashlib
import itertools
import math
import os
import random
import statistics

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np


def mean(values):
    return math.fsum(values) / len(values)


def stable_seed(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2
        for i in order[start:end]:
            result[i] = rank
        start = end
    return result


def rank_correlation(left, right):
    if any(value is None or not math.isfinite(value) for value in list(left) + list(right)):
        return None
    a, b = ranks(left), ranks(right)
    a, b = [v - mean(a) for v in a], [v - mean(b) for v in b]
    denominator = math.sqrt(math.fsum(v*v for v in a) * math.fsum(v*v for v in b))
    return math.fsum(x*y for x, y in zip(a, b)) / denominator if denominator else None


def top_two_regret(prediction, oracle):
    if len(prediction) != 8 or len(oracle) != 8:
        raise ValueError("The fixed selector needs exactly eight ordered focals")
    if any(value is None or not math.isfinite(value) for value in list(prediction) + list(oracle)):
        return None
    chosen = sorted(range(8), key=lambda i: (-prediction[i], i))[:2]
    best = sorted(range(8), key=lambda i: (-oracle[i], i))[:2]
    return mean([oracle[i] for i in best]) - mean([oracle[i] for i in chosen])


def p5_arrays(setting, method):
    shape = (len(setting["blocks"]), 6, 8)
    oracle = np.asarray(setting["oracle"], dtype=float).reshape(shape)
    prediction = np.asarray(setting["scores"][method], dtype=float).reshape(shape)
    reference = oracle.copy()
    for background, seed in itertools.product((0, 1), (0, 1, 2)):
        other = [background * 3 + value for value in (0, 1, 2) if value != seed]
        reference[:, background * 3 + seed] = np.mean(oracle[:, other], axis=1)
    return prediction, reference


def p5_metric_cells(setting, method):
    prediction, oracle = p5_arrays(setting, method)
    rank = np.asarray([
        [rank_correlation(p.tolist(), o.tolist()) for p, o in zip(pb, ob)]
        for pb, ob in zip(prediction, oracle)
    ], dtype=float)
    regret = np.asarray([
        [top_two_regret(p.tolist(), o.tolist()) for p, o in zip(pb, ob)]
        for pb, ob in zip(prediction, oracle)
    ], dtype=float)
    return {
        "spearman": rank, "top_two_regret": regret,
        "median_absolute_error": np.abs(prediction - oracle).reshape(len(setting["blocks"]), -1),
    }


def primary_cells(inputs, specification):
    setting = specification["setting"]
    endpoint = specification["endpoint"]
    if specification["pilot"] == "P4":
        p4 = inputs["p4"][setting]
        values = np.asarray(p4["endpoints"][endpoint], dtype=float)
        return values, None, "mean", p4["strata"]
    p5 = inputs["p5"][setting]
    left = p5_metric_cells(p5, specification["left"])[endpoint]
    if specification.get("right") is None:
        return left - specification["null"], None, "mean", None
    right = p5_metric_cells(p5, specification["right"])[endpoint]
    if endpoint == "median_absolute_error":
        return right, left, "median", None
    return (left-right if endpoint == "spearman" else right-left), None, "mean", None


def statistic(values, kind):
    if kind == "mean":
        return mean([mean(row.tolist()) for row in values])
    if kind == "median":
        return statistics.median(values.ravel().tolist())
    raise ValueError(kind)


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered)-1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high]-ordered[low]) * (position-low)


def tuple_state(value):
    return tuple(tuple_state(child) for child in value) if isinstance(value, list) else value


def bootstrap(values, *, subtract=None, kind="mean", confidence=.95, seed=0,
              replicates=5000, strata=None, checkpoint=None, resume=None, chunk=1000):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    other = None if subtract is None else np.asarray(subtract, dtype=float).reshape(values.shape)
    finite = np.isfinite(values).all() and (other is None or np.isfinite(other).all())
    result = {
        "estimate": None, "interval": None, "clusters": len(values),
        "total_cells": int(values.size), "full_metric_defined": bool(finite),
        "undefined_cells_count": int(np.sum(~np.isfinite(values))) +
        (0 if other is None else int(np.sum(~np.isfinite(other)))),
        "statistic": "median_abs_error_right_minus_median_abs_error_left"
        if kind == "median" and other is not None else kind,
        "confidence": confidence, "replicates": replicates, "cluster_seed": seed,
    }
    if not finite:
        result["status"] = "undefined_cells_retained_no_complete_inference"
        return result
    result["estimate"] = statistic(values, kind) - (statistic(other, kind) if other is not None else 0)
    labels = strata if strata is not None else ["all"] * len(values)
    groups = [[i for i in range(len(values)) if labels[i] == label] for label in sorted(set(labels))]
    if len(labels) != len(values) or any(len(group) < 2 for group in groups):
        result["status"] = "insufficient_clusters_within_stratum"
        return result
    rng = random.Random(seed)
    boot = []
    if resume is not None:
        if resume["seed"] != seed or resume["replicates"] != replicates:
            raise ValueError("Checkpoint seed or replicate count drift")
        boot = resume["bootstrap_values"]
        if len(boot) != resume["completed"]:
            raise ValueError("Checkpoint draw count mismatch")
        rng.setstate(tuple_state(resume["rng_state"]))
    block_means = [mean(row.tolist()) for row in values] if kind == "mean" else None
    if kind == "mean" and other is not None:
        block_means = [a-mean(row.tolist()) for a, row in zip(block_means, other)]
    for start in range(len(boot), replicates, chunk):
        size = min(chunk, replicates-start)
        draws = np.asarray([
            [group[rng.randrange(len(group))] for group in groups for _ in group]
            for _ in range(size)
        ], dtype=np.int32)
        if kind == "mean":
            boot.extend(mean([block_means[index] for index in selected]) for selected in draws)
        else:
            expanded = values[draws].reshape(size, -1)
            current = np.median(expanded, axis=1)
            if other is not None:
                current -= np.median(other[draws].reshape(size, -1), axis=1)
            boot.extend(current.tolist())
        if checkpoint is not None and (len(boot) % 10000 == 0 or len(boot) == replicates):
            checkpoint({
                "seed": seed, "replicates": replicates, "completed": len(boot),
                "rng_state": rng.getstate(), "bootstrap_values": boot,
            })
    tail = (1-confidence)/2
    result.update({
        "interval": [percentile(boot, tail), percentile(boot, 1-tail)],
        "expected_draws_per_tail": replicates * tail,
        "status": "fixed_law_cluster_percentile",
    })
    return result


def directional(result, direction):
    interval = result["interval"]
    if interval is None:
        return False
    return interval[0] > 0 if direction == "greater" else interval[1] < 0
