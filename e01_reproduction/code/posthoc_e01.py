"""Post-hoc E01 descriptives: outside the pre-registered five-test family.

Run from the artifact root:  python -B code/posthoc_e01.py results/POSTHOC_E01.json
Nominal 95% whole-block bootstrap intervals (20,000 draws, fixed documented seed). Nothing here
selects, replaces or reinterprets a primary endpoint. Because every new IN fit reproduces its
historical twin bit for bit, theta_far - theta_near equals NLL(OUT_far) - NLL(OUT_near).
"""
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEED = int(hashlib.sha256(b"e01-posthoc-deep-dive-20261011").hexdigest()[:16], 16)
DRAWS = 20000


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r, i = [0.0] * len(xs), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx, syy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return None if sxx == 0 or syy == 0 else sxy / math.sqrt(sxx * syy)


def block_ci(block_values, label):
    vals = [block_values[b] for b in sorted(block_values)]
    rng = random.Random(int(hashlib.sha256((str(SEED) + label).encode()).hexdigest()[:16], 16))
    boots = sorted(statistics.fmean(vals[rng.randrange(len(vals))] for _ in vals) for _ in range(DRAWS))
    return {"estimate": statistics.fmean(vals), "nominal95": [boots[int(0.025 * DRAWS)], boots[int(0.975 * DRAWS) - 1]],
            "blocks_positive": sum(v > 0 for v in vals), "blocks_negative": sum(v < 0 for v in vals)}


def main(out_path):
    far = load("inputs/E01_RETURN_ALL.json")
    near = {c["context_id"]: c for c in load("inputs/NEAR_INPUT_BUNDLE.json")["contexts"]}
    cells = {}
    for c in far["contexts"]:
        q = c["focal_query_ids"]
        fits = {(f["role"], f.get("focal_index")): f for f in c["fits"]}
        L_in = [fits[("IN", None)]["focal_nll"][x] for x in q]
        L_out = [fits[("OUT_FAR", k)]["focal_nll"][q[k]] for k in range(8)]
        L_g20 = [fits[("G20_FAR", None)]["focal_nll"][x] for x in q]
        n = near[c["context_id"]]
        assert n["focal_query_ids"] == q and n["all_in_focal_nll"] == L_in, "IN replay must be bit-identical"
        cells[(c["block_id"], c["background"], c["seed_label"])] = {
            "theta_far": [o - i for o, i in zip(L_out, L_in)], "theta_near": n["theta_near"],
            "g20_far": [g - i for g, i in zip(L_g20, L_in)], "g20_near": n["scores"]["G20"],
            "members": n["focal_member_ids"]}
    blocks = sorted({k[0] for k in cells})

    def per_block(fn):
        return {b: statistics.fmean(fn(v) for k, v in cells.items() if k[0] == b) for b in blocks}

    out = {"label": "post hoc; descriptive; outside the pre-registered E01 family; nominal 95% whole-block bootstrap",
           "seed": SEED, "draws": DRAWS}
    out["mean_theta_near"] = statistics.fmean(x for v in cells.values() for x in v["theta_near"])
    out["mean_theta_far"] = statistics.fmean(x for v in cells.values() for x in v["theta_far"])
    out["theta_far_minus_near"] = block_ci(per_block(lambda v: statistics.fmean(
        f - n for f, n in zip(v["theta_far"], v["theta_near"]))), "D1")
    out["share_cells_far_gt_near"] = statistics.fmean(f > n for v in cells.values()
                                                      for f, n in zip(v["theta_far"], v["theta_near"]))
    out["G20_far_minus_near"] = block_ci(per_block(lambda v: statistics.fmean(
        f - n for f, n in zip(v["g20_far"], v["g20_near"]))), "D7")
    out["rank_agreement_far_near_same_context"] = block_ci(per_block(
        lambda v: spearman(v["theta_far"], v["theta_near"])), "D2")
    out["rank_agreement_note"] = "same-context theta_far and theta_near share the same IN fit, which inflates agreement"

    def seed_rel(key):
        res = {}
        for b in blocks:
            res[b] = statistics.fmean(spearman(cells[(b, g, s)][key], cells[(b, g, t)][key])
                                      for g in (0, 1) for s, t in ((0, 1), (0, 2), (1, 2)))
        return res
    rf, rn = seed_rel("theta_far"), seed_rel("theta_near")
    out["seed_reliability_far"] = block_ci(rf, "D3far")
    out["seed_reliability_near"] = block_ci(rn, "D3near")
    out["seed_reliability_far_minus_near"] = block_ci({b: rf[b] - rn[b] for b in blocks}, "D3diff")
    desc = load("inputs/FAR_MAP_DESCRIPTORS.json")["focal_units"]
    delta = {}
    for v in cells.values():
        for m, f, n in zip(v["members"], v["theta_far"], v["theta_near"]):
            delta.setdefault(m, []).append(f - n)
    units = [u for u in desc if u["member_id"] in delta]
    out["gap_vs_benefit_change_spearman"] = {
        "units": len(units),
        "spearman": spearman([u["near_cosine"] - u["far_cosine"] for u in units],
                             [statistics.fmean(delta[u["member_id"]]) for u in units]),
        "note": "association across focal units; not causal; not a test"}
    out["supply_limited_focal_units"] = [{"member_id": u["member_id"], "far_pool_size": u["far_pool_size"],
                                          "mean_theta_far_minus_near": statistics.fmean(delta[u["member_id"]])}
                                         for u in units if u["far_pool_size"] <= 5]
    pathlib.Path(out_path).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("mean_theta_near", "mean_theta_far", "theta_far_minus_near")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
