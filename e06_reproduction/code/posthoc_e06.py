"""Post-hoc E06 descriptions, outside the pre-registered eight-test family. Read-only on all inputs.

Every quantity here was chosen after the frozen analysis ran, so each is labelled post hoc and reported with
a nominal 95% whole-block interval (20,000 draws; seed = stable_seed(root, "e06-posthoc", item)). None of
them selects, replaces or reinterprets a family endpoint.

The per-context table is built by the frozen analysis program itself (imported, never modified), so every
definition (theta^(e), T^(e), A^(e), G20^(e), N^(e), the gradient scores, Spearman, top-two regret, block
weighting) is the frozen one.

Items
- P1  G20 versus U4 across exposure (rank and regret), the open head-to-head of the main paper.
- P2  Paired gradient comparisons on the published target: Adam - Plain, Adam - D, Adam - A.
- P3  Gradient-variant sensitivity: single-checkpoint Adam scores, self-influence-only scores, and the best
      variant after seeing outcomes (biased toward the gradient score) against G20 and U4.
- P4  Exposure-matched gradient scores: Adam accumulated to epoch e against T^(e).
- P5  One-pass decomposition: N^(e) (same seed) against T^(e) (other seeds); mean N^(e)/theta^(e);
      mean loss drop A^(e).
- P6  Reference point: one seed's own effect theta^(e)_s as a score for T^(e) (not a ceiling).
- P7  Block sign counts and leave-one-block-out ranges of the eight family estimates (from the frozen output).
"""
import argparse
import importlib.util
import json
import math
import pathlib
import sys

REPLICATES = 20000
CONFIDENCE = 0.95


def load_frozen(path):
    spec = importlib.util.spec_from_file_location("analysis_e06_frozen", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", required=True, help="frozen analysis_e06.py")
    parser.add_argument("--return", dest="return_path", required=True)
    parser.add_argument("--near", required=True)
    parser.add_argument("--far", required=True)
    parser.add_argument("--e01-in-epochs", required=True)
    parser.add_argument("--result", required=True, help="frozen E06_ANALYSIS_RESULT.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    m = load_frozen(args.analysis)
    near_index = m.validate_near_bundle_e06(read(args.near))
    far_contexts = m.validate_far_return(read(args.far), near_index)
    contexts, _, _ = m.validate_return(m.read_return_json(args.return_path), near_index)
    e01_in = m.validate_e01_in_epochs(read(args.e01_in_epochs), near_index)
    table = m.build_table(near_index, contexts, far_contexts, e01_in)
    frozen = read(args.result)
    root = m.FAMILY["root_seed"]
    BG0, ALL = m.EXPOSURE_KEYS, m.CONTEXT_KEYS

    # Extra per-context vectors (post hoc): single-checkpoint and self-only gradient scores, cumulative Adam.
    for key in ALL:
        row = table[key]
        q = contexts[key]["gradient"]
        comps, eta = q["components"], q["eta"]
        for e in m.EPOCHS:
            row[f"Adam_e{e}"] = m._vector([eta * (comps[e][k]["S_qUq"] - comps[e][k]["S_qUc"]) for k in range(m.FOCALS)], [])
            row[f"Plain_e{e}"] = m._vector([eta * (comps[e][k]["S_qq"] - comps[e][k]["S_qc"]) for k in range(m.FOCALS)], [])
            row[f"Adam_cum@{e}"] = m._vector([math.fsum(eta * (comps[f][k]["S_qUq"] - comps[f][k]["S_qUc"])
                                                        for f in m.EPOCHS if f <= e) for k in range(m.FOCALS)], [])
        row["Adam_self"] = m._vector([math.fsum(eta * comps[e][k]["S_qUq"] for e in m.EPOCHS) for k in range(m.FOCALS)], [])
        row["Plain_self"] = m._vector([math.fsum(eta * comps[e][k]["S_qq"] for e in m.EPOCHS) for k in range(m.FOCALS)], [])
        if key[1] == 0:
            for e in m.EPOCHS:
                row[f"thetaS@{e}"] = row[f"theta@{e}"]  # one seed's own effect as a score
        row["thetaS"] = row["theta"]

    def describe(values, keys, item, **extra):
        blocks = m.block_means(values, keys)
        record = {"item": item, "units_available": sum(values[k] is not None for k in keys),
                  "units_expected": len(keys), "blocks_available": len(blocks), **extra}
        if len(blocks) < m.MIN_AVAILABLE_BLOCKS:
            return {**record, "estimate": None, "interval": None, "status": "unavailable"}
        block_values = [value for _, value, _ in blocks]
        seed = m.stable_seed(root, "e06-posthoc", item)
        interval, stream = m.bootstrap_interval(block_values, seed=seed, replicates=REPLICATES, confidence=CONFIDENCE)
        return {**record, "estimate": m.mean(block_values), "interval": interval, "seed": seed,
                "index_stream_sha256": stream, "positive_blocks": sum(v > 0 for v in block_values),
                "status": "available_posthoc"}

    def absolute(keys, score, target, metric, item):
        return describe(m._absolute_values(table, keys, score, target, metric), keys, item,
                        score=score, target=target, metric=metric)

    def paired(keys, first, second, target, metric, item):
        cells = {key: m.context_cell(table[key], first, second, target, metric)["value"] for key in keys}
        return describe(cells, keys, item, first=first, second=second, target=target, metric=metric,
                        context_value=(f"{metric}({first},{target}) - {metric}({second},{target})" if metric == "spearman"
                                       else f"R({first},{target}) - R({second},{target})"))

    out = {"label": "post_hoc_outside_family_nominal_95_not_multiplicity_adjusted",
           "replicates": REPLICATES, "confidence": CONFIDENCE,
           "seed_rule": f"stable_seed({root}, 'e06-posthoc', item)",
           "frozen_result_sha256": m.file_sha256(args.result)}

    # P1: G20 versus U4 across exposure (background 0), and on the published target over both backgrounds.
    p1 = {}
    for e in m.EPOCHS:
        p1[f"e{e}"] = {
            "rank_G20_minus_U4": paired(BG0, f"G20@{e}", "U4", f"T@{e}", "spearman", f"P1.rank.G20-U4.e{e}"),
            "regret_U4_minus_G20": paired(BG0, "U4", f"G20@{e}", f"T@{e}", "top_two_regret", f"P1.regret.U4-G20.e{e}")}
    p1["published_pooled"] = {
        "rank_G20_minus_U4": paired(ALL, "G20", "U4", "T", "spearman", "P1.rank.G20-U4.pooled"),
        "regret_U4_minus_G20": paired(ALL, "U4", "G20", "T", "top_two_regret", "P1.regret.U4-G20.pooled")}
    out["P1_G20_vs_U4_by_exposure"] = p1

    # P2: paired gradient comparisons on the published target, both backgrounds.
    out["P2_gradient_pairs_pooled"] = {
        "rank_Adam_minus_Plain": paired(ALL, "Adam", "Plain", "T", "spearman", "P2.rank.Adam-Plain"),
        "regret_Plain_minus_Adam": paired(ALL, "Plain", "Adam", "T", "top_two_regret", "P2.regret.Plain-Adam"),
        "rank_Adam_minus_D": paired(ALL, "Adam", "D", "T", "spearman", "P2.rank.Adam-D"),
        "regret_D_minus_Adam": paired(ALL, "D", "Adam", "T", "top_two_regret", "P2.regret.D-Adam"),
        "rank_Adam_minus_A": paired(ALL, "Adam", "A", "T", "spearman", "P2.rank.Adam-A"),
        "regret_A_minus_Adam": paired(ALL, "A", "Adam", "T", "top_two_regret", "P2.regret.A-Adam"),
    }

    # P3: gradient-variant sensitivity (pooled, published target).
    variants = ["Adam", "Precond", "Plain", "LESS_cosine", "Adam_self", "Plain_self"] + \
               [f"Adam_e{e}" for e in m.EPOCHS] + [f"Plain_e{e}" for e in m.EPOCHS]
    p3 = {name: {metric: absolute(ALL, name, "T", metric, f"P3.{name}.{metric}") for metric in m.METRICS}
          for name in variants}
    best = max(variants, key=lambda name: p3[name]["spearman"]["estimate"])
    best_regret = min(variants, key=lambda name: p3[name]["top_two_regret"]["estimate"])
    out["P3_gradient_variants_pooled"] = {
        "variants": p3,
        "best_rank_variant_selected_after_outcomes": best,
        "best_regret_variant_selected_after_outcomes": best_regret,
        "rank_G20_minus_best": paired(ALL, "G20", best, "T", "spearman", "P3.rank.G20-best"),
        "rank_U4_minus_best": paired(ALL, "U4", best, "T", "spearman", "P3.rank.U4-best"),
        "regret_best_minus_G20": paired(ALL, best_regret, "G20", "T", "top_two_regret", "P3.regret.best-G20"),
        "regret_best_minus_U4": paired(ALL, best_regret, "U4", "T", "top_two_regret", "P3.regret.best-U4"),
        "note": "The best variant is chosen after seeing outcomes, which favours the gradient score.",
    }

    # P4: exposure-matched gradient scores (background 0): Adam accumulated to epoch e against T^(e).
    out["P4_exposure_matched_gradient_bg0"] = {
        f"e{e}": {"Adam_cum": {metric: absolute(BG0, f"Adam_cum@{e}", f"T@{e}", metric, f"P4.Adam_cum.{metric}.e{e}")
                               for metric in m.METRICS},
                  "rank_G20e_minus_Adam_cum": paired(BG0, f"G20@{e}", f"Adam_cum@{e}", f"T@{e}", "spearman",
                                                     f"P4.rank.G20-Adam_cum.e{e}"),
                  "rank_U4_minus_Adam_cum": paired(BG0, "U4", f"Adam_cum@{e}", f"T@{e}", "spearman",
                                                   f"P4.rank.U4-Adam_cum.e{e}")}
        for e in m.EPOCHS}

    # P5: one-pass decomposition (background 0). N^(e) and theta^(e) from the same seed; T^(e) from the others.
    p5 = {}
    for e in m.EPOCHS:
        ratio = {}
        for key in BG0:
            n, t = table[key][f"N@{e}"]["values"], table[key][f"theta@{e}"]["values"]
            ratio[key] = m.mean(n) / m.mean(t) if m.mean(t) > 0 else None
        p5[f"e{e}"] = {
            "rank_N_vs_T": absolute(BG0, f"N@{e}", f"T@{e}", "spearman", f"P5.rank.N-T.e{e}"),
            "context_mean_N_over_mean_theta": describe(ratio, BG0, f"P5.ratio.N-theta.e{e}"),
            "mean_A": describe(m._context_mean(table, BG0, f"A@{e}"), BG0, f"P5.mean.A.e{e}",
                               unit_of_value="nats/token, focal mean"),
        }
    out["P5_one_pass_decomposition_bg0"] = p5

    # P6: one seed's own effect as a score (reference point, not a ceiling).
    p6 = {f"e{e}": {metric: absolute(BG0, f"thetaS@{e}", f"T@{e}", metric, f"P6.thetaS.{metric}.e{e}")
                    for metric in m.METRICS} for e in m.EPOCHS}
    p6["e1"]["rank_U4_minus_thetaS"] = paired(BG0, "U4", "thetaS@1", "T@1", "spearman", "P6.rank.U4-thetaS.e1")
    p6["e1"]["rank_G20e1_minus_thetaS"] = paired(BG0, "G20@1", "thetaS@1", "T@1", "spearman", "P6.rank.G20-thetaS.e1")
    out["P6_same_seed_effect_reference_bg0"] = p6

    # P7: block sign counts and leave-one-block-out ranges of the frozen family estimates.
    p7 = {}
    for endpoint in frozen["endpoints"]:
        values = [row["value"] for row in endpoint["block_values"]]
        n = len(values)
        lobo = [(math.fsum(values) - v) / (n - 1) for v in values]
        p7[endpoint["id"]] = {"estimate": endpoint["estimate"], "interval": endpoint["interval"],
                              "blocks": n, "positive_blocks": sum(v > 0 for v in values),
                              "leave_one_block_out_min": min(lobo), "leave_one_block_out_max": max(lobo)}
    out["P7_family_block_robustness"] = p7

    path = pathlib.Path(args.output)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "best_rank_variant": best, "best_regret_variant": best_regret}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
