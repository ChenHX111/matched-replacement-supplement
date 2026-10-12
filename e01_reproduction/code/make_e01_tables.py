"""Generate the E01 appendix tables from the artifact's result files (no hand-typed numbers).

Run from the artifact root:  python -B code/make_e01_tables.py
Near-control reference values are recomputed from the near bundle with the frozen module's own
functions and must equal the published values before any table is written.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import analysis_e01 as frozen  # noqa: E402

PUBLISHED_NEAR = {"E1": "0.7517", "E2": "0.2505", "E3": "0.3044"}
LABELS = {1: "Far: $G_{20}$ rank $-$ loss-drop rank", 2: "Far: $U_4$ rank $-$ difficulty rank",
          3: "Far: loss-drop regret $-$ $G_{20}$ regret", 4: "Far $-$ near: ($U_4$ $-$ difficulty) rank",
          5: "Far $-$ near: ($G_{20}$ $-$ loss drop) rank"}
BRANCH = {"RESOLVED_POSITIVE": "positive", "RESOLVED_NEGATIVE": "negative", "UNRESOLVED": "unresolved",
          "SUPPORT": "support", "CONTRARY": "contrary"}


def f4(x):
    return f"{x:.4f}"


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def near_values(result, bundle):
    by = {e["number"]: e for e in result["endpoints"]}
    comp = lambda e: frozen.mean([frozen.mean(b["near_context_values"]) for b in e["block_values"]])
    e1, e2 = comp(by[5]), comp(by[4])
    ctx = {(c["block_id"], c["background"], c["seed_label"]): c for c in bundle["contexts"]}
    blocks = sorted({k[0] for k in ctx})
    per_block = []
    for b in blocks:
        vals = []
        for g in (0, 1):
            theta = {s: ctx[(b, g, s)]["theta_near"] for s in (0, 1, 2)}
            for s in (0, 1, 2):
                target = frozen.other_two_seed_target(theta, s)
                sc = ctx[(b, g, s)]["scores"]
                vals.append(frozen.top_two_regret(sc["audit_loss_drop"], target) - frozen.top_two_regret(sc["G20"], target))
        per_block.append(frozen.mean(vals))
    e3 = frozen.mean(per_block)
    got = {"E1": f4(e1), "E2": f4(e2), "E3": f4(e3)}
    if got != PUBLISHED_NEAR:
        sys.exit(f"near reference values {got} differ from the published {PUBLISHED_NEAR}; refusing to write tables")
    return {1: e1, 2: e2, 3: e3, 4: e2, 5: e1}


def main():
    result, post, bundle = load("results/E01_ANALYSIS_RESULT.json"), load("results/POSTHOC_E01.json"), \
        load("inputs/NEAR_INPUT_BUNDLE.json")
    near = near_values(result, bundle)
    prov = {}
    rows = []
    for e in result["endpoints"]:
        n = e["number"]
        lo, hi = e["interval"]
        near_cell = f4(near[n])
        rows.append(f"E{n} & {LABELS[n]} & {f4(e['estimate'])} [{f4(lo)}, {f4(hi)}] & {near_cell} & {BRANCH.get(e['branch'], e['branch'])} \\\\")
        prov[f"E{n}"] = {"source": "results/E01_ANALYSIS_RESULT.json", "endpoint_id": e["id"],
                         "estimate": e["estimate"], "interval": e["interval"], "branch": e["branch"]}
    endpoints = "\n".join([
        r"\begin{table*}[tp]", r"\centering", r"\small", r"\begin{tabular}{@{}lllll@{}}", r"\toprule",
        r"& Endpoint & Far estimate [99\%] & Near & Result \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}",
        r"\caption{Pre-registered far-control family: 24 main blocks, two backgrounds, three seeds, eight focals; "
        r"equal block weights, 20,000 whole-block draws, Bonferroni 99\% intervals over five tests. "
        r"E1--E3 evaluate scores against the far-control cross-seed benefit. E4--E5 subtract, block by block, "
        r"the published nearest-control advantage shown under Near. Near values are the published point "
        r"estimates, recomputed from retained values. An unresolved interval does not establish invariance.}",
        r"\label{tab:e01_endpoints}", r"\end{table*}", ""])
    d = result["descriptive_outside_family"]["items"]
    names = [("audit_loss_drop", "Total loss drop", "-0.0913", "0.3914"),
             ("base_difficulty_nll", "Base difficulty", "0.3495", "0.2240"),
             ("uncovered_base_nll_k4", "$U_4$", "0.6000", "0.1251"),
             ("G20_far", "$G_{20}$ (far reference)", "0.6604", "0.0870")]
    drow = []
    for key, label, near_rank, near_regret in names:
        r, g = d[f"far.{key}.spearman"], d[f"far.{key}.top_two_regret"]
        drow.append(f"{label} & {f4(r['estimate'])} [{f4(r['interval'][0])}, {f4(r['interval'][1])}] & {near_rank} & "
                    f"{f4(g['estimate'])} [{f4(g['interval'][0])}, {f4(g['interval'][1])}] & {near_regret} \\\\")
        prov[f"desc.{key}"] = {"source": "results/E01_ANALYSIS_RESULT.json", "rank": r, "regret": g}
    m = d["far.mismatched_near_G20.spearman"]
    drow.append(f"Near $G_{{20}}$ on far target & {f4(m['estimate'])} [{f4(m['interval'][0])}, {f4(m['interval'][1])}] & --- & --- & --- \\\\")
    descriptive = "\n".join([
        r"\begin{table*}[tp]", r"\centering", r"\small", r"\begin{tabular}{@{}lllll@{}}", r"\toprule",
        r"Score & Far rank [95\%] & Near rank & Far regret [95\%] & Near regret \\", r"\midrule", *drow,
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Descriptive far-control levels outside the family (nominal 95\%, whole-block bootstrap). "
        r"Near values are the published main-setting point estimates (Table~\ref{tab:review_fidelity}). "
        r"Rank is cross-seed Spearman; regret is top-two selection regret in nats/token. "
        r"The last row scores the published nearest-control reference against the far target, a mismatched "
        r"description.}",
        r"\label{tab:e01_descriptive}", r"\end{table*}", ""])
    p = post
    lines = [
        ("Mean individual benefit, near controls", f4(p["mean_theta_near"]), "---"),
        ("Mean individual benefit, far controls", f4(p["mean_theta_far"]), "---"),
        ("Far $-$ near individual benefit", f4(p["theta_far_minus_near"]["estimate"]),
         f"[{f4(p['theta_far_minus_near']['nominal95'][0])}, {f4(p['theta_far_minus_near']['nominal95'][1])}]"),
        ("Far $-$ near shared-reference effect", f4(p["G20_far_minus_near"]["estimate"]),
         f"[{f4(p['G20_far_minus_near']['nominal95'][0])}, {f4(p['G20_far_minus_near']['nominal95'][1])}]"),
        ("Seed reliability of effects, far", f4(p["seed_reliability_far"]["estimate"]),
         f"[{f4(p['seed_reliability_far']['nominal95'][0])}, {f4(p['seed_reliability_far']['nominal95'][1])}]"),
        ("Seed reliability of effects, near", f4(p["seed_reliability_near"]["estimate"]),
         f"[{f4(p['seed_reliability_near']['nominal95'][0])}, {f4(p['seed_reliability_near']['nominal95'][1])}]"),
        ("Lexical gap vs.\\ benefit change (192 focals)", f4(p["gap_vs_benefit_change_spearman"]["spearman"]), "---"),
    ]
    prov["posthoc"] = {"source": "results/POSTHOC_E01.json"}
    posthoc = "\n".join([
        r"\begin{table*}[tp]", r"\centering", r"\small", r"\begin{tabular}{@{}lll@{}}", r"\toprule",
        r"Post-hoc quantity & Value & Nominal 95\% \\", r"\midrule",
        *[f"{a} & {b} & {c} \\\\" for a, b, c in lines], r"\bottomrule", r"\end{tabular}",
        r"\caption{Post-hoc far-control descriptions, outside the pre-registered family. Benefit is in nats/token; "
        r"reliability is the mean pairwise seed Spearman of individual effects; the last row is a Spearman "
        r"association across focal units between the reconstructed near-minus-far cosine gap and the mean "
        r"benefit change. None is a test or a causal estimate.}",
        r"\label{tab:e01_posthoc}", r"\end{table*}", ""])
    out = ROOT / "generated"
    for name, text in (("e01_endpoints.tex", endpoints), ("e01_descriptive.tex", descriptive), ("e01_posthoc.tex", posthoc)):
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    (out / "DISPLAY_PROVENANCE.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("near references:", {k: f4(v) for k, v in near.items() if k <= 3}, "| tables written")


if __name__ == "__main__":
    main()
