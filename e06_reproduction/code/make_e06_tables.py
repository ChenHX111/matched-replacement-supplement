"""Generate the E06 tables from the artifact's result files (no hand-typed numbers).

Run from the artifact root:  python -B code/make_e06_tables.py
Writes generated/e06_main.tex (main text), generated/e06_endpoints.tex, generated/e06_exposure.tex,
generated/e06_gradient.tex, generated/e06_posthoc.tex (appendix) and generated/DISPLAY_PROVENANCE.json.
Before writing, the epoch-4 replay must equal the published background-0 values, and the family must be the
frozen eight-test family at 20,000 draws.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAMILY_IDS = ["E06_X1_G20e1_vs_Ae1_rank", "E06_X2_U4_vs_D_rank_e1", "E06_X3_Ae1_minus_G20e1_regret",
              "E06_X4_D_minus_U4_regret_e1", "E06_Y1_G20_vs_Adam_rank", "E06_Y2_U4_vs_Adam_rank",
              "E06_Y3_Adam_minus_G20_regret", "E06_Y4_Adam_minus_U4_regret"]
SHORT = {"E06_X1_G20e1_vs_Ae1_rank": ("X1", r"$G_{20}$ over loss drop, rank"),
         "E06_X2_U4_vs_D_rank_e1": ("X2", r"$U_4$ over difficulty, rank"),
         "E06_X3_Ae1_minus_G20e1_regret": ("X3", r"$G_{20}$ over loss drop, regret"),
         "E06_X4_D_minus_U4_regret_e1": ("X4", r"$U_4$ over difficulty, regret"),
         "E06_Y1_G20_vs_Adam_rank": ("Y1", r"$G_{20}$ over Adam, rank"),
         "E06_Y2_U4_vs_Adam_rank": ("Y2", r"$U_4$ over Adam, rank"),
         "E06_Y3_Adam_minus_G20_regret": ("Y3", r"$G_{20}$ over Adam, regret"),
         "E06_Y4_Adam_minus_U4_regret": ("Y4", r"$U_4$ over Adam, regret")}
VECTOR = {"G20@1": r"$G_{20}^{(1)}$", "A@1": r"$A^{(1)}$", "U4": r"$U_4$", "D": r"$D$", "G20": r"$G_{20}$",
          "Adam": "Adam"}
RESULT = {"HOLDS_AFTER_ONE_PASS": "holds", "UNRESOLVED_AFTER_ONE_PASS": "unresolved",
          "REVERSED_AFTER_ONE_PASS": "reversed", "ACCESS_AWARE_OUTRANKS_GRADIENT": "access-aware",
          "GRADIENT_OUTRANKS_ACCESS_AWARE": "gradient", "UNRESOLVED": "unresolved", "UNAVAILABLE": "unavailable"}


def f4(x):
    return f"{x:.4f}"


def cell(record):
    """'estimate [lower, upper]' from a record with 'estimate' and 'interval'."""
    lo, hi = record["interval"]
    return f"{f4(record['estimate'])} [{f4(lo)}, {f4(hi)}]"


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def table(env, spec, header, rows, caption, label, size=r"\small"):
    return "\n".join([rf"\begin{{{env}}}[tp]", r"\centering", size,
                      r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
                      rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header, r"\midrule", *rows, r"\bottomrule",
                      r"\end{tabular}}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\end{{{env}}}", ""])


def main():
    result, post = load("results/E06_ANALYSIS_RESULT.json"), load("results/POSTHOC_E06.json")
    if result["status"] != "frozen_law_output" or result["family"]["bootstrap_replicates"] != 20000:
        sys.exit("not the frozen E06 family output; refusing to write tables")
    endpoints = {row["id"]: row for row in result["endpoints"]}
    if list(endpoints) != FAMILY_IDS:
        sys.exit("unexpected endpoint set")
    desc = result["descriptive_outside_family"]
    for spec in desc["exposure_contrasts"]["e4"].values():
        if not spec["replay_equals_history"]:
            sys.exit("epoch-4 replay differs from the published background-0 values; refusing to write tables")
    prov = {}

    # Main-text table: the eight-test family.
    rows = [r"\multicolumn{3}{@{}l}{\emph{After one pass (background 0; 72 contexts)}} \\"]
    for i, endpoint_id in enumerate(FAMILY_IDS):
        if i == 4:
            rows.append(r"\multicolumn{3}{@{}l}{\emph{Against the Adam gradient score (both backgrounds; 144)}} \\")
        e = endpoints[endpoint_id]
        tag, text = SHORT[endpoint_id]
        rows.append(f"{tag} & {text} & {cell(e)} \\\\")
        prov[f"e06_main.{tag}"] = {"source": "results/E06_ANALYSIS_RESULT.json", "endpoint_id": endpoint_id,
                                   "estimate": e["estimate"], "interval": e["interval"], "branch": e["branch"]}
    main_tex = table("table", "@{}lll@{}", r"& Advantage & Estimate [99.375\%] \\", rows,
                     r"Pre-registered replay family: Bonferroni 99.375\% intervals, 20,000 whole-block draws "
                     r"over 24 blocks. Positive values favour the first score (higher rank or lower top-two "
                     r"regret against cross-seed benefit). Only X4 is unresolved.",
                     "tab:e06_main")

    # Appendix: the family with components, block counts and leave-one-block-out ranges.
    robust = post["P7_family_block_robustness"]
    rows = []
    for endpoint_id in FAMILY_IDS:
        e, r = endpoints[endpoint_id], robust[endpoint_id]
        tag, text = SHORT[endpoint_id]
        first, second = e["components"]["first"], e["components"]["second"]
        rows.append(f"{tag} & {text} & {VECTOR[first['vector']]} {f4(first['estimate'])} & "
                    f"{VECTOR[second['vector']]} {f4(second['estimate'])} & {cell(e)} & "
                    f"{r['positive_blocks']}/{r['blocks']} & [{f4(r['leave_one_block_out_min'])}, "
                    f"{f4(r['leave_one_block_out_max'])}] & {RESULT[e['branch']]} \\\\")
        prov[f"e06_endpoints.{tag}"] = {"source": "results/E06_ANALYSIS_RESULT.json and results/POSTHOC_E06.json",
                                        "endpoint_id": endpoint_id, "components": e["components"],
                                        "positive_blocks": r["positive_blocks"],
                                        "leave_one_block_out": [r["leave_one_block_out_min"],
                                                                r["leave_one_block_out_max"]]}
    endpoints_tex = table(
        "table*", "@{}llllllll@{}",
        r"& Advantage & First & Second & Estimate [99.375\%] & Blocks $>0$ & Leave one block out & Result \\",
        rows,
        r"Replay family in full. First and Second are the block-weighted levels of the endpoint's two scores "
        r"(rank correlation, or top-two regret in nats/token, against the endpoint's target), and the estimate "
        r"is First minus Second. Regret rows list the comparator first, so a positive estimate always favours "
        r"the advantaged score. Blocks $>0$ counts block values favouring it. Leave-one-block-out gives the "
        r"range of the 24 estimates that each drop one block (post hoc). X-tests use background 0 and "
        r"one-epoch states; Y-tests use both backgrounds and the published target.",
        "tab:e06_endpoints", size=r"\footnotesize")

    # Appendix: exposure trajectory (descriptive, nominal 95%; P5 and P6 are post hoc).
    means, rep, comps = desc["theta_and_nonfocal_means"], desc["cross_seed_repeatability"], desc["exposure_components"]
    p5, p6 = post["P5_one_pass_decomposition_bg0"], post["P6_same_seed_effect_reference_bg0"]
    rows = [r"\multicolumn{6}{@{}l}{\emph{Levels: mean loss drop, individual effect and nonfocal learning "
            r"(nats/token); cross-seed repeatability; nonfocal rank}} \\",
            r"$e$ & $A^{(e)}$ & $\theta^{(e)}$ & $N^{(e)}$ & Seed rank of $\theta^{(e)}$ & "
            r"$N^{(e)}_s$ vs.\ $\bar\theta^{(e)}_{-s}$ \\"]
    for e in (1, 2, 3, 4):
        rows.append(f"{e} & {cell(p5[f'e{e}']['mean_A'])} & {cell(means[f'theta.e{e}'])} & "
                    f"{cell(means[f'N.e{e}'])} & {cell(rep[f'theta.e{e}'])} & {cell(p5[f'e{e}']['rank_N_vs_T'])} \\\\")
    for metric, title in (("spearman", "Rank correlation"), ("top_two_regret", "Top-two regret (nats/token)")):
        rows += [r"\midrule", rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{title} against same-exposure cross-seed "
                              r"benefit $\bar\theta^{(e)}_{-s}$}} \\",
                 r"$e$ & Loss drop $A^{(e)}$ & $G_{20}^{(e)}$ & Difficulty $D$ & $U_4$ & One-seed $\theta^{(e)}_s$ \\"]
        for e in (1, 2, 3, 4):
            c = comps[f"e{e}"]
            rows.append(f"{e} & {cell(c['A'][metric])} & {cell(c['G20'][metric])} & {cell(c['D'][metric])} & "
                        f"{cell(c['U4'][metric])} & {cell(p6[f'e{e}'][metric])} \\\\")
    exposure_tex = table(
        "table*", "@{}llllll@{}", r"\multicolumn{6}{@{}l}{Background 0, 72 contexts, epoch states $e=1,\dots,4$} \\",
        rows,
        r"Exposure trajectory of the replay (descriptive, nominal 95\% whole-block intervals). The state after "
        r"epoch $e$ is an exact $e$-epoch fine-tune; $e=4$ reproduces the published background-0 values. "
        r"Seed rank is the mean within-block Spearman correlation of $\theta^{(e)}$ between seed pairs. "
        r"The nonfocal rank pairs one seed's $N^{(e)}$ with the other two seeds' benefit, and the one-seed "
        r"$\theta^{(e)}_s$ column uses a seed's own effect as a score; both are post hoc and are reference "
        r"points, not ceilings. $A^{(e)}$ means are post hoc.",
        "tab:e06_exposure", size=r"\footnotesize")
    prov["e06_exposure"] = {"source": "results/E06_ANALYSIS_RESULT.json descriptive_outside_family "
                                      "(theta_and_nonfocal_means, cross_seed_repeatability, exposure_components); "
                                      "results/POSTHOC_E06.json (P5 mean_A, rank_N_vs_T; P6)"}

    # Appendix: gradient scores (pooled, published target) and far-control target.
    attr, far = desc["attribution_absolute"]["pooled"], desc["far_targets"]
    p3 = post["P3_gradient_variants_pooled"]["variants"]
    label = {"Adam": "Adam (primary)", "Precond": "Preconditioner only", "Plain": "Plain (TracIn-style)",
             "LESS_cosine": "LESS cosine", "Adam_self": "Adam, self term only", "Plain_self": "Plain, self term only",
             **{f"Adam_e{e}": f"Adam, epoch {e} only" for e in (1, 2, 3, 4)},
             **{f"Plain_e{e}": f"Plain, epoch {e} only" for e in (1, 2, 3, 4)}}
    rows = [r"\multicolumn{4}{@{}l}{\emph{Pre-registered gradient scores (sum over the four checkpoints)}} \\"]
    for name in ("Adam", "Precond", "Plain", "LESS_cosine"):
        rows.append(f"{label[name]} & {cell(attr[name]['spearman'])} & {cell(attr[name]['top_two_regret'])} & "
                    f"{cell(far[name]['spearman'])} \\\\")
    rows.append(r"\multicolumn{4}{@{}l}{\emph{Post-hoc variants}} \\")
    for name in ("Adam_e1", "Adam_e2", "Adam_e3", "Adam_e4", "Adam_self", "Plain_e1", "Plain_e2", "Plain_e3",
                 "Plain_e4", "Plain_self"):
        rows.append(f"{label[name]} & {cell(p3[name]['spearman'])} & {cell(p3[name]['top_two_regret'])} & -- \\\\")
    rows.append(r"\multicolumn{4}{@{}l}{\emph{Reference scores}} \\")
    for name, text, far_name in (("G20", r"$G_{20}$", "G20_far"), ("U4", r"$U_4$", "U4"), ("D", r"Difficulty $D$", None),
                                 ("A", r"Loss drop $A$", None)):
        far_cell = cell(far[far_name]["spearman"]) if far_name else "--"
        rows.append(f"{text} & {cell(attr[name]['spearman'])} & {cell(attr[name]['top_two_regret'])} & {far_cell} \\\\")
    gradient_tex = table(
        "table*", "@{}llll@{}", r"Score & Rank & Top-two regret & Rank, far-control target \\", rows,
        r"Full-network checkpoint-gradient scores over all 144 contexts against published cross-seed benefit, "
        r"and against the far-control benefit of Appendix~\ref{app:farcontrol} (descriptive, nominal 95\%). "
        r"Each score sums, over the epoch checkpoints, the learning rate times the query gradient's inner "
        r"product with the member's minus the control's update: the plain gradient, the preconditioned "
        r"gradient, or the exact AdamW update direction (Adam); LESS cosine normalizes each product. Post-hoc "
        r"variants keep one checkpoint or only the member's self term. For the far target, $G_{20}$ is the "
        r"far reference. The pre-registered Adam score has the best rank and regret of all fourteen variants.",
        "tab:e06_gradient", size=r"\footnotesize")
    prov["e06_gradient"] = {"source": "results/E06_ANALYSIS_RESULT.json descriptive_outside_family "
                                      "(attribution_absolute.pooled, far_targets); results/POSTHOC_E06.json (P3)"}

    # Appendix: post-hoc exposure comparisons.
    p1, p4 = post["P1_G20_vs_U4_by_exposure"], post["P4_exposure_matched_gradient_bg0"]
    rows = []
    for e in (1, 2, 3, 4):
        a, b = p1[f"e{e}"], p4[f"e{e}"]
        rows.append(f"{e} & {cell(a['rank_G20_minus_U4'])} & {cell(a['regret_U4_minus_G20'])} & "
                    f"{cell(b['Adam_cum']['spearman'])} & {cell(b['rank_G20e_minus_Adam_cum'])} & "
                    f"{cell(b['rank_U4_minus_Adam_cum'])} \\\\")
    posthoc_tex = table(
        "table*", "@{}llllll@{}",
        r"$e$ & $G_{20}^{(e)}$ over $U_4$, rank & $G_{20}^{(e)}$ over $U_4$, regret & Adam to $e$, rank & "
        r"$G_{20}^{(e)}$ over Adam to $e$ & $U_4$ over Adam to $e$ \\",
        rows,
        r"Post-hoc exposure comparisons on background 0 against same-exposure cross-seed benefit (nominal "
        r"95\%). Positive values favour the first score. ``Adam to $e$'' sums the Adam score over checkpoints "
        r"$1,\dots,e$, matching the model's exposure.",
        "tab:e06_posthoc", size=r"\footnotesize")
    prov["e06_posthoc"] = {"source": "results/POSTHOC_E06.json (P1, P4)"}

    out = ROOT / "generated"
    for name, text in (("e06_main.tex", main_tex), ("e06_endpoints.tex", endpoints_tex),
                       ("e06_exposure.tex", exposure_tex), ("e06_gradient.tex", gradient_tex),
                       ("e06_posthoc.tex", posthoc_tex)):
        (out / name).write_bytes(text.encode("utf-8"))
    (out / "DISPLAY_PROVENANCE.json").write_bytes(
        (json.dumps(prov, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    print("wrote", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
