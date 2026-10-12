"""Generate the post-hoc saved-data appendix tables from this artifact's result files (no hand-typed numbers).

Run from the repository root:  python -B panel_reproduction/code/make_panel_tables.py
Writes panel_reproduction/generated/*.tex and DISPLAY_PROVENANCE.json (one record per displayed interval).
Before writing, published values the tables restate must be reproduced by the results.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "generated"
CKPT, FV = "last_layer_checkpoint_alignment", "for_value_full_vocab_replacement"
SETTING = {"p5_main": "Main", "p5_corpus": "IMDb", "p5_model": "GPT-2"}
PROVENANCE = {}


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def f4(x):
    return f"{x:.4f}"


def cell(key, record, interval_key, source, path):
    lo, hi = record[interval_key]
    PROVENANCE[key] = {"estimate": record["estimate"], "interval": [lo, hi], "source": source, "path": path}
    return f"{f4(record['estimate'])} [{f4(lo)}, {f4(hi)}]"


def blocks(record):
    return f"{record['favourable_blocks']}/{record['blocks']}"


def write(name, text, wrap=()):
    """Write a table file; tabulars whose 0-based index is in ``wrap`` are scaled to the line width."""
    parts, index, start = [], 0, 0
    while True:
        begin = text.find("\\begin{tabular}", start)
        if begin < 0:
            parts.append(text[start:])
            break
        end = text.index("\\end{tabular}", begin) + len("\\end{tabular}")
        body = text[begin:end]
        parts.append(text[start:begin])
        shrink = "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n"  # shrink only
        parts.append(shrink + body + "}" if index in wrap else body)
        index, start = index + 1, end
    (OUT / name).write_text("".join(parts), encoding="utf-8", newline="\n")


def head_to_head(checks, checks2):
    rows = []
    paired = checks2["paired"]
    spec = [("p5_main", "G20", "difficulty", "$G_{20}$ vs difficulty"),
            ("p5_main", "G20", "U4", "$G_{20}$ vs $U_4$"),
            ("p5_main", "G20", "one_seed_effect", "$G_{20}$ vs one-seed effect")]
    for setting, first, second, label in spec:
        rank = paired[setting][f"{first}_minus_{second}_rank"]
        regret = paired[setting][f"{second}_minus_{first}_regret"]
        rows.append((SETTING[setting], label,
                     cell(f"h2h:{setting}:{first}-{second}:rank", rank, "nominal95", "SAVED_DATA_CHECKS_2.json",
                          f"paired.{setting}.{first}_minus_{second}_rank"), blocks(rank),
                     cell(f"h2h:{setting}:{first}-{second}:regret", regret, "nominal95", "SAVED_DATA_CHECKS_2.json",
                          f"paired.{setting}.{second}_minus_{first}_regret"), blocks(regret)))
    adapters = checks["adapter_pairs"]
    for mine, other, label in (("G20", CKPT, "$G_{20}$ vs checkpoint adapter"),
                               ("U4", CKPT, "$U_4$ vs checkpoint adapter"),
                               ("G20", FV, "$G_{20}$ vs For-Value adapter"),
                               ("U4", FV, "$U_4$ vs For-Value adapter")):
        rank = adapters[f"{mine}_minus_{other}_rank"]
        regret = adapters[f"{other}_minus_{mine}_regret"]
        short = "ckpt" if other == CKPT else "forvalue"
        rows.append((SETTING["p5_main"], label,
                     cell(f"h2h:p5_main:{mine}-{short}:rank", rank, "nominal95", "SAVED_DATA_CHECKS.json",
                          f"adapter_pairs.{mine}_minus_{other}_rank"), blocks(rank),
                     cell(f"h2h:p5_main:{mine}-{short}:regret", regret, "nominal95", "SAVED_DATA_CHECKS.json",
                          f"adapter_pairs.{other}_minus_{mine}_regret"), blocks(regret)))
    for setting in ("p5_corpus", "p5_model"):
        for first, second, label in (("G20", "difficulty", "$G_{20}$ vs difficulty"), ("G20", "U4", "$G_{20}$ vs $U_4$")):
            rank = paired[setting][f"{first}_minus_{second}_rank"]
            regret = paired[setting][f"{second}_minus_{first}_regret"]
            rows.append((SETTING[setting], label,
                         cell(f"h2h:{setting}:{first}-{second}:rank", rank, "nominal95", "SAVED_DATA_CHECKS_2.json",
                              f"paired.{setting}.{first}_minus_{second}_rank"), blocks(rank),
                         cell(f"h2h:{setting}:{first}-{second}:regret", regret, "nominal95",
                              "SAVED_DATA_CHECKS_2.json", f"paired.{setting}.{second}_minus_{first}_regret"),
                         blocks(regret)))
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    write("panel_head_to_head.tex", wrap=(0,), text=r"""\begin{table*}[tp]
\centering
\small
\begin{tabular}{@{}llllll@{}}
\toprule
Setting & Comparison & Rank gain [95\%] & Blocks & Regret reduction [95\%] & Blocks \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{Post-hoc paired head-to-head comparisons, computed from saved scores (Main is MultiEURLEX/Pythia;
IMDb is IMDb/Pythia; GPT-2 is MultiEURLEX/GPT-2). Rank gain is the
first score's within-context Spearman rank minus the second's; regret reduction is the second score's top-two
regret minus the first's (nats/token), so positive values favour the first score in both columns. Targets are
the published cross-seed benefits; both backgrounds; nominal 95\% intervals from 20,000 whole-block draws over
24 main or eight transport blocks; Blocks counts blocks favouring the first score. The one-seed effect is the
same-realization individual effect used as a score. The checkpoint and For-Value adapters are the
output-linear attribution adapters of Appendix~\ref{app:scores}. No comparison here was specified in advance.}
\label{tab:panel_head_to_head}
\end{table*}
""")


def gradient(comparator):
    summary = comparator["block_level_summary"]
    names = {"plain_gradient": "Full-network checkpoint gradient", "G20": "$G_{20}$", "U4": "$U_4$",
             "difficulty": "Difficulty", "loss_drop": "Loss drop"}
    rows = [f"{names[m]} & {f4(summary[m]['rank'])} & {f4(summary[m]['regret'])} \\\\"
            for m in ("plain_gradient", "G20", "U4", "difficulty", "loss_drop")]
    labels = {"G20_minus_plain_rank": ("$G_{20}$ $-$ gradient", "rank"),
              "plain_minus_G20_regret": ("$G_{20}$ $-$ gradient", "regret"),
              "U4_minus_plain_rank": ("$U_4$ $-$ gradient", "rank"),
              "plain_minus_U4_regret": ("$U_4$ $-$ gradient", "regret"),
              "difficulty_minus_plain_rank": ("Difficulty $-$ gradient", "rank"),
              "plain_minus_difficulty_regret": ("Difficulty $-$ gradient", "regret"),
              "loss_drop_minus_plain_rank": ("Loss drop $-$ gradient", "rank"),
              "plain_minus_loss_drop_regret": ("Loss drop $-$ gradient", "regret")}
    level = comparator["contrasts"][0]["bonferroni_level"]
    contrast_rows = []
    for record in comparator["contrasts"]:
        label, metric = labels[record["contrast"]]
        contrast_rows.append(f"{label} & {metric} & "
                             + cell("grad:" + record["contrast"], record, "bonferroni_family",
                                    "POSTHOC_GRADIENT_COMPARATOR.json", "contrasts." + record["contrast"])
                             + f" & {record['positive_blocks']}/{record['blocks']} \\\\")
    write("panel_gradient.tex", wrap=(1,), text=r"""\begin{table}[tp]
\centering
\small
\begin{tabular}{@{}lrr@{}}
\toprule
Score (background 0) & Rank & Regret \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{0.6em}
\begin{tabular}{@{}lllr@{}}
\toprule
Contrast & Metric & Estimate [""" + f"{100 * level:.3f}".rstrip("0") + r"""\%] & Blocks \\
\midrule
""" + "\n".join(contrast_rows) + r"""
\bottomrule
\end{tabular}
\caption{Post-hoc comparison with the follow-up's full-network checkpoint-gradient score: the
learning-rate-weighted sum over the four epoch checkpoints of member-minus-control gradient alignment with the
query, over all 405,334,016 parameters (TracIn-style; no optimizer preconditioning). It was computed only on
background 0 (24 main blocks, three seeds). Upper panel: block-mean rank and top-two regret against the
published cross-seed benefit. Lower panel: paired contrasts, positive favouring the first score in both
metrics; Bonferroni intervals over these eight contrasts from 20,000 whole-block draws.}
\label{tab:panel_gradient}
\end{table}
""")


def p_text(p, permutations):
    if p <= 1.5 / permutations:
        return "$<10^{-5}$"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def robustness(checks, checks2):
    labels = {"C1_G20_minus_lossdrop_rank": "$G_{20}$ $-$ loss drop, rank",
              "C2_U4_minus_difficulty_rank": "$U_4$ $-$ difficulty, rank",
              "C3_lossdrop_minus_G20_regret": "Loss drop $-$ $G_{20}$, regret",
              "C4_difficulty_minus_U4_regret": "Difficulty $-$ $U_4$, regret"}
    rows = []
    for key, label in labels.items():
        single = checks["single_heldout_seed_targets"][key]
        lobo = checks["leave_one_block_out"][key]
        alt = checks2["primary_alternative_inference"][key]
        t_record = {"estimate": alt["estimate"], "interval": alt["cluster_t_99_545"]["interval"]}
        if key.endswith("_rank"):
            single_text = cell("robust:single:" + key, single, "nominal95", "SAVED_DATA_CHECKS.json",
                               "single_heldout_seed_targets." + key)
        else:  # fixed selections: a paired regret contrast is linear in the target (oracle term cancels)
            single_text = "unchanged (linear)"
        rows.append(f"{label} & {f4(lobo['full'])} & " + single_text
                    + f" & {f4(lobo['min'])}--{f4(lobo['max'])} & "
                    + p_text(alt["sign_flip_p_two_sided"], checks2["permutations"]) + " & "
                    + cell("robust:cluster_t:" + key, t_record, "interval", "SAVED_DATA_CHECKS_2.json",
                           "primary_alternative_inference." + key + ".cluster_t_99_545")
                    + f" & {alt['positive_blocks']}/24 \\\\")
    write("panel_robustness.tex", wrap=(0,), text=r"""\begin{table*}[tp]
\centering
\small
\begin{tabular}{@{}llllll@{}}
\toprule
Primary contrast & Published & Single held-out seed [95\%] & Leave one block out & Sign-flip $p$ & Cluster-$t$ [99.545\%] \\
\midrule
""".replace("@{}llllll@{}", "@{}lllllll@{}").replace(r"Cluster-$t$ [99.545\%] \\", r"Cluster-$t$ [99.545\%] & Blocks \\")
          + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Post-hoc robustness of the four primary main-setting contrasts (both backgrounds, 24 blocks).
Published is the frozen-family estimate. Single held-out seed evaluates each score against each of the other
two seeds' individual effects separately, instead of their mean (nominal 95\%, 20,000 whole-block draws). For
fixed selections a paired regret contrast is linear in the target, because the oracle top-two term cancels, so the
regret contrasts are unchanged by construction.
Leave one block out gives the range of the 24 estimates that each drop one block. Sign-flip is a two-sided
permutation test of the block mean (200,000 sign assignments). Cluster-$t$ uses the 24 block values with 23
degrees of freedom at the original family's per-test level. Blocks counts blocks with a positive contrast.}
\label{tab:panel_robustness}
\end{table*}
""")


def selection(checks):
    curves = checks["selection_curves"]
    names = [("G20", "$G_{20}$"), ("U4", "$U_4$"), ("difficulty", "Difficulty"), ("loss_drop", "Loss drop"),
             (CKPT, "Checkpoint adapter")]
    header = " & ".join(f"$k={k}$" for k in range(1, 8))
    rows = [f"{label} & " + " & ".join(f4(curves[key][str(k)]) for k in range(1, 8)) + r" \\"
            for key, label in names]
    contrast_rows = []
    for k in range(1, 8):
        pair = curves["contrasts"][str(k)]
        contrast_rows.append(f"$k={k}$ & "
                             + cell(f"curve:{k}:G20", pair["lossdrop_minus_G20_regret"], "nominal95",
                                    "SAVED_DATA_CHECKS.json", f"selection_curves.contrasts.{k}.lossdrop_minus_G20_regret")
                             + " & "
                             + cell(f"curve:{k}:U4", pair["difficulty_minus_U4_regret"], "nominal95",
                                    "SAVED_DATA_CHECKS.json", f"selection_curves.contrasts.{k}.difficulty_minus_U4_regret")
                             + r" \\")
    write("panel_selection.tex", wrap=(0, 1), text=r"""\begin{table*}[tp]
\centering
\small
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Top-$k$ regret & """ + header + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{0.6em}
\begin{tabular}{@{}lll@{}}
\toprule
Budget & Loss drop $-$ $G_{20}$ regret [95\%] & Difficulty $-$ $U_4$ regret [95\%] \\
\midrule
""" + "\n".join(contrast_rows) + r"""
\bottomrule
\end{tabular}
\caption{Post-hoc selection at every audit budget $k$ (main setting, both backgrounds): top-$k$ regret is the
mean cross-seed benefit of the best $k$ of eight focals minus that of the $k$ a score selects (nats/token;
ties to the frozen focal order). $k=2$ is the pre-registered endpoint. Lower panel: paired regret reductions,
nominal 95\% intervals from 20,000 whole-block draws.}
\label{tab:panel_selection}
\end{table*}
""")


def reliability(checks, checks2):
    rows = []
    for setting in ("p5_main", "p5_corpus", "p5_model"):
        rel = checks["reliability"][setting]
        var = checks2["variance_decomposition"][setting]["share"]
        rep = checks2["repeatability"][setting]
        obs = rel["observed_over_ceiling"]
        rows.append(f"{SETTING[setting]} & {f4(rel['mean_pairwise_seed_rank'])} & "
                    f"{f4(rel['spearman_brown_two_seed_reliability'])} & "
                    f"{f4(rel['approx_ceiling_for_perfect_expected_benefit_predictor'])} & "
                    f"{obs['G20']:.2f} & {obs['U4']:.2f} & {obs['difficulty']:.2f} & "
                    f"{var['focal']:.2f} & {var['focal_x_background']:.2f} & {var['seed_residual']:.2f} & "
                    f"{f4(rep['G20'])} \\\\")
    write("panel_reliability.tex", wrap=(0,), text=r"""\begin{table*}[tp]
\centering
\small
\begin{tabular}{@{}lrrrrrrrrrr@{}}
\toprule
& \multicolumn{3}{c}{Target reliability} & \multicolumn{3}{c}{Observed / ceiling} & \multicolumn{3}{c}{Variance share of $\theta$} & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Setting & Seed rank & Two-seed & Ceiling & $G_{20}$ & $U_4$ & Diff. & Focal & Focal$\times$bg & Seed & $G_{20}$ repeat. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Post-hoc target reliability (Main is MultiEURLEX/Pythia; IMDb is IMDb/Pythia; GPT-2 is MultiEURLEX/GPT-2).
Seed rank is the mean within-context Spearman correlation between two
seeds' individual effects. Two-seed applies the Spearman--Brown formula to the two-seed mean target, and
Ceiling is its square root: the approximate rank a perfect predictor of expected benefit would reach against
this noisy target (a classical-test-theory approximation applied to ranks, assuming additive independent
seed noise; indicative only). Observed / ceiling divides each
score's published rank by that ceiling. Variance shares decompose within-context variation of the individual
effect into a stable focal component, a focal-by-background interaction and seed residual (method of moments,
truncated at zero). $G_{20}$ repeat. is the same seed-rank statistic for $G_{20}$ itself.}
\label{tab:panel_reliability}
\end{table*}
""")


def support(checks):
    rows = []
    for setting in ("p5_main", "p5_corpus", "p5_model"):
        row = checks["support_structure"][setting]
        rows.append(f"{SETTING[setting]} & {100 * row['mean_uncovered_fraction']:.2f}\\% & "
                    f"{100 * row['mean_retained_base_loss_mass']:.2f}\\% & "
                    f"{row['mean_within_context_sd_uncovered_fraction']:.4f} & "
                    f"{f4(row['mean_within_context_rank_U4_vs_difficulty'])} & "
                    f"{row['contexts_with_identical_top_two']}/{row['contexts']} \\\\")
    write("panel_support.tex", r"""\begin{table}[tp]
\centering
\small
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Setting & Uncovered & Retained & SD within & Rank($U_4$, D) & Same top two \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Post-hoc input-only support structure (settings as in Table~\ref{tab:panel_reliability}). Uncovered is
the mean share of scored four-gram endpoints
absent from the remaining training corpus; Retained is $U_4$ divided by base loss, averaged over focals; SD
within is the within-context standard deviation of the uncovered share; Rank($U_4$, D) is the mean
within-context Spearman correlation of $U_4$ with difficulty; Same top two counts contexts where $U_4$ and
difficulty select the same two focals. All are computable before training.}
\label{tab:panel_support}
\end{table}
""")


def decomposition(checks3):
    names = [("G20", r"$G_{20}=L(\mathrm{OUT}_{20})-L(\mathrm{IN})$"),
             ("member_free_reference_loss", r"Member-free loss $L(\mathrm{OUT}_{20})$"),
             ("final_loss", r"Final loss $L(\mathrm{IN})$"), ("U4", r"$U_4$"), ("difficulty", "Difficulty"),
             ("loss_drop", "Loss drop")]
    pairs = [("G20_over_member_free_reference_loss", r"$G_{20}$ over member-free loss"),
             ("G20_over_final_loss", r"$G_{20}$ over final loss"),
             ("member_free_reference_loss_over_difficulty", "Member-free loss over difficulty"),
             ("member_free_reference_loss_over_U4", r"Member-free loss over $U_4$")]
    src = "SAVED_DATA_CHECKS_3.json"
    rows, prows, ops = [], [], []
    for setting in ("p5_main", "p5_corpus", "p5_model"):
        res = checks3["settings"][setting]
        for i, (key, label) in enumerate(names):
            a = res["absolute"][key]
            base = f"settings.{setting}"
            rows.append(f"{SETTING[setting] if i == 0 else ''} & {label} & "
                        + cell(f"decomp:{setting}:{key}:rank", a["rank"], "nominal95", src,
                               f"{base}.absolute.{key}.rank") + " & "
                        + cell(f"decomp:{setting}:{key}:regret", a["regret"], "nominal95", src,
                               f"{base}.absolute.{key}.regret") + " & "
                        + cell(f"decomp:{setting}:{key}:capture", res["capture_share_top2"][key], "nominal95", src,
                               f"{base}.capture_share_top2.{key}") + r" \\")
        for i, (key, label) in enumerate(pairs):
            p = res["paired"][key]
            base = f"settings.{setting}.paired.{key}"
            prows.append(f"{SETTING[setting] if i == 0 else ''} & {label} & "
                         + cell(f"decomp:{setting}:{key}:rank", p["rank"], "nominal95", src, base + ".rank")
                         + f" & {blocks(p['rank'])} & "
                         + cell(f"decomp:{setting}:{key}:regret", p["regret_reduction"], "nominal95", src,
                                base + ".regret_reduction") + r" \\")
        if setting != "p5_model":
            rows.append(r"\midrule")
            prows.append(r"\midrule")
        o = res["G20_operating_points"]
        ops.append(f"{SETTING[setting]} {o['top_two_equal_to_cross_seed_top_two']}/{o['contexts']} contexts and "
                   f"{o['cells_signed_error_above_0_5']}/{o['focal_cells']} cells")
    write("panel_decomposition.tex", r"""\begin{table*}[tp]
\centering
\footnotesize
\begin{tabular}{@{}lllll@{}}
\toprule
Setting & Score & Rank & Top-two regret & Top-two capture \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\medskip
\begin{tabular}{@{}lllll@{}}
\toprule
Setting & Paired comparison & Rank difference & Blocks & Regret reduction \\
\midrule
""" + "\n".join(prows) + r"""
\bottomrule
\end{tabular}
\caption{Post-hoc decomposition of the shared reference (nominal 95\% whole-block intervals, 20,000 draws).
$G_{20}$ is the member-free reference loss $L(\mathrm{OUT}_{20})$ minus the final loss $L(\mathrm{IN})$; both
components are recovered from published scores. Top-two capture is the cross-seed benefit of a score's top two
focals divided by that of the best two. Positive paired values favour the first score; Blocks counts blocks
favouring it on rank. $G_{20}$'s top two equal the cross-seed top two, and its signed error against cross-seed
benefit exceeds 0.5 nats/token, in """ + "; ".join(ops) + r""".}
\label{tab:panel_decomposition}
\end{table*}
""", wrap=(0, 1))


def main():
    checks, checks2 = load("SAVED_DATA_CHECKS.json"), load("SAVED_DATA_CHECKS_2.json")
    comparator = load("POSTHOC_GRADIENT_COMPARATOR.json")
    gate = checks["reproduction_gate"]
    assert all(gate[m]["match"] for m in ("G20", "loss_drop", "U4", "difficulty", CKPT)), gate
    assert (gate["C1"]["value"], gate["C2"]["value"]) == (0.7517, 0.2505), gate
    assert all(row["match"] for row in comparator["reproduction_of_published_bg0"].values())
    OUT.mkdir(exist_ok=True)
    head_to_head(checks, checks2)
    gradient(comparator)
    robustness(checks, checks2)
    selection(checks)
    reliability(checks, checks2)
    support(checks)
    decomposition(load("SAVED_DATA_CHECKS_3.json"))
    (OUT / "DISPLAY_PROVENANCE.json").write_text(json.dumps(PROVENANCE, indent=1, sort_keys=True) + "\n",
                                                 encoding="utf-8", newline="\n")
    print(json.dumps({"tables": sorted(p.name for p in OUT.glob("*.tex")), "intervals": len(PROVENANCE)}))


if __name__ == "__main__":
    main()
