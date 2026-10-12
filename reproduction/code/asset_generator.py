"""Deterministic LaTeX and two dependency-free PDF figures from source-linked scalars."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SETTINGS = ("p5_main", "p5_corpus", "p5_model")
SHORT = {"p5_main": "Main", "p5_corpus": "IMDb", "p5_model": "GPT-2"}
LONG = {
    "p5_main": "MultiEURLEX / Pythia", "p5_corpus": "IMDb / Pythia", "p5_model": "MultiEURLEX / GPT-2",
    "p4_main": "Main", "p4_corpus": "IMDb", "p4_model": "GPT-2",
}
METHOD = {
    "G20": "G20", "audit_loss_drop": "Original A", "base_difficulty_nll": "Base difficulty",
    "raw_negative_nll": "Final -NLL", "base_negative_nll": "Base -NLL",
    "uncovered_base_nll_k4": "Weighted k4", "uncovered_base_nll_k8": "Weighted k8",
    "MinK20": "Min-K20", "MinK++20": "Min-K++20",
    "last_layer_checkpoint_alignment": "Checkpoint adapter",
    "for_value_full_vocab_replacement": "Forward adapter",
}
PRIMARY_LABELS = {
    "p4_main_later_D": r"Donor later $D_{\mathrm{donor}}$",
    "p4_main_later_eta": r"Donor later $\eta$",
    "p4_main_prefix_D": r"Donor prefix $D_{\mathrm{donor}}$",
    "p4_main_prefix_eta": r"Donor prefix $\eta$",
    "p5_main_G_positive_rank": r"G20 rank above zero",
    "p5_main_G_vs_A_median_absolute_error": r"A--G20 medianAE",
    "p5_main_G_vs_A_spearman": r"G20--A rank",
    "p5_main_G_vs_A_top_two_regret": r"A--G20 regret",
    "p5_main_corpus_vs_A_rank": r"k4--A rank",
    "p5_main_corpus_vs_difficulty_rank": r"k4--difficulty rank",
    "p5_main_corpus_vs_difficulty_regret": r"Difficulty--k4 regret",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_one(rows, **keys):
    result = [row for row in rows if all(row.get(key) == value for key, value in keys.items())]
    if len(result) != 1:
        raise ValueError((keys, len(result)))
    return result[0]


def interval(row, digits=4):
    limits = row.get("interval")
    if limits is None and row.get("ci_low") is not None:
        limits = [row["ci_low"], row["ci_high"]]
    if row.get("estimate") is None:
        return "---"
    point = f"{row['estimate']:.{digits}f}"
    return point if limits is None else point + f" [{limits[0]:.{digits}f}, {limits[1]:.{digits}f}]"


def tex(text):
    result = str(text)
    for left, right in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                        ("&", r"\&"), ("#", r"\#")):
        result = result.replace(left, right)
    return result


def tabular(headers, rows, alignment=None):
    alignment = alignment or "l" * len(headers)
    return "\n".join([
        r"\begin{tabular}{" + alignment + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        *[" & ".join(map(str, row)) + r" \\" for row in rows],
        r"\bottomrule",
        r"\end{tabular}",
    ])


def fragment(out, name, caption, body, source_ids, index, note=None, wide=False):
    environment = "table*" if wide else "table"
    value = (
        "% Generated from bound saved scalar outputs; do not hand-edit numbers.\n"
        f"\\begin{{{environment}}}[t]\n\\centering\n\\footnotesize\n\\setlength{{\\tabcolsep}}{{3pt}}\n{body}\n"
        f"\\caption{{{caption}}}\n\\label{{tab:{name.removeprefix('table_')}}}\n"
        + (r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize " + note
           + r"\end{minipage}" + "\n" if note else "")
        + f"\\end{{{environment}}}\n"
    )
    path = out / (name + ".tex")
    path.write_text(value, encoding="utf-8", newline="\n")
    index[path.name] = {"source_record_ids": source_ids, "sha256": sha(path)}


class PDF:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.commands = []

    def text(self, x, y, value, size=8, bold=False, center=False):
        value = str(value).encode("ascii", errors="strict").decode("ascii")
        if center:
            x -= len(value) * size * .245
        value = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        self.commands.append(f"0 0 0 rg BT /{'F2' if bold else 'F1'} {size:g} Tf {x:.3f} {y:.3f} Td ({value}) Tj ET")

    def line(self, x1, y1, x2, y2, color=(0, 0, 0), width=.6):
        self.commands.append(
            f"{color[0]} {color[1]} {color[2]} RG {width:g} w "
            f"{x1:.3f} {y1:.3f} m {x2:.3f} {y2:.3f} l S"
        )

    def point(self, x, y, color, radius=2.2):
        k = radius * .55228475
        r = radius
        self.commands.append(
            f"{color[0]} {color[1]} {color[2]} rg "
            f"{x+r:.3f} {y:.3f} m "
            f"{x+r:.3f} {y+k:.3f} {x+k:.3f} {y+r:.3f} {x:.3f} {y+r:.3f} c "
            f"{x-k:.3f} {y+r:.3f} {x-r:.3f} {y+k:.3f} {x-r:.3f} {y:.3f} c "
            f"{x-r:.3f} {y-k:.3f} {x-k:.3f} {y-r:.3f} {x:.3f} {y-r:.3f} c "
            f"{x+k:.3f} {y-r:.3f} {x+r:.3f} {y-k:.3f} {x+r:.3f} {y:.3f} c f"
        )

    def save(self, path, title):
        stream = ("\n".join(self.commands) + "\n").encode("ascii")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
             "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>").encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream",
            f"<< /Producer (Saved-scalar asset generator) /Title ({title}) >>".encode("ascii"),
        ]
        output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for number, body in enumerate(objects, 1):
            offsets.append(len(output))
            output.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
        xref = len(output)
        output.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode())
        output.extend(
            f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R /Info 7 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        )
        Path(path).write_bytes(output)


def draw_interval(pdf, x0, x1, y, limits, point, domain, color):
    def x(value):
        return x0 + (value-domain[0])/(domain[1]-domain[0]) * (x1-x0)
    if point is None or limits is None:
        pdf.text(x0, y-2, "undefined", 7)
        return
    if not domain[0] <= limits[0] <= point <= limits[1] <= domain[1]:
        raise ValueError("Plot domain clips a reported interval")
    pdf.line(x(limits[0]), y, x(limits[1]), y, color, 1)
    pdf.line(x(limits[0]), y-2, x(limits[0]), y+2, color)
    pdf.line(x(limits[1]), y-2, x(limits[1]), y+2, color)
    pdf.point(x(point), y, color)


def figures(out, records, counts, index):
    pdf = PDF(555, 276)
    methods = ("G20", "audit_loss_drop", "base_difficulty_nll", "uncovered_base_nll_k4")
    names = ("G20", "A", "Base", "k4")
    colors = ((.10, .28, .55), (.72, .27, .12), (.42, .42, .42), (.10, .48, .30))
    for column, setting in enumerate(SETTINGS):
        left = 6 + column*183
        pdf.text(left+91, 260, LONG[setting], 8.2, True, True)
        pdf.text(left+91, 248, "24 blocks" if setting == "p5_main" else "8 blocks", 7, center=True)
        for row, endpoint in enumerate(("spearman", "top_two_regret")):
            top = 220-row*117
            domain = (-.30, 1.03) if endpoint == "spearman" else (0, .52)
            pdf.text(left+90, top+12, "Rank (higher better)" if row == 0 else "Regret (lower better)", 7.5, center=True)
            x0, x1 = left+35, left+174
            for i, (method, short, color) in enumerate(zip(methods, names, colors)):
                y = top-i*16
                pdf.text(left+1, y-2.5, short, 7)
                metric = records[f"metric:{setting}:{method}:{endpoint}"]
                draw_interval(pdf, x0, x1, y, metric["interval"], metric["estimate"], domain, color)
            axis_y = top-58
            pdf.line(x0, axis_y, x1, axis_y, (.35, .35, .35))
            for tick in ((0, .5, 1) if endpoint == "spearman" else (0, .25, .5)):
                x = x0+(tick-domain[0])/(domain[1]-domain[0])*(x1-x0)
                pdf.line(x, axis_y-2, x, axis_y+2, (.4, .4, .4))
                pdf.text(x, axis_y-12, f"{tick:g}", 6.5, center=True)
    pdf.text(278, 8, "Frozen descriptive 95% intervals; same-background other-two-seed mean oracle; no tokenizer-scale pooling.",
             7, center=True)
    name = "figure_p5_fidelity.pdf"
    pdf.save(out / name, "Individual-benefit fidelity and transport boundary")
    index[name] = {
        "source_record_ids": [f"metric:{setting}:{method}:{endpoint}" for setting in SETTINGS
                              for method in methods for endpoint in ("spearman", "top_two_regret")],
        "sha256": sha(out / name),
    }
    pdf = PDF(555, 228)
    rows = [(setting, right) for setting in SETTINGS
            for right in ("uncovered_base_nll_k4", "base_difficulty_nll")]
    for column, endpoint in enumerate(("spearman", "top_two_regret")):
        left = 5 + column*277
        x0, x1 = left+82, left+261
        domain = (-.34, .46) if endpoint == "spearman" else (-.16, .17)
        pdf.text(left+139, 211, "Count - comparator rank" if column == 0 else "Comparator - count regret", 8.5, True, True)
        zero = x0+(0-domain[0])/(domain[1]-domain[0])*(x1-x0)
        pdf.line(zero, 58, zero, 196, (.65, .65, .65), .5)
        for i, (setting, right) in enumerate(rows):
            y = 187-i*23
            label = SHORT[setting] + (" / k4" if right == "uncovered_base_nll_k4" else " / base")
            pdf.text(left, y-2.5, label, 7.1)
            row = exact_one(counts["comparisons"], setting=setting, right=right, endpoint=endpoint)
            color = (.10, .28, .55) if right == "uncovered_base_nll_k4" else (.45, .45, .45)
            draw_interval(pdf, x0, x1, y, [row["ci_low"], row["ci_high"]], row["estimate"], domain, color)
        pdf.line(x0, 47, x1, 47, (.35, .35, .35))
        for tick in ((-.3, 0, .3) if endpoint == "spearman" else (-.1, 0, .1)):
            x = x0+(tick-domain[0])/(domain[1]-domain[0])*(x1-x0)
            pdf.line(x, 45, x, 49, (.35, .35, .35))
            pdf.text(x, 34, f"{tick:g}", 7, center=True)
    pdf.text(278, 17, "Post-hoc paired 95% intervals; positive favors unweighted count/255. All settings and both comparators retained.",
             7, center=True)
    name = "figure_posthoc_count.pdf"
    pdf.save(out / name, "Post-hoc unweighted coverage comparisons")
    index[name] = {"source_record_ids": [row["id"] for row in counts["comparisons"]],
                   "source": "COUNT_CHECK.json", "sha256": sha(out / name)}


def generate(values, counts, bootstrap, out):
    out.mkdir(parents=True, exist_ok=True)
    records = values["records"]
    index = {}
    rows = []
    for row in bootstrap["rows"]:
        original, sensitivity = row["original"], row["sensitivity"]
        rows.append([PRIMARY_LABELS[row["id"]], interval(original), interval(sensitivity),
                     "yes" if original["direction_supported"] else "no"])
    sensitivity_note = (
        "No sign decision changes." if not bootstrap["changed_sign_decisions"] else
        "Sensitivity changes the sign decision for " + tex(", ".join(bootstrap["changed_sign_decisions"])) +
        "; qualify those claims rather than replacing the original decisions."
    )
    fragment(out, "table_primary_family",
             "Complete original 11-test LM family. Both columns use 99.545\\% intervals; 100,000 draws are a post-hoc Monte Carlo sensitivity, not added independent evidence.",
             tabular(["Endpoint", "Original 5,000", "Sensitivity 100,000", "Original support"], rows, "lrrc"),
             [row["id"] for row in bootstrap["rows"]], index,
             note="Original decisions are retained. " + sensitivity_note +
             " G20's absolute test is rank above zero, not a practical error/regret threshold. The later-window donor joint substitution criterion remains unmet.", wide=True)
    methods = ("G20", "audit_loss_drop", "base_difficulty_nll", "uncovered_base_nll_k4",
               "raw_negative_nll", "base_negative_nll", "uncovered_base_nll_k8",
               "MinK20", "MinK++20", "last_layer_checkpoint_alignment", "for_value_full_vocab_replacement")
    access = {
        "G20": "1 reference retrain", "audit_loss_drop": "Base + final NLL",
        "base_difficulty_nll": "Base NLL", "uncovered_base_nll_k4": "Base NLL + corpus",
        "raw_negative_nll": "Final token NLL", "base_negative_nll": "Base NLL",
        "uncovered_base_nll_k8": "Base NLL + corpus",
        "MinK20": "Final token NLL", "MinK++20": "Vocab. moments",
        "last_layer_checkpoint_alignment": "4 epoch readouts",
        "for_value_full_vocab_replacement": "Base output factors",
    }
    rows, ids = [], []
    for method in methods:
        rank_id, regret_id = [f"metric:p5_main:{method}:{endpoint}" for endpoint in ("spearman", "top_two_regret")]
        error_id = f"metric:p5_main:{method}:median_absolute_error"
        ids += [rank_id, regret_id]
        if method in {"G20", "audit_loss_drop"}:
            ids.append(error_id)
        rows.append([tex(METHOD[method]), interval(records[rank_id]),
                     interval(records[error_id]) if method in {"G20", "audit_loss_drop"} else "---",
                     interval(records[regret_id]), access[method]])
    fragment(out, "table_p5_absolute",
             "All 11 applicable original scores in the main individual-benefit study (24 blocks), evaluated against the same-background other-two-seed mean individual oracle. Intervals are nominal 95\\%.",
             tabular(["Fixed score", "Rank", "MedianAE", "Top-two regret", "Access"], rows, "lrrrl"),
             ids, index, note="Magnitude errors are shown only for G20/A; difficulty, k4/k8 and feature scores are uncalibrated. Regret is in oracle nats/token. k8 is the fixed sensitivity; G8 reuses main blocks. Ties retain frozen focal order. Controls are nearest lexical same-label, high-relatedness alternatives. Membership classification is not tested. Checkpoint and Forward adapters are output-linear checkpoint (TracIn-style) and forward (For-Value-style) adapters, not exact AdamW influence; self-query/tied-weight limitations remain.", wide=True)
    rows, ids = [], []
    for setting in ("p5_corpus", "p5_model"):
        for method in ("G20", "audit_loss_drop", "base_difficulty_nll", "uncovered_base_nll_k4"):
            rank_id, regret_id = [f"metric:{setting}:{method}:{endpoint}"
                                  for endpoint in ("spearman", "top_two_regret")]
            error_id = f"metric:{setting}:{method}:median_absolute_error"
            ids.extend([rank_id, regret_id])
            if method in {"G20", "audit_loss_drop"}:
                ids.append(error_id)
            rows.append([SHORT[setting], METHOD[method], interval(records[rank_id]),
                         interval(records[error_id]) if method in {"G20", "audit_loss_drop"} else "---",
                         interval(records[regret_id])])
    for setting in ("p5_corpus", "p5_model"):
        root = setting.removeprefix("p5_")
        for label, prefix in (
            ("G20 / A", f"comparison:p5_{root}_G_vs_A_"),
            ("G20 / difficulty", f"exploratory:{setting}:G20:difficulty:"),
            ("k4 / difficulty", f"comparison:p5_{root}_corpus_vs_difficulty_"),
        ):
            rid = prefix + ("rank" if label == "k4 / difficulty" else "spearman")
            gid = prefix + ("regret" if label == "k4 / difficulty" else "top_two_regret")
            eid = prefix + "median_absolute_error"
            ids.extend([rid, gid])
            if label == "G20 / A":
                ids.append(eid)
    fragment(out, "table_transport_boundary",
             "Core four fixed scores in the two transports, eight blocks each. Absolute intervals are frozen descriptive, nominal 95\\%; positive correlation alone is not relative success.",
             tabular(["Setting", "Fixed score", "Rank", "MedianAE (G20/A)", "Top-two regret"], rows, "llrrr"),
             ids, index, note="Difficulty and k4 are uncalibrated. IMDb: G20/A improves rank/regret nominally, not medianAE. G20/difficulty does not establish a rank/regret advantage; k4/difficulty also does not establish a rank/regret advantage, with exactly identical top-two choices. GPT-2 relative gains are nominal; G20/difficulty comparisons are exploratory. IMDb lacks movie/reviewer identifiers. NLL scales are not pooled across tokenizers.", wide=True)
    rows, ids = [], []
    for row in values["g20_tails"]:
        setting = row["setting"]
        rid = f"reliability:{setting}:oracle_cross_seed_stability"
        eid = f"repeat_error:{setting}"
        ids.extend([rid, eid])
        rows.append([SHORT[setting], f"{row['median_absolute_error']:.4f}", f"{row['p95_absolute_error']:.4f}",
                     f"{row['signed_bias_mean']:.4f}", interval(records[rid]), interval(records[eid])])
    body = tabular(["Setting", "G20 medianAE", "G20 P95 AE", "Signed bias", "Seed-oracle rank", "Repeat-oracle medianAE"],
                   rows, "lrrrrr")
    g8_rows = []
    for endpoint, name in (("spearman", "Rank gain"), ("median_absolute_error", "MedianAE reduction"),
                           ("top_two_regret", "Regret reduction")):
        rid = "comparison:p5_group8_G8_vs_G20_" + endpoint
        ids.append(rid)
        g8_rows.append([name, interval(records[rid])])
    body += "\n\\par\\medskip\n" + tabular(["G8 versus G20: eight nested blocks", "Paired nominal 95\\%"], g8_rows, "lr")
    cost_rows = []
    for setting, variant, label in (
        ("p5_main", "ALL_OUT", "G20 reference fits"),
        ("p5_main", "FOCAL_OUT_individual", "Individual-oracle retrains"),
        ("p5_main", "ALL_IN", "Shared instrumented ALL\\_IN"),
        ("p5_main_group8", "FOCALS_OUT", "G8 stress reference fits"),
        ("BASE", "BASE", "Shared BASE actions, all LM"),
    ):
        row = exact_one(values["costs"], setting=setting, variant=variant)
        cost_rows.append([label, row["native_count"], f"{row['summed_gpu_process_hours_total']:.4f}",
                          f"{row['summed_gpu_process_hours_diagnostics']:.4f}"])
    body += "\n\\par\\medskip\n" + tabular(["Measured cost", "Actions/fits", "Job-wall hours", "Diagnostic hours"], cost_rows, "lrrr")
    fragment(out, "table_p5_error_cost_variability",
             "Residual error, oracle repeatability, group-size stress, and actual access costs. Error tails and oracle-repeat error are existing exploratory summaries; repeatability is not a proven noise floor.",
             body, ids+["existing_g20_tails", "saved_variant_costs"], index,
             note="Given ALL\\_IN, G20 requires one matched retrain versus eight for individual oracles. All fits score 48 queries with 255 targets each. Corpus scores need base token losses and aligned-corpus matching; attribution requires hidden/full-vocabulary factors. Hours sum GPU-process wall times, not elapsed campaign time or energy. No practical accuracy threshold was registered.", wide=True)
    rows, ids = [], []
    for setting in ("p4_main", "p4_corpus", "p4_model"):
        for window in ("prefix", "later"):
            rids = [f"p4:{setting}:{window}:high:{endpoint}" for endpoint in ("D", "I", "eta")]
            ids.extend(rids)
            rows.append([LONG[setting], window, *[interval(records[rid]) for rid in rids]])
    body = tabular(["Setting", "Window", r"$D_{\mathrm{donor}}$", r"$I_{\mathrm{donor}}$",
                    r"$\eta=I_{\mathrm{donor}}-D_{\mathrm{donor}}$"], rows, "llrrr")
    rows = []
    for window in ("prefix", "later"):
        rids = [f"p4:p4_edits:{window}:overlap_edit-minus-sham_edit:{endpoint}" for endpoint in ("D", "eta")]
        ids.extend(rids)
        rows.append([window, *[interval(records[rid]) for rid in rids]])
    body += "\n\\par\\medskip\n" + tabular(["Direct overlap-edit minus sham", r"$D_{\mathrm{donor}}$", r"$\eta$"], rows, "lrr")
    rows = []
    for setting in ("p4_main", "p4_corpus", "p4_model"):
        rid = f"panel:{setting}:high:OUT:mean_minus_base_nll"
        pid = f"paired_panel:{setting}:high:D"
        ids.extend([rid, pid])
        paired = records[pid]
        adverse_direction = {"estimate": -paired["estimate"],
                             "interval": [-paired["interval"][1], -paired["interval"][0]]}
        rows.append([LONG[setting], interval(records[rid]), interval(adverse_direction)])
    body += "\n\\par\\medskip\n" + tabular(["Fixed 64-query panel", "High OUT minus base NLL", "High OUT minus refresh OUT"], rows, "lrr")
    fragment(out, "table_p4_scope",
             "Supporting results of the donor study. Intervals are nominal 95\\%; the complete corrected family is reported separately. Positive panel NLL changes mean harm.",
             body, ids, index,
             note="Joint substitution requires $D_{\\mathrm{donor}}>0$ and $\\eta<0$. Main later eta and direct edit-minus-sham effects remain imprecise. The natural reduced-overlap arm was omitted after 0/48 feasible input matches, not tested as a null. No semantic-isolation, equivalence-to-zero, or downstream-utility claim.", wide=True)
    rows = []
    for setting in SETTINGS:
        for right in ("uncovered_base_nll_k4", "base_difficulty_nll"):
            rank = exact_one(counts["comparisons"], setting=setting, right=right, endpoint="spearman")
            regret = exact_one(counts["comparisons"], setting=setting, right=right, endpoint="top_two_regret")
            rows.append([SHORT[setting], "weighted k4" if right == "uncovered_base_nll_k4" else "difficulty",
                         interval(rank), interval(regret)])
    body = tabular(["Setting", "Comparator", "Count rank gain", "Count regret reduction"], rows, "llrr")
    absolute_rows = []
    for setting in SETTINGS:
        rank = exact_one(counts["absolute_unweighted_metrics"], setting=setting, endpoint="spearman")
        regret = exact_one(counts["absolute_unweighted_metrics"], setting=setting, endpoint="top_two_regret")
        absolute_rows.append([SHORT[setting], interval(rank), interval(regret)])
    body += "\n\\par\\medskip\n" + tabular(["Count/255 absolute performance", "Rank", "Top-two regret"], absolute_rows, "lrr")
    fragment(out, "table_posthoc_count",
             "The fixed post-hoc unweighted k4 check: saved uncovered count divided by the same 255 scored targets. Paired block intervals are nominal 95\\%, with no multiplicity correction.",
             body, [row["id"] for row in counts["comparisons"]+counts["absolute_unweighted_metrics"]], index,
             note="Positive contrasts favor count. No raw magnitude error is computed for this dimensionless score. Main count improves beyond difficulty, but its difference from weighted k4 is imprecise: do not claim equivalence or unique weighting value. GPT-2 weighted k4 has a nominal regret advantage; IMDb remains an adverse/uncertain boundary. The original main weighted-k4 result is retained.", wide=True)
    figures(out, records, counts, index)
    sources = {
        "schema": "r03c-paper-asset-index-v1", "protocol_sha256": values["protocol_sha256"],
        "assets": index, "vector_figure_count": 2, "latex_fragment_count": 6,
        "record_sources": {rid: record["source"] for rid, record in records.items()},
        "new_result_sources": ["COUNT_CHECK.json", "BOOTSTRAP_SENSITIVITY.json"],
        "no_manual_numeric_copying": True, "new_model_calls": 0,
        "scalar_scope": values["scalar_scope"], "control_design": values["control_design"],
    }
    (out / "ASSET_INDEX.json").write_text(
        json.dumps(sources, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n",
    )
    notes = (
        "# Generated paper assets\n\n"
        "Six source-linked LaTeX fragments require `booktabs`; include each with `\\input{...}`. "
        "Two compact PDFs use only vector lines, marks and standard PDF fonts; include with `graphicx`.\n\n"
        "All figures use nominal95% intervals. The primary-family table alone uses99.545% intervals. "
        "Original5000-draw decisions remain authoritative; the100000-draw column is a post-hoc precision sensitivity. "
        "The unweighted count check is also post-hoc. Neither interval overlap nor nonsignificance establishes equivalence.\n\n"
        "See `ASSET_INDEX.json` for every row's source record and file SHA256. "
        "For-Value/checkpoint feature-scale errors are not displayed as calibrated nats. "
        "Error tails are existing diagnostics, not a new error analysis. "
        "Transport scales and labels remain separate. Main plus two transports contain1920 unique focal/context cells; "
        "G8 reuses main cells and adds no independent blocks. Fixed-score ties retain frozen focal order. "
        "The main individual-benefit study uses nearest lexical same-label, high-relatedness controls. "
        "No model weights or raw text are needed for these numerical assets.\n"
    )
    (out / "ASSET_USAGE.md").write_text(notes, encoding="utf-8", newline="\n")
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", type=Path, required=True)
    parser.add_argument("--count", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values, counts, bootstrap = [read(path) for path in (args.values, args.count, args.bootstrap)]
    assert values["protocol_sha256"] == counts["protocol_sha256"] == bootstrap["protocol_sha256"]
    result = generate(values, counts, bootstrap, args.output)
    print("GENERATED_ASSETS", result["latex_fragment_count"], "LaTeX fragments;",
          result["vector_figure_count"], "vector PDFs")


if __name__ == "__main__":
    main()
