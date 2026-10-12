"""Post-hoc: how does the full-network checkpoint-gradient (TracIn-style) score compare with the main scores?

Background 0 only (the only background on which the follow-up computed that score), 24 main blocks x 3 seeds.
Reads two committed artifact inputs; writes one JSON. No training, no GPU, nothing re-estimated from text.

Usage: python -I posthoc_gradient_comparator.py <paper_repo_root> <output_json>
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
NEAR = ROOT / "e01_reproduction" / "inputs" / "NEAR_INPUT_BUNDLE.json"
FOLLOW = ROOT / "followup_reproduction" / "inputs" / "CONTEXTS.json"
DRAWS = 20000
ROOT_SEED = 202610111930  # fixed here, before any contrast below was computed


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(root, name):
    return int.from_bytes(hashlib.sha256(f"{root}:{name}".encode()).digest()[:8], "big")


def ranks(values):
    """Average ranks (ties share the mean rank), as scipy.stats.rankdata(method='average')."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values))
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        out[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def spearman(score, target):
    a, b = ranks(score), ranks(target)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def top_two_regret(score, target):
    """Mean target of the two best focals minus mean target of the score's top two (ties: frozen focal order)."""
    score, target = np.asarray(score, float), np.asarray(target, float)
    by_score = sorted(range(len(score)), key=lambda k: (-score[k], k))[:2]
    by_target = sorted(range(len(target)), key=lambda k: (-target[k], k))[:2]
    return float(target[by_target].mean() - target[by_score].mean())


def main():
    near = json.loads(NEAR.read_text(encoding="utf-8"))
    follow = json.loads(FOLLOW.read_text(encoding="utf-8"))
    by_id = {c["context_id"]: c for c in near["contexts"]}
    theta = {cid: np.asarray(c["focal_out_focal_nll"], float) - np.asarray(c["all_in_focal_nll"], float)
             for cid, c in by_id.items()}
    plain = {c["context_id"]: c for c in follow["contexts"]
             if c["context_id"].startswith("p5_main-") and c["independent_plain_epoch_score"] is not None}
    assert len(plain) == 72, len(plain)

    methods = ("G20", "U4", "difficulty", "loss_drop", "plain_gradient")
    per_context, blocks = [], {}
    for cid, fc in sorted(plain.items()):
        c = by_id[cid]
        # Focal order must agree between the two artifacts before any score is paired:
        # same member positions, and bitwise-equal difficulty, individual effects and G20.
        assert list(fc["focal_order"]) == list(range(8)), cid
        assert list(fc["focal_member_positions"]) == list(c["focal_member_positions"]), cid
        assert list(fc["difficulty_nll"]) == list(c["scores"]["base_difficulty_nll"]), cid
        assert list(fc["original_individual_effect"]) == list(theta[cid]), cid
        assert list(fc["original_G20_score"]) == list(c["scores"]["G20"]), cid
        others = [f"{c['block_id']}-b{c['background']}-s{s}" for s in (0, 1, 2) if s != int(cid[-1])]
        target = np.mean([theta[o] for o in others], axis=0)
        scores = {"G20": c["scores"]["G20"], "U4": c["scores"]["uncovered_base_nll_k4"],
                  "difficulty": c["scores"]["base_difficulty_nll"], "loss_drop": c["scores"]["audit_loss_drop"],
                  "plain_gradient": fc["independent_plain_epoch_score"]}
        row = {"context_id": cid, "block_id": c["block_id"]}
        for m in methods:
            row[f"{m}_rank"] = spearman(scores[m], target)
            row[f"{m}_regret"] = top_two_regret(scores[m], target)
        per_context.append(row)
        blocks.setdefault(c["block_id"], []).append(row)

    assert len(blocks) == 24 and all(len(v) == 3 for v in blocks.values())
    block_ids = sorted(blocks)
    block_mean = {m: {k: np.array([np.mean([r[f"{m}_{k}"] for r in blocks[b]]) for b in block_ids])
                      for k in ("rank", "regret")} for m in methods}
    summary = {m: {k: float(block_mean[m][k].mean()) for k in ("rank", "regret")} for m in methods}

    # Published background-0 descriptive values (appendix follow-up methods table) must reproduce first.
    published = {"plain_gradient": (0.222884, 0.258979), "G20": (0.637235, 0.090005),
                 "difficulty": (0.326058, 0.228715)}
    reproduction = {m: {"rank": round(summary[m]["rank"], 6), "regret": round(summary[m]["regret"], 6),
                        "published": published[m],
                        "match": (round(summary[m]["rank"], 6), round(summary[m]["regret"], 6)) == published[m]}
                    for m in published}

    # Fixed post-hoc family: rank and regret contrasts against the gradient score; 8 contrasts.
    family = []
    for other in ("G20", "U4", "difficulty", "loss_drop"):
        family.append((f"{other}_minus_plain_rank", block_mean[other]["rank"] - block_mean["plain_gradient"]["rank"]))
        family.append((f"plain_minus_{other}_regret",
                       block_mean["plain_gradient"]["regret"] - block_mean[other]["regret"]))
    alpha_bonf = 0.05 / len(family)
    results = []
    for name, diff in family:
        rng = random.Random(stable_seed(ROOT_SEED, name))
        n = len(diff)
        boots = np.array([diff[[rng.randrange(n) for _ in range(n)]].mean() for _ in range(DRAWS)])
        lo95, hi95 = np.quantile(boots, [0.025, 0.975])
        lob, hib = np.quantile(boots, [alpha_bonf / 2, 1 - alpha_bonf / 2])
        results.append({"contrast": name, "estimate": float(diff.mean()),
                        "nominal95": [float(lo95), float(hi95)],
                        "bonferroni_family": [float(lob), float(hib)],
                        "bonferroni_level": 1 - alpha_bonf,
                        "positive_blocks": int((diff > 0).sum()), "blocks": n})

    record = {
        "schema": "posthoc-gradient-comparator-v1",
        "status": "post hoc; background 0 only; specified after the main analyses",
        "inputs": {"near_bundle": {"path": str(NEAR.relative_to(ROOT)), "sha256": sha(NEAR)},
                   "followup_contexts": {"path": str(FOLLOW.relative_to(ROOT)), "sha256": sha(FOLLOW)}},
        "score_definitions": {
            "plain_gradient": "sum over 4 epoch-end live checkpoints of eta*(g_q.g_q - g_q.g_c), all 405,334,016 parameters, "
                              "g = gradient of 255-target mean NLL of member query (q) / matched control window (c)",
            "target": "mean near-control theta over the other two seeds, same background"},
        "draws": DRAWS, "root_seed": ROOT_SEED, "unit": "block (24), seeds averaged within block",
        "reproduction_of_published_bg0": reproduction,
        "block_level_summary": summary,
        "contrasts": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({"reproduction": reproduction, "summary": summary,
                      "contrasts": [(r["contrast"], round(r["estimate"], 4), [round(x, 4) for x in r["bonferroni_family"]],
                                     f"{r['positive_blocks']}/{r['blocks']}") for r in results]}, indent=1))


if __name__ == "__main__":
    main()
