"""Post-hoc saved-data checks of the main comparisons. No training, no GPU.

Reads the committed main-study scalar inputs (reproduction/data/*.json) and writes one JSON record.
All analyses are post hoc and descriptive; intervals are nominal 95% whole-block bootstrap (20,000 draws).

Usage: python -I saved_data_checks.py <paper_repo_root> <output_json>
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
ASSET = ROOT / "reproduction" / "data" / "asset_scalars.json"
CPU = ROOT / "reproduction" / "data" / "cpu_inputs.json"
DRAWS = 20000
ROOT_SEED = 202610112015  # fixed before any contrast below was computed
SETTINGS = {"p5_main": "MultiEURLEX/Pythia (main)", "p5_corpus": "IMDb/Pythia", "p5_model": "MultiEURLEX/GPT-2"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(root, name):
    return int.from_bytes(hashlib.sha256(f"{root}:{name}".encode()).digest()[:8], "big")


def ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    out, sv, i = np.empty(len(values)), values[order], 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sv[j + 1] == sv[i]:
            j += 1
        out[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def regret_at(score, target, k):
    score, target = np.asarray(score, float), np.asarray(target, float)
    chosen = sorted(range(len(score)), key=lambda i: (-score[i], i))[:k]
    best = sorted(range(len(target)), key=lambda i: (-target[i], i))[:k]
    return float(target[best].mean() - target[chosen].mean())


def top(score, k=2):
    return tuple(sorted(sorted(range(len(score)), key=lambda i: (-score[i], i))[:k]))


def boot(diff, name):
    rng = random.Random(stable_seed(ROOT_SEED, name))
    n = len(diff)
    draws = np.array([diff[[rng.randrange(n) for _ in range(n)]].mean() for _ in range(DRAWS)])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"estimate": float(diff.mean()), "nominal95": [float(lo), float(hi)],
            "favourable_blocks": int((diff > 0).sum()), "blocks": n}


def load(setting, asset, cpu):
    a, c = asset["p5"][setting], cpu["p5"][setting]
    order = [tuple(x) for x in c["context_order"]]
    assert order == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    theta = np.asarray(a["oracle"], float)                       # [block, context, focal]
    scores = {k: np.asarray(v, float) for k, v in a["scores"].items()}
    scores["loss_drop"] = scores.pop("audit_loss_drop")
    scores["difficulty"] = scores.pop("base_difficulty_nll")
    scores["U4"] = scores.pop("uncovered_base_nll_k4")
    uncovered = np.asarray(c["uncovered_count_k4"], float)
    return order, theta, scores, uncovered, int(c["denominator"])


def targets(theta, order, mode):
    """Per (block, context) target vectors. mode 'pair_mean' = mean of other two seeds (paper);
    'single' = list of the two single other-seed effects."""
    out = {}
    for ci, (bg, s) in enumerate(order):
        others = [order.index((bg, t)) for t in (0, 1, 2) if t != s]
        if mode == "pair_mean":
            out[ci] = [theta[:, others].mean(axis=1)]
        else:
            out[ci] = [theta[:, o] for o in others]
    return out


def block_metric(theta, order, score, metric, mode="pair_mean", k=2):
    """Block-level mean over contexts (and over single-seed targets when mode='single')."""
    tg = targets(theta, order, mode)
    nb = theta.shape[0]
    vals = np.zeros(nb)
    for b in range(nb):
        cell = []
        for ci in range(len(order)):
            for t in tg[ci]:
                if metric == "rank":
                    cell.append(spearman(score[b, ci], t[b]))
                else:
                    cell.append(regret_at(score[b, ci], t[b], k))
        vals[b] = np.mean(cell)
    return vals


def main():
    asset = json.loads(ASSET.read_text(encoding="utf-8"))
    cpu = json.loads(CPU.read_text(encoding="utf-8"))
    record = {"schema": "panel-saved-data-checks-v1",
              "status": "post hoc and descriptive; specified after the main analyses",
              "inputs": {p.name: {"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in (ASSET, CPU)},
              "draws": DRAWS, "root_seed": ROOT_SEED, "unit": "block; contexts averaged within block"}

    order, theta, sc, unc, denom = load("p5_main", asset, cpu)
    ckpt, fv = "last_layer_checkpoint_alignment", "for_value_full_vocab_replacement"
    main_scores = ["G20", "U4", "difficulty", "loss_drop", ckpt, fv]
    bm = {m: {"rank": block_metric(theta, order, sc[m], "rank"),
              "regret": block_metric(theta, order, sc[m], "regret")} for m in main_scores}
    record["main_block_means"] = {m: {k: float(v.mean()) for k, v in bm[m].items()} for m in main_scores}

    # Reproduction gate against published main values (Table 1 / Table 4) before any new number is used.
    pub = {"G20": 0.6604, "loss_drop": -0.0913, "U4": 0.6000, "difficulty": 0.3495, ckpt: 0.5227}
    record["reproduction_gate"] = {m: {"rank": round(record["main_block_means"][m]["rank"], 4), "published": v,
                                       "match": round(record["main_block_means"][m]["rank"], 4) == v}
                                   for m, v in pub.items()}
    record["reproduction_gate"]["regret_ckpt"] = {"value": round(record["main_block_means"][ckpt]["regret"], 4),
                                                  "published": 0.1318}
    record["reproduction_gate"]["regret_U4"] = {"value": round(record["main_block_means"]["U4"]["regret"], 4),
                                                "published": 0.1251}
    primary = {"C1_G20_minus_lossdrop_rank": bm["G20"]["rank"] - bm["loss_drop"]["rank"],
               "C2_U4_minus_difficulty_rank": bm["U4"]["rank"] - bm["difficulty"]["rank"]}
    record["reproduction_gate"]["C1"] = {"value": round(float(primary["C1_G20_minus_lossdrop_rank"].mean()), 4),
                                         "published": 0.7517}
    record["reproduction_gate"]["C2"] = {"value": round(float(primary["C2_U4_minus_difficulty_rank"].mean()), 4),
                                         "published": 0.2505}

    # (1) Paired comparisons with the strongest available attribution adapter (and For-Value).
    record["adapter_pairs"] = {}
    for other in (ckpt, fv):
        for mine in ("G20", "U4"):
            r = f"{mine}_minus_{other}_rank"
            g = f"{other}_minus_{mine}_regret"
            record["adapter_pairs"][r] = boot(bm[mine]["rank"] - bm[other]["rank"], r)
            record["adapter_pairs"][g] = boot(bm[other]["regret"] - bm[mine]["regret"], g)

    # (2) Primary advantages against each single held-out seed instead of the two-seed mean.
    single = {m: {"rank": block_metric(theta, order, sc[m], "rank", "single"),
                  "regret": block_metric(theta, order, sc[m], "regret", "single")}
              for m in ("G20", "U4", "difficulty", "loss_drop")}
    contrasts_single = {
        "C1_G20_minus_lossdrop_rank": single["G20"]["rank"] - single["loss_drop"]["rank"],
        "C2_U4_minus_difficulty_rank": single["U4"]["rank"] - single["difficulty"]["rank"],
        "C3_lossdrop_minus_G20_regret": single["loss_drop"]["regret"] - single["G20"]["regret"],
        "C4_difficulty_minus_U4_regret": single["difficulty"]["regret"] - single["U4"]["regret"]}
    record["single_heldout_seed_targets"] = {k: boot(v, "single:" + k) for k, v in contrasts_single.items()}
    record["single_heldout_seed_targets"]["score_ranks"] = {m: float(single[m]["rank"].mean()) for m in single}

    # (3) Leave-one-block-out range of the paper-target primary contrasts.
    pair = {"C1_G20_minus_lossdrop_rank": primary["C1_G20_minus_lossdrop_rank"],
            "C2_U4_minus_difficulty_rank": primary["C2_U4_minus_difficulty_rank"],
            "C3_lossdrop_minus_G20_regret": bm["loss_drop"]["regret"] - bm["G20"]["regret"],
            "C4_difficulty_minus_U4_regret": bm["difficulty"]["regret"] - bm["U4"]["regret"]}
    record["leave_one_block_out"] = {}
    for k, v in pair.items():
        lobo = np.array([np.delete(v, b).mean() for b in range(len(v))])
        record["leave_one_block_out"][k] = {"full": float(v.mean()), "min": float(lobo.min()),
                                            "max": float(lobo.max()),
                                            "block_dropped_at_min": int(lobo.argmin())}

    # (4) Selection curves: regret at k = 1..7 for each score.
    record["selection_curves"] = {}
    for m in ("G20", "U4", "difficulty", "loss_drop", ckpt):
        record["selection_curves"][m] = {str(k): float(block_metric(theta, order, sc[m], "regret", k=k).mean())
                                         for k in range(1, 8)}
    record["selection_curves"]["contrasts"] = {}
    for k in range(1, 8):
        d1 = block_metric(theta, order, sc["loss_drop"], "regret", k=k) - block_metric(theta, order, sc["G20"], "regret", k=k)
        d2 = block_metric(theta, order, sc["difficulty"], "regret", k=k) - block_metric(theta, order, sc["U4"], "regret", k=k)
        record["selection_curves"]["contrasts"][str(k)] = {
            "lossdrop_minus_G20_regret": boot(d1, f"curve:{k}:G20"),
            "difficulty_minus_U4_regret": boot(d2, f"curve:{k}:U4")}

    # (5) Target reliability and approximate attenuation ceilings, per setting.
    record["reliability"] = {}
    for setting, label in SETTINGS.items():
        o, th, s, u, dn = load(setting, asset, cpu)
        pairs = []
        for b in range(th.shape[0]):
            for bg in (0, 1):
                idx = [o.index((bg, t)) for t in (0, 1, 2)]
                for i in range(3):
                    for j in range(i + 1, 3):
                        pairs.append(spearman(th[b, idx[i]], th[b, idx[j]]))
        rho = float(np.nanmean(pairs))
        rho2 = 2 * rho / (1 + rho)
        ceiling = float(np.sqrt(rho2))
        obs = {m: float(block_metric(th, o, s[m], "rank").mean()) for m in ("G20", "U4", "difficulty", "loss_drop")}
        record["reliability"][setting] = {
            "label": label, "mean_pairwise_seed_rank": rho,
            "spearman_brown_two_seed_reliability": rho2,
            "approx_ceiling_for_perfect_expected_benefit_predictor": ceiling,
            "observed_rank": obs, "observed_over_ceiling": {m: v / ceiling for m, v in obs.items()},
            "caveat": "classical-test-theory approximation applied to Spearman ranks; parallel-seed assumption"}

        # (6) Support structure: does U4 carry information beyond difficulty in this corpus?
        frac = u / dn                                              # uncovered fraction per focal
        ratio = s["U4"] / s["difficulty"]                          # retained base-loss mass per focal
        same_top2, rho_ud, sd_frac, sd_ratio = [], [], [], []
        for b in range(th.shape[0]):
            for ci in range(len(o)):
                same_top2.append(top(s["U4"][b, ci]) == top(s["difficulty"][b, ci]))
                rho_ud.append(spearman(s["U4"][b, ci], s["difficulty"][b, ci]))
                sd_frac.append(float(np.std(frac[b, ci])))
                sd_ratio.append(float(np.std(ratio[b, ci])))
        record.setdefault("support_structure", {})[setting] = {
            "label": label, "mean_uncovered_fraction": float(frac.mean()),
            "mean_retained_base_loss_mass": float(ratio.mean()),
            "mean_within_context_sd_uncovered_fraction": float(np.mean(sd_frac)),
            "mean_within_context_sd_retained_mass": float(np.mean(sd_ratio)),
            "mean_within_context_rank_U4_vs_difficulty": float(np.nanmean(rho_ud)),
            "contexts_with_identical_top_two": int(sum(same_top2)), "contexts": len(same_top2)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("reproduction_gate", "main_block_means")}, indent=1))


if __name__ == "__main__":
    main()
