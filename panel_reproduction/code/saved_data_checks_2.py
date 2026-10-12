"""Second set of post-hoc saved-data checks. No training, no GPU.

Adds: paired G20-vs-difficulty / G20-vs-U4 / G20-vs-one-seed-effect contrasts in every setting,
cross-seed repeatability of each seed-dependent score, a within-block variance decomposition of the
individual effect, and sign-flip permutation and cluster-t checks of the four primary main contrasts.

Usage: python -I saved_data_checks_2.py <paper_repo_root> <output_json>
"""
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from saved_data_checks import (ASSET, CPU, SETTINGS, block_metric, boot, load, sha,  # noqa: E402
                               spearman, stable_seed)

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
ROOT_SEED = 202610112045  # fixed before any number below was computed
PERMUTATIONS = 200000


def boot2(diff, name):
    rng = random.Random(stable_seed(ROOT_SEED, name))
    n = len(diff)
    draws = np.array([diff[[rng.randrange(n) for _ in range(n)]].mean() for _ in range(20000)])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"estimate": float(diff.mean()), "nominal95": [float(lo), float(hi)],
            "favourable_blocks": int((diff > 0).sum()), "blocks": n}


def one_seed_effect_scores(theta):
    """Same-seed individual effect used as a score (the paper's one-seed reference point)."""
    return theta.copy()


def seed_repeatability(score, order):
    vals = []
    for b in range(score.shape[0]):
        for bg in (0, 1):
            idx = [order.index((bg, s)) for s in (0, 1, 2)]
            for i in range(3):
                for j in range(i + 1, 3):
                    vals.append(spearman(score[b, idx[i]], score[b, idx[j]]))
    return float(np.nanmean(vals))


def variance_decomposition(theta, order):
    """Within-block decomposition of theta[b, (bg,s), focal] into focal, focal x background and
    seed-residual components (balanced two-way layout with replication; method of moments)."""
    nb, nf = theta.shape[0], theta.shape[2]
    ss = {"focal": 0.0, "focal_x_background": 0.0, "seed_residual": 0.0}
    df = {"focal": 0, "focal_x_background": 0, "seed_residual": 0}
    for b in range(nb):
        cube = np.zeros((2, 3, nf))
        for ci, (bg, s) in enumerate(order):
            cube[bg, s] = theta[b, ci]
        # Remove background-by-seed context means: rankings are within context, so they are irrelevant.
        cube = cube - cube.mean(axis=2, keepdims=True)
        cell = cube.mean(axis=1)                         # [bg, focal]
        focal = cell.mean(axis=0)                        # [focal]
        ss["focal"] += 6 * float(((focal - focal.mean()) ** 2).sum())
        ss["focal_x_background"] += 3 * float(((cell - focal[None, :]) ** 2).sum())
        ss["seed_residual"] += float(((cube - cell[:, None, :]) ** 2).sum())
        df["focal"] += nf - 1
        df["focal_x_background"] += (nf - 1) * 1
        df["seed_residual"] += 2 * 2 * nf - 2 * 2   # per bg: (3-1)*nf minus context-mean constraints
    ms = {k: ss[k] / df[k] for k in ss}
    var_seed = ms["seed_residual"]
    var_fb = max((ms["focal_x_background"] - var_seed) / 3, 0.0)
    var_f = max((ms["focal"] - ms["focal_x_background"]) / 6, 0.0)
    total = var_f + var_fb + var_seed
    return {"variance": {"focal": var_f, "focal_x_background": var_fb, "seed_residual": var_seed},
            "share": {"focal": var_f / total, "focal_x_background": var_fb / total,
                      "seed_residual": var_seed / total},
            "method": "balanced within-block method of moments after removing context means"}


def sign_flip_p(diff, name):
    rng = np.random.default_rng(stable_seed(ROOT_SEED, "perm:" + name) % (2 ** 63))
    observed = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(PERMUTATIONS, len(diff)))
    null = np.abs((signs * diff[None, :]).mean(axis=1))
    return float((1 + (null >= observed - 1e-15).sum()) / (PERMUTATIONS + 1))


def cluster_t(diff, level):
    n = len(diff)
    mean, se = diff.mean(), diff.std(ddof=1) / math.sqrt(n)
    # Student-t quantile by bisection on the regularized incomplete beta (no SciPy dependency).
    def t_cdf(t, v):
        x = v / (v + t * t)
        return 1 - 0.5 * betainc(v / 2, 0.5, x) if t > 0 else 0.5 * betainc(v / 2, 0.5, x)
    lo_q, hi_q = 0.0, 50.0
    target = 1 - (1 - level) / 2
    for _ in range(200):
        mid = (lo_q + hi_q) / 2
        if t_cdf(mid, n - 1) < target:
            lo_q = mid
        else:
            hi_q = mid
    q = (lo_q + hi_q) / 2
    return {"level": level, "interval": [float(mean - q * se), float(mean + q * se)], "t_quantile": q}


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b) by continued fraction (Numerical Recipes betacf)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    front = math.exp(lbeta)
    def cf(a, b, x):
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1 - qab * x / qap
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d
            d = 1 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1 + aa / c if abs(c) > 1e-300 else 1 + aa / 1e-300
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d
            d = 1 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1 + aa / c if abs(c) > 1e-300 else 1 + aa / 1e-300
            de = d * c
            h *= de
            if abs(de - 1) < 1e-15:
                break
        return h
    if x < (a + 1) / (a + b + 2):
        return front * cf(a, b, x) / a
    return 1 - math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x)
                        + b * math.log(1 - x)) * cf(b, a, 1 - x) / b


def main():
    asset = json.loads(ASSET.read_text(encoding="utf-8"))
    cpu = json.loads(CPU.read_text(encoding="utf-8"))
    rec = {"schema": "panel-saved-data-checks-2-v1",
           "status": "post hoc and descriptive; specified after the main analyses",
           "inputs": {p.name: {"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in (ASSET, CPU)},
           "root_seed": ROOT_SEED, "permutations": PERMUTATIONS, "bootstrap_draws": 20000}

    rec["paired"], rec["repeatability"], rec["variance_decomposition"] = {}, {}, {}
    for setting, label in SETTINGS.items():
        order, theta, sc, _, _ = load(setting, asset, cpu)
        sc = dict(sc)
        sc["one_seed_effect"] = one_seed_effect_scores(theta)
        bm = {m: {"rank": block_metric(theta, order, sc[m], "rank"),
                  "regret": block_metric(theta, order, sc[m], "regret")}
              for m in ("G20", "U4", "difficulty", "loss_drop", "one_seed_effect")}
        out = {"label": label, "score_means": {m: {k: float(v.mean()) for k, v in bm[m].items()} for m in bm}}
        for other in ("difficulty", "U4", "one_seed_effect"):
            out[f"G20_minus_{other}_rank"] = boot2(bm["G20"]["rank"] - bm[other]["rank"], f"{setting}:{other}:rank")
            out[f"{other}_minus_G20_regret"] = boot2(bm[other]["regret"] - bm["G20"]["regret"],
                                                     f"{setting}:{other}:regret")
        rec["paired"][setting] = out
        rec["repeatability"][setting] = {
            "label": label,
            "individual_effect": seed_repeatability(theta, order),
            "G20": seed_repeatability(sc["G20"], order),
            "loss_drop": seed_repeatability(sc["loss_drop"], order),
            "note": "mean pairwise cross-seed Spearman within (block, background); difficulty and U4 are seed-free"}
        rec["variance_decomposition"][setting] = {"label": label, **variance_decomposition(theta, order)}

    # Permutation and cluster-t checks of the four primary main contrasts (two-seed-mean target).
    order, theta, sc, _, _ = load("p5_main", asset, cpu)
    bm = {m: {"rank": block_metric(theta, order, sc[m], "rank"),
              "regret": block_metric(theta, order, sc[m], "regret")}
          for m in ("G20", "U4", "difficulty", "loss_drop")}
    primary = {"C1_G20_minus_lossdrop_rank": bm["G20"]["rank"] - bm["loss_drop"]["rank"],
               "C2_U4_minus_difficulty_rank": bm["U4"]["rank"] - bm["difficulty"]["rank"],
               "C3_lossdrop_minus_G20_regret": bm["loss_drop"]["regret"] - bm["G20"]["regret"],
               "C4_difficulty_minus_U4_regret": bm["difficulty"]["regret"] - bm["U4"]["regret"]}
    rec["primary_alternative_inference"] = {
        k: {"estimate": float(v.mean()), "sign_flip_p_two_sided": sign_flip_p(v, k),
            "cluster_t_95": cluster_t(v, 0.95), "cluster_t_99_545": cluster_t(v, 1 - 0.05 / 11),
            "positive_blocks": int((v > 0).sum())}
        for k, v in primary.items()}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    f = lambda d: f"{d['estimate']:+.4f} [{d['nominal95'][0]:+.4f}, {d['nominal95'][1]:+.4f}] {d['favourable_blocks']}/{d['blocks']}"
    for s, o in rec["paired"].items():
        print("==", o["label"], {m: round(v["rank"], 4) for m, v in o["score_means"].items()})
        for k, v in o.items():
            if isinstance(v, dict) and "estimate" in v:
                print("  ", k, f(v))
    print("== repeatability", json.dumps({s: {k: (round(v, 4) if isinstance(v, float) else v)
                                              for k, v in r.items() if k != "note"}
                                          for s, r in rec["repeatability"].items()}))
    for s, v in rec["variance_decomposition"].items():
        print("== variance", v["label"], {k: round(x, 3) for k, x in v["share"].items()})
    for k, v in rec["primary_alternative_inference"].items():
        print("== alt", k, round(v["estimate"], 4), "p_perm", v["sign_flip_p_two_sided"],
              "t95", [round(x, 4) for x in v["cluster_t_95"]["interval"]],
              "t99.5", [round(x, 4) for x in v["cluster_t_99_545"]["interval"]], f"{v['positive_blocks']}/24")


if __name__ == "__main__":
    main()
