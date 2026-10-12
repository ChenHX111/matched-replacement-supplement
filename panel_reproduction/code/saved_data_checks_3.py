"""Third set of post-hoc saved-data checks. No training, no GPU.

Decomposes the shared reference G20 = L(OUT20) - L(IN) using committed scalars only:
- the member-free reference loss L(OUT20) on its own as a score (one retrain, no IN readout);
- the final loss L(IN) on its own (the post-hoc final-NLL score);
- paired rank and regret contrasts of G20 against each component and of L(OUT20) against difficulty;
- G20 operating points: how often its top two equal the cross-seed top two, and how often its signed error
  against cross-seed benefit exceeds 0.5 nats/token;
- the share of the best two focals' cross-seed benefit that each score's top-two choice captures.

In every setting L(IN) = base difficulty - loss drop and L(OUT20) = G20 + L(IN), from the published scores.

Usage: python -I saved_data_checks_3.py <paper_repo_root> <output_json>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from saved_data_checks import (ASSET, CPU, SETTINGS, block_metric, load, sha, spearman,  # noqa: E402
                               stable_seed, targets, top)
import saved_data_checks as base  # noqa: E402

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
base.ROOT_SEED = 202610121930  # fixed before any number below was computed; separate stream from earlier checks
PUBLISHED = {"p5_main": {"G20": 0.6604, "final": 0.4722}}


def boot(diff, name):
    return base.boot(np.asarray(diff, float), name)


def capture_share(theta, order, score):
    """Block means of (mean target over the score's top two) / (mean target over the target's top two)."""
    tg = targets(theta, order, "pair_mean")
    nb = theta.shape[0]
    vals = np.zeros(nb)
    for b in range(nb):
        cell = []
        for ci in range(len(order)):
            t = tg[ci][0][b]
            chosen = list(top(score[b, ci]))
            best = list(top(t))
            denom = t[best].mean()
            cell.append(t[chosen].mean() / denom if denom > 0 else np.nan)
        vals[b] = np.nanmean(cell)
    return vals


def main():
    asset = json.loads(ASSET.read_text(encoding="utf-8"))
    cpu = json.loads(CPU.read_text(encoding="utf-8"))
    out = {"label": "post_hoc_saved_data_checks_3", "draws": base.DRAWS, "root_seed": base.ROOT_SEED,
           "interval": "nominal 95% whole-block percentile bootstrap",
           "inputs": {"asset_scalars_sha256": sha(ASSET), "cpu_inputs_sha256": sha(CPU)}, "settings": {}}
    for setting in ("p5_main", "p5_corpus", "p5_model"):
        order, theta, scores, _, _ = load(setting, asset, cpu)
        final = scores["difficulty"] - scores["loss_drop"]             # L(IN)
        member_free = scores["G20"] + final                             # L(OUT20)
        candidates = {"G20": scores["G20"], "member_free_reference_loss": member_free, "final_loss": final,
                      "difficulty": scores["difficulty"], "U4": scores["U4"], "loss_drop": scores["loss_drop"]}
        res = {"label": SETTINGS[setting], "blocks": int(theta.shape[0]), "absolute": {}, "paired": {},
               "capture_share_top2": {}}
        for name, sc in candidates.items():
            rank = block_metric(theta, order, sc, "rank")
            regret = block_metric(theta, order, sc, "regret")
            res["absolute"][name] = {"rank": boot(rank, f"{setting}:abs:rank:{name}"),
                                     "regret": boot(regret, f"{setting}:abs:regret:{name}")}
            res["capture_share_top2"][name] = boot(capture_share(theta, order, sc), f"{setting}:capture:{name}")
        if setting in PUBLISHED:
            got = {"G20": round(res["absolute"]["G20"]["rank"]["estimate"], 4),
                   "final": round(res["absolute"]["final_loss"]["rank"]["estimate"], 4)}
            if got != PUBLISHED[setting]:
                raise SystemExit(f"published ranks not reproduced: {got} != {PUBLISHED[setting]}")
        pairs = [("G20", "member_free_reference_loss"), ("G20", "final_loss"),
                 ("member_free_reference_loss", "difficulty"), ("member_free_reference_loss", "U4")]
        for first, second in pairs:
            a, b = candidates[first], candidates[second]
            rank = block_metric(theta, order, a, "rank") - block_metric(theta, order, b, "rank")
            regret = block_metric(theta, order, b, "regret") - block_metric(theta, order, a, "regret")
            res["paired"][f"{first}_over_{second}"] = {
                "rank": boot(rank, f"{setting}:pair:rank:{first}:{second}"),
                "regret_reduction": boot(regret, f"{setting}:pair:regret:{first}:{second}")}
        # G20 operating points against the cross-seed target.
        tg = targets(theta, order, "pair_mean")
        same_top, contexts, over, cells = 0, 0, 0, 0
        for b in range(theta.shape[0]):
            for ci in range(len(order)):
                t = tg[ci][0][b]
                g = scores["G20"][b, ci]
                same_top += top(g) == top(t)
                contexts += 1
                over += int(np.sum(g - t > 0.5))
                cells += len(t)
        res["G20_operating_points"] = {"contexts": contexts, "top_two_equal_to_cross_seed_top_two": same_top,
                                       "share_top_two_equal": same_top / contexts,
                                       "focal_cells": cells, "cells_signed_error_above_0_5": over,
                                       "share_signed_error_above_0_5": over / cells}
        out["settings"][setting] = res
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    main_ = out["settings"]["p5_main"]
    print(json.dumps({k: round(v["rank"]["estimate"], 4) for k, v in main_["absolute"].items()}))


if __name__ == "__main__":
    main()
