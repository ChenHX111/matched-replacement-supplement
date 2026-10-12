"""Post-hoc, input-only description of how much the main study's training backgrounds share rows.

Each main context trains on 748 background rows plus 20 member slots. Backgrounds are hash-ordered draws from
one fitting pool, so different blocks can share background rows. Whole-block resampling treats blocks as
exchangeable and does not model that sharing; this script quantifies it. It uses row identifiers only (no
text, no model outputs) and computes no outcome.

Usage: python -I background_overlap.py <paper_repo_root> <output_json>
"""
import collections
import hashlib
import itertools
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
INPUT = Path(__file__).resolve().parent.parent / "inputs" / "BACKGROUND_ROWS.json"


def summary(values):
    return {"min": min(values), "mean": statistics.fmean(values), "max": max(values)}


def main():
    data = INPUT.read_bytes()
    doc = json.loads(data)
    backgrounds = {(row["block_id"], row["background"]): row["rows"] for row in doc["backgrounds"]}
    if len(backgrounds) != 48 or any(len(rows) != 748 or len(set(rows)) != 748 for rows in backgrounds.values()):
        raise SystemExit("expected 48 backgrounds of 748 distinct rows")
    blocks = sorted({block for block, _ in backgrounds})
    multiplicity = collections.Counter(row for rows in backgrounds.values() for row in rows)
    within = [len(set(backgrounds[(b, 0)]) & set(backgrounds[(b, 1)])) for b in blocks]
    union = {b: set(backgrounds[(b, 0)]) | set(backgrounds[(b, 1)]) for b in blocks}
    shared, jaccard = [], []
    for a, c in itertools.combinations(blocks, 2):
        common = len(union[a] & union[c])
        shared.append(common)
        jaccard.append(common / len(union[a] | union[c]))
    elsewhere = [len(union[b] & set().union(*(union[c] for c in blocks if c != b))) / len(union[b]) for b in blocks]
    same_label = {}
    for label in (0, 1):
        pairs = [len(set(backgrounds[(a, label)]) & set(backgrounds[(c, label)]))
                 for a, c in itertools.combinations(blocks, 2)]
        same_label[f"background_{label}"] = summary(pairs)
    result = {
        "label": "post_hoc_input_only_description",
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "backgrounds": len(backgrounds), "rows_per_background": 748, "blocks": len(blocks),
        "distinct_background_rows": len(multiplicity),
        "backgrounds_containing_a_row": dict(sorted(collections.Counter(multiplicity.values()).items())),
        "within_block_rows_shared_by_its_two_backgrounds": summary(within),
        "block_background_union_rows": summary([len(union[b]) for b in blocks]),
        "between_block_pairs": len(shared),
        "between_block_shared_union_rows": summary(shared),
        "between_block_union_jaccard": summary(jaccard),
        "between_block_shared_rows_same_background_label": same_label,
        "fraction_of_block_union_rows_present_in_another_block": summary(elsewhere),
        "note": "Whole-block resampling treats blocks as exchangeable; it does not model this sharing.",
    }
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("distinct_background_rows", "within_block_rows_shared_by_its_two_backgrounds",
                                             "between_block_shared_union_rows", "between_block_union_jaccard")}))


if __name__ == "__main__":
    main()
