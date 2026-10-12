# Far-control rerun: scalar artifact

This artifact replays the far-control analysis (Appendix "Far-Control Rerun") from the per-context
values returned by the training runs. It does not retrain models, and it contains no corpus text or
model states.

## Contents

| Path | What it is |
|---|---|
| `inputs/E01_RETURN_ALL.json` | Focal-query losses and identity records for all 144 contexts, as returned by the six workers (concatenated, otherwise unchanged) |
| `inputs/NEAR_INPUT_BUNDLE.json` | Retained nearest-control values used for the paired endpoints; local provenance paths are redacted |
| `inputs/FAR_MAP_DESCRIPTORS.json` | Reconstructed TF–IDF cosines and eligible-pool sizes for the 192 focal units (input-only) |
| `code/analysis_e01.py` | The frozen analysis law for the five pre-registered endpoints |
| `code/posthoc_e01.py` | Post-hoc descriptions, outside the pre-registered family |
| `code/make_e01_tables.py` | Writes the appendix tables from the result files |
| `results/` | `E01_ANALYSIS_RESULT.json` and `POSTHOC_E01.json` |
| `generated/` | The three appendix tables and a display-provenance map |
| `verify_e01.py` | Checks the manifest, then replays everything byte-for-byte in a temporary copy |

## Replay

```bash
python -B e01_reproduction/verify_e01.py
```

Standard-library Python 3.11 or later is enough.

## Scope

- The five endpoints form their own family, with Bonferroni 99% intervals over 20,000 whole-block draws. An unresolved interval does not establish invariance.
- Post-hoc quantities are descriptive and are never used to select or replace a primary result.
- Far-control cosines are reconstructed with the original vectorizer, not retained measurements.
