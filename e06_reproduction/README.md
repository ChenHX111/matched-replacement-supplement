# Exposure-resolved replay with preconditioned gradients: scalar artifact

This artifact replays the analysis of the pre-registered replay study (Appendix "Exposure-Resolved Replay
with Preconditioned Gradients") from the per-context values returned by the training runs. It does not
retrain models, and it contains no corpus text or model states.

## What the study did

- It replayed 792 historical main-setting fits exactly, on the original runtime: the all-member fit of all
  144 contexts, and on background 0 the eight individual replacements and the shared 20-row reference.
- There is no learning-rate schedule, and each epoch's row order depends only on the seed and the epoch.
  The state after epoch e is therefore an exact e-epoch fine-tune. Every fit read its 48 query losses after
  each epoch, and every replay reproduced its historical losses bit for bit.
- Each all-member replay also recorded, at the four epoch checkpoints, full-network inner products between
  each focal member's query gradient and the plain, preconditioned and exact AdamW update directions of the
  member and its control.

## Contents

| Path | What it is |
|---|---|
| `inputs/E06_RETURN_ALL.json` | Focal-query losses per epoch, gradient components and identity records for all 144 contexts, as returned by the six workers (merged, otherwise unchanged) |
| `inputs/E01_IN_EPOCH_READOUTS.json` | Pre-registered fallback input: the far-control rerun's bitwise-historical all-member epoch readouts (not used: every replay passed) |
| `code/analysis_e06.py` | The frozen analysis law for the eight pre-registered endpoints, unchanged since it was frozen before any return existed |
| `code/posthoc_e06.py` | Post-hoc descriptions, outside the pre-registered family |
| `code/make_e06_tables.py` | Writes the main-text and appendix tables from the result files |
| `results/` | `E06_ANALYSIS_RESULT.json` and `POSTHOC_E06.json` |
| `generated/` | The five tables and a display-provenance map |
| `verify_e06.py` | Checks the manifest and the external inputs, then replays everything byte for byte in a temporary copy |

The analysis also reads three inputs committed by the other artifacts, unchanged:
`e01_reproduction/inputs/NEAR_INPUT_BUNDLE.json`, `e01_reproduction/inputs/E01_RETURN_ALL.json` and
`followup_reproduction/inputs/CONTEXTS.json`.

## Replay

```bash
python -B e06_reproduction/verify_e06.py
```

Standard-library Python 3.11 or later is enough.

## Scope

- The eight endpoints form their own family, with Bonferroni 99.375% intervals over 20,000 whole-block draws.
  The original eleven-test family, the follow-up family and the far-control family are unchanged.
- The exposure arm covers background 0 of the main setting. A one-epoch state is one pass over the same
  768 rows in a four-epoch run, not a single exposure during pretraining.
- The gradient scores use exact full-network inner products at four epoch checkpoints. They are not
  projected, curvature-based or unrolled influence estimates.
- Post-hoc quantities are descriptive and are never used to select or replace a primary result.
