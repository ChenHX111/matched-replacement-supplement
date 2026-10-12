# Post-hoc saved-data checks

Everything here is **post hoc**: it was specified after the main analyses were complete. No original test,
interval or decision changes. No training, GPU or text is used.

## Inputs (read-only, from the repository's other artifacts)

- `reproduction/data/asset_scalars.json` and `reproduction/data/cpu_inputs.json`: published main-study
  individual effects and scores for every setting (both backgrounds, three seeds).
- `e01_reproduction/inputs/NEAR_INPUT_BUNDLE.json`: the published near-control effects and scores.
- `followup_reproduction/inputs/CONTEXTS.json`: the follow-up's full-network checkpoint-gradient score
  (background 0).
- `inputs/BACKGROUND_ROWS.json` (this artifact's own input): the 748 background row identifiers of each
  main block and background, taken from the sealed historical all-member jobs. Identifiers only, no text.

Each result file records the SHA-256 of the inputs it read.

## Code and results

| Script | Result | Content |
|---|---|---|
| `code/posthoc_gradient_comparator.py` | `results/POSTHOC_GRADIENT_COMPARATOR.json` | gradient score vs G20, U4, difficulty, loss drop (background 0; Bonferroni over 8) |
| `code/saved_data_checks.py` | `results/SAVED_DATA_CHECKS.json` | adapter pairs; single-seed targets; leave-one-block-out; top-k curves; reliability; support structure |
| `code/saved_data_checks_2.py` | `results/SAVED_DATA_CHECKS_2.json` | paired head-to-heads in all settings; repeatability; variance decomposition; sign-flip and cluster-t |
| `code/saved_data_checks_3.py` | `results/SAVED_DATA_CHECKS_3.json` | decomposition of the shared reference into its member-free loss and the final loss; top-two capture; reference operating points |
| `code/background_overlap.py` | `results/BACKGROUND_OVERLAP.json` | input-only: how many background rows the main blocks share |
| `code/make_panel_tables.py` | `generated/*.tex`, `generated/DISPLAY_PROVENANCE.json` | appendix tables; one provenance record per displayed interval |

Each script first reproduces published values: the background-0 follow-up summaries, Table 1's ranks and
regrets, and the primary contrasts 0.7517 and 0.2505. It refuses to write tables if any differ.

## Verify

```
python -B panel_reproduction/verify_panel.py
```

This checks `MANIFEST.json` against `MANIFEST.sha256` and every listed file's hash. It then re-runs the
analysis scripts and the table generator in a temporary copy and requires byte-identical outputs. Python 3.11
and numpy 1.26.4 were used.
