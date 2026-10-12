# Supplementary material

Anonymous supplementary material for the submission *Does Loss Drop Track an Example's Effect on Its Own
Loss? Matched-Replacement Evidence from Small Fine-Tunes*.

The repository holds the scalar artifacts behind the paper's numbers. Each module replays its analyses
from saved per-context values (query losses, scores and identity records returned by the training runs) and
checks every output byte for byte against the committed files.

Nothing here trains a model. No corpus text, token sequence or model weight is included.

## Modules

| Folder | Paper section | What it replays | Needs NumPy |
|---|---|---|---|
| `reproduction/` | Main study, transports and donor study (Sections 3–5) | The original eleven-test family (5,000 whole-block draws), descriptive transports, the post-hoc count score and the 100,000-draw sensitivity | yes |
| `followup_reproduction/` | Main-setting reference construction | The background-0 reference-construction follow-up and its own eleven-test family | no |
| `review_reproduction/` | Appendix "Saved-Data Diagnostics" | 204 post-hoc diagnostic summaries | yes |
| `e01_reproduction/` | "Changing the control preserves both orderings"; appendix "Far-Control Rerun" | The pre-registered far-control rerun (five endpoints) | no |
| `panel_reproduction/` | Appendix "Further Saved-Data Checks"; background-overlap counts | Post-hoc head-to-heads, robustness, reliability, support structure and the decomposition of the shared reference | yes |
| `e06_reproduction/` | "One pass, and a preconditioned gradient score"; appendix "Exposure-Resolved Replay with Preconditioned Gradients" | The pre-registered exact replay (eight endpoints): one-pass exposure and full-network gradient scores | no |

Each folder has its own README with its inputs, scope and commands.

## Verify everything

Use Python 3.11 (validated with 3.11.9) and NumPy 1.26.4. From the repository root, run:

```bash
python -B verify_all.py
```

This runs each module's own tests and verifiers in a temporary copy, and takes about two minutes. Your
checkout is never modified.

Add `--full` to also replay the 100,000-draw sensitivity in `reproduction/`, which is slow.

## Scope

**What the modules do not do.** They reproduce analysis arithmetic from retained projections. They do not
retrain models, re-tokenize corpora or reconstruct the private input selection.

**Pre-registration.** "Pre-registered" means that a study's design, endpoints and analysis code were frozen
and hash-recorded before any of its outcomes were returned. These records are internal, not entries in a
public registry.

**Post-hoc analyses.** These are labelled as such. They never replace a pre-registered result.

**Corpora and models.** MultiEURLEX, IMDb, Pythia-410M and GPT-2 are not redistributed and remain under their
own terms.

## License

MIT; see `LICENSE`.
