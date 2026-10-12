# Separate saved-data diagnostic artifact

This module adds post-hoc diagnostics without modifying `reproduction` or
`followup_reproduction`. It requires the original sibling `reproduction` module,
Python with the original NumPy dependency, and no model, tokenization, network
access or additional package. Neural imports are explicitly denied.

From this directory:

```text
python -B test_saved_diagnostics.py
python -B saved_diagnostics.py --output replayed
python -B verify_review.py --replayed replayed
```

The first command tests the new diagnostic arithmetic and immutable targets.
The private projection-helper test is explicitly skipped outside its author
workspace; it was run there separately. The second command reconstructs all
204 summaries, using 20,000 block draws for new analyses and the exact original
5,000-draw streams for 21 retained diagnostics. The last command verifies the
manifest and byte-compares every fresh numerical output with `results`.
The manifest-only command is `python -B verify_review.py`.

## Scientific scope

The three original settings retain 24/8/8 blocks, two backgrounds, three seeds
and eight ordered focals. Each scoring seed targets the other two original
seeds in the same background. Complete blocks, not seeds or token losses, are
resampled; all blocks keep equal weight. No high-reliability or successful
subset changes the target. All new intervals are exploratory nominal 95%.
No new confirmatory family or change to the two original eleven-test families
is made. Original displayed intervals remain in the original artifacts.

- `SUMMARY.json/.csv`: all absolute metrics, paired final-NLL comparisons,
  component associations, input-half summaries and selection-stability intervals.
- `CELLS.json`: complete ordered target/predictor tensors and diagnostic cells.
- `STABILITY.json`: every fixed seed, background and block description, including
  leave-one-block-out means without changing other-two-seed targets.
- `INPUT_STRATA.json`: both input-only splits, ranges and complete memberships.
- `generated`: manuscript table sources and interval-origin metadata.
- `review_inputs.json`: input-defined control descriptors, exact retained
  diagnostic values and hashes binding the original scalar inputs.

Positive final NLL is a post-hoc reversal of the original likelihood direction;
the old direction remains untouched. One seed's individual effect is an empirical
predictor, not a ceiling or noise-floor estimate. Different-seed component
associations avoid a shared OUT realization across operands but do not identify
a causal mechanism. The fixed input halves do not test another control policy.

Control coverage is positional trailing-four-gram coverage in the selected
control's bounded prefix, divided by 255. It is **not** the original TF–IDF
matching similarity. The latter was not saved with the selected pairs.
The author-side extraction verified original identities, roles, token windows
and scalar joins before discarding private identifiers and token sequences.
This module replays descriptor-based arithmetic, not the original private
extraction, full matching pipeline, training, qualification or timers.

This is a prepared reviewer snapshot, not a claim of venue attachment, public
release or dataset/model redistribution permission.
