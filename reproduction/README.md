# Anonymous numerical reproduction artifact

This package reproduces numerical paper values from de-identified, saved scalar
readouts. It does not train, download or construct a model, collect a dataset,
rerun tokenization, reconstruct raw corpus masks, or reproduce the original
training process.

## Run

Use Python3.11 (validated3.11.9) with NumPy1.26.4. No other third-party dependency
is needed. Set PYTHONUTF8=1 and PYTHONDONTWRITEBYTECODE=1. The code pins numerical
threads to one.

From this directory:

    python -B code\test_core.py
    python -B code\reproduce.py

The full command first exactly replays the original eleven5000-draw primary
results, then the fixed unweighted-count check, then the *same*100000-draw
sensitivity sequence. This is a deterministic reproduction, not another selected
Monte Carlo run. It also replays previously reported paper summaries and generates
six LaTeX fragments and two vector PDFs. The eight paper assets must match the
provided byte hashes. Outputs go under `reproduced`.

Use `--resume` after an interrupted full run to resume the exact saved RNG state.
Use `--quick` to verify the original5000, count and existing paper scalars without
replaying100000 draws; its report explicitly records that limitation.

## Scientific contract

Individual-benefit inference clusters are24 main blocks and8 blocks per transport. Scores are
compared with the mean individual oracle at the other two optimizer seeds in
the same background. Focals, backgrounds and seeds are not extra independent
units. The donor study has48 main targets, equally weighted and resampled within two fixed
24-target strata.

The main and two transport settings contain1920 unique focal/context cells.
G8 reuses the specified main block/oracle rows and adds no independent blocks;
its anonymous reference indices are included. Main individual-benefit controls
are nearest lexical same-label, high-relatedness alternatives. Magnitude errors
are displayed only for G20/A. Difficulty and corpus
scores are uncalibrated predictors despite sharing NLL units.

The post-hoc count score is saved uncovered complete trailing-k4 endpoints/255.
The denominator includes ordinary scored EOS; incomplete first3 content positions,
EOS and padding have zero numerator. The original same-source OR duplicate-group
exclusion remains fixed. No k, weights, signs, masks or thresholds were tuned.
Rank uses averaged ties; top-two ties use the saved focal vector order. Any
undefined rank invalidates the complete metric rather than being dropped.

Positive paired rank contrasts mean left minus right. Positive paired error and
regret improvements mean right minus left. MedianAE is a median of absolute errors;
paired medianAE effects are differences of medians, not mean absolute error or
medians of signed differences. No magnitude error is evaluated for count/255.
Model/tokenizer NLL scales are not pooled.

All original eleven primary tests and the single Bonferroni family remain intact.
Original5000-draw decisions are preserved. The fixed100000-draw sensitivity adds
no independent blocks or targets. Both new CPU checks are post-hoc; no overlapping
interval or nonsignificant result establishes equivalence. The frozen main k4
comparison is not erased if unweighted coverage is competitive.

## Inputs and provenance

`data\cpu_inputs.json` contains only the exact anonymized scalar arrays needed for
the two checks. `data\asset_scalars.json` contains only additional scalar vectors
needed to replay the paper's existing summaries. `data\paper_values.json` records
the original scalar estimates/intervals and source-file hashes. Costs are saved
aggregate wall-time telemetry, not simulated FLOPs, energy or campaign elapsed
time. Existing summaries are retained, with a1e-10 replay tolerance for scalar
rounding; the eleven original primary estimates/intervals match exactly.

`protocol.json` preserves the pre-computation law, seeds, joins and endpoint sets.
The bound private protocol's digest and sanitized source hashes are included;
private paths and the reverse pseudonym map are deliberately absent. Anonymous
ordinal labels preserve original cluster ordering and focal tie order.
`MANIFEST.json` hashes every shipped numerical input, reference and source file.
Its digest is recorded in `MANIFEST.sha256`. Local `reproduced` outputs are not
part of the immutable input manifest.

The mask checks reported in provenance were performed on the original saved
tokenized evidence. This artifact can check counts, denominators, scalar joins and
downstream calculations but cannot independently reconstruct source/duplicate
exclusion masks without the intentionally omitted raw source material.

## Scope and redistribution

No raw documents, token sequences, source-query identifiers, model weights,
machine/people identities or reviewer opinions are included. This is local
anonymous numerical staging, not an upload or license grant for the original
datasets or checkpoints. Authors must resolve the numerical artifact's release
license before distribution; reproducing these scalars requires no assumption
that a model or dataset can be redistributed.
