# Reference definitions and scalar-replay boundary

The source locations below are **relative logical locations**, not runtime
dependencies. Exact origin-file hashes are in `inputs/SOURCE_ORIGINS.json`.
All construction checks concern the frozen inputs, not an outcome-fitted recipe.

## Manuscript-ready definitions

**Matched-budget reference scores.** Each reference score is the focal's
per-token natural-log NLL under a reference model minus its NLL under the
matched ALL_IN model. All fits retain the 768-row budget, original training
recipe and optimizer seed/order law. The G5/G20 interventions use their original
matched controls and physical training slots; they are not deletions. The
half-background reference retains those controls, whereas the role-disjoint
reference changes them as well as the background rows. Readouts use exactly
255 shifted active targets including true EOS. G20 replaces all twenty members
with their matched controls while retaining the 748 background rows. The
individual-effect labels remain the original one-pair replacement contrasts.

**Reference-group composition.** The twenty original member/control pairs are
partitioned into **four groups of five**, with **exactly two evaluated focal
members per group**. A group-of-five reference replaces all five members of
the focal's assigned group with their respective controls, including the
focal's own matched control. Each focal is scored using its assigned reference,
not an average over the four fits.

Each policy therefore costs **four reference fits per context** (four coherent
and four dispersed fits, eight across the two policies), not one fit per policy.

“Coherent” and “dispersed” are labels for an input-only four-gram-support
construction, **not semantic clustering or an outcome-trained optimizer**.
The frozen PCG64 procedure generates 256 candidate partitions by separately
permuting the eight focal positions into pairs and twelve nonfocal positions
into triples. For each query, the candidate objective counts the union of
four-gram completion endpoints supported by the other four companion members,
minus the union supported by their matched controls. It then averages over
the eight focal queries with denominator 255. The query's own focal member
**and own matched control are excluded from this companion-support objective**;
rows with the same source identity or duplicate group as the query are also
excluded. This support exclusion does **not** exclude the focal control from
the replacement training intervention.

The coherent partition maximizes this fixed net-support criterion; the
dispersed partition minimizes it. Ties use the lexicographic partition key of
member-row IDs. The input-derived seed does not depend on optimizer seed or
scientific outcomes, so a block/background's partition is shared across the
three optimizer seeds. This numerical artifact preserves the selected
partitions; it does not rerun their token-level candidate search.

**Half-background replacement reference.** Start from ALL_OUT: the twenty
member slots already contain matched controls. Choose 374 of the **748
background slot positions** and replace those rows with 374 eligible fresh
rows. Retain the other **374 original background rows plus all twenty
matched controls**. Thus this reference shares **394/768 rows with ALL_OUT**
and **374/768 rows with ALL_IN**. It changes 374 positions relative to ALL_OUT
and 394 relative to ALL_IN (the latter also counts the twenty member-to-control
replacements). “Half-overlap” is faithful only as shorthand for **half of the
original background**; it does not mean half of all 768 rows. “Half-background
replacement reference” is the more precise label.

**Role-disjoint reference.** All 768 rows come from the frozen eligible shipped
pool, excluding source identity or duplicate group against the selected P5
query/member/control/background roles across settings. The half-reference's
374 fresh rows are the first 374 rows of this 768-row draw: the two reference
training sets intentionally **share 374 fresh rows**. “Disjoint” therefore
means disjoint from the specified original P5 roles, **not** disjoint from
the half reference, globally unseen, or absent from pretraining, P4 or every
near-duplicate relation. “New/fresh” here means eligibility for this selected-P5
source/duplicate-group draw, not a global novelty claim.
Different blocks draw from common finite pools and can overlap; they are not
claimed to use globally mutually disjoint corpora.

**Q1 is a joint reference-training-data change.** G20 versus the role-disjoint
reference changes **all 748 background rows and the twenty matched controls:
all 768 slots**. It is not an isolated background-information intervention.
Use “matched reference versus role-disjoint reference” or “reference training
data,” not a claim that Q1 alone identifies a background-only effect. The
half-background arm retains controls and changes 374/748 background rows, but
its comparison is descriptive, not an additional registered inferential test.

**Fixed reference reuse.** A recipient uses the disjoint model from the **next
original block**, at the same background and optimizer-seed label: modulo24
for the main population, modulo8 within each transport. The donor's supplemental
readouts score the recipient's eight focals. Do not choose a favorable donor,
skip development blocks, or advance within only an available/evaluation subset.
No extra reference fit is charged for these already-computed donor readouts.

## Explicit slot-to-row audit

The source audit maps every stored `H_background_slots` integer through the
original ALL_OUT `train_ids` array to obtain the **actual original row ID**.
It verifies that the same ID occupies that ALL_IN background slot, that the
corresponding fresh ID occupies precisely that slot in the half reference,
and that every complementary background slot is unchanged. It separately
maps each member slot to its matched control and checks all twenty retained
controls. It never compares a set of slot integers with a set of row IDs.

All 120 scheduled contexts passed. `inputs/CONSTRUCTION_AUDIT.json` contains
the counts and hashes of the explicit slot-to-original/replacement-row maps.
Full native row IDs remain outside this anonymous artifact. This numerical
replay checks projected scalar/partition identities; it does not reconstruct
training-row assembly from omitted row files.

## Input-only composition contrast and IMDb limitation

The composition manipulation is the coherent-minus-dispersed **net companion
k4 support fraction**, not loss, rank or regret. The original feasibility gate
averaged backgrounds **0 and 1 within each block**. Only the main population
was gated: at least 12/24 main block means had to reach 0.02; all 24 did.
IMDb was **not gated** by that requirement. This gate was not redefined after
the background0-only execution amendment.

| Input-only scope | Main support-contrast range | IMDb support-contrast range |
|---|---|---|
| Original feasibility: each block's mean over backgrounds0 and1 | 0.04485294117647059 to 0.09142156862745097 | 0.006127450980392157 to 0.017892156862745097 |
| Executed background0: each block/background assignment | 0.043627450980392155 to 0.09215686274509804 | 0.0058823529411764705 to 0.018627450980392157 |

Thus shorthand ranges approximately **0.045–0.091 main / 0.006–0.018 IMDb**
refer to the **original both-background feasibility**, not the executed
background0 projection. The latter ranges are shown separately, derived
directly from the corresponding frozen assignment numerators.
The ungated, smaller IMDb input contrast limits comparability of the
composition manipulation. It is not an outcome-based explanation, a new
test, a reason to omit IMDb, or evidence of equivalence or a between-corpus
effect difference. `inputs/INPUT_FEASIBILITY.json` and the generated
feasibility tables preserve both scopes and every block.

## Targets, support association and descriptive limits

- Prediction from seed s is evaluated against the arithmetic mean of the
  original individual effects from the **other two seeds in the same
  background**. Same-realization quantities use the matched seed separately.
- E01 companion error is **mean** absolute error, coherent minus dispersed,
  not the older study's median-error statistic. Selection regret chooses two
  of eight focals; score ties use the frozen focal-vector order.
- Q3 is exactly, **within each eight-focal context**,
  `Spearman(disjoint score - G20 score, ORIGINAL ALL_OUT covered fraction)`,
  where the k4 covariate is **`1 - uncovered_count/255`**, including its
  four-position structural offset. It is not support measured in the disjoint
  reference. It is a score-difference association, not causal mediation or
  “more support makes the matched reference help ranking more.”
- Means average the three scheduled seeds within each original block and then
  equally average complete blocks. The inference family stays eleven tests,
  including six unavailable main comparisons.
- The descriptive stability data include the contrary E01 selection seed
  exception: seed0 is **+0.0037440015281149652**; seed1 is
  **-0.04022950915775216** and seed2 **-0.047478320797564706**.
  Main block signs are 17 negative, six positive and one zero. These are
  descriptive checks, not extra significance tests; favorable Q1/Q3 stability
  is not reported in isolation.

## Source locations

| Definition | Logical source and lines | Origin ID |
|---|---|---|
| Net k4 endpoint/exclusion rule | `reference_inputs/core.py:236-255` | `k4_support_rule` |
| Input seed, constrained candidate partitions, extrema and ties | `reference_inputs/core.py:322-411` | `E01_construction` |
| Original two-background, main-only feasibility gate | `reference_inputs/core.py:450-478` | `inputs/INPUT_FEASIBILITY.json` gate-source hash |
| Half/disjoint draw, slot replacement and shared fresh rows | `reference_inputs/core.py:690-718` | `E03_construction` |
| Invariant budget, matched controls and G5 two-focal assignment | `reference_inputs/assemble_inputs.py:145-202` | `budget_and_reuse` |
| Next-original-block mapping and recipient readouts | `reference_inputs/assemble_inputs.py:206-247` | `budget_and_reuse` |
| Scalar scores, original-ALL_OUT Q3 and fixed endpoint inference | `analysis/endpoint_analysis.py:223-399` | `accepted_endpoint_rules` |
| Mean, ranks, focal-order regret and stable seed | `analysis_bootstrap.py:18-59` | `frozen_bootstrap_helper` |
| Frozen block-resampling/linear-percentile law | `analysis_bootstrap.py:113-186` | `frozen_interval_law` |

The construction source hashes were checked against the input build record,
and frozen assignment/draw hashes against the sealed input files. The build
record itself is deliberately not distributed as an administrative file.
