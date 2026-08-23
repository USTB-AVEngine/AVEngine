# Exact 48-point held-out selector checkpoint (2026-08-24)

## Scope

This checkpoint adds the CPU-only selector for the first future multi-view QA
production partition. It consumes an accepted `questions.json` plus the point
`spec.json` directories and publishes four ordinary JSON files:

- `heldout_point_ids.json`;
- `heldout_questions.json`;
- `train_eligible_questions.json`;
- `selection_report.json`.

The selector fixes the held-out size at exactly 48 points. A point and every
question belonging to that point move together; it never drops an individual
question to repair answer balance. The undirected transitive closure of every
`spec.twin_of` reference is the indivisible selection unit, including chains
longer than one direct twin edge.

## Selection behavior

Inputs fail explicitly when a required point/question field is absent,
`twin_of` references an unknown point, question metadata disagrees with its
spec, a binary answer cannot be mapped uniquely to A/B, or a point has no
accepted question. Point specs must declare room, viewpoint, source assets and
pair kind, motion case, program, and twin relationship; the selector does not
invent a room or viewpoint for legacy data.

The search first runs a point-count-only subset-sum check over whole transitive
twin components. It therefore reports “exact 48 is impossible” separately
from the case where 48 whole-group points are reachable but the binary answer
constraints are not. It then solves one mixed-integer linear program (MILP):
one binary variable selects each whole twin component, exact point count is an
equality, and each binary type constrains both held-out and train-eligible A/B
difference to at most one. All questions at a selected point contribute to
those coefficients.

The same MILP globally minimizes equal-dimension mean absolute deviation from
population-proportional counts across room ID, viewpoint ID, canonical
source-asset pair, pair kind, motion case, sound/program, question type, and
binary type-plus-answer key. Absolute-deviation auxiliary variables keep the
problem linear. Integer scaling makes the primary objective integral; a
sub-unit sorted group-rank objective provides the stable deterministic
tie-break without worsening the primary optimum. Input rows and group
variables are canonically sorted before solving.

The implementation uses `scipy.optimize.milp`/HiGHS from the repository's
existing `native` dependency set. Mathematical infeasibility is reported only
for solver status `infeasible`; dependency, solver, or resource failure is a
separate explicit error and is never relabeled as “answer balance impossible.”
This avoids retaining an unbounded Python dictionary of multi-type answer
delta states for several-hundred-point populations.

The report lists population, held-out, and train-eligible counts/fractions for
every category. It also records empty point intersection, zero canonical
question-row intersection, complete question partitioning, and no split twin
component.

Publication uses a fresh sibling staging directory and the repository's
existing atomic no-replace directory publisher. Serialization failure or a
destination race removes only that invocation's exact staging directory and
never replaces the racing destination.

No hash, frozen schema, baseline, or permanent gate was added.

## Hermetic CPU evidence

The server worktree passed the selector tests plus the adjacent answer-balance,
question-generation, and path-policy tests:

```text
41 passed
```

The selector suite covers exact 48 points, transitive three-point twin chains,
whole-point question preservation, point/question/twin zero overlap, all eight
distribution dimensions, per-type A/B balance in held-out and train-eligible
partitions, count and balance impossibility as distinct failures, required
field rejection, input-reorder identity, and a proportional optimum that
requires a two-group-for-two-group change. It also exercises the same solver
over 320 points with six independently varying binary types, fresh no-clobber
publication, publication race cleanup, and serialization cleanup. Ruff passed
on the three new code/test files.

These are synthetic algorithm and filesystem-boundary results only. They do
not prove that a retained native multi-view population exists or passes its
visual/audio review.

## Retained-input dry selection boundary

A fresh dry selection was attempted against the retained batch2 inputs without
modifying them:

- questions:
  `/data/avengine_external/review/qa_v2_batch2_questions_v1/questions.json`;
- specs:
  `/data/avengine_external/review/qa_v2_batch2_inputs_20260823T065059Z`;
- evidence:
  `/data/jzy/tmp/avengine-heldout48-retained-dry-aopxMW5X`.

The command exited 2 with:

```text
spec b001 is missing non-empty room_id
```

It reported `output_written=false` and created no selection directory. This is
the correct fail-closed result: batch2 is a retained legacy single-view batch,
not the required new multi-view production population. No room ID, viewpoint,
answer metadata, or other authority was fabricated to make it selectable.

Therefore this checkpoint does **not** claim a completed real 48-point
held-out. The next data action is to generate the fresh accepted multi-view
questions/specs with all declared dimensions, then rerun this selector into a
new no-clobber output and review its distribution report.

All outputs remain `research_only`, `episode_counted=false`, and formal Episode
count zero. The earlier pilot48 is recorded only as a weak probe, not the
primary held-out.
