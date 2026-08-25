# QA v2 per-type answer balance checkpoint (2026-08-24)

## Scope

This slice makes question publication fail closed unless every recognized binary
question type is balanced to `50/50 +/- 1`. It also provides the exact design
variant selector that a later `design_qa_batch.py` wiring slice will call.

This is a question-generation and selection checkpoint only. It does not claim
that Apartment viewpoint rotation, design-time point replacement, held-out
generation, or any native visual/audio pilot is complete.

## Implemented behavior

- The normal CLI path is strict. `--report-only` is the sole explicit path for
  auditing an unbalanced retained batch without publishing it as balanced.
- Every point resolves its own `spec.viewpoint_id` through an explicit viewpoint
  bank. Strict mode rejects missing, duplicate, or unknown viewpoint IDs.
- Known binary questions reject non-two-option, duplicate-option, and
  answer-not-in-options rows. Numeric/non-binary questions must declare their
  answer arity explicitly.
- The design selector uses exact dynamic programming. An optional resource limit
  reports `search_limit_exhausted`; it is not misreported as mathematical
  impossibility.
- Publication uses sibling staging and the existing atomic no-replace directory
  publisher, including cleanup on serialization failure or a publication race.

## Verification

The server worktree was tested with its configured AVEngine runtime:

```text
PYTHONPATH=src .../avengine-habitat-runtime/bin/python -m pytest -q \
  tests/unit/test_answer_balance.py \
  tests/unit/test_generate_qa_v2_questions.py

17 passed
```

The independent read-only review also ran adjacent path-policy/question tests
(`26 passed`), Ruff, and `git diff --check`.

The retained 192-point batch at
`/data/avengine_external/review/qa_v2_batch2_inputs_20260823T065059Z` was
recompiled in explicit report-only mode without publishing over retained data:

- `909 accepted / 27 refused`, unchanged;
- accepted/refused rows remain identical after excluding the newly reported
  `answer_arity`, `semantic_answer_source_slot`, and `viewpoint_id` fields;
- the report continues to expose the existing imbalance: T4 `93/96`, T9
  `87/105`; the other three binary types are balanced.

Ruff passed on all four changed code/test files, and `git diff --check` passed.

## Remaining work

- Wire the exact selector and per-point viewpoint authority into
  `tools/qa/design_qa_batch.py` so imbalance is repaired by choosing legal
  point/path/source variants instead of deleting questions.
- Generate a fresh balanced multi-view batch and the stratified 48-point
  held-out, then run the required native visual/audio/gate review.

All evidence in this checkpoint remains `research_only`; it does not change the
formal dataset denominator or mark any Episode admitted.
