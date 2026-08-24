# QA producer modality-mask checkpoint (2026-08-24)

## Scope

This producer-only slice is implemented on the server worktree
`/data/jzy/code/AVEngine-codex-qa-modality-mask`, branch
`codex/qa-modality-mask`, from integration `2b0d4f9`.

The code commit is `a3994ca408e4a674dd1474bb596b35f3df2e5c4d`.
Only `tools/qa/generate_qa_v2_questions.py` and its unit test were changed in
that commit. No region-planner file, answer rule, onset schedule, candidate
retention rule, or refusal-quota algorithm was changed.

## Producer behavior

Every accepted ordinary question and every selected refusal question now has
an explicit `modality_mask` with exactly boolean `audio` and `video` keys.
Rows rejected by an existing question gate also receive the mask before being
placed in the report's `refused` list.

The mask is derived from explicit facts, in this order:

- row-level `provided_modalities`/`withheld_modalities` or `modality_mask`;
- point-level `input_modalities` or `modality_mask` when the row has no more
  specific declaration; and
- the validated point AudioProgram's non-empty event list for the audio-input
  fact.

Missing modality facts fail closed. The producer never assumes that an
ordinary row is dual-modal merely because it is an ordinary row. A row-level
refusal may intentionally withhold video even when the point's underlying
input set contains video; it must still resolve to exactly
`{"audio": true, "video": false}`. Ordinary rows whose expectation is
`dual_required` or `vision_dominant_control` must resolve to
`{"audio": true, "video": true}`. Conflicting declarations, unsupported
modality names, duplicate/overlapping lists, non-boolean masks, and a refusal
mask that exposes video fail before publication.

The existing question text, answer, question type, onset/timing fields,
answer-balance report, candidate filtering, and 10--15% refusal quota are
unchanged. The mask is descriptive input metadata, not a new gate.

## Verification

The focused plus adjacent CPU suite passed `63 tests`:

```text
PYTHONPATH=src:. /data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python -m pytest -q \
  tests/unit/test_generate_qa_v2_questions.py \
  tests/unit/test_answer_balance.py \
  tests/unit/test_extended_questions.py \
  tests/unit/test_heldout_selector.py \
  tests/unit/test_design_qa_batch.py
```

The exact test log is
`/data/jzy/tmp/qa_modality_mask_20260824_tests_r1.log`.
Ruff check/format, byte compilation, `git diff --check`, and clean-worktree
evidence are in
`/data/jzy/tmp/qa_modality_mask_a3994ca_quality_r1.log`.

Tests cover ordinary dual-modal rows, audio-only refusal rows, explicit
audio-only derivation for an audio-sufficient question, missing-fact failure,
normal dual-modal conflict, refusal conflict, existing refusal quota and
output-count preservation, plus the existing T5/off-screen, balance, held-out
and design-batch suites.

## Retained-input boundary

A fresh/no-clobber CPU CLI run was attempted with
`--report-only --enable-refusal` against the retained batch2 input:

`/data/avengine_external/review/qa_v2_batch2_inputs_20260823T065059Z`

The run correctly stopped at the first legacy point because
`b001/spec.json` has neither an input modality declaration nor a row-level
modality fact:

`b001: modality_mask cannot be derived without explicit
provided/withheld_modalities or input_modalities`

The failure log is
`/data/jzy/tmp/qa_modality_mask_retained_20260824_r1.log`; its output
directory was not created. Therefore this slice makes no retained-batch claim
about refusal percentage, answer counts, or per-row masks. No legacy field was
invented to make that batch run.

## Explicit non-claims

This commit proves producer serialization and validation only. Trainer/collator
consumption has not been verified end to end. T5, formal off-screen,
exact-48 held-out, current accepted multi-view native input, and formal dataset
admission remain pending. No GPU job ran, and no hash, frozen contract,
baseline, or permanent gate was added.
