# QA v2 T5 DoA, refusal, and off-screen checkpoint (2026-08-24)

## Scope

This slice adds strict question-generation support for three work-order items:

- T5 numeric listener-local DoA with the approved `+/-15 deg` full-credit and
  `+/-30 deg` partial-credit bands;
- native-pixel-certified off-screen left/right questions;
- a deterministic 10--15 percent audio-only refusal quota.

It extends the question generator only. No fresh production batch, trainer
modality masking, new native capture, or dataset admission is claimed here.

## Evidence authority

T5 and off-screen generation require a per-point native capture directory with
both `frame_records.json` and `research_receipt.json`. The loader requires:

- exactly 75 contiguous frames at 15 fps and 3,200 ticks/frame;
- passing camera/source root readback summaries;
- `research_only=true` and `episode_counted=false`;
- camera and actor anchor/state agreement;
- a per-frame camera match to the resolved viewpoint at `1e-6`.

T5 gold uses the emitting source's UE anchor readback and camera pose readback
at the second event frame. Azimuth is listener-local in `[-180, 180]`, positive
to the right. The rows also use the repository's existing numeric-banded field
shape (`format`, `answer_numeric`, `unit`, `scoring_bands`). A callable scorer
parses one unambiguous numeric prediction, applies circular wraparound, and
returns `1.0`, `0.5`, or `0.0` at the approved 15/30-degree boundaries.

Off-screen questions additionally require the existing
`avengine_qa_pixel_visibility_truth_v1` artifact with
`computed_modal_target_only_v1` status. An event is accepted only when the
same-renderer target-only record says `state=out_of_view`,
`visible_pixels=0`, and `target_pixels=0` for every authored frame touched by
the complete sound event. A start-frame sample, center-frustum estimate, or
authored `offscreen_candidate` flag alone is insufficient.

## Refusal quota

Refusal rows ask for a visual attribute while explicitly providing audio only;
the gold answer is `cannot determine from audio alone`. The selector finds an
integer count whose final-batch fraction is within 10--15 percent and chooses
evenly spaced point candidates deterministically. It never deletes ordinary
questions to meet the quota.

The rows carry `provided_modalities=[audio]` and `withheld_modalities=[video]`.
The downstream dataset/collator must still enforce that modality mask before a
real training batch can count as complete.

## Verification boundary

CPU tests cover:

- 75-frame UE receipt/readback validation and viewpoint mismatch rejection;
- positive-right azimuth and event-frame conversion;
- native target-only `out_of_view` enforcement;
- per-instance native pixel frame ordering and state/count consistency;
- T5 scoring metadata and readback-derived numeric gold;
- executable circular numeric scoring, including +/-180-degree wraparound;
- off-screen binary answer-key accounting;
- deterministic in-range refusal quota without base-question deletion;
- fail-closed behavior when native evidence roots are absent;
- fail-closed handling for empty events and malformed endpoint lists.

With all three extensions disabled, an in-memory recompile of the retained
192-point input produced the unchanged `909 accepted / 27 refused`; accepted
and refused rows were byte-equivalent after removing the already documented
additive `answer_arity`, `semantic_answer_source_slot`, and `viewpoint_id`
fields.

A second read-only in-memory run enabled T5 against all 192 retained UE capture
directories. All 192 receipts/readbacks and camera cross-checks passed, yielding
192 T5 rows over 192 unique points. Numeric gold ranged from `-40.5 deg` to
`+34.9 deg` (`100` left, `92` right), and every row carried the same approved
`+/-15 deg` / `+/-30 deg` circular scoring record. This proves the T5 readback
path on real retained captures, but it remains an old single-view research
batch; the overall legacy binary answer-balance report still fails and no new
output was published.

Enabling T5 and refusal together on the same retained inputs produced `1,101`
ordinary rows plus `157` refusal rows (`1,258` total), a refusal fraction of
`0.124801`. The 157 rows came from 157 unique points, all declared audio-only
with video withheld. This remains an in-memory generator check: the trainer has
not yet enforced the modality mask and therefore this is not an end-to-end
refusal training result.

A fresh multi-view native batch is still required to prove real T5/off-screen
questions and the 10--15 percent refusal quota end to end. All outputs remain
research-only and outside the formal dataset denominator.
