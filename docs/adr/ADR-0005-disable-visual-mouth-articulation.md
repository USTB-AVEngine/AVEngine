# ADR-0005: Disable Visual Mouth Articulation

- Status: Accepted
- Date: 2026-07-16

## Context

Visible mouth motion could reveal the vocalizing identity directly and create
an undesirable visual shortcut in source-grounding experiments. The current
asset pipeline does not provide audited facial animation.

## Decision

Disable visual mouth articulation. In timeline v2, `open_ratio` is `0.0` and
`vocalizing` describes only audio-event activity. Episode manifests record
`disabled_for_shortcut_control`.

## Alternatives considered

- Generate lip sync or visemes.
- Remove mouth fields from timeline v2.
- Leave mouth state unspecified.

## Consequences

The dataset can study AV grounding without a direct mouth-open cue. Papers and
README files must not claim lip synchronization. Vocalization still anchors to
the head/muzzle semantic emitter.

## Validation plan

Reject episodes with nonzero `open_ratio` or pose differences caused by
vocalization assignment. Counterfactual pairs must retain identical visuals.

## Reversal criteria

A future facial-articulation task requires a new schema/ADR and a separate
benchmark condition; it must not alter v2 samples retroactively.
