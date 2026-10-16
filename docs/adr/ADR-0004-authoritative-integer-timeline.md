# ADR-0004: Authoritative Integer Timeline

- Status: Accepted
- Date: 2026-07-16

## Context

Free-running animation and view-outer capture gave different cameras different
gait phases while audio was reused. Floating clocks and fixed samples-per-frame
rounding cannot prove exact five-second alignment.

## Decision

Use one 48 kHz integer timeline as authority for root/joint poses, contacts,
semantic emitters, camera capture, video PTS, audio samples and events. Preserve
timeline v2 unchanged.

## Alternatives considered

- Free-run animation and mux with `-shortest`.
- Independent audio/video clocks reconciled after rendering.
- Floating-point seconds as the stored authority.

## Consequences

All systems require exact seek/evaluation APIs and semantic cross-field
validation. Five seconds maps exactly to 75 video frames and 80,000 samples.

## Validation plan

Validate frame/sample coverage, referential integrity, same-frame pose hashes,
event/contact alignment and exact output counts after readback.

## Reversal criteria

New rates or episode lengths require a new versioned schema and migration, not
a silent semantic edit to v2.
