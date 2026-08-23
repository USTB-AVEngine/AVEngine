# Apartment QA design viewpoint wiring checkpoint (2026-08-24)

## Scope and concrete failure

`tools/qa/design_qa_batch.py` previously authored every point with the legacy
Apartment camera `(-70, 65, 147.1)` / yaw `-145`, the global straight-corridor
bank, and one fixed off-screen corner. Its `spec.json` had no `viewpoint_id`.
Rotating only the later question generator therefore could not rotate visual or
listener authority, and the old corner could become visible from another pose.

The only current multi-view artifact is the deterministic 2-D raster prefilter
from `tools/m6x/build_apartment_viewpoint_bank.py`. It explicitly carries
`native_pixel_validation_status=pending`; Git identity, numeric types, M1 schema
validation, and ordinary unit tests do not establish native pixels, composition,
side calibration, or off-screen visibility. This slice rejects that specific
draft-to-production failure. It adds no hash, frozen schema, baseline, or general
publication gate.

## Implemented boundary

- Production design requires an explicit declared-pass `--viewpoint-bank` containing 4–8
  uniquely identified viewpoints in bank order. `--viewpoint-id` may explicitly
  select one; otherwise primaries rotate deterministically by cell and point,
  and each attribute twin inherits its source point's viewpoint.
- Every accepted viewpoint must carry native-pixel status `pass`, a passing
  authentication record with existing evidence, a valid corresponding M1
  request, a passing corridor subset with evidence, and a passing per-view
  off-screen point with evidence. The bank camera is cross-checked against the
  M1 camera at `1e-6`; timeline author arguments are then taken from that M1
  authority.
- Corridor geometry is read per viewpoint. Both Apartment UE-centimetre and
  prefilter Habitat-XZ-metre records are normalized to UE centimetres before
  planning. Side `c` is recomputed from the selected camera yaw and position.
- Every `spec.json` records `viewpoint_id`, viewpoint authority, and the exact M1
  request path/id. The manifest records selection mode and per-view point counts.
- Output is assembled in a sibling staging directory and published through the
  existing atomic no-replace directory publisher. Any timeline-author failure
  removes staging and publishes nothing.
- The historical `apartment_view0` path remains available only through the
  explicit `--legacy-view0-report-only` option. It requires the old off-screen
  point to be supplied explicitly and is never a production fallback.

## Evidence and remaining native work

CPU tests cover pending-prefilter refusal, 4–8 completeness, unique IDs,
evidence presence, M1/camera agreement, deterministic rotation and twin
inheritance, per-view off-screen positions, yaw-dependent side calculation,
timeline camera arguments, explicit single-view selection, and the absence of a
silent legacy fallback.

The `declared_certified_viewpoint_input` authority label means only that the
input bank declares the required reviews complete. The loader verifies those
declared `pass` states, validates the M1 request,
and requires every evidence reference to resolve. File existence alone is not a
new native authentication result: the bank producer and human review remain the
authority for promoting those records to `pass`, and the synthetic unit-test
evidence files are fixtures only.

No authenticated 4–8-view bank is fabricated or committed here. No UE, GPU,
native pixel, audio, RIR, or mini-pilot run was performed, and no existing
artifact was modified. Human Studio placement, calibration frames, per-view
four-point visual/audio pilots, and promotion of their evidence to `pass` remain
required before a fresh multi-view production batch can run. The formal dataset
denominator remains zero.
