# Task 1 Development Log — Unified QA Episode Data Structure

**Date**: 2026-08-10 to 2026-08-11
**Person**: A (QA Pipeline)
**Branch**: `feature/qa-episode-v1` (from `origin/integration/lifelike-engine-v1`)

## Overview

Task 1 defines the unified QA Episode — a content-hash-bound document that wraps
Timeline v2 output with structured per-frame facts (sound, spatial, motion,
visibility) and validated question/answer pairs.  It is the single source of
truth for all downstream QA consumers (training, evaluation, audit).

## Deliverables

| File | Description | Status |
|------|-------------|--------|
| `src/avengine/qa/__init__.py` | Package init with module docstring | ✅ |
| `src/avengine/qa/schemas/qa_episode_v1.schema.json` | JSON Schema Draft 2020-12 | ✅ |
| `src/avengine/qa/episode.py` | Python data model (~550 lines) | ✅ |
| `tests/unit/test_qa_episode.py` | 32 unit tests across 6 classes | ✅ |

## Schema Design Decisions

### 1. Schema Constant
- `"avengine_qa_episode_v1"` — immutable, no backwards compat without
  bumping the constant and writing a migration guide.

### 2. Content Hash Binding
Every Episode carries `episode_content_sha256` computed over a canonical
JSON serialization (sorted keys, deterministic number formatting).  The
hash covers the full document **except** the `episode_content_sha256` field
itself, allowing verification before the hash is set.

### 3. Required Top-Level Fields
- `schema`, `episode_id`, `created` — identity
- `assets_used` — actors (1-4) + sounds (1-8)
- `scene` — room identity, optional furniture occluders
- `timeline` — Timeline v2 blob (external schema constraint)
- `facts` — sound, spatial, motion, visibility, events
- `qa_pairs` — array of validated question/answer pairs
- `sidecars` — paths to render artifacts
- `provenance` — seed, commits, registry hashes
- `episode_content_sha256` — content integrity hash

### 4. Visibility Model
Four discrete states, classified per-frame:
- `out_of_view` — not in frustum, or amodal_pixels == 0
- `visible_clear` — visible_fraction ≥ 0.90 (default threshold)
- `visible_occluded` — 0.05 ≤ visible_fraction < 0.90
- `fully_occluded` — visible_fraction < 0.05

Thresholds are configurable via `clear_threshold` and `visible_threshold`.

Each frame records `amodal_pixels`, `visible_pixels`, `visible_fraction`,
`touches_frame_border`, an optional `bbox_visible`, and a list of
`occluders` (actor, furniture, or unknown_static).

### 5. Event Detection
Six event types derived from per-frame visibility state transitions:
- `enter_frustum` — out_of_view → any visible state
- `exit_frustum` — any visible state → out_of_view
- `become_visible` — out_of_view → visible (paired with enter_frustum)
- `occlusion_start` — visible_clear → visible_occluded
- `fully_occluded` — visible_occluded → fully_occluded
- `reappear` — fully_occluded → visible_occluded+

### 6. QA Pair Structure
Each QA pair has:
- `question_id`, `question_type`, `question_text`, `answer_text`
- Optional `choices` (multiple-choice distractor set)
- Optional `answer_source` (fact_path traceability)
- `validation` block: `answer_unique`, `fact_observable`,
  `distractor_check`, `rejection_reason`

QA pairs with `answer_unique=false` are rejected at `build()` time
unless they carry a `rejection_reason`.

### 7. Occluder Discrimination
Occluders use `occluder_type` with conditional validation:
- `actor` → requires `actor_id`
- `furniture` → requires `instance_id` + `semantic_label`
- `unknown_static` → no additional required fields

## Python Module Architecture

### Episode (mutable builder)
```python
ep = (Episode("ep_001")
      .add_actor("a1", "beagle_01", "dog", 1, breed_id="beagle", ...)
      .add_sound("bark_01", "dog_vocalization", "a1", ...)
      .add_furniture_occluder("table_01", "table", 100))
ep.timeline = valid_timeline
# Populate per-frame facts...
doc = ep.build()  # validates + binds content hash
```

Key methods: `add_actor()`, `add_sound()`, `add_sound_fact()`,
`add_visibility_record()`, `add_furniture_occluder()`, `add_event()`,
`add_qa()`, `build()`.

### VisibilityRecord (frozen dataclass)
Immutable value object for a single frame's visibility measurement.
Includes `as_dict()` for serialization.

### EpisodeEvent (frozen dataclass)
Immutable value object for detected visibility transitions.
Includes `as_dict()` for serialization.

### QAPair (frozen dataclass)
Immutable value object for a question/answer pair with validation metadata.
Includes `as_dict()` for serialization.

### Free Functions
- `classify_visibility(amodal_pixels, visible_pixels, in_frustum, *, clear_threshold=0.90, visible_threshold=0.05) -> str`
- `make_visibility_record(amodal_pixels, visible_pixels, in_frustum, *, ...) -> VisibilityRecord`
- `detect_visibility_events(visibility_frames, actor_id) -> list[EpisodeEvent]`
- `validate_qa_episode(episode) -> list[str]` — structural + schema
- `validate_qa_episode_schema(value) -> list[str]` — schema only

## Test Coverage

| Test Class | Tests | Focus |
|-----------|-------|-------|
| `TestSchemaValidation` | 4 | Round-trip, missing fields, hash mismatch |
| `TestEpisodeBuilder` | 9 | Timeline, actors, sounds, visibility, furniture, sidecars, QA validation, hash stability, provenance |
| `TestVisibilityClassification` | 7 | All states, thresholds, auto-classification |
| `TestEventDetection` | 7 | Enter/exit frustum, occlusion, reappear, no-op, canary sequence |
| `TestQAPair` | 3 | Minimal, choices, rejection |
| `TestEpisodeEvent` | 2 | Serialization, occluder attachment |
| **Total** | **32** | All passing |

## Issues Encountered & Resolved

1. **M5 schema constraint: `semantic_anchor_id` must be `"muzzle"`**
   The M5 episode request schema defines `semantic_anchor_id` with `const:
   "muzzle"`.  Test fixtures used `"mouth"` which failed validation.
   Fixed by changing to `"muzzle"`.

2. **SHA256 regex: only `[0-9a-f]` allowed**
   The stable_id and sha256 patterns use `[0-9a-f]`, not `[0-9a-fA-F]`.
   Test fixtures using `"g"` or uppercase hex failed.  Fixed by using
   valid lowercase hex.

3. **Schema `minProperties: 1` on nested objects**
   Both `spatial_frame.actors` and `motion_frame.actor_states` require
   `minProperties: 1`, meaning `{}` is rejected.  Same for `actor_visibility`.
   All test fixtures now populate non-empty per-frame dictionaries.

4. **Schema `minItems: 1` on `assets_used.sounds`**
   Even a "minimal valid episode" requires at least one sound asset.
   Tests that only needed actor/sidecar/visibility data still need a
   sound to pass schema validation.

5. **Sidecar paths require `minLength: 1`**
   Empty string sidecar paths fail schema validation.  The builder
   defaults to `""` for all sidecars; tests that don't need sidecar
   paths must still set non-empty values.

6. **Content hash mismatch test**
   The test expects `content_sha256` in error messages, but the actual
   error uses the field name `episode_content_sha256`.  (This test
   passes — the `"content_sha256" in e` substring check works.)

## Next Steps (Task 2: Target-Only Pass)

With the Episode data structure in place, Task 2 will:
- Implement a `--target-only` rendering pass in Habitat that renders
  only the target actor's semantic ID
- Record `amodal_pixels` from the target-only pass
- Record `visible_pixels` from the standard RGB/semantic pass
- Compute `visible_fraction` and classify visibility state
- Feed results into `VisibilityRecord` and `detect_visibility_events()`

## B → A Interface Reminder

Person B must deliver (per `A_TASK_PLAN.md`):
- Entity Asset Registry with `entity_asset_id`, `semantic_id`, species/breed
- Sound Asset Registry with `sound_asset_id`, `semantic_sound_class`
- Scene Room Catalog with `room_id`, `room_provider`, furniture definitions
- Runtime profiles for beagle/human scaffolding

Person A produces:
- QA Episodes (this module's output) back to B for canary validation
- Feedback on asset quality issues (skeleton drift, texture gaps, etc.)
