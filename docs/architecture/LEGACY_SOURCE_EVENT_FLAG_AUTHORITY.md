# Legacy Source/Event/Flag Authority

Status: M6 compatibility authority audit  
Scope: existing M5.1 source/event/flag v1 semantics and the bounded M6 adapter  
Not in scope: a final dataset-item schema, natural-language QA, SelectTSL labels, or a replacement dense motion schema

## Decision

M5.1 v1 remains the compatibility authority for the migrated legacy AVEngine source, event, trajectory and flag semantics. M6 adds a data-driven definition registry and provider adapter; it does not rename v1 fields, change thresholds, coerce unknown facts to false, or replace the existing evaluator with required per-frame velocity records.

The stable relationship is:

```text
Room/entity/source providers
  -> compact authoritative facts
  -> M6 compatibility adapter
  -> existing M5.1 flag IDs, thresholds and tri-state assessments
  -> existing OR/AND clip aggregation
```

## Executable authority chain

| Layer | Authoritative for | Not authoritative for |
|---|---|---|
| `examples/m5_1/legacy_apartment/route_manifest.json` | Legacy camera route, actor/source root routes, frame slice and route identities | Actual animated mouth/muzzle joint after capture |
| `schemas/m5_1_source_manifest_v1.schema.json` | Manifest structure, enums, fixed thresholds and flag assessment shape | Cross-field trajectory/event/hash correctness |
| `src/avengine/m5_1/source_contracts.py` | Semantic validation, exact tick/frame/sample math, trajectory-derived flags, pair flags, tri-state and clip aggregation | Room-specific visibility facts that were never supplied |
| Checked-in M5.1 `source_manifest.json` | Declared source identities, taxonomy, audio provenance, event program, pre-execution paths and assessments | A claim that all visibility or acoustic facts were evaluated |
| Post-capture `actual_emitter_trajectory_record` | Actual animated link trajectory used by retained acoustics/delivery evidence | The original route declaration |
| `delivery.py`, top-down/review overlays and muxed videos | Human-readable presentation of validated facts | Source, event, trajectory or flag truth by themselves |

`source_contracts.load_source_manifest()` is fail-closed. It resolves the actual route authority, verifies file/content hashes, validates every keyframe against the bound route plus the declared Y emitter offset, verifies audio provenance and reconstructs event and flag semantics.

## Legacy origin and migration boundary

The older SPEAR implementation is retained under `AVEngine/external/SPEAR`, with the historical locations summarized in `docs/migration/LEGACY_SOURCE_LOCATIONS.md`. Relevant legacy modules include:

- `tools/spike_rlr/flag_definitions.py`;
- `tools/spike_rlr/flag_verifier.py`;
- `tools/spike_rlr/event_constraints.py`;
- `tools/spike_rlr/source_trajectory.py`;
- `tools/spike_rlr/rejection_sampler.py`;
- `tools/spike_rlr/scene_generator.py` and `dataset_runner.py`;
- `tools/spike_rlr/build_review_videos.py`.

The old implementation supplied the useful flag vocabulary and rejection-sampling intent, but represented assessments as flat booleans. It did not preserve unknown state, reasons, evidence or content hashes, and it did not reauthenticate animated emitter links after rendering.

Old SPEAR/UE trajectory code also uses a Z-up world and XY horizontal plane in several paths. Habitat-native M5.1 uses right-handed Y-up world coordinates and the XZ horizontal plane. Old emitter offsets were applied on Z; M5.1 offsets are applied on Y. M5.1 also corrected the camera-local lateral/azimuth sign. Legacy coordinate functions and old AABB visibility output therefore cannot be copied into a Habitat provider.

## Source and trajectory authority

M5.1 distinguishes two valid but different trajectory records:

1. Before execution, a source manifest binds the actor root route and a nominal emitter-height offset. This supports deterministic planning and preflight flag computation.
2. After capture, actual animated mouth/muzzle link positions are recorded and hash-bound separately. This is the authority for runtime acoustic emitter placement and post-capture evidence.

Neither record overwrites the other. A provider must state which one it supplies. A root path cannot be relabeled as an actual joint path, and an actual joint record cannot silently rewrite the checked-in route declaration.

## Event authority

M5.1 events are identified by stable `event_id`, `source_id`, taxonomy and dry-audio provenance. Event windows are half-open:

```text
[start_frame, end_frame_exclusive)
[start_sample, end_sample_exclusive)
```

At the frozen 48 kHz timebase, 15 fps and 16 kHz render rate, the exact nearest-sample frame boundary used by M5.1 frame-bound events is:

```text
B(f) = (3200 * f + 1) // 3
```

The validator reconstructs every `frame_event_state`, rejects overlapping events on one source, recomputes pair overlap windows, and checks deterministic crop, resample and zero-tail-padding arithmetic. Delivery expands these already-validated windows; it does not define them.

M5's earlier Timeline v2 audio windows are sample-authoritative and are not required to land on video-frame boundaries. For example, the retained first M5 call is `[6400, 11200)` samples, or `[19200, 33600)` ticks; its end is not a 15 Hz frame boundary. M6 AudioProgram therefore preserves required sample boundaries and verifies `tick = sample * 3`. Frame activity is derived on demand by testing the frame PTS sample against the half-open event window. It is never produced by rounding an event to a different frame interval.

Timeline v2 deliberately has no `source_id` inside its audio event. Dynamic source routing remains a sidecar/source manifest responsibility. M6 AudioProgram follows that boundary and does not modify Timeline v2. Its checked-in one-active-of-N example reuses all six exact M5 `source0` sample windows and the exact `[3200, 8000)` source slice, while keeping the second registered endpoint silent.

## Frozen flag registry

The public M6 view is checked in at:

```text
examples/m6/registries/legacy_m5_1_flags_v1.json
```

Its validator and access API are in `src/avengine/m6/flags.py`. The checked-in order is inherited from M5.1 and is part of the contract; it is intentionally not replaced by alphabetic sorting.

| Flag ID | Scope | Clip aggregation | Evaluator/fact dependency |
|---|---|---|---|
| `occluded_by_furniture` | source | OR | room-provider visibility and occluder class |
| `occluded_by_wall` | source | OR | room-provider visibility and occluder class |
| `never_occluded` | source | AND | complete room-provider visibility evidence |
| `leaves_camera_fov` | source | OR | calibrated camera and frame FOV membership |
| `stays_in_camera_fov` | source | AND | calibrated camera and frame FOV membership |
| `crosses_azimuth_zero` | source | OR | observer yaw and source trajectory |
| `passes_close_to_mic` | source | OR | observer position and source trajectory |
| `far_from_mic_whole_clip` | source | AND | observer position and source trajectory |
| `stationary` | source | OR | source trajectory and frame rate |
| `steady_walk` | source | AND | source trajectory and frame rate |
| `stop_and_go` | source | OR | source trajectory and frame rate |
| `sources_pass_each_other` | pair | OR | synchronized paired source trajectories |

### Frozen thresholds

| Threshold | Value | Comparison semantics |
|---|---:|---|
| `passes_close_to_mic_m` | 1.0 m | minimum 3D distance is strictly less than threshold |
| `far_from_mic_whole_clip_m` | 4.0 m | minimum 3D distance is strictly greater than threshold |
| `stationary_mean_speed_mps` | 0.1 m/s | mean adjacent-frame speed is strictly less |
| `steady_walk_min_mean_speed_mps` | 0.15 m/s | mean speed is greater than or equal |
| `steady_walk_max_speed_cv` | 0.4 | speed coefficient of variation is strictly less |
| `stop_and_go_stop_speed_mps` | 0.05 m/s | stopped is strictly less; moving is greater than or equal |
| `stop_and_go_min_stopped_frames` | 3 | at least three stopped speed samples |
| `stop_and_go_min_moving_frames` | 3 | at least three moving speed samples |
| `sources_pass_each_other_m` | 0.5 m | minimum synchronized horizontal XZ separation is strictly less |

`steady_walk` additionally requires at least three adjacent-frame speed samples. `crosses_azimuth_zero` requires camera-local lateral coordinates to span both a strictly negative and a strictly positive value.

## Tri-state assessments

Each assessment preserves:

```text
present       -> true
absent        -> false
not_evaluated -> null
```

`not_evaluated` requires `missing_dependency` or `manual_review` evidence. An evaluated state cannot rely only on missing/manual evidence. Missing raycast, camera calibration, semantic occluder class or provider output is never interpreted as `absent`.

Mutual exclusions retained by M5.1 include:

- `never_occluded` versus either occlusion-present flag;
- `leaves_camera_fov` versus `stays_in_camera_fov` when both are known;
- `stationary` versus `steady_walk` or `stop_and_go`;
- `steady_walk` versus `stop_and_go`;
- `passes_close_to_mic` versus `far_from_mic_whole_clip`.

## Clip aggregation

OR flags use:

```text
any present      -> present
all absent       -> absent
otherwise        -> not_evaluated
```

AND flags use:

```text
any absent       -> absent
all present      -> present
otherwise        -> not_evaluated
```

The stable M6 public function is `aggregate_legacy_status()`. It duplicates no new semantics: focused tests compare it and the full adapter against the checked-in M5.1 source manifest.

## M6 provider adapter

`evaluate_legacy_flags()` accepts only the compact facts required by the existing evaluator:

- fixed observer position and yaw;
- frame rate;
- synchronized source positions in Habitat Y-up world coordinates;
- optional frame-aligned FOV and occlusion facts from a room provider.

It returns a source/pair/clip report with the existing flag IDs, scopes, tri-state values, reasons and hash-bound evidence. It computes velocity internally from adjacent positions only where an existing flag needs it. It does not require or define `world_velocity`, `listener_radial_velocity`, `angular_velocity`, path summaries, LOS transition arrays or a final episode schema.

Optional visibility facts use:

```text
in_fov_by_frame: true | false | null
occlusion_by_frame: clear | furniture | wall | other | null
```

An omitted provider or a `null` fact propagates to `not_evaluated` unless another known fact is already sufficient for the relevant Boolean definition.

## Why MP3D cannot copy Legacy Apartment spatial flags

`src/avengine/m5_1/mp3d_delivery.py` deliberately reuses only source taxonomy, event timing/audio programs and dry-audio provenance. It excludes the Legacy Apartment observer, trajectories, source/clip/pair spatial flags and visual/migration provenance.

MP3D has a different scene, route, navmesh, camera visibility and actual articulated anchor evidence. Its spatial flags must be recomputed from MP3D provider facts. Copying a legacy `present` or `absent` assessment would create a false geometry claim. If MP3D lacks a required raycast or semantic fact, the correct result is `not_evaluated`.

## Delivery-only data

The following are derived review aids rather than authorities:

- overlay `true_flags` strings;
- top-down arrows, labels and event captions;
- review-video muxes;
- MP3D review labels such as `center_navmesh_pass` and `visible_all_frames`;
- `source_program_reuse_record.json` as a record of permitted reuse.

They may link to authoritative evidence but cannot independently promote a room, source or flag assessment.

## Compatibility tests

The M6 tests enforce:

- exact flag ID membership and order;
- exact v1 thresholds and scopes;
- exact OR/AND tri-state truth tables;
- missing visibility provider facts remain `not_evaluated`;
- the M6 trajectory/pair/clip report matches every status and value in the checked-in M5.1 Legacy Apartment manifest;
- AudioProgram uses the same half-open frame/sample boundary arithmetic;
- new entity/endpoint registries resolve through explicit adapters without changing the M5.1 manifest.

Any future flag definition, threshold or scope change requires a new schema/definition revision and an explicit migration. It must not mutate `m5_1_v1` in place.
