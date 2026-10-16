# Event Scenario Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic event-based apartment demo where one visible dog idles in front of the camera while another real dog source moves behind the camera from listener-left to listener-right without entering the camera view.

**Architecture:** Keep event construction separate from source identity, audio selection, trajectory generation, and constraint verification. The demo emits a normal `apartment_v1` spec so existing UE/RLR/topdown/review tools remain the renderer of record.

**Tech Stack:** Python, NumPy, existing SPEAR `tools/spike_rlr` pipeline, pytest in `spear-env`, RLR rendering in `ss2`.

## Global Constraints

- Work in `/data/jzy/code/AVEngine/external/SPEAR` and do not reset existing dirty files.
- Do not use seed search for the requested demo; construct the event directly in listener-local coordinates.
- Use real dog audio for dog tags and real cat audio for cat tags by default.
- A scenario must verify hard constraints before rendering: behind-camera, not-visible, left-to-right, stationary, no wall collision, and minimum actor distance.
- Generated MP4/WAV/tmp output remains under `external/SPEAR/tmp` and must not be committed.

---

### Task 1: Animal Audio Resolver

**Files:**
- Create: `tools/spike_rlr/animal_audio.py`
- Modify: `tools/spike_rlr/run_audio_pass_rlr.py`
- Modify: `data/audio_library_v1.json`
- Test: `tests/tools/spike_rlr/test_animal_audio.py`

**Interfaces:**
- Produces: `resolve_animal_audio_path(tag: str, audio_lookup: str | None = None, explicit_path: str | None = None) -> str`
- Produces: `is_synthetic_audio_path(path: str | None) -> bool`
- Consumes: RLR `_load_dry_source(..., source_spec=...)`

- [ ] Write tests that dog tags resolve to real dog files, cat tags resolve to real cat files, explicit paths win, and `dog_husky` does not default to synthetic piano.
- [ ] Run the new test and confirm it fails before implementation.
- [ ] Implement `animal_audio.py`.
- [ ] Modify RLR dry-source loading to prefer each spec source's `audio_path`, then resolver fallback, then existing synthetic fallback only when explicitly requested.
- [ ] Run the animal-audio tests and RLR import tests.

### Task 2: Event Constraint Primitives

**Files:**
- Create: `tools/spike_rlr/event_constraints.py`
- Test: `tests/tools/spike_rlr/test_event_constraints.py`

**Interfaces:**
- Produces: `listener_local_xy(points_xyz, mic_pos_m, mic_yaw_deg) -> np.ndarray`
- Produces: `ConstraintResult(name: str, passed: bool, details: dict)`
- Produces: `verify_constraints(...) -> list[ConstraintResult]`

- [ ] Write tests for behind-camera, visible/not-visible contradiction, left-to-right listener-local motion, stationary motion, actor distance, and unsatisfied reports.
- [ ] Run the new test and confirm it fails before implementation.
- [ ] Implement the smallest constraint functions needed by the demo.
- [ ] Run the constraint tests.

### Task 3: Deterministic Demo Scenario

**Files:**
- Create: `tools/spike_rlr/demo_scenarios.py`
- Test: `tests/tools/spike_rlr/test_demo_scenarios.py`

**Interfaces:**
- Produces: `compose_front_idle_rear_left_to_right_demo(base_spec_path: Path, out_spec_path: Path | None = None) -> dict`

- [ ] Write tests that the generated spec has `dog_golden` idle in front, `dog_beagle_v2` walking behind left-to-right, both with real dog audio, and all hard constraints passing.
- [ ] Run the new test and confirm it fails before implementation.
- [ ] Implement the scenario builder using listener-local placement and trajectory helpers.
- [ ] Run the scenario tests.

### Task 4: Smoke Generation

**Files:**
- Generated only under `tmp/spike_output_apartment_v2_rear_pass_demo`

**Interfaces:**
- Consumes: normal `dataset_runner` per-clip render path through generated `spec.json`.

- [ ] Generate one spec-only demo clip and inspect `spec.json` plus constraint report.
- [ ] If spec-only passes, run UE/RLR/topdown/review for the one demo clip.
- [ ] Report final absolute paths for review videos and solo per-source audio.
