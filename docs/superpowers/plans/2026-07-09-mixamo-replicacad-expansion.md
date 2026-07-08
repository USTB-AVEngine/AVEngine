# Mixamo + ReplicaCAD Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the current animal audio-visual pipeline with a minimal Mixamo animation probe and a minimal ReplicaCAD room probe while preserving review-video and audio/visual consistency gates.

**Architecture:** Keep the existing apartment pipeline as the reference path. Add small resolver/probe modules first, then implement one vertical slice for external animation and room assets. Large datasets live in `/data/datasets`; the repo stores code, docs, manifests, and tiny reports only.

**Tech Stack:** Python 3.11, SPEAR RPC/UE 5.5, Habitat/RLR via `ss2`, `trimesh`, pytest, ffmpeg, Git/GitHub remotes `origin` and `eastforward`.

## Global Constraints

- Read `/data/jzy/code/AVEngine/AGENTS.md` before each work session and update it when a new path convention or trap is discovered.
- Do not commit `/data/datasets`, `external/SPEAR/tmp`, generated MP4/WAV/PNG outputs, Mixamo FBX files, ReplicaCAD scene payloads, UE `Saved`, `Intermediate`, or cooked output.
- Preserve current review artifacts: UE video, topdown, `actor_visual_metadata.json`, RLR metadata, and `side_by_side_review_annotated.mp4`.
- Use `spear-env` for SPEAR/UE/render/review tests and `ss2` for `trimesh`/RLR mesh tests.
- Low-confidence direction checks block automatic batch usage and write a report; they do not silently flip assets.

---

## File Structure

- Create: `docs/superpowers/status/2026-07-09-mixamo-replicacad-expansion-status.md`
  - Final overnight report for the user.
- Modify: `AGENTS.md`
  - External data paths and discovered dataset/download traps.
- Create: `external/SPEAR/tools/spike_rlr/external_data_paths.py`
  - Single source for ReplicaCAD and Mixamo dataset roots.
- Create: `external/SPEAR/tests/tools/spike_rlr/test_external_data_paths.py`
  - Missing-data and override tests.
- Create: `external/SPEAR/tools/spike_rlr/direction_gate.py`
  - Automatic orientation confidence report.
- Create: `external/SPEAR/tests/tools/spike_rlr/test_direction_gate.py`
  - Report schema and pass/block tests.
- Create: `external/SPEAR/tools/spike_rlr/mixamo_probe.py`
  - FBX discovery and import status writer.
- Create: `external/SPEAR/tests/tools/spike_rlr/test_mixamo_probe.py`
  - Missing-root and discovery tests.
- Create: `external/SPEAR/tools/spike_rlr/replicacad_probe.py`
  - ReplicaCAD path discovery and scene inventory writer.
- Create: `external/SPEAR/tests/tools/spike_rlr/test_replicacad_probe.py`
  - Missing-root and inventory tests using a tiny synthetic fixture.

## Task 1: Freeze And Push Current Work

**Files:**
- Modify: Git index in `/data/jzy/code/AVEngine`
- Modify: Git index in `/data/jzy/code/AVEngine/external/SPEAR`

**Interfaces:**
- Consumes: current dirty worktree.
- Produces: pushed commits or a status note explaining why push was blocked.

- [ ] **Step 1: Inspect root repository status**

Run:

```bash
cd /data/jzy/code/AVEngine
git status --short
git log --oneline --decorate -5
git remote -v
```

Expected: root repo shows docs/AGENTS files only or a short list that can be
separated from SPEAR code changes.

- [ ] **Step 2: Inspect SPEAR repository status**

Run:

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
git status --short
git log --oneline --decorate -5
git remote -v
```

Expected: dirty SPEAR tree is large. Stage only source/test/docs files relevant
to current work; do not stage `tmp` or generated media.

- [ ] **Step 3: Run focused verification before commit**

Run:

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_review_videos.py \
  tests/tools/spike_rlr/test_room_conventions.py \
  tests/tools/spike_rlr/test_hy3d_generate_and_audit.py \
  tests/tools/test_species_rig_map_approved_assets.py \
  tests/tools/spike_rlr/test_run_audio_pass_cli.py \
  tests/tools/test_hy3d_bake_diffuse_paths.py
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_auto_orient_ingest.py
```

Expected: both commands exit 0.

- [ ] **Step 4: Commit root docs**

Run:

```bash
cd /data/jzy/code/AVEngine
git add AGENTS.md \
  docs/superpowers/specs/2026-07-07-rlr-vs-gpurir-spike-design.md \
  docs/superpowers/plans/2026-07-07-rlr-vs-gpurir-spike.md \
  docs/superpowers/specs/2026-07-09-mixamo-replicacad-expansion-design.md \
  docs/superpowers/plans/2026-07-09-mixamo-replicacad-expansion.md
git diff --cached --stat
git commit -m "docs: plan Mixamo and ReplicaCAD expansion"
```

Expected: commit succeeds or reports nothing to commit. If nothing commits,
write the reason in the status file.

- [ ] **Step 5: Push root docs**

Run:

```bash
cd /data/jzy/code/AVEngine
git push origin main
```

Expected: push succeeds. If rejected, run `git status --short` and write the
reason in the status file.

- [ ] **Step 6: Commit and push focused SPEAR changes**

Run:

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
git add \
  tools/spike_rlr/build_review_videos.py \
  tools/spike_rlr/run_audio_pass_rlr.py \
  tools/spike_rlr/run_render_pass_apartment.py \
  tools/spike_rlr/hy3d_generate_and_audit.py \
  tools/hy3d_bake_diffuse.py \
  tools/species_rig_map.py \
  tools/spike_rlr/auto_orient_ingest.py \
  tools/spike_rlr/review_ui_server.py \
  tests/tools/spike_rlr/test_review_videos.py \
  tests/tools/spike_rlr/test_room_conventions.py \
  tests/tools/spike_rlr/test_run_audio_pass_cli.py \
  tests/tools/spike_rlr/test_hy3d_generate_and_audit.py \
  tests/tools/spike_rlr/test_auto_orient_ingest.py \
  tests/tools/test_species_rig_map_approved_assets.py \
  tests/tools/test_hy3d_bake_diffuse_paths.py
git diff --cached --stat
git commit -m "fix(review): align actor markers, audio yaw, and painted assets"
git push eastforward HEAD
```

Expected: push succeeds. If one of the listed paths is untracked/missing, adjust
the `git add` list to existing focused files only and record the exact list in
the status file.

## Task 2: External Data Path Resolvers

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/external_data_paths.py`
- Create: `external/SPEAR/tests/tools/spike_rlr/test_external_data_paths.py`
- Modify: `AGENTS.md`

**Interfaces:**
- Produces: `dataset_root(name: str) -> pathlib.Path`
- Produces: `require_dataset_root(name: str) -> pathlib.Path`
- Produces: `DatasetMissingError`

- [ ] **Step 1: Write failing tests**

Create `external/SPEAR/tests/tools/spike_rlr/test_external_data_paths.py` with:

```python
from pathlib import Path

import pytest

from tools.spike_rlr.external_data_paths import (
    DatasetMissingError,
    dataset_root,
    require_dataset_root,
)


def test_dataset_root_defaults():
    assert dataset_root("replicacad") == Path("/data/datasets/replica_cad")
    assert dataset_root("mixamo") == Path("/data/datasets/mixamo")


def test_dataset_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("AVENGINE_REPLICACAD_ROOT", str(tmp_path / "rc"))
    monkeypatch.setenv("AVENGINE_MIXAMO_ROOT", str(tmp_path / "mx"))
    assert dataset_root("replicacad") == tmp_path / "rc"
    assert dataset_root("mixamo") == tmp_path / "mx"


def test_require_dataset_root_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("AVENGINE_REPLICACAD_ROOT", str(tmp_path / "missing"))
    with pytest.raises(DatasetMissingError) as exc:
        require_dataset_root("replicacad")
    msg = str(exc.value)
    assert "AVENGINE_REPLICACAD_ROOT" in msg
    assert "ReplicaCAD" in msg


def test_unknown_dataset_name():
    with pytest.raises(KeyError):
        dataset_root("unknown")
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_external_data_paths.py
```

Expected: import failure because `external_data_paths.py` does not exist.

- [ ] **Step 3: Implement resolver**

Create `external/SPEAR/tools/spike_rlr/external_data_paths.py` with:

```python
"""External dataset path helpers for AVEngine/SPEAR experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    label: str
    default_path: Path
    env_var: str
    acquisition_hint: str


DATASETS: dict[str, DatasetSpec] = {
    "replicacad": DatasetSpec(
        name="replicacad",
        label="ReplicaCAD",
        default_path=Path("/data/datasets/replica_cad"),
        env_var="AVENGINE_REPLICACAD_ROOT",
        acquisition_hint=(
            "ReplicaCAD: run python -m habitat_sim.utils.datasets_download --uids "
            "replica_cad_dataset --data-path /data/datasets --no-replace. "
            "Set AVENGINE_REPLICACAD_ROOT when using a non-default location."
        ),
    ),
    "mixamo": DatasetSpec(
        name="mixamo",
        label="Mixamo",
        default_path=Path("/data/datasets/mixamo"),
        env_var="AVENGINE_MIXAMO_ROOT",
        acquisition_hint=(
            "Place user-downloaded Mixamo FBX files under this directory or set "
            "AVENGINE_MIXAMO_ROOT to the FBX dataset directory."
        ),
    ),
}


class DatasetMissingError(FileNotFoundError):
    """Raised when an expected external dataset root is missing."""


def dataset_spec(name: str) -> DatasetSpec:
    return DATASETS[name]


def dataset_root(name: str) -> Path:
    spec = dataset_spec(name)
    return Path(os.environ.get(spec.env_var, spec.default_path)).expanduser()


def require_dataset_root(name: str) -> Path:
    spec = dataset_spec(name)
    root = dataset_root(name)
    if root.exists():
        return root
    raise DatasetMissingError(
        f"{spec.label} dataset root does not exist: {root}. "
        f"Set {spec.env_var} or create the default path. {spec.acquisition_hint}"
    )
```

- [ ] **Step 4: Verify tests pass**

Run:

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_external_data_paths.py
```

Expected: all tests pass.

- [ ] **Step 5: Update `AGENTS.md`**

Add the external dataset paths and environment override names under Workspace
Layout:

```markdown
- External ReplicaCAD dataset root: `/data/datasets/replica_cad`
  (`AVENGINE_REPLICACAD_ROOT` overrides it).
- External Mixamo dataset root: `/data/datasets/mixamo`
  (`AVENGINE_MIXAMO_ROOT` overrides it).
```

## Task 3: Automatic Direction Gate Report

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/direction_gate.py`
- Create: `external/SPEAR/tests/tools/spike_rlr/test_direction_gate.py`

**Interfaces:**
- Produces: `direction_gate_report(mesh_path: Path, direction_json: Path | None = None) -> dict`
- Produces: `write_direction_gate_report(mesh_path: Path, out_path: Path, direction_json: Path | None = None) -> dict`

- [ ] **Step 1: Write failing tests**

Create `external/SPEAR/tests/tools/spike_rlr/test_direction_gate.py` with tests
that build small box meshes via `trimesh` and assert:

- Long horizontal mesh with valid direction metadata returns `decision == "pass"`.
- Nearly symmetric cube returns `decision == "block"`.
- Missing mesh raises `FileNotFoundError`.
- JSON writer creates a report with `mesh_path`, `checks`, `confidence`, and
  `decision`.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_direction_gate.py
```

Expected: import failure because `direction_gate.py` does not exist.

- [ ] **Step 3: Implement minimal geometry gate**

Implement:

- Load mesh with `trimesh.load(..., force="scene")`.
- Compute combined bounds.
- Compute extents `(x, y, z)`.
- Pass when horizontal major/minor ratio is at least `1.2`, up extent is
  finite, and direction metadata has either `human_approved: true` or a
  `human_applied_rotation_matrix`.
- Block symmetric or invalid assets.
- Write report as JSON with stable float rounding.

- [ ] **Step 4: Verify tests pass**

Run:

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_direction_gate.py
```

Expected: all tests pass.

## Task 4: Mixamo Probe

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/mixamo_probe.py`
- Create: `external/SPEAR/tests/tools/spike_rlr/test_mixamo_probe.py`

**Interfaces:**
- Produces: `discover_mixamo_fbx(root: Path) -> list[Path]`
- Produces: CLI `python tools/spike_rlr/mixamo_probe.py --out PATH`

- [ ] **Step 1: Write failing discovery tests**

Tests create a temporary Mixamo-like tree with two `.fbx` files and one
non-FBX file. Assert discovery returns sorted relative FBX paths and missing
root writes a status with `state == "missing_data"`.

- [ ] **Step 2: Implement discovery/status writer**

The CLI must:

- Resolve root via `external_data_paths.dataset_root("mixamo")`.
- If missing, write JSON:
  `{"state": "missing_data", "dataset": "mixamo", "root": "...", "manual_action": "..."}`
- If present, write JSON with `state == "ready"`, `fbx_count`, and `fbx_files`.

- [ ] **Step 3: Verify tests pass**

Run:

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_mixamo_probe.py
```

Expected: all tests pass.

- [ ] **Step 4: Run real probe**

Run:

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python tools/spike_rlr/mixamo_probe.py \
  --out tmp/mixamo_probe/status.json
cat tmp/mixamo_probe/status.json
```

Expected: either `ready` with FBX inventory or `missing_data` with manual action.

## Task 5: ReplicaCAD Probe

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/replicacad_probe.py`
- Create: `external/SPEAR/tests/tools/spike_rlr/test_replicacad_probe.py`

**Interfaces:**
- Produces: `discover_replicacad_scenes(root: Path) -> list[Path]`
- Produces: CLI `python tools/spike_rlr/replicacad_probe.py --out PATH`

- [ ] **Step 1: Write failing inventory tests**

Tests create a temporary ReplicaCAD-like tree containing `.scene_instance.json`,
`.glb`, and `.ply` files. Assert the probe returns sorted scene candidates and
reports missing roots as `state == "missing_data"`.

- [ ] **Step 2: Implement inventory/status writer**

The CLI must:

- Resolve root via `external_data_paths.dataset_root("replicacad")`.
- Search for `*.scene_instance.json`, `*.glb`, and `*.ply`.
- Write JSON with `state`, `root`, `scene_instance_count`, `mesh_count`, and up
  to 20 sample paths.

- [ ] **Step 3: Verify tests pass**

Run:

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_replicacad_probe.py
```

Expected: all tests pass.

- [ ] **Step 4: Run real probe**

Run:

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python tools/spike_rlr/replicacad_probe.py \
  --out tmp/replicacad_probe/status.json
cat tmp/replicacad_probe/status.json
```

Expected: either `ready` with scene inventory or `missing_data` with acquisition
instructions.

## Task 6: Try Data Acquisition Without Blocking Blindly

**Files:**
- Modify: `docs/superpowers/status/2026-07-09-mixamo-replicacad-expansion-status.md`
- Modify: `AGENTS.md` if a real download command/path trap is discovered.

**Interfaces:**
- Consumes: real network availability and dataset licensing.
- Produces: completed dataset root or a status file explaining manual action.

- [ ] **Step 1: Check local dataset roots**

Run:

```bash
ls -la /data/datasets/replica_cad /data/datasets/mixamo 2>&1 || true
```

Expected: either directories exist or the error is recorded.

- [ ] **Step 2: Look for official/local ReplicaCAD download tooling**

Run local searches first:

```bash
cd /data/jzy/code/AVEngine
rg -n -i "ReplicaCAD|replica cad|scene_dataset|habitat_download|hm3d|hssd" .
```

Expected: locate an existing script or confirm the repo has no downloader.

- [ ] **Step 3: If a non-interactive official download command exists, run it**

Run the command into `/data/datasets` and keep the terminal session
open until it finishes. If it is gated or requires login/license acceptance,
write that exact reason in the status file and continue with code/tests.

- [ ] **Step 4: Mixamo acquisition**

Mixamo normally requires browser/user account download. Do not scrape it. If
`/data/datasets/mixamo` is missing, write `missing_data` status and continue
with the probe/test implementation.

## Task 7: Minimal Smoke Or Graceful Stop

**Files:**
- Modify: `docs/superpowers/status/2026-07-09-mixamo-replicacad-expansion-status.md`

**Interfaces:**
- Consumes: tasks 2-6.
- Produces: review artifact paths or explicit missing-data blockers.

- [ ] **Step 1: Run all new lightweight tests**

Run:

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_external_data_paths.py \
  tests/tools/spike_rlr/test_mixamo_probe.py \
  tests/tools/spike_rlr/test_replicacad_probe.py
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_direction_gate.py
```

Expected: all new tests pass.

- [ ] **Step 2: Run existing focused regressions**

Run the Task 1 verification commands again.

Expected: all previously passing tests remain passing.

- [ ] **Step 3: If ReplicaCAD is ready, choose one scene for manual smoke**

Run `replicacad_probe.py`, pick the first scene candidate, and write its path in
the status file. Do not start a full render until the adapter can prove a shared
coordinate frame and topdown extent.

- [ ] **Step 4: Write final status**

Write `docs/superpowers/status/2026-07-09-mixamo-replicacad-expansion-status.md`
with:

- Completed work
- Tests run and results
- Push results
- Dataset readiness
- Any generated probe/status JSON paths
- Unfinished items and exact reasons
