# Source Asset Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a registry-backed source asset layer so dataset specs can select approved reusable assets by `asset_id` while downstream SPEAR code still receives compatible `tag` and `audio_lookup` fields.

**Architecture:** Keep registry manifests under `external/SPEAR/data/source_assets_v1`. Add a small loader in `tools/spike_rlr/source_asset_registry.py` that validates approval and resolves `asset_id` entries into legacy-compatible source-pool entries. Update `scene_generator.py` and the M1 dataset spec to use this loader without changing render/audio consumers.

**Tech Stack:** Python standard library, pytest, JSON manifests, existing SPEAR `tools/spike_rlr` import style.

## Global Constraints

- Register only reusable assets with approved review status.
- Dataset specs should reference `asset_id`; per-clip sources/events should keep `asset_id`.
- Keep legacy `tag` and `audio_lookup` output for existing render/audio code.
- Preserve backward compatibility for old `source_pool` entries that already contain `tag`.
- Store intended generation descriptions separately from measured texture colors.
- Do not move large binary assets in this change.

---

### Task 1: Registry Fixtures And Loader Tests

**Files:**
- Create: `external/SPEAR/tests/tools/spike_rlr/test_source_asset_registry.py`
- Create later: `external/SPEAR/tools/spike_rlr/source_asset_registry.py`

**Interfaces:**
- Produces test expectations for:
  - `load_registry(registry_root: Path | None = None) -> dict`
  - `load_asset(asset_id: str, registry_root: Path | None = None) -> dict`
  - `approved_assets(registry_root: Path | None = None, asset_class: str | None = None, category: str | None = None) -> list[dict]`
  - `resolve_source_pool_entry(entry: dict, registry_root: Path | None = None, require_approved: bool = True) -> dict`
  - `resolve_source_pool(pool: list[dict], registry_root: Path | None = None, require_approved: bool = True) -> list[dict]`

- [ ] Write tests that load the production registry, resolve an approved
      `asset_id`, preserve legacy entries, allow audio override, and reject an
      unapproved fixture asset.
- [ ] Run `pytest tests/tools/spike_rlr/test_source_asset_registry.py -q` and
      verify the new module import fails.

### Task 2: Registry Loader

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/source_asset_registry.py`

**Interfaces:**
- `source_asset_registry.py` must be importable from tests that insert
  `tools/spike_rlr` directly on `sys.path`.
- `resolve_source_pool_entry()` must return a copy of the input entry with
  `asset_id`, `tag`, `audio_lookup`, `asset_class`, `category`, and `family`
  populated for registry entries.

- [ ] Implement JSON loading and path resolution.
- [ ] Validate registry schema values `source_assets_v1` and
      `source_asset_v1`.
- [ ] Reject missing assets, duplicate `asset_id` values, missing `legacy_tag`,
      missing `audio.default_lookup`, and unapproved assets when
      `require_approved=True`.
- [ ] Run `pytest tests/tools/spike_rlr/test_source_asset_registry.py -q` and
      verify it passes.

### Task 3: Production Registry Manifests

**Files:**
- Create: `external/SPEAR/data/source_assets_v1/registry.json`
- Create: `external/SPEAR/data/source_assets_v1/dog/golden_retriever/dog_golden_0001/asset.json`
- Create: `external/SPEAR/data/source_assets_v1/dog/beagle/dog_beagle_0002/asset.json`
- Create: `external/SPEAR/data/source_assets_v1/cat/british_shorthair/cat_british_shorthair_0002/asset.json`

**Interfaces:**
- Manifests use `legacy_tag` values consumed by existing render/audio passes.
- Manifests include generation text, intended color label, measured dominant
  colors, rig info, audio defaults, visual asset paths, and review status.

- [ ] Add manifests for the three currently approved M1 source assets.
- [ ] Use repo-relative SPEAR paths for `visual_assets`.
- [ ] Confirm referenced files exist under
      `tmp/hy3d_batch/approved/{legacy_tag}`.
- [ ] Run the registry tests again.

### Task 4: Scene Generator Integration

**Files:**
- Modify: `external/SPEAR/tools/spike_rlr/scene_generator.py`
- Modify: `external/SPEAR/tests/tools/spike_rlr/test_scene_generator.py`

**Interfaces:**
- `sample_scene()` consumes either legacy source-pool entries or registry
  entries with `asset_id`.
- Output `source_specs` include `asset_id` when the selected pool entry came
  from the registry.

- [ ] Add a failing test for `sample_scene()` with `source_pool` entries that
      only contain `asset_id`.
- [ ] Import and call `resolve_source_pool()` before source count validation.
- [ ] Preserve old tests for explicit `tag`/`audio_lookup` source pools.
- [ ] Run `pytest tests/tools/spike_rlr/test_scene_generator.py -q`.

### Task 5: M1 Dataset Spec Migration

**Files:**
- Modify: `external/SPEAR/data/apartment_v2_m1_dataset_spec.json`
- Modify: `external/SPEAR/tests/tools/spike_rlr/test_scene_generator.py`

**Interfaces:**
- M1 `source_pool` uses `asset_id`.
- The test resolves that pool through the registry and asserts the legacy tags
  and audio mappings are still correct.

- [ ] Replace M1 `source_pool` entries with `asset_id` entries.
- [ ] Update the M1 source-pool test to resolve the registry-backed pool.
- [ ] Run `pytest tests/tools/spike_rlr/test_scene_generator.py -q`.

### Task 6: Documentation And Verification

**Files:**
- Modify: `AGENTS.md`

**Interfaces:**
- Future agents know where the registry lives and when assets may be
  registered.

- [ ] Add the source asset registry path and lifecycle trap to `AGENTS.md`.
- [ ] Run focused tests:
      `pytest tests/tools/spike_rlr/test_source_asset_registry.py tests/tools/spike_rlr/test_scene_generator.py -q`.
- [ ] Check `git status --short` in AVEngine and SPEAR and report exactly
      which files were changed.
