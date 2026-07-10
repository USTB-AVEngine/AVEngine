# Rocketbox Source Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the two sampled Rocketbox avatars with verified official textures and produce unanimated multi-view source-review artifacts that the user can approve before any retargeting work begins.

**Architecture:** A pure Python helper owns Rocketbox source discovery, Git blob verification, atomic texture download, inspection metadata, and review-gate semantics. A separate Blender script imports the untouched rigged FBX, reconnects the verified textures without replacing material slots, renders fixed review views and a turntable, and emits a machine-readable render manifest. Generated binaries remain under `external/SPEAR/tmp/rocketbox_human_review`; reusable code and tests live under `external/SPEAR/tools` and `external/SPEAR/tests`.

**Tech Stack:** Python 3.10+, standard library `urllib`, `hashlib`, `json`, pytest, Blender 4.2.1 Python API, Eevee Next, FFmpeg.

## Global Constraints

- Read `/data/jzy/code/AVEngine/AGENTS.md` before every implementation batch.
- Do not use the existing Hunyuan `trimesh` review baker on rigged Rocketbox FBXs; it strips armature data.
- Do not assign locomotion actions during this phase. The armature must render in `REST` pose.
- Do not clear the original `body`, `head`, or `opacity` material slots.
- Official files must match both Git tree size and Git blob SHA-1 before use.
- Missing body color, head color, or opacity color blocks textured source review.
- Generated PNG, MP4, JSON, TGA, FBX, and GLB artifacts stay out of git.
- Do not stage or overwrite the existing unrelated `AGENTS.md`, `.superpowers/`, or `docs/superpowers/plans/2026-07-09-event-scenario-demo.md` changes.
- This plan stops after source-review artifacts are delivered to the user. Source-motion retargeting starts only after explicit approval.

---

## File Structure

- Create `external/SPEAR/tools/spike_rlr/rocketbox_human_review.py`
  - Source catalog, Git tree parsing, Git blob hashing, atomic official-file download, inspection payload, pending review payload, and approval gate.
- Create `external/SPEAR/tools/blender_render_rocketbox_source_review.py`
  - Blender-only material reconstruction, rest-pose multi-view rendering, turntable rendering, and render manifest.
- Create `external/SPEAR/tests/tools/spike_rlr/test_rocketbox_human_review.py`
  - Pure Python unit tests for catalog, hashing, download verification, payload schema, and gate behavior.
- Create `external/SPEAR/tests/tools/test_blender_rocketbox_source_review_static.py`
  - Static contract tests preventing action assignment/material clearing and checking required CLI/output contracts.
- Modify `AGENTS.md`
  - Record the new review paths, official-texture requirement, and the rule that old locomotion pages are not accepted retarget evidence. Do not stage this pre-existing dirty file with code commits.
- Generate `external/SPEAR/tmp/rocketbox_human_review/<asset_id>/...`
  - Download/inspection metadata, five still views, close views, contact sheets, turntable, and pending `source_review.json`.

---

### Task 1: Verified Rocketbox Source Catalog And Downloader

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/rocketbox_human_review.py`
- Test: `external/SPEAR/tests/tools/spike_rlr/test_rocketbox_human_review.py`

**Interfaces:**
- Consumes: official Git tree JSON at `external/SPEAR/tmp/human_motion_source_probe/rocketbox_tree.json`, sample root `/data/datasets/rocketbox/sample`, and raw GitHub base `https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/`.
- Produces: `RocketboxReviewAsset`, `load_review_assets(tree_json: Path, sample_root: Path) -> dict[str, RocketboxReviewAsset]`, `git_blob_sha1(path: Path) -> str`, `verify_official_file(path: Path, expected_size: int, expected_git_sha: str) -> None`, and `ensure_official_files(asset: RocketboxReviewAsset, opener: Callable | None = None) -> list[Path]`.

- [ ] **Step 1: Write failing source-catalog and Git-blob tests**

Add tests that create a tiny Git tree fixture and assert exact avatar fields:

```python
def test_load_review_assets_selects_two_sampled_adults(tmp_path):
    tree = {
        "tree": [
            {"path": "Assets/Avatars/Adults/Male_Adult_01/Export/Male_Adult_01.fbx", "size": 3, "sha": "fbx-m"},
            {"path": "Assets/Avatars/Adults/Male_Adult_01/Textures/m002_body_color.tga", "size": 4, "sha": "tex-m"},
            {"path": "Assets/Avatars/Adults/Female_Adult_01/Export/Female_Adult_01.fbx", "size": 3, "sha": "fbx-f"},
            {"path": "Assets/Avatars/Adults/Female_Adult_01/Textures/f001_body_color.tga", "size": 4, "sha": "tex-f"},
        ]
    }
    tree_path = tmp_path / "tree.json"
    tree_path.write_text(json.dumps(tree), encoding="utf-8")

    assets = load_review_assets(tree_path, tmp_path / "sample")

    assert sorted(assets) == ["rocketbox_female_adult_01", "rocketbox_male_adult_01"]
    assert assets["rocketbox_male_adult_01"].forward_axis == "-Y"
    assert assets["rocketbox_female_adult_01"].up_axis == "+Z"


def test_git_blob_sha1_matches_git_object_format(tmp_path):
    path = tmp_path / "hello.bin"
    path.write_bytes(b"hello\n")

    assert git_blob_sha1(path) == "ce013625030ba8dba906f756967f9e9ca394464a"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_rocketbox_human_review.py
```

Expected: collection or import failure because `rocketbox_human_review.py` does not exist.

- [ ] **Step 3: Implement the catalog and blob verifier**

Implement immutable records and select required files by exact asset prefix:

```python
@dataclass(frozen=True)
class OfficialFile:
    rel_path: str
    size: int
    git_sha: str
    local_path: Path


@dataclass(frozen=True)
class RocketboxReviewAsset:
    asset_id: str
    gender: str
    avatar_dir: str
    texture_prefix: str
    fbx: OfficialFile
    textures: tuple[OfficialFile, ...]
    up_axis: str = "+Z"
    forward_axis: str = "-Y"
```

The required texture suffixes are `body_color`, `body_normal`, `body_specular`, `head_color`, `head_normal`, `head_specular`, and `opacity_color`. Female `head_normal_wrinkle` is optional and included when present.

`verify_official_file` must raise `OfficialFileError` with the path, actual/expected size, and actual/expected Git SHA. The Git blob digest is:

```python
header = f"blob {path.stat().st_size}\0".encode("ascii")
digest = hashlib.sha1()
digest.update(header)
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
return digest.hexdigest()
```

- [ ] **Step 4: Add failing atomic-download tests**

Use a fake opener returning `io.BytesIO` and assert that:

```python
def test_ensure_official_files_downloads_atomically_and_verifies(tmp_path):
    payload = b"texture"
    expected = OfficialFile(
        rel_path="Assets/Test/texture.tga",
        size=len(payload),
        git_sha=git_blob_sha1_bytes(payload),
        local_path=tmp_path / "texture.tga",
    )
    asset = _asset_with_files(tmp_path, textures=(expected,))

    paths = ensure_official_files(asset, opener=lambda request, timeout: io.BytesIO(payload))

    assert paths == [expected.local_path]
    assert expected.local_path.read_bytes() == payload
    assert not expected.local_path.with_suffix(".tga.part").exists()


def test_ensure_official_files_rejects_corrupt_download(tmp_path):
    expected = _official_file(tmp_path, payload=b"correct")
    asset = _asset_with_files(tmp_path, textures=(expected,))

    with pytest.raises(OfficialFileError, match="Git blob SHA"):
        ensure_official_files(asset, opener=lambda request, timeout: io.BytesIO(b"wrong!!"))

    assert not expected.local_path.exists()
```

- [ ] **Step 5: Run tests and verify RED for downloader behavior**

Run the same focused pytest command. Expected: failures because `ensure_official_files` and byte-digest helpers are missing.

- [ ] **Step 6: Implement atomic download**

Use `urllib.request.Request` with a fixed user agent, URL-quote each path segment, write `<name>.part`, fsync and close, verify the part file, then `os.replace(part, final)`. Existing valid files return without network access. Invalid existing files must be renamed to `<name>.invalid` rather than overwritten silently.

- [ ] **Step 7: Run focused tests and verify GREEN**

Expected: all tests in `test_rocketbox_human_review.py` pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add external/SPEAR/tools/spike_rlr/rocketbox_human_review.py \
  external/SPEAR/tests/tools/spike_rlr/test_rocketbox_human_review.py
git commit -m "Add verified Rocketbox human source catalog"
```

Do not stage `AGENTS.md` or generated files.

---

### Task 2: Inspection And Pending Review Gate

**Files:**
- Modify: `external/SPEAR/tools/spike_rlr/rocketbox_human_review.py`
- Modify: `external/SPEAR/tests/tools/spike_rlr/test_rocketbox_human_review.py`

**Interfaces:**
- Consumes: `RocketboxReviewAsset` and verified local files from Task 1.
- Produces: `build_source_inspection(asset: RocketboxReviewAsset) -> dict[str, Any]`, `write_pending_source_review(asset: RocketboxReviewAsset, output_dir: Path, inspection: dict[str, Any]) -> Path`, `assert_source_review_approved(path: Path) -> dict[str, Any]`, and CLI subcommands `download`, `inspect`, and `approve`.

- [ ] **Step 1: Write failing payload and gate tests**

```python
def test_pending_source_review_records_axes_hashes_and_pending_status(tmp_path):
    asset = _complete_local_asset(tmp_path)
    inspection = build_source_inspection(asset)

    review_path = write_pending_source_review(asset, tmp_path / "out", inspection)
    review = json.loads(review_path.read_text(encoding="utf-8"))

    assert review["schema_version"] == "rocketbox_human_source_review_v1"
    assert review["up_axis"] == "+Z"
    assert review["forward_axis"] == "-Y"
    assert review["geometry_status"] == "pending"
    assert review["appearance_status"] == "pending"
    assert review["direction_status"] == "pending"
    assert review["source_sha256"] == sha256_file(asset.fbx.local_path)
    assert review["official_files"] == inspection["official_files"]


def test_source_review_gate_requires_all_three_human_approvals(tmp_path):
    review_path = _write_review(
        tmp_path,
        geometry_status="approved",
        appearance_status="pending",
        direction_status="approved",
    )

    with pytest.raises(SourceReviewNotApproved, match="appearance_status"):
        assert_source_review_approved(review_path)
```

- [ ] **Step 2: Run focused tests and verify RED**

Expected: missing function failures.

- [ ] **Step 3: Implement inspection, pending payload, and approval gate**

Inspection must record absolute local path, official relative path, size, Git blob SHA-1, and SHA-256 for every FBX and texture. The gate requires all three status fields to equal `approved`, a non-empty `approved_by`, and an ISO-8601 `approved_at`.

The `approve` CLI requires explicit flags and never auto-approves:

```bash
python tools/spike_rlr/rocketbox_human_review.py approve \
  --review-json /absolute/path/source_review.json \
  --reviewer jzy \
  --geometry approved \
  --appearance approved \
  --direction approved \
  --notes "reviewed in generated contact sheet and turntable"
```

- [ ] **Step 4: Add and pass CLI parser tests**

Test `parse_args([...])` for each subcommand without network or Blender. Expected: exact `command`, paths, and review statuses.

- [ ] **Step 5: Run focused tests and verify GREEN**

Expected: all source-review tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add external/SPEAR/tools/spike_rlr/rocketbox_human_review.py \
  external/SPEAR/tests/tools/spike_rlr/test_rocketbox_human_review.py
git commit -m "Add Rocketbox source review gate"
```

---

### Task 3: Rig-Preserving Blender Review Renderer

**Files:**
- Create: `external/SPEAR/tools/blender_render_rocketbox_source_review.py`
- Create: `external/SPEAR/tests/tools/test_blender_rocketbox_source_review_static.py`

**Interfaces:**
- Consumes: `--asset-id`, `--fbx`, `--texture-dir`, `--output-dir`, `--forward-axis=-Y`, and `--up-axis=+Z`.
- Produces: `front.png`, `back.png`, `left.png`, `right.png`, `top.png`, `face_close.png`, `arms_close.png`, `feet_close.png`, `turntable.mp4`, and `render_manifest.json`.

- [ ] **Step 1: Write failing static contract tests**

```python
def test_renderer_never_clears_materials_or_assigns_actions():
    source = SCRIPT.read_text(encoding="utf-8")

    assert ".materials.clear(" not in source
    assert "animation_data.action" not in source
    assert "pose_position = \"REST\"" in source


def test_renderer_declares_required_outputs_and_eevee():
    source = SCRIPT.read_text(encoding="utf-8")

    for name in ("front.png", "back.png", "left.png", "right.png", "top.png", "turntable.mp4", "render_manifest.json"):
        assert name in source
    assert '"BLENDER_EEVEE_NEXT"' in source
```

- [ ] **Step 2: Run static tests and verify RED**

Run:

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/test_blender_rocketbox_source_review_static.py
```

Expected: failure because the Blender script does not exist.

- [ ] **Step 3: Implement Blender CLI and import validation**

The script must start from factory settings, import exactly one FBX, select the largest mesh and first armature, set `armature.data.pose_position = "REST"`, clear animation data from imported objects, and validate:

```python
if len(armature.data.bones) != 80:
    raise RuntimeError(f"expected 80 Rocketbox avatar bones, got {len(armature.data.bones)}")
if [slot.material.name for slot in mesh.material_slots] != expected_material_names:
    raise RuntimeError("unexpected Rocketbox material slots")
```

Do not delete the armature, vertex groups, modifiers, UV layers, or material slots.

- [ ] **Step 4: Implement explicit material reconstruction**

Match materials by suffix `_body`, `_head`, and `_opacity`. Build Eevee node graphs with:

- color TGA as sRGB Base Color;
- normal TGA as Non-Color through a Normal Map node;
- specular TGA as Non-Color into `Specular IOR Level` when that socket exists;
- opacity TGA alpha or grayscale into Alpha, with hashed/dithered transparency when supported by Blender 4.2.

Raise with the exact missing texture path instead of rendering a flat fallback in the textured-review command.

- [ ] **Step 5: Implement deterministic views and turntable**

Use evaluated mesh bounds to frame orthographic cameras. Camera semantics are fixed:

```python
VIEWS = {
    "front": (0.0, -1.0, 0.15),
    "back": (0.0, 1.0, 0.15),
    "left": (-1.0, 0.0, 0.15),
    "right": (1.0, 0.0, 0.15),
    "top": (0.0, 0.0, 1.0),
}
```

Render at 1200x1600 for body views, 1200x900 for close views, and 1280x720 at 24 fps for a 96-frame turntable. Add a non-rotating legend with `UP +Z`, `FRONT -Y`, asset ID, source filename, and `REST POSE / NO ACTION`.

- [ ] **Step 6: Emit render manifest**

Record Blender version, source FBX SHA-256, mesh name, vertex/polygon counts, UV layer count, armature name, bone count, material slot names, view files, video file, axes, and `animation_attached: false`.

- [ ] **Step 7: Run static tests and verify GREEN**

Expected: all static contract tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add external/SPEAR/tools/blender_render_rocketbox_source_review.py \
  external/SPEAR/tests/tools/test_blender_rocketbox_source_review_static.py
git commit -m "Add Rocketbox rest-pose review renderer"
```

---

### Task 4: Real Official-Texture Preparation

**Files:**
- Generate only under `/data/datasets/rocketbox/sample/Assets/Avatars/Adults/{Male_Adult_01,Female_Adult_01}/Textures`
- Generate: `external/SPEAR/tmp/rocketbox_human_review/<asset_id>/source_inspection.json`
- Generate: `external/SPEAR/tmp/rocketbox_human_review/<asset_id>/source_review.json`

**Interfaces:**
- Consumes: Task 1 and Task 2 CLI.
- Produces: verified local TGA files and pending review metadata for both avatars.

- [ ] **Step 1: Run the complete focused unit suite**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_rocketbox_human_review.py \
  tests/tools/test_blender_rocketbox_source_review_static.py
```

Expected: all tests pass before network work.

- [ ] **Step 2: Download and verify male official files**

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python \
  tools/spike_rlr/rocketbox_human_review.py download \
  --asset-id rocketbox_male_adult_01 \
  --tree-json tmp/human_motion_source_probe/rocketbox_tree.json \
  --sample-root /data/datasets/rocketbox/sample
```

Expected sentinel: `ROCKETBOX_OFFICIAL_FILES_OK asset_id=rocketbox_male_adult_01 files=8`.

- [ ] **Step 3: Download and verify female official files**

Run the equivalent command for `rocketbox_female_adult_01`. Expected sentinel includes `files=9` because the wrinkle normal is included as optional provenance.

- [ ] **Step 4: Write pending inspection/review records**

Run `inspect` for each asset with output roots:

```text
tmp/rocketbox_human_review/rocketbox_male_adult_01
tmp/rocketbox_human_review/rocketbox_female_adult_01
```

Expected: each directory contains `source_inspection.json` and `source_review.json`; all review statuses remain `pending`.

- [ ] **Step 5: Independently verify downloaded files**

Re-run the helper's verification command with network disabled or a failing opener. Expected: every official file validates from disk and no `.part` files remain.

---

### Task 5: Render And Verify First Human Review Checkpoint

**Files:**
- Generate: `external/SPEAR/tmp/rocketbox_human_review/rocketbox_male_adult_01/*`
- Generate: `external/SPEAR/tmp/rocketbox_human_review/rocketbox_female_adult_01/*`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: verified FBX/textures and Blender renderer.
- Produces: the first user-review checkpoint. No approval JSON is changed automatically.

- [ ] **Step 1: Render male rest-pose review**

```bash
blender --background --python tools/blender_render_rocketbox_source_review.py -- \
  --asset-id rocketbox_male_adult_01 \
  --fbx /data/datasets/rocketbox/sample/Assets/Avatars/Adults/Male_Adult_01/Export/Male_Adult_01.fbx \
  --texture-dir /data/datasets/rocketbox/sample/Assets/Avatars/Adults/Male_Adult_01/Textures \
  --output-dir tmp/rocketbox_human_review/rocketbox_male_adult_01
```

Expected sentinel: `ROCKETBOX_SOURCE_REVIEW_RENDER_OK asset_id=rocketbox_male_adult_01`.

- [ ] **Step 2: Render female rest-pose review**

Run the equivalent command for the female asset and expect the female sentinel.

- [ ] **Step 3: Build contact sheets**

Use FFmpeg `xstack` to combine front/back/left/right/top into `source_views_contact.png`, and face/arms/feet into `joint_close_contact.png`. Labels are already embedded by Blender; FFmpeg must not rescale individual panels non-uniformly.

- [ ] **Step 4: Verify every rendered artifact**

Checks:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames \
  -of json tmp/rocketbox_human_review/rocketbox_male_adult_01/turntable.mp4
```

Expected: `1280x720`, `24/1`, and 96 frames for each turntable. Use Pillow to open every PNG and assert nonzero bounding-box variance. Validate `render_manifest.json` has `bone_count=80`, `animation_attached=false`, three original material slots, and at least one UV layer.

- [ ] **Step 5: Perform visual inspection**

Inspect both source contact sheets and turntables. Confirm visible official textures, face direction matches `FRONT -Y`, body is upright on `+Z`, no locomotion pose is attached, opacity regions do not render as opaque cards, and no text overlaps the character.

- [ ] **Step 6: Update AGENTS.md without staging unrelated changes**

Append the exact output roots, texture-fetch rule, render sentinel, and the warning that `locomotion_pages/*.mp4` are direct-action negative evidence rather than approved retarget results.

- [ ] **Step 7: Run final focused verification**

Run both pytest files, verify all output images/videos/manifests, and run `git diff --check`. Expected: zero test failures and zero malformed artifacts.

- [ ] **Step 8: Stop for human review**

Provide clickable links to both `source_views_contact.png`, both `joint_close_contact.png`, both `turntable.mp4`, and both pending `source_review.json` files. Do not run the approval CLI, source-motion renderer, retargeter, or 68-action batch until the user explicitly approves or rejects each source avatar.
