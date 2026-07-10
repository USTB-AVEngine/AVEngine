# Rocketbox Neutral Walk Web Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rest-correct the official male and female Rocketbox neutral walks onto the already approved textured avatars, render browser-playable review evidence, and expose independent approve/reject controls whose paired gate protects the remaining 68-action batch.

**Architecture:** A pure-Python contract module owns hashes, review decisions, and the two-avatar gate. A small NumPy module owns rest-basis and root-frame math, while Blender adapters import the FBXs, bake a new target action, export/re-import GLB, and render the evidence. A dedicated Flask application serves only allowlisted review media and writes atomic decisions; it never mutates or moves source assets.

**Tech Stack:** Python 3.11 (`spear-env`), NumPy, pytest, Blender 4.2.1 LTS Python API, Flask, HTML/CSS/vanilla JavaScript, H.264 MP4, glTF 2.0.

## Global Constraints

- User review is only of the bound, textured target avatar; source FBX, JSON, and unbound skeleton reports are internal diagnostics.
- The reviewed avatar convention is `FRONT -Y`, `UP +Z`; root-frame conversion is applied exactly once and no arbitrary container rotation is accepted.
- Preserve target mesh topology, UV layers, material slots, vertex groups, skin weights, and facial bones.
- Finger animation is optional, but shoulders, elbows, wrists, hips, knees, ankles, palms, and feet must remain connected and visually stable.
- Male and female decisions are independent; batching remains locked until both current manifests and all current media hashes are approved.
- Source and target FBXs, exported GLBs, and rendered evidence are immutable to the review server.
- The first scope is `m_walk_neutral.max.fbx` and `f_walk_neutral.max.fbx` only; do not batch the other 68 actions.
- Do not start FLUX, Qwen, Hunyuan, LongCat, FireRed, or any other model probe in this implementation.
- Keep unrelated dirty and untracked files untouched.

---

### Task 1: Motion Review Contract And Paired Gate

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/rocketbox_motion_review.py`
- Create: `external/SPEAR/tests/tools/spike_rlr/test_rocketbox_motion_review.py`

**Interfaces:**
- Consumes: `retarget_manifest.json` with `schema_version`, `asset_id`, immutable input hashes, `media`, and `automatic_checks`.
- Produces: `ensure_pending_review(review_dir: Path) -> dict`, `record_decision(review_dir: Path, decision: str, reviewer: str, notes: str) -> dict`, and `assert_pair_approved(review_root: Path) -> dict[str, dict]`.

- [ ] **Step 1: Write failing tests for current-media hashing and atomic decisions**

```python
def test_record_decision_pins_current_manifest_and_media(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")

    result = record_decision(review_dir, "approved", "jzy", "motion looks stable")

    assert result["schema_version"] == "rocketbox_motion_review_v1"
    assert result["decision"] == "approved"
    assert result["retarget_manifest_sha256"] == sha256_file(
        review_dir / "retarget_manifest.json"
    )
    assert result["media_sha256"]["front"] == sha256_file(
        review_dir / "front.mp4"
    )
    assert not (review_dir / "motion_review.json.tmp").exists()


def test_changed_media_invalidates_approval(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    record_decision(review_dir, "approved", "jzy", "approved")
    (review_dir / "front.mp4").write_bytes(b"rerendered")

    with pytest.raises(MotionReviewNotApproved, match="front.*hash"):
        assert_motion_approved(review_dir)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_rocketbox_motion_review.py
```

Expected: collection or import failure because `rocketbox_motion_review` does not exist.

- [ ] **Step 3: Implement the manifest validator and decision writer**

The module must define the exact required media names and never accept manifest paths that escape the asset directory:

```python
REQUIRED_MEDIA = (
    "front",
    "side",
    "top",
    "joints",
    "feet",
    "source_target",
    "contact_sheet",
)
EXPECTED_ASSET_IDS = (
    "rocketbox_male_adult_01",
    "rocketbox_female_adult_01",
)


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def record_decision(
    review_dir: Path, decision: str, reviewer: str, notes: str
) -> dict:
    manifest, media_paths = validate_ready_manifest(review_dir)
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")
    if not reviewer.strip():
        raise ValueError("reviewer must be non-empty")
    payload = {
        "schema_version": "rocketbox_motion_review_v1",
        "asset_id": manifest["asset_id"],
        "decision": decision,
        "reviewer": reviewer.strip(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes.strip(),
        "retarget_manifest_sha256": sha256_file(
            review_dir / "retarget_manifest.json"
        ),
        "media_sha256": {
            name: sha256_file(path) for name, path in media_paths.items()
        },
    }
    _atomic_write_json(review_dir / "motion_review.json", payload)
    return payload
```

- [ ] **Step 4: Add failing then passing tests for the paired gate**

Cover missing female, rejected female, stale male media, failed automatic checks, and two current approvals. `assert_pair_approved()` must return both records only in the final case.

- [ ] **Step 5: Run contract tests and commit**

Run:

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_rocketbox_motion_review.py
```

Expected: all tests pass.

Commit in `external/SPEAR`:

```bash
git add tools/spike_rlr/rocketbox_motion_review.py \
  tests/tools/spike_rlr/test_rocketbox_motion_review.py
git commit -m "Add Rocketbox motion review contract"
```

### Task 2: Rest-Basis And Root-Frame Math

**Files:**
- Create: `external/SPEAR/tools/rocketbox_retarget_math.py`
- Create: `external/SPEAR/tests/tools/test_rocketbox_retarget_math.py`

**Interfaces:**
- Consumes: 4x4 source/target rest and pose matrices plus horizontal source/target forward vectors.
- Produces: `rest_delta()`, `apply_rest_delta()`, `horizontal_alignment()`, `scaled_root_translation()`, and `loop_residual()` using NumPy arrays.

- [ ] **Step 1: Write a failing synthetic rest-correction test**

```python
def test_rest_delta_preserves_motion_across_different_bind_axes():
    source_rest = rotation_z(45)
    source_pose = source_rest @ rotation_x(30)
    target_rest = rotation_z(-20)

    delta = rest_delta(source_rest, source_pose)
    target_pose = apply_rest_delta(target_rest, delta)

    assert_matrix_close(np.linalg.inv(target_rest) @ target_pose, rotation_x(30))
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/test_rocketbox_retarget_math.py
```

Expected: import failure because `rocketbox_retarget_math` does not exist.

- [ ] **Step 3: Implement parent-local rest-delta math**

```python
def rest_delta(rest_local: np.ndarray, pose_local: np.ndarray) -> np.ndarray:
    return np.linalg.inv(rest_local) @ pose_local


def apply_rest_delta(target_rest_local: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return target_rest_local @ delta


def parent_local(parent_matrix: np.ndarray | None, child_matrix: np.ndarray) -> np.ndarray:
    return child_matrix if parent_matrix is None else np.linalg.inv(parent_matrix) @ child_matrix
```

- [ ] **Step 4: Add RED/GREEN tests for root alignment and loop residual**

Test that `horizontal_alignment([0, 1, 0], [0, -1, 0])` is a 180-degree Z rotation derived from the two motion frames, while identity input remains identity. Test that root translation is scaled once and that loop residual subtracts the expected cycle displacement before measuring the seam.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/test_rocketbox_retarget_math.py
```

Expected: all tests pass.

Commit:

```bash
git add tools/rocketbox_retarget_math.py tests/tools/test_rocketbox_retarget_math.py
git commit -m "Add Rocketbox rest correction math"
```

### Task 3: Blender Neutral-Walk Retarget And Export

**Files:**
- Create: `external/SPEAR/tools/blender_retarget_rocketbox_walk.py`
- Create: `external/SPEAR/tests/tools/test_blender_retarget_rocketbox_walk_static.py`

**Interfaces:**
- Consumes: approved `source_review.json`, official avatar FBX and textures, one Rocketbox motion FBX, asset ID, and output directory.
- Produces: `retarget.blend`, `retarget.glb`, `retarget_metrics.json`, `retarget_manifest.json`, and sentinel `ROCKETBOX_RETARGET_OK asset_id=<id>`.

- [ ] **Step 1: Write static contract tests before the Blender script**

Assert the CLI exposes `--asset-id`, `--avatar-fbx`, `--texture-dir`, `--texture-prefix`, `--motion-fbx`, `--source-review-json`, and `--output-dir`. Assert the source-review gate is called, source-only `Nub` bones are ignored, required body bones are explicit, target materials are not cleared, the target action is newly created and keyframed, and both GLB export and re-import verification are present.

- [ ] **Step 2: Run the static test and verify RED**

Run:

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/test_blender_retarget_rocketbox_walk_static.py
```

Expected: failure because the script is absent.

- [ ] **Step 3: Implement import validation and explicit body mapping**

Use all shared non-`Nub` target bones, but fail unless this core set exists in both rigs:

```python
CORE_BONES = (
    "Bip01 Pelvis", "Bip01 Spine", "Bip01 Spine1", "Bip01 Spine2",
    "Bip01 Neck", "Bip01 Head",
    "Bip01 L Clavicle", "Bip01 L UpperArm", "Bip01 L Forearm", "Bip01 L Hand",
    "Bip01 R Clavicle", "Bip01 R UpperArm", "Bip01 R Forearm", "Bip01 R Hand",
    "Bip01 L Thigh", "Bip01 L Calf", "Bip01 L Foot", "Bip01 L Toe0",
    "Bip01 R Thigh", "Bip01 R Calf", "Bip01 R Foot", "Bip01 R Toe0",
)
```

Import and rebuild official materials through the existing functions in `blender_render_rocketbox_source_review.py`. Capture pre-retarget vertex, polygon, UV-layer, material-slot, vertex-group, and bone counts.

- [ ] **Step 4: Implement evaluated-frame retarget baking**

For every source frame, evaluate the source action, solve mapped bones parent-first from source parent-local rest deltas into target parent-local rest bases, assign target pose matrices, and insert quaternion/location/scale keys. Derive one horizontal root-frame alignment from measured source travel to reviewed `-Y`; transform root displacement and root yaw with that matrix exactly once. Keep facial target bones at their target rest unless a same-named source curve drives them.

- [ ] **Step 5: Add export and re-import invariants**

Export only the target mesh and armature to `retarget.glb`, save `retarget.blend`, then re-import the GLB into a fresh temporary scene and confirm one skinned mesh, an armature, a non-empty action, unchanged mesh/material counts, and the required core bones. Write hashes, source frame range/FPS, mapped/unmapped bones, rest-angle statistics, root alignment, travel vector, loop residual, floor metrics, and all invariants to JSON.

- [ ] **Step 6: Run static tests and one male Blender smoke**

Run:

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/test_blender_retarget_rocketbox_walk_static.py

blender --background --python tools/blender_retarget_rocketbox_walk.py -- \
  --asset-id rocketbox_male_adult_01 \
  --avatar-fbx /data/datasets/rocketbox/sample/Assets/Avatars/Adults/Male_Adult_01/Export/Male_Adult_01.fbx \
  --texture-dir /data/datasets/rocketbox/sample/Assets/Avatars/Adults/Male_Adult_01/Textures \
  --texture-prefix m002 \
  --motion-fbx /data/datasets/rocketbox/sample/Assets/Animations/all_animations_max_motextr_xy/m_walk_neutral.max.fbx \
  --source-review-json tmp/rocketbox_human_review/rocketbox_male_adult_01/source_review.json \
  --output-dir tmp/rocketbox_motion_review/rocketbox_male_adult_01
```

Expected: tests pass and Blender prints `ROCKETBOX_RETARGET_OK asset_id=rocketbox_male_adult_01`.

- [ ] **Step 7: Commit**

```bash
git add tools/blender_retarget_rocketbox_walk.py \
  tests/tools/test_blender_retarget_rocketbox_walk_static.py
git commit -m "Retarget Rocketbox neutral walks"
```

### Task 4: Bound-Avatar Review Renderer

**Files:**
- Create: `external/SPEAR/tools/blender_render_rocketbox_motion_review.py`
- Create: `external/SPEAR/tests/tools/test_blender_render_rocketbox_motion_review_static.py`

**Interfaces:**
- Consumes: `retarget.blend`, `retarget.glb`, `retarget_metrics.json`, source motion FBX, and output directory.
- Produces: `front.mp4`, `side.mp4`, `top.mp4`, `joints.mp4`, `feet.mp4`, `source_target.mp4`, `contact_sheet.png`, and a ready `retarget_manifest.json` with hashes for every artifact.

- [ ] **Step 1: Write failing renderer contract tests**

Assert all seven media names are declared, videos are H.264 MP4 at 1280x720 and source FPS, the target textured mesh is visible in every video, `FRONT -Y`/`UP +Z` labels are present, the top view draws root trajectory and facing arrow, and joints/feet views use root-following cameras rather than changing target transforms.

- [ ] **Step 2: Run the renderer test and verify RED**

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/test_blender_render_rocketbox_motion_review_static.py
```

Expected: failure because the renderer is absent.

- [ ] **Step 3: Implement deterministic review views**

Load `retarget.blend`, validate the GLB re-import result recorded by Task 3, and render two gait cycles. Front and side keep the entire translated path framed; joints and feet track `Bip01 Pelvis`; top shows the complete path and a red `FRONT -Y` arrow. `source_target.mp4` places a thin source skeleton beside the bound target while keeping the target larger and fully visible.

- [ ] **Step 4: Implement contact and automatic QA evidence**

Sample at least eight evenly spaced gait frames into `contact_sheet.png`. Update automatic checks for core-map coverage, forward/travel dot product, normalized loop residual, floor penetration, preserved mesh/material counts, and GLB re-import. Any failed check leaves the manifest non-reviewable.

- [ ] **Step 5: Run static tests and render both review roots**

Run the renderer for the male output, then create the female retarget with:

```bash
blender --background --python tools/blender_retarget_rocketbox_walk.py -- \
  --asset-id rocketbox_female_adult_01 \
  --avatar-fbx /data/datasets/rocketbox/sample/Assets/Avatars/Adults/Female_Adult_01/Export/Female_Adult_01.fbx \
  --texture-dir /data/datasets/rocketbox/sample/Assets/Avatars/Adults/Female_Adult_01/Textures \
  --texture-prefix f001 \
  --motion-fbx /data/datasets/rocketbox/sample/Assets/Animations/all_animations_max_motextr_xy/f_walk_neutral.max.fbx \
  --source-review-json tmp/rocketbox_human_review/rocketbox_female_adult_01/source_review.json \
  --output-dir tmp/rocketbox_motion_review/rocketbox_female_adult_01
```

Invoke `blender_render_rocketbox_motion_review.py` once for each completed
retarget root. Expected output in each root is the seven media artifacts plus a
manifest whose `automatic_checks.overall` equals `passed`.

- [ ] **Step 6: Inspect generated media metadata and commit**

Run:

```bash
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate \
  -of json tmp/rocketbox_motion_review/rocketbox_male_adult_01/front.mp4
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/test_blender_render_rocketbox_motion_review_static.py
```

Expected: H.264, 1280x720, source frame rate, and passing tests.

Commit:

```bash
git add tools/blender_render_rocketbox_motion_review.py \
  tests/tools/test_blender_render_rocketbox_motion_review_static.py
git commit -m "Render Rocketbox motion review evidence"
```

### Task 5: Dedicated Motion Review Web UI

**Files:**
- Create: `external/SPEAR/tools/spike_rlr/rocketbox_motion_review_server.py`
- Create: `external/SPEAR/tests/tools/spike_rlr/test_rocketbox_motion_review_server.py`

**Interfaces:**
- Consumes: a review root containing the two asset directories from Tasks 3-4.
- Produces: Flask routes `GET /`, `GET /asset/<asset_id>`, `GET /media/<asset_id>/<kind>`, `POST /decision/<asset_id>`, and `GET /gate`; writes decisions only through `record_decision()`.

- [ ] **Step 1: Write failing Flask test-client tests**

```python
def test_asset_page_shows_bound_video_tabs_and_decision_controls(workspace):
    client = create_app(workspace).test_client()
    response = client.get("/asset/rocketbox_male_adult_01")

    assert response.status_code == 200
    assert b'<video' in response.data
    for label in (b"Front", b"Side", b"Top", b"Joints", b"Feet", b"Source + Target"):
        assert label in response.data
    assert b"Approve" in response.data
    assert b"Reject" in response.data


def test_media_route_rejects_unknown_kind_and_path_traversal(workspace):
    client = create_app(workspace).test_client()
    assert client.get("/media/rocketbox_male_adult_01/front").status_code == 200
    assert client.get("/media/rocketbox_male_adult_01/unknown").status_code == 404
    assert client.get("/media/../front").status_code in {404, 405}
```

- [ ] **Step 2: Run server tests and verify RED**

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_rocketbox_motion_review_server.py
```

Expected: import failure because the server does not exist.

- [ ] **Step 3: Implement the operational review page**

Use one unframed main review area, a compact male/female status rail, stable 16:9 video dimensions, media tabs, a notes textarea, and approve/reject buttons. Videos use `controls`, `loop`, `muted`, and `playsinline`. The page must show the current decision and the paired gate state, but hide raw JSON unless a compact diagnostics disclosure is opened.

- [ ] **Step 4: Implement safe routes and decision replacement**

Resolve assets only from `EXPECTED_ASSET_IDS` and media only from `REQUIRED_MEDIA`. `POST /decision/<asset_id>` calls `record_decision()` and redirects to the same asset so a corrected rerender or changed decision can be reviewed without moving directories.

- [ ] **Step 5: Run server and contract tests, then commit**

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_rocketbox_motion_review.py \
  tests/tools/spike_rlr/test_rocketbox_motion_review_server.py
```

Expected: all tests pass.

Commit:

```bash
git add tools/spike_rlr/rocketbox_motion_review_server.py \
  tests/tools/spike_rlr/test_rocketbox_motion_review_server.py
git commit -m "Add Rocketbox motion review web UI"
```

### Task 6: End-To-End Verification And Live Review URL

**Files:**
- Modify only if a newly discovered durable trap must be recorded: `AGENTS.md`
- Generated, untracked evidence: `external/SPEAR/tmp/rocketbox_motion_review/`

**Interfaces:**
- Consumes: all prior tasks and the approved male/female source records.
- Produces: verified male/female review artifacts, a running local server, screenshots, and the URL given to the user.

- [ ] **Step 1: Run the full focused test suite**

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_rocketbox_human_review.py \
  tests/tools/test_blender_rocketbox_source_review_static.py \
  tests/tools/test_rocketbox_retarget_math.py \
  tests/tools/test_blender_retarget_rocketbox_walk_static.py \
  tests/tools/test_blender_render_rocketbox_motion_review_static.py \
  tests/tools/spike_rlr/test_rocketbox_motion_review.py \
  tests/tools/spike_rlr/test_rocketbox_motion_review_server.py
```

Expected: all tests pass with no warnings or collection errors.

- [ ] **Step 2: Verify artifact hashes and media playback**

Call `validate_ready_manifest()` for both asset roots, run `ffprobe` over every MP4, decode at least the first and midpoint frames with ffmpeg, and reject blank/near-uniform frames before opening review.

- [ ] **Step 3: Start the Flask server on an unused port**

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python \
  tools/spike_rlr/rocketbox_motion_review_server.py \
  --review-root tmp/rocketbox_motion_review \
  --host 0.0.0.0 --port 8091
```

Keep the process running and report `http://<server-host>:8091/` plus the SSH-forward form if the IDE cannot reach the server directly.

- [ ] **Step 4: Verify desktop and mobile layouts with Playwright**

Use Playwright at 1440x900 and 390x844. Capture screenshots under `tmp/rocketbox_motion_review/ui_qa/`, confirm the video canvas is nonblank, exercise every media tab, verify no text or controls overlap, and submit a decision only against a disposable copied fixture rather than the real male/female records.

- [ ] **Step 5: Perform final repository and process checks**

Confirm the Flask session is alive, no Blender render process remains, no model download/probe session was started, and only intended tracked files were committed. Preserve all unrelated dirty/untracked files.

- [ ] **Step 6: Hand the visual gate to the user**

Report the live URL, list the male and female asset IDs, and state the five user-visible checks: facing/travel direction, shoulder/elbow/wrist integrity, hip/knee/ankle bending, foot contact/sliding, and overall naturalness. Do not ask the user to inspect FBX or JSON.
