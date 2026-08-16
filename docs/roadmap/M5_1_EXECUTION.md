# M5.1 execution

M5.1 is a bounded 270-frame/18-second research review.  It is not a
Timeline-v2 five-second production Episode, a room qualification, or an
equivalence result.  This page describes the current AVEngine-owned MP3D
entrypoint and clearly separates it from the retained checkout-era evidence.

## Current MP3D runtime boundary

Run the M5.1 MP3D writer from this repository plus three independently
provisioned external inputs:

- a non-Git installed Habitat runtime prefix, containing the selected
  `habitat_sim` facade, binding, and default physics configuration;
- a licensed MP3D data root containing `scene_datasets/`; and
- a Corrade/Magnum Python site compatible with the active interpreter.

The prefix is runtime code, not a source checkout or build tree.  MP3D is
licensed external data and is never copied into Git.  Corrade/Magnum remains
an explicit external Python dependency; it is neither installed into the
Habitat prefix nor inferred from an old checkout.

```bash
export REPO=/path/to/AVEngine
export HABPY=/path/to/python
export HABITAT_PREFIX=/path/to/installed-habitat-runtime
export MP3D_ROOT=/path/to/licensed-mp3d-data
export HABITAT_MAGNUM_PYTHON_SITE=/path/to/magnum-python/site-packages

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export AVENGINE_HABITAT_RUNTIME_PREFIX="$HABITAT_PREFIX"
export AVENGINE_MP3D_ROOT="$MP3D_ROOT"
export AVENGINE_HABITAT_MAGNUM_PYTHON_SITE="$HABITAT_MAGNUM_PYTHON_SITE"
cd "$REPO"
```

`$MP3D_ROOT` must contain `scene_datasets/mp3d_example/` with the selected
scene GLB, semantic PLY, `.house`, navmesh, and dataset configuration.  The
current small external-data manifest is
`examples/m1/rooms/habitat_mp3d_example/room_manifest.json`; every MP3D path
in it resolves through `AVENGINE_MP3D_ROOT`.

The M5.1 route authority itself is the checked-in migrated
`examples/m5_1/mp3d_articulated_review/route_manifest.json`.  Its route is
still paired with the historical articulated room/request inputs under
`examples/m2/rooms/habitat_mp3d_articulated_review/`; those files expand the
old checkout-root variable and therefore are historical evidence, not current
runtime manifests.  A fresh M5.1 replay must first use an adopted matching
room/request pair whose scene, assets, and navmesh all resolve through
`AVENGINE_MP3D_ROOT`.  Do not substitute an old checkout merely to make that
replay run.

## Current command form

Once that adopted M5.1 room/request pair and the separately authorized human
and animal runtime inputs are available, use the AVEngine entrypoint in this
form, with a fresh output path:

```bash
RUN_ID=replace_with_unique_tag

"$HABPY" tools/m5_1/capture_human_beagle_mp3d.py \
  --route-manifest examples/m5_1/mp3d_articulated_review/route_manifest.json \
  --room-manifest /path/to/adopted_m5_1_mp3d_room_manifest.json \
  --m1-request /path/to/adopted_m5_1_mp3d_capture_request.json \
  --human-runtime-glb /path/to/authorized_human_runtime.glb \
  --beagle-manifest /path/to/authorized_beagle_asset_manifest.json \
  --beagle-m2-request /path/to/authorized_beagle_request.json \
  --runtime-prefix "$HABITAT_PREFIX" \
  --mp3d-root "$MP3D_ROOT" \
  --magnum-python-site "$HABITAT_MAGNUM_PYTHON_SITE" \
  --output "tmp/m5_1/mp3d_mixed_${RUN_ID}"
```

`--runtime-prefix` is the preferred spelling.  For a room already migrated to
`AVENGINE_MP3D_ROOT`, the CLI and Python writer keep `runtime_root` only as a
compatibility alias for that same non-Git installed prefix; a path inside a Git
checkout is rejected.  An already configured
`AVENGINE_HABITAT_RUNTIME_PREFIX`, together with the MP3D and Magnum
environment variables above, selects the same installed path.  The writer
does not fall back to `discover_runtime_root` or direct checkout imports after
one of those new inputs is present.

The output must be new.  The command performs one visual capture and one
PathFinder preflight for its request; it does not create RLR evidence, qualify
the room, or turn this M5.1 review into a production Episode.

## Retained historical material

The former commands in this document that named a Habitat checkout, its
checkout-relative scene manifests, or a root-style runtime argument were
instructions for the retained 2026 M5.1 review.  They are past-tense
reproducibility records only and must not be copied into a current invocation.
Their captures, acoustics, deliveries, and limitations remain described in
[M5.1 status](M5_1_STATUS.md).  The historical Legacy Apartment and ReplicaCAD
routes also retain their own checkout-based compatibility branches until their
separate source/data migrations are complete; they are not MP3D runtime
fallbacks.

The M5.1 source/event/flag semantics remain readable through
`examples/m5_1/legacy_apartment/source_manifest.json`, but that retained
semantic record does not authorize its historical external asset paths for a
new run.

## Focused regression coverage

The source-level selection behavior is covered without native rendering:

```bash
PYTHONPATH="$PWD/src" "$HABPY" -m pytest -q \
  tests/unit/test_m5_1_mixed_capture.py \
  tests/unit/test_m5_1_mp3d_capture.py \
  tests/unit/test_m1_habitat_capture.py
```

These ordinary tests cover the installed-prefix alias, prefix environment
selection, checkout rejection, and no-new-input legacy branch.  A passing unit
suite does not replace a native Habitat capture or the required fresh
pre/post-equivalence matrix.
