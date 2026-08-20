# M5 Execution

## Current installed MP3D visual research

`capture-current-visual` is a separate, visual-only research path. It uses an
installed Habitat prefix, an external MP3D data root, and an external Magnum
Python site. It creates only co-located RGB/depth/semantic observations; it
does not invoke the M5 v1 counterfactual writer, RLR, M3, M4, dry-audio
assembly, or an audio sensor.

All three runtime inputs are explicit. `--runtime-root` is deliberately not an
option for this command, so an old Habitat checkout cannot be selected as a
compatibility fallback.

```bash
python -m avengine.cli m5 capture-current-visual \
  --animal-manifest /path/to/beagle_asset_manifest.json \
  --m2-request /path/to/75_state_request_authored_for_this_mp3d_room.json \
  --room-manifest examples/m1/rooms/habitat_mp3d_example/room_manifest.json \
  --m1-request examples/m1/requests/habitat_mp3d_example.json \
  --runtime-prefix "$HABITAT_PREFIX" \
  --mp3d-root "$MP3D_ROOT" \
  --magnum-python-site "$MAGNUM_PYTHON_SITE" \
  --output /external/fresh/m5_current_visual
```

The M2 request's `room_id` must equal the selected MP3D room. The retained
Beagle request is authored in Blender coordinates, so it returns `not_run`
before activating Habitat rather than rendering those coordinates in MP3D.
Do not change its room ID or translate coordinates by hand. Supply a new M2
request authored for the same MP3D room before attempting a native run.

For a same-room request, the joined M1/M2 capture context and declared MP3D
asset graph are checked before any runtime activation or output creation. After
the declared navmesh loads, the final Habitat scene graph is checked before
actors are created. One transient no-actor semantic observation rejects a scene
that already uses either selected actor semantic ID; every one of the 75
rendered frames must contain both actor IDs. These checks are per-run safety
checks only: they do not add a new receipt schema, evidence reader, hash,
baseline, or gate.


The fresh output contains raw visual arrays, frame readback records, and a
plain `research_receipt.json`. It is research-only, does not count an episode,
and is not interpreted by the retained M5 v1 reader.


## Current MP3D two-Beagle route author

`author-current-mp3d-two-beagle-route` creates a fresh, research-only
75-frame route for the current MP3D sample. It reads a user-provided,
non-Git, canary-qualified Beagle package and its matching source M2 request;
the source request is read-only and is never renamed, relabelled, or changed.
The author preserves the 75-frame Idle/Walk/Idle action timing, joint states,
contacts, mouth policy, and pose hashes, then recomputes only the existing M2
`applied_state_hash` values required for the new root transforms.

M2 v1 remains a single-articulated-asset request. Therefore the output contains
one normal M2-compatible primary request, one new static research M1
camera/listener request, and a plain research explanation. M5 current-visual
owns the two same-asset instances and its fixed offsets; the explanation
records the two resulting skin-root paths rather than pretending a new
multi-actor M2 schema exists.

```bash
python -m avengine.cli m5 author-current-mp3d-two-beagle-route \
  --source-animal-manifest "$BEAGLE_PACKAGE/asset_manifest.json" \
  --source-m2-request "$BEAGLE_M2_SOURCE_REQUEST" \
  --runtime-prefix "$HABITAT_PREFIX" \
  --mp3d-root "$MP3D_ROOT" \
  --magnum-python-site "$MAGNUM_PYTHON_SITE" \
  --output /data/avengine_external/review/current_mp3d_two_beagle_route_unique
```

All paths must be canonical, external, and outside Git checkouts. The output
must be a fresh immediate child of `/data/avengine_external/review`; the writer
will not create it until the current room graph, native Habitat PathFinder,
shared M5.1 no-sliding checks, M6x source-center feasibility, one navmesh
island, and two-Beagle center separation have all passed.

The author leaves the checked-in M1 request and its camera rig untouched. It
creates a new static `research_m1_request.json` for this research scenario:
the camera X/Z is directly rechecked with the loaded native PathFinder
(`is_navigable`, snap error, clearance, and island), its height uses the room
agent height, and its yaw points at the two-Beagle path midpoint. A conservative
frustum filter is followed by native all-75-frame semantic
readback of both real same-asset instances before any output is written. The
new M1 request is still an ordinary M1-compatible `camera_rig_0/view0`
request so it can be passed explicitly to `capture-current-visual`; its
research-only status is carried only by the surrounding explanation, never by
changing the retained M1 schema or old reader.

If no source-length route/camera pair passes native navmesh, no-sliding,
frustum, separation, and semantic checks, the author returns a blocker without
creating output. It does not shorten or fold source motion, move the existing
M1 camera, or change instance offsets.

After a successful author run, pass its two new request files explicitly to the
existing visual-only capture command; keep its output fresh as well:

```bash
python -m avengine.cli m5 capture-current-visual \
  --animal-manifest "$BEAGLE_PACKAGE/asset_manifest.json" \
  --m2-request /data/avengine_external/review/current_mp3d_two_beagle_route_unique/primary_m2_request.json \
  --room-manifest examples/m1/rooms/habitat_mp3d_example/room_manifest.json \
  --m1-request /data/avengine_external/review/current_mp3d_two_beagle_route_unique/research_m1_request.json \
  --runtime-prefix "$HABITAT_PREFIX" \
  --mp3d-root "$MP3D_ROOT" \
  --magnum-python-site "$MAGNUM_PYTHON_SITE" \
  --output /data/avengine_external/review/current_mp3d_two_beagle_visual_unique
```

This is research-only: it adds no new schema, persistent route hash, baseline,
contract, or formal gate, and it makes no body-volume collision, audio/RLR,
formal episode, or equivalence claim.


## Offline current-M1 research audio assembly

`render-current-m1-research-audio` joins one completed current-M1 FOA receipt
and one completed current-M1 native-binaural receipt. It is a CPU-only offline
handoff: the command has no runtime, SDK, HRTF, native, UE, or GPU argument and
does not activate RLR again.

Before writing output it requires both receipts to agree on their M1 request,
simulation request, acoustic package and room, listener pose, `source0` and
`source1` positions, propagation switches, current installed-runtime identity,
package-QA report, 16 kHz rate, and non-formal research boundary. FOA must say
`hrtf_used=false`; binaural must retain passing HRTF and native-SOFA preflights.
Each pair is joined by `source_id` and its authenticated existing WAV sidecar,
not by JSON-list position, and both the WAV and sidecar must remain inside the
receipt directory.

```bash
export FOA_RECEIPT=/external/review/current_m1_foa/research_receipt.json
export BINAURAL_RECEIPT=/external/review/current_m1_binaural/research_receipt.json
export M5_AUDIO=/external/review/current_m1_audio_<RUN_ID>

"$HABPY" -m avengine.cli m5 render-current-m1-research-audio \
  --foa-receipt "$FOA_RECEIPT" \
  --binaural-receipt "$BINAURAL_RECEIPT" \
  --output "$M5_AUDIO"
```

The deterministic dry buses are exactly 80,000 samples: `source0` is a 440 Hz,
0.25-amplitude sine at phase 0 and `source1` is a 660 Hz, 0.25-amplitude sine
at phase pi/2. A single sample-0 static keyframe reuses the ordinary M5 dynamic
stem assembler for FOA and binaural separately. It records each full linear
convolution length but writes only the explicit five-second `[0,80000)` episode
crop, with no resampling, normalization, or limiter:

```text
dry/{source0,source1}.wav
foa/stems/{source0,source1}.wav
foa/mix.wav
binaural/stems/{source0,source1}.wav
binaural/mix.wav
research_receipt.json
```

All eight float32 WAVs retain the existing adjacent WAV sidecar format. The
ordinary research receipt has no new schema, baseline, or verifier gate and
keeps `research_only=true`, `episode_counted=false`,
`formal_dataset_count=0`, and `qualification=false`.


### Offline human review

Only after a completed `research_only` receipt exists, generate a fresh local
review directory with synchronized RGB, depth and semantic panels:

```bash
python -m avengine.cli m5 review-current-visual \
  --research-output /external/fresh/m5_current_visual \
  --review-output /external/review/m5_current_visual
```

Omit `--review-output` to create the no-clobber `review/` directory beneath the
research output. The page presents frame/time, planned actor pose, runtime
source readback and semantic visibility when those arrays are supplied. Its
free-text notes stay in browser-local storage unless the reviewer explicitly
downloads them. It never rewrites `research_receipt.json`, creates a pass or
review decision, or changes formal/admission evidence.

## Historical M5 v1 counterfactual

Run from the AVEngine Habitat-native repository with the pinned Habitat Python
environment. The output directory must not already exist, and a formal run
requires a clean AVEngine worktree because the evidence binds the source
commit before creating its staging directory.

```bash
export PATH=/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin:$PATH
export PYTHONPATH="$PWD:$PWD/src"

python -m avengine.cli m5 run-canary \
  --request examples/m5/blender_custom/two_dog_simultaneous_counterfactual_request.json \
  --animal-manifest tmp/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5/asset_manifest.json \
  --m2-request tmp/m2/rocketbox_beagle_m2_formal_request_v7_world_contact_r5.json \
  --room-manifest examples/m1/rooms/blender_custom/room_manifest.json \
  --m1-request examples/m1/requests/blender_custom.json \
  --acoustic-package-manifest tmp/m3/formal_20260717_01/compile/low_absorption/manifest.json \
  --m4-request examples/m4/blender_custom/multi_source_canary_request.json \
  --runtime-root /data/jzy/code/habitat-sim-AVEngine \
  --hrtf /usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa \
  --hrtf-license /usr/share/doc/libmysofa1/copyright \
  --beagle-dry /data/jzy/code/AVEngine/external/SPEAR/tmp/animal_audio_event_audit_v1/dog_beagle_v2_scheduled_dry.wav \
  --golden-dry /data/jzy/code/AVEngine/external/SPEAR/tmp/animal_audio_event_audit_v1/dog_golden_scheduled_dry.wav \
  --output tmp/m5/formal_20260718_02
```

The dry-audio paths above are local research inputs. If the exact workspace
uses different filenames, use the paths whose hashes match the request; the
runner rejects an undeclared substitution.

Independently verify the completed bundle:

```bash
python -m avengine.cli m5 verify-canary \
  tmp/m5/formal_20260718_02/evidence.json
```

The verifier returns `pass` only after reading and rebuilding the retained
evidence. The primary listening videos are under `videos/`; exact FOA and
binaural WAV authority, source stems, timelines, RIR arrays, spatial reports,
and input snapshots remain inside the evidence tree.
