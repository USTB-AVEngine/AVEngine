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
