# M5 Execution

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
