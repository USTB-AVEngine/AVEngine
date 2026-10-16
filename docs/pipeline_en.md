# Current Animal Audio-Visual Dataset Construction Pipeline

> Snapshot date: 2026-07-06
>
> Scope: current SPEAR-side pipeline for generating animal visual assets,
> importing animated/static animals into Unreal, composing GPURIR-aligned
> scenes, rendering videos, generating 4-channel room audio, and muxing playable
> MP4 outputs.

## 1. Inputs

The pipeline currently combines four data sources:

1. Visual animal prompts and reference images
   - Prompt templates live in `tools/batch_animal_pipeline.py`.
   - The important prompt constraints are side profile view, plain white
     background, separated legs, and a tail or appendage pose that avoids
     contact with legs/body.

2. Hunyuan3D assets
   - Hunyuan3D-2.1 lives at `/data/jzy/code/Hunyuan3D-2.1`.
   - Local weights are under `/data/jzy/code/Hunyuan3D-2.1/pretrained_models`.
   - Batch outputs are under `/data/jzy/code/SPEAR/tmp/hy3d_batch/<tag>/`.

3. Source rigs
   - Quaternius source rigs are read from
     `/data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/`.
   - Current animated rigs:
     - `Dog.glb` for `dog_golden`, `dog_husky`
     - `Cat.glb` for `cat_persian`, `cat_tabby`, `chipmunk`
   - Large ungulates are currently imported as static meshes, not animated
     skeletal meshes.

4. Source audio
   - Primary source: `/data/datasets/omniaudio/train-data-az-360-large`.
   - Audio lookup rules live in `tools/gpurir_scenes/audio_registry.py`.
   - Stable Audio Open is the fallback for tags without a usable local match.

## 2. Visual Asset Generation

The batch visual asset generator is `tools/batch_animal_pipeline.py`.

For each tag in `SPECIES_LIST`, it runs:

1. Flux reference image generation
   - Input: prompt template + species/breed name.
   - Output: `reference.png`.
   - Prompt constraints are tuned for downstream rig transfer, not only image
     quality: legs separated, body side-on, clean white background, and tail or
     wings away from the torso.

2. Hunyuan3D-Shape
   - Uses `Hunyuan3DDiTFlowMatchingPipeline`.
   - Output: `shape.glb`.

3. Hunyuan3D-Paint
   - Wrapper: `tools/hy3d_bake_diffuse.py`.
   - Outputs:
     - `hy3d_textured.obj`
     - `hy3d_diffuse.jpg`
     - `hy3d_metallic.jpg`
     - `hy3d_roughness.jpg`

4. Blender turntable QC
   - Produces `turntable_grid.png`.
   - This checks whether the Hunyuan mesh is visually recognizable before
     spending time on rig transfer or Unreal import.

The earlier 10-species batch succeeded for:

`dog_golden`, `dog_husky`, `cat_persian`, `cat_tabby`, `mouse_white`,
`mouse_gray`, `chicken_hen`, `chicken_rooster`, `duck_mallard`,
`bird_canary`.

## 3. Rig Transfer And Animation

Animated animals are defined by `ANIMATED_RIG_MAP` in
`tools/species_rig_map.py`.

Current animated tags:

`cat_persian`, `cat_tabby`, `chipmunk`, `dog_golden`, `dog_husky`

Rig transfer is performed by:

`tools/blender_robust_swap_mesh_keep_rig.py`

The robust transfer strategy is:

1. Import the source rig GLB and the Hunyuan target mesh.
2. Align target mesh to the source rig coordinate system.
3. Remove obvious ground cards and low limb-bridge faces when present.
4. Segment the target mesh into coarse semantic regions:
   - tail
   - torso
   - head
   - front left/right legs
   - hind left/right legs
5. Transfer skin weights only from compatible source regions.
6. Inpaint missing weights over the mesh graph.
7. Keep top-k normalized bone weights per vertex.
8. Optionally damp head/tail/foot rotations to reduce strange motion.
9. Export a UE-safe GLB with the source rig's animations preserved.

Gate check command:

```bash
bash tools/gate_check_animal.sh <tag>
```

This produces a side-view walking video under:

`/tmp/gate_check_v4/<tag>_side.mp4`

## 4. Static Animal Import Path

Static animals are defined by `STATIC_MESH_MAP` in `tools/species_rig_map.py`.

Current static tags:

`goat`, `sheep`, `pig`, `horse`, `cattle_bovinae`, `yak`, `donkey_ass`

Static mesh import is handled by:

- `tools/gpurir_scenes/render_gate_animal_editor.py`
- `tools/gpurir_scenes/build_static_meshes.sh`

The imported UE Blueprint path format is:

`/Game/MyAssets/Audioset/Blueprints/gate_static_<tag>/BP_gate_static_<tag>`

These animals can be placed and rendered in scenes, but they do not currently
walk. They still participate in audio simulation as static sound sources.

## 5. Unreal Import And Cook

Animated animals are imported by `tools/import_gate_animal_editor.py`.

For each animated tag, it creates:

- Skeletal mesh assets under
  `/Game/MyAssets/Audioset/Meshes/gate_<tag>/`
- A Blueprint under
  `/Game/MyAssets/Audioset/Blueprints/gate_<tag>/BP_gate_<tag>`
- A SkeletalMeshComponent configured to play the Walking animation by default.

Static animals create equivalent `gate_static_<tag>` Blueprints with a
StaticMeshComponent.

After import, Unreal cook is run through the existing UAT wrapper so the
Standalone SPEAR runtime can load the assets.

## 6. Scene Specification

Scene composition lives in `tools/gpurir_scenes/scene_spec.py`.

The deterministic scene contract is:

- Room size: `5.2 m x 4.4 m x 2.8 m`
- Reverberation: `T60 = 0.45 s`
- Mic position: `(2.6, 2.2, 1.2)` meters
- Duration: `5 s`
- Video frames: `75` frames at `15 fps`
- Audio sample rate: `16 kHz`
- Number of animals: `1` or `2`
- Camera yaws per room: four fixed views
- Animated trajectories: 75-frame xy trajectories with per-frame yaw
- Static placements: one safe xy position and static yaw

Collision is not delegated to Unreal physics. SPEAR uses `AlwaysSpawn` and
per-frame teleporting, so runtime collision can be bypassed. Therefore the
scene spec validates geometry before rendering:

- wall footprint clearance
- pairwise animal clearance
- same-frame animated-vs-animated separation
- fallback local trajectories when a full-room trajectory cannot be safely
  placed

The hand-authored two-dog demo uses `tools/gpurir_scenes/scene_two_dogs.py`.
It has an explicit L-shaped husky path plus an idle golden retriever and
stricter checks for the apartment corridor.

## 7. Rendering

Rendering is handled by `tools/gpurir_scenes/run_render_pass.py`.

It supports two visual rooms:

1. `apartment`
   - Uses SPEAR's native `apartment_0000` map.
   - SceneSpec coordinates are mapped to an apartment-local mic anchor.
   - Camera view 0 faces the apartment window direction.

2. `shoebox`
   - Builds a GPURIR-aligned synthetic room with matching dimensions.
   - Uses apartment material instances for floor and walls.
   - Shoebox frames are horizontally flipped after render so visual right
     matches audio +X.

Each room produces:

- `view0.mp4`
- `view1.mp4`
- `view2.mp4`
- `view3.mp4`

Important: these raw `view*.mp4` files are video-only. They are expected to be
silent.

## 8. Audio Generation

Audio generation is handled by `tools/gpurir_scenes/run_audio_pass.py`.

For each animal source:

1. Pick a source clip via `audio_registry.py`.
   - Prefer local OmniAudio filename keyword matches.
   - Use Stable Audio Open fallback when no local clip exists.

2. Convert source clip to mono, `16 kHz`, `5 s`.
   - Longer clips are trimmed.
   - Shorter clips are padded with silence.
   - Clips are not looped.

3. Simulate a 4-capsule tetrahedral microphone.
   - Mic center: `(2.6, 2.2, 1.2)` meters.
   - Capsule radius: `0.042 m`.

4. Use gpuRIR for room impulse responses.
   - Moving animals use `gpuRIR.simulateTrajectory`.
   - Static animals use one static RIR convolution.

5. Mix all sources and normalize peak to 0.9.

Output:

`audio.wav`

Shape for current 5 s scenes:

`80000 samples x 4 channels`

## 9. Muxing

Muxing is handled by `tools/gpurir_scenes/mux_audio_video.py`.

It performs:

1. Downmix 4-channel tetrahedral audio to stereo:
   - `FL = 0.5*c0 + 0.5*c2`
   - `FR = 0.5*c1 + 0.5*c3`

2. Mux the stereo WAV into every rendered MP4.

Output filenames:

- `view0_with_audio.mp4`
- `view1_with_audio.mp4`
- `view2_with_audio.mp4`
- `view3_with_audio.mp4`

For playback or qualitative review, use the `_with_audio.mp4` files, not the
raw `view*.mp4` files.

## 10. End-To-End Scene Drivers

General seeded scene:

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python \
  tools/gpurir_scenes/run_scene.py \
  --seed <seed> \
  --out-root /data/jzy/code/SPEAR/tmp/gpurir_scenes_v1
```

This runs:

1. `compose_scene(seed)`
2. `run_audio_pass.py`
3. `run_render_pass.py --room apartment`
4. `run_render_pass.py --room shoebox`
5. `mux_audio_video.py`

Hand-authored two-dog demo:

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python \
  tools/gpurir_scenes/scene_two_dogs.py \
  --out-root /data/jzy/code/SPEAR/tmp/gpurir_scenes_v1
```

Current v19 apartment-only repair was rendered manually, so audio had to be
added afterward:

```bash
CUDA_VISIBLE_DEVICES=1 /data/jzy/miniconda3/envs/sao-env/bin/python - <<'PY'
import json
import numpy as np
import sys
sys.path.insert(0, '/data/jzy/code/SPEAR/tools')
from gpurir_scenes.run_audio_pass import run_audio_pass
from gpurir_scenes.scene_two_dogs import compose_two_dog_scene, _spec_to_json
spec = compose_two_dog_scene()
scene_dir = '/data/jzy/code/SPEAR/tmp/gpurir_scenes_v19/two_dogs'
with open(scene_dir + '/trajectory.json', 'w') as f:
    json.dump(_spec_to_json(spec), f, indent=2)
rng = np.random.default_rng(999 + 10000)
run_audio_pass(spec, scene_dir + '/audio.wav', rng)
PY

/data/jzy/miniconda3/envs/spear-env/bin/python \
  tools/gpurir_scenes/mux_audio_video.py \
  --scene-dir tmp/gpurir_scenes_v19/two_dogs
```

## 11. Output Layout

Typical output directory:

```text
tmp/gpurir_scenes_v*/scene_XX/
  trajectory.json
  audio.wav
  audio_stereo.wav
  apartment/
    view0.mp4
    view0_with_audio.mp4
    ...
  shoebox/
    view0.mp4
    view0_with_audio.mp4
    ...
```

Two-dog demo output:

```text
tmp/gpurir_scenes_v*/two_dogs/
  trajectory.json
  audio.wav
  audio_stereo.wav
  apartment/
    view0.mp4
    view0_with_audio.mp4
    ...
```

## 12. Current Known Limits

1. Raw render MP4s are silent by design.
   - Use `_with_audio.mp4` for playback.

2. Runtime physics collision is not active for the teleported actors.
   - The reliable solution is trajectory and footprint validation before
     rendering.

3. Only five tags currently have walking animation assets integrated.
   - Dogs use Dog rig.
   - Cats and chipmunk use Cat rig.
   - Large ungulates are static for now.

4. The apartment has real furniture and map geometry, but the current geometry
   validator only knows the abstract room footprint, not every furniture mesh.
   - Hand-authored scenes need additional corridor/path constraints when they
     use apartment-specific space.

5. Hunyuan meshes can contain low bridges between legs or tail/body contact.
   - Current robust transfer removes low limb bridges and regularizes regions,
     but prompt design remains the first line of defense.

## 13. Current Verification Snapshot

For the current v19 two-dog apartment render:

- Raw video-only file:
  - `tmp/gpurir_scenes_v19/two_dogs/apartment/view0.mp4`
  - stream check: h264 video only

- Muxed file:
  - `tmp/gpurir_scenes_v19/two_dogs/apartment/view0_with_audio.mp4`
  - stream check: h264 video + stereo AAC audio
  - volume check: mean volume around `-18.2 dB`, max around `-1.7 dB`

- Audio file:
  - `tmp/gpurir_scenes_v19/two_dogs/audio.wav`
  - generated shape: `80000 x 4`

