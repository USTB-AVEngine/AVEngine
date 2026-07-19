# M6.y Optional SPEAR/UE Comparison Visuals

M6.y is an optional visual-comparison workstream. It answers a narrow question:
can the old SPEAR/Unreal presentation layer render the same actor states and
rooms while the current Habitat-native AVEngine keeps control of the episode?
It is not a second AVEngine runtime and it does not change dataset admission.

## Authority boundary

- Habitat-native AVEngine remains authoritative for Timeline or retained-route
  state, navigation, source-center placement, source identity and programs,
  binaural audio, Topdown, flags and metadata.
- SPEAR/UE has the role `comparison_visual`: it may load native UE assets and
  render pixels, but it may not replan trajectories, move source centers or
  recompute the authoritative audio.
- Actor placement is checked at the authored source/actor root center. M6.y does
  not add a full-body collision requirement.
- No new release manifest, symlink policy or leaf-hash checklist is introduced.
  Code and environment versions remain the reproducibility controls; each local
  run keeps one human-readable evidence JSON beside its media.

## Current result

| Room/render | Status | What has been demonstrated | Claim boundary |
| --- | --- | --- | --- |
| Native SPEAR Apartment S0/S3/S4 | `pass` | Real UE execution at 1280x720, 15 fps and 75 frames per scenario; native Apartment map, materials, outdoor view and lighting; root/yaw, animation phase, Beagle floor/upright and media readback gates pass | UE supplies visual pixels only; the Habitat-native binaural stream and Topdown are reused unchanged |
| Habitat Apartment S0--S5 | `pass` | Fresh 1280x720 capture and all eight variants use the `natural_v3` shallow neutral window-key/bounce profile with HBAO; trajectory, semantic, Topdown and audio invariants remain unchanged | This improves the fixed-camera Habitat review, but its exterior remains a direction-projected panel rather than UE glass, HDRIBackdrop, reflection capture or Lumen |
| MP3D `17DRP5sb8fy` | `pass` | Real 270-frame/18-second UE execution; 23/23 base-color bindings use fresh-reloaded sRGB texture views while 23/23 AO bindings retain linear views; exposure, aggregate color retention, root/yaw, animation and media gates pass | This is the retained M5.1 compatibility route, not the 75-frame Timeline-v2 clock; each root moves only 1.1 m/18 s, so it is not a normal-speed result; review lights do not reconstruct Matterport capture lighting, and remaining holes are scan geometry |
| ReplicaCAD `apt_0` | `pass` | Real editor import/reload and 270-frame UE execution; 120 logical instances become 171 tagged runtime mesh actors; room-local lights 0/1/2 stay active while exterior lights 3/4 are zero; an optional route-center mode adds and reads back exactly one neutral ceiling fill in both UE and Habitat | This replays the retained M5.1 compatibility route, not Timeline v2; both roots travel only 1.2 m in 18 seconds; the generated fill is explicitly visual-only, not dataset-authored or acoustic truth |

The Apartment pass covers exactly S0 routing sanity, S3 moving source and S4
overlapping sources because those are the three requested visual comparisons.
The UE map is the native SPEAR package
`/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000`; Habitat's temporary
window/exterior proxy and debug source markers are not inserted into UE.

MP3D textures already contain illumination captured by the scan. The original
glTF uses each of its 23 images for both base color and occlusion; the old UE
import treated all of them as linear and produced the observed white/grey
render. The corrected isolated-project import creates an sRGB base-color view
and preserves a separate linear AO view. A fresh editor reload verifies both
slots, and the 270-frame UE/Habitat mean-chroma ratio is `1.006`. Fixed exposure
and weak shadow-review lighting still do not recover the original Matterport
lights. Its 1.1 m/18 s actor route is useful for compatibility only.

ReplicaCAD uses the imported PBR scene and dataset light locations. The two
strongest positive lights (IDs 3/4) lie outside the imported stage-shell AABB;
because that shell has an open roof and holes, they dominated the previous
render like an exterior key light. The `room_local_review` profile sets only
those two to zero, retains indoor lights 0/1/2 at an explicit UE scale of 2.0,
and moves/adds no lights. A real Habitat run with the same selection was darker
and less natural even at its independent scale of 4.0, so it remains a research
comparison; the maintained Habitat view stays `no_lights + HBAO`. UE lumens and
Habitat light colors are not treated as interchangeable physical units. The two
plain source-light profiles and optional `route_center_fill_review` live in
`examples/m6y/replicacad_apt0_lighting_profiles.json` for lab maintenance.
The optional mode averages the human and Beagle route endpoints, clamps that
horizontal point inside the stage-shell bounds and places one light below the
stage ceiling. UE and Habitat keep separate intensity values. The added light
is counted and read back separately from the five positive dataset lights.

## Recommended visual backend by room

- MP3D: Habitat-Sim is the primary presentation. Its scan textures already
  contain captured illumination, so adding UE lights can create double-lighting
  and make scan holes more conspicuous. The UE import remains a compatibility
  diagnostic, not the preferred final render.
- ReplicaCAD: UE/SPEAR is the primary presentation because its PBR materials,
  point-light shadows and exposure are more coherent. Habitat remains useful
  for navigation and a research visual comparison.
- SPEAR Apartment: the native UE map is the primary realism presentation.
  Habitat remains the authoritative protocol/sensor/source-logic render and can
  use one warm route-area point fill in addition to its directional review
  lights.

This routing changes only the visual presentation backend. Habitat-native
Timeline/route state, source centers, binaural audio, Topdown, flags and
metadata remain the single episode authority.

## Local evidence and review

Generated results remain under ignored `tmp/` directories and are not Git
content. The current evidence roots are:

- `tmp/m6y/spear_apartment_native_suite_20260720_04/evidence.json`
- `tmp/m6x/fixed_apartment_natural_lighting_20260720_01/bundle_manifest.json`
- `tmp/m6y/spear_mp3d_full_20260720_02/evidence.json`
- `tmp/m6y/spear_replicacad_room_local_full_20260720_01/evidence.json`
- `tmp/m6y/spear_replicacad_route_fill_full_20260720_01/evidence.json`
- `tmp/m6y/habitat_replicacad_route_fill_20260720_02/evidence.json`
- `tmp/m6x/fixed_apartment_route_fill_trial_20260720_01/bundle_manifest.json`

Build one local review page with:

```bash
python tools/m6y/build_review_index.py \
  --habitat-apartment-bundle tmp/m6x/fixed_apartment_natural_lighting_20260720_01 \
  --apartment-suite tmp/m6y/spear_apartment_native_suite_20260720_04 \
  --mp3d-run tmp/m6y/spear_mp3d_full_20260720_02 \
  --replicacad-run tmp/m6y/spear_replicacad_room_local_full_20260720_01 \
  --output tmp/m6y/LIGHTING_REVIEW_INDEX.html
```

The page embeds the Habitat Apartment natural-light S3 review, native UE
Apartment media, corrected-color MP3D, and room-local-lit ReplicaCAD. It links
the full S0--S5 Habitat page and each evidence document. The builder reports the
status recorded by each real run rather than promoting it by convention.

## Closeout boundary

M6.y is complete when the optional-backend regression suite passes and this
work is merged into `feature/habitat-native-avengine`. Its three room results
remain bounded comparison media rather than room/material qualification or
dataset admission. New normal-speed MP3D and ReplicaCAD episodes are separate
follow-up work because each requires one coherent Habitat-native route, audio,
Topdown and metadata regeneration rather than a visual-backend-only edit.
