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
| MP3D `17DRP5sb8fy` | `pass` | Real 270-frame/18-second UE execution; imported PBR scan, root/yaw and animation readback, Beagle floor/upright gate, fixed-exposure QA and media readback pass | This is the retained M5.1 compatibility route, not the 75-frame Timeline-v2 clock; each root moves only 1.1 m/18 s, so it is not a normal-speed result; weak review lights improve actor-shadow readability but do not reconstruct Matterport capture lighting |
| ReplicaCAD `apt_0` | `pass` | Real editor import/reload and 270-frame UE execution; 120 logical instances become 171 tagged runtime mesh actors, all five positive dataset point lights cast shadows, and root/yaw, animation, Beagle floor/upright, source-center, exposure and media gates pass | This replays the retained M5.1 compatibility route, not Timeline v2; both roots travel only 1.2 m in 18 seconds, so it proves scene/constraint compatibility but not a normal-speed route |

The Apartment pass covers exactly S0 routing sanity, S3 moving source and S4
overlapping sources because those are the three requested visual comparisons.
The UE map is the native SPEAR package
`/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000`; Habitat's temporary
window/exterior proxy and debug source markers are not inserted into UE.

MP3D textures already contain illumination captured by the scan. The comparison
therefore disables eye adaptation, applies one fixed output gain and uses only a
weak shadow-casting key plus skylight. Passing exposure QA means the result is
neither black nor clipped; it is not evidence of physically recovered lights.
Like the retained ReplicaCAD review below, its 1.1 m/18 s actor route is useful
for visual compatibility only and does not close the normal-speed requirement.

ReplicaCAD uses the imported PBR scene and the dataset lighting description.
The dataset declares seven signed lights: UE instantiates the five positive
point lights with shadows, records the two negative fills that UE cannot express
as point lights, and adds no review light. The import/reload evidence closes 101
source GLBs, 127 StaticMeshes and 120 logical instances; runtime readback closes
all 171 tagged mesh actors. Its intentionally retained slow route remains useful
for comparing the old Habitat delivery with UE pixels, but a future normal-speed
route must first be authored and requalified in Habitat together with matching
audio, Topdown and source metadata. UE is not allowed to invent that route.

## Local evidence and review

Generated results remain under ignored `tmp/` directories and are not Git
content. The current evidence roots are:

- `tmp/m6y/spear_apartment_native_suite_20260720_04/evidence.json`
- `tmp/m6y/spear_mp3d_full_20260720_01/evidence.json`
- `tmp/m6y/spear_replicacad_full_20260720_01/evidence.json`

Build one local review page with:

```bash
python tools/m6y/build_review_index.py \
  --apartment-suite tmp/m6y/spear_apartment_native_suite_20260720_04 \
  --mp3d-run tmp/m6y/spear_mp3d_full_20260720_01 \
  --replicacad-run tmp/m6y/spear_replicacad_full_20260720_01 \
  --output tmp/m6y/REVIEW_INDEX.html
```

The page embeds the Apartment clean/Topdown videos and the MP3D clean,
Topdown and three-panel comparison, plus ReplicaCAD clean, Topdown and
three-panel comparison media, and links each evidence JSON. The builder reports
the status recorded by each real run rather than promoting it by convention.

## Closeout boundary

M6.y is complete when the optional-backend regression suite passes and this
work is merged into `feature/habitat-native-avengine`. Its three room results
remain bounded comparison media rather than room/material qualification or
dataset admission. New normal-speed MP3D and ReplicaCAD episodes are separate
follow-up work because each requires one coherent Habitat-native route, audio,
Topdown and metadata regeneration rather than a visual-backend-only edit.
