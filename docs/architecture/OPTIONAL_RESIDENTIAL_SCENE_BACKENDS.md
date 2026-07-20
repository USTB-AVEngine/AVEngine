# Optional residential scene backends

AVEngine does not need one renderer to be best for every room family. The
maintained policy is to choose the visual presentation backend per room while
keeping one episode authority. Habitat-native AVEngine owns Timeline state,
navigation, source centers, source programs, audio, Topdown, flags and
metadata. UE/SPEAR is optional and has the bounded role `comparison_visual`.

## Recommended routing

| Room family | Preferred visual presentation | Why | Current status |
| --- | --- | --- | --- |
| MP3D | Habitat-Sim | The reconstructed scan texture already contains captured color and illumination. Treating it as a clean PBR room and adding UE lights can double-light it and emphasize scan holes. | Habitat is primary; the corrected-color UE import remains a compatibility diagnostic. |
| ReplicaCAD | UE/SPEAR | The scene is assembled from PBR assets, so UE material evaluation, local point-light shadows and exposure are more coherent. | UE is the preferred review presentation; Habitat remains authoritative for navigation and episode state. |
| SPEAR `apartment_0000` | Native UE/SPEAR map | The authored UE map already contains its glass, exterior presentation, materials, lighting and post-process setup. | Native UE is the preferred realism presentation; the Habitat package remains the protocol/source-logic reference. |
| InteriorAgent/Kujiale | Optional UE/SPEAR external-USD adapter | It supplies structured USD scenes, MDL materials, lights and `rooms.json`. The adapter converts the useful MDL inputs to portable USD PreviewSurface without copying the source scene. | The real `kujiale_0020` living-room episode and one kitchen-zone/open-plan episode pass. A full-home material adapter passes; bathroom visual review fails and bedroom/balcony dynamic review remains `not_run`. |
| InteriorNet | Rendered evaluation data, not a runtime room backend | The public release is chiefly a rendered RGB-D/inertial benchmark. The site says rights to the furniture, layouts and scenes remain with Kujiale and directs asset requests to them. | Consider it for RGB/depth/SLAM visual evaluation only unless separate scene-asset permission is obtained. |
| 3D-FRONT + 3D-FUTURE | Structured-room adapter candidate | Its layouts, semantics and textured furniture are a good match for controllable synthetic rooms. The shared compiler can consume normalized room polygons and object bounds. | A human+Beagle UE canary passes on the official Toolbox five-object sample proxy. Full-house qualification still requires an authorized complete 3D-FRONT release. |

This table changes only visual presentation. It does not create parallel
Timeline, navigation or acoustic implementations.

## Lighting policy

- MP3D keeps its scan-baked appearance and does not receive a claimed
  reconstruction of the original Matterport lights.
- ReplicaCAD uses room-local dataset lights in UE. A separately named neutral
  route-center fill may be enabled for review, but it is always labelled as a
  generated visual aid and never as acoustic truth.
- Native SPEAR Apartment uses the native UE lighting stack for comparison
  pixels. An editable JSON file exposes an untouched `native` preset and an
  optional two-light `warm_indoor_fill` preset. The Habitat review may use its
  own bounded warm indoor profile without claiming UE-equivalent rendering.
- InteriorAgent episode profiles declare their review lights in JSON. Positions
  may follow fixture-light prims or explicit room-local fill locations, but UE
  photometric values are review settings rather than recovered physical
  calibration and do not alter the acoustic scene.
- The 3D-FRONT Toolbox sample proxy uses two explicitly generated soft review
  lights. They illuminate the moving actors and projected furniture geometry;
  they are neither dataset-authored lights nor acoustic truth.

## Shared human and dog episode contract

All passing M6.z source videos use the same AVEngine-owned contract:

- exactly 75 frames at 15 fps and 80,000 audio samples at 16 kHz;
- one reviewed Beagle and one Rocketbox adult human, both moving with route
  tangent-aligned heading and deterministic walk phase;
- simultaneous human speech and three dog-bark windows with independent source
  trajectories and binaural stems;
- source-center-only room/obstacle qualification, with rugs treated as
  walkable and elevated fixtures excluded from ground blockers; the Kujiale
  extractor uses a 5 cm low-height descendant-triangle footprint instead of a
  potentially over-large parent AABB;
- a camera/listener Topdown showing room polygon, object roles, both routes,
  visual HFOV and the 360-degree microphone; and
- no audio cutoff when a source leaves the camera view.

The pure planner is `src/avengine/optional_backends/residential_episode.py`.
`tools/m6z/build_residential_source_episode.py` generates Timeline, source,
audio and Topdown records. `tools/m6z/run_spear_residential_episode.py` replays
that plan in UE and is not allowed to replan either route.

## InteriorAgent adapter boundary

InteriorAgent scenes are user-downloaded external research data. The repository
contains only:

- a small JSON visual profile;
- a pure-Python plan compiler;
- an optional Pixar USD material adapter;
- a UE editor map creator that stores one external USD reference; and
- SPEAR four-view and shared human+Beagle episode runners with human-readable
  evidence JSON.

The generated adapter layer references selected source scopes and texture
files. It neither embeds nor redistributes the downloaded dataset. The original
living-room proof adapts 407 materials, including 132 textured and 9 glass
materials, and repairs the UE-facing material-binding/primvar metadata for
1,470 selected meshes. The full-home derivative covers all declared room and
shell scopes with 2,053 meshes, 553 adapted materials, 170 textured materials
and 13 glass materials. Materials without useful authored surface inputs remain
explicitly unadapted. Production use should continue to split or selectively
load stages by room/object rather than assuming every semantic room has a
usable camera, route and collision interpretation.

The follow-on room review deliberately keeps mixed results. The kitchen-zone
episode passes, but the bathroom profile is withheld because manual inspection
found actor/fixture intersection even though the center-only grid gate passed.
Bedroom and balcony candidate views remain `not_run` for dynamic admission.
This manual visual gate does not silently expand the requested placement rule
into full-body collision; it only prevents an obviously broken video from being
called usable.

The official InteriorAgent page describes USD/USDA scenes, MDL materials,
lighting and room polygons. Its Terms of Use restrict the data to
non-commercial research and education and prohibit redistribution of the
downloaded data:

- <https://huggingface.co/datasets/spatialverse/InteriorAgent>
- <https://kloudsim-usa-cos.kujiale.com/InteriorAgent/InteriorAgent_Terms_of_Use.pdf>

## InteriorNet and 3D-FRONT boundary

InteriorNet is useful evidence for rendered visual/depth tasks, but its public
site does not grant AVEngine a general redistributable 3D-room source. It
explicitly reserves Kujiale's rights to the furniture models, layouts and
scenes and asks researchers to contact the listed owner for asset access:

- <https://interiornet.org/>

3D-FRONT is a stronger future runtime candidate because the release is a
structured room dataset paired with 3D-FUTURE furniture. Its license is for
scientific research only, is revocable/non-transferable, prohibits
commercialization and prohibits making the original dataset available to third
parties. Therefore AVEngine may ship an adapter, but must keep the downloaded
data outside Git and require each researcher to obtain it under the official
agreement:

- <https://gw.alicdn.com/bao/uploaded/TB1ZJUfK.z1gK0jSZLeXXb9kVXa.pdf?file=TB1ZJUfK.z1gK0jSZLeXXb9kVXa.pdf>
- <https://github.com/3D-FRONT-FUTURE/3D-FUTURE-ToolBox>

The local machine currently has only the official Toolbox sample: five posed
3D-FUTURE furniture meshes and one reference rendering, not a complete house
JSON, shell or texture library. The retained canary projectively maps that
official image onto the five meshes, adds a clearly labelled review shell,
imports the derived GLB into persistent UE mesh/material/texture assets and
replays the same human+Beagle episode. It is reported as
`official Toolbox five-object sample proxy`, never as a complete 3D-FRONT
house. Downloading or qualifying the full release remains a separate,
authorized user action.

Do not obtain either dataset from an unofficial mirror merely to bypass its
access agreement.

## What SPEAR itself currently includes

The maintained SPEAR checkout can control arbitrary compatible UE projects,
but its default cooked-map list is not a catalogue of residential datasets.
It contains:

- one authored residential map: `apartment_0000`;
- two SPEAR debug maps: `debug_0000` and `debug_0001`; and
- UE example/template maps: `Advanced_Lighting`, `Minimal_Default`,
  `StarterMap`, `ThirdPersonMap`, `VehicleExampleMap` and
  `VehicleOffroadExampleMap`.

Thus, Apartment is the only bundled residential room in the current checkout.
Additional Kujiale rooms arrive through the external InteriorAgent adapter, not
through an assumption that hundreds of rooms are already packaged with SPEAR.
Modern SPEAR is a general UE controller, so this same optional-backend pattern
can later target another authorized UE project:

- <https://github.com/spear-sim/spear>

## Reproducibility and maintenance

Code and environment versions are the reproducibility controls. Local runs
write one readable evidence JSON beside ignored media. Lighting is maintained
through small JSON profiles, not runtime locks. This work introduces no asset
leaf hashes, release-manifest locks or symlink policy. External dataset
locations are supplied at runtime and are never committed.
