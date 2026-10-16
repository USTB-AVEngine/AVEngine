# M6.z External residential scene adapters

M6.z is a bounded optional-backend workstream. It answers whether authorized
external residential scenes can supply Apartment-like UE/SPEAR pixels while
Habitat-native AVEngine remains the only Timeline, navigation, source, audio,
Topdown and metadata authority.

## Result

Status: `pass` for the two original retained scopes:

1. the real InteriorAgent/Kujiale `kujiale_0020` living room, including the
   original four-view material proof and a human+Beagle source episode; and
2. a clearly labelled 3D-FRONT official Toolbox five-object sample proxy with
   the same human+Beagle source episode.

The follow-on `kujiale_0020` full-home exploration has mixed, explicitly
reported outcomes:

- the reference-only full-home USD adapter is `pass` with 2,053 meshes, 553
  adapted materials, 170 textured materials and 13 glass materials;
- a kitchen-zone/open-plan human+Beagle episode is a bounded `pass` after
  runtime/media checks and manual visual review;
- the bathroom run is manual visual `fail`: its automatic center-only gate
  passed, but the rendered human visibly intersects the toilet/wall area, so
  the profile is not shipped as a usable example; and
- bedroom and balcony dynamic episodes are `not_run` because the tested views
  are near the scene shell, occluded or black.

This mixed result is intentional. A programmatic source-center pass does not
override an obvious failure in the final visual review, and M6.z does not claim
that every semantic room inside one downloaded home is immediately usable.

Each source episode has exactly 75 frames at 15 fps and 80,000 audio samples at
16 kHz. The human and dog move simultaneously on independently qualified
routes, face their route tangents and replay deterministic walk phases. Human
speech overlaps three dog-bark events. The listening video contains a two-
channel binaural mix and the AVEngine Topdown; audio remains 360 degrees and is
not cut off by the camera HFOV.

The Kujiale scene loads through one external USD stage. The material adapter
preserves useful wood, fabric, stone, glass and fixture detail. Review-light
locations and photometric values are explicit JSON per episode. They are tuned
visual settings, not recovered physical calibration. The kitchen result uses
two 450-lumen soft fills; its remaining wall hotspot follows the imported
fixture/emissive appearance and is recorded rather than hidden.

The local 3D-FRONT data is not a full house. The official Toolbox sample ships
five posed 3D-FUTURE meshes and one reference rendering but no complete house
JSON, room shell or texture library. The canary projectively maps the official
image onto those meshes, adds a generated review shell and imports the derived
GLB into persistent UE assets. This repaired the black result caused by the
earlier transient USD material cache. The result must always be called an
`official Toolbox five-object sample proxy`.

## Checks

| Check | Kujiale | 3D-FRONT sample proxy |
| --- | ---: | ---: |
| Runtime room representation | 1 external USD stage | 9 persistent StaticMesh actors |
| Persistent imported material assets | external USD PreviewSurface | 4 |
| Persistent imported texture assets | external references | 3 |
| Episode frames | 75 | 75 |
| Frame rate | 15 fps | 15 fps |
| Binaural channels/rate | 2 / 16 kHz | 2 / 16 kHz |
| Source-center route gate | pass | pass |
| Actor root readback | pass | pass |
| Walk-phase readback | pass | pass |
| Beagle floor/contact frame gate | pass | pass |

The original Kujiale four-view material preparation additionally records 1,470
selected meshes, 407 adapted MDL materials, 132 textured materials, 9 explicit
glass materials and four captured views. Materials without usable authored
surface inputs remain explicitly unadapted.

## Implementation

- `src/avengine/optional_backends/interioragent_kujiale.py`: dependency-free
  Kujiale profile/material planning and authority/license boundary;
- `src/avengine/optional_backends/residential_episode.py`: shared exact
  human+Beagle Timeline, constant-arc-length polyline route, tangent heading,
  source and center-gate compiler;
- `examples/m6z/`: editable Kujiale and 3D-FRONT sample profiles;
- `tools/m6z/prepare_interioragent_kujiale_adapter.py`: MDL-to-USD
  PreviewSurface adapter;
- `tools/m6z/prepare_3d_front_toolbox_sample_proxy.py`: explicit bounded sample
  proxy builder, with embedded-texture GLB output for persistent UE import;
- `tools/m6z/create_spear_kujiale_map_editor.py` and
  `tools/m6z/import_spear_3d_front_sample_editor.py`: isolated UE map creation;
- `tools/m6z/build_residential_source_episode.py`: Timeline, binaural and
  Topdown generation; and
- `tools/m6z/run_spear_residential_episode.py`: UE replay, mux and readback.

`tools/m6z/extract_interioragent_scene_metadata.py` now derives a 5 cm
low-height triangle footprint for ground blockers. This avoids treating a rug,
tabletop or separated/L-shaped furniture group as one solid parent AABB. It
still implements the project-owner-selected source-center-only rule; it is not
a full human or animal body-volume collision test.

The fast tests do not import UE, SPEAR, USD or Blender.

## Local evidence

Generated data stays under ignored `tmp/` directories. Current review videos:

- Kujiale Topdown+binaural:
  `tmp/m6z/kujiale_0020_human_dog_spear_20260720_01/ue_topdown_binaural.mp4`;
- Kujiale clean+binaural:
  `tmp/m6z/kujiale_0020_human_dog_spear_20260720_01/ue_clean_binaural.mp4`;
- Kujiale kitchen-zone Topdown+binaural:
  `tmp/m6z/spear_kujiale_0020_kitchen_20260720_05/ue_topdown_binaural.mp4`;
- rejected bathroom diagnostic:
  `tmp/m6z/spear_kujiale_0020_bathroom_20260720_01/ue_topdown_binaural.mp4`;
- 3D-FRONT sample Topdown+binaural:
  `tmp/m6z/3d_front_official_toolbox_human_dog_spear_20260720_04/ue_topdown_binaural.mp4`;
- 3D-FRONT sample clean+binaural:
  `tmp/m6z/3d_front_official_toolbox_human_dog_spear_20260720_04/ue_clean_binaural.mp4`.

The matching `evidence.json` files sit beside each video. No dataset bytes,
derived GLB/USD, UE map, media or evidence output is committed.

For local review, open
`tmp/m6z/kujiale_rooms_and_apartment_lighting_review_20260720_01/REVIEW_INDEX.html`.
It places the accepted living/kitchen media beside the rejected bathroom
diagnostic and the Apartment native/warm-light A/B.

## Claim and authority boundary

- Both room backends are `comparison_visual`; neither may replan a route.
- Placement checks the source center only, per the project-owner decision. It
  does not claim whole-body collision clearance.
- Binaural review audio uses the explicitly declared generic directional
  acoustic proxy. It is not exact Kujiale/3D-FRONT material or RT60 truth.
- Generated UE review lights are visual aids, not dataset-authored acoustic
  sources.
- InteriorAgent and 3D-FRONT/3D-FUTURE data remain external research data and
  are not redistributed.
- A complete 3D-FRONT house adapter and qualification remain pending until the
  user obtains the full official data under its agreement.

For license links, InteriorNet scope and the current SPEAR room list, see
[OPTIONAL_RESIDENTIAL_SCENE_BACKENDS.md](../architecture/OPTIONAL_RESIDENTIAL_SCENE_BACKENDS.md).
