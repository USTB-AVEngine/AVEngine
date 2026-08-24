# Kujiale target-agnostic multi-room CPU camera catalogs (2026-08-24)

## Scope

This bounded slice applies the shared target-agnostic camera-placement iterator
to current `kujiale_0020_full_home_v1` room instances. It takes each room
instance polygon only as a geometric region boundary, rasterizes the retained
descendant-mesh ground footprints, and crosses every sampled CPU free-space
camera center with all declared camera heights and yaw
`0,30,...,330` degrees.

Room type and room scope are retained as downstream annotations and provenance.
They do not choose a position, yaw, source path or target. The generic builder
therefore behaves identically for a previously unseen room-type label.

The catalog is complete on its declared 0.25 m camera-position sampling lattice:
there is no maximum-position truncation, diversity selection, or semantic,
source/path, continuous-60-cm, side, crossing/exchange, visibility, occlusion,
composition or difficulty filter. Those facts remain downstream joins or
descriptors.

## Current-scene binding

The existing ordinary cross-input validation now lives in
`avengine.optional_backends.kujiale_scene_binding`. Both the retained viewpoint
prefilter and this catalog builder use the same implementation. It compares
dataset, scene, current UE map, current source-stage filename and structural
room scope before candidate generation. An old `kujiale_0020_live_v2` source
profile still fails before any output is published.

This is not a new hash, frozen contract, baseline, release or admission gate.
It prevents one concrete cross-wiring failure among independently loaded JSON
files, which Git identity, versions, primary keys, transactions, uniqueness,
types and tests on each file alone cannot prevent.

## Authority boundary

The current authority is only the room polygon plus the retained 5 cm
descendant-mesh ground-footprint CPU raster. It is not a native navmesh and it
does not prove a full camera/actor body can occupy the point. A catalog entry
does not certify UE/SPEAR pixels, camera/listener readback, visibility, M3,
RLR, audio or a mini-pilot.

All declared ground-blocker footprints are excluded from camera floor
positions even if the furniture top is below the camera eye. This is a floor
placement fact, not an image-quality or difficulty condition. Height variants
then change the camera center above the same free floor position.

## Code and tests

- `src/avengine/optional_backends/kujiale_scene_binding.py`
- `tools/m6z/build_kujiale_camera_placement_catalogs.py`
- `tools/m6x/compile_kujiale_feasibility_bank.py`
- `tests/unit/test_kujiale_camera_placement_catalogs.py`

Focused tests cover complete position/height/yaw crossing, exact twelve-yaw
enumeration, low-furniture footprint exclusion, unknown room labels, null
semantic targets, pending downstream joins, profile mismatch rejection,
multi-room identity uniqueness, fresh atomic publication and no-clobber.

## Fresh retained-data evidence

The implementation commit is
`3c6078cb0c77af67a0d180e764a7926ee51db7d6`. Five metadata files were
extracted fresh from the current derived full-home stage and its retained
`rooms.json` into:

`/data/jzy/tmp/kujiale_full_home_room_metadata_20260824T065213Z_r1`

The clean-code multi-room output is:

`/data/jzy/tmp/kujiale_full_home_camera_catalogs_3c6078c_r1`

It uses a 5 cm raster, a complete 0.25 m camera-position sampling lattice,
camera heights 1.05/1.50/1.85 m above the declared floor, zero additional
center-clearance erosion, and exactly twelve 30-degree yaws. Observed counts
are:

| Room instance | Free-space raster pixels | Sampled floor positions | Position x height x yaw placements |
| --- | ---: | ---: | ---: |
| `livingroom_491` | 18,702 | 761 | 27,396 |
| `bedroom_579` | 3,937 | 153 | 5,508 |
| `balcony_628` | 3,370 | 153 | 5,508 |
| `kitchen_736` | 835 | 33 | 1,188 |
| `bathroom_781525` | 1,679 | 63 | 2,268 |
| **Total** | **28,523** | **1,163** | **41,868** |

Every room satisfies `placements = sampled positions x 3 x 12`. The build
completed in 2.75 s wall time with maximum RSS 82,456 KiB; its log is
`/data/jzy/tmp/kujiale_full_home_camera_catalogs_3c6078c_r1_build.log`.

The independent record-level verifier checked each sampled pixel against its
stored free-space raster, recomputed pixel-center XYZ and height, required
every position/height/yaw tuple to be unique, checked all 41,868 downstream
status fields, and reconciled per-room and aggregate counts. It reports
`status=pass` at:

`/data/jzy/tmp/kujiale_full_home_camera_catalogs_3c6078c_r1_verification_r3.log`

A fresh actual negative run used the retained old `kujiale_0020_live_v2`
source profile. It exited before publication with `source profile and visual
profile map_path differ`; no output directory exists. The log is:

`/data/jzy/tmp/kujiale_full_home_camera_catalogs_old_live_v2_rejection_3c6078c_r1.log`

Focused and adjacent verification records `85 passed` at
`/data/jzy/tmp/kujiale_room_catalogs_3c6078c_unit_r1.log`; Ruff records
`All checks passed` at
`/data/jzy/tmp/kujiale_room_catalogs_3c6078c_ruff_r1.log`.

Native package/launch endpoint, M3, RLR, UE/SPEAR first-person/Topdown/readback,
4--8 certified viewpoint selection and four-point mini-pilots remain pending
or not run for every room.

No Studio, UE, GPU, Habitat or RLR process is started by this slice. All
outputs remain `research_only=true`, `episode_counted=false`, formal dataset
count zero and `qualification_claim=false`.
