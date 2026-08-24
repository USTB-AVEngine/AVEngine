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

The five current room-instance metadata extractions and clean-code catalog run
will be recorded here after execution. Until then, native package, launch
endpoint, M3, RLR, UE/SPEAR first-person/Topdown/readback and four-point
mini-pilot statuses remain pending or not run.

No Studio, UE, GPU, Habitat or RLR process is started by this slice. All
outputs remain `research_only=true`, `episode_counted=false`, formal dataset
count zero and `qualification_claim=false`.
