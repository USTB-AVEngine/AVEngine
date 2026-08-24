# Kujiale current full-home profile binding (20260824)

## Scope and concrete failure

This is a bounded CPU-only wiring correction. The retained living-room metadata
and its eight-view prefilter came from
`kujiale_0020_full_home_preview_surface.usda`, while the source episode profile
still selected the older UE map `kujiale_0020_live_v2`. The viewpoint builder
previously compared only `scene_id`, so those individually valid JSON inputs
could be cross-wired and still produce a successful planning receipt. Git
identity, schema versions, primary keys, transactions, uniqueness, types and
the old unit tests do not compare identities across three independently loaded
files, so none of them prevented this concrete failure.

The checked-in living source profile now selects
`/Game/AVEngine/Optional/Kujiale/kujiale_0020_full_home_v1`. The current
full-home visual profile names the corresponding retained stage file. The
builder requires that visual profile and compares:

- dataset and scene identity across metadata, source profile and visual profile;
- UE map identity across source and visual profiles;
- the metadata source-stage filename against the visual profile; and
- the structural room scope against the scopes present in that visual profile.

The accepted identity is recorded in `viewpoint_bank.json` and `delivery.json`.
This is ordinary input validation and provenance, not a hash, frozen baseline,
release gate or claim that an external asset can never be revised.

## Claim boundary

The builder remains a 2D room-polygon plus descendant-footprint CPU prefilter.
It does not create or inspect a Habitat navmesh and does not certify native
navigation, UE/SPEAR pixels, full-body collision, visibility, M3 or RLR. Room
semantics do not generate camera candidates; the checked scope is structural
stage membership, while room labels remain descriptive metadata.

This slice does not promote continuous 60 cm separation, image-side margin,
visibility, crossing or swapping to geometric legality. Those retained values
belong to later difficulty analysis. The pre-existing KJ1 corridor/profile
selection is not claimed as the owner-approved generic 30-degree camera
enumerator and remains to be replaced/integrated in that separate slice.

## Fresh evidence

A fresh/no-clobber eight-view build uses:

- metadata: `/data/jzy/tmp/kujiale_living_metadata_20260824T0110Z.json`;
- source profile:
  `examples/m6z/interioragent_kujiale_0020_source_episode.json`;
- visual profile:
  `examples/m6z/interioragent_kujiale_0020_full_home_visual_profile.json`; and
- output:
  `/data/jzy/tmp/kujiale_living_viewpoint_prefilter_current_profile_20260824T054651Z`.

The build used a 30-degree yaw step and completed with 8 viewpoints. Its
`delivery.json` records the current full-home map and stage,
`research_only=true`, `episode_counted=false`, formal dataset count 0, and all
native/mini-pilot statuses pending.

A separate fresh negative run used the untouched historical `live_v2` source
profile from its original worktree:

`/data/jzy/tmp/kujiale_current_profile_old_source_rejection_20260824T054651Z`

It stopped before corridor/viewpoint generation and wrote only `FAILED.json`
with `source profile and visual profile map_path differ`.

Focused/adjacent verification is 36 passed tests in 12.37 seconds. Ruff,
`py_compile`, JSON parsing and `git diff --check` also pass.

No UE, SPEAR, Studio, GPU, native pixel, M3, RLR or mini-pilot job ran in this
slice. Outputs remain `research_only`, `episode_counted=false`, formal dataset
count 0, with UE/SPEAR and native-pixel certification pending.

## Remaining work

The external current full-home package, launch endpoint and current runtime map
still need live UE/SPEAR verification. Each target room still needs the generic
camera/source enumerator, native first-person and Topdown review, readback and
four-point mini-pilot. M3 geometry/acoustics and RLR remain `not_run` here.
