# MP3D target-agnostic camera catalog (2026-08-24)

## Scope

This slice adds the second backend adapter for the shared camera-placement
architecture. Apartment consumes a room obstacle raster; MP3D can instead
consume the complete legal floor positions already exported from the declared
live Habitat PathFinder. Both paths enumerate camera placement before semantic
targets or source paths are joined.

`iter_declared_floor_camera_placements` crosses every canonical caller-declared
floor XYZ with low/base/high camera heights and the stable yaw set
`0,30,...,330` degrees. Exact duplicate floor XYZ values are removed. It knows
nothing about room names, semantic targets, actors, source paths, visibility,
difficulty or selection.

`tools/m5/build_mp3d_camera_placement_catalog.py` binds that iterator to one
validated MP3D room/M1 request and a matching native floor export. It publishes
every placement to a fresh/no-clobber ordinary JSON catalog with candidate
filtering `none`, semantic fields null, and downstream source/visibility/
difficulty states pending. It does not materialize thousands of duplicate full
M1 request documents and does not launch Habitat.

## Boundary

The floor export remains the native navigation authority; the CPU catalog does
not re-probe PathFinder. A placement is not a certified viewpoint. Source/path
joining, native dynamic frames, visible-pixel evidence, difficulty strata,
deduplicated 4--8-view selection, audio/RIR and mini-pilots remain downstream.

No Studio placement, manual coordinate authoring, semantic generation
dependency, hash, frozen contract, baseline or permanent gate is introduced.

## Verification before real run

The focused server log is:

`/data/jzy/tmp/mp3d_camera_catalog_unit_r1.log`

It records `25 passed` across the declared-floor iterator, target-agnostic
catalog, existing camera placement, MP3D viewpoints and native floor export.

Unit coverage proves stable enumeration under reordered/duplicated samples,
three height roles, exactly twelve 30-degree yaw values, unknown room names,
null semantic fields, no source/path filtering, fresh publication, no-clobber
and cross-room floor-authority rejection. Existing MP3D viewpoint tests also
prove that moving the shared height variants does not change their values.

A real native-floor catalog run remains pending until the implementation is
committed, so the first artifact can name clean code provenance.

## Clean-code real native-floor catalog

The implementation was committed as
`c9c2274bf6cde77a4151b9124140cd38862fd7e8`. The fresh output is:

`/data/jzy/tmp/mp3d_camera_placement_catalog_c9c2274_r1`

It consumes all 128 retained native PathFinder floor samples and emits exactly
4,608 placements: 128 floor points x three camera heights x twelve yaw values.
Heights above each floor point are low `1.2 m`, base `1.5 m` and high `1.8 m`;
yaw is exactly `0,30,...,330` degrees. The JSON artifact is 4,535,250 bytes.

Every placement has null semantic room/target fields, source/path join pending,
native dynamic visibility pending, difficulty not evaluated and candidate
filtering `none`. Studio was not used. No full M1 request is duplicated 4,608
times; the base request identity and ordinary pose values are retained for later
materialization.

The verifier is:

`/data/jzy/tmp/mp3d_camera_placement_catalog_c9c2274_r1_verification.log`

It reports `status=pass` for the exact count, unique placement IDs, three
heights, twelve yaws, no filtering, no Studio, null semantics and pending joins.

This completes the raw MP3D camera-position/yaw enumeration layer. It does not
yet join the retained emitter tracks, choose a native-probe subset, certify
4--8 viewpoints, render audio/RIR or run a mini-pilot.

## Expanded full scanned-floor domain

The 128-floor result above was complete only relative to the exporter's default
`maximum_sample_count=128`. The same native scan had actually examined 860
Topdown pixels and found 632 passing positions; therefore 128 was a bounded
first run, not the complete discrete scanned feasible domain.

After a fresh GPU/process preflight, current integration reran the same live
PathFinder export with `maximum_sample_count=1000`:

`/data/jzy/tmp/mp3d_floor_samples_all_8c4e7df_r1`

All 860 positions were examined: 632 pass and 228 fail, and all 632 passing
positions were retained. The target-agnostic CPU catalog was then regenerated:

`/data/jzy/tmp/mp3d_camera_placement_catalog_all_8c4e7df_r1`

It contains exactly 22,752 placements: 632 floor positions x three heights x
twelve 30-degree yaws. The JSON is 22,399,723 bytes. Candidate filtering remains
`none`; semantic fields are null and source/visibility/difficulty joins remain
pending.

The previous 4,608-placement artifact is retained as the bounded first run and
is not relabeled. All downstream native-probe selection must use the expanded
22,752 catalog unless a later fresh PathFinder export supersedes it.
