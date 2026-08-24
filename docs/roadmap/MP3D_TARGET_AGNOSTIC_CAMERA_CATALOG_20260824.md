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
