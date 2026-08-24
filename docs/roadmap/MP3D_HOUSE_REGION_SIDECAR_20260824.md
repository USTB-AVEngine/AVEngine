# MP3D static `.house` region sidecar (2026-08-24)

## Scope and authority

This slice is a post-hoc annotation over the complete target-agnostic MP3D
camera catalog. It parses the source dataset's `H`, `L`, `R`, `S`, and `V`
records, uses only `F` surface polygons, and applies the same
`mp3d_z_up_y_front_to_habitat` transform registered by the M3 compiler:
`[x,y,z] -> [x,z,-y]`. Region names and category codes are labels after camera
enumeration; they do not generate camera candidates.

The parser distinguishes the two MP3D `P` record layouts. A 13-field `P`
record is a panorama and a 15-field `P` record is a portal. Those counts are
checked independently against the `H` record, so the 48 panorama records in
`17DRP5sb8fy.house` are not misreported as 48 door portals when the declared
portal count is zero. Habitat runtime `SemanticRegion.contains` is not used as
authority.

For every canonical native floor sample, the binder checks every catalog
record's `floor_sample_index` and exact `floor_position_m`, plus the complete
three-height x twelve-yaw cross product. It then publishes a separate sidecar;
the source catalog is never rewritten. Unique, exact-boundary, unknown and
overlapping/ambiguous memberships are all retained. The distance to the
nearest polygon boundary and the `<= 0.05 m` descriptor are recorded, but are
not candidate filters or legality gates. Unknown category codes are preserved
with a null category name rather than rejected or coerced to a known class.

The classification uses Habitat XZ polygon containment and the source region's
transformed Y bounding interval. It does not force an unknown point into the
nearest room. It makes no native-pixel, visibility, source-route, audio/RIR,
viewpoint-certification, difficulty, ranking or formal-Episode claim.

## Failure cases covered

Ordinary validation fails before publication when a floor index and coordinate
drift apart, a per-floor height/yaw record is missing or duplicated, the input
artifacts no longer describe the same unfiltered native floor authority, a
house count/reference is inconsistent, a coordinate is non-finite, or a floor
polygon is degenerate/self-intersecting. Overlapping valid polygons are not an
error: the sample is retained as `ambiguous` with every member ID. Unknown
samples likewise remain in the sidecar.

This is ordinary input and geometry validation. It adds no hash, frozen
contract, baseline or permanent gate. Existing research boundaries remain
`research_only=true`, `episode_counted=false`, formal release `not_run`,
candidate filtering `none`, and candidate-legality effect `none`.

## Clean-code verification and current scene evidence

The implementation commit is
`93dc679a5f9fe24e0de19c2daadd5cd7e78af1b5`. From that clean commit, Ruff,
Ruff format checking, `git diff --check`, and 69 related CPU tests passed. The
tests cover this binder together with the existing MP3D floor exporter, camera
catalog, full-catalog probe selector, viewpoint selector, and M3 compiler. The
fresh log is:

`/data/jzy/tmp/mp3d_regions_93dc679_unit_r1.log`

The clean-code current-scene run consumed the complete 632-sample native floor
export and the unfiltered 22,752-placement catalog. Its fresh/no-clobber output
is:

`/data/jzy/tmp/mp3d_house_regions_93dc679_r1`

`camera_region_sidecar.json` is 32,282,250 bytes. It contains 632 floor records
and 22,752 placement records, exactly 36 placements per floor point. The
`.house` parser reports 10 region instances, 48 panorama records, zero portal
records, 10 floor surfaces, and 57 vertices. The membership counts are:

- floor: 615 unique, zero exact-boundary, 17 unknown, zero ambiguous;
- placement: 22,140 unique, zero exact-boundary, 612 unknown, zero ambiguous;
- 20 floor points are within 5 cm of the nearest polygon boundary; all remain
  present and that descriptor changes no legal status.

In numeric region-index order, the retained floor/placement counts are:

| Region | Category | Floor points | Placements |
|---:|---|---:|---:|
| 0 | bedroom | 108 | 3,888 |
| 1 | bathroom | 65 | 2,340 |
| 2 | toilet | 7 | 252 |
| 3 | bathroom | 26 | 936 |
| 4 | tv | 73 | 2,628 |
| 5 | dining room | 61 | 2,196 |
| 6 | entryway/foyer/lobby | 85 | 3,060 |
| 7 | living room | 133 | 4,788 |
| 8 | familyroom/lounge | 47 | 1,692 |
| 9 | kitchen | 10 | 360 |

An independent standard-library verifier reparsed the raw `.house` records,
reimplemented the source-to-Habitat projection and point-in-polygon operation,
and checked every floor record and every placement record against both input
artifacts. It reproduced all counts, all member IDs, exact index/coordinate
joins, all 36 placement IDs per floor sample, every retained pose/yaw, the 20
near-boundary descriptors, and the research/no-filter boundary. Its log is:

`/data/jzy/tmp/mp3d_house_regions_93dc679_r1_verification.log`

The 17 unknown floor points and their 612 placements remain explicit unknowns;
no nearest-region coercion was used. This evidence completes the static
additive region sidecar for the current scene. It does not certify any native
viewpoint or complete the downstream 4--8-view-per-region route, dynamic
capture, actor-path generation, RLR/SoundSpaces audio, or four-point
mini-pilots.
