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

The implementation commit, focused CPU test log, fresh artifact and independent
per-record verifier will be recorded here after they exist. Until then, this
section makes no completion claim.
