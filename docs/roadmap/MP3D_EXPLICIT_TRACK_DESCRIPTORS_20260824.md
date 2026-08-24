# MP3D explicit viewpoint-track descriptors (2026-08-24)

## Scope

`tools/m5/describe_mp3d_viewpoint_tracks.py` is a CPU-only consumer for a
retained MP3D viewpoint bank, its exact two-actor route explanation and one
passing native static-probe receipt per view. It does not launch Habitat and it
does not select or reject cameras.

For every viewpoint the tool:

- loads both caller-supplied `actor_skin_root_paths_m` arrays and requires
  exactly 75 finite frames without endpoint interpolation;
- preserves the two actor IDs through every frame, including crossings or
  endpoint exchanges;
- derives the listener position and forward/right/up basis from the exact M1
  `world_from_rig` quaternion;
- calls the shared explicit-track descriptor with Habitat Y-up mapping
  `(horizontal X/Z, vertical Y)`;
- records per-frame positions, horizontal/XYZ distance, closest frames,
  observed coincidence/order changes and listener-local azimuth/elevation;
- binds the request to its passing intended-floor native receipt and retains
  the RGB/depth/semantic/Topdown artifact records.

The route authority is explicitly `visual_actor_skin_root_not_audio_emitter`.
Audio endpoint/emitter tracks remain separate work. The static native probes do
not contain either actor, so projection, in-frame state, occlusion and visible
pixels remain `pending/null`; the tool never infers them from endpoint geometry.

## Legality and difficulty boundary

The complete legal raw viewpoint catalog is retained. Side margin, pair
distance, crossing/exchange, composition, blank-wall fraction, projection,
occlusion and view similarity do not filter this output. Native collision and
runtime actor-pose readback are also not claimed by this CPU descriptor.

The output is an ordinary research artifact with `research_only=true`,
`episode_counted=false`, formal release not run, candidate filtering `none` and
fresh/no-clobber atomic publication. No new hash, frozen contract, baseline or
permanent gate is introduced.

## Verification before real-data execution

Focused server verification is retained at:

`/data/jzy/tmp/mp3d_track_descriptor_unit_r1.log`

It records `28 passed` across the new tool, the shared explicit-track helper and
the MP3D viewpoint selector. Tests prove exact hold/motion frame preservation,
identity binding, Habitat axis mapping, explicit pending visibility, no-clobber,
nonpassing-receipt failure, exact 75-frame requirement and bank-path containment.
Ruff, formatting, Python compilation and `git diff --check` also pass.

A real-artifact run remains pending until this implementation is committed, so
the first runtime artifact can name clean code provenance rather than a dirty
worktree.
