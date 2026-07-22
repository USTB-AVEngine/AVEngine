# Real-reference animal coat workflow

Status: `research_candidate`; optional SPEAR/Blender asset-preparation route.

AVEngine's template-authoritative animal decision remains unchanged: accepted
geometry, UVs, skin, skeleton and actions are runtime authority. Generative
models may change breed-scoped appearance, but they may not silently replace
topology or animation.

This workflow starts only after a source asset already represents the requested
breed and morphotype. A new breed or materially different body shape is not a
coat edit of another breed; it must follow the
[`generated animal asset and instance contract`](GENERATED_ANIMAL_ASSET_AND_INSTANCE_CONTRACT.md)
and create its own mesh, rig and source-asset identity first.

The reusable coat route validated on a Blue Abyssinian is:

```text
rights-reviewed real photos for one breed × coat
  → deterministic appearance board
  → fixed rest-pose front/back/left/right renders of the accepted GLB
  → one undistilled FLUX.2 Klein Base edit
     image 1 = geometry/layout authority
     image 2 = coat appearance authority only
  → human four-view review
  → spatial chroma projection onto the original UV texture
  → sRGB identity check
  → one-skin, Idle/Walking GLB readback
  → moving visual review
```

The implementation belongs to the independent SPEAR repository:

Current reviewed implementation: SPEAR commit `6b6d9199` on
`feature/plan2-flag-generator-m1`.

- `tools/build_animal_coat_reference_board.py`
- `tools/blender_render_generated_animal_coat_views.py`
- `tools/flux2_edit_animal_multiview_coat.py`
- `tools/blender_project_animal_multiview_coat.py`
- `docs/generated_animal_real_reference_coat_workflow.md`

This route is intentionally not part of default Habitat imports or runtime
episode generation. It is an offline asset build step: run it once per accepted
breed/coat asset, then reuse the resulting GLB across trajectories, RIRs and
audio-event combinations.

## Attribute boundary

- `coat`: this workflow may implement one of the three breed-specific coat
  profiles using real references and a reviewed spatial transfer.
- `size`: remains `small / medium / large` under the existing bounded instance
  policy; a coat edit does not alter size.
- `body_build`: remains `slim / standard / stocky`; material colour cannot
  pretend to change anatomy.
- `life_stage`: remains explicit and separately validated.
- A new species, breed silhouette or morphotype still needs its own mesh,
  binding, deformation and runtime review. It cannot reuse another animal's
  shape merely because the texture looks plausible.

## Admission boundary

A successful coat video is necessary visual evidence, not formal admission.
The asset stays `research_candidate` until its ordinary source rights,
geometry, skin, action, deformation, contact, collision, room and human-review
gates pass. Large photos, generated images, textures, GLBs and videos remain in
ignored artifact storage; Git tracks the code and compact workflow only.
