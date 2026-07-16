# Room and Acoustic Scene Package

## Separation of concerns

A visual PBR material is not an acoustic material. Base color, roughness,
metallicity and normal maps do not provide reliable absorption, scattering or
transmission coefficients. Acoustic categories and parameters must be
explicit, versioned and auditable.

## Room package

```text
room_package/
  room_manifest.json
  provenance_manifest.json
  visual/
    scene.glb
    scene_dataset_config.json
    semantic_descriptor.json
    navmesh.navmesh
  acoustic/
    vertices.npy
    triangles.npy
    triangle_material_ids.npy
    material_categories.json
    material_database.json
    acoustic_proxy.glb
  qa/
    geometry_report.json
    material_coverage.json
    ray_leakage.json
    visual_acoustic_parity.json
```

Optional visual/semantic/navmesh files remain optional only when the manifest
states why they are absent.

## Acoustic Scene Package contract

The runtime-facing package records:

- Vertices, triangle indices and object transforms.
- Triangle-to-material-category assignments.
- Material-category-to-RLR parameter mappings.
- Unit scale and coordinate convention.
- Source scene revision and geometry/material hashes.
- Mapping confidence, fallback usage and human overrides.

The runtime adapter uploads this package through the modern RLR object/mesh
API and can export the resulting debug scene mesh for parity inspection.

## Geometry policy

Axis-aligned bounding boxes are allowed only for fast debug. Production
acoustic proxies must preserve door/window openings, room connectivity, major
occluders/reflectors, normals and meaningful concavity. Simplification must be
measured against the visual scene rather than replacing objects with boxes.

## Initial dynamic scope

M3-M5 assume static or quasi-static room geometry with dynamic semantic point
emitters. Animal deformation does not participate in per-frame acoustic
reflection. Any future dynamic acoustic geometry requires a new ADR and
performance/validity canaries.

## Admission gates

- Every production triangle has a known material category.
- No unintended default/fallback material.
- Geometry is watertight enough for the selected RLR configuration or its
  openings are intentional and documented.
- Ray leakage and scene parity checks pass.
- High- versus low-absorption canaries create a repeatable change beyond
  simulation variance.
