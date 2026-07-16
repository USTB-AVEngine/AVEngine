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

## M1 observation and transform contract

The room manifest records the coordinate frame, unit scale, scene assets,
openings and navigation QA declarations. The separate M1 capture request
records one logical `camera_rig_0`, its co-located RGB/depth/semantic
calibration, the co-located `listener0`, and independently named source
transforms. `world_from_rig` is the formal camera/listener viewpoint itself;
the MVP `rig_from_sensor` and `rig_from_listener` mounts are identity. The
formal room canary emits exactly `view0`; a top-down navigation map is a
labeled QA artifact and is excluded from dataset observations, timelines and
benchmark inputs. M1 requires at least two uniquely named sources whose world
transforms are pairwise distinct. Its listener is a pose anchor only: M1 does
not instantiate an AudioSensor or run RLR, and multi-source propagation remains
the M4 gate. See
[ADR-0009](../adr/ADR-0009-single-view-multimodal-sensor-rig.md).

## M1 visual loading and navigation closure

A room is not admitted merely because every declared file exists. M1 closes
the dataset search paths and selected scene instance or path, then records and
checks the stage and render/collision/semantic assets that Habitat actually
loaded. For handle-based scenes it also checks live source-marker objects and
lighting. Every selected handle must resolve to exactly one declared file.

The official Habitat, Blender custom and legacy UE canaries all use
`navmesh_policy: load_declared`. Capture explicitly loads the declared navmesh
into the active Pathfinder and into an independent Pathfinder. Their full
fingerprints—every navmesh setting and canonical vertex/index hashes—must
match, and the embedded agent settings must agree with the room manifest.

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
This is an M3/M4 architecture contract, not an M1 execution claim.

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
