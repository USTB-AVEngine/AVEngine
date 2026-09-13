# Prepare reusable navigation from the room surface

Navigation preparation runs once per room package, before episode requests reach the sampler. It reads the same render surface used by the room geometry pipeline and returns an ordinary `walkable_space` registration. It does not choose a renderer or require a particular room name.

Use the Python API from other code:

```python
from avengine.rooms.navigation_preparation import prepare_render_surface_navigation

prepared = prepare_render_surface_navigation(
    room_manifest_path,
    fresh_navigation_directory,
    runtime_prefix=habitat_runtime_prefix,
    magnum_python_site=magnum_site,
    rlr_sdk_root=rlr_sdk_root,
)
room_package["walkable_space"] = prepared["walkable_space"]
```

The manifest supplies `navigation.agent_height_m`, `navigation.agent_radius_m` and one `assets` entry with role `render_surface_mesh`. That mesh uses shared metre/+Y coordinates. For an existing scan stage with another orientation, pass its `stage_config` explicitly. Its render mesh supplies navigation geometry while its declared orientation is retained. Store host-specific package paths through the existing catalog path bindings when shipping a RoomPackage.

The equivalent command is `tools/rooms/build_render_surface_navigation.py --room-manifest ... --output ... --runtime-prefix ... --magnum-site ... --rlr-sdk-root ...`. The output contains `navigation.navmesh`, its stage description and `build_result.json`. Reuse that directory for later episode requests; rebuilding navigation per seed wastes work.

`load_planning_resources` routes `walkable_space.kind` to the existing path planner and keeps the room's declared UE/SPEAR or Habitat rendering backend. Existing valid navmeshes and retained grids remain supported. The camera-first solver prefers readable target endpoints and competitor positions that offer an angularly valid event window within its existing candidate limits.

These early preparations aim to avoid major wall intersections and expensive discarded routes. Small clothing or limb intersections can remain acceptable. Existing native pixels and sound evidence determine whether a requested question was realized; navigation construction does not claim zero mesh penetration or question acceptance.
