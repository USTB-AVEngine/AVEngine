# M1 Status: Habitat Visual and Three-Room Canary

Status: `pass` on 2026-07-16 for the clean AVEngine commit recorded by formal
evidence and the Habitat runtime pinned by the M1 profile selected through
`runtime.lock.yaml`. Formal evidence
is generated under ignored `tmp/` output and remains valid only while the
checked-out AVEngine commit, runtime commit, runtime binary, input manifests
and scene assets match its recorded hashes.

## Contract proved by M1

- There is exactly one formal dataset viewpoint: `camera_rig_0/view0`.
- RGB, depth and semantic are three co-located, co-oriented modalities of that
  viewpoint. They are not three cameras at different viewpoints.
- `listener0` is a co-located pose anchor. M1 does not invoke RLR or claim an
  impulse response.
- At least two sources have unique stable names and pairwise-distinct world
  transforms, and pass transform round-trip checks. Multi-source acoustic
  propagation is deferred to M4.
- Capture does not advance world state. Same-process observations are
  byte-repeatable, and a second OS process with a different recorded process
  UUID must match a self-contained copy of the first run.
- Top-down navigation QA maps are diagnostic artifacts and never enter
  `formal_view_ids`.

## Executed room canaries

| Room | RGB color std (alpha excluded) | Positive finite depth | Maximum depth | Raw semantic IDs | Geometry/navigation result |
| --- | ---: | ---: | ---: | --- | --- |
| Official Habitat MP3D example | 50.656 | 99.9857% | 3.629 m | 18 IDs | declared navmesh and cross-room path pass |
| Blender custom two-zone room | 21.659 | 100% | 6.500 m | `0, 101, 102` | modeled door/window clear rays, solid-wall controls and cross-door path pass |
| Legacy SPEAR UE apartment | 29.079 | 100% | 4.790 m | `0, 101` visible | two paths pass on audited UE render LOD0 surfaces |

The custom room is a small tracked fixture. The official MP3D data is local
only under its upstream terms. The legacy apartment GLB is about 63 MiB and
remains under ignored `tmp/`; ordinary Git history contains the exporter,
packager and audit code, not the large derived scene.

## Habitat loaded-graph and navigation closure

For each room, M1 first proves the static dataset configuration and search-path
closure and then inspects the Simulator that performed the capture. The active
dataset and selected scene/stage, render and collision assets, navmesh and
semantic assets must resolve exactly to the room manifest. Handle-based custom
and legacy scenes additionally bind their selected scene instance, source
object templates and meshes, live semantic IDs and pairwise-declared poses,
and selected/current lighting setup. Ambiguous or alternate same-content paths
fail; matching bytes alone do not excuse loading the wrong file.

All three canaries declare `navmesh_policy: load_declared`. Capture explicitly
loads that file into the active Habitat Pathfinder and independently loads it
into a second Pathfinder. Their full fingerprints match, including all 17
navmesh settings and the shapes and hashes of canonical vertex and index
buffers; the stored agent settings also match each room contract.

## Legacy real-surface provenance

The admitted legacy canary is not the retired 252-triangle actor-bounds proxy.
It was freshly exported under UE 5.5.x after deleting stale manifest/output
targets. Its audited export contains 782,305 triangles, 463,873 vertices, 118
meshes and 48 materials. The UE export manifest, Blender mesh audit and GLB
agree on SHA-256
`3c0f53d0a46d99cad90b016763d420d53618cfd0ca881851f142db4d91d399b2`.
The package also binds the source map to SPEAR commit
`7fbf3632fdb63cc2eceea564811c9597cabfb199` and the tracked
`apartment_0000.umap` SHA-256
`042e2a7d9b7123c5ae1a1aa2aa775459e0827067b65d4b07bd7067689a7b7a8c`.
The selected render world closes over 144 `/Game` StaticMesh/material `.uasset`
packages: every recorded path is inside the SPEAR project, Git-tracked and
SHA-256 matched. Its one `/Engine` material reference is recorded separately
rather than misrepresented as a SPEAR file. The counts describe this admitted
apartment export; the verifier enforces the same path/hash closure for any
future selected-package count.

## Evidence and negative gates

The completed evidence schema is executed with JSON Schema Draft 2020-12 and
then independently replayed semantically. Verification rehashes raw arrays,
artifacts, inputs, scene assets, the native Habitat binding, runtime lock and
the self-contained first-run reference. It also recomputes observation
statistics, sensor/listener alignment, source transforms, connectivity and ray
thresholds, navigation QA pixels, the recorded actual Habitat loading graph,
active/independent Pathfinder fingerprints, producer-process identity, and
custom/legacy surface provenance. A structurally valid blocked attempt stays
`blocked`; it cannot be aggregated as a completed room.

The M1 implementation unit suite contains 91 tests. The completion procedure
is in [M1_EXECUTION.md](M1_EXECUTION.md). M1 passes only when the three fresh
second-process evidence files each verify and their exact three-room aggregate
returns `pass` from a clean worktree.

## Carried M0 failures and M1 non-claims

Two pinned-runtime failures remain explicit:

- direct `import habitat_sim` aborts unless `numpy-quaternion` is imported
  first; M1 uses the documented import order;
- 21 `GreedyGeodesicFollower.find_path` binding cases fail with a PyCapsule
  iterator error; M1 uses `ShortestPath`, not that follower.

M1 does not prove articulated animal playback, acoustic material activation,
RLR propagation, per-source stems, the 75-frame/80,000-sample timeline,
counterfactual invariance, dataset admission or benchmark quality. Those remain
M2-M7 gates.
