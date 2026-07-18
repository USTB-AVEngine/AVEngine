# M6 Room Registry and Qualification Matrix

This file records the implemented room-interface boundary. It does not promote
historical ignored `tmp/` artifacts into portable release evidence. The checked-in
reports use `evidence_basis = audited_historical`; a fresh native run must use
`current_execution` and populate hash-bound `evidence_artifacts`.

## Lineage boundary

- `scene_lineage_id` identifies the source building/scan/procedural scene.
- `room_geometry_id` identifies the geometry revision.
- `layout_variant_id` identifies furniture/layout state.
- `acoustic_profile_id` identifies material parameters and is not a new room.
- `episode_layout_id` identifies actor/listener/camera/event placement and is not a
  new room.

The MP3D raw scan and derived acoustic proxy share the same scene lineage. They
remain separate acoustic representations and have separate qualification reports.

## Current honest matrix

Every result names its evidence basis. `historical` means retained lower-
milestone or research evidence; it is not a pass for the current M6 release
candidate. `current native not_run` means the M6 attempt did not rerun that
Habitat/RLR/Blender/media layer. A current read-only hash, provider or topology
inspection may report its own result without promoting a native layer.

| Room / representation | Evidence basis | Visual | Nav | Acoustic geometry | Material | Rays | Physical truth | Episode | Placement | Admission |
|---|---|---|---|---|---|---|---|---|---|---|
| `blender_custom_two_zone_v1` | audited historical M1/M3/M5; current M6 native `not_run` | historical pass; current native `not_run` | historical pass; current native `not_run` | historical controlled pass; current native `not_run` | historical controlled pass; current native `not_run` | historical controlled pass; current native `not_run` | historical `controlled_profile` | historical M5 pass; formal M6 commit-bound canary pending | current `not_run` | false |
| ReplicaCAD `apt_0` target proxy | current read-only provider resolution; current native `not_run` | resource resolved; native `not_run` | navmesh resolved; native `not_run` | proxy generation/native load `not_run` | `not_run` | `not_run` | unqualified | native `not_run` | `not_run` | false |
| Legacy UE Apartment real surface | audited historical migration plus current read-only package inspection; current native `not_run` | historical pass; current native `not_run` | historical pass; current native `not_run` | current static topology/parity gate fail; native load `not_run` | current placeholder/unqualified; native readback `not_run` | current native `not_run` | unqualified | historical research pass; current native `not_run` | current `not_run` | false |
| MP3D raw source | immutable raw-resource/hash inspection plus historical M5.1 research evidence; current native `not_run` | historical pass; current native `not_run` | historical pass; current native `not_run` | current diagnostic fail: 458 zero-area faces; raw native upload historically failed | historical placeholder; current native readback `not_run` | current native `not_run` | unqualified | historical raw RLR fail; current native `not_run` | current `not_run` | false |
| MP3D declared proxy v2 | current declared-derivation inspection plus historical research runtime evidence; current native `not_run` | inherited historical visual pass only | inherited historical nav pass only | derivation integrity pass; topology fail; historical solver load pass; current native `not_run` | placeholder/blocked; semantic mapping and current readback incomplete | current native `not_run` | unqualified | historical research pass only; current native `not_run` | current `not_run` | false |
| Independent corrupted fixture | current hermetic fixture evaluation; no native claim | n/a | n/a | fail by design | fail by design | fail by design | none | false | n/a | false |

No `pass` in one dimension implies a single overall room pass. Historical
research execution, current read-only qualification and current native
execution are separate claims. `dataset_admission` is a separate fail-closed
decision and remains false for every row in this snapshot.

## Provider API

```python
from avengine.m6.rooms import load_room_registry, find_room_record
from avengine.m6.room_providers import provider_for_id

registry = load_room_registry("examples/m6/rooms/room_registry.json")
record = find_room_record(registry, "replicacad_apt_0")
provider = provider_for_id(record["provider_id"])
resolved = provider.resolve_room(
    record,
    repository_root=".",
    environment={"AVENGINE_REPLICACAD_ROOT": "/declared/dataset/root"},
)
acoustic = provider.acoustic_representation(
    record,
    "replicacad_apt_0_acoustic_proxy_v1",
    repository_root=".",
    environment={"AVENGINE_REPLICACAD_ROOT": "/declared/dataset/root"},
)
```

All existing inputs are resolved through the central
`WorkspacePathPolicy`. Missing external roots are `blocked`; declared generated
packages are `not_run`; hash mismatch or root escape is `fail`. Providers never
fall back to private `/data/...` defaults.

## Placement-feasibility API

Native/provider code supplies measured rays to
`evaluate_placement_feasibility(probe)`. The pure evaluator requires:

- downward support rays at `center`, `front_left`, `front_right`, `rear_left`,
  and `rear_right`;
- at least four horizontal body-clearance rays;
- camera-frustum rays classified as `surface_hit`, `opening_exit`, or
  `scene_escape`;
- explicit allowed door/window IDs for legal `opening_exit` rays.

Any unsupported body corner, too-close obstacle, scene escape, or unlisted opening
fails the placement gate. This rejects unsafe episode positions; it does not modify
or claim to repair the original MP3D scan.

## Building an actual qualification report

```python
from avengine.m6.qualification import (
    build_qualification_report,
    evaluate_placement_feasibility,
)

placement = evaluate_placement_feasibility(measured_probe)
report = build_qualification_report(
    report_id="replicacad_apt_0_native_run_001",
    subject=subject,
    evidence_basis="current_execution",
    evidence_artifacts=[
        {
            "artifact_id": "native_room_evidence",
            "path": "canary_run/room_runtime_evidence.json",
            "sha256": measured_evidence_sha256,
        }
    ],
    dimensions=measured_dimension_checks,
    placement_feasibility=placement,
    acoustic_diagnostics=measured_acoustic_diagnostics,
    provenance=provenance,
    promote_if_eligible=False,
)
```

`build_qualification_report` computes eligibility but does not promote by default.
Promotion requires `promote_if_eligible=True` and every required dimension,
placement check, and acoustic diagnostic to pass with admissible material truth.

For all four checked-in audit reports, the main native execution still needs to
replace the empty `evidence_artifacts` array with current immutable artifact paths
and SHA-256 values. In particular:

- controlled room: current M6 one-active-of-N episode and placement evidence;
- ReplicaCAD: visual/PBR, nav/placement, derived package, material readback, rays,
  and RLR load evidence;
- Legacy Apartment: rebuilt real-surface/package hashes, the one-triangle parity
  resolution, topology/material/ray results, and full placement evidence;
- MP3D raw/derived: raw asset hash, derivation manifest/hash, spatial parity,
  solver upload, topology, semantic material/readback, ray, and placement evidence.

Historical reports remain `audited_historical`; their prose references are not a
replacement for current hashed evidence.

## Independent negative fixture API

```python
from avengine.contracts.json_io import load_json
from avengine.m6.qualification import qualify_corrupted_acoustic_fixture

result = qualify_corrupted_acoustic_fixture(
    load_json("tests/fixtures/m6/corrupted_acoustic_package/fixture.json")
)
assert result.report["dataset_admission"] is False
```

The fixture contains a zero-area face, material-count mismatch, fallback material,
source-hash mismatch, and failed ray expectation. It is not a fifth real room and
cannot be cleaned into admission by the report aggregator.
