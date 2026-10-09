# InteriorAgent/Kujiale CAD room selection

The adapter reads original `rooms.json` in source-world metres, the reviewed
`geometry.source_to_canonical.matrix_row_major` in each existing unfiltered
acoustic package, and the exact navmesh used by the earlier CPU placement
search. It applies the recorded matrix once; it never searches signs or rotates
an image until it appears plausible. The old upward-facing USD floor measurement
clips CAD to actual authored floor support, preserving holes and components.
`input_plan_v1.json` records both CAD area and supported area.

The new adapter modules are in `tools/rooms/room_split_kujiale/`. The selection
pipeline depends on the committed HM3D v6 `shape_quality_geometry` and
`shape_quality_repair` modules. Merge that revision before a final run. A missing
v6 dependency is an error; uncommitted files in another worktree are never an
execution dependency. Owner-approved limits are 6–35 m² and short side >=2.4 m.
New cuts also need a certified 2.4 m disk in the real delivered polygon. Original
subcap rooms do not receive a new visibility-only admission threshold.

Doors precede shape construction. The v6 repair uses only each room's outline,
then wall-axis and +/-15 degree straight cuts; the adapter's last fallback uses
the existing HM3D two-leg orthogonal generator. The full designed chord must
cross <=0.5 m of shape-preserving furniture. Directly annotated stair triangles
are removed only in their measured floor-height band. Other CAD room floors are
excluded from any connectivity bridge. Far-part paths are rechecked on the old
navmesh and must stay in the original room plus its distinct-part pair bridges.
A failure stays unresolved; no original scene, checkpoint or production record
is rewritten.

Frozen navigation area, dominant floor fraction, grid, clearance, distance, FOV
and ray budgets come from the existing `thresholds.frozen.yaml`. The task's
6 m² minimum overrides its older 4 m² area setting; the source file stays read
only. Scan black-area incompleteness is explicitly not applicable to authored
CAD/USD. Every retained room has a CPU three-ray placement witness, followed by
independent checks that the camera and two sources are inside the actual polygon
and exactly supported by the old navmesh. The ray surface is the unchanged,
unfiltered old surface, numerically compared with the original USD vertices.

`prepare_scene` reads the original composed USD mesh transforms and authored
material colours, including material subsets. It produces depth-buffered CPU
orthographic JPEG renders. Material colour uses linear authored colour and the
mean decoded texture colour; these are review images with neutral diffuse light,
not a simulation of UE lighting. The original ceiling is hidden for a top-down
review and nonwall surfaces are clipped at floor+2.2 m. Render metadata supplies
`path`, `projection`, `view`, `floor_y_m`, `span_m`, and `size_px` for the owner's
redraw script. Whole-house production geometry remains intact.

Run all tools with the dedicated room-selection Python, `PYTHONPATH=src:.`,
`PYTHONDONTWRITEBYTECODE=1`, empty `CUDA_VISIBLE_DEVICES`, single-thread BLAS,
`NUMBA_NUM_THREADS=1`, `PXR_WORK_THREAD_LIMIT=1`, and nice 10–15. `prepare_scene`
and `pipeline` allow at most four workers with 20 GiB address-space limits and a
10 GiB parent, for a combined 90 GiB ceiling. Use one compute batch at a time.
`resources.py` observes only that parent's process tree in a thread.

```bash
python -m tools.rooms.room_split_kujiale.prepare_scene --root "$TASK_OUTPUT" --workers 4
python -m tools.rooms.room_split_kujiale.stairs --root "$TASK_OUTPUT"
python -m tools.rooms.room_split_kujiale.pipeline --root "$TASK_OUTPUT" --workers 4
python -m tools.rooms.room_split_kujiale.delivery --root "$TASK_OUTPUT"
# Run the original Claude check and redraw tools, writing only new task outputs.
python -m tools.rooms.room_split_kujiale.review --root "$TASK_OUTPUT"
```

Prepare `input_plan_v1.json` and `navmesh_map_v1.json` with
`adapter.plan_inputs(old_root, dataset_root, output)`, then call
`adapter.coordinate_evidence` for every house. Output is exclusive creation:
existing products are never replaced. A rerun uses a new attempt output root.

The delivery keeps the HM3D `regions/*.json`, `rooms/*.json`, `summary.json`
layout. The draft list has `room_list_draft_v1.json`, `rooms.csv`, `pending.json`
and `summary.json`. Leakage remains null pending the separate acoustic task:
<=5% permits test, 5–15% only training, and >15% exclusion. Geometric retention is
not a claim that acoustic or new native production-media admission has passed.

The review HTML embeds JPEGs as JSON text, leaving every image `src` empty.
`IntersectionObserver` with zero root margin loads only intersecting images;
manual clicking is required in browsers without that API. Enlarged images load
only on click. A static page check is reported separately from a real browser
check. The additional all-source audit view changes `requires_split` only to
select all 151 rows in the owner's split-only audit/redraw scripts; it cannot be
used as the production delivery.

InteriorAgent inputs and derived review images remain noncommercial research
only and must not be redistributed. No dataset assets or generated media enter
Git.


Optional original-room physical evidence may live in `original_physical_v1/`
under the artifact root. A previously found witness is reused only for an
identical real polygon, identical frozen parameters, and the same recorded
navigation and raw surface input paths. The final worker independently checks
its camera and two source points inside that exact polygon, their current
navmesh support, and all three current raw-mesh segment rays. Failed or absent
evidence falls back to the normal frozen search. Original evidence is a
precheck and never substitutes for final v6 shape/connectivity admission.


Connectivity groups are re-evaluated on every delivered leaf after geometric
cuts. A cut may detach a tiny part that was joined in the parent. Unsupported
components below 6 m² are separate discarded fragments; larger components
become separate room candidates and go through the normal shape/navigation/
placement checks. A valid pair bridge still keeps a supported small part in
its room. This does not enlarge the real delivered floor geometry.
