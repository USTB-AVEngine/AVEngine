# M6 Status — Feasibility Interfaces and Room Canary

> Historical pre-release snapshot. M6 was subsequently closed; the final
> result is recorded in [M6_FINAL_REPORT.md](../../release/M6_FINAL_REPORT.md).

Recorded: 2026-07-19 implementation snapshot, before release metadata commit.
Milestone closeout is determined only by the verified release manifest; this
snapshot is **pending** when no such manifest/tag exists. A later tagged
candidate may add `release/M6_FINAL_REPORT.md` without rewriting this pre-release
record.

This page is the human-readable M6 status record. It is not a release manifest
and cannot promote a room, asset or episode. Executable checks use only
`pass`, `fail`, `blocked`, and `not_run`; `implemented`, `research_only`,
`qualified revision`, and `pending` describe scope or lifecycle, not a
substitute verification result.

## Authority and scope

The authoritative task is
`CODEX_GOAL_AVENGINE_M6_FEASIBILITY_INTERFACES_ROOM_CANARY_V2.md`. It replaces
the earlier M6 direction that tried to freeze a complete dataset record or
generate natural-language QA. The room/extensibility, SelectTSL-aligned and
2026-07-18 code-review documents informed interface boundaries, but do not
override the v2 goal.

M6 establishes a task-neutral evidence foundation:

- Habitat-native remains the default runtime; UE/SPEAR/gpuRIR are explicit
  legacy/optional paths.
- The current audio interpretation remains 360° binaural, with independent
  per-source stems/RIR evidence. No 180° folded-DoA paper setting is imported
  into the runtime contract.
- Entity, emitter anchor, source endpoint and audio event remain distinct.
- The M5.1 source/event/flag definitions, thresholds, three-state values and
  OR/AND clip aggregation remain authoritative.
- Stable IDs and hashes support a later read-only task exporter. M6 does not
  define a final dataset-item schema, target source, SelectTSL heatmap, prompt,
  question or answer.
- Room qualification is multidimensional; no overall `pass` may hide a
  `fail`, `blocked`, `not_run` or unqualified physical-truth claim.

Hashes are evidence identities rather than feature locks. Git identifies
checked-in code, schemas and configuration; explicit versions identify the
toolchain. Content hashes are required only for result-changing bytes outside
that Git identity, generated package closures, formal test receipts and release
evidence. Leaf-file records may live inside one machine-readable directory
manifest so its closure digest is meaningful, but they are not repeated as
human-facing milestone locks. Temporary previews, logs and uncited
intermediates are not release gates.

## Current implementation matrix

The final column is deliberately conservative. A module being present does not
mean its native or release canary ran.

| Area | Implemented boundary | Current executable status | Closeout requirement |
| --- | --- | --- | --- |
| Workspace trust | Central `WorkspacePathPolicy`, declared roots, canonical paths, input/hash checks and no-clobber atomic publication | targeted unit evidence must be recorded in the final test matrix | keep `strict_untrusted_linux` unavailable until a real `openat2()` implementation and Linux integration tests exist |
| Bootstrap | Habitat-native `manifest.yaml`, `paths.yaml`, editable setup, schema validation and fast-test layer | dry-run/targeted checks exist; fresh-checkout run remains required for release | run from a clean checkout without private absolute paths or legacy dependency installation |
| Release authority | Versioned manifest schema and verifier | current manifest/tag: `not_run` / pending | generate `release/avengine_release_manifest_v1.json`, verify it from clean worktrees, and bind an annotated tag |
| Entity registry | Versioned entity asset records with stable IDs, hashes, anchors, provenance and admission state | targeted schema/unit evidence pending final aggregation | retain only evidence-backed assets and never infer admission from schema validity |
| Animal templates | Versioned body-plan/morphotype records, morphology ranges and structured OOD rejection | bounded Beagle registry route implemented; broad-species qualification is not claimed | prove no silent generic-Dog fallback and preserve size/build/life-stage plus breed-scoped three-value coat domains |
| Source/sound/program | Stable endpoints, independent dry sound assets and data-driven AudioProgram modes | contract/unit route and retained-evidence semantic materialization implemented; formal A3 bundle pending; native Habitat/RLR `not_run` | prove two endpoints exist while only the declared endpoint emits during specified windows; retain both stems and mixture |
| Legacy flags | Stable registry/access API and provider adapter over M5.1 v1 | contract/unit path implemented; controlled-canary report pending | preserve all IDs, thresholds, `present`/`absent`/`not_evaluated`, and legacy clip aggregation |
| Room provider | Portable providers, room registry, split qualification report and placement-feasibility evaluator | six-case read-only attempt implemented; attempt-verifier `pass` means report/artifact consistency, not six qualified rooms | replace prose-only history with current evidence where a new native or qualification pass is claimed |
| Negative fixture | Independent corrupted acoustic package and fail-closed evaluator | deterministic unit route implemented; final result must be retained | admission must remain false without treating MP3D as the permanent negative case |
| Future exporter boundary | Immutable stable-ID/hash view and protocol only | interface implemented; no task exporter is supplied | remain read-only and simulator-independent; do not add QA or model-specific labels in M6 |
| Controlled M6 materialization | Required bundle layout and request/evidence schemas | retained semantic-materialization semantics implemented; formal A3 evidence pending; native Habitat, native RLR and episode feasibility `not_run` | verify the retained M5 authority, deterministic one-active-of-N view, 360° binaural/FOA bytes, per-source IR/stems, flags and no QA text without promoting it to a native rerun |

## Registry and extension matrix

| Interface | Current canary entry | Not currently admitted | Extension method |
| --- | --- | --- | --- |
| Animal Template | audited Beagle/dog template | arbitrary cat, horse, bird or generated mesh; any out-of-range morphology | add a versioned body-plan package and validator evidence; OOD requests return structured rejection |
| Entity Asset | bounded articulated Beagle records and other evidence-backed examples only | unaudited assets and any asset whose rights/QA state is incomplete | add a registry record with byte hashes, realized attributes, anchors, provenance and explicit admission state |
| Source Endpoint | named entity/anchor endpoints, including two candidate muzzle endpoints | undeclared anchors or endpoint IDs inferred from file order | add an endpoint record referencing an existing entity anchor |
| Sound Asset | bounded dry-audio records independent of mesh identity | unreviewed sounds or implicit path-only audio | add a sound record with content hash, format, normalization, provenance, rights and admissibility |
| AudioProgram | `one_active_of_n` canary program; data contracts also express overlap, sequence, intermittent, swap and silence | claims that every mode has a native canary | add or compile program data without branching episode core on species/room IDs |
| Room Provider | custom, ReplicaCAD, Legacy Apartment and MP3D provider records | HM3D/HM3DSem batch support and undeclared local paths | add a provider adapter and room record; keep geometry/material/profile/layout lineages distinct |
| Task Exporter | read-only evidence-bundle protocol | SelectTSL export, QA generation and final training-item schema | implement a later deterministic exporter over stable IDs/hashes without rerunning Habitat |

Animal appearance domains remain explicit and reusable across later assets:
`size = small|medium|large`, `body_build = slim|standard|stocky`, a declared
life stage, and exactly three sensible coat values scoped to the species/breed
profile. A cross-breed name such as `golden` must not be silently reused as a
generic color.

## Source/event/flag continuity

The detailed authority audit is
[LEGACY_SOURCE_EVENT_FLAG_AUTHORITY.md](../architecture/LEGACY_SOURCE_EVENT_FLAG_AUTHORITY.md).
M6 reads that authority; it does not mutate the v1 schema or canonical hashes.

Required continuity:

- source identity and event windows originate in the route/source program and
  are resolved against actual emitter trajectories;
- source-level and pair-level assessments carry evidence and one of
  `present`, `absent`, or `not_evaluated`;
- unavailable FOV, visibility, occlusion or ray facts remain
  `not_evaluated`, never `absent`;
- clip aggregation keeps the existing flag-specific OR/AND rule;
- delivery annotations are overlays, not a second flag authority;
- future task records may derive labels from retained facts, but cannot rewrite
  the meaning of an existing M5.1 flag.

## Room result matrix

The attempt contains six report cases but only four visual room lineages.
MP3D raw and derived share one MP3D scene, and the corrupted fixture is not a
room.

The detailed, provider-facing matrix and placement contract are in
[M6_ROOM_MATRIX.md](M6_ROOM_MATRIX.md). The closeout snapshot must retain the
following independent columns even when several share the same result:

| Room / representation | Visual | Nav | Acoustic geometry | Material | Ray leakage | Physical truth | Episode feasibility | Admission |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `blender_custom_two_zone_v1` | historical `pass`; current native `not_run` | historical `pass`; current native `not_run` | historical `pass`; current native `not_run` | historical `pass`; current native `not_run` | historical `pass`; current native `not_run` | `controlled_profile` | retained M5 semantic materialization pending formal A3 verification; current native episode `not_run` | false |
| ReplicaCAD `apt_0` | development v2 native review `pass` (270 frames); formal A3 layer pending | development nav/LOS/semantic/furnished rigid-object root-center placement 19/19 checks `pass`; formal A3 layer pending | research RLR load/render `pass`, but stage-surface-only topology `fail` after cleanup (10,629 duplicate triangles, 102 boundary edges, 8,097 nonmanifold edges) | `research_placeholder`; unqualified | `not_run` | unqualified | 18 s human/Beagle research episode and 90-keyframe two-source binaural render `pass`; both dry/stem buses active in declared event windows; not formal qualification | false |
| Legacy Apartment real surface | historical `pass` | historical `pass` | historical gate `fail`/blocked | blocked/unqualified | `not_run` | unqualified | historical research review `pass` | false |
| MP3D raw source | historical `pass`; current native `not_run` | historical `pass`; current native `not_run` | `fail`: known zero-area faces block raw upload | placeholder/blocked | `not_run` | unqualified | historical raw RLR `fail`; current native `not_run` | false |
| MP3D declared proxy revision | inherited historical visual only; current native `not_run` | inherited historical nav only; current native `not_run` | 13-file descriptor/package binding and declared derivation pass in development; topology `fail` (45 duplicate triangles, 8,339 boundary edges, 269 nonmanifold edges); current RLR upload `not_run` | coverage exists but qualification is blocked by `research_placeholder`; semantic mapping/readback incomplete | `not_run`; no scene-specific opening/enclosure checks | unqualified | current native episode and placement `not_run` | false |
| Independent corrupted fixture | n/a | n/a | `fail` by design | `fail` by design | `fail`/`not_run` by design | none | false | false |

This is an honest starting matrix, not the desired final shape. A new
qualified revision may be created only from current immutable evidence after
all declared gates pass. The MP3D raw asset must remain unchanged; a legal
derived proxy reports raw identity, derivation integrity, visual-to-acoustic
spatial parity, solver loadability, topology, material and rays separately.

The development review bundle also includes one six-case, four-lineage video.
ReplicaCAD uses its real 18-second Habitat+Topdown+binaural clip; MP3D raw and
derived deliberately share visual bytes; unavailable raw/corrupted audio is
replaced by labelled stereo silence. This media is for human inspection and
does not change any room qualification or release-layer status.

Placement feasibility is an episode gate, not a repair of MP3D. It requires
floor support below the feet/body center and four corners, horizontal clearance
around the body, camera-frustum escape checks, and an explicit whitelist for
legal door/window openings.

## Test-layer matrix

This table is a closeout template. Replace `not_run` only with the exact command,
result and hash-bound evidence from the current implementation/release commits.
Historical M2–M5 results may be referenced, but cannot stand in for a current
M6 layer.

| Layer | Current M6 closeout status | Evidence required before changing status |
| --- | --- | --- |
| `fast-unit` | pending final M6-inclusive run | fresh editable install, schema validation and complete fast-unit totals |
| `slow-hermetic` | `not_run` | exact command, fixture identities and totals |
| `native-habitat` | `not_run` for the M6 release candidate | fork commit, build identity, native command and report hash |
| `rlr-audio` | `not_run` for the M6 release candidate | RLR commit/binary hash, all-pair/controlled-canary command and evidence hash |
| `blender-assets` | `not_run` | Blender version, package inputs and validator evidence |
| `media-readback` | `not_run` for the M6 controlled bundle | lossless WAV and MP4/readback identities and exact duration/channel checks |
| `release-canary` | `not_run` | verified release manifest, controlled evidence bundle and annotated tag |

## Definition-of-Done state

The M6 milestone remains open until all of the following are true in one
coherent release state:

1. bootstrap, trust policy and archived legacy instructions are verified;
2. the unique release manifest exists and verifies both repositories, native
   binaries, schemas, environment, test layers, evidence bundles and tag;
3. every registry/Room/AudioProgram contract and OOD/fail-closed behavior has
   current tests;
4. the controlled one-active-of-N retained-evidence materialization produces
   and verifies the required structured bundle, 360° binaural audio, stems/RIR
   bytes and legacy-compatible flags, while native Habitat/RLR execution stays
   independently reported;
5. custom, ReplicaCAD, Legacy Apartment and MP3D each have an honest current
   qualification attempt or an exact retained blocker;
6. the independent corrupted fixture reliably prevents admission;
7. every native/RLR/Blender/media layer is either executed with evidence or
   explicitly `not_run`;
8. no natural-language QA pair or substitute dense motion schema has been
   introduced.

## Claim boundary

When the controlled canary, room attempts and release manifest are actually
retained, the maximum defensible claim is that AVEngine has extensible
Habitat-native entity/source/sound/room interfaces, preserves its source/event/
flag semantics, verifies deterministic M6 semantic materialization over
independently verified retained M5 controlled acoustic evidence, attempts
auditable qualification across structured, migrated and scanned rooms, and
fails closed on a deliberately corrupted package.

M6 cannot claim a complete training dataset, large room coverage, arbitrary
animal/action generalization, measured physical materials for scanned rooms, a
QA benchmark, or a qualified MP3D revision unless the corresponding current
evidence independently passes every declared gate.
