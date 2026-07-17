# Acoustic Scene and Material Architecture

This document defines the M3 boundary between AVEngine-owned scene/material
compilation and the reused RLR acoustic propagation algorithm. It also fixes
what evidence is sufficient to claim that an explicit scene reached the
runtime without silently becoming an axis-aligned box, an implicit material,
or an unrelated mesh.

## Ownership boundary

| Layer | Owner | Responsibility |
| --- | --- | --- |
| Geometric acoustic propagation, ray tracing and impulse-response synthesis | RLR Audio Propagation | Reused propagation algorithm; AVEngine does not replace or reimplement it |
| RLR context lifetime, strict configuration transfer, explicit object upload and native readback | `habitat-sim-AVEngine` | Isolated runtime adapter over the modern RLR API |
| Source-room extraction, canonical transforms, per-triangle material assignment, evidence replay and admission policy | AVEngine | Explicit compiler/adapter inputs, fail-closed validation and evidence |

The runtime fork may expose RLR operations and return owned result buffers, but
that is not a new acoustic solver. AVEngine's M3 contribution is the explicit,
hash-bound path into RLR and the independent evidence around that path.

## Data flow

```text
Room manifest + source GLB
  + reviewed source-to-canonical transform
  + explicit source-material mapping
  + versioned acoustic material database
    -> AVEngine Acoustic Scene Package compiler
    -> canonical arrays, object partitions and RLR material database
    -> compiler QA and source-to-package replay evidence
    -> habitat-sim-AVEngine RLRAcousticContext adapter
    -> exact upload receipts + resolved material blocks + geometry readback
    -> RLR impulse response and ray results
    -> AVEngine metric recomputation and evidence verification
```

Each arrow is fail-closed. A valid JSON document, a successful native call, or
an exported OBJ is never sufficient by itself.

## Acoustic Scene Package v1

[`acoustic_scene_package_v1.schema.json`](../../schemas/acoustic_scene_package_v1.schema.json)
defines the runtime-facing manifest. A package contains:

- canonical little-endian `vertices`, `triangles` and
  `triangle_material_ids` arrays;
- stable object partitions and baked canonical transforms;
- exact source room, geometry, mapping and material-database lineage;
- category metadata and the generated RLR material database;
- geometry, material-coverage, opening/ray and compiler-parity reports;
- a compiler-side debug OBJ and hashes of the compiler implementation;
- a package content hash covering the declared files and manifest fields.

The package copies the source mapping and source material database into its
provenance directory. Verification nevertheless reopens the hash-bound source
GLB, mapping and databases, recompiles them, and compares the expected arrays,
object partitions, categories and RLR database with the package. Internal
self-consistency alone cannot establish lineage.

### Geometry policy

Production-mode geometry must be `real_surface_mesh` or
`controlled_surface_mesh`. `debug_aabb_proxy` is never a production
representation. The compiler expands the selected GLB scene graph, applies
the declared source-to-canonical transform, bakes node instances into the
canonical world, preserves triangle material identity and records the source
geometry hash.

The strict GLB route intentionally rejects unsupported constructs instead of
guessing: non-triangle primitive modes, sparse or compressed accessors,
skinning, morph targets, and unsupported component layouts do not receive a
silent approximation. Door/window clear rays and solid-wall control rays are
evaluated against the emitted canonical triangles.

## Acoustic material contract

Visual PBR slots are identifiers, not measurements of acoustic behavior.
Base color, roughness, metallic response and texture names cannot establish
absorption, scattering, transmission, damping, density or sound speed.

The mapping and database contracts therefore remain separate:

- [`m3_acoustic_material_mapping_v1.schema.json`](../../schemas/m3_acoustic_material_mapping_v1.schema.json)
  maps every exact source material slot to one stable acoustic category and
  material key;
- [`m3_acoustic_material_database_v1.schema.json`](../../schemas/m3_acoustic_material_database_v1.schema.json)
  supplies explicit band frequencies, coefficients, units, provenance and
  intended use;
- material category names must resolve by exact label in the generated RLR
  database; substring or first-match selection is not allowed;
- every source slot used by the geometry must be mapped exactly once, every
  triangle must receive an ID, and fallback/random assignment remains false.

### Semantics and admission

| Material semantics | Intended use | Permitted claim |
| --- | --- | --- |
| `controlled_canary` | Deliberate high/low activation experiment | `synthetic_activation_test_only` |
| `reviewed_physical` | Separately reviewed physical profile | `reviewed_physical_material_profile` |
| `research_placeholder` | Compiler and mapping diagnostics | `unqualified_research_placeholder` |

`package_mode: production` means the package took the strict production
compiler path. It does not override the material semantics and does not grant
dataset admission. In particular, the controlled custom-room high/low pair is
a **synthetic activation test**, not physical material truth for its floor,
walls, ceiling or door frame.

Visual-slot proposals are always research inputs. They use
`mapping_source_kind: visual_material_slot_proposal`, have no physical
qualification claim, and cannot be promoted merely by compiling successfully.

### M3.1 user-controlled material profiles

[`avengine_m3_acoustic_material_profile_v1.schema.json`](../../schemas/avengine_m3_acoustic_material_profile_v1.schema.json)
is a convenience layer over the existing explicit mapping and database. It
does not alter the source mesh, runtime package or RLR API. Resolution is
deterministic and field-wise:

```text
base material database < global override < exact material override
```

Curve scalars are expanded to every base-database frequency band. Explicit
arrays must have exactly the same length as `bands_hz`; unspecified fields
retain the value from the preceding layer. The resolver rejects unknown,
duplicate and conflicting selectors. A `source_material_name` selector must
resolve to a material key used by exactly one source slot; when several slots
share a key, the caller must select that key to change all of them or author
separate slots. The resolved mapping, complete effective database, field-level
lineage, input/output hashes and selector resolutions are emitted before the
same strict compiler path runs. A global override is therefore expanded into
explicit records, not retained as a hidden runtime fallback.

Profile values remain subject to the base database's declared material
semantics. User-selected coefficients do not become physical measurements or
dataset admission merely because resolution and compilation pass. If a
profile changes a `reviewed_physical` base, the effective database is
automatically downgraded to `research_placeholder`; recovering a physical
claim requires a separate review of the complete resolved database.

### Decay target boundary

RT60/T60 and EDT are room-response metrics derived from an RIR under declared
geometry, material coefficients, solver settings and source/listener anchors.
They are not properties that can be assigned independently to a floor or wall,
and the pinned RLR API has no direct RT60 setter. `global_volume` scales output
amplitude; `max_ir_seconds` bounds IR duration and can truncate the decay tail.
Neither controls reverberation time.

AVEngine exposes a bounded `calibrate_broadband_edt_seconds` search core. It
adjusts one uniform absorption scalar by repeatedly calling a caller-owned EDT
evaluator, requires higher absorption to produce no longer EDT, and checks
repeat spread, reachability, monotonicity and convergence. A successful result
records the full target/achieved/error trace. This is specifically broadband
EDT, not frequency-band RT60 or recovered physical material truth. Native
multi-anchor target-decay orchestration and evidence remain `not_run` until
they bind the mesh, anchors, simulation configuration, retained RIRs and
achieved tolerance.

## Controlled high/low counterfactual

The tracked Blender custom-room fixture assigns four explicit author slots.
The low and high databases deliberately use a synthetic `0.02` / `0.60`
absorption pair at every band to prove that material selection reaches the
propagation runtime. Their contract freezes:

- source room and source geometry;
- canonical vertices, triangles, object partitions and material IDs;
- material keys, labels, names, bands, scattering, transmission, damping,
  density, speed and provenance structure;
- listener/source anchors and the complete RLR configuration.

Only `database_id` and `materials[*].absorption` may differ, and every high
absorption coefficient must be strictly greater than its low counterpart.
The experiment must show the declared EDT, DRR and late-energy directions and
minimum effects while repeat spread remains below the fixed thresholds.

The high value is intentionally below the exploratory `0.90` condition. At
`0.90`, the unchanged direct impulse held about 68% of total energy and the
Schroeder curve fell about 5 dB immediately, so a single line over the genuine
0 to -10 dB EDT interval was not supported (`R² = 0.770`). Excluding that
direct impulse would instead measure a T10-like reverberant tail and must not
be relabeled as EDT. The tracked `0.60` condition preserves the unchanged EDT
definition and fixed `R² >= 0.90` gate while retaining a large controlled
material contrast. This is fixture design, not physical calibration.

Passing this canary proves material-path activation under the controlled
configuration. It does not calibrate room materials, establish perceptual
quality, or validate real-room acoustic coefficients.

The authoritative formal result, measurements and evidence hashes are recorded
in [M3_STATUS.md](../roadmap/M3_STATUS.md).

## Runtime ingestion evidence

M3 uses a fresh RLR context for every canary repeat, records strict
configuration readback, and binds the native Habitat and RLR binaries. Scene
ingestion is proved by complementary records:

1. the package arrays and object/material partitions are independently replayed
   from the hash-bound source inputs;
2. exact API receipts report object IDs, object/vertex/triangle/category counts
   and triangle counts by resolved material category;
3. the adapter resolves and records the material blocks that were supplied to
   RLR and checks them against the compiled database;
4. a post-ingestion native scene OBJ is parsed independently and its canonical
   vertex/triangle coordinate multisets, counts and file hash are compared with
   the package;
5. declared opening and solid-control rays are recomputed on the package mesh
   and compared with native any-hit and first-hit results.

RLR documents the first-hit surface normal as un-normalized: its magnitude
depends on the hit triangle geometry and is not an acceptance invariant. A hit
therefore requires a finite, non-zero normal, while first-hit misses and every
any-hit query must retain the binding's finite zero distance/normal sentinels.
Hit agreement and first-hit distance tolerances remain unchanged.

The post-ingestion OBJ is useful geometry and resolved-material-block evidence,
but its format does **not** expose a recoverable per-face material-ID array.
It must not be described as proving `triangle_material_ids` by itself. Exact
per-face assignment is instead verified by the source replay plus exact upload
receipts and resolved material blocks, while the OBJ independently verifies
post-ingestion geometry.

The final evidence verifier rereads raw IR arrays and recomputes direct arrival,
EDT, DRR, late-energy ratio, repeat spread and high/low comparisons. Boolean
fields already present in an evidence JSON are not trusted as the calculation.

## Real-room research boundary

MP3D and the legacy UE real-surface apartment are useful compiler stress
tests. Their current visual-slot mappings and material databases are only
`research_candidate` proposals:

- no visual material name is treated as acoustic truth;
- no mapping confidence or physical coefficient review is inferred;
- geometry or transform diagnostics remain visible rather than being waived;
- successful package generation cannot promote either room to a production
  acoustic scene or dataset asset.

Promotion would require an explicit reviewed mapping, a reviewed physical
material profile, passing geometry/opening evidence and a separate admission
decision.

## M3 versus M4

M3 may use one named source and one named listener to exercise a controlled
material canary. That does not complete multi-source identity semantics.

M4 still owns:

- at least two simultaneously registered, stably named sources and exactly one
  MVP listener;
- independently readable source/listener-pair IRs and identity-bound stems;
- source-registration order invariance;
- reset and temporal-coherence policy;
- a multi-source performance report.

No M3 evidence may claim those outcomes. See
[`MILESTONES.md`](../roadmap/MILESTONES.md) for the sequential gate boundary.
