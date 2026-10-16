# Single-Repository Functional Equivalence Plan

Status: active execution plan; migration is not complete.

This document organizes a behavior-preserving source consolidation into
`USTB-AVEngine/AVEngine`. It does not create a new baseline, frozen contract,
hash policy, admission gate or release authority. Existing schemas, validators,
ordinary tests, native checks and the release manifest retain their current
authority.

## 1. Objective

Keep all required distributable AVEngine source, selected Habitat integration
source, AVEngine-owned RLR adapter source and selected SPEAR integration source,
together with small configuration and license/provenance records, in the
canonical AVEngine repository.

After cutover, setup, build and runtime must not clone, initialize a submodule
for, or resolve code from a separate AVEngine, Habitat, RLR or SPEAR Git
checkout.

This is a source-repository boundary, not an attempt to vendor every runtime
input. Unreal Engine installations, Epic content, datasets, native room assets,
USD/MDL scenes, HRTFs, model weights, compiled libraries, generated media,
caches and build products remain outside Git.

## 2. Acceptance floor and scope ceiling

### Acceptance floor

The refactor is functionally equivalent only when all of the following remain
true for the supported production surface:

- MP3D, Apartment and InteriorAgent/Kujiale remain runnable through their exact
  production visual routes and authority split.
- Existing public CLI names, required arguments, defaults, exit behavior,
  schemas, IDs, status vocabulary and no-clobber semantics remain compatible.
- Timeline remains integer time at 48 kHz: 75 formal frames, 5 seconds, 15 fps,
  3,200 ticks per frame and 80,000 samples at 16 kHz.
- Source/listener/camera identity, source ordering, coordinates, pose semantics
  and frame/sample intervals remain unchanged.
- FOA remains `[W, Y, Z, X]` in ACN/N3D under the existing AVEngine coordinate
  convention; binaural remains `[left, right]`.
- Required artifact names, arrays, dtypes, shapes, units, channel counts,
  runtime readbacks and ordinary validator outcomes remain present.
- Existing failure classification, fresh-output requirements and formal/research
  status semantics are preserved.
- No production room is silently replaced by another backend or by a diagnostic
  capture.

### Scope ceiling

This task consolidates source and adapts ownership boundaries. It does not:

- redesign visual or acoustic quality;
- add a room family, dataset, schema generation, benchmark or release claim;
- import UE, data, room assets, weights, binaries or generated output into Git;
- promise bit-identical output from a native backend that does not already
  promise determinism; or
- preserve uncalled examples, editor conveniences or unrelated upstream
  features merely because they existed in a source repository.

Any quality or feature improvement begins only after equivalence is established
and is reviewed as a separate change.

## 3. Production routing authority

| Room family | Production visual execution | Other authority |
| --- | --- | --- |
| MP3D | Habitat-Sim scene, pixels, sensors and articulated pose | RLR uses SoundSpaces material authority on the same Habitat scene/state; AVEngine owns Episode, Timeline and labels |
| SPEAR `apartment_0000` | Native UE/SPEAR map | AVEngine owns Timeline, task/source state, navigation semantics, audio, Topdown, labels and admission |
| InteriorAgent/Kujiale | UE/SPEAR USD/MDL adapter over an explicitly selected external scene | AVEngine owns Timeline, task/source state, navigation semantics, audio, Topdown, labels and admission |

An MP3D UE import remains `comparison_visual` and cannot satisfy the MP3D
production row. Skokloster remains excluded and is neither run nor counted
unless the project owner explicitly reauthorizes it for a named task.

## 4. Source and dependency treatment

Every selected path is recorded in
`docs/provenance/UPSTREAM_ADAPTATIONS.md` as one of:

- `adapted`: selected upstream source retained with bounded AVEngine changes;
- `reimplemented`: AVEngine-owned source reproducing a required behavior or
  interface; or
- `external runtime/data`: configured installation or input whose bytes stay
  outside this repository.

The migration includes only code exercised by supported routes and the small
configuration needed to build or select it. Applicable license notices travel
with adapted source.

The current RLR pin contains headers, configuration and a precompiled shared
library, not propagation-engine source. Therefore strict all-source completion
cannot be claimed from the current pin. Before cutover, either obtain and review
redistributable propagation-engine source or receive an explicit owner decision
that the RLR shared library remains an installed external runtime dependency.
Neither choice may require a separate RLR Git checkout, and the precompiled
library itself does not enter Git.

SPEAR-owned MIT source is distinct from Unreal Engine and from its Boost
(BSL-1.0), rpclib (MIT) and yaml-cpp (MIT) dependencies. Precompiled third-party
libraries do not enter Git. UE binary assets currently used to provide camera
or render-pass behavior must be replaced by selected source/configuration or by
a reviewed build-time generated equivalent; they cannot be copied into the
source repository.

## 5. Comparison rules

### Exact comparison

Compare exactly where behavior is discrete or deterministically serialized:

- selected backend, backend role and authority boundary;
- CLI projection, defaults, relevant exit code and error class;
- schema names/versions, required keys, types and status vocabulary;
- stable request, Episode, scenario, actor, source, listener, camera and asset
  IDs;
- planned Timeline, pose and event records;
- frame/sample/tick counts, rates, channel counts and channel ordering;
- FOA/binaural convention and coordinate conversion;
- expected artifact names, required artifact presence and no-clobber behavior;
- deterministic AVEngine assembly, labels, Fact/Question results, assignments,
  deduplication keys and split membership from identical native inputs.

### Existing tolerance or ordinary review

Do not require byte identity where the current backend does not promise it:

- RLR impulse-response samples and derived spatial/acoustic measurements use
  the repository's existing native verifier and existing thresholds.
- UE pixels/video use existing frame, array, planned-versus-live readback,
  visibility and human visual-review procedures; dimensions, frame count and
  timing remain exact.
- Habitat/native floating-point pose and sensor results use their existing
  comparator when byte equality is not already required.
- Performance measurements remain informational unless an existing project
  requirement already classifies them.

No new numeric tolerance is introduced merely for this migration. Functional
equivalence means exact discrete semantics plus the same existing native
validation behavior, not an unsupported cross-run byte-equality claim.

## 6. Reference execution matrix

All entries use the server worktree and real retained inputs. A row is run only
after its prerequisites are available.

| Layer | Current entrypoint or authority | Required reference |
| --- | --- | --- |
| Hermetic software | relevant unit/schema suites and repository CLI | same tests and ordinary validators before and after each source batch |
| MP3D visual | a new reviewed Habitat-native strict two-human 75-frame runner is required | one fresh full-75 production Episode from one Habitat Simulator; an imported-GLB UE capture is diagnostic only |
| Apartment strict Episode | `tools/qa/capture_spear_native_pixel_episode.py` with a newly generated production-role suite and no `--frame-index` | fresh 75-frame RGB, metric depth, object IDs, target-only masks/truth, actor/camera readbacks, video/audio and manifest |
| Apartment route smoke | `tools/m6y/run_spear_apartment_canary.py` using the selected Apartment runtime profile | fresh native-map visual/runtime evidence independent of the strict Episode finalizer |
| Kujiale adapter smoke | `tools/m6z/run_spear_kujiale_canary.py` | fresh four-view adapter/map/material evidence; this is not a Timeline Episode substitute |
| Kujiale Episode candidate | `tools/m6z/run_spear_residential_episode.py` over an external authorized episode root and generated map | fresh 75-frame route with Timeline, audio and actor/camera readback; it is not an accepted reference until the live map/input path is restored and the route is reviewed |
| M4 named-pair RLR | `python -m avengine.cli m4 run-canary`, then `m4 verify-canary`/`verify-bundle` | one fresh run using the request-declared repeat count and the same request, package, runtime and HRTF inputs |
| Semantic MP3D RIR | `tools/m6x/render_rir_cache.py --semantic-no-file-evidence` with explicit plan/package/simulation/HRTF inputs | one fresh CPU cache with exact plan/job/source/listener/layout semantics and existing native receipt checks |
| CPU Episode semantics | declared strict-two-human builder/execution plan, `tools/qa/bind_native_pixel_fact_episode.py`, full-Episode assignment and dataset-index tools | exact request/Episode IDs, Timeline, audio program, Fact/Question output, deduplication and split semantics |

Use the exact argv declared by the selected request/execution plan when one
exists. Command lines, inputs and output roots are recorded per run rather than
copied into this document as a second authority.

## 7. Preconditions and current blockers

Before a complete pre-migration reference can run, these issues must be closed:

1. MP3D has no accepted strict two-human Habitat-native full-75 visual runner.
   Existing MP3D UE sparse captures are comparison diagnostics and cannot be
   promoted. Implement and review one-Simulator/two-human Habitat execution
   before recording the MP3D reference.
2. The live Kujiale UE content path lacks the required map. The current
   `run_spear_kujiale_canary.py` is a four-view adapter canary, not a full
   Timeline Episode. Rebuild a fresh map from the retained authorized external
   USD/MDL input, use the retained external Episode root, and review the
   75-frame residential runner before recording the reference. The map, USD and
   Episode data stay outside Git.
3. Apartment reference evidence must be regenerated from a suite that carries
   `production_visual` consistently at suite/scenario/plan level; historical
   comparison-role evidence is not rewritten.
4. The current Habitat transition commit is not available from the documented
   USTB Habitat repository revision. Select the required changes from the
   retained server fork before retiring it.
5. The RLR engine-source decision in Section 4 remains unresolved.
6. Selected SPEAR Python/client, extension and UE plugin source plus required
   source-generated camera/render-pass behavior must be inventoried before its
   checkout can be retired.

These are source/reference prerequisites, not new admission gates.

## 8. Fresh pre-migration procedure

Before each native row:

1. Confirm the expected branch/commit and an otherwise clean worktree.
2. Confirm required inputs resolve within their declared authority roots and
   satisfy each runner's existing regular-file and symlink rules.
3. Confirm the output target is absent and not a symlink; do not use resume.
4. Record CPU/GPU/UE availability, GPU index/UUID/utilization/compute processes,
   relevant ports and existing Habitat/RIR/capture processes. Do not kill an
   unknown process.
5. Execute one declared row. Stop on its first error.
6. Preserve the fresh output, exact command, runtime versions, process/device
   receipt and existing validator result.

Use distinct roots such as
`tmp/refactor_equivalence/pre_refactor_<UTC>/<row>/`. These are ordinary,
ignored comparison outputs, not a checked-in baseline or release bundle.

Run order:

1. hermetic unit/schema/CLI checks;
2. CPU Episode semantics;
3. M4 RLR and MP3D semantic-RIR repeats;
4. MP3D Habitat full 75;
5. Apartment strict Episode and route smoke;
6. Kujiale Episode; and
7. cross-row comparison report using existing validators.

## 9. Migration sequence

Use ordinary, reviewable commits on the integration branch:

1. close the reference blockers and record the fresh pre-migration matrix;
2. inventory exact source/config/license dependencies;
3. integrate selected Habitat runtime and binding changes while preserving
   upstream-compatible defaults;
4. integrate the AVEngine/Habitat RLR adapter, headers and small build/config
   surface, then resolve the propagation-engine-source decision;
5. integrate selected SPEAR client, Python extension, UE plugin/control source,
   build helpers and notices; reimplement binary-asset-only behavior in source;
6. point setup/build/runtime resolution at repository-owned source paths;
7. remove sibling-checkout, clone and submodule fallbacks;
8. run hermetic tests after every component batch; and
9. run the complete post-migration matrix.

Do not squash unrelated histories into one opaque import. Each adapted file or
bounded subtree keeps its upstream source/revision mapping. Do not rewrite
historical release evidence.

## 10. Fresh post-migration procedure

Repeat the same declared commands, input identities, room routes and validator
layers using non-existing roots such as
`tmp/refactor_equivalence/post_refactor_<UTC>/<row>/`.

Compare only corresponding production rows:

- MP3D Habitat against MP3D Habitat;
- native Apartment UE against native Apartment UE;
- Kujiale UE/USD against Kujiale UE/USD;
- deterministic AVEngine metadata/assembly exactly; and
- RLR/UE/Habitat numerical or visual behavior through the same existing native
  validators and review procedures.

No post run reuses, resumes or overwrites a pre-run artifact.

## 11. First-failure protocol

- Stop at the first failed command, missing dependency, backend mismatch,
  validator failure or attempted overwrite.
- Preserve the failed output and exact command.
- Classify the failure as source selection, build, runtime, routing, artifact
  structure, deterministic mismatch or native numerical/visual mismatch.
- Repair only that layer and rerun it with another fresh output path.
- Do not continue downstream, weaken a validator, edit evidence or substitute
  another backend to obtain a pass.

## 12. Cutover and merge conditions

Cutover is allowed only when:

- all selected distributable source, small configuration, provenance and
  license notices are in the canonical repository;
- no setup/build/runtime path requires another AVEngine, Habitat, RLR or SPEAR
  Git checkout;
- no UE/data/asset/weight/binary/generated output was added to Git;
- ordinary hermetic tests pass;
- every production-route post run completes fresh and passes its exact and
  existing-native comparisons;
- RLR/CPU native checks pass under the documented source or explicitly approved
  external-library resolution;
- no unresolved first-failure record remains; and
- an independent review finds no P0/P1 issue.

Merge the reviewed integration branch into `main` without rewriting the
existing AVEngine history. Verify a clean checkout of canonical `main` can
configure and build without another product Git repository. Only then may the
old Habitat and SPEAR repositories be marked retired or archived. Their
historical commits and release records remain available as provenance; external
datasets and runtime inputs are not deleted by this source migration.
