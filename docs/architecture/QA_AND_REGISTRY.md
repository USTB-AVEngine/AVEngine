# QA and Registry

## Principle

Successful execution is not dataset approval. Every asset, scene and episode
is admitted only after deterministic checks, required visual review and
provenance validation.

## Result vocabulary

Verification steps use only:

- `pass`: executed and met its acceptance criteria.
- `fail`: executed and violated an acceptance criterion.
- `not_run`: not executed in the recorded environment.
- `blocked`: execution could not start because a declared dependency or input
  was unavailable.

QA measurements may also include non-admission warnings, but warnings never
replace a required hard-gate result.

## QA families

| Family | Representative hard failures |
|---|---|
| Static animal geometry | Missing limbs, floor fusion, severe bridges |
| Skeleton and skinning | Wrong semantic chain, cross-limb weights, invalid sums |
| Animation and contact | Flips, self-intersection, joint limits, sliding, penetration |
| Room geometry | Closed openings, disconnected rooms, ray leakage |
| Acoustic materials | Unassigned production triangles, unintended fallback |
| Multi-source runtime | Identity loss, pair-IR mismatch, order dependence |
| Visual synchronization | Different pose hashes for the same frame |
| AV synchronization | Wrong frame/sample counts or event boundaries |
| Counterfactual integrity | A frozen visual variable changed |
| Provenance | Missing hashes, commits, sources or license decisions |
| Runtime determinism | Same request and seed produce incompatible manifests |

Each report records status, threshold, measured value, worst-case location,
artifact path and failure reason.

## Admission states

Admission uses a separate machine vocabulary from verification results:

- `research_candidate`: retained for research or migration; required evidence
  is not yet complete.
- `research_candidate_pending_revalidation`: mandatory state assigned to an
  imported legacy `approved` record until fresh evidence is executed.
- `canary_qualified`: sufficient for a bounded milestone canary, but not for
  dataset production or release.
- `rejected`: a named hard gate failed; the failure and evidence remain
  addressable.
- `admission_blocked`: a required dependency, rights decision, or review is
  unavailable; this is never equivalent to approval.
- `approved_for_dataset`: all versioned hard gates, rights checks, and required
  reviews passed under the central M6 Dataset Admission contract.

These exact underscore-separated values are the canonical M0 vocabulary. A
rejected, blocked, `research_candidate`, or `canary_qualified` record cannot
resolve through the production asset/sample API. Transitions retain the prior
state and evidence; no importer may synthesize `approved_for_dataset`.

## Provenance minimum

- AVEngine, runtime fork, upstream Habitat and RLR commits.
- Schema, template, scene, action and material revisions.
- Input/output hashes, commands, parameters and random seeds.
- Model and asset sources, licenses, derivative policy and redistribution flag.
- Test environment and the exact `pass/fail/not_run/blocked` record.

## Human review

Human review decisions bind to immutable media and manifest hashes. Rebuilding
an input invalidates the old decision. Visual acceptance cannot override a
failed deterministic safety or rights gate.
