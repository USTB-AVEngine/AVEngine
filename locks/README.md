# Historical Runtime Locks

Files in this directory and root [`runtime.lock.yaml`](../runtime.lock.yaml)
are immutable inputs or receipts for already-recorded milestone evidence. They
answer which bytes an earlier M2/M3/M4/M5 result used; they are not a mutable
description of the newest source tree.

The only current cross-repository release authority is:

```text
release/avengine_release_manifest_v1.json
```

If that file is absent or invalid, release state is pending. Its schema and
verifier can exist before a release candidate, but neither constitutes a
release. A valid manifest must bind the AVEngine implementation commit, Habitat fork commit,
upstream Habitat commit, RLR commit, schema set, native binaries, environment,
test-layer statuses, evidence bundles and annotated release tag.

Rules:

- Do not edit a historical lock to match a newer commit or binary.
- Do not use a historical `pass` as evidence for an unrun current test layer.
- If a historical artifact is retained by a new release, reference its exact
  hash from the release manifest and label its evidence basis explicitly.
- If a runtime changes, create a new release manifest or a new versioned
  historical lock; never rewrite the old record.
- README and roadmap prose may explain evidence, but they never override the
  machine-readable current manifest.

The apparent mismatch between an old lock's milestone name and current code is
therefore expected. It is provenance, not a feature flag or a runtime gate.
