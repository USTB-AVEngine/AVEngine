# Historical Runtime Profiles

Root [`runtime.lock.yaml`](../runtime.lock.yaml) is a lightweight,
Git-tracked compatibility index. It contains no test outcomes, native-binary
hashes or duplicated evidence identities. M1--M4 consumers resolve one explicit
milestone profile from that index:

- [`m1_runtime_v1.yaml`](m1_runtime_v1.yaml)
- [`m2_runtime_v1.yaml`](m2_runtime_v1.yaml)
- [`m3_runtime_v1.yaml`](m3_runtime_v1.yaml)
- [`m4_runtime_v1.json`](m4_runtime_v1.json)

The M1--M3 files preserve the exact historical lock bytes used by already
recorded evidence. Their old status/test fields are archival compatibility data,
not current project status. Do not copy those fields into the root index,
README prose or a new release manifest. M4 already used a dedicated bounded
runtime profile.

The only current cross-repository release authority is:

```text
release/avengine_release_manifest_v1.json
```

If that file is absent or invalid, release state is pending. A valid manifest
binds current repository commits, environment versions and only those external
result-changing bundles required by the release.

Identity rules:

- Git commits identify checked-in source, schemas, configuration and the
  historical profile files themselves. Do not repeat their content hashes in
  the root index or human-facing status pages.
- Environment records use explicit tool/package versions.
- External assets, generated closures, precompiled native artifacts and formal
  evidence each expose one authoritative logical bundle identity. Leaf hashes
  may remain inside that bundle's machine-readable closure but are not copied
  into unrelated locks.
- Do not use an archived `pass` as evidence for an unrun current test layer.
- Do not rewrite a historical profile to match a newer commit or binary.
  Current work belongs in the release manifest or a newly versioned profile.
- README and roadmap prose may explain scope, but never override the current
  manifest or promote historical evidence.

The mismatch between an archived profile and current code is expected. These
profiles are compatibility inputs, not feature flags or current runtime gates.
