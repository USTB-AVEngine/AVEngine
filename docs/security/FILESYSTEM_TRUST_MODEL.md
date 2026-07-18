# Filesystem trust model

AVEngine's default filesystem mode is `trusted_research_workspace`. It is for
a controlled research server where the operator chooses the repository,
dataset and output roots. It prevents accidental path escape and replacement
of immutable evidence; it is not an anonymous-upload security boundary.

## Guarantees

All formal M6 entrypoints use `WorkspacePathPolicy` with explicit roots. The
policy canonicalizes absolute paths, follows ordinary symlinks, and then
requires the resolved target to remain beneath one declared root. It rejects
missing or empty file inputs, root escape, malformed or mismatched SHA-256
bindings and any pre-existing output name. Evidence files are written to a
sibling temporary file and committed with a no-clobber hard link. Complete
directories are staged beside their destination and published with Linux
`renameat2(RENAME_NOREPLACE)`.

A dataset's convenience symlink is valid when its resolved target is still
inside a declared dataset root. A symlink that resolves outside all declared
roots is rejected. This distinction allows versioned local datasets such as
ReplicaCAD without treating every symlink as hostile.

## Non-guarantees

This mode does not claim to resist a malicious local process racing symlink or
mount changes. It does not claim complete TOCTOU resistance, portable
directory `O_NOFOLLOW` behavior, equal Windows/Linux semantics or containment
of arbitrary subprocesses. Inputs are authenticated at the time recorded;
the operator is responsible for preventing concurrent hostile mutation.

`strict_untrusted_linux` is a reserved future name, not an implemented mode.
It may be enabled only after path resolution is centralized around Linux
`openat2()` with `RESOLVE_BENEATH` and `RESOLVE_NO_SYMLINKS`, all subsequent IO
uses held descriptors, and dedicated Linux race/integration tests pass.

## Operational rules

- Declare the narrowest useful roots in workspace configuration.
- Bind formal input bytes by SHA-256 and retain the resolved canonical path.
- Never reuse an evidence output directory. Choose a new run ID.
- Build a bundle in a hidden sibling staging directory, fsync critical files,
  verify it, then publish it once with no replacement.
- Treat a policy error as `fail` for malformed input and `blocked` only when a
  required external root or platform facility is genuinely unavailable.
