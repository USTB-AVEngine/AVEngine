# M6 release manifest tool

The release is intentionally assembled in two Git commits:

1. Commit **A** contains all implementation, schemas and documentation. The
   AVEngine and Habitat worktrees must both be clean.
2. `prepare` reads hash-bound formal evidence and atomically creates
   `release/avengine_release_manifest_v1.json` without replacing an existing
   file.
3. Commit only the manifest and its declared `release/` allowlist. This
   single-parent commit is **B**, and `B^` must be A.
4. Create an annotated tag on B and run `verify` from clean worktrees.

The manifest records A, not B, and never records its own hash. B and the tag
are observed from Git by the verifier. This avoids a self-referential hash or
commit fixed point while still binding the complete release state.

The build request is local input and is not part of the release schema. Keep it
under an ignored evidence directory after A exists. Its shape is:

For the two `m6_evidence` role bundles, list only the authority entry
(`evidence.json` or `attempt_manifest.json`). `prepare` runs the corresponding
semantic verifier and expands that entry into an exact per-file closure in the
release manifest: the entry itself plus every declared artifact. Missing,
tampered, duplicate, symlinked or extra undeclared retained files fail before
the manifest is written. In particular, the controlled canary's
`release_manifest_ref.json` is part of the recorded closure, not an unrecorded
side read.

```json
{
  "schema": "avengine_release_build_request_v1",
  "release": {
    "release_id": "avengine_m6_feasibility_v1",
    "tag": "avengine-m6-feasibility-v1",
    "state": "candidate",
    "current_milestone": "M6",
    "manifest_path": "release/avengine_release_manifest_v1.json"
  },
  "repositories": {
    "implementation_commit": "<commit-A>",
    "expected_habitat_commit": "<fork-commit>",
    "upstream_commit": "<upstream-habitat-commit>",
    "expected_rlr_commit": "<RLR-gitlink-commit>",
    "rlr_submodule_path": "src/deps/rlr-audio-propagation",
    "expected_avengine_repository": "https://github.com/Eastforward/AVEngine.git",
    "expected_habitat_repository": "https://github.com/Eastforward/habitat-sim-AVEngine.git",
    "expected_upstream_repository": "https://github.com/facebookresearch/habitat-sim.git",
    "expected_rlr_repository": "https://github.com/facebookresearch/rlr-audio-propagation.git"
  },
  "native_artifacts": {
    "habitat_sim_binding": {
      "root_id": "habitat_runtime",
      "path": "<relative-binding-path>"
    },
    "rlr_binary": {
      "root_id": "habitat_runtime",
      "path": "<relative-RLR-binary-path>"
    }
  },
  "environment": {
    "compiler": {"id": "gcc", "command": "c++"},
    "python_dependencies": ["avengine", "jsonschema", "numpy", "Pillow", "PyYAML"]
  },
  "evidence_bundles": [
    {
      "evidence_id": "m6-controlled-canary-formal",
      "status": "pass",
      "artifacts": [{"root_id": "avengine", "path": "tmp/m6/<run>/evidence.json"}]
    },
    {
      "evidence_id": "m6-room-qualification-formal",
      "status": "pass",
      "artifacts": [{"root_id": "avengine", "path": "tmp/m6/<rooms>/attempt_manifest.json"}]
    },
    {
      "evidence_id": "m6-test-layers-formal",
      "status": "pass",
      "artifacts": [{"root_id": "avengine", "path": "tmp/m6/<tests>/report.json"}]
    }
  ],
  "m6_evidence": {
    "controlled_canary_bundle_id": "m6-controlled-canary-formal",
    "room_qualification_bundle_id": "m6-room-qualification-formal"
  },
  "test_layers": {
    "fast-unit": {"status": "pass", "command": ["..."], "evidence_bundle_ids": ["m6-test-layers-formal"]},
    "slow-hermetic": {"status": "pass", "command": ["..."], "evidence_bundle_ids": ["m6-test-layers-formal"]},
    "native-habitat": {"status": "pass", "command": ["..."], "evidence_bundle_ids": ["m6-test-layers-formal"]},
    "rlr-audio": {"status": "pass", "command": ["..."], "evidence_bundle_ids": ["m6-test-layers-formal"]},
    "blender-assets": {"status": "pass", "command": ["..."], "evidence_bundle_ids": ["m6-test-layers-formal"]},
    "media-readback": {"status": "pass", "command": ["..."], "evidence_bundle_ids": ["m6-test-layers-formal"]},
    "release-canary": {
      "status": "pass",
      "command": ["..."],
      "evidence_bundle_ids": [
        "m6-controlled-canary-formal",
        "m6-room-qualification-formal",
        "m6-test-layers-formal"
      ]
    }
  }
}
```

Prepare and later verify:

```bash
python tools/release/build_manifest.py prepare \
  --request tmp/m6/release_build_request.json \
  --avengine-root . \
  --habitat-runtime-root ../habitat-sim-AVEngine

git add -- release/avengine_release_manifest_v1.json
git commit -m "release: bind M6 candidate metadata"
git tag -a avengine-m6-feasibility-v1 -m "AVEngine M6 feasibility release"

python tools/release/build_manifest.py verify \
  --manifest release/avengine_release_manifest_v1.json \
  --avengine-root . \
  --habitat-runtime-root ../habitat-sim-AVEngine
```

For evidence outside either repository, add the same
`--artifact-root ROOT_ID=/path` mapping to both commands. Paths stored in the
manifest remain root-relative.
