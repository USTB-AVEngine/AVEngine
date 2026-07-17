# Variant package assembler evidence contract

The JSON files in this directory are only variant specifications. A spec or an
upstream source-manifest snapshot is not sufficient evidence for assembly.
Every API call must construct `VariantPackageEvidence` with both of these
separate, regular files:

- `appearance_lineage`: a passing
  `avengine_m2_appearance_variant_lineage_v1` file. Its authenticated
  `pre_rebase_visual_glb` must match `rebase_report.source`, and
  `rebase_report.output` must match the exact `visual_glb` byte size and
  SHA-256.
- `material_normalization_report`: a passing
  `avengine_m2_material_normalization_v2` file with `force_opaque=true`. Its
  output must match the exact `visual_glb` byte size and SHA-256.

Immediately before compilation, the assembler independently parses the actual
final GLB and rejects implicit/default materials or metallic, emissive,
specular, unsupported-extension, and alpha bypasses. Copies embedded inside
`source_manifest` do not replace either required file.

For Beagle L9 lineage, the assembler also reruns the complete producer
spec/lineage validator, including request-derived spec identity, declared
realization operations, topology/skin/action invariants, and output readback.
This strict L9 path currently requires a repository checkout containing
`tools/m2/build_appearance_variant_inputs.py`; it fails closed when that
producer validator is unavailable. The canonical file is loaded by absolute
repository path from its authenticated bytes, so the CLI does not depend on
the caller adding the repository root to `PYTHONPATH`. Cross-species diagnostic
lineage uses its separate validator under `src/avengine/m2`.

The CLI exposes the same contract:

```bash
.venv/bin/python tools/m2/assemble_variant_package.py \
  --spec examples/m2/variant_packages/rocketbox_beagle_review_spec_v1.json \
  --visual-glb /absolute/path/to/final_visual.glb \
  --appearance-lineage /absolute/path/to/appearance_lineage.json \
  --material-normalization-report /absolute/path/to/material_normalization.json \
  --actions-npz /absolute/path/to/actions.npz \
  --rebase-report /absolute/path/to/rebase.json \
  --rebase-deformation-report /absolute/path/to/rebase_deformation.json \
  --action-report /absolute/path/to/action_report.json \
  --static-qa /absolute/path/to/static_geometry.json \
  --deformation-qa /absolute/path/to/deformation.json \
  --animation-qa /absolute/path/to/animation.json \
  --habitat-static-probe /absolute/path/to/habitat_static_probe.json \
  --habitat-animation-review /absolute/path/to/habitat_animation_review.json \
  --contact-phases /absolute/path/to/contact_phases.json \
  --source-manifest /absolute/path/to/source_manifest.json \
  --license-snapshot /absolute/path/to/license_snapshot.json \
  --output /absolute/path/to/new_package_directory
```

The output directory must not already exist.
