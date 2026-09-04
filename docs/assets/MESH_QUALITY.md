# Bounded mesh quality measurement

`src/avengine/assets/mesh_quality.py` and `tools/assets/inspect_mesh_quality.py` provide a read-only geometry measurement for triangle GLBs. The checker reports vertex and face counts, tiny faces, connected components, small-component counts, the largest-component fraction, finite bounds, and whether an explicitly supplied support-plane manifest exists.

The connected-component pass uses a chunked union-find over face indices. Temporary face geometry is bounded by `--chunk-size`; it never creates a vertices-by-vertices adjacency matrix. The checker does not delete components, rewrite the GLB, or infer a support plane from vertex height.

Without `--quality-policy`, the report status is always `measured_unclassified`. The descriptive cutoffs for tiny faces and small components are recorded in the report and do not classify an asset. To classify a result, pass a JSON policy whose `asset_category` and limits were chosen for that asset category:

```json
{
  "schema": "avengine_mesh_quality_policy_v1",
  "asset_category": "owner_selected_category",
  "measurement": {
    "tiny_face_area_threshold": 0.0,
    "small_component_max_faces": 1
  },
  "limits": {
    "max_tiny_faces": 0,
    "max_small_component_count": 0,
    "max_small_component_faces": 0,
    "min_largest_component_fraction": 0.0,
    "require_support_plane": false
  }
}
```

A policy is optional and has no repository-wide default limits. Declared limits produce `pass` or `review_required`; malformed policies fail closed. A support plane is counted as present only when the caller supplies `--support-plane-manifest` and that file is a readable JSON object.

Example measurement:

```text
PYTHONPATH=src python tools/assets/inspect_mesh_quality.py \
  --input /path/mesh.glb \
  --output /path/fresh/mesh_quality.json \
  --tiny-area-threshold 1e-12 \
  --small-component-max-faces 10 \
  --chunk-size 100000
```

Reports include the resolved input path, the measurement parameters, the process peak RSS, and explicit `input_modified: false` / `components_deleted: false` records. Generated media and reports belong under the configured external `tmp` storage path; they are not source assets or registration evidence.
