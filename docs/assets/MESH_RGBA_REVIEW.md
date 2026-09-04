# Mesh RGBA review previews

`tools/assets/render_mesh_rgba_review.py` renders an existing mesh to a fresh transparent RGBA PNG for visual review. It never writes the input asset or registers a new asset. The renderer is explicitly Cycles on CPU with bounded samples and threads.

The provenance mode is explicit:

- `--source-kind registered_asset` (the default) requires `--asset-id`, a unique runtime registry record, and a matching `source_manifest.json`. The manifest must bind the input path and mesh URI to the same registered record.
- `--source-kind generated_candidate` requires `--pixal-receipt`. The receipt must use an AVEngine Pixal3D schema, have `status: passed`, and declare an `output.path` that resolves exactly to the input GLB. This mode records the result as a generated candidate; it does not claim canonical input, registration, or replacement.

Both the PNG and its JSON sidecar are checked before Blender starts, so an existing review is preserved. The sidecar records `source_kind`, the applicable provenance binding, the actual Cycles engine/device, and the non-registration claims.

Example for a registered asset:

```text
blender --background --factory-startup --python tools/assets/render_mesh_rgba_review.py -- \
  --input /path/registered_mesh.glb \
  --output /path/fresh/registered_front.png \
  --source-kind registered_asset \
  --asset-id example_asset \
  --registry examples/runtime/source_asset_runtime_profiles.json \
  --source-manifest /path/source_manifest.json
```

Example for a Pixal3D candidate:

```text
blender --background --factory-startup --python tools/assets/render_mesh_rgba_review.py -- \
  --input /path/pixal_result.glb \
  --output /path/fresh/pixal_front.png \
  --source-kind generated_candidate \
  --pixal-receipt /path/pixal3d_receipt.json
```
