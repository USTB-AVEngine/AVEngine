# Pixal3D inference closure

Status: AVEngine-local adapted inference source, 2026-09-04.

`tools/assets/run_pixal3d_mesh.py` now loads the selected Pixal3D inference
closure from `src/avengine/assets/pixal3d` and the selected local NAF
upsampler from `src/avengine/assets/naf`. It never adds an external Git
checkout to `sys.path` and never executes an external `inference.py` with
`runpy`.

## Selected source

The source carrier was the retained Pixal3D working tree. Its checkout Git
pointer was unavailable during this migration, so no source commit is asserted
for the selected files. The package keeps the upstream package layout for the
runtime model, sparse modules, attention, pipeline, representations, samplers,
and projection feature extractor.

Selected Pixal3D source areas:

- `inference.py` adapted as `pixal3d/inference.py`.
- `models/` model registry plus the sparse structure, SLat, and SC-VAE model
  implementations used by the cached `pipeline.json` checkpoint entries.
- `modules/` sparse tensors, sparse convolution/attention/transformer
  implementations, normalization, spatial helpers, and utilities used by the
  selected models.
- `pipelines/` local-only pipeline base, Pixal3D pipeline, and Euler/guidance
  samplers.
- `representations/` mesh and voxel representations required by GLB export.
- `trainers/flow_matching/mixins/image_conditioned_proj.py` projection DINOv3
  extractor used by inference.
- `utils/` only the local utility modules needed by the selected inference
  closure.

Training datasets, training-only trainer modules, Trellis2 entry points, the
unused rembg implementation, and renderer/data-toolkit files are excluded.
The input is required to be an already-matted non-opaque RGBA image, so the
remote background-removal model in the upstream pipeline is not constructed.

## Local model inputs

`examples/assets/model_roots_v1.json` names the local Pixal3D, MoGe, DINOv3,
and NAF roots. The runner checks every directory/file before importing the
inference closure. The local pipeline and model loaders reject missing
checkpoint files instead of falling back to Hugging Face.

The Pixal3D pipeline checkpoint uses the cached `TencentARC/Pixal3D` snapshot;
MoGe and DINOv3 are installed model inputs, and NAF is loaded from the local
checkpoint through the migrated five-file NAF source slice. Installed Torch,
natten, flex_gemm, cumesh, o_voxel, MoGe, and other binary/SDK dependencies
remain external runtime dependencies; they are not copied into this source
tree.

## License and provenance

Pixal3D source is covered by its MIT license and upstream NOTICE. The selected
files retain those terms through `LICENSES/PIXAL3D-MIT.txt` and
`LICENSES/PIXAL3D-NOTICE.txt`. The NOTICE also lists DINOv2 under Apache-2.0
and TRELLIS.2, Direct3D-S2, and MoGe under MIT; those terms remain applicable
to the corresponding model/runtime components.

The NAF source slice is covered by the Apache-2.0 text in
`LICENSES/NAF-APACHE-2.0.txt`. Its `rope.py` retains the upstream DINOv3 license in
`LICENSES/DINOV3-LICENSE.md` and its license header; both must be kept. Model-card terms and training
data provenance remain separate from source-code licenses.

Validation is limited to local package imports without model construction,
source syntax, model-root path checks, and fresh-output/alpha checks. No GPU
inference, model download, or UE process is started by the focused tests.
## Runtime dependencies and relocation

The production entry requires an installed inference environment containing
the modules PyTorch, torchvision, transformers, diffusers, safetensors, einops,
Pillow, OpenCV, and the installed MoGe package (moge.model.v2), plus the
installed native/runtime packages o_voxel, cumesh, flex_gemm, and natten.
xformers and flash-attn are optional acceleration backends; the local defaults
use PyTorch SDPA. spconv/torchsparse are optional only when an explicit sparse
convolution backend selects them. The AVEngine runner sets the Hugging Face
and Datasets offline flags, and all model loaders receive validated local
paths.

The model-root registry has host-local defaults but every runner supports
per-root CLI or AVENGINE_MODEL_* overrides. Pixal3D autotuning state defaults
to the system temporary directory (or AVENGINE_PIXAL3D_AUTOTUNE_CACHE), never
to the source checkout. The runner and animal chain resolve source paths from
their own script locations and use an explicit runtime Python for all Python
steps.
