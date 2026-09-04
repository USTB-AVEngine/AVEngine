# SkinTokens inference closure

Status: AVEngine-local adapted TokenRig inference source, 2026-09-04.

tools/assets/run_skintokens_rig.py runs the released VAST-AI SkinTokens
checkpoint through source under src/avengine/assets/skintokens. It has one
mesh input and one fresh output. The model stage uses the local checkpoint and
Qwen configuration roots from examples/assets/model_roots_v1.json; the
Blender stage is a separate single-request HTTP server over a per-job Unix
socket, with a 0700 private directory and 0600 socket. It opens no TCP listener;
filesystem permissions restrict the trusted pickle transport to the invoking
user's processes. The runner rejects --rigger-root and the obsolete --port,
removes inherited PYTHONPATH
for the Blender child, and never executes an upstream checkout, Gradio app,
training dataloader, or network model download.

## Upstream and selected closure

The upstream source is
https://github.com/VAST-AI-Research/SkinTokens at
273b691d35989d71cd17ff2895fdc735097b92d1. The selected source follows the
released src/model, src/tokenizer, src/data transform/sampling, and
src/rig_package runtime paths needed by TokenRig.load_from_system_checkpoint
and GLB import/export. The model checkpoint is the external VAST-AI/SkinTokens
snapshot; the transformer is constructed from the explicit local Qwen3-0.6B
config root. Checkpoint-relative VAE and skeleton paths are rewritten to
their validated local roots before model construction.

demo.py, the Gradio UI, training datasets, training-only data paths, and
server/model-selection code are not part of the execution path. The
AVEngine-owned runner keeps the original raw-mesh skin-generation path: it
does not replace generation with an existing rigged GLB. For a raw mesh with
no skin, the prediction transform drops only the checkpoint's skin vertex
group, which otherwise assumes an already rigged input, and retains affine,
normalization, and surface sampling.

The Blender-side parser uses the upstream asset representation and NumPy
normals. Blender nearest-neighbor transfer uses its built-in mathutils KDTree;
when neither that index nor scipy is available, the generic NumPy fallback
visits both dimensions in fixed tiles rather than building an all-pairs array. Voxel postprocessing remains
explicitly dependency-gated; --use-postprocess fails with the missing
dependency named rather than silently changing the output.

## License and model-data boundary

The selected upstream source is MIT-licensed; the text is retained at
LICENSES/SKINTOKENS-MIT.txt. The model card's training-data description
contains ArticulationXL 2.0, VRoid Hub, and ModelsResource sources. That
description is evidence about model provenance, not a clearance of those
datasets or of generated assets. Keep the checkpoint, Qwen config/weights,
input media, and outputs outside Git under their respective terms.

CPU checkpoint loading was verified with CUDA_VISIBLE_DEVICES empty:
SkinVAEModel loaded 252 state keys and TokenRig loaded 672 state keys with
strict=True and no missing or unexpected keys. Blender 4.2.1 loaded the actual
Border Collie mesh (48,885 vertices and 68,576 faces) through the private Unix
socket; directory/socket modes and cleanup were checked. Both recovered animal
meshes passed 54,000-point preprocessing, and the Blender KDTree was exercised
at 80,000 queries against 54,000 reference points. GPU rig inference remains
unverified at this stage.

## Runtime dependencies

The model process requires the host inference environment's PyTorch,
transformers, diffusers, einops, PyYAML, numpy, scipy, and trimesh packages.
Lightning and flash-attn are optional because the selected source carries
inference fallbacks. The Blender process supplies bpy and mathutils; it does
not require the model environment's scipy or trimesh. Open3D is required only
when the explicit voxel postprocess option is used. Checkpoint, VAE, Qwen
configuration, and skeleton paths are all resolved from explicit local roots.
