import os
import argparse
import math
import time
import tempfile
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# This entry accepts only explicit local model roots; avoid implicit hub or
# dataset downloads when it is invoked directly rather than through the
# AVEngine wrapper.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# SDPA is available in supported PyTorch builds; flash attention remains an explicit opt-in.
os.environ.setdefault("ATTN_BACKEND", "sdpa")
# Keep autotuning state outside the source tree so a relocated/read-only
# checkout remains a valid production entry. Override explicitly when a
# persistent cache is desired.
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(tempfile.gettempdir(), "avengine_pixal3d_autotune_cache.json"),
)
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'


# ============================================================================
# Constants & Defaults
# ============================================================================

MODEL_PATH = None
MOGE_MODEL_NAME = None

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": None,
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": None,
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": None,
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": None,
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

# ============================================================================
# Model Loading
# ============================================================================

class Pixal3DLocalModelError(RuntimeError):
    """A required Pixal3D model must be supplied as a local path."""


def _require_local_path(value: str | Path | None, *, label: str, directory: bool) -> Path:
    if value is None:
        raise Pixal3DLocalModelError(
            f"{label} must be supplied as a local path; remote model IDs are disabled"
        )
    path = Path(value).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise Pixal3DLocalModelError(f"{label} {kind} is missing: {path}")
    return path


def build_image_cond_model(
    config: dict,
    *,
    dinov3_model_path: str | Path,
    naf_model_path: str | Path,
):
    from .trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )

    local_config = dict(config)
    local_config["model_name"] = str(
        _require_local_path(
            dinov3_model_path, label="DINOv3 model root", directory=True
        )
    )
    if local_config.get("use_naf_upsample"):
        local_config["naf_model_path"] = str(
            _require_local_path(
                naf_model_path, label="NAF checkpoint", directory=False
            )
        )
    model = DinoV3ProjFeatureExtractor(**local_config)
    model.eval()
    return model


def load_moge_model(*, device="cuda", model_name: str | Path):
    from moge.model.v2 import MoGeModel

    local_path = _require_local_path(
        model_name, label="MoGe model root", directory=True
    )
    checkpoint_path = local_path / "model.pt"
    if not checkpoint_path.is_file():
        raise Pixal3DLocalModelError(
            f"MoGe checkpoint is missing locally: {checkpoint_path}"
        )
    # MoGe's loader accepts a local checkpoint file; a local path bypasses its
    # Hugging Face download branch.
    moge_model = MoGeModel.from_pretrained(
        str(checkpoint_path), local_files_only=True
    )
    moge_model = moge_model.to(device)
    moge_model.eval()
    return moge_model


def init_pipeline(
    model_path: str | Path | None,
    *,
    dinov3_model_path: str | Path | None,
    naf_model_path: str | Path | None,
    device="cuda",
    low_vram=False,
):
    # Import the local pipeline only when a GPU pipeline is actually being
    # built. Its optional flex_gemm backend initializes a CUDA driver at import.
    from .pipelines import Pixal3DImageTo3DPipeline

    local_model_path = _require_local_path(
        model_path, label="Pixal3D model root", directory=True
    )
    local_dinov3_path = _require_local_path(
        dinov3_model_path, label="DINOv3 model root", directory=True
    )
    local_naf_path = _require_local_path(
        naf_model_path, label="NAF checkpoint", directory=False
    )
    print(f"[Pipeline] Loading from {local_model_path}...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(local_model_path)

    print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(
        IMAGE_COND_CONFIGS["ss"],
        dinov3_model_path=local_dinov3_path,
        naf_model_path=local_naf_path,
    )
    pipeline.image_cond_model_shape_512 = build_image_cond_model(
        IMAGE_COND_CONFIGS["shape_512"],
        dinov3_model_path=local_dinov3_path,
        naf_model_path=local_naf_path,
    )
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(
        IMAGE_COND_CONFIGS["shape_1024"],
        dinov3_model_path=local_dinov3_path,
        naf_model_path=local_naf_path,
    )
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(
        IMAGE_COND_CONFIGS["tex_1024"],
        dinov3_model_path=local_dinov3_path,
        naf_model_path=local_naf_path,
    )

    if low_vram:
        # Low-VRAM mode: models stay on CPU, loaded to GPU on-demand per stage.
        # Peak VRAM = one flow model + one DinoV3, not all ~18 GB at once.
        print("[NAF] Preloading local NAF upsampler weights (CPU only)...")
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
        print("[Pipeline] Low-VRAM mode enabled.")
    else:
        # Standard mode: all models loaded to GPU at once (faster, needs more VRAM).
        pipeline.low_vram = False
        pipeline.cuda()
        pipeline.image_cond_model_ss.cuda()
        pipeline.image_cond_model_shape_512.cuda()
        pipeline.image_cond_model_shape_1024.cuda()
        pipeline.image_cond_model_tex_1024.cuda()
        print("[NAF] Pre-loading NAF upsampler model...")
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        print("[Pipeline] Standard mode (all models on GPU).")

    return pipeline

# ============================================================================
# Camera Estimation
# ============================================================================

def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    y_ndc = -(yt - image_resolution / 2.0)
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def get_camera_params_wild_moge(image_path, moge_model, device="cuda", mesh_scale=1.0, extend_pixel=0, image_resolution=512):
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        camera_angle_x, grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale, image_resolution
    )["distance_from_x"]
    return {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}

# ============================================================================
# Main Inference
# ============================================================================

def run_inference(
    image_path: str,
    output_path: str,
    seed: int = 42,
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_sampling_steps: int = 12,
    ss_rescale_t: float = 5.0,
    shape_slat_guidance_strength: float = 7.5,
    shape_slat_guidance_rescale: float = 0.5,
    shape_slat_sampling_steps: int = 12,
    shape_slat_rescale_t: float = 3.0,
    tex_slat_guidance_strength: float = 1.0,
    tex_slat_guidance_rescale: float = 0.0,
    tex_slat_sampling_steps: int = 12,
    tex_slat_rescale_t: float = 3.0,
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
    max_num_tokens: int = 49152,
    model_path: str | Path | None = None,
    moge_model_path: str | Path | None = None,
    dinov3_model_path: str | Path | None = None,
    naf_model_path: str | Path | None = None,
    manual_fov: float = -1.0,
    low_vram: bool = False,
    resolution: int = -1,
    pipeline=None,
):
    # Batch workers may pass an already loaded pipeline so multiple assets can
    # share the same model residency. Standalone callers retain the original
    # behavior and initialize the pipeline here.
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise Pixal3DLocalModelError(
            f"refusing to replace existing output: {output}"
        )
    if pipeline is None:
        pipeline = init_pipeline(
            model_path,
            dinov3_model_path=dinov3_model_path,
            naf_model_path=naf_model_path,
            low_vram=low_vram,
        )
    else:
        print("[Pipeline] Reusing caller-provided persistent pipeline.")

    # The AVEngine runner has already validated the non-opaque alpha channel.
    # MoGe is loaded afterwards so both never occupy VRAM at the same time.
    print(f"[Inference] Processing image: {image_path}")
    img = Image.open(image_path)
    image_preprocessed = pipeline.preprocess_image(img)

    # Save preprocessed image for MoGe
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), f"_tmp_preprocessed_{int(time.time()*1000)}.png")
    image_preprocessed.save(tmp_path)

    # Camera estimation
    if manual_fov > 0:
        # Use manually specified FOV (in radians)
        camera_angle_x = float(manual_fov)
        grid_point = torch.tensor([-1.0, 0.0, 0.0])
        distance = distance_from_fov(
            camera_angle_x, grid_point,
            torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
            mesh_scale, image_resolution
        )["distance_from_x"]
        camera_params = {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}
        print(f"[Inference] Using manual FOV: {math.degrees(manual_fov):.2f}° ({manual_fov:.4f} rad), distance={distance:.4f}")
    else:
        print("[MoGe-2] Loading model for camera estimation...")
        moge_model = load_moge_model(
            device="cuda", model_name=moge_model_path
        )
        print("[Inference] Estimating camera parameters...")
        camera_params = get_camera_params_wild_moge(
            tmp_path, moge_model, device="cuda",
            mesh_scale=mesh_scale, extend_pixel=extend_pixel,
            image_resolution=image_resolution,
        )
        print(f"  camera_angle_x={camera_params['camera_angle_x']:.4f}, distance={camera_params['distance']:.4f}")
        # MoGe is only needed for camera estimation; free its VRAM for inference.
        moge_model.cpu()
        del moge_model
        torch.cuda.empty_cache()
    os.remove(tmp_path)

    # Run pipeline
    print("[Inference] Running 3D generation pipeline...")
    torch.manual_seed(seed)

    ss_sampler_override = {
        "steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
        "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t,
    }
    shape_sampler_override = {
        "steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
        "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t,
    }
    tex_sampler_override = {
        "steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t,
    }

    pipeline_type = f"{resolution if resolution > 0 else (1024 if low_vram else 1536)}_cascade"
    print(f"[Inference] Using pipeline_type={pipeline_type}")
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        image_preprocessed,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params=ss_sampler_override,
        shape_slat_sampler_params=shape_sampler_override,
        tex_slat_sampler_params=tex_sampler_override,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )

    mesh = mesh_list[0]

    # Extract GLB
    print("[Inference] Extracting GLB...")
    import o_voxel
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000, texture_size=4096,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
    )

    # Apply rotation
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)

    # Export
    output.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(output), extension_webp=True)
    print(f"[Done] GLB saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVEngine Pixal3D inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="./output.glb", help="Output GLB file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fov", type=float, default=-1.0,
                        help="Manual camera FOV in radians; auto mode uses local MoGe")
    parser.add_argument("--model-root", type=str, required=True,
                        help="Local Pixal3D model snapshot root")
    parser.add_argument("--moge-root", type=str, required=True,
                        help="Local MoGe model snapshot root")
    parser.add_argument("--dinov3-root", type=str, required=True,
                        help="Local DINOv3 model snapshot root")
    parser.add_argument("--naf-root", type=str, required=True,
                        help="Local NAF checkpoint file")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable the upstream low-VRAM execution mode")
    parser.add_argument("--resolution", type=int, default=-1,
                        help="Pipeline resolution (1024 or 1536)")

    args = parser.parse_args()
    run_inference(
        image_path=args.image,
        output_path=args.output,
        seed=args.seed,
        manual_fov=args.fov,
        model_path=args.model_root,
        moge_model_path=args.moge_root,
        dinov3_model_path=args.dinov3_root,
        naf_model_path=args.naf_root,
        low_vram=args.low_vram,
        resolution=args.resolution,
    )
