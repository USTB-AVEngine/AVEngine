"""Local SkinTokens/TokenRig checkpoint and configuration resolution.

This module keeps the upstream checkpoint format while making every runtime
input explicit: source code is this AVEngine package, model weights are resolved
from the AVEngine model-root registry, and the copied skeleton YAML is used
instead of a checkout-relative path.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_RELATIVE = Path(
    "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
)
VAE_RELATIVE = Path("experiments/skin_vae_2_10_32768/last.ckpt")


@dataclass(frozen=True)
class SkinTokensPaths:
    model_root: Path
    checkpoint: Path
    vae_checkpoint: Path
    qwen_root: Path
    skeleton_root: Path


def resolve_paths(model_root: Path | str, qwen_root: Path | str) -> SkinTokensPaths:
    """Validate local model roots and return concrete checkpoint paths."""
    model_root = Path(model_root).expanduser().resolve()
    qwen_root = Path(qwen_root).expanduser().resolve()
    if not model_root.is_dir():
        raise ValueError(f"SkinTokens model root is missing: {model_root}")
    if not qwen_root.is_dir():
        raise ValueError(f"Qwen model root is missing: {qwen_root}")
    checkpoint = model_root / CHECKPOINT_RELATIVE
    vae_checkpoint = model_root / VAE_RELATIVE
    if not checkpoint.is_file():
        raise ValueError(f"SkinTokens checkpoint is missing: {checkpoint}")
    if not vae_checkpoint.is_file():
        raise ValueError(f"SkinTokens VAE checkpoint is missing: {vae_checkpoint}")
    qwen_config = qwen_root / "config.json"
    if not qwen_config.is_file():
        raise ValueError(f"Qwen config is missing: {qwen_config}")
    skeleton_root = Path(__file__).resolve().parent / "configs" / "skeleton"
    for name in ("mixamo.yaml", "vroid.yaml"):
        if not (skeleton_root / name).is_file():
            raise ValueError(f"local SkinTokens skeleton config is missing: {skeleton_root / name}")
    return SkinTokensPaths(
        model_root=model_root,
        checkpoint=checkpoint,
        vae_checkpoint=vae_checkpoint,
        qwen_root=qwen_root,
        skeleton_root=skeleton_root,
    )


def _replace_skeleton_paths(config: dict[str, Any], skeleton_root: Path) -> None:
    for key in ("order", "order_config"):
        section = config.get(key)
        if not isinstance(section, dict):
            continue
        skeleton_path = section.get("skeleton_path")
        if isinstance(skeleton_path, dict):
            section["skeleton_path"] = {
                name: str(skeleton_root / Path(value).name)
                for name, value in skeleton_path.items()
            }


def checkpoint_configs(
    paths: SkinTokensPaths,
    *,
    has_skin: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read and adapt only checkpoint hyperparameters needed by inference.

    The released predict transform contains a skin vertex group because its
    original dataloader primarily handled already rigged examples. A raw mesh
    has no skin to sample; for that case the vertex group is omitted while
    retaining the trained affine/normalization and surface sampling stages.
    This lets TokenRig generate the skeleton and skin for the raw mesh.
    """
    import torch

    checkpoint = torch.load(
        paths.checkpoint,
        map_location="meta",
        weights_only=False,
    )
    try:
        hparams = checkpoint["hyper_parameters"]
        model_config = deepcopy(hparams["model_config"])
        transform_config = deepcopy(hparams["transform_config"])
        tokenizer_config = deepcopy(hparams["tokenizer_config"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"SkinTokens checkpoint has incomplete hyperparameters: {exc}") from exc

    model_config["pretrained_vae"] = str(paths.vae_checkpoint)
    llm_config = model_config.get("llm")
    if not isinstance(llm_config, dict):
        raise ValueError("SkinTokens checkpoint model_config.llm is missing")
    llm_config["pretrained_model_name_or_path"] = str(paths.qwen_root)
    llm_config["local_files_only"] = True
    llm_config["attn_implementation"] = "sdpa"
    mesh_encoder = model_config.get("mesh_encoder")
    if isinstance(mesh_encoder, dict):
        # The local source has an SDPA fallback; flash-attn is an optional
        # acceleration and must not become a hidden dependency.
        mesh_encoder["flash"] = bool(mesh_encoder.get("flash", False))

    predict = transform_config.get("predict_transform")
    if not isinstance(predict, dict):
        raise ValueError("SkinTokens checkpoint transform_config.predict_transform is missing")
    _replace_skeleton_paths(predict, paths.skeleton_root)
    if not has_skin:
        predict.pop("vertex_groups", None)

    _replace_skeleton_paths(tokenizer_config, paths.skeleton_root)
    return model_config, transform_config, tokenizer_config


def load_model(
    paths: SkinTokensPaths,
    *,
    device: str = "cpu",
    has_skin: bool = False,
):
    """Construct and load TokenRig from the local checkpoint."""
    from .model.tokenrig import TokenRig

    model_config, transform_config, tokenizer_config = checkpoint_configs(
        paths,
        has_skin=has_skin,
    )
    model = TokenRig.load_from_system_checkpoint(
        checkpoint_path=str(paths.checkpoint),
        model_config=model_config,
        transform_config=transform_config,
        tokenizer_config=tokenizer_config,
        strict=True,
    )
    model = model.to(device)
    model.eval()
    return model, transform_config, tokenizer_config
