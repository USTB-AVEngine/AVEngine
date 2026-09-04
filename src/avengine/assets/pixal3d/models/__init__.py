"""Selected Pixal3D inference model registry (local checkpoints only)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from safetensors.torch import load_file


__attributes = {
    "SparseStructureEncoder": "sparse_structure_vae",
    "SparseStructureDecoder": "sparse_structure_vae",
    "SparseStructureFlowModel": "sparse_structure_flow",
    "SLatFlowModel": "structured_latent_flow",
    "ElasticSLatFlowModel": "structured_latent_flow",
    "SparseUnetVaeEncoder": "sc_vaes.sparse_unet_vae",
    "SparseUnetVaeDecoder": "sc_vaes.sparse_unet_vae",
    "FlexiDualGridVaeEncoder": "sc_vaes.fdg_vae",
    "FlexiDualGridVaeDecoder": "sc_vaes.fdg_vae",
}

__all__ = list(__attributes)


def __getattr__(name):
    if name not in __attributes:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{__attributes[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def from_pretrained(path: str | Path, **kwargs):
    """Load one model from a local JSON/safetensors pair without network fallback."""
    root = Path(path).expanduser().resolve()
    config_file = Path(f"{root}.json")
    model_file = Path(f"{root}.safetensors")
    if not config_file.is_file() or not model_file.is_file():
        raise FileNotFoundError(
            f"Pixal3D model checkpoint pair is missing locally: {config_file} "
            f"and {model_file}"
        )
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("name"), str):
        raise ValueError(f"invalid Pixal3D model config: {config_file}")
    model = __getattr__(config["name"])(**config.get("args", {}), **kwargs)
    model.load_state_dict(load_file(str(model_file)), strict=False)
    return model
