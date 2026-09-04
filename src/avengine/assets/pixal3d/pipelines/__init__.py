"""Selected local-only Pixal3D pipeline entry point."""

from __future__ import annotations

import importlib
from pathlib import Path


__attributes = {"Pixal3DImageTo3DPipeline": "pixal3d_image_to_3d"}
__submodules = ("samplers",)
__all__ = [*__attributes, *__submodules]


def __getattr__(name):
    if name in __attributes:
        module = importlib.import_module(f".{__attributes[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in __submodules:
        value = importlib.import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def from_pretrained(path: str | Path):
    """Load the selected pipeline from an explicit local model root."""
    root = Path(path).expanduser().resolve()
    if not (root / "pipeline.json").is_file():
        raise FileNotFoundError(f"Pixal3D pipeline config is missing locally: {root}")
    pipeline_class = __getattr__("Pixal3DImageTo3DPipeline")
    return pipeline_class.from_pretrained(root)
