from pathlib import Path
import json
from typing import *

import torch
import torch.nn as nn

from .. import models


class Pipeline:
    """Selected local-only Pixal3D pipeline base."""

    def __init__(self, models: dict[str, nn.Module] = None):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_pretrained(
        cls, path: str | Path, config_file: str = "pipeline.json"
    ) -> "Pipeline":
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Pixal3D model root is missing: {root}")
        config_path = root / config_file
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Pixal3D pipeline config is missing locally: {config_path}"
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        args = config.get("args") if isinstance(config, dict) else None
        model_specs = args.get("models") if isinstance(args, dict) else None
        if not isinstance(model_specs, dict):
            raise ValueError(f"Pixal3D pipeline config lacks args.models: {config_path}")

        loaded = {}
        for name, relative in model_specs.items():
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"invalid local checkpoint path for {name}")
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"checkpoint path must stay lexically under Pixal3D model root: {relative}"
                )
            # Hugging Face snapshots use symlinks from the snapshot directory
            # into its sibling blobs cache. Check traversal lexically, then
            # resolve the declared snapshot link for the local loader.
            model_root = (root / relative_path).resolve()
            loaded[name] = models.from_pretrained(model_root)

        new_pipeline = cls(loaded)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, "_device"):
            return self._device
        for model in self.models.values():
            if hasattr(model, "device"):
                return model.device
        for model in self.models.values():
            if hasattr(model, "parameters"):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
