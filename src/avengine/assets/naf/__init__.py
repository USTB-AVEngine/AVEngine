"""AVEngine-local inference slice of ValeoAI NAF upsampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .naf import NAF


def load_pretrained(path: str | Path, *, device: str | torch.device = "cpu") -> NAF:
    """Load NAF weights from an explicit local file without network fallback."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"NAF checkpoint is missing: {checkpoint_path}")
    model = NAF().to(device)
    state: Any = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"NAF checkpoint is not a state-dict: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


__all__ = ["NAF", "load_pretrained"]
