"""Where shared model weights live, resolved instead of hard-coded.

Weights are shared across repositories and are far too large to vendor, so they
sit in a public directory outside this tree. Baking that directory into a tool as
a module-level default couples the tool to one machine, which is what this
replaces.

Resolution order, first hit wins:

1. the tool's own command-line argument, which always overrides everything;
2. the environment variable named for the entry (``AVENGINE_MODEL_FLUX2_KLEIN``
   for ``flux2_klein``), for a one-off run against a different copy;
3. ``AVENGINE_MODEL_ROOTS`` pointing at a replacement registry file, for a host
   whose whole layout differs;
4. ``examples/assets/model_roots_v1.json`` in this repository.

A missing entry is an error naming the entry and the variable that would supply
it, because a wrong weight silently produces a plausible wrong asset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPOSITORY_ROOT / "examples/assets/model_roots_v1.json"


def registry_path() -> Path:
    override = os.environ.get("AVENGINE_MODEL_ROOTS")
    return Path(override) if override else DEFAULT_REGISTRY


def environment_variable(name: str) -> str:
    return "AVENGINE_MODEL_" + name.upper()


def resolve(name: str, *, override: Path | str | None = None) -> Path:
    """The directory or file for one logical model entry."""
    if override:
        return Path(override)
    from_environment = os.environ.get(environment_variable(name))
    if from_environment:
        return Path(from_environment)
    path = registry_path()
    if not path.is_file():
        raise SystemExit(
            f"model root registry not found at {path}; set "
            f"{environment_variable(name)} or AVENGINE_MODEL_ROOTS")
    entries = json.loads(path.read_text(encoding="utf-8")).get("models", {})
    entry = entries.get(name)
    if not entry:
        raise SystemExit(
            f"no model root named {name!r} in {path}; add it there or set "
            f"{environment_variable(name)}")
    return Path(entry["path"])
