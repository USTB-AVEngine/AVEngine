"""Launch settings for AVEngine's external SPEAR/UE runtime adapter."""

from __future__ import annotations

import os
from pathlib import Path

# Copyright (c) 2025 The SPEAR Development Team
# Copyright (c) 2022 Intel
# SPDX-License-Identifier: MIT
#
# Behavior reimplemented from
# spear-sim/spear@251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7,
# examples/render_in_apartment.py::parallel_instance_settings.
# The upstream MIT text is retained at LICENSES/SPEAR-MIT.txt.


def _is_within_git_checkout(path: Path) -> bool:
    """Return whether a lexical path sits below a worktree marker."""

    for directory in (path.parent, *path.parent.parents):
        marker = directory / ".git"
        if marker.is_dir() or marker.is_file():
            return True
    return False


def validate_current_production_spear_executable(
    spear_executable: Path,
) -> Path:
    """Validate an external packaged game at a current-production launch edge."""

    lexical_path = Path(spear_executable).expanduser().absolute()
    if not lexical_path.is_file():
        raise RuntimeError(
            "current production SPEAR executable is missing or not a regular file: "
            f"{lexical_path}"
        )
    if not os.access(lexical_path, os.X_OK):
        raise RuntimeError(
            f"current production SPEAR executable is not executable: {lexical_path}"
        )
    if _is_within_git_checkout(lexical_path):
        raise RuntimeError(
            "current production SPEAR executable lexical path is inside a Git "
            f"checkout: {lexical_path}"
        )
    resolved_path = lexical_path.resolve()
    if _is_within_git_checkout(resolved_path):
        raise RuntimeError(
            "current production SPEAR executable resolved path is inside a Git "
            f"checkout: {resolved_path}"
        )
    return resolved_path


def parallel_instance_settings(
    rpc_port: object, graphics_adapter: object | None = None
) -> dict[str, int | str | None]:
    """Return collision-free SPEAR/UE process settings for one render worker."""

    port = int(rpc_port)
    if port < 1024 or port > 65535:
        raise ValueError(f"rpc_port must be in [1024, 65535], got {port}")

    adapter = None
    if graphics_adapter is not None:
        adapter = int(graphics_adapter)
        if adapter < 0:
            raise ValueError(
                f"graphics_adapter must be non-negative, got {adapter}"
            )

    return {
        "rpc_port": port,
        "graphics_adapter": adapter,
        "temp_dir": f"tmp/spear_instance_{port}",
        "log": f"SpearSim_rpc_{port}.log",
        "shared_memory_initial_unique_id": port * 10000,
    }
