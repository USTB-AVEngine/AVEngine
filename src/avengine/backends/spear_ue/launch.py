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


def instance_settings_for_lease(lease: object) -> dict[str, int | str | None]:
    """Build one instance's collision-free settings from a granted lease.

    The allocator owns which port and which graphics device this worker got;
    this function only turns that into the upstream SPEAR settings, so the
    port that isolates the temp directory, the log file and the shared-memory
    id is the same port the allocator reserved and probed.
    """

    port = getattr(lease, "rpc_port", None)
    if port is None:
        raise ValueError(
            "this lease reserved no RPC port; a SPEAR instance needs one, so "
            "its backend profile must declare needs_rpc_port"
        )
    return parallel_instance_settings(
        port, graphics_adapter=getattr(lease, "device_index", None)
    )


def launch_arguments_for_lease(lease: object) -> dict[str, int | None]:
    """The two values a granted lease decides for an external game launch.

    ``launch_external_game_instance`` already takes ``rpc_port`` and
    ``graphics_adapter`` and passes them to ``parallel_instance_settings``, so
    a runner that has a lease only needs to hand these through. Nothing has to
    be renamed or replaced for the allocator's choice to take effect.
    """

    port = getattr(lease, "rpc_port", None)
    if port is None:
        raise ValueError(
            "this lease reserved no RPC port; a SPEAR instance needs one, so "
            "its backend profile must declare needs_rpc_port"
        )
    return {
        "rpc_port": int(port),
        "graphics_adapter": getattr(lease, "device_index", None),
    }


def describe_instance_isolation(
    settings: dict[str, int | str | None],
    *,
    uproject: str | Path | None = None,
    ddc_directory: str | Path | None = None,
    concurrency_trial: dict[str, object] | None = None,
) -> dict[str, object]:
    """Say what one instance isolates by port and what it still shares.

    The temp directory, the log file and the shared-memory id are per port, so
    two instances do not collide there. A shared `.uproject` and a shared
    derived-data cache are not isolated by port at all. Whether two instances
    may use them at once is a measurement, not an assumption: without a
    recorded trial this reports `not_run` rather than claiming it is safe.
    """

    trial = dict(concurrency_trial or {})
    status = str(trial.get("status") or "not_run")
    return {
        "isolated_by_rpc_port": {
            "rpc_port": settings.get("rpc_port"),
            "temp_dir": settings.get("temp_dir"),
            "log": settings.get("log"),
            "shared_memory_initial_unique_id": settings.get(
                "shared_memory_initial_unique_id"
            ),
        },
        "graphics_adapter": settings.get("graphics_adapter"),
        "shared_between_instances": {
            "uproject": None if uproject is None else str(uproject),
            "ddc_directory": None if ddc_directory is None else str(ddc_directory),
        },
        "shared_input_concurrency": {
            "status": status,
            "detail": trial,
            "note": (
                "a shared uproject and derived-data cache are not separated by "
                "the RPC port; run a representative two-instance trial before "
                "treating concurrent use of them as safe"
            ),
        },
    }
