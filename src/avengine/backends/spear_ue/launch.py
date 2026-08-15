"""Launch settings for AVEngine's external SPEAR/UE runtime adapter."""

from __future__ import annotations

# Copyright (c) 2025 The SPEAR Development Team
# Copyright (c) 2022 Intel
# SPDX-License-Identifier: MIT
#
# Behavior reimplemented from
# spear-sim/spear@251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7,
# examples/render_in_apartment.py::parallel_instance_settings.
# The upstream MIT text is retained at LICENSES/SPEAR-MIT.txt.


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
