"""Discovery and process-local loading for a user-installed RLR SDK.

The AVEngine source tree contains the adapter that calls the modern RLR C API,
but not the RLR SDK, headers, or shared object.  This module deliberately uses
the same ``AVENGINE_RLR_SDK_ROOT`` layout as ``native/habitat/cmake/RlrSdk.cmake``
and rejects Git-checkout roots or symlink escapes before native code is loaded.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any


RLR_SDK_ROOT_ENVIRONMENT_VARIABLE = "AVENGINE_RLR_SDK_ROOT"
_HEADER_RELATIVE_PATH = Path("headers") / "RLRAudioPropagation.h"
_LIBRARY_RELATIVE_PATH = (
    Path("libs") / "linux" / "x64" / "libRLRAudioPropagation.so"
)
_PRELOADED_RLR_SDK_HANDLES: dict[Path, Any] = {}


class ExternalRlrSdkError(RuntimeError):
    """The declared external RLR SDK is absent, unsafe, or cannot be loaded."""


@dataclass(frozen=True)
class ExternalRlrSdk:
    """Canonical paths to the only RLR files AVEngine consumes at runtime."""

    root: Path
    header: Path
    library: Path


def require_outside_git_checkout(path: str | Path, *, owner: str) -> Path:
    """Resolve *path* and reject it when any ancestor is a Git checkout.

    This is the runtime counterpart of the CMake SDK containment policy.  It
    also rejects a root reached through a symlink into an old checkout.
    """

    resolved = Path(path).resolve()
    candidate = resolved
    while True:
        marker = candidate / ".git"
        if marker.exists() or marker.is_symlink():
            raise ExternalRlrSdkError(
                f"{owner} must resolve outside a Git checkout: {resolved}"
            )
        parent = candidate.parent
        if parent == candidate:
            return resolved
        candidate = parent


def _require_file_within_root(root: Path, relative: Path, *, owner: str) -> Path:
    requested = root / relative
    resolved = requested.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ExternalRlrSdkError(
            f"{owner} must resolve inside {RLR_SDK_ROOT_ENVIRONMENT_VARIABLE}: "
            f"{resolved}"
        ) from error
    if not resolved.is_file():
        raise ExternalRlrSdkError(f"{owner} is missing: {requested}")
    return resolved


def discover_external_rlr_sdk(
    explicit_root: str | Path | None = None,
) -> ExternalRlrSdk:
    """Resolve the explicit, non-checkout RLR SDK required by the adapter."""

    configured = (
        explicit_root
        if explicit_root is not None
        else os.environ.get(RLR_SDK_ROOT_ENVIRONMENT_VARIABLE)
    )
    if not configured:
        raise ExternalRlrSdkError(
            f"Set {RLR_SDK_ROOT_ENVIRONMENT_VARIABLE} to the user-installed "
            "RLRAudioPropagationPkg directory before invoking native RLR."
        )
    root = Path(configured).resolve()
    if not root.is_dir():
        raise ExternalRlrSdkError(
            f"{RLR_SDK_ROOT_ENVIRONMENT_VARIABLE} is not a directory: {root}"
        )
    root = require_outside_git_checkout(
        root, owner=RLR_SDK_ROOT_ENVIRONMENT_VARIABLE
    )
    return ExternalRlrSdk(
        root=root,
        header=_require_file_within_root(
            root,
            _HEADER_RELATIVE_PATH,
            owner=(
                f"{RLR_SDK_ROOT_ENVIRONMENT_VARIABLE} "
                "headers/RLRAudioPropagation.h"
            ),
        ),
        library=_require_file_within_root(
            root,
            _LIBRARY_RELATIVE_PATH,
            owner=(
                f"{RLR_SDK_ROOT_ENVIRONMENT_VARIABLE} "
                "libs/linux/x64/libRLRAudioPropagation.so"
            ),
        ),
    )


def _mapped_rlr_libraries() -> tuple[Path, ...]:
    """Return loaded RLR mappings on Linux without inferring from a neighbor."""

    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return ()
    values: set[Path] = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        raw_path = fields[5]
        if raw_path.endswith(" (deleted)"):
            raw_path = raw_path[: -len(" (deleted)")]
        candidate = Path(raw_path)
        if candidate.name != "libRLRAudioPropagation.so":
            continue
        if candidate.is_absolute():
            values.add(candidate.resolve())
    return tuple(sorted(values))


def validate_loaded_external_rlr_sdk(sdk: ExternalRlrSdk) -> None:
    """Ensure no binding-neighbor or checkout RLR library was selected."""

    if not sys.platform.startswith("linux"):
        return
    observed = _mapped_rlr_libraries()
    if observed != (sdk.library,):
        rendered = [str(path) for path in observed]
        raise ExternalRlrSdkError(
            "Loaded RLR shared objects do not resolve only to the declared "
            f"{RLR_SDK_ROOT_ENVIRONMENT_VARIABLE} library {sdk.library}: {rendered}"
        )


def preload_external_rlr_sdk(sdk: ExternalRlrSdk) -> None:
    """Load the SDK before importing the Habitat extension that links it.

    The handle is process-local and intentionally retained.  This avoids an
    implicit sibling ``libRLRAudioPropagation.so`` beside the Python binding or
    a process-wide ``LD_LIBRARY_PATH`` lookup while retaining a precise error
    when the legal external SDK cannot load on this host.
    """

    if sdk.library not in _PRELOADED_RLR_SDK_HANDLES:
        mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
        try:
            _PRELOADED_RLR_SDK_HANDLES[sdk.library] = ctypes.CDLL(
                str(sdk.library), mode=mode
            )
        except OSError as error:
            raise ExternalRlrSdkError(
                "Unable to preload the declared external RLR shared library "
                f"{sdk.library}: {error}"
            ) from error
    validate_loaded_external_rlr_sdk(sdk)


__all__ = [
    "ExternalRlrSdk",
    "ExternalRlrSdkError",
    "RLR_SDK_ROOT_ENVIRONMENT_VARIABLE",
    "discover_external_rlr_sdk",
    "preload_external_rlr_sdk",
    "require_outside_git_checkout",
    "validate_loaded_external_rlr_sdk",
]
