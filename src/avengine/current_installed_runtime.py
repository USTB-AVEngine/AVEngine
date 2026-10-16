"""Validate non-pinned current-installed Habitat/RLR runtime identities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from avengine.backends.rlr.sdk import ExternalRlrSdkError, require_outside_git_checkout


CURRENT_INSTALLED_RUNTIME_IDENTITY_KEYS = frozenset(
    {
        "identity_schema",
        "mode",
        "habitat_runtime_prefix",
        "habitat_sim_module",
        "habitat_sim_binding",
        "magnum_python_site",
        "rlr_sdk_root",
        "rlr_sdk_header",
        "rlr_sdk_library",
        "rlr_adapter_enabled",
        "binding_api",
    }
)


def _canonical_existing_path(value: Any, *, directory: bool) -> Path | None:
    """Return an existing canonical non-checkout path matching its receipt text."""

    if not isinstance(value, str) or not value:
        return None
    declared = Path(value)
    if not declared.is_absolute():
        return None
    try:
        resolved = require_outside_git_checkout(
            declared,
            owner="current-installed runtime identity",
        )
    except (ExternalRlrSdkError, OSError, RuntimeError):
        return None
    if value != str(resolved):
        return None
    if directory:
        return resolved if resolved.is_dir() else None
    return resolved if resolved.is_file() else None


def _is_contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def is_current_installed_runtime_identity(value: Any) -> bool:
    """Require an observed, accessible non-checkout current runtime identity.

    This validates installation provenance and component containment without
    recording any historical binary hash or reusable runtime baseline.
    """

    if (
        not isinstance(value, Mapping)
        or set(value) != CURRENT_INSTALLED_RUNTIME_IDENTITY_KEYS
        or value.get("identity_schema")
        != "avengine_current_installed_rlr_runtime_v1"
        or value.get("mode") != "current-installed"
        or value.get("rlr_adapter_enabled") is not True
        or value.get("binding_api") != "habitat_sim.RLRAcousticContext_v1"
    ):
        return False
    prefix = _canonical_existing_path(
        value.get("habitat_runtime_prefix"),
        directory=True,
    )
    module = _canonical_existing_path(
        value.get("habitat_sim_module"),
        directory=False,
    )
    binding = _canonical_existing_path(
        value.get("habitat_sim_binding"),
        directory=False,
    )
    magnum_site = _canonical_existing_path(
        value.get("magnum_python_site"),
        directory=True,
    )
    sdk_root = _canonical_existing_path(value.get("rlr_sdk_root"), directory=True)
    sdk_header = _canonical_existing_path(
        value.get("rlr_sdk_header"),
        directory=False,
    )
    sdk_library = _canonical_existing_path(
        value.get("rlr_sdk_library"),
        directory=False,
    )
    if any(
        item is None
        for item in (
            prefix,
            module,
            binding,
            magnum_site,
            sdk_root,
            sdk_header,
            sdk_library,
        )
    ):
        return False
    assert (
        prefix is not None
        and module is not None
        and binding is not None
        and magnum_site is not None
        and sdk_root is not None
        and sdk_header is not None
        and sdk_library is not None
    )
    return (
        _is_contained(module, prefix)
        and _is_contained(binding, prefix)
        and _is_contained(sdk_header, sdk_root)
        and _is_contained(sdk_library, sdk_root)
    )


__all__ = ["is_current_installed_runtime_identity"]
