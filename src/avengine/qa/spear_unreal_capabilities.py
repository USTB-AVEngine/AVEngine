"""Strict compatibility helpers for SPEAR-backed Unreal object proxies.

The helpers in this module are deliberately duck typed and do not import
SPEAR.  They probe the capabilities exposed by a live proxy and return an
evidence record describing the selected path.  Compatibility never means
silently accepting ambiguity: zero matches, duplicate normalized names,
invalid handles, and readback drift are errors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class SpearUnrealCapabilityError(RuntimeError):
    """Raised when a live Unreal proxy cannot satisfy a strict capability."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SpearUnrealCapabilityError(message)


def _return_value(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    for key, item in value.items():
        if str(key).replace("_", "").lower() == "returnvalue":
            return item
    return value


def live_handle(value: Any, *, owner: str) -> int:
    """Return a positive UObject handle from an integer or proxy object."""

    raw = value
    if not isinstance(value, int) or isinstance(value, bool):
        raw = getattr(value, "uobject", None)
        if raw is None:
            raw = getattr(value, "_uobject", None)
    try:
        handle = int(raw)
    except (TypeError, ValueError) as exc:
        raise SpearUnrealCapabilityError(
            f"{owner} did not expose a live Unreal handle"
        ) from exc
    _require(
        not isinstance(raw, bool) and handle > 0,
        f"{owner} exposed an invalid Unreal handle",
    )
    return handle


def read_handle_capability(
    target: Any,
    *,
    owner: str,
    getter_name: str,
    property_name: str,
    getter_kwargs: Mapping[str, Any] | None = None,
    property_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read one UObject handle across callable-getter and property proxies.

    A callable getter is authoritative when present.  Its runtime exceptions
    are intentionally not hidden by falling back to a property.  A missing or
    non-callable getter selects the explicit property-read capability.
    """

    getter = getattr(target, getter_name, None)
    if callable(getter):
        value = getter(**dict(getter_kwargs or {}))
        strategy = "callable_getter"
    else:
        reader = getattr(target, "get_property_value", None)
        _require(
            callable(reader),
            f"{owner} exposes neither callable {getter_name} nor a readable "
            f"{property_name} property",
        )
        kwargs = {"property_name": property_name, **dict(property_kwargs or {})}
        value = reader(**kwargs)
        strategy = "property_readback"
    return {
        "schema": "avengine_spear_handle_capability_readback_v1",
        "status": "pass",
        "owner": owner,
        "handle": live_handle(_return_value(value), owner=owner),
        "strategy": strategy,
        "getter_name": getter_name,
        "property_name": property_name,
    }


def set_numeric_property_with_readback(
    components: Mapping[str, Any],
    *,
    owner: str,
    property_name: str,
    requested_value: float,
    required_names: Sequence[str] | None = None,
    tolerance: float = 1.0e-6,
    require_distinct_handles: bool = True,
) -> dict[str, Any]:
    """Set/read a numeric property while proving component handle stability."""

    names = tuple(required_names or components.keys())
    _require(bool(names), f"{owner} has no selected components")
    _require(len(set(names)) == len(names), f"{owner} component names repeat")
    missing = [name for name in names if name not in components]
    _require(not missing, f"missing named camera components for {owner}: {missing}")
    selected = {name: components[name] for name in names}
    handles_before = {
        name: live_handle(component, owner=f"{owner} {name}")
        for name, component in selected.items()
    }
    if require_distinct_handles:
        _require(
            len(set(handles_before.values())) == len(handles_before),
            f"{owner} named components do not have distinct live handles",
        )
    requested = float(requested_value)
    _require(math.isfinite(requested), f"{owner} requested value is non-finite")
    _require(
        math.isfinite(float(tolerance)) and float(tolerance) >= 0.0,
        f"{owner} tolerance is invalid",
    )
    observed: dict[str, float] = {}
    for name, component in selected.items():
        component.set_property_value(
            property_name=property_name,
            property_value=requested,
        )
        value = float(component.get_property_value(property_name=property_name))
        _require(math.isfinite(value), f"{owner} {name} readback is non-finite")
        _require(
            abs(value - requested) <= float(tolerance),
            f"{owner} {name} {property_name} readback drift",
        )
        observed[name] = value
    handles_after = {
        name: live_handle(component, owner=f"{owner} {name} after write")
        for name, component in selected.items()
    }
    _require(
        handles_after == handles_before,
        f"{owner} component handle drift during {property_name} write/readback",
    )
    return {
        "schema": "avengine_spear_numeric_property_readback_v1",
        "status": "pass",
        "owner": owner,
        "property_name": property_name,
        "requested_value": requested,
        "tolerance": float(tolerance),
        "component_handles": handles_before,
        "observed_by_component": observed,
    }


def normalize_unreal_name(value: str) -> str:
    """Normalize an Unreal FName while retaining a strict uniqueness gate."""

    _require(isinstance(value, str) and bool(value), "bone name must be non-empty")
    normalized = "".join(
        character for character in value.casefold() if character.isalnum()
    )
    _require(bool(normalized), f"bone name {value!r} normalizes to an empty value")
    return normalized


def _bone_index(component: Any, name: str, *, owner: str) -> int:
    try:
        return int(_return_value(component.GetBoneIndex(BoneName=name)))
    except (TypeError, ValueError) as exc:
        raise SpearUnrealCapabilityError(
            f"{owner} GetBoneIndex returned an invalid value for {name!r}"
        ) from exc


def resolve_component_bone_names(
    component: Any,
    requested_names: Sequence[str],
    *,
    owner: str,
) -> dict[str, Any]:
    """Resolve semantic bone names against a complete live UE inventory.

    Rocketbox Interchange imports may expose ``Bip01 L Foot`` as
    ``Bip01-L-Foot``.  Resolution therefore uses a normalized comparison, but
    the full live inventory must contain exactly one normalized match.  The
    returned *actual* FName is the only value consumers may pass to subsequent
    GetBoneIndex/GetBoneTransform calls.
    """

    try:
        count = int(_return_value(component.GetNumBones()))
    except (TypeError, ValueError) as exc:
        raise SpearUnrealCapabilityError(f"{owner} bone count is invalid") from exc
    _require(count > 0, f"{owner} has no live bones")
    inventory: list[dict[str, Any]] = []
    for index in range(count):
        actual = _return_value(component.GetBoneName(BoneIndex=index))
        _require(
            isinstance(actual, str) and bool(actual),
            f"{owner} bone {index} has an invalid live name",
        )
        inventory.append(
            {
                "inventory_index": index,
                "actual_name": actual,
                "normalized_name": normalize_unreal_name(actual),
            }
        )
    requested = tuple(requested_names)
    _require(bool(requested), f"{owner} requested no bones")
    _require(
        len(set(requested)) == len(requested),
        f"{owner} requested duplicate bone names",
    )
    resolutions: list[dict[str, Any]] = []
    for name in requested:
        normalized = normalize_unreal_name(name)
        matches = [item for item in inventory if item["normalized_name"] == normalized]
        _require(
            len(matches) == 1,
            f"{owner} bone {name!r} did not resolve exactly once "
            f"(normalized_matches={len(matches)})",
        )
        match = matches[0]
        actual = str(match["actual_name"])
        inventory_index = int(match["inventory_index"])
        actual_probe = _bone_index(component, actual, owner=owner)
        _require(
            actual_probe == inventory_index,
            f"{owner} live bone {actual!r} index differs from its inventory index",
        )
        requested_probe = _bone_index(component, name, owner=owner)
        _require(
            requested_probe in (-1, inventory_index),
            f"{owner} requested bone {name!r} resolved to an unexpected index",
        )
        if name == actual:
            _require(
                requested_probe == inventory_index,
                f"{owner} exact bone {name!r} did not resolve to its live index",
            )
            mode = "direct_exact_fname"
        elif requested_probe == inventory_index:
            mode = "requested_alias_fname_accepted"
        else:
            mode = "sanitized_live_fname_required"
        resolutions.append(
            {
                "requested_name": name,
                "requested_normalized_name": normalized,
                "actual_live_name": actual,
                "actual_normalized_name": str(match["normalized_name"]),
                "inventory_index": inventory_index,
                "requested_probe_index": requested_probe,
                "actual_probe_index": actual_probe,
                "resolution_mode": mode,
            }
        )
    return {
        "schema": "avengine_spear_bone_name_resolution_v1",
        "status": "pass",
        "owner": owner,
        "normalization": "unicode_casefold_then_alphanumeric_only",
        "bone_count": count,
        "inventory": inventory,
        "resolutions": resolutions,
    }
