"""Explicit per-request asset-pair policy for QA-v3 scene design.

The policy chooses visual assets from the source runtime registry while keeping
visual entity class separate from the sound class an endpoint may play. A
resolved context is an ordinary request value; no module global is changed
when a different pair is selected.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


VALID_MOTION = frozenset({"must_move", "must_be_still", "any"})
VALID_FACING = frozenset({"toward_camera", "keep_mesh_forward"})


class AssetPolicyError(ValueError):
    """The request asset policy is incomplete or inconsistent."""


def _nonempty(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssetPolicyError(f"{owner} must be a nonempty string")
    return value


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssetPolicyError(f"{owner} must be an object")
    return value


def _motion(value: Any, *, owner: str) -> str:
    value = _nonempty(value, owner=owner)
    if value not in VALID_MOTION:
        raise AssetPolicyError(
            f"{owner} must be one of {sorted(VALID_MOTION)}, got {value!r}"
        )
    return value


def load_asset_policy(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssetPolicyError(f"cannot read asset policy {path}: {error}") from error
    if not isinstance(value, dict):
        raise AssetPolicyError("asset policy must be a JSON object")
    if value.get("schema") != "qa_v3_asset_policy_v1":
        raise AssetPolicyError("asset policy schema is invalid")
    if not isinstance(value.get("pairs"), Mapping) or not value["pairs"]:
        raise AssetPolicyError("asset policy pairs must be a nonempty object")
    return value


def _registry_family(record: Mapping[str, Any]) -> str:
    if record.get("entity_class") == "rigid_object":
        identity = record.get("identity")
        if isinstance(identity, Mapping):
            category = identity.get("category")
            if isinstance(category, str) and category:
                return category
        return "rigid_object"
    identity = record.get("identity")
    if isinstance(identity, Mapping):
        species = identity.get("species_id")
        if isinstance(species, str) and species:
            return species
    return str(record.get("entity_class") or "")


def _asset_spec(
    policy: Mapping[str, Any],
    asset_id: str,
    *,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    specs = _mapping(policy.get("assets"), owner="asset policy assets")
    raw = _mapping(specs.get(asset_id), owner=f"asset policy asset {asset_id}")
    label = _nonempty(raw.get("label"), owner=f"{asset_id}.label")
    phrase = _nonempty(
        raw.get("referent_phrase", f"the {label}"),
        owner=f"{asset_id}.referent_phrase",
    )
    classes = raw.get("allowed_sound_class_ids")
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(item, str) or not item for item in classes)
        or len(set(classes)) != len(classes)
    ):
        raise AssetPolicyError(
            f"{asset_id}.allowed_sound_class_ids must be unique nonempty strings"
        )
    action = _nonempty(
        raw.get("sound_action_phrase", "makes a sound"),
        owner=f"{asset_id}.sound_action_phrase",
    )
    family = _nonempty(
        raw.get("visual_family", _registry_family(record)),
        owner=f"{asset_id}.visual_family",
    )
    spear = record.get("runtime_backends", {}).get("spear_unreal", {})
    static_forward_yaw = (
        spear.get("ue_static_forward_yaw_deg")
        if isinstance(spear, Mapping)
        else None
    )
    return {
        "asset_id": asset_id,
        "label": label,
        "referent_phrase": phrase,
        "allowed_sound_class_ids": list(classes),
        "sound_action_phrase": action,
        "visual_family": family,
        "entity_class": record.get("entity_class"),
        "ue_static_forward_yaw_deg": static_forward_yaw,
    }


def resolve_asset_policy(
    policy: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    pair_id: str | None = None,
    profiles: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve one configured pair against the current runtime registry."""
    pairs = _mapping(policy.get("pairs"), owner="asset policy pairs")
    selected_id = pair_id or policy.get("default_pair_id")
    selected_id = _nonempty(selected_id, owner="asset policy pair_id")
    raw_pair = _mapping(
        pairs.get(selected_id), owner=f"asset policy pair {selected_id}"
    )
    asset_ids = raw_pair.get("asset_ids")
    if (
        not isinstance(asset_ids, list)
        or len(asset_ids) != 2
        or len(set(asset_ids)) != 2
        or any(not isinstance(item, str) or not item for item in asset_ids)
    ):
        raise AssetPolicyError(
            f"asset policy pair {selected_id} must contain two distinct asset_ids"
        )
    records = {
        record.get("asset_id"): record
        for record in registry.get("assets", ())
        if isinstance(record, Mapping)
    }
    missing = [asset_id for asset_id in asset_ids if asset_id not in records]
    if missing:
        raise AssetPolicyError(
            f"asset policy pair {selected_id} references assets absent from registry: {missing}"
        )
    specs = {
        asset_id: _asset_spec(policy, asset_id, record=records[asset_id])
        for asset_id in asset_ids
    }
    families = [specs[asset_id]["visual_family"] for asset_id in asset_ids]
    family_rule = _nonempty(
        raw_pair.get("family_rule", "explicit_pair"),
        owner=f"{selected_id}.family_rule",
    )
    if family_rule == "same_family_distinct_instances" and families[0] != families[1]:
        raise AssetPolicyError(
            f"{selected_id} requires same visual family, got {families}"
        )
    if family_rule == "same_family_distinct_instances" and asset_ids[0] == asset_ids[1]:
        raise AssetPolicyError(f"{selected_id} requires distinct assets")
    pair_kind = _nonempty(
        raw_pair.get("pair_kind", policy.get("default_pair_kind", "dog")),
        owner=f"{selected_id}.pair_kind",
    )
    raw_motion_by_role = raw_pair.get("motion_by_role", {})
    motion_by_role = _mapping(
        raw_motion_by_role, owner=f"{selected_id}.motion_by_role"
    )
    for role, value in motion_by_role.items():
        if role not in {"target", "other"}:
            raise AssetPolicyError(
                f"{selected_id}.motion_by_role has unknown role {role!r}"
            )
        _motion(value, owner=f"{selected_id}.motion_by_role.{role}")
    raw_motion_by_asset = raw_pair.get("motion_by_asset", {})
    motion_by_asset = _mapping(
        raw_motion_by_asset, owner=f"{selected_id}.motion_by_asset"
    )
    for asset_id, value in motion_by_asset.items():
        if asset_id not in specs:
            raise AssetPolicyError(
                f"{selected_id}.motion_by_asset references an unselected asset {asset_id!r}"
            )
        _motion(value, owner=f"{selected_id}.motion_by_asset.{asset_id}")
    raw_facing_by_asset = raw_pair.get("facing_by_asset", {})
    facing_by_asset = _mapping(
        raw_facing_by_asset, owner=f"{selected_id}.facing_by_asset"
    )
    for asset_id, value in facing_by_asset.items():
        if asset_id not in specs:
            raise AssetPolicyError(
                f"{selected_id}.facing_by_asset references an unselected asset {asset_id!r}"
            )
        value = _nonempty(value, owner=f"{selected_id}.facing_by_asset.{asset_id}")
        if value not in VALID_FACING:
            raise AssetPolicyError(
                f"{selected_id}.facing_by_asset.{asset_id} must be one of "
                f"{sorted(VALID_FACING)}"
            )
    allowed_kinds = raw_pair.get("allowed_answer_kinds")
    if allowed_kinds is not None:
        if (
            not isinstance(allowed_kinds, list)
            or not allowed_kinds
            or any(not isinstance(item, str) or not item for item in allowed_kinds)
            or len(set(allowed_kinds)) != len(allowed_kinds)
        ):
            raise AssetPolicyError(
                f"{selected_id}.allowed_answer_kinds must be unique nonempty strings"
            )
    context = {
        "policy_id": _nonempty(
            policy.get("policy_id"), owner="asset policy policy_id"
        ),
        "pair_id": selected_id,
        "family_rule": family_rule,
        "pair_kind": pair_kind,
        "asset_ids": tuple(asset_ids),
        "asset_specs": specs,
        "motion_by_role": dict(motion_by_role),
        "motion_by_asset": dict(motion_by_asset),
        "facing_by_asset": dict(facing_by_asset),
        "allowed_answer_kinds": list(allowed_kinds) if allowed_kinds is not None else None,
        "answer_depends_on_source_displacement": bool(
            raw_pair.get("answer_depends_on_source_displacement", False)
        ),
        "raw": copy.deepcopy(dict(raw_pair)),
    }
    if profiles is not None:
        for profile in profiles:
            kind = profile.get("answer_kind", "azimuth_band")
            if context["allowed_answer_kinds"] is not None and kind not in context[
                "allowed_answer_kinds"
            ]:
                raise AssetPolicyError(
                    f"{selected_id} does not support profile {profile.get('id')!r} "
                    f"answer_kind {kind!r}"
                )
    return context


def motion_constraint(
    context: Mapping[str, Any],
    *,
    asset_id: str,
    role: str,
) -> str:
    role_value = context.get("motion_by_role", {}).get(role)
    if role_value is not None and role_value != "any":
        return str(role_value)
    asset_value = context.get("motion_by_asset", {}).get(asset_id)
    if asset_value is not None:
        return str(asset_value)
    return "any"


def asset_spec(context: Mapping[str, Any], asset_id: str) -> Mapping[str, Any]:
    try:
        return context["asset_specs"][asset_id]
    except (KeyError, TypeError) as error:
        raise AssetPolicyError(f"asset {asset_id!r} is absent from policy context") from error


def slot_context(
    context: Mapping[str, Any],
    *,
    assets_by_slot: Mapping[str, str],
    target_slot: str = "source1",
) -> dict[str, Any]:
    labels = {}
    phrases = {}
    sound_actions = {}
    sound_classes = {}
    motion = {}
    for slot, asset_id in assets_by_slot.items():
        spec = asset_spec(context, asset_id)
        labels[slot] = spec["label"]
        phrases[slot] = spec["referent_phrase"]
        sound_actions[slot] = spec["sound_action_phrase"]
        sound_classes[slot] = list(spec["allowed_sound_class_ids"])
        role = "target" if slot == target_slot else "other"
        motion[slot] = motion_constraint(context, asset_id=asset_id, role=role)
    return {
        "labels_by_slot": labels,
        "referent_phrases_by_slot": phrases,
        "sound_action_phrases_by_slot": sound_actions,
        "allowed_sound_class_ids_by_slot": sound_classes,
        "motion_by_slot": motion,
    }


__all__ = [
    "AssetPolicyError",
    "asset_spec",
    "load_asset_policy",
    "motion_constraint",
    "resolve_asset_policy",
    "slot_context",
]
