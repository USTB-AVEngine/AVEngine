"""Selected rig-query helpers for AVEngine's external SPEAR/UE runtime.

Adapted from the Eastforward SPEAR fork, not from the official SPEAR source.
Only the three runner-facing public functions below are exposed; calibration,
CLI, file-writing, and other transition-tool behavior is deliberately absent.
"""

# Adapted from the Eastforward/spear fork.
# Original path: tools/spike_rlr/rig_direction_check.py.
# Fork behavior origins: 0a9ba3ded8ffa07a3bc3684279845da22dc123e0 (initial),
# c8ba04076a32060e35020deb8f706c4b13951cae (runner-facing component selection), and
# ff6e44736f68c72ce4140152e2dadb4b58dc0b28 (current carried rig behavior).
# Selected bytes are carried by local MIT transition snapshot 251bd5e0d3d1e7297ec072bb9b0df9ef63f864b7
# (SPEAR-lead-b); that local carrier is not represented as a public fork ref.
# AVEngine modifications: retain only the helper closure used by its runners;
# expose only the three listed functions; support helpers are private and
# calibration/CLI/file behavior is removed.
# License: MIT; see LICENSES/SPEAR-MIT.txt.

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = [
    "select_skeletal_mesh_component",
    "sample_body_bone_position_in_frame",
    "sample_body_basis_in_frame",
]


_BODY_BASIS_BONE_CANDIDATES = {
    "pelvis": ("Bip01 Pelvis", "Bip02 Pelvis", "Pelvis", "Hips"),
    "spine": ("Bip01 Spine2", "Bip02 Spine2", "Spine2", "Spine1", "Spine"),
    "left_clavicle": (
        "Bip01 L Clavicle",
        "Bip02 L Clavicle",
        "LeftClavicle",
        "LeftShoulder",
        "mixamorig LeftShoulder",
    ),
    "right_clavicle": (
        "Bip01 R Clavicle",
        "Bip02 R Clavicle",
        "RightClavicle",
        "RightShoulder",
        "mixamorig RightShoulder",
    ),
}


_QUADRUPED_BASIS_BONE_CANDIDATES = {
    "rear": ("Bone", "Hips"),
    "front": ("Bone.002", "Bone.001", "Shoulders"),
    "body": ("Bone", "Hips", "Body"),
    "left_foot": ("Bone.010", "IKBackLeft", "BackFoot.L"),
    "right_foot": ("Bone.013", "IKBackRight", "BackFoot.R"),
}


_QUATERNIUS_NATIVE_NAMED_BASIS_BONE_CANDIDATES = {
    "rear": ("Back",),
    "front": ("Torso3", "Neck1", "Torso2"),
    "body": ("Back", "Torso"),
    "left_foot": (
        "BackLowerLeg.L_end",
        "BackLowerLeg.L",
        "IKBackLeg.L",
        "FFB.L",
    ),
    "right_foot": (
        "BackLowerLeg.R_end",
        "BackLowerLeg.R",
        "IKBackLeg.R",
        "FFB.R",
    ),
}


_PREFIXED_BIP_QUADRUPED_BASIS_BONE_CANDIDATES = {
    "rear": ("Pelvis",),
    "front": ("Spine2", "Neck"),
    "body": ("Pelvis",),
    "left_foot": ("L Foot",),
    "right_foot": ("R Foot",),
}


_PREFIXED_BIP_QUADRUPED_REQUIRED_MARKERS = (("Tail",),)


_EXPLICIT_QUADRUPED_SEMANTIC_ROLES = (
    "rear",
    "front",
    "body",
    "left_foot",
    "right_foot",
)


def _integer_return_value(value) -> int:
    """Normalize direct and as-dict Unreal integer return values."""
    current = value
    for _ in range(2):
        if not isinstance(current, dict):
            return int(current)
        lowered = {str(key).lower(): item for key, item in current.items()}
        if "returnvalue" not in lowered:
            break
        current = lowered["returnvalue"]
    return int(current)


def _name_return_value(value) -> str:
    current = value
    for _ in range(2):
        if not isinstance(current, dict):
            return str(current)
        lowered = {str(key).lower(): item for key, item in current.items()}
        if "returnvalue" not in lowered:
            break
        current = lowered["returnvalue"]
    return str(current)


def _normalized_bone_name(name: str) -> str:
    return "".join(character.lower() for character in str(name) if character.isalnum())


def _match_explicit_bone_name(available_names, *, role: str, requested_name) -> str:
    if not isinstance(requested_name, str) or not requested_name.strip():
        raise ValueError(f"explicit {role} bone name must be a non-empty string")
    matches = [
        name
        for name in available_names
        if name == requested_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"explicit {role} bone {requested_name!r} matched "
            f"{len(matches)} component bones; expected exactly one"
        )
    return matches[0]


def _match_explicit_quadruped_semantic_roles(
    available_names,
    semantic_bone_names,
):
    if not isinstance(semantic_bone_names, Mapping):
        raise TypeError("semantic_bone_names must be a mapping")
    expected_roles = set(_EXPLICIT_QUADRUPED_SEMANTIC_ROLES)
    actual_roles = set(semantic_bone_names)
    if actual_roles != expected_roles:
        missing = sorted(expected_roles - actual_roles)
        unexpected = sorted(
            (repr(role) for role in actual_roles - expected_roles)
        )
        raise ValueError(
            "semantic_bone_names must contain exactly "
            f"{list(_EXPLICIT_QUADRUPED_SEMANTIC_ROLES)!r}; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    return {
        role: _match_explicit_bone_name(
            available_names,
            role=role,
            requested_name=semantic_bone_names[role],
        )
        for role in _EXPLICIT_QUADRUPED_SEMANTIC_ROLES
    }


def _component_bone_names(component):
    bone_count = _integer_return_value(component.GetNumBones())
    return [
        _name_return_value(component.GetBoneName(BoneIndex=index))
        for index in range(bone_count)
    ]


def _match_common_prefixed_roles(
    available_names,
    candidate_map,
    *,
    required_marker_groups=(),
):
    """Match a complete semantic scheme under one non-empty bone prefix."""
    by_normalized_name = {
        _normalized_bone_name(name): name for name in available_names
    }
    first_role = next(iter(candidate_map))
    prefixes = set()
    for actual_normalized in by_normalized_name:
        for candidate in candidate_map[first_role]:
            suffix = _normalized_bone_name(candidate)
            if actual_normalized.endswith(suffix) and len(actual_normalized) > len(suffix):
                prefixes.add(actual_normalized[: -len(suffix)])

    for prefix in sorted(prefixes):
        marker_groups_present = all(
            any(
                prefix + _normalized_bone_name(marker) in by_normalized_name
                for marker in marker_group
            )
            for marker_group in required_marker_groups
        )
        if not marker_groups_present:
            continue
        matched = {}
        for role, candidates in candidate_map.items():
            for candidate in candidates:
                actual = by_normalized_name.get(
                    prefix + _normalized_bone_name(candidate)
                )
                if actual is not None:
                    matched[role] = actual
                    break
        if len(matched) == len(candidate_map):
            return matched, prefix
    return {}, None


def _unit_vector(vector, *, label: str):
    value = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if length < 1e-6:
        raise ValueError(f"{label} vector is degenerate")
    return value / length


def _body_basis_from_positions(*, pelvis, spine, left_clavicle, right_clavicle):
    """Build a semantic body basis from same-frame world-space bone positions."""
    pelvis = np.asarray(pelvis, dtype=np.float64)
    spine = np.asarray(spine, dtype=np.float64)
    left_clavicle = np.asarray(left_clavicle, dtype=np.float64)
    right_clavicle = np.asarray(right_clavicle, dtype=np.float64)

    up = _unit_vector(spine - pelvis, label="body up")
    right = right_clavicle - left_clavicle
    right = right - up * float(np.dot(right, up))
    right = _unit_vector(right, label="body right")
    # UE's body frame is +X forward, +Y right, +Z up, so right x up
    # recovers forward. Reversing this cross product points through the back.
    forward = _unit_vector(np.cross(right, up), label="body forward")
    forward_xy = forward[:2]
    if float(np.linalg.norm(forward_xy)) < 1e-6:
        raise ValueError("body forward has no horizontal component")

    return {
        "up_vector_ue": up.tolist(),
        "right_vector_ue": right.tolist(),
        "forward_vector_ue": forward.tolist(),
        "forward_yaw_ue_deg": float(np.degrees(np.arctan2(
            forward[1], forward[0]
        ))),
        "up_alignment_z": float(up[2]),
    }


def _quadruped_basis_from_positions(
    *, rear, front, body, left_foot, right_foot
):
    """Build a quadruped body basis from torso and paired-foot anchors."""
    rear = np.asarray(rear, dtype=np.float64)
    front = np.asarray(front, dtype=np.float64)
    body = np.asarray(body, dtype=np.float64)
    left_foot = np.asarray(left_foot, dtype=np.float64)
    right_foot = np.asarray(right_foot, dtype=np.float64)

    feet_center = 0.5 * (left_foot + right_foot)
    up = _unit_vector(body - feet_center, label="quadruped up")
    forward = front - rear
    forward = forward - up * float(np.dot(forward, up))
    forward = _unit_vector(forward, label="quadruped forward")
    right = _unit_vector(np.cross(up, forward), label="quadruped right")
    forward_xy = forward[:2]
    if float(np.linalg.norm(forward_xy)) < 1e-6:
        raise ValueError("quadruped forward has no horizontal component")
    anatomical_right = right_foot - left_foot
    anatomical_right = anatomical_right - up * float(np.dot(anatomical_right, up))
    anatomical_alignment = float(
        np.dot(right, _unit_vector(anatomical_right, label="quadruped anatomical right"))
    )
    return {
        "basis_kind": "quadruped_longitudinal_v1",
        "up_vector_ue": up.tolist(),
        "right_vector_ue": right.tolist(),
        "forward_vector_ue": forward.tolist(),
        "forward_yaw_ue_deg": float(
            np.degrees(np.arctan2(forward[1], forward[0]))
        ),
        "up_alignment_z": float(up[2]),
        "anatomical_right_alignment": anatomical_alignment,
    }


def sample_body_basis_in_frame(
    actor,
    *,
    unreal_service=None,
    diagnostics=None,
    semantic_bone_names=None,
):
    """Sample humanoid or quadruped body axes inside an active SPEAR frame."""
    if unreal_service is None:
        raise RuntimeError(
            "unreal_service is required for safe UClass handle marshalling"
        )
    try:
        component = select_skeletal_mesh_component(
            unreal_service=unreal_service,
            actor=actor,
            diagnostics=diagnostics,
        )
        if component is None:
            return None
        available_names = _component_bone_names(component)
        by_normalized_name = {
            _normalized_bone_name(name): name for name in available_names
        }
        def match_roles(candidate_map):
            matched = {}
            for role, candidates in candidate_map.items():
                for candidate in candidates:
                    actual = by_normalized_name.get(_normalized_bone_name(candidate))
                    if actual is not None:
                        matched[role] = actual
                        break
            return matched

        if semantic_bone_names is not None:
            matched_names = _match_explicit_quadruped_semantic_roles(
                available_names,
                semantic_bone_names,
            )
            basis_builder = _quadruped_basis_from_positions
            basis_kind = "authenticated_generated_quadruped_longitudinal_v1"
            positions = {}
            for role, bone_name in matched_names.items():
                position = sample_body_bone_position_in_frame(
                    actor,
                    bone_name,
                    unreal_service=unreal_service,
                    diagnostics=diagnostics,
                )
                if position is None:
                    return None
                positions[role] = position
            basis = basis_builder(**positions)
            basis["basis_kind"] = basis_kind
            basis["bone_names"] = matched_names
            basis["positions_ue_cm"] = {
                role: np.asarray(position, dtype=np.float64).tolist()
                for role, position in positions.items()
            }
            return basis

        human_names = match_roles(_BODY_BASIS_BONE_CANDIDATES)
        quadruped_names = match_roles(_QUADRUPED_BASIS_BONE_CANDIDATES)
        native_named_names = match_roles(
            _QUATERNIUS_NATIVE_NAMED_BASIS_BONE_CANDIDATES
        )
        prefixed_bip_names, prefixed_bip_namespace = _match_common_prefixed_roles(
            available_names,
            _PREFIXED_BIP_QUADRUPED_BASIS_BONE_CANDIDATES,
            required_marker_groups=_PREFIXED_BIP_QUADRUPED_REQUIRED_MARKERS,
        )
        if len(human_names) == len(_BODY_BASIS_BONE_CANDIDATES):
            matched_names = human_names
            basis_builder = _body_basis_from_positions
            basis_kind = "humanoid_semantic_v1"
        elif len(native_named_names) == len(
            _QUATERNIUS_NATIVE_NAMED_BASIS_BONE_CANDIDATES
        ):
            matched_names = native_named_names
            basis_builder = _quadruped_basis_from_positions
            basis_kind = "quaternius_native_named_longitudinal_v1"
        elif len(prefixed_bip_names) == len(
            _PREFIXED_BIP_QUADRUPED_BASIS_BONE_CANDIDATES
        ):
            matched_names = prefixed_bip_names
            basis_builder = _quadruped_basis_from_positions
            basis_kind = "prefixed_bip_quadruped_longitudinal_v1"
        elif len(quadruped_names) == len(_QUADRUPED_BASIS_BONE_CANDIDATES):
            matched_names = quadruped_names
            basis_builder = _quadruped_basis_from_positions
            basis_kind = "quadruped_longitudinal_v1"
        else:
            if diagnostics is not None:
                diagnostics.append({
                    "stage": "body_basis_bone_lookup",
                    "candidate_schemes": {
                        "humanoid": {
                            role: list(candidates)
                            for role, candidates in _BODY_BASIS_BONE_CANDIDATES.items()
                        },
                        "quadruped": {
                            role: list(candidates)
                            for role, candidates in _QUADRUPED_BASIS_BONE_CANDIDATES.items()
                        },
                        "quaternius_native_named": {
                            role: list(candidates)
                            for role, candidates in (
                                _QUATERNIUS_NATIVE_NAMED_BASIS_BONE_CANDIDATES.items()
                            )
                        },
                        "prefixed_bip_quadruped": {
                            role: list(candidates)
                            for role, candidates in (
                                _PREFIXED_BIP_QUADRUPED_BASIS_BONE_CANDIDATES.items()
                            )
                        },
                    },
                    "matched_humanoid_roles": sorted(human_names),
                    "matched_quadruped_roles": sorted(quadruped_names),
                    "matched_quaternius_native_named_roles": sorted(
                        native_named_names
                    ),
                    "matched_prefixed_bip_roles": sorted(prefixed_bip_names),
                    "matched_prefixed_bip_namespace": prefixed_bip_namespace,
                    "available_bone_names": available_names,
                })
            return None

        positions = {}
        for role, bone_name in matched_names.items():
            position = sample_body_bone_position_in_frame(
                actor,
                bone_name,
                unreal_service=unreal_service,
                diagnostics=diagnostics,
            )
            if position is None:
                return None
            positions[role] = position

        basis = basis_builder(**positions)
        # The shared quadruped math is reused by multiple independently
        # validated bone-name schemes.  Publish the selected semantic scheme,
        # not only the generic math helper's default label.
        basis["basis_kind"] = basis_kind
        basis["bone_names"] = matched_names
        basis["positions_ue_cm"] = {
            role: np.asarray(position, dtype=np.float64).tolist()
            for role, position in positions.items()
        }
        return basis
    except Exception as error:
        if diagnostics is not None:
            diagnostics.append({
                "stage": "body_basis",
                "error_type": type(error).__name__,
                "error": str(error),
            })
        return None


def select_skeletal_mesh_component(*, unreal_service, actor, diagnostics=None):
    """Select the actor's populated skeletal component.

    A SkeletalMeshActor Blueprint can expose an empty inherited component
    before its actual imported 80-bone mesh component. Unreal's singular
    GetComponentByClass-style lookup therefore is not sufficient here.
    """
    components = unreal_service.get_components_by_class(
        actor=actor,
        uclass="USkeletalMeshComponent",
    )
    if components is None:
        components = []
    elif not isinstance(components, (list, tuple)):
        components = [components]

    populated = []
    inventory = []
    for index, component in enumerate(components):
        try:
            bone_count = _integer_return_value(component.GetNumBones())
        except Exception as error:
            inventory.append({
                "component_index": index,
                "error_type": type(error).__name__,
                "error": str(error),
            })
            continue
        inventory.append({
            "component_index": index,
            "bone_count": bone_count,
        })
        if bone_count > 0:
            populated.append((bone_count, -index, component))

    if populated:
        return max(populated, key=lambda item: (item[0], item[1]))[2]

    if diagnostics is not None:
        diagnostics.append({
            "stage": "component_selection",
            "error_type": "MissingRiggedComponent",
            "error": "no USkeletalMeshComponent with one or more bones",
            "component_inventory": inventory,
        })
    return None


def sample_body_bone_position_in_frame(
    actor,
    bone_name: str,
    *,
    unreal_service=None,
    diagnostics=None,
):
    """Query a bone's world-space location via SPEAR RPC.

    IMPORTANT: caller must be inside an active `instance.begin_frame()`
    context — this function does NOT open its own frame. That was the bug
    in v1: opening a nested begin_frame after the render loop tore down
    frame state triggered engine_service.begin_frame:157 assert False.

    Returns np.ndarray shape (3,) in UE cm world frame, or None if the bone
    doesn't exist.
    """
    if unreal_service is None:
        raise RuntimeError(
            "unreal_service is required for safe UClass handle marshalling"
        )
    try:
        # The service wrapper resolves USkeletalMeshComponent to a real UClass
        # handle. Calling actor.GetComponentByClass with a class-path string
        # reaches SPEAR's native pointer parser and asserts on non-0x input.
        comp = select_skeletal_mesh_component(
            unreal_service=unreal_service,
            actor=actor,
            diagnostics=diagnostics,
        )
        if comp is None:
            return None
        bone_index = int(comp.GetBoneIndex(BoneName=bone_name))
        if bone_index < 0:
            if diagnostics is not None:
                diagnostics.append({
                    "bone_name": str(bone_name),
                    "stage": "bone_lookup",
                    "error_type": "MissingBone",
                    "error": f"GetBoneIndex returned {bone_index}",
                })
            return None
        tf = comp.GetBoneTransform(
            InBoneName=bone_name,
            TransformSpace="RTS_World",
            as_dict=True,
        )
        if isinstance(tf, dict) and "ReturnValue" in tf:
            tf = tf["ReturnValue"]
        if isinstance(tf, dict):
            lowered = {str(key).lower(): value for key, value in tf.items()}
            loc = lowered.get("translation") or lowered.get("location")
        else:
            loc = getattr(tf, "Translation", None) or getattr(tf, "Location", None)
        if loc is None:
            if diagnostics is not None:
                diagnostics.append({
                    "bone_name": str(bone_name),
                    "stage": "parse",
                    "error_type": type(tf).__name__,
                    "error": f"missing Translation in {repr(tf)[:500]}",
                })
            return None
        if isinstance(loc, dict):
            lowered_loc = {str(key).lower(): value for key, value in loc.items()}
            return np.array(
                [lowered_loc["x"], lowered_loc["y"], lowered_loc["z"]],
                dtype=np.float64,
            )
        return np.array([loc.x, loc.y, loc.z], dtype=np.float64)
    except Exception as error:
        if diagnostics is not None:
            diagnostics.append({
                "bone_name": str(bone_name),
                "stage": "query",
                "error_type": type(error).__name__,
                "error": str(error),
            })
        return None
