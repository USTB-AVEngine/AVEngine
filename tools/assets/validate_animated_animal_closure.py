#!/usr/bin/env python3
"""Validate an animated animal GLB and write one fresh closure report.

The GLB is imported by Blender and is never saved back.  The level and
retarget manifests are checked for the support-plane and source-motion
properties that make an animation closure review meaningful.  The validator
samples every requested action at its first, middle and last frame, checks
that the pose changes, and checks that the first and last poses close within
configurable tolerances.

No animal-specific bone count is assumed.  Expected bone and vertex-group
counts are optional assertions for a caller that already knows the target
rig.  The only file this tool writes is the requested report, and it refuses
an existing report path.

Blender invocation::

    blender --background --factory-startup \
      --python tools/assets/validate_animated_animal_closure.py -- \
      animated.glb level.json retarget.json validation.json \
      --required-actions Walking Idle
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

try:  # Blender supplies bpy; ordinary unit tests deliberately do not.
    import bpy  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised by normal Python tests
    bpy = None  # type: ignore[assignment]


REPORT_SCHEMA = "avengine_animal_chain_cpu_closure_validation_v1"
DEFAULT_REQUIRED_ACTIONS = ("Walking", "Idle")
DEFAULT_WEIGHT_TOLERANCE = 1.0e-3
DEFAULT_MIN_POSE_TRANSLATION_DELTA = 1.0e-7
DEFAULT_MIN_POSE_ROTATION_DELTA_DEG = 1.0e-5
DEFAULT_MAX_CYCLE_TRANSLATION_DELTA = 1.0e-3
DEFAULT_MAX_CYCLE_ROTATION_DELTA_DEG = 0.1


class AnimatedAnimalClosureError(ValueError):
    """The asset or one of its closure manifests is invalid."""


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnimatedAnimalClosureError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AnimatedAnimalClosureError(f"{owner} must be a finite number")
    return result


def _nonnegative_number(value: Any, *, owner: str) -> float:
    result = _finite_number(value, owner=owner)
    if result < 0.0:
        raise AnimatedAnimalClosureError(f"{owner} must be non-negative")
    return result


def _nonempty_text(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnimatedAnimalClosureError(f"{owner} must be a non-empty string")
    return value


def normalize_required_actions(actions: Sequence[str]) -> tuple[str, ...]:
    """Validate and normalize the action names requested by a caller."""

    if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
        raise AnimatedAnimalClosureError("required actions must be a sequence")
    result = tuple(
        _nonempty_text(value, owner=f"required_actions[{index}]")
        for index, value in enumerate(actions)
    )
    if not result:
        raise AnimatedAnimalClosureError("at least one required action is needed")
    if len(set(result)) != len(result):
        raise AnimatedAnimalClosureError("required action names must be unique")
    return result


def _input_file(path: str | Path, *, owner: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise AnimatedAnimalClosureError(f"{owner} must not be a symbolic link: {candidate}")
    if not candidate.is_file():
        raise AnimatedAnimalClosureError(f"{owner} is not a regular file: {candidate}")
    return candidate.resolve()


def _new_output_file(path: str | Path, *, owner: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists() or candidate.is_symlink():
        raise AnimatedAnimalClosureError(
            f"{owner} already exists; refusing to overwrite: {candidate}"
        )
    return candidate


def _json_object(path: Path, *, owner: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AnimatedAnimalClosureError(f"could not read {owner}: {path}") from error
    if not isinstance(value, dict):
        raise AnimatedAnimalClosureError(f"{owner} must contain a JSON object")
    return value


def _declared_local_path(value: Any, *, owner: str) -> Path:
    text = _nonempty_text(value, owner=owner)
    if "://" in text:
        raise AnimatedAnimalClosureError(f"{owner} must be a local path")
    return Path(text).expanduser().resolve()


def _require_same_declared_path(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_owner: str,
    right_owner: str,
) -> None:
    left_path = _declared_local_path(left.get("path"), owner=left_owner)
    right_path = _declared_local_path(right.get("path"), owner=right_owner)
    if left_path != right_path:
        raise AnimatedAnimalClosureError(
            f"{left_owner} and {right_owner} refer to different files"
        )


def validate_level_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate support-plane evidence without assuming four feet or a breed."""

    if not isinstance(value, Mapping):
        raise AnimatedAnimalClosureError("level manifest must be a JSON object")
    support = value.get("support_plane")
    if not isinstance(support, Mapping):
        raise AnimatedAnimalClosureError("level manifest lacks support_plane")
    foot_leaves = support.get("foot_leaves")
    if (
        not isinstance(foot_leaves, Sequence)
        or isinstance(foot_leaves, (str, bytes))
        or not foot_leaves
        or any(not isinstance(item, str) or not item for item in foot_leaves)
    ):
        raise AnimatedAnimalClosureError(
            "support_plane.foot_leaves must be a non-empty list of names"
        )
    plane_source = _nonempty_text(
        support.get("plane_source"), owner="support_plane.plane_source"
    )
    dual = support.get("dual_authority")
    if not isinstance(dual, Mapping):
        raise AnimatedAnimalClosureError(
            "support_plane.dual_authority must be an object"
        )
    agreement = dual.get("agreement")
    if not isinstance(agreement, Mapping) or agreement.get("passed") is not True:
        raise AnimatedAnimalClosureError(
            "support_plane dual-authority agreement did not pass"
        )
    if dual.get("fallback_used") is not False:
        raise AnimatedAnimalClosureError(
            "support_plane dual-authority fallback must be explicitly false"
        )

    residual = _nonnegative_number(
        support.get("maximum_residual_ratio_of_mesh_diagonal"),
        owner="support_plane.maximum_residual_ratio_of_mesh_diagonal",
    )
    reviewed_residual = _nonnegative_number(
        support.get("maximum_reviewed_residual_ratio_of_mesh_diagonal"),
        owner="support_plane.maximum_reviewed_residual_ratio_of_mesh_diagonal",
    )
    if reviewed_residual < residual:
        raise AnimatedAnimalClosureError(
            "support-plane residual exceeds its reviewed threshold"
        )
    tilt = _nonnegative_number(
        support.get("tilt_deg"), owner="support_plane.tilt_deg"
    )
    maximum_tilt = _nonnegative_number(
        support.get("maximum_tilt_deg"), owner="support_plane.maximum_tilt_deg"
    )
    if tilt > maximum_tilt:
        raise AnimatedAnimalClosureError("support-plane tilt exceeds its threshold")

    output = value.get("output")
    if not isinstance(output, Mapping):
        raise AnimatedAnimalClosureError("level manifest lacks output")
    output_path = _nonempty_text(output.get("path"), owner="level.output.path")
    return {
        "foot_leaves": list(foot_leaves),
        "plane_source": plane_source,
        "agreement_passed": True,
        "fallback_used": False,
        "maximum_residual_ratio": residual,
        "maximum_reviewed_residual_ratio": reviewed_residual,
        "tilt_deg": tilt,
        "maximum_tilt_deg": maximum_tilt,
        "output_path": output_path,
    }


def validate_retarget_manifest(
    value: Mapping[str, Any], *, required_actions: Sequence[str]
) -> dict[str, Any]:
    """Validate action declarations and ensure donor geometry did not leak."""

    actions = normalize_required_actions(required_actions)
    if not isinstance(value, Mapping):
        raise AnimatedAnimalClosureError("retarget manifest must be a JSON object")
    target = value.get("target")
    if not isinstance(target, Mapping):
        raise AnimatedAnimalClosureError("retarget manifest lacks target")
    target_path = _nonempty_text(target.get("path"), owner="retarget.target.path")
    source_motion = value.get("source_motion")
    if not isinstance(source_motion, Mapping):
        raise AnimatedAnimalClosureError("retarget manifest lacks source_motion")
    source_path = _nonempty_text(
        source_motion.get("path"), owner="retarget.source_motion.path"
    )
    if source_motion.get("geometry_used") is not False:
        raise AnimatedAnimalClosureError(
            "retarget source geometry must be explicitly unused"
        )
    if source_motion.get("weights_used") is not False:
        raise AnimatedAnimalClosureError(
            "retarget source weights must be explicitly unused"
        )
    export = value.get("export")
    if not isinstance(export, Mapping):
        raise AnimatedAnimalClosureError("retarget manifest lacks export")
    declared_actions = export.get("action_names")
    if declared_actions != list(actions):
        raise AnimatedAnimalClosureError(
            "retarget export action_names do not match required actions"
        )
    export_path = _nonempty_text(export.get("path"), owner="retarget.export.path")
    return {
        "target_path": target_path,
        "source_motion_path": source_path,
        "geometry_used": False,
        "weights_used": False,
        "action_names": list(actions),
        "export_path": export_path,
    }


def _vector(value: Any, *, length: int, owner: str) -> tuple[float, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != length
    ):
        raise AnimatedAnimalClosureError(f"{owner} must have {length} numeric values")
    return tuple(
        _finite_number(item, owner=f"{owner}[{index}]")
        for index, item in enumerate(value)
    )


def _snapshot_delta(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    *,
    owner: str,
) -> dict[str, float]:
    if set(left) != set(right) or not left:
        raise AnimatedAnimalClosureError(
            f"{owner} pose snapshots have different or empty bone sets"
        )
    maximum_translation = 0.0
    maximum_rotation = 0.0
    for name in left:
        left_bone = left[name]
        right_bone = right[name]
        if not isinstance(left_bone, Mapping) or not isinstance(right_bone, Mapping):
            raise AnimatedAnimalClosureError(f"{owner} bone {name!r} is invalid")
        left_location = _vector(
            left_bone.get("location"), length=3, owner=f"{owner}.{name}.location"
        )
        right_location = _vector(
            right_bone.get("location"), length=3, owner=f"{owner}.{name}.location"
        )
        translation = math.sqrt(
            sum((left_location[index] - right_location[index]) ** 2 for index in range(3))
        )
        left_quaternion = _vector(
            left_bone.get("quaternion"), length=4, owner=f"{owner}.{name}.quaternion"
        )
        right_quaternion = _vector(
            right_bone.get("quaternion"), length=4, owner=f"{owner}.{name}.quaternion"
        )
        left_norm = math.sqrt(sum(item * item for item in left_quaternion))
        right_norm = math.sqrt(sum(item * item for item in right_quaternion))
        if left_norm <= 0.0 or right_norm <= 0.0:
            raise AnimatedAnimalClosureError(f"{owner} bone {name!r} has a zero quaternion")
        dot = abs(
            sum(left_quaternion[index] * right_quaternion[index] for index in range(4))
            / (left_norm * right_norm)
        )
        dot = min(1.0, max(-1.0, dot))
        rotation = math.degrees(2.0 * math.acos(dot))
        maximum_translation = max(maximum_translation, translation)
        maximum_rotation = max(maximum_rotation, rotation)
    return {
        "translation": maximum_translation,
        "rotation_deg": maximum_rotation,
    }


def summarize_action_samples(
    action_name: str,
    *,
    frame_range: Sequence[float],
    sample_frames: Sequence[float],
    samples: Sequence[Mapping[str, Mapping[str, Any]]],
    minimum_pose_translation_delta: float = DEFAULT_MIN_POSE_TRANSLATION_DELTA,
    minimum_pose_rotation_delta_deg: float = DEFAULT_MIN_POSE_ROTATION_DELTA_DEG,
    maximum_cycle_translation_delta: float = DEFAULT_MAX_CYCLE_TRANSLATION_DELTA,
    maximum_cycle_rotation_delta_deg: float = DEFAULT_MAX_CYCLE_ROTATION_DELTA_DEG,
    require_closed_cycle: bool = True,
) -> dict[str, Any]:
    """Summarize first/middle/last pose changes and cyclic closure."""

    name = _nonempty_text(action_name, owner="action_name")
    if len(frame_range) != 2 or len(sample_frames) != 3 or len(samples) != 3:
        raise AnimatedAnimalClosureError(
            f"{name} requires a start/middle/end sample and a two-value frame range"
        )
    start = _finite_number(frame_range[0], owner=f"{name}.frame_range[0]")
    end = _finite_number(frame_range[1], owner=f"{name}.frame_range[1]")
    if end < start:
        raise AnimatedAnimalClosureError(f"{name} frame range is reversed")
    frames = [
        _finite_number(value, owner=f"{name}.sample_frames[{index}]")
        for index, value in enumerate(sample_frames)
    ]
    if not (frames[0] <= frames[1] <= frames[2]):
        raise AnimatedAnimalClosureError(f"{name} sample frames are not ordered")
    minimum_translation = _nonnegative_number(
        minimum_pose_translation_delta,
        owner="minimum_pose_translation_delta",
    )
    minimum_rotation = _nonnegative_number(
        minimum_pose_rotation_delta_deg,
        owner="minimum_pose_rotation_delta_deg",
    )
    maximum_translation = _nonnegative_number(
        maximum_cycle_translation_delta,
        owner="maximum_cycle_translation_delta",
    )
    maximum_rotation = _nonnegative_number(
        maximum_cycle_rotation_delta_deg,
        owner="maximum_cycle_rotation_delta_deg",
    )
    pair_specs = (
        ("first_middle", samples[0], samples[1]),
        ("middle_last", samples[1], samples[2]),
        ("first_last", samples[0], samples[2]),
    )
    pair_deltas = {
        pair_name: _snapshot_delta(left, right, owner=f"{name}.{pair_name}")
        for pair_name, left, right in pair_specs
    }
    max_sample_translation = max(item["translation"] for item in pair_deltas.values())
    max_sample_rotation = max(item["rotation_deg"] for item in pair_deltas.values())
    changed = (
        max_sample_translation > minimum_translation
        or max_sample_rotation > minimum_rotation
    )
    if not changed:
        raise AnimatedAnimalClosureError(
            f"{name} action does not change any sampled bone pose"
        )
    first_last = pair_deltas["first_last"]
    cycle_closed = (
        first_last["translation"] <= maximum_translation
        and first_last["rotation_deg"] <= maximum_rotation
    )
    if require_closed_cycle and not cycle_closed:
        raise AnimatedAnimalClosureError(
            f"{name} action does not close: "
            f"translation={first_last['translation']}, "
            f"rotation_deg={first_last['rotation_deg']}"
        )
    return {
        "action_name": name,
        "action_frame_range": [start, end],
        "sample_frames": frames,
        "pair_deltas": pair_deltas,
        "max_sample_translation_delta": max_sample_translation,
        "max_sample_rotation_delta_deg": max_sample_rotation,
        "max_first_last_translation_delta": first_last["translation"],
        "max_first_last_rotation_delta_deg": first_last["rotation_deg"],
        "cycle_closed": cycle_closed,
        "cycle_tolerances": {
            "translation": maximum_translation,
            "rotation_deg": maximum_rotation,
        },
    }


def _require_blender() -> Any:
    if bpy is None:
        raise AnimatedAnimalClosureError(
            "this validator must run inside Blender; ordinary Python can only test its pure helpers"
        )
    return bpy


def _resolve_action(actions: Sequence[Any], required_name: str) -> Any:
    matches = [
        action
        for action in actions
        if action.name == required_name or action.name.startswith(required_name + "_")
    ]
    if len(matches) != 1:
        raise AnimatedAnimalClosureError(
            f"required action {required_name!r} is missing or ambiguous; "
            f"available={[action.name for action in actions]}"
        )
    return matches[0]


def _blender_pose_snapshot(armature: Any, bone_names: Sequence[str]) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for name in bone_names:
        try:
            pose_bone = armature.pose.bones[name]
        except (KeyError, TypeError) as error:
            raise AnimatedAnimalClosureError(
                f"sample bone {name!r} is missing from the imported armature"
            ) from error
        world_matrix = armature.matrix_world @ pose_bone.matrix
        translation = world_matrix.to_translation()
        quaternion = world_matrix.to_quaternion()
        result[name] = {
            "location": [float(value) for value in translation],
            "quaternion": [float(value) for value in quaternion],
        }
    return result


def _validate_imported_glb(
    animated_path: Path,
    *,
    required_actions: Sequence[str],
    expected_bone_count: int | None,
    expected_vertex_group_count: int | None,
    sample_bones: Sequence[str] | None,
    weight_tolerance: float,
    minimum_pose_translation_delta: float,
    minimum_pose_rotation_delta_deg: float,
    maximum_cycle_translation_delta: float,
    maximum_cycle_rotation_delta_deg: float,
    require_closed_cycle: bool,
) -> dict[str, Any]:
    blender = _require_blender()
    blender.ops.wm.read_factory_settings(use_empty=True)
    result = blender.ops.import_scene.gltf(filepath=str(animated_path))
    if "FINISHED" not in result:
        raise AnimatedAnimalClosureError(f"GLB import did not finish: {result}")

    objects = list(blender.context.scene.objects)
    armatures = [obj for obj in objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise AnimatedAnimalClosureError(
            f"expected exactly one armature, got {[obj.name for obj in armatures]}"
        )
    armature = armatures[0]
    meshes = [obj for obj in objects if obj.type == "MESH"]
    skinned = [
        obj
        for obj in meshes
        if any(
            modifier.type == "ARMATURE" and modifier.object == armature
            for modifier in obj.modifiers
        )
    ]
    if len(skinned) != 1:
        raise AnimatedAnimalClosureError(
            f"expected exactly one mesh skinned to the armature, got "
            f"{[obj.name for obj in skinned]}"
        )
    mesh = skinned[0]
    bone_count = len(armature.data.bones)
    vertex_group_count = len(mesh.vertex_groups)
    if expected_bone_count is not None and bone_count != expected_bone_count:
        raise AnimatedAnimalClosureError(
            f"bone count {bone_count} differs from expected {expected_bone_count}"
        )
    if (
        expected_vertex_group_count is not None
        and vertex_group_count != expected_vertex_group_count
    ):
        raise AnimatedAnimalClosureError(
            f"vertex-group count {vertex_group_count} differs from expected "
            f"{expected_vertex_group_count}"
        )

    tolerance = _nonnegative_number(weight_tolerance, owner="weight_tolerance")
    weight_sums = [
        sum(float(group.weight) for group in vertex.groups)
        for vertex in mesh.data.vertices
    ]
    if not weight_sums or any(not math.isfinite(value) for value in weight_sums):
        raise AnimatedAnimalClosureError("skinned mesh has no finite vertex weights")
    if min(weight_sums) < 1.0 - tolerance or max(weight_sums) > 1.0 + tolerance:
        raise AnimatedAnimalClosureError(
            f"vertex weights fall outside 1 +/- {tolerance}: "
            f"{min(weight_sums)}..{max(weight_sums)}"
        )

    actions = list(blender.data.actions)
    names = normalize_required_actions(required_actions)
    resolved_actions = {name: _resolve_action(actions, name) for name in names}
    if armature.animation_data is None:
        armature.animation_data_create()
    for track in armature.animation_data.nla_tracks:
        track.mute = True
    if sample_bones is None:
        bone_names = tuple(pose_bone.name for pose_bone in armature.pose.bones)
    else:
        bone_names = normalize_required_actions(sample_bones)
    if not bone_names:
        raise AnimatedAnimalClosureError("imported armature has no pose bones to sample")

    action_reports: dict[str, Any] = {}
    for name in names:
        action = resolved_actions[name]
        try:
            frame_start, frame_end = (float(value) for value in action.frame_range)
        except (TypeError, ValueError) as error:
            raise AnimatedAnimalClosureError(
                f"action {name!r} has an invalid frame range"
            ) from error
        sample_frames = (
            frame_start,
            (frame_start + frame_end) * 0.5,
            frame_end,
        )
        armature.animation_data.action = action
        samples = []
        for frame in sample_frames:
            blender.context.scene.frame_set(frame)
            blender.context.view_layer.update()
            samples.append(_blender_pose_snapshot(armature, bone_names))
        report = summarize_action_samples(
            name,
            frame_range=(frame_start, frame_end),
            sample_frames=sample_frames,
            samples=samples,
            minimum_pose_translation_delta=minimum_pose_translation_delta,
            minimum_pose_rotation_delta_deg=minimum_pose_rotation_delta_deg,
            maximum_cycle_translation_delta=maximum_cycle_translation_delta,
            maximum_cycle_rotation_delta_deg=maximum_cycle_rotation_delta_deg,
            require_closed_cycle=require_closed_cycle,
        )
        report["imported_action_name"] = action.name
        report["sample_bone_count"] = len(bone_names)
        action_reports[name] = report

    return {
        "target": {
            "mesh_object": mesh.name,
            "vertices": len(mesh.data.vertices),
            "faces": len(mesh.data.polygons),
            "bones": bone_count,
            "vertex_groups": vertex_group_count,
            "weight_sum_range": [min(weight_sums), max(weight_sums)],
        },
        "actions": action_reports,
    }


def validate_animated_animal_closure(
    animated_glb: str | Path,
    level_manifest: str | Path,
    retarget_manifest: str | Path,
    *,
    required_actions: Sequence[str] = DEFAULT_REQUIRED_ACTIONS,
    expected_bone_count: int | None = None,
    expected_vertex_group_count: int | None = None,
    sample_bones: Sequence[str] | None = None,
    weight_tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
    minimum_pose_translation_delta: float = DEFAULT_MIN_POSE_TRANSLATION_DELTA,
    minimum_pose_rotation_delta_deg: float = DEFAULT_MIN_POSE_ROTATION_DELTA_DEG,
    maximum_cycle_translation_delta: float = DEFAULT_MAX_CYCLE_TRANSLATION_DELTA,
    maximum_cycle_rotation_delta_deg: float = DEFAULT_MAX_CYCLE_ROTATION_DELTA_DEG,
    require_closed_cycle: bool = True,
) -> dict[str, Any]:
    """Validate manifests and a Blender-imported GLB, returning a report."""

    actions = normalize_required_actions(required_actions)
    animated_path = _input_file(animated_glb, owner="animated GLB")
    level_path = _input_file(level_manifest, owner="level manifest")
    retarget_path = _input_file(retarget_manifest, owner="retarget manifest")
    level = _json_object(level_path, owner="level manifest")
    retarget = _json_object(retarget_path, owner="retarget manifest")
    level_summary = validate_level_manifest(level)
    retarget_summary = validate_retarget_manifest(
        retarget, required_actions=actions
    )
    _require_same_declared_path(
        {"path": level_summary["output_path"]},
        {"path": retarget_summary["target_path"]},
        left_owner="level.output.path",
        right_owner="retarget.target.path",
    )
    _require_same_declared_path(
        {"path": retarget_summary["export_path"]},
        {"path": str(animated_path)},
        left_owner="retarget.export.path",
        right_owner="animated GLB",
    )

    for value, owner in (
        (expected_bone_count, "expected_bone_count"),
        (expected_vertex_group_count, "expected_vertex_group_count"),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise AnimatedAnimalClosureError(f"{owner} must be a positive integer")
    if sample_bones is not None:
        sample_bones = normalize_required_actions(sample_bones)

    blender_summary = _validate_imported_glb(
        animated_path,
        required_actions=actions,
        expected_bone_count=expected_bone_count,
        expected_vertex_group_count=expected_vertex_group_count,
        sample_bones=sample_bones,
        weight_tolerance=weight_tolerance,
        minimum_pose_translation_delta=minimum_pose_translation_delta,
        minimum_pose_rotation_delta_deg=minimum_pose_rotation_delta_deg,
        maximum_cycle_translation_delta=maximum_cycle_translation_delta,
        maximum_cycle_rotation_delta_deg=maximum_cycle_rotation_delta_deg,
        require_closed_cycle=require_closed_cycle,
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "validator": "tools/assets/validate_animated_animal_closure.py",
        "animated_glb": str(animated_path),
        "required_actions": list(actions),
        "target": blender_summary["target"],
        "actions": blender_summary["actions"],
        "support_plane": {
            key: value
            for key, value in level_summary.items()
            if key != "output_path"
        },
        "retarget": {
            key: value
            for key, value in retarget_summary.items()
            if key != "target_path" and key != "export_path"
        },
        "validation_options": {
            "expected_bone_count": expected_bone_count,
            "expected_vertex_group_count": expected_vertex_group_count,
            "sample_bone_count": next(
                iter(blender_summary["actions"].values())
            )["sample_bone_count"],
            "require_closed_cycle": require_closed_cycle,
            "weight_tolerance": float(weight_tolerance),
        },
    }


def write_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Write one report with no-clobber semantics."""

    output = _new_output_file(path, owner="validation report")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as error:
        raise AnimatedAnimalClosureError(
            f"could not write validation report: {output}"
        ) from error
    return output


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a non-negative finite number") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("expected a non-negative finite number")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = (
            sys.argv[sys.argv.index("--") + 1 :]
            if "--" in sys.argv
            else sys.argv[1:]
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("animated_glb", type=Path)
    parser.add_argument("level_manifest", type=Path)
    parser.add_argument("retarget_manifest", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--required-actions",
        nargs="+",
        default=list(DEFAULT_REQUIRED_ACTIONS),
        metavar="ACTION",
    )
    parser.add_argument("--expected-bone-count", type=_positive_int)
    parser.add_argument("--expected-vertex-group-count", type=_positive_int)
    parser.add_argument("--sample-bones", nargs="+", metavar="BONE")
    parser.add_argument(
        "--weight-tolerance", type=_nonnegative_float, default=DEFAULT_WEIGHT_TOLERANCE
    )
    parser.add_argument(
        "--minimum-pose-translation-delta",
        type=_nonnegative_float,
        default=DEFAULT_MIN_POSE_TRANSLATION_DELTA,
    )
    parser.add_argument(
        "--minimum-pose-rotation-delta-deg",
        type=_nonnegative_float,
        default=DEFAULT_MIN_POSE_ROTATION_DELTA_DEG,
    )
    parser.add_argument(
        "--maximum-cycle-translation-delta",
        type=_nonnegative_float,
        default=DEFAULT_MAX_CYCLE_TRANSLATION_DELTA,
    )
    parser.add_argument(
        "--maximum-cycle-rotation-delta-deg",
        type=_nonnegative_float,
        default=DEFAULT_MAX_CYCLE_ROTATION_DELTA_DEG,
    )
    parser.add_argument(
        "--allow-open-cycle",
        action="store_true",
        help="report cycle_closed=false instead of failing an open action",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_animated_animal_closure(
            args.animated_glb,
            args.level_manifest,
            args.retarget_manifest,
            required_actions=args.required_actions,
            expected_bone_count=args.expected_bone_count,
            expected_vertex_group_count=args.expected_vertex_group_count,
            sample_bones=args.sample_bones,
            weight_tolerance=args.weight_tolerance,
            minimum_pose_translation_delta=args.minimum_pose_translation_delta,
            minimum_pose_rotation_delta_deg=args.minimum_pose_rotation_delta_deg,
            maximum_cycle_translation_delta=args.maximum_cycle_translation_delta,
            maximum_cycle_rotation_delta_deg=args.maximum_cycle_rotation_delta_deg,
            require_closed_cycle=not args.allow_open_cycle,
        )
        report_path = write_report(args.report, report)
    except AnimatedAnimalClosureError as error:
        print(f"animated animal closure validation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "ANIMAL_CHAIN_CLOSURE_VALIDATION_OK", "validation": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
