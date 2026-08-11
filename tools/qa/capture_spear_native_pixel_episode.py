#!/usr/bin/env python3
"""Capture a full native SPEAR RGB/depth/pixel-truth Episode.

The tool replays every formal frame from an existing native Apartment suite
plan.  A normal pass records RGB, metric depth and raw object IDs.  One
show-only pass per controlled source records target-only metric depth from the
same ``BP_CameraSensor``.  The QA pixel compiler derives visibility from exact
normal-vs-target depth agreement and verifies renderer/camera/frame identity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.pixel_visibility import (
    PIXEL_VISIBILITY_DEPTH_AUTHORITY,
    compile_depth_pixel_visibility_truth,
)

SPIKE_PATH = REPOSITORY / "tools/qa/spike_spear_native_pixel_visibility.py"
SPIKE_SPEC = importlib.util.spec_from_file_location(
    "lead_a_native_pixel_spike", SPIKE_PATH
)
if SPIKE_SPEC is None or SPIKE_SPEC.loader is None:
    raise RuntimeError(f"cannot import {SPIKE_PATH}")
SPIKE = importlib.util.module_from_spec(SPIKE_SPEC)
SPIKE_SPEC.loader.exec_module(SPIKE)
RUNNER = SPIKE.RUNNER

SCHEMA = "avengine_qa_native_spear_pixel_episode_v1"
ABSOLUTE_TOLERANCE_M = 0.01
RELATIVE_TOLERANCE = 0.002
TARGET_ONLY_BACKGROUND_DEPTH_M = 65504.0
RUNTIME_ASSET_SAMPLE_FRAME_INDICES = (0, 37, 74)
GROUND_CONTACT_BONES = {
    "left": {"foot": "Bip01 L Foot", "toe": "Bip01 L Toe0"},
    "right": {"foot": "Bip01 R Foot", "toe": "Bip01 R Toe0"},
}
GROUND_TRACE_START_OFFSET_CM = 25.0
GROUND_TRACE_LENGTH_CM = 300.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_artifact_record(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"artifact file is missing: {path}")
    return {
        "kind": "file",
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _directory_artifact_record(path: Path) -> dict[str, Any]:
    _require(path.is_dir(), f"artifact directory is missing: {path}")
    inventory = [
        {
            "relative_path": str(file.relative_to(path)),
            "size_bytes": file.stat().st_size,
            "sha256": _sha256(file),
        }
        for file in sorted(item for item in path.rglob("*") if item.is_file())
    ]
    _require(inventory, f"artifact directory is empty: {path}")
    return {
        "kind": "directory",
        "path": str(path.resolve()),
        "file_count": len(inventory),
        "total_size_bytes": sum(item["size_bytes"] for item in inventory),
        "inventory": inventory,
        "inventory_root_sha256": _canonical_json_sha256(inventory),
    }


def _raw_object_ids(component: Any) -> np.ndarray:
    bgra = component.read_pixels()["arrays"]["data"].copy()
    return np.ascontiguousarray(bgra).view(np.uint32).reshape(
        bgra.shape[:2]
    ) & np.uint32(0x00FFFFFF)


def _depth_native(component: Any) -> np.ndarray:
    return component.read_pixels()["arrays"]["data"][:, :, 0].copy()


def _rgb_bgr(component: Any) -> np.ndarray:
    return component.read_pixels()["arrays"]["data"][:, :, :3].copy()


def _safe_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "rawId",
        "actorStableName",
        "actorName",
        "componentStableName",
        "componentName",
        "materialName",
    ):
        value = descriptor.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


def _descriptor_raw_ids(
    descriptors: Sequence[Mapping[str, Any]], stable_name: str
) -> list[int]:
    return [
        int(descriptor["rawId"])
        for descriptor in descriptors
        if descriptor.get("actorStableName") == stable_name
    ]


def _return_value(value: Any) -> Any:
    """Unwrap SPEAR's direct and ``as_dict`` return conventions."""

    current = value
    for _ in range(2):
        if not isinstance(current, Mapping):
            return current
        lowered = {str(key).lower(): item for key, item in current.items()}
        if "returnvalue" not in lowered:
            return current
        current = lowered["returnvalue"]
    return current


def _mapping_value(value: Mapping[str, Any], key: str) -> Any:
    lowered = {str(name).lower(): item for name, item in value.items()}
    _require(key.lower() in lowered, f"Unreal result is missing {key}")
    return lowered[key.lower()]


def _xyz(value: Any, *, owner: str) -> list[float]:
    if isinstance(value, Mapping):
        lowered = {str(key).lower(): item for key, item in value.items()}
        result = [float(lowered[axis]) for axis in ("x", "y", "z")]
    else:
        result = [float(value.x), float(value.y), float(value.z)]
    _require(all(np.isfinite(item) for item in result), f"{owner} is nonfinite")
    return result


def _safe_unreal_reference(value: Any) -> str | int | float | bool | None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _live_bone_world_position(
    component: Any, *, bone_name: str, actor_id: str
) -> list[float]:
    bone_count = int(_return_value(component.GetNumBones()))
    _require(bone_count > 0, f"{actor_id} has no live bones")
    available_names = [
        str(_return_value(component.GetBoneName(BoneIndex=index)))
        for index in range(bone_count)
    ]
    _require(
        available_names.count(bone_name) == 1,
        f"{actor_id} bone {bone_name!r} did not resolve exactly once",
    )
    bone_index = int(component.GetBoneIndex(BoneName=bone_name))
    _require(bone_index >= 0, f"{actor_id} bone {bone_name!r} has invalid index")
    transform = component.GetBoneTransform(
        InBoneName=bone_name,
        TransformSpace="RTS_World",
        as_dict=True,
    )
    transform = _return_value(transform)
    _require(isinstance(transform, Mapping), f"{actor_id} bone transform is invalid")
    translation = _mapping_value(transform, "translation")
    return _xyz(translation, owner=f"{actor_id} {bone_name} world position")


def _line_trace_floor(
    *,
    game: Any,
    position_ue_cm: Sequence[float],
    actors_to_ignore: Sequence[Any],
    owner: str,
) -> dict[str, Any]:
    start = {
        "X": float(position_ue_cm[0]),
        "Y": float(position_ue_cm[1]),
        "Z": float(position_ue_cm[2]) + GROUND_TRACE_START_OFFSET_CM,
    }
    end = dict(start)
    end["Z"] = start["Z"] - GROUND_TRACE_LENGTH_CM
    kismet = game.get_unreal_object(uclass="UKismetSystemLibrary")
    trace = kismet.LineTraceSingleByProfile(
        Start=start,
        End=end,
        ProfileName="BlockAll",
        bTraceComplex=True,
        ActorsToIgnore=list(actors_to_ignore),
        DrawDebugType="None",
        bIgnoreSelf=True,
        TraceColor={"R": 1.0, "G": 0.0, "B": 0.0, "A": 1.0},
        TraceHitColor={"R": 0.0, "G": 1.0, "B": 0.0, "A": 1.0},
        DrawTime=0.0,
        as_dict=True,
    )
    _require(isinstance(trace, Mapping), f"{owner} floor trace result is invalid")
    _require(bool(_return_value(trace)), f"{owner} downward trace did not hit")
    hit = _mapping_value(trace, "outhit")
    _require(isinstance(hit, Mapping), f"{owner} floor hit is invalid")
    location = _xyz(_mapping_value(hit, "location"), owner=f"{owner} hit point")
    normal = _xyz(_mapping_value(hit, "normal"), owner=f"{owner} hit normal")
    horizontal_error_cm = max(
        abs(location[0] - float(position_ue_cm[0])),
        abs(location[1] - float(position_ue_cm[1])),
    )
    _require(
        horizontal_error_cm <= 1.0e-4,
        f"{owner} floor trace moved horizontally",
    )
    return {
        "status": "hit",
        "profile_name": "BlockAll",
        "trace_complex": True,
        "actors_to_ignore_count": len(actors_to_ignore),
        "start_ue_cm": [start[axis] for axis in ("X", "Y", "Z")],
        "end_ue_cm": [end[axis] for axis in ("X", "Y", "Z")],
        "hit_actor": _safe_unreal_reference(_mapping_value(hit, "actor")),
        "hit_component": _safe_unreal_reference(_mapping_value(hit, "component")),
        "hit_point_ue_cm": location,
        "hit_normal_ue": normal,
        "horizontal_error_cm": horizontal_error_cm,
    }


def _runtime_ground_contact_readback(
    *, game: Any, runtimes: Mapping[str, Mapping[str, Any]], actor_id: str
) -> dict[str, Any]:
    """Measure live Rocketbox foot/toe bones against runtime floor traces.

    This is deliberately a measurement-only record.  A later release gate must
    supply an asset- and animation-specific contact interval; this function does
    not infer one from the actor root or from imported mesh bounds.
    """

    runtime = runtimes[actor_id]
    ignored_actors = [
        actor
        for item in runtimes.values()
        for actor in (item.get("anchor"), item.get("visual_actor"))
        if actor is not None
    ]
    sides: dict[str, Any] = {}
    for side, bone_names in GROUND_CONTACT_BONES.items():
        anchors: dict[str, Any] = {}
        for anchor_kind, bone_name in bone_names.items():
            position = _live_bone_world_position(
                runtime["component"], bone_name=bone_name, actor_id=actor_id
            )
            trace = _line_trace_floor(
                game=game,
                position_ue_cm=position,
                actors_to_ignore=ignored_actors,
                owner=f"{actor_id} {bone_name}",
            )
            clearance_cm = position[2] - float(trace["hit_point_ue_cm"][2])
            _require(np.isfinite(clearance_cm), f"{actor_id} clearance is nonfinite")
            anchors[anchor_kind] = {
                "bone_name": bone_name,
                "bone_index": int(
                    runtime["component"].GetBoneIndex(BoneName=bone_name)
                ),
                "world_position_ue_cm": position,
                "floor_trace": trace,
                "bone_to_floor_clearance_cm": clearance_cm,
            }
        sides[side] = {
            "status": "observed",
            "anchors": anchors,
            "minimum_bone_to_floor_clearance_cm": min(
                float(value["bone_to_floor_clearance_cm"]) for value in anchors.values()
            ),
        }
    latest_snap = runtime.get("latest_runtime_visual_ground_snap")
    if latest_snap is None:
        snap_record = {
            "schema": "ue_dynamic_ground_snap_v1",
            "status": "not_requested",
            "timeline_anchor_mutated": False,
            "emitter_or_rir_mutated": False,
        }
    else:
        _require(
            latest_snap.get("status") == "passed",
            f"{actor_id} runtime visual ground snap did not pass",
        )
        snap_record = deepcopy(dict(latest_snap))
    return {
        "schema": "avengine_native_live_ground_contact_readback_v1",
        "status": "pass_instrumented_measurement_only",
        "authority": (
            "live_USkeletalMeshComponent_GetBoneTransform_RTS_World_plus_"
            "complex_BlockAll_downward_trace_v1"
        ),
        "claim_boundary": (
            "no_contact_release_without_a_declared_asset_animation_contact_profile"
        ),
        "ue_length_unit": "centimeter",
        "trace_policy": {
            "start_offset_cm": GROUND_TRACE_START_OFFSET_CM,
            "length_cm": GROUND_TRACE_LENGTH_CM,
            "ignored_runtime_actor_count": len(ignored_actors),
        },
        "runtime_visual_ground_snap": snap_record,
        "sides": sides,
    }


def _actor_location_ue_cm(actor: Any, *, owner: str) -> list[float]:
    location = actor.K2_GetActorLocation(as_dict=True)
    location = _return_value(location)
    _require(isinstance(location, Mapping), f"{owner} location is invalid")
    return _xyz(location, owner=f"{owner} location")


def _visual_bounds_ue_cm(actor: Any, *, owner: str) -> dict[str, Any]:
    bounds = actor.GetActorBounds(bOnlyCollidingComponents=False, as_dict=True)
    bounds = _return_value(bounds)
    _require(isinstance(bounds, Mapping), f"{owner} bounds are invalid")
    origin = _xyz(_mapping_value(bounds, "origin"), owner=f"{owner} bounds origin")
    extent = _xyz(_mapping_value(bounds, "boxextent"), owner=f"{owner} bounds extent")
    _require(all(value > 0.0 for value in extent), f"{owner} bounds are degenerate")
    return {
        "origin_ue_cm": origin,
        "extent_ue_cm": extent,
        "minimum_ue_cm": [origin[index] - extent[index] for index in range(3)],
        "maximum_ue_cm": [origin[index] + extent[index] for index in range(3)],
        "bottom_z_ue_cm": origin[2] - extent[2],
    }


def _runtime_visual_ground_snap_contract(
    declaration: Mapping[str, Any], *, actor_id: str
) -> Mapping[str, Any] | None:
    profile = declaration.get("ground_contact_release_profile")
    if profile is None:
        return None
    _require(isinstance(profile, Mapping), f"{actor_id} ground profile is invalid")
    snap = profile.get("runtime_visual_ground_snap")
    _require(isinstance(snap, Mapping), f"{actor_id} ground snap is missing")
    _require(
        snap.get("schema") == "ue_dynamic_ground_snap_v1"
        and snap.get("target") == "attached_visual_actor_root_component"
        and snap.get("timeline_anchor_mutation_allowed") is False
        and snap.get("emitter_or_rir_mutation_allowed") is False,
        f"{actor_id} ground snap authority drift",
    )
    for key in ("maximum_abs_correction_cm", "residual_tolerance_cm"):
        value = snap.get(key)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(float(value)),
            f"{actor_id} {key} is invalid",
        )
    _require(
        0.0 < float(snap["maximum_abs_correction_cm"]) <= 15.0
        and 0.0 <= float(snap["residual_tolerance_cm"]) <= 0.1,
        f"{actor_id} ground snap exceeds normalization limits",
    )
    authority = snap.get("normalization_manifest_authority")
    _require(
        isinstance(authority, str) and Path(authority).is_file(),
        f"{actor_id} normalization manifest authority is missing",
    )
    manifest = _load_json_object(Path(authority))
    expected = manifest.get("expected_ue_qa", {})
    runtime_motion = manifest.get("runtime_motion_contract", {})
    _require(
        runtime_motion.get("dynamic_ground_snap_to_floor_required") is True
        and expected.get("ground_snap_to_floor") is True
        and float(expected.get("ground_snap_max_abs_correction_cm", -1.0))
        == float(snap["maximum_abs_correction_cm"])
        and float(expected.get("ground_snap_residual_tolerance_cm", -1.0))
        == float(snap["residual_tolerance_cm"]),
        f"{actor_id} ground snap differs from normalization authority",
    )
    return snap


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _reset_runtime_visual_ground_snap(runtime: Mapping[str, Any]) -> None:
    correction_cm = float(runtime.get("runtime_visual_ground_snap_z_cm", 0.0))
    if abs(correction_cm) > 0.0:
        runtime["visual_root"].K2_AddRelativeLocation(
            DeltaLocation={"X": 0.0, "Y": 0.0, "Z": -correction_cm},
            bSweep=False,
            bTeleport=True,
        )
    runtime["runtime_visual_ground_snap_z_cm"] = 0.0
    runtime["latest_runtime_visual_ground_snap"] = None


def _apply_runtime_visual_ground_snap(
    *,
    game: Any,
    runtimes: Mapping[str, Mapping[str, Any]],
    declaration: Mapping[str, Any],
    actor_id: str,
    frame_index: int,
) -> dict[str, Any]:
    contract = _runtime_visual_ground_snap_contract(declaration, actor_id=actor_id)
    if contract is None:
        return {
            "schema": "ue_dynamic_ground_snap_v1",
            "status": "not_requested",
            "frame_index": frame_index,
            "timeline_anchor_mutated": False,
            "emitter_or_rir_mutated": False,
        }
    runtime = runtimes[actor_id]
    anchor_before = _actor_location_ue_cm(
        runtime["anchor"], owner=f"{actor_id} timeline anchor"
    )
    bounds_before = _visual_bounds_ue_cm(
        runtime["visual_actor"], owner=f"{actor_id} visual actor before snap"
    )
    ignored_actors = [
        actor
        for item in runtimes.values()
        for actor in (item.get("anchor"), item.get("visual_actor"))
        if actor is not None
    ]
    floor_trace = _line_trace_floor(
        game=game,
        position_ue_cm=anchor_before,
        actors_to_ignore=ignored_actors,
        owner=f"{actor_id} root-floor authority",
    )
    floor_z_cm = float(floor_trace["hit_point_ue_cm"][2])
    correction_cm = floor_z_cm - float(bounds_before["bottom_z_ue_cm"])
    maximum_abs_correction_cm = float(contract["maximum_abs_correction_cm"])
    _require(
        np.isfinite(correction_cm) and abs(correction_cm) <= maximum_abs_correction_cm,
        f"{actor_id} ground snap correction {correction_cm:.6f} cm exceeds "
        f"{maximum_abs_correction_cm:.6f} cm",
    )
    runtime["visual_root"].K2_AddRelativeLocation(
        DeltaLocation={"X": 0.0, "Y": 0.0, "Z": correction_cm},
        bSweep=False,
        bTeleport=True,
    )
    runtime["runtime_visual_ground_snap_z_cm"] = correction_cm
    bounds_after = _visual_bounds_ue_cm(
        runtime["visual_actor"], owner=f"{actor_id} visual actor after snap"
    )
    residual_cm = float(bounds_after["bottom_z_ue_cm"]) - floor_z_cm
    residual_tolerance_cm = float(contract["residual_tolerance_cm"])
    _require(
        np.isfinite(residual_cm) and abs(residual_cm) <= residual_tolerance_cm,
        f"{actor_id} ground snap residual {residual_cm:.6f} cm exceeds "
        f"{residual_tolerance_cm:.6f} cm",
    )
    anchor_after = _actor_location_ue_cm(
        runtime["anchor"], owner=f"{actor_id} timeline anchor"
    )
    anchor_error_cm = max(
        abs(before - after) for before, after in zip(anchor_before, anchor_after)
    )
    _require(
        anchor_error_cm <= 1.0e-6,
        f"{actor_id} ground snap mutated the Timeline/acoustic anchor",
    )
    record = {
        "schema": "ue_dynamic_ground_snap_v1",
        "status": "passed",
        "frame_index": frame_index,
        "target": "attached_visual_actor_root_component",
        "floor_trace": floor_trace,
        "floor_z_cm": floor_z_cm,
        "visual_bounds_before": bounds_before,
        "applied_z_correction_cm": correction_cm,
        "visual_bounds_after": bounds_after,
        "residual_clearance_cm": residual_cm,
        "maximum_abs_correction_cm": maximum_abs_correction_cm,
        "residual_tolerance_cm": residual_tolerance_cm,
        "timeline_anchor_before_ue_cm": anchor_before,
        "timeline_anchor_after_ue_cm": anchor_after,
        "maximum_timeline_anchor_error_cm": anchor_error_cm,
        "timeline_anchor_mutated": False,
        "emitter_or_rir_mutated": False,
        "bounds_role": "action_only_not_release_evidence",
    }
    runtime["latest_runtime_visual_ground_snap"] = record
    return record


def _apply_exact_frame_with_ground_snap(
    *,
    game: Any,
    camera: Any,
    runtimes: Mapping[str, Mapping[str, Any]],
    scenario: Mapping[str, Any],
    frame: Mapping[str, Any],
) -> dict[str, Any]:
    declarations = {actor["actor_id"]: actor for actor in scenario["plan"]["actors"]}
    _require(set(declarations) == set(runtimes), "ground-snap actor closure failed")
    for runtime in runtimes.values():
        _reset_runtime_visual_ground_snap(runtime)
    readback = SPIKE._apply_exact_frame(camera=camera, runtimes=runtimes, frame=frame)
    readback["visual_ground_snaps"] = {
        actor_id: _apply_runtime_visual_ground_snap(
            game=game,
            runtimes=runtimes,
            declaration=declarations[actor_id],
            actor_id=actor_id,
            frame_index=int(frame["frame_index"]),
        )
        for actor_id in sorted(runtimes)
    }
    return readback


def _runtime_asset_readbacks(
    *,
    game: Any,
    scenario: Mapping[str, Any],
    runtimes: Mapping[str, Mapping[str, Any]],
    stable_names: Mapping[str, str],
    raw_descriptors: Sequence[Mapping[str, Any]],
    frame: Mapping[str, Any],
) -> dict[str, Any]:
    """Read exact live BP/mesh/skeleton/action and mouth-root bindings."""

    declarations = {item["actor_id"]: item for item in scenario["plan"]["actors"]}
    states = {item["actor_id"]: item for item in frame["actor_states"]}
    _require(
        set(declarations) == set(runtimes) == set(states),
        "runtime asset readback actor closure failed",
    )
    records: dict[str, Any] = {}
    for actor_id in sorted(runtimes):
        runtime = runtimes[actor_id]
        declaration = declarations[actor_id]
        state = states[actor_id]
        instance_id = actor_id.removesuffix("_actor")
        stable_name = stable_names[instance_id]
        raw_ids = _descriptor_raw_ids(raw_descriptors, stable_name)
        _require(raw_ids, f"{actor_id} stable actor tag has no proxy descriptor")

        blueprint_path = declaration.get("blueprint_class_path")
        _require(
            isinstance(blueprint_path, str) and blueprint_path,
            f"{actor_id} Blueprint path is missing",
        )
        expected_blueprint_handle = int(
            game.unreal_service.load_class(
                uclass="AActor",
                name=blueprint_path,
                as_handle=True,
            )
        )
        observed_blueprint_handle = int(
            game.unreal_service.get_class(
                uobject=runtime["visual_actor"],
                as_handle=True,
            )
        )
        blueprint_match = observed_blueprint_handle == expected_blueprint_handle
        _require(blueprint_match, f"{actor_id} live Blueprint class mismatch")

        mesh_path = declaration.get("skeletal_mesh_path")
        skeleton_path = declaration.get("skeleton_path")
        _require(
            isinstance(mesh_path, str) and mesh_path,
            f"{actor_id} skeletal mesh path is missing",
        )
        _require(
            isinstance(skeleton_path, str) and skeleton_path,
            f"{actor_id} Skeleton path is missing",
        )
        expected_mesh_handle = int(
            game.unreal_service.load_object(
                uclass="USkeletalMesh",
                name=mesh_path,
                as_handle=True,
            )
        )
        observed_mesh_handle, mesh_method = RUNNER._skeletal_mesh_handle(
            runtime["component"]
        )
        _require(
            observed_mesh_handle == expected_mesh_handle,
            f"{actor_id} live SkeletalMesh mismatch",
        )
        observed_mesh = game.get_unreal_object(uobject=observed_mesh_handle)
        try:
            observed_skeleton_handle = int(observed_mesh.GetSkeleton(as_handle=True))
            skeleton_readback_method = "USkeletalMesh.GetSkeleton"
        except (AttributeError, RuntimeError):
            observed_skeleton_handle = int(
                observed_mesh.get_property_value(
                    property_name="Skeleton",
                    as_handle=True,
                )
            )
            skeleton_readback_method = "USkeletalMesh.Skeleton_property"
        expected_skeleton_handle = int(
            game.unreal_service.load_object(
                uclass="USkeleton",
                name=skeleton_path,
                as_handle=True,
            )
        )
        _require(
            observed_skeleton_handle == expected_skeleton_handle,
            f"{actor_id} live Skeleton mismatch",
        )

        idle_path = declaration.get("idle_animation")
        _require(
            isinstance(idle_path, str) and idle_path,
            f"{actor_id} Standing_Idle path is missing",
        )
        walking_path = declaration.get("walking_animation")
        _require(
            isinstance(walking_path, str) and walking_path,
            f"{actor_id} Walking path is missing",
        )
        action_paths = {"idle": idle_path, "walk": walking_path}
        action_id = state.get("action_id")
        _require(
            action_id in action_paths,
            f"{actor_id} frame action is not Idle/Walking",
        )
        action_path = action_paths[action_id]
        _require(
            state.get("ue_animation") == action_path,
            f"{actor_id} frame action/animation binding mismatch",
        )
        expected_idle_handle = int(
            game.unreal_service.load_object(
                uclass="UAnimationAsset",
                name=idle_path,
                as_handle=True,
            )
        )
        runtime_idle_handle = int(runtime["animations"][idle_path].uobject)
        _require(
            runtime_idle_handle == expected_idle_handle,
            f"{actor_id} Standing_Idle asset was not loaded exactly",
        )
        expected_action_handle = int(
            game.unreal_service.load_object(
                uclass="UAnimationAsset",
                name=action_path,
                as_handle=True,
            )
        )
        runtime_action_handle = int(runtime["animations"][action_path].uobject)
        try:
            anim_instance_handle = int(
                runtime["component"].GetAnimInstance(as_handle=True)
            )
            anim_instance_readback_method = "USkeletalMeshComponent.GetAnimInstance"
        except (AttributeError, RuntimeError):
            anim_instance_handle = int(
                runtime["component"].get_property_value(
                    property_name="AnimScriptInstance",
                    as_handle=True,
                )
            )
            anim_instance_readback_method = (
                "USkeletalMeshComponent.AnimScriptInstance_property"
            )
        _require(
            anim_instance_handle != 0,
            f"{actor_id} has no live AnimScriptInstance",
        )
        anim_instance = game.get_unreal_object(uobject=anim_instance_handle)
        try:
            observed_action_handle = int(
                anim_instance.GetAnimationAsset(as_handle=True)
            )
            action_readback_method = "UAnimSingleNodeInstance.GetAnimationAsset"
        except (AttributeError, RuntimeError):
            observed_action_handle = int(
                anim_instance.get_property_value(
                    property_name="CurrentAsset",
                    as_handle=True,
                )
            )
            action_readback_method = "UAnimSingleNodeInstance.CurrentAsset_property"
        _require(
            runtime_action_handle == expected_action_handle
            and observed_action_handle == expected_action_handle
            and runtime["current_animation"] == action_path,
            f"{actor_id} live current-action binding mismatch",
        )
        observed_animation_seconds = float(runtime["component"].GetPosition())
        expected_animation_seconds = RUNNER.animation_position_seconds(
            float(state["action_phase"]),
            float(runtime["lengths"][action_path]),
        )
        animation_error = abs(observed_animation_seconds - expected_animation_seconds)
        _require(
            animation_error <= RUNNER.ANIMATION_TOLERANCE_SECONDS,
            f"{actor_id} live current-action position mismatch",
        )

        anchor_readback = RUNNER._actor_readback(
            runtime["anchor"], int(frame["frame_index"])
        )
        location_cm = anchor_readback["location_cm"]
        observed_root_m = [
            float(location_cm[0]) / 100.0,
            float(location_cm[2]) / 100.0,
            float(location_cm[1]) / 100.0,
        ]
        expected_root_m = [float(value) for value in state["translation_m"]]
        root_error_m = max(
            abs(observed - expected)
            for observed, expected in zip(observed_root_m, expected_root_m)
        )
        _require(root_error_m <= 1.0e-6, f"{actor_id} live root readback drift")
        emitter_offset_m = declaration.get("emitter_offset_m")
        _require(
            isinstance(emitter_offset_m, Sequence) and len(emitter_offset_m) == 3,
            f"{actor_id} emitter offset is missing",
        )
        observed_emitter_m = [
            observed_root_m[index] + float(emitter_offset_m[index])
            for index in range(3)
        ]
        expected_emitter_m = [
            expected_root_m[index] + float(emitter_offset_m[index])
            for index in range(3)
        ]
        emitter_error_m = max(
            abs(observed - expected)
            for observed, expected in zip(observed_emitter_m, expected_emitter_m)
        )
        _require(
            emitter_error_m <= 1.0e-6,
            f"{actor_id} live emitter binding drift",
        )
        ground_contact = _runtime_ground_contact_readback(
            game=game,
            runtimes=runtimes,
            actor_id=actor_id,
        )
        records[instance_id] = {
            "status": "pass",
            "actor_id": actor_id,
            "asset_id": declaration["asset_id"],
            "asset_revision": declaration["asset_revision"],
            "stable_actor_tag": {
                "status": "pass",
                "value": stable_name,
                "descriptor_match_count": len(raw_ids),
                "raw_object_ids": raw_ids,
            },
            "blueprint": {
                "status": "pass",
                "expected_path": blueprint_path,
                "expected_class_handle": expected_blueprint_handle,
                "observed_class_handle": observed_blueprint_handle,
                "spawned_actor_exact_class_match": blueprint_match,
            },
            "skeletal_mesh": {
                "status": "pass",
                "expected_path": mesh_path,
                "expected_handle": expected_mesh_handle,
                "observed_handle": observed_mesh_handle,
                "readback_method": mesh_method,
            },
            "skeleton": {
                "status": "pass",
                "expected_path": skeleton_path,
                "expected_handle": expected_skeleton_handle,
                "observed_mesh_skeleton_handle": observed_skeleton_handle,
                "readback_method": skeleton_readback_method,
            },
            "standing_idle": {
                "status": "pass",
                "expected_path": idle_path,
                "expected_handle": expected_idle_handle,
                "runtime_loaded_handle": runtime_idle_handle,
                "play_length_seconds": float(runtime["lengths"][idle_path]),
            },
            "current_action": {
                "status": "pass",
                "action_id": action_id,
                "expected_path": action_path,
                "expected_handle": expected_action_handle,
                "runtime_loaded_handle": runtime_action_handle,
                "anim_script_instance_handle": anim_instance_handle,
                "anim_instance_readback_method": anim_instance_readback_method,
                "observed_animation_asset_handle": observed_action_handle,
                "readback_method": action_readback_method,
                "current_animation_path": runtime["current_animation"],
                "play_length_seconds": float(runtime["lengths"][action_path]),
                "expected_position_seconds": expected_animation_seconds,
                "observed_position_seconds": observed_animation_seconds,
                "absolute_position_error_seconds": animation_error,
            },
            "emitter_native_readback": {
                "status": "pass",
                "authority": "native_actor_root_plus_declared_profile_offset",
                "claim_boundary": (
                    "does_not_claim_live_mouth_bone_or_socket_geometry_readback"
                ),
                "anchor_id": declaration["emitter_anchor_id"],
                "offset_m": [float(value) for value in emitter_offset_m],
                "observed_root_m": observed_root_m,
                "expected_root_m": expected_root_m,
                "observed_world_emitter_m": observed_emitter_m,
                "expected_world_emitter_m": expected_emitter_m,
                "maximum_absolute_error_m": emitter_error_m,
            },
            "live_ground_contact_readback": ground_contact,
        }
    return {
        "schema": "avengine_native_spear_runtime_asset_readbacks_v1",
        "status": "pass",
        "frame_index": int(frame["frame_index"]),
        "per_instance": records,
    }


def _bundle_runtime_asset_samples(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(
        [int(sample.get("frame_index", -1)) for sample in samples]
        == list(RUNTIME_ASSET_SAMPLE_FRAME_INDICES),
        "runtime asset sample frame closure failed",
    )
    _require(
        all(sample.get("status") == "pass" for sample in samples),
        "runtime asset sample failed",
    )
    bundled = deepcopy(dict(samples[-1]))
    bundled["sampling_contract"] = {
        "schema": "avengine_native_spear_runtime_asset_sampling_v1",
        "status": "pass",
        "frame_indices": list(RUNTIME_ASSET_SAMPLE_FRAME_INDICES),
        "purpose": (
            "close actor root, emitter, live action asset, animation position, "
            "and measurement-only foot/toe-to-floor traces at the beginning, "
            "midpoint, and end of the full75 Episode"
        ),
    }
    bundled["sampled_frames"] = [deepcopy(dict(sample)) for sample in samples]
    return bundled


def _derive_masks(
    *,
    normal_depths: Sequence[np.ndarray],
    target_depths_by_instance: Mapping[str, Sequence[np.ndarray]],
    semantic_ids: Mapping[str, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frame_count = len(normal_depths)
    height, width = normal_depths[0].shape
    modal = np.zeros((frame_count, height, width), dtype=np.uint8)
    target_masks = {
        instance_id: np.zeros((frame_count, height, width), dtype=np.uint8)
        for instance_id in semantic_ids
    }
    for frame_index, normal_value in enumerate(normal_depths):
        normal = np.asarray(normal_value, dtype=np.float32)
        best_residual = np.full((height, width), np.inf, dtype=np.float32)
        for instance_id in sorted(semantic_ids):
            target = np.asarray(
                target_depths_by_instance[instance_id][frame_index],
                dtype=np.float32,
            )
            footprint = target < TARGET_ONLY_BACKGROUND_DEPTH_M
            target_masks[instance_id][frame_index][footprint] = semantic_ids[
                instance_id
            ]
            residual = np.abs(normal - target)
            tolerance = ABSOLUTE_TOLERANCE_M + RELATIVE_TOLERANCE * target
            visible = footprint & (residual <= tolerance)
            wins = visible & (residual < best_residual)
            modal[frame_index][wins] = semantic_ids[instance_id]
            best_residual[wins] = residual[wins]
    return modal, target_masks


def _maximum_readback_drift(
    normal_readbacks: Sequence[Mapping[str, Any]],
    target_readbacks: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    maximum_location_cm = 0.0
    maximum_rotation_deg = 0.0
    for readbacks in target_readbacks.values():
        _require(
            len(readbacks) == len(normal_readbacks),
            "target-only runtime readback count differs from normal pass",
        )
        for normal, target in zip(normal_readbacks, readbacks):
            _require(
                normal["camera"]["expected_pose_hash"]
                == target["camera"]["expected_pose_hash"],
                "target-only camera pose hash differs from normal pass",
            )
            for owner in ["camera", *sorted(normal["actors"])]:
                normal_value = (
                    normal[owner] if owner == "camera" else normal["actors"][owner]
                )
                target_value = (
                    target[owner] if owner == "camera" else target["actors"][owner]
                )
                location = max(
                    abs(float(left) - float(right))
                    for left, right in zip(
                        normal_value["location_cm"], target_value["location_cm"]
                    )
                )
                rotation = max(
                    abs(((float(left) - float(right) + 180.0) % 360.0) - 180.0)
                    for left, right in zip(
                        normal_value["rotation_deg"], target_value["rotation_deg"]
                    )
                )
                maximum_location_cm = max(maximum_location_cm, location)
                maximum_rotation_deg = max(maximum_rotation_deg, rotation)
    _require(maximum_location_cm <= 1.0e-4, "target pass location drift")
    _require(maximum_rotation_deg <= 1.0e-4, "target pass rotation drift")
    return {
        "normal_frame_count": len(normal_readbacks),
        "target_pass_count": len(target_readbacks),
        "maximum_location_drift_cm": maximum_location_cm,
        "maximum_rotation_drift_deg": maximum_rotation_deg,
    }


def _run_checked(command: Sequence[str]) -> None:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed: "
            + " ".join(command)
            + "\n"
            + completed.stdout
            + completed.stderr
        )


def _ffprobe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def run(args: argparse.Namespace) -> Path:
    import cv2

    suite = json.loads(args.suite_plan.read_text(encoding="utf-8"))
    scenarios = {item["scenario_id"]: item for item in suite["scenarios"]}
    _require(args.scenario_id in scenarios, "scenario is absent from suite plan")
    scenario = scenarios[args.scenario_id]
    all_frames = scenario["plan"]["frames"]
    _require(
        len(all_frames) == RUNNER.FRAME_COUNT,
        "scenario is not a 75-frame Episode",
    )
    selected_indices = (
        list(range(RUNNER.FRAME_COUNT))
        if args.frame_index is None
        else list(args.frame_index)
    )
    _require(
        selected_indices
        and len(selected_indices) == len(set(selected_indices))
        and all(0 <= index < RUNNER.FRAME_COUNT for index in selected_indices),
        "selected formal frame indices must be unique values in [0,74]",
    )
    frames = [all_frames[index] for index in selected_indices]
    _require(args.audio_wav.is_file(), "authoritative binaural WAV is missing")
    args.output.mkdir(parents=True)
    rgb_directory = args.output / "rgb_frames"
    rgb_directory.mkdir()

    configure_args = argparse.Namespace(
        spear_root=args.spear_root,
        rpc_port=args.rpc_port,
        graphics_adapter=args.graphics_adapter,
    )
    instance, spear_root = RUNNER._configure_instance(
        configure_args, native_map=str(suite["native_map"])
    )
    game = instance.get_game()
    runtimes: dict[str, Any] = {}
    components: dict[str, Any] = {}
    camera = None
    normal_depths: list[np.ndarray] = []
    normal_object_ids: list[np.ndarray] = []
    normal_readbacks: list[dict[str, Any]] = []
    target_depths: dict[str, list[np.ndarray]] = {}
    target_readbacks: dict[str, list[dict[str, Any]]] = {}
    target_object_id_foreground_counts: dict[str, list[int]] = {}
    descriptors: list[dict[str, Any]] = []
    stable_names: dict[str, str] = {}
    direct_modal_pixel_counts: dict[str, list[int]] = {}
    runtime_asset_samples: list[dict[str, Any]] = []
    runtime_asset_readbacks: dict[str, Any] = {}
    try:
        with instance.begin_frame():
            camera, components = SPIKE._spawn_multimodal_camera(game)
            runtimes = RUNNER._spawn_runtime_actors(game, scenario, spear_root)
            for actor_id, runtime in runtimes.items():
                stable_name = f"lead_a_native_{actor_id}"
                game.unreal_service.set_stable_name_for_actor(
                    actor=runtime["visual_actor"], stable_name=stable_name
                )
                stable_names[actor_id.removesuffix("_actor")] = stable_name
            _apply_exact_frame_with_ground_snap(
                game=game,
                camera=camera,
                runtimes=runtimes,
                scenario=scenario,
                frame=frames[0],
            )
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                bPaused=False
            )
        with instance.end_frame():
            pass
        instance.step(num_frames=args.warmup_frames)

        with instance.begin_frame():
            game.segmentation_service.initialize()
            components["depth"].PrimitiveRenderMode = "PRM_RenderScenePrimitives"
            components["depth"].ShowOnlyActors = []
            _apply_exact_frame_with_ground_snap(
                game=game,
                camera=camera,
                runtimes=runtimes,
                scenario=scenario,
                frame=frames[0],
            )
        with instance.end_frame():
            pass
        instance.step(num_frames=2)

        with instance.begin_frame():
            raw_descriptors = game.segmentation_service.get_mesh_proxy_geometry_descs(
                include_debug_info=False, as_global=True
            )
        with instance.end_frame():
            pass
        descriptors = [_safe_descriptor(item) for item in raw_descriptors]
        raw_ids_by_instance = {
            instance_id: _descriptor_raw_ids(raw_descriptors, stable_name)
            for instance_id, stable_name in stable_names.items()
        }
        _require(
            all(raw_ids_by_instance.values()),
            "dynamic source proxy descriptors are missing",
        )
        direct_modal_pixel_counts = {instance_id: [] for instance_id in stable_names}

        for capture_index, frame in enumerate(frames):
            with instance.begin_frame():
                readback = _apply_exact_frame_with_ground_snap(
                    game=game,
                    camera=camera,
                    runtimes=runtimes,
                    scenario=scenario,
                    frame=frame,
                )
                if int(frame["frame_index"]) in RUNTIME_ASSET_SAMPLE_FRAME_INDICES:
                    runtime_asset_samples.append(
                        _runtime_asset_readbacks(
                            game=game,
                            scenario=scenario,
                            runtimes=runtimes,
                            stable_names=stable_names,
                            raw_descriptors=raw_descriptors,
                            frame=frame,
                        )
                    )
            with instance.end_frame():
                rgb = _rgb_bgr(components["rgb"])
                depth = _depth_native(components["depth"])
                raw_ids = _raw_object_ids(components["object_ids"])
            _require(
                cv2.imwrite(str(rgb_directory / f"frame_{capture_index:06d}.png"), rgb),
                f"could not write RGB frame {frame['frame_index']}",
            )
            normal_depths.append(depth)
            normal_object_ids.append(raw_ids)
            normal_readbacks.append(readback)
            for instance_id, expected_raw_ids in raw_ids_by_instance.items():
                direct_modal_pixel_counts[instance_id].append(
                    int(
                        np.count_nonzero(
                            np.isin(
                                raw_ids,
                                np.asarray(expected_raw_ids, dtype=np.uint32),
                            )
                        )
                    )
                )

        for instance_id in sorted(stable_names):
            actor_id = f"{instance_id}_actor"
            with instance.begin_frame():
                manager = game.segmentation_service.proxy_component_manager
                manager.SetAllowedActors(
                    AllowedActors=[runtimes[actor_id]["visual_actor"]]
                )
                game.segmentation_service.initialize()
                components["depth"].PrimitiveRenderMode = "PRM_UseShowOnlyList"
                components["depth"].ShowOnlyActors = [
                    runtimes[actor_id]["visual_actor"]
                ]
                _apply_exact_frame_with_ground_snap(
                    game=game,
                    camera=camera,
                    runtimes=runtimes,
                    scenario=scenario,
                    frame=frames[0],
                )
            with instance.end_frame():
                pass
            instance.step(num_frames=2)
            target_depths[instance_id] = []
            target_readbacks[instance_id] = []
            target_object_id_foreground_counts[instance_id] = []
            for frame in frames:
                with instance.begin_frame():
                    readback = _apply_exact_frame_with_ground_snap(
                        game=game,
                        camera=camera,
                        runtimes=runtimes,
                        scenario=scenario,
                        frame=frame,
                    )
                with instance.end_frame():
                    depth = _depth_native(components["depth"])
                    raw_ids = _raw_object_ids(components["object_ids"])
                target_depths[instance_id].append(depth)
                target_readbacks[instance_id].append(readback)
                target_object_id_foreground_counts[instance_id].append(
                    int(np.count_nonzero(raw_ids != 0))
                )
        runtime_asset_readbacks = _bundle_runtime_asset_samples(runtime_asset_samples)
    finally:
        if runtimes:
            try:
                RUNNER._destroy_runtime_actors(instance, runtimes)
            except Exception:  # noqa: BLE001, S110
                pass
        instance.close(force=True)

    frame_indices = [int(frame["frame_index"]) for frame in frames]
    camera_pose_ids = [frame["camera_state"]["pose_hash"] for frame in frames]
    semantic_ids = {"source1": 1, "source2": 2}
    _require(set(target_depths) == set(semantic_ids), "unexpected source slots")
    common_context = {
        "renderer_backend": "spear_unreal_native_apartment",
        "rgb_renderer_backend": "spear_unreal_native_apartment",
        "camera_contract_id": "lead_a_native_spear_bp_camera_sensor_v1",
        "semantic_id_namespace": "lead_a_native_spear_metric_depth_instances_v1",
        "resolution_hw": [RUNNER.HEIGHT, RUNNER.WIDTH],
        "frame_indices": frame_indices,
        "camera_pose_ids": camera_pose_ids,
    }
    truth = compile_depth_pixel_visibility_truth(
        normal_depth_m_frames=normal_depths,
        target_only_depth_m_frames_by_instance=target_depths,
        semantic_ids_by_instance=semantic_ids,
        normal_context={"pass_kind": "modal_scene", **common_context},
        target_only_contexts_by_instance={
            instance_id: {
                "pass_kind": "target_only",
                "target_instance_id": instance_id,
                **common_context,
            }
            for instance_id in semantic_ids
        },
        target_only_background_depth_m=TARGET_ONLY_BACKGROUND_DEPTH_M,
        absolute_tolerance_m=ABSOLUTE_TOLERANCE_M,
        relative_tolerance=RELATIVE_TOLERANCE,
    )
    _require(
        truth["authority"] == PIXEL_VISIBILITY_DEPTH_AUTHORITY,
        "pixel compiler returned the wrong authority",
    )
    modal_masks, target_masks = _derive_masks(
        normal_depths=normal_depths,
        target_depths_by_instance=target_depths,
        semantic_ids=semantic_ids,
    )
    alignment = _maximum_readback_drift(normal_readbacks, target_readbacks)

    normal_depth_array = np.stack(normal_depths)
    normal_object_id_array = np.stack(normal_object_ids)
    target_depth_arrays = {
        instance_id: np.stack(values) for instance_id, values in target_depths.items()
    }
    depth_path = args.output / "metric_depth_native.npz"
    np.savez_compressed(
        depth_path,
        normal_depth_m=normal_depth_array,
        target_only_source1_depth_m=target_depth_arrays["source1"],
        target_only_source2_depth_m=target_depth_arrays["source2"],
    )
    object_id_path = args.output / "normal_object_ids_uint32.npz"
    np.savez_compressed(object_id_path, normal_object_ids=normal_object_id_array)
    mask_path = args.output / "native_pixel_masks_depth_authority_v1.npz"
    np.savez_compressed(
        mask_path,
        depth_derived_modal_semantic=modal_masks,
        modal_visible_source1=modal_masks == semantic_ids["source1"],
        modal_visible_source2=modal_masks == semantic_ids["source2"],
        target_only_source1=target_masks["source1"],
        target_only_source2=target_masks["source2"],
    )
    truth_path = args.output / "pixel_visibility_truth.json"
    _write_json(truth_path, truth)
    readbacks_path = args.output / "runtime_readbacks.json"
    _write_json(
        readbacks_path,
        {
            "schema": "avengine_native_spear_multimodal_runtime_readbacks_v1",
            "normal": normal_readbacks,
            "target_only": target_readbacks,
        },
    )
    descriptor_path = args.output / "normal_object_id_descriptors.json"
    _write_json(
        descriptor_path,
        {
            "schema": "avengine_native_spear_object_id_descriptors_v1",
            "descriptors": descriptors,
        },
    )
    runtime_asset_readbacks_path = args.output / "runtime_asset_readbacks.json"
    _write_json(runtime_asset_readbacks_path, runtime_asset_readbacks)

    visual_path = args.output / "native_rgb_visual_only.mp4"
    muxed_path = args.output / "native_rgb_binaural.mp4"
    _run_checked(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(RUNNER.FPS),
            "-start_number",
            "0",
            "-i",
            str(rgb_directory / "frame_%06d.png"),
            "-frames:v",
            str(len(frames)),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(RUNNER.FPS),
            str(visual_path),
        ]
    )
    _run_checked(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(visual_path),
            "-i",
            str(args.audio_wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-ar",
            "16000",
            "-ac",
            "2",
            "-shortest",
            str(muxed_path),
        ]
    )
    video_probe = _ffprobe(muxed_path)
    intrinsics = SPIKE._camera_intrinsics(
        width=RUNNER.WIDTH, height=RUNNER.HEIGHT, hfov_deg=105.0
    )
    depth_statistics = {
        "dtype": str(normal_depth_array.dtype),
        "shape_nhw": list(normal_depth_array.shape),
        "units": "meters",
        "finite_positive_fraction": float(
            np.mean(np.isfinite(normal_depth_array) & (normal_depth_array > 0.0))
        ),
        "minimum_m": float(normal_depth_array.min()),
        "maximum_m": float(normal_depth_array.max()),
        "per_frame_minimum_m": [float(np.min(frame)) for frame in normal_depth_array],
        "per_frame_maximum_m": [float(np.max(frame)) for frame in normal_depth_array],
    }
    artifact_paths = {
        "rgb_frames": rgb_directory,
        "native_rgb_visual_only": visual_path,
        "native_rgb_binaural": muxed_path,
        "metric_depth": depth_path,
        "normal_object_ids": object_id_path,
        "pixel_masks": mask_path,
        "pixel_visibility_truth": truth_path,
        "runtime_readbacks": readbacks_path,
        "runtime_asset_readbacks": runtime_asset_readbacks_path,
        "object_id_descriptors": descriptor_path,
    }
    artifact_records = {
        name: (
            _directory_artifact_record(path)
            if path.is_dir()
            else _file_artifact_record(path)
        )
        for name, path in artifact_paths.items()
    }
    manifest = {
        "schema": SCHEMA,
        "status": "pass",
        "benchmark_qualification_claim": False,
        "native_pixel_fact_binding_claim": True,
        "claim_boundary": (
            "native SPEAR Apartment RGB/depth/pixel evidence bound to an existing "
            "controlled Episode; this optional comparison render is not a release "
            "qualification claim"
        ),
        "scenario_id": args.scenario_id,
        "authoritative_capture_request": scenario.get("authoritative_capture_request"),
        "static_camera_upgrade": scenario.get("static_camera_upgrade"),
        "native_map": suite["native_map"],
        "frame_contract": {
            "frame_count": len(frames),
            "formal_episode_frame_count": len(all_frames),
            "captured_frame_indices": frame_indices,
            "frame_rate_hz": RUNNER.FPS,
            "resolution_hw": [RUNNER.HEIGHT, RUNNER.WIDTH],
            "camera_pose_ids": camera_pose_ids,
        },
        "camera_intrinsics": intrinsics,
        "runtime_alignment": alignment,
        "runtime_assets": {
            "status": runtime_asset_readbacks["status"],
            "frame_index": runtime_asset_readbacks["frame_index"],
            "per_instance_status": {
                instance_id: record["status"]
                for instance_id, record in runtime_asset_readbacks[
                    "per_instance"
                ].items()
            },
        },
        "pixel_visibility": {
            "authority": truth["authority"],
            "mask_array_contract": {
                "depth_derived_modal_semantic": (
                    "uint8 semantic IDs derived from metric depth; not direct "
                    "raw skeletal object IDs"
                ),
                "modal_visible_source1": "boolean depth-derived modal mask",
                "modal_visible_source2": "boolean depth-derived modal mask",
                "target_only_source1": "uint8 semantic-ID target footprint",
                "target_only_source2": "uint8 semantic-ID target footprint",
            },
            "per_instance_state_counts": {
                instance_id: entry["state_counts"]
                for instance_id, entry in truth["per_instance"].items()
            },
        },
        "direct_object_id_limitation": {
            "dynamic_skeletal_raw_ids_usable": all(
                any(count > 0 for count in counts)
                for counts in direct_modal_pixel_counts.values()
            ),
            "per_instance_direct_modal_pixel_counts": direct_modal_pixel_counts,
            "reason": (
                "cooked M_Emissive lacks bUsedWithSkeletalMesh; static environment "
                "object IDs remain retained for occluder analysis"
            ),
            "target_only_object_id_foreground_counts": (
                target_object_id_foreground_counts
            ),
        },
        "metric_depth": depth_statistics,
        "audio": {
            "authoritative_wav": str(args.audio_wav.resolve()),
            "sha256": _sha256(args.audio_wav),
        },
        "artifacts": {
            name: str(path.resolve()) for name, path in artifact_paths.items()
        },
        "artifact_records": artifact_records,
        "sha256": {
            name: record["sha256"]
            for name, record in artifact_records.items()
            if record["kind"] == "file"
        },
        "ffprobe": video_probe,
    }
    manifest_path = args.output / "manifest.json"
    _write_json(manifest_path, manifest)
    print(f"SPEAR_NATIVE_PIXEL_EPISODE_OK output={args.output}", flush=True)
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-plan", required=True, type=Path)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--audio-wav", required=True, type=Path)
    parser.add_argument("--spear-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rpc-port", type=int, default=39474)
    parser.add_argument("--graphics-adapter", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=40)
    parser.add_argument(
        "--frame-index",
        action="append",
        type=int,
        help="Capture only this formal Episode frame; repeat for a sparse capture.",
    )
    args = parser.parse_args(argv)
    if not 1024 <= args.rpc_port <= 65535:
        parser.error("--rpc-port must be in [1024,65535]")
    if args.graphics_adapter < 0 or args.warmup_frames < 0:
        parser.error("GPU and warmup values must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
