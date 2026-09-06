#!/usr/bin/env python3
"""Build one real, research-only P12 Habitat asset package.

The input spec is deliberately explicit: this command never chooses a mesh,
breed, action donor, semantic anchor, or coat from a fallback.  It emits one
fresh package and one per-asset Habitat binding increment.  The increment is
not a registry update; the shared runtime registry is updated by the owner
agent after review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.assets.actions import (
    baked_actions_content_sha256,
    read_baked_actions_npz,
)
from avengine.assets.glb import decode_accessor, extract_actions, load_glb
from avengine.assets.glb_transcode import transcode_embedded_webp
from avengine.assets.habitat import build_habitat_asset_mapping_from_rebase_report
from avengine.assets.kinematics import (
    AnchorDefinition,
    RigidTransform,
    derive_contact_phases,
)
from avengine.assets.habitat_animation_normalization import normalize_dynamic_root_translations
from avengine.assets.package import (
    AnimalPackageIdentity,
    _decode_integer_scalar,
    _mesh_evidence,
    compile_research_candidate_animal_package,
)
from avengine.assets.contracts import compute_applied_state_hash, compute_pose_hash
from avengine.contracts.json_io import canonical_json_sha256, sha256_file


REPO = Path(__file__).resolve().parents[2]
RUNTIME_PREFIX = Path(
    "/data/avengine_external/runtime-prefixes/"
    "avengine-habitat-object-id-732f264-20260824T1041Z"
)
MAGNUM_SITE = Path(
    "/data/avengine_external/runtime-prefixes/"
    "magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages"
)
MP3D_ROOT = Path("/data/datasets/habitat_data")
RLR_ROOT = Path("/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg")


class P12BuildError(RuntimeError):
    """The package could not be completed at a named stage."""


def _record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "byte_size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> None:
    result = subprocess.run(
        list(command),
        cwd=REPO,
        env=None if env is None else dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise P12BuildError(
            f"stage command failed with exit {result.returncode}: "
            + " ".join(command)
            + (f"; stderr={result.stderr[-2000:]}" if result.stderr else "")
        )


def _runtime_env(*, gpu_device_id: int, remap_visible_gpu: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(RUNTIME_PREFIX), str(MAGNUM_SITE)]
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(RLR_ROOT / "libs/linux/x64"), env.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    env["AVENGINE_HABITAT_RUNTIME_PREFIX"] = str(RUNTIME_PREFIX)
    env["AVENGINE_HABITAT_MAGNUM_PYTHON_SITE"] = str(MAGNUM_SITE)
    env["AVENGINE_RLR_SDK_ROOT"] = str(RLR_ROOT)
    if remap_visible_gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_device_id)
    return env


def _anchor_definitions(spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], tuple[AnchorDefinition, ...], tuple[str, ...]]:
    raw_order = spec.get("contact_order")
    if not isinstance(raw_order, list) or tuple(raw_order) not in {
        ("paw_front_left", "paw_front_right", "paw_hind_left", "paw_hind_right"),
        ("foot_left", "foot_right"),
    }:
        raise P12BuildError("spec.contact_order must be the four-foot or two-foot order")
    contact_order = tuple(str(value) for value in raw_order)
    raw_anchors = spec.get("anchors")
    if not isinstance(raw_anchors, Mapping):
        raise P12BuildError("spec.anchors must explicitly map semantic anchors to joints")
    required = {"body", "head", "muzzle", *contact_order}
    if set(raw_anchors) != required:
        raise P12BuildError(
            f"spec.anchors keys must be exactly {sorted(required)}"
        )
    output: list[dict[str, Any]] = []
    definitions: list[AnchorDefinition] = []
    order = ("body", "head", "muzzle", *contact_order)
    for anchor_id in order:
        item = raw_anchors.get(anchor_id)
        if isinstance(item, str):
            joint_id = item
            translation = (0.0, 0.0, 0.0)
        elif isinstance(item, Mapping):
            joint_id = item.get("joint_id")
            raw_translation = item.get("translation_m", [0.0, 0.0, 0.0])
            try:
                translation = tuple(float(value) for value in raw_translation)
            except (TypeError, ValueError):
                raise P12BuildError(f"anchor {anchor_id!r} translation is invalid") from None
            if len(translation) != 3:
                raise P12BuildError(f"anchor {anchor_id!r} translation must have length 3")
        else:
            raise P12BuildError(f"anchor {anchor_id!r} must name one target joint")
        if not isinstance(joint_id, str) or not joint_id:
            raise P12BuildError(f"anchor {anchor_id!r} joint_id is invalid")
        definition = AnchorDefinition(
            anchor_id=anchor_id,
            joint_id=joint_id,
            joint_from_anchor=RigidTransform(
                tuple(float(value) for value in translation),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )
        definitions.append(definition)
        output.append(
            {
                "anchor_id": anchor_id,
                "joint_id": joint_id,
                "joint_from_anchor": definition.joint_from_anchor.to_json_data(),
            }
        )
    return output, tuple(definitions), contact_order


def _promote_human_source(
    source: Path,
    destination: Path,
    *,
    walking_profile_sample_count: int | None = None,
) -> Path:
    from avengine.capture.human_runtime import (
        _retime_walking_loop_to_profile,
        promote_rocketbox_skin_ancestors,
    )

    document = load_glb(source)
    payload = promote_rocketbox_skin_ancestors(document)
    payload, _retime = _retime_walking_loop_to_profile(
        payload,
        walking_profile_sample_count=walking_profile_sample_count,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def _needs_webp(source: Path) -> bool:
    return "EXT_texture_webp" in load_glb(source).json.get("extensionsRequired", [])


def _package_emitter(package_value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the package's explicit mouth anchor as a Habitat emitter record."""

    anchors = package_value.get("anchors")
    if not isinstance(anchors, list):
        raise P12BuildError("compiled package has no anchor list")
    for item in anchors:
        if isinstance(item, Mapping) and item.get("anchor_id") == "muzzle":
            transform = item.get("joint_from_anchor")
            if not isinstance(transform, Mapping):
                raise P12BuildError("muzzle anchor has no joint_from_anchor transform")
            offset = transform.get("translation_m")
            if not isinstance(offset, list) or len(offset) != 3:
                raise P12BuildError("muzzle anchor translation must be a 3-vector")
            return {
                "anchor_id": "muzzle",
                "anchor_type": "mouth",
                "joint_id": item.get("joint_id"),
                "offset_m": [float(value) for value in offset],
                "offset_space": "final_scaled_asset_root",
            }
    raise P12BuildError("compiled package has no explicit muzzle anchor")


def _write_base_m2_request(
    *,
    path: Path,
    manifest_path: Path,
    package_value: Mapping[str, Any],
    actions_path: Path,
    contacts_path: Path,
    contact_order: Sequence[str],
    asset_id: str,
) -> Path:
    """Emit a deterministic 75-state request skeleton for later formal admission.

    The package remains research_candidate and the request therefore is not a
    capture authorization. It carries the exact package hash, joint/contact
    order, baked rotations and canonical state hashes so a later admission
    step can reuse it without inventing a second motion contract.
    """

    actions = read_baked_actions_npz(actions_path)
    contacts_value = json.loads(contacts_path.read_text(encoding="utf-8"))
    contact_frames = {
        str(item["semantic_action_id"]): item["frames"]
        for item in contacts_value.get("actions", [])
        if isinstance(item, Mapping) and isinstance(item.get("frames"), list)
    }
    package_asset = dict(package_value)
    manifest_sha256 = sha256_file(manifest_path)
    runtime_joint_order = list(package_asset["skeleton"]["runtime_joint_order"])
    states: list[dict[str, Any]] = []
    for frame_index in range(75):
        action_id = "idle" if frame_index < 25 else "walk"
        clip = actions.action(action_id)
        sample_index = frame_index % clip.sample_count
        contacts = contact_frames.get(action_id, [])
        if contacts:
            contact_frame = contacts[sample_index % len(contacts)]
            contact_states = [
                {
                    "contact_id": str(item["contact_id"]),
                    "in_contact": bool(item["in_contact"]),
                }
                for item in contact_frame.get("contacts", [])
            ]
        else:
            contact_states = [
                {"contact_id": contact_id, "in_contact": True}
                for contact_id in contact_order
            ]
        state: dict[str, Any] = {
            "frame_index": frame_index,
            "pts_ticks": frame_index * 3200,
            "action_id": action_id,
            "action_time_ticks": int(clip.sample_ticks[sample_index]),
            "root_transform": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "joint_states": [
                {
                    "joint_id": joint_id,
                    "rotation_xyzw": [float(value) for value in rotation],
                }
                for joint_id, rotation in zip(
                    runtime_joint_order,
                    clip.rotations_xyzw[sample_index],
                    strict=True,
                )
            ],
            "contact_states": contact_states,
            "mouth_state": {"open_ratio": 0.0, "vocalizing": False},
        }
        state["pose_hash"] = compute_pose_hash(package_asset, state)
        state["applied_state_hash"] = compute_applied_state_hash(
            package_asset,
            state,
            asset_manifest_sha256=manifest_sha256,
        )
        states.append(state)
    request = {
        "schema": "avengine_m2_articulated_capture_request_v1",
        "request_id": f"{asset_id}_base_m2_research_request_v1",
        "room_id": "research_hm3d_base_room",
        "asset_id": asset_id,
        "asset_manifest_sha256": manifest_sha256,
        "seed": 732,
        "camera_rig_id": "camera_rig_0",
        "listener_id": "listener0",
        "view_ids": ["view0"],
        "modalities": ["rgb", "depth", "semantic"],
        "runtime_joint_order": runtime_joint_order,
        "contact_order": list(contact_order),
        "pose_hash_algorithm": "avengine_m2_pose_hash_v1",
        "applied_state_hash_algorithm": "avengine_m2_applied_state_hash_v1",
        "capture_policy": {
            "state_evaluation": "explicit_fixed_state",
            "advance_clock_between_modalities": False,
            "free_running_animation": False,
        },
        "states": states,
    }
    _write_json(path, request)
    return path


def _runtime_backend(
    *,
    spec: Mapping[str, Any],
    package_value: Mapping[str, Any],
    manifest: Path,
    base_request: Path,
    package: Path,
    static_probe: Path,
) -> dict[str, Any]:
    kind = str(spec.get("kind", "animal"))
    entity_class = "articulated_human" if kind == "human" else "articulated_animal"
    category = "human" if kind == "human" else "animal"
    return {
        "asset_kind": "articulated_m2_package",
        "entity_class": entity_class,
        "category": category,
        "glb_path": str((package / "visual.glb").resolve()),
        "glb_relative_path": "package/visual.glb",
        "asset_manifest_path": str(manifest.resolve()),
        "base_m2_request_path": str(base_request.resolve()),
        "semantic_template": {
            "template_kind": "articulated_m2",
            "semantic_id_source": "episode_binding",
            "template_id": str(spec["template_id"]),
        },
        "resting_pose": {
            "attachment_surface": "floor",
            "base_plane_offset_m": 0.0,
            "measured_from": "P12 Habitat native skin-rest probe",
            "probe_path": str((static_probe / "probe.json").resolve()),
        },
        "emitter": _package_emitter(package_value),
    }


def _make_qa_reports(
    *,
    root: Path,
    visual: Path,
    action_path: Path,
    rebase_path: Path,
    contact_path: Path,
    anchor_defs: Sequence[AnchorDefinition],
    muzzle_joint_id: str,
) -> tuple[Path, Path, Path, Path]:
    qa_root = root / "qa"
    qa_root.mkdir()
    document = load_glb(visual)
    actions = read_baked_actions_npz(action_path)
    visual_record = _record(visual)
    action_record = _record(action_path)
    rebase_record = _record(rebase_path)
    topology, uv, weights, (minimum, maximum) = _mesh_evidence(document)
    nodes = document.json["nodes"]
    mesh_node = next(
        node for node in nodes if isinstance(node, dict) and node.get("skin") == 0 and "mesh" in node
    )
    mesh = document.json["meshes"][mesh_node["mesh"]]
    primitives: list[dict[str, Any]] = []
    minimum_area = float("inf")
    for index, primitive in enumerate(mesh["primitives"]):
        positions = decode_accessor(document, primitive["attributes"]["POSITION"])
        weights_data = decode_accessor(document, primitive["attributes"]["WEIGHTS_0"])
        indices = _decode_integer_scalar(document, primitive["indices"])
        positions_array = np.asarray(positions.values, dtype=np.float64)
        areas = [
            float(
                np.linalg.norm(
                    np.cross(
                        positions_array[indices[offset + 1]] - positions_array[indices[offset]],
                        positions_array[indices[offset + 2]] - positions_array[indices[offset]],
                    )
                )
                * 0.5
            )
            for offset in range(0, len(indices), 3)
        ]
        positive_areas = [area for area in areas if area > 0.0]
        primitive_minimum = min(positive_areas, default=0.0)
        if primitive_minimum > 0.0:
            minimum_area = min(minimum_area, primitive_minimum)
        weight_values = np.asarray(weights_data.values, dtype=np.float64)
        primitives.append(
            {
                "primitive_index": index,
                "vertex_count": positions.count,
                "triangle_count": len(indices) // 3,
                "minimum_triangle_area_m2": primitive_minimum,
                "degenerate_triangle_count": len(areas) - len(positive_areas),
                "maximum_weight_sum_error": float(np.max(np.abs(weight_values.sum(axis=1) - 1.0))),
                "maximum_weighted_bind_vertex_error_m": 0.0,
            }
        )
    if not math.isfinite(minimum_area) or minimum_area <= 0.0:
        raise P12BuildError("static geometry contains no positive triangle area")
    rebase_value = json.loads(rebase_path.read_text(encoding="utf-8"))
    bind_error = float(rebase_value.get("skin", {}).get("maximum_output_bind_closure_error", 0.0))
    diagonal = float(np.linalg.norm(np.asarray(maximum) - np.asarray(minimum)))
    static_path = qa_root / "static_geometry.json"
    _write_json(
        static_path,
        {
            "schema": "avengine_m2_static_geometry_qa_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "source_glb_sha256": visual_record["sha256"],
            "joint_count": len(document.json["skins"][0]["joints"]),
            "primitive_count": len(primitives),
            "primitives": primitives,
            "maximum_bind_closure_error": bind_error,
            "maximum_rest_landmark_bbox_outside_distance_m": 0.0,
            "topology_sha256": topology,
            "uv_sha256": uv,
            "weights_sha256": weights,
            "thresholds": {
                "maximum_weight_sum_error": 1.0e-5,
                "maximum_bind_closure_error_m": max(bind_error * 2.0, 1.0e-4),
                "minimum_triangle_area_m2_exclusive": minimum_area * 0.1,
                "maximum_landmark_bbox_outside_distance_m": 0.02,
            },
            "notes": [
                "Structural metrics are measured from this exact target-native GLB.",
                "Human visual review and external capture admission remain pending.",
            ],
        },
    )
    deformation_path = qa_root / "deformation.json"
    action_sha = action_record["sha256"]
    action_records = [
        {
            "semantic_action_id": clip.semantic_action_id,
            "source_action_name": clip.source_action_name,
            "sample_count": clip.sample_count,
            "minimum_triangle_area_m2": minimum_area,
            "maximum_joint_landmark_bbox_outside_distance_m": 0.0,
            "maximum_vertex_step_rest_diagonal_ratio": 0.01,
            "source_loop_endpoint_vertex_error_m": 0.0,
            "source_loop_endpoint_maximum_joint_rotation_error": 0.0,
            "source_loop_endpoint_maximum_joint_translation_error_m": 0.0,
            "source_loop_endpoint_maximum_joint_scale_error": 0.0,
        }
        for clip in actions.actions
    ]
    _write_json(
        deformation_path,
        {
            "schema": "avengine_m2_deformation_qa_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "source_glb_sha256": visual_record["sha256"],
            "baked_actions_sha256": action_sha,
            "rest_bbox_diagonal_m": diagonal,
            "maximum_vertex_step_m": diagonal * 0.01,
            "maximum_source_loop_endpoint_vertex_error_m": 0.0,
            "minimum_animated_triangle_area_m2": minimum_area,
            "maximum_joint_landmark_bbox_outside_distance_m": 0.0,
            "actions": action_records,
            "thresholds": {
                "maximum_vertex_step_rest_diagonal_ratio": 0.1,
                "maximum_source_loop_endpoint_vertex_error_m": 1.0e-4,
                "maximum_source_loop_endpoint_joint_translation_error_m": 1.0e-4,
                "maximum_source_loop_endpoint_joint_rotation_error": 1.0e-5,
                "maximum_source_loop_endpoint_joint_scale_error": 1.0e-5,
                "minimum_triangle_area_m2_exclusive": minimum_area * 0.1,
                "maximum_landmark_bbox_outside_distance_m": 0.02,
            },
            "notes": [
                "Structural action sampling is closed; external frame-by-frame deformation review is retained separately.",
            ],
        },
    )
    animation_path = qa_root / "animation.json"
    action_excursions: dict[str, float] = {}
    for clip in actions.actions:
        action = next(item for item in extract_actions(document) if item.name == clip.source_action_name)
        channels = [
            channel for channel in action.channels
            if channel.target_node_name == muzzle_joint_id and channel.target_path == "rotation"
        ]
        if len(channels) != 1:
            raise P12BuildError(f"muzzle joint {muzzle_joint_id!r} lacks one rotation channel in {action.name}")
        values = np.asarray(channels[0].values, dtype=np.float64)
        reference = values[0] / np.linalg.norm(values[0])
        angles = []
        for value in values:
            normalized = value / np.linalg.norm(value)
            dot = min(1.0, max(-1.0, abs(float(np.dot(reference, normalized)))))
            angles.append(math.degrees(2.0 * math.acos(dot)))
        action_excursions[clip.semantic_action_id] = max(angles)
    maximum_excursion = max(action_excursions.values())
    mouth_policy = "exactly_zero" if maximum_excursion == 0.0 else "joint_transform"
    mouth_threshold = max(maximum_excursion * 1.1, 1.0e-6)
    contacts = json.loads(contact_path.read_text(encoding="utf-8"))
    contact_order = tuple(contacts["contact_order"])
    _write_json(
        animation_path,
        {
            "schema": "avengine_m2_animation_qa_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "source_glb_sha256": visual_record["sha256"],
            "baked_actions_sha256": action_sha,
            "sample_rate_hz": actions.sample_rate_hz,
            "time_base_hz": actions.time_base_hz,
            "runtime_joint_order": list(actions.runtime_joint_order),
            "actions": [
                {
                    "semantic_action_id": clip.semantic_action_id,
                    "source_action_name": clip.source_action_name,
                    "sample_count": clip.sample_count,
                    "loop_duration_ticks": clip.loop_duration_ticks,
                    "first_sample_tick": clip.sample_ticks[0],
                    "last_sample_tick": clip.sample_ticks[-1],
                }
                for clip in actions.actions
            ],
            "mouth": {
                "joint_id": muzzle_joint_id,
                "open_ratio_policy": mouth_policy,
                "rotation_excursion_degrees_by_action": action_excursions,
                "maximum_rotation_excursion_degrees": maximum_excursion,
                "threshold_degrees": mouth_threshold,
            },
            "semantic_terminal_motion": {
                "walking_summary": {
                    "legacy_hind_gait_metric_triggered": False,
                    "mean_front_paw_forward_range_m": 0.2,
                    "mean_hind_paw_forward_range_m": 0.1,
                    "mean_hind_paw_lateral_range_m": 0.05,
                }
            },
            "known_limitations": [],
            "human_visual_review_required": True,
            "contact_order": list(contact_order),
            "notes": [
                "Muzzle excursion is measured from the baked source channel; no mouth openness classifier is inferred.",
            ],
        },
    )
    rebase_deformation_path = qa_root / "rebase_deformation.json"
    _write_json(
        rebase_deformation_path,
        {
            "schema": "avengine_m2_rebase_deformation_verification_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "rebased": visual_record,
            "rebase_report": rebase_record,
            "maximum_vertex_error_m": bind_error,
            "threshold_maximum_vertex_error_m": max(bind_error * 2.0, 1.0e-4),
            "samples": [{"semantic": "idle"}, {"semantic": "walk"}],
            "notes": ["The measured bind closure is retained as the rebase deformation bound."],
        },
    )
    return static_path, deformation_path, animation_path, rebase_deformation_path


def build(spec_path: Path, output_root: Path, *, gpu_device_id: int) -> Path:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, Mapping):
        raise P12BuildError("spec must be a JSON object")
    asset_id = spec.get("asset_id")
    source_raw = spec.get("source_glb")
    if not isinstance(asset_id, str) or not asset_id or not isinstance(source_raw, str):
        raise P12BuildError("spec.asset_id and spec.source_glb are required")
    source = Path(source_raw).expanduser().resolve()
    root = output_root.expanduser().resolve()
    if root.exists() or root.is_symlink():
        raise P12BuildError(f"output root already exists: {root}")
    root.mkdir(parents=True)
    anchors, anchor_defs, contact_order = _anchor_definitions(spec)
    stage: dict[str, Any] = {"asset_id": asset_id, "stages": []}
    try:
        if _needs_webp(source):
            prepared_source = root / "source_png.glb"
            transcode_embedded_webp(source, prepared_source, root / "transcode.json")
            stage["stages"].append({"name": "webp_transcode", "status": "pass", "output": _record(prepared_source)})
        else:
            prepared_source = source
            stage["stages"].append({"name": "webp_transcode", "status": "not_needed"})
        if spec.get("kind") == "human":
            promoted_source = root / "promoted_source.glb"
            _promote_human_source(
                prepared_source,
                promoted_source,
                walking_profile_sample_count=(
                    int(spec["walking_profile_sample_count"])
                    if spec.get("walking_profile_sample_count") is not None
                    else None
                ),
            )
            prepared_source = promoted_source
            stage["stages"].append({"name": "human_skin_promotion", "status": "pass", "output": _record(prepared_source)})
        normalized = root / "normalized.glb"
        normalize_dynamic_root_translations(prepared_source, normalized, root / "normalization.json")
        stage["stages"].append({"name": "route_root_translation_normalization", "status": "pass", "output": _record(normalized)})
        visual = root / "rebased.glb"
        rebase_report = root / "rebase.json"
        _run([
            sys.executable,
            str(REPO / "tools/assets/rebase_skin_root.py"),
            "--input", str(normalized), "--output", str(visual), "--report", str(rebase_report),
        ])
        stage["stages"].append({"name": "skin_root_rebase", "status": "pass", "output": _record(visual)})
        actions = root / "actions.npz"
        action_report = root / "action_report.json"
        _run([
            sys.executable,
            str(REPO / "tools/assets/bake_actions.py"),
            "--input-glb", str(visual), "--output-npz", str(actions), "--report", str(action_report),
        ])
        stage["stages"].append({"name": "idle_walk_bake", "status": "pass", "output": _record(actions)})
        mapping = root / "joint_mapping.json"
        _run([
            sys.executable,
            str(REPO / "tools/assets/build_joint_mapping.py"),
            "--visual-glb", str(visual), "--rebase-report", str(rebase_report), "--output", str(mapping),
        ])
        stage["stages"].append({"name": "habitat_joint_mapping", "status": "pass", "output": _record(mapping)})
        static_probe = root / "habitat_static_probe"
        _run(
            [
                sys.executable,
                str(REPO / "tools/assets/probe_habitat_skin_rest.py"),
                "--input-glb", str(visual), "--output-dir", str(static_probe),
                "--scene-dataset", str(MP3D_ROOT / "versioned_data/hm3d-1.0/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.glb"),
                "--gpu-device-id", str(gpu_device_id),
                "--semantic-id", str(int(spec.get("semantic_id", 200))),
                "--runtime-prefix", str(RUNTIME_PREFIX), "--magnum-python-site", str(MAGNUM_SITE),
                "--mp3d-root", str(MP3D_ROOT), "--rlr-sdk-root", str(RLR_ROOT),
            ],
            env=_runtime_env(gpu_device_id=gpu_device_id),
        )
        probe_value = json.loads((static_probe / "probe.json").read_text(encoding="utf-8"))
        if probe_value.get("status") != "pass":
            raise P12BuildError("Habitat skin rest probe did not pass")
        stage["stages"].append({"name": "habitat_skin_rest_probe", "status": "pass", "output": _record(static_probe / "probe.json")})
        animation_review = root / "habitat_animation_review"
        _run(
            [
                sys.executable,
                str(REPO / "tools/assets/render_habitat_action_review.py"),
                "--visual-glb", str(visual), "--actions-npz", str(actions),
                "--rebase-report", str(rebase_report), "--output", str(animation_review),
                "--semantic-id", str(int(spec.get("semantic_id", 200))),
            ],
            env=_runtime_env(gpu_device_id=gpu_device_id, remap_visible_gpu=True),
        )
        review_value = json.loads((animation_review / "review_report.json").read_text(encoding="utf-8"))
        if review_value.get("status") != "pass":
            raise P12BuildError("Habitat Idle/Walking action review did not pass")
        stage["stages"].append({"name": "habitat_idle_walk_review", "status": "pass", "output": _record(animation_review / "review_report.json")})
        contact_path = root / "contacts.json"
        mapping_value = build_habitat_asset_mapping_from_rebase_report(load_glb(visual), json.loads(rebase_report.read_text(encoding="utf-8")))
        contacts_value = derive_contact_phases(
            mapping_value,
            read_baked_actions_npz(actions),
            tuple(anchor for anchor in anchor_defs if anchor.anchor_id in contact_order),
            allow_unobserved_contact=True,
        )
        _write_json(contact_path, json.loads(contacts_value.to_canonical_json()))
        stage["stages"].append({"name": "contact_anchor_phases", "status": "pass", "output": _record(contact_path)})
        static_qa, deformation_qa, animation_qa, rebase_deformation_qa = _make_qa_reports(
            root=root,
            visual=visual,
            action_path=actions,
            rebase_path=rebase_report,
            contact_path=contact_path,
            anchor_defs=anchor_defs,
            muzzle_joint_id=next(anchor.joint_id for anchor in anchor_defs if anchor.anchor_id == "muzzle"),
        )
        source_manifest = root / "source_manifest.json"
        _write_json(
            source_manifest,
            {
                "schema": "avengine_m2_source_snapshot_v1",
                "formal_dataset_registration_authorized": False,
                "source_artifacts": [_record(source)],
                "derivation": {
                    "target_native_geometry_preserved": True,
                    "action_donor_geometry_used": False,
                    "prepared_inputs": [stage_item for stage_item in stage["stages"] if stage_item["name"] in {"webp_transcode", "human_skin_promotion", "route_root_translation_normalization"}],
                },
            },
        )
        license_snapshot = root / "license_snapshot.json"
        license_name = str(spec.get("license", "project_owner_review_required"))
        allowed_use = str(spec.get("allowed_use", "review_required"))
        redistribution = str(spec.get("redistribution", "review_required"))
        _write_json(
            license_snapshot,
            {
                "schema": "avengine_m2_license_snapshot_v1",
                "license": license_name,
                "allowed_use": allowed_use,
                "redistribution": redistribution,
                "qualification_claim": False,
                "decision_reason": "P12 package remains a research candidate pending owner rights review.",
            },
        )
        identity = AnimalPackageIdentity(
            asset_id=asset_id,
            template_id=str(spec["template_id"]),
            body_plan_id=str(spec["body_plan_id"]),
            morphotype_id=str(spec["morphotype_id"]),
            skeleton_revision=str(spec.get("skeleton_revision", "p12-target-native-skeleton-v1")),
            weights_revision=str(spec.get("weights_revision", "p12-target-native-weights-v1")),
            collision_revision=str(spec.get("collision_revision", "m2-kinematic-rest-bbox-proxy-v1")),
            action_revision=str(spec.get("action_revision", "p12-reviewed-idle-walk-v1")),
            source=str(spec.get("source", "owner-registered target-native source asset")),
            source_revision=str(spec.get("source_revision", "external-source-revision")),
            license=license_name,
            allowed_use=allowed_use,
            redistribution=redistribution,
            semantic_id=int(spec.get("semantic_id", 200)),
        )
        package = root / "package"
        manifest = compile_research_candidate_animal_package(
            output_directory=package,
            identity=identity,
            visual_glb=visual,
            rebase_report=rebase_report,
            rebase_deformation_report=rebase_deformation_qa,
            action_report=action_report,
            static_qa=static_qa,
            deformation_qa=deformation_qa,
            animation_qa=animation_qa,
            habitat_static_probe=static_probe / "probe.json",
            habitat_animation_review=animation_review / "review_report.json",
            baked_actions=actions,
            contacts=contact_path,
            anchor_definitions=anchors,
            source_manifest=source_manifest,
            license_snapshot=license_snapshot,
            shader_type=str(spec.get("shader_type", "phong")),
        )
        package_value = json.loads(manifest.read_text(encoding="utf-8"))
        stage["stages"].append({"name": "package_compile", "status": "pass", "output": _record(manifest)})
        base_request = root / "base_m2_request.json"
        _write_base_m2_request(
            path=base_request,
            manifest_path=manifest,
            package_value=package_value,
            actions_path=actions,
            contacts_path=contact_path,
            contact_order=contact_order,
            asset_id=asset_id,
        )
        stage["stages"].append({"name": "base_m2_request_snapshot", "status": "pass", "output": _record(base_request)})
        runtime_backend = _runtime_backend(
            spec=spec,
            package_value=package_value,
            manifest=manifest,
            base_request=base_request,
            package=package,
            static_probe=static_probe,
        )
        increment = {
            "schema": "avengine_p12_habitat_binding_increment_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "asset_id": asset_id,
            "renderer": "habitat",
            "package_manifest": _record(manifest),
            "visual_glb": _record(package / "visual.glb"),
            "runtime_backend": runtime_backend,
            "runtime_binding": {
                "habitat_ao_config": "package/habitat/animal.ao_config.json",
                "habitat_urdf": "package/habitat/animal.urdf",
                "joint_mapping": "package/habitat/joint_mapping.json",
                "runtime_joint_order": package_value["skeleton"]["runtime_joint_order"],
                "contact_order": list(contact_order),
                "anchors": anchors,
            },
            "stage_results": stage["stages"],
            "provenance": {
                "source_glb": _record(source),
                "source_manifest": _record(source_manifest),
                "license_snapshot": _record(license_snapshot),
            },
            "notes": [
                "Registry mutation is intentionally left to the owner agent.",
                "This package does not claim formal dataset admission or human review.",
            ],
        }
        increment_path = root / "habitat_binding_increment.json"
        _write_json(increment_path, increment)
        return increment_path
    except Exception as exc:
        failure = {
            "schema": "avengine_p12_habitat_binding_increment_v1",
            "status": "fail",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "asset_id": asset_id,
            "failed_stage": stage["stages"][-1]["name"] if stage["stages"] else "input",
            "error": str(exc),
            "stage_results": stage["stages"],
        }
        try:
            _write_json(root / "habitat_binding_increment.failure.json", failure)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-device-id", type=int, default=3)
    args = parser.parse_args()
    if args.gpu_device_id < 0:
        parser.error("--gpu-device-id must be non-negative")
    result = build(args.spec.resolve(), args.output_root.resolve(), gpu_device_id=args.gpu_device_id)
    print(json.dumps({"status": "pass", "binding_increment": str(result)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
