#!/usr/bin/env python3
"""Render hash-bound M2 Idle/Walk review media in Habitat.

This is a human-review producer, not an asset qualification command or trusted
execution attester.  It writes the exact baked joint pose before every
observation, never advances Habitat's physics/animation clock, and captures
co-located RGB/depth/semantic modalities in one observation call.  Its local
hashes establish configuration-and-output integrity only.  The two camera
angles are explicitly QA-only; a formal M2 capture still has one ``view0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence

import numpy as np
from PIL import Image

from avengine.contracts.json_io import load_json
from avengine.assets.actions import read_baked_actions_npz
from avengine.assets.glb import load_glb
from avengine.assets.habitat import (
    HabitatLinkJointBlock,
    bind_habitat_link_layout,
    build_habitat_ao_config_data,
    build_habitat_asset_mapping_from_rebase_report,
)


_SEMANTIC_ID = 200
_MIN_SEMANTIC_PIXELS = 256
_QA_VIEWS = ("side", "front_quarter")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _output_directory(path: Path) -> Path:
    output = path.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must not already contain files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _make_configuration() -> tuple[Any, Any, Any]:
    # This pinned build must import numpy-quaternion before habitat_sim.
    import quaternion as qt

    import habitat_sim
    import magnum as mn

    sensor_specs: list[Any] = []
    for uuid, sensor_type in (
        ("rgb", habitat_sim.SensorType.COLOR),
        ("depth", habitat_sim.SensorType.DEPTH),
        ("semantic", habitat_sim.SensorType.SEMANTIC),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = mn.Vector2i([480, 640])
        spec.hfov = 60.0
        spec.near = 0.02
        # QA framing is derived from source-asset bounds; large research assets
        # can require substantially more than the old fixed 20 m range.
        spec.far = 100.0
        spec.gpu2gpu_transfer = False
        # CameraSensorSpec otherwise hides a default 1.5 m eye-height offset.
        spec.position = mn.Vector3(0.0, 0.0, 0.0)
        spec.orientation = mn.Vector3(0.0, 0.0, 0.0)
        if uuid != "rgb":
            spec.channels = 1
        sensor_specs.append(spec)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = True
    sim_cfg.gpu_device_id = 0
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {}
    return habitat_sim.Configuration(sim_cfg, [agent_cfg]), qt, mn


def _yaw_matrix(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = [
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ]
    return result


def _apply_root_transform(
    articulated_object: Any,
    world_from_skin_root: np.ndarray,
    *,
    qt: Any,
    mn: Any,
) -> None:
    rotation = world_from_skin_root[:3, :3]
    quaternion_wxyz = qt.as_float_array(qt.from_rotation_matrix(rotation))
    articulated_object.translation = mn.Vector3(world_from_skin_root[:3, 3])
    articulated_object.rotation = mn.Quaternion(
        mn.Vector3(quaternion_wxyz[1:]), float(quaternion_wxyz[0])
    )


def _world_visual_framing(
    articulated_object: Any, mn: Any
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    node = articulated_object.root_scene_node
    node.compute_cumulative_bb()
    bounds = node.cumulative_bb
    local_minimum = np.asarray(bounds.min, dtype=np.float64)
    local_maximum = np.asarray(bounds.max, dtype=np.float64)
    world_points = np.asarray(
        [
            node.absolute_transformation().transform_point(mn.Vector3([x, y, z]))
            for x in (local_minimum[0], local_maximum[0])
            for y in (local_minimum[1], local_maximum[1])
            for z in (local_minimum[2], local_maximum[2])
        ],
        dtype=np.float64,
    )
    world_minimum = world_points.min(axis=0)
    world_maximum = world_points.max(axis=0)
    world_center = 0.5 * (world_minimum + world_maximum)
    radius = float(np.max(np.linalg.norm(world_points - world_center, axis=1)))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"invalid articulated-object visual radius: {radius}")
    return world_center, radius, world_minimum, world_maximum


def _camera_states(
    target: np.ndarray, visual_radius: float, habitat_sim: Any
) -> tuple[dict[str, tuple[Any, Any]], float]:
    from habitat_sim.utils.common import quat_from_two_vectors

    # The sensor is 640x480 with a 60-degree horizontal FOV.  Frame a bounding
    # sphere inside 78% of the more restrictive vertical half-FOV.  The prior
    # fixed 1.45 m offset could place the camera inside large source assets and
    # still satisfy a pixel-count visibility check.
    horizontal_half_fov = math.radians(60.0 / 2.0)
    vertical_half_fov = math.atan(math.tan(horizontal_half_fov) * (480.0 / 640.0))
    camera_distance = max(
        1.45,
        visual_radius / math.sin(vertical_half_fov * 0.78),
    )
    directions = {
        "side": np.asarray([1.0, 0.055, 0.0], dtype=np.float64),
        "front_quarter": np.asarray([1.0, 0.12, -1.0], dtype=np.float64),
    }
    positions = {
        qa_id: target + camera_distance * direction / np.linalg.norm(direction)
        for qa_id, direction in directions.items()
    }
    return (
        {
            qa_id: (
                position,
                quat_from_two_vectors(np.asarray([0.0, 0.0, -1.0]), target - position),
            )
            for qa_id, position in positions.items()
        },
        camera_distance,
    )


def _save_previews(
    output: Path,
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    semantic: np.ndarray,
) -> None:
    Image.fromarray(rgb[..., :3].astype(np.uint8), mode="RGB").save(
        output.with_name(output.name + "_rgb.png")
    )
    finite = depth[np.isfinite(depth)]
    maximum = float(finite.max()) if finite.size else 1.0
    normalized = np.clip(depth / max(maximum, 1.0e-12), 0.0, 1.0)
    Image.fromarray((normalized * 65535).astype(np.uint16), mode="I;16").save(
        output.with_name(output.name + "_depth.png")
    )
    Image.fromarray(np.asarray(semantic, dtype=np.uint16), mode="I;16").save(
        output.with_name(output.name + "_semantic.png")
    )


def _write_contact_sheet(frame_paths: Sequence[Path], destination: Path) -> None:
    selected_indices = np.linspace(0, len(frame_paths) - 1, 5, dtype=np.int64)
    frames = [
        Image.open(frame_paths[index]).convert("RGB") for index in selected_indices
    ]
    try:
        width, height = frames[0].size
        sheet = Image.new("RGB", (width * len(frames), height), color=(0, 0, 0))
        for index, frame in enumerate(frames):
            sheet.paste(frame, (index * width, 0))
        sheet.save(destination)
    finally:
        for frame in frames:
            frame.close()


def _encode_video(frame_directory: Path, destination: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        "15",
        "-start_number",
        "0",
        "-i",
        str(frame_directory / "frame_%04d.png"),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
        "-threads",
        "1",
        str(destination),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "byte_size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def render_review(
    *,
    visual_glb: Path,
    actions_npz: Path,
    rebase_report_path: Path,
    output_path: Path,
    actor_yaw_degrees: float,
    shader_type: str = "phong",
) -> dict[str, Any]:
    output = _output_directory(output_path)
    document = load_glb(visual_glb)
    actions = read_baked_actions_npz(actions_npz)
    rebase_report = load_json(rebase_report_path)
    mapping = build_habitat_asset_mapping_from_rebase_report(document, rebase_report)
    if actions.source_glb_sha256 != document.sha256:
        raise ValueError("actions NPZ source SHA-256 does not match the visual GLB")
    if actions.runtime_joint_order != mapping.runtime_joint_order:
        raise ValueError("actions and Habitat mapping joint orders differ")

    visual_copy = output / "visual.glb"
    urdf_path = output / "animal.urdf"
    config_path = output / "animal.ao_config.json"
    mapping_path = output / "habitat_joint_mapping.json"
    shutil.copyfile(visual_glb, visual_copy)
    urdf_path.write_text(mapping.render_urdf(), encoding="utf-8")
    ao_config = build_habitat_ao_config_data(
        render_asset=visual_copy.name,
        urdf_filepath=urdf_path.name,
        semantic_id=_SEMANTIC_ID,
        shader_type=shader_type,
    )
    _write_json(config_path, ao_config)
    _write_json(mapping_path, mapping.joint_mapping_data())

    configuration, qt, mn = _make_configuration()
    import habitat_sim

    world_from_actor = _yaw_matrix(actor_yaw_degrees)
    actor_from_skin_root = np.asarray(mapping.actor_from_skin_root, dtype=np.float64)
    world_from_skin_root = world_from_actor @ actor_from_skin_root
    runs: list[dict[str, Any]] = []
    emitted_artifacts: list[Path] = []

    with habitat_sim.Simulator(configuration) as simulator:
        loaded = simulator.metadata_mediator.ao_template_manager.load_configs(
            str(config_path)
        )
        handle_prefix = config_path.stem.removesuffix(".ao_config")
        handles = simulator.metadata_mediator.ao_template_manager.get_template_handles(
            handle_prefix
        )
        if len(loaded) != 1 or len(handles) != 1:
            raise RuntimeError(
                f"expected one AO template, got ids={loaded}, handles={handles}"
            )
        articulated_object = simulator.get_articulated_object_manager().add_articulated_object_by_template_handle(
            handles[0]
        )
        if articulated_object is None:
            raise RuntimeError("Habitat failed to instantiate the articulated dog")
        articulated_object.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        _apply_root_transform(articulated_object, world_from_skin_root, qt=qt, mn=mn)

        actual_names = {articulated_object.get_link_name(-1)} | {
            articulated_object.get_link_name(link_id)
            for link_id in articulated_object.get_link_ids()
        }
        if actual_names != set(mapping.joint_order):
            raise RuntimeError(
                "Habitat link names differ from the asset mapping: "
                f"missing={set(mapping.joint_order) - actual_names}, "
                f"extra={actual_names - set(mapping.joint_order)}"
            )
        blocks = [
            HabitatLinkJointBlock(
                link_name=articulated_object.get_link_name(link_id),
                joint_position_offset=int(
                    articulated_object.get_link_joint_pos_offset(link_id)
                ),
                joint_position_count=int(
                    articulated_object.get_link_num_joint_pos(link_id)
                ),
            )
            for link_id in articulated_object.get_link_ids()
        ]
        binding = bind_habitat_link_layout(
            mapping.runtime_joint_order,
            blocks,
            joint_position_count=len(articulated_object.joint_positions),
        )
        runtime_binding_path = output / "habitat_runtime_binding.json"
        runtime_binding = binding.to_json_data()
        _write_json(runtime_binding_path, runtime_binding)

        first_pose = actions.action("idle").rotations_xyzw[0]
        articulated_object.joint_positions = np.asarray(binding.map_pose(first_pose))
        target, visual_radius, visual_minimum, visual_maximum = _world_visual_framing(
            articulated_object, mn
        )
        camera_states, camera_distance = _camera_states(
            target, visual_radius, habitat_sim
        )
        state = habitat_sim.AgentState()
        first_position, first_rotation = camera_states[_QA_VIEWS[0]]
        state.position = first_position
        state.rotation = first_rotation
        agent = simulator.initialize_agent(0, state)

        world_time_before = float(simulator.get_world_time())
        for clip in actions.actions:
            clip_rotations = np.asarray(clip.rotations_xyzw, dtype=np.float64)
            for qa_id in _QA_VIEWS:
                position, rotation = camera_states[qa_id]
                state.position = position
                state.rotation = rotation
                agent.set_state(state, reset_sensors=True)
                frame_directory = output / "frames" / clip.semantic_action_id / qa_id
                frame_directory.mkdir(parents=True, exist_ok=True)
                semantic_counts: list[int] = []
                observation_hashes: list[dict[str, str]] = []
                frame_paths: list[Path] = []
                for frame_index, pose in enumerate(clip_rotations):
                    _apply_root_transform(
                        articulated_object, world_from_skin_root, qt=qt, mn=mn
                    )
                    articulated_object.joint_positions = np.asarray(
                        binding.map_pose(pose), dtype=np.float64
                    )
                    observations = simulator.get_sensor_observations()
                    rgb = np.asarray(observations["rgb"])
                    depth = np.asarray(observations["depth"])
                    semantic = np.asarray(observations["semantic"])
                    if not (rgb.shape[:2] == depth.shape == semantic.shape):
                        raise RuntimeError("co-located modality resolutions differ")
                    count = int(np.count_nonzero(semantic == _SEMANTIC_ID))
                    semantic_counts.append(count)
                    observation_hashes.append(
                        {
                            "rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
                            "depth": hashlib.sha256(depth.tobytes()).hexdigest(),
                            "semantic": hashlib.sha256(semantic.tobytes()).hexdigest(),
                        }
                    )
                    frame_path = frame_directory / f"frame_{frame_index:04d}.png"
                    Image.fromarray(rgb[..., :3].astype(np.uint8), mode="RGB").save(
                        frame_path
                    )
                    frame_paths.append(frame_path)
                    if frame_index == 0:
                        _save_previews(
                            output / f"{clip.semantic_action_id}_{qa_id}_frame0",
                            rgb=rgb,
                            depth=depth,
                            semantic=semantic,
                        )

                video_path = output / f"{clip.semantic_action_id}_{qa_id}.mp4"
                contact_sheet_path = (
                    output / f"{clip.semantic_action_id}_{qa_id}_contact_sheet.png"
                )
                _encode_video(frame_directory, video_path)
                _write_contact_sheet(frame_paths, contact_sheet_path)
                emitted_artifacts.extend([video_path, contact_sheet_path])
                runs.append(
                    {
                        "semantic_action_id": clip.semantic_action_id,
                        "source_action_name": clip.source_action_name,
                        "qa_view_id": qa_id,
                        "sample_ticks": list(clip.sample_ticks),
                        "sample_count": clip.sample_count,
                        "minimum_semantic_pixel_count": min(semantic_counts),
                        "maximum_semantic_pixel_count": max(semantic_counts),
                        "all_frames_visible": min(semantic_counts)
                        >= _MIN_SEMANTIC_PIXELS,
                        "observation_hashes": observation_hashes,
                        "video": _artifact_record(video_path, output),
                        "contact_sheet": _artifact_record(contact_sheet_path, output),
                    }
                )
        world_time_after = float(simulator.get_world_time())

    automatic_status = (
        "pass"
        if all(run["all_frames_visible"] for run in runs)
        and world_time_before == world_time_after
        else "fail"
    )
    report = {
        "schema": "avengine_m2_habitat_action_review_v1",
        "status": automatic_status,
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "evidence_scope": {
            "local_report_claim": "artifact_integrity_only",
            "trusted_runtime_attestation": False,
            "runtime_execution_conclusion_source": "external_capture_audit_only",
        },
        "producer_source_integrity": {
            "path": str(Path(__file__).resolve()),
            "byte_size": Path(__file__).stat().st_size,
            "sha256": _sha256_file(Path(__file__)),
        },
        "render_configuration_integrity": {
            "configured_shader_type": shader_type,
            "ao_config_artifact": {
                **_artifact_record(config_path, output),
                "snapshot": ao_config,
            },
        },
        "runtime_artifact_integrity": {
            "runtime_binding_artifact": {
                **_artifact_record(runtime_binding_path, output),
                "snapshot": runtime_binding,
            },
            "observation_artifacts": [
                _artifact_record(path, output) for path in emitted_artifacts
            ],
        },
        "source": {
            "visual_glb": {
                "path": str(visual_glb.resolve()),
                "sha256": document.sha256,
                "byte_size": visual_glb.stat().st_size,
            },
            "actions_npz": {
                "path": str(actions_npz.resolve()),
                "sha256": _sha256_file(actions_npz),
                "byte_size": actions_npz.stat().st_size,
            },
            "rebase_report": {
                "path": str(rebase_report_path.resolve()),
                "sha256": _sha256_file(rebase_report_path),
                "byte_size": rebase_report_path.stat().st_size,
            },
        },
        "runtime_contract": {
            "root_joint_id": mapping.root_joint_id,
            "runtime_joint_order": list(mapping.runtime_joint_order),
            "joint_position_count": binding.joint_position_count,
            "world_from_actor": world_from_actor.tolist(),
            "actor_from_skin_root": actor_from_skin_root.tolist(),
            "world_from_skin_root_formula": ("world_from_actor @ actor_from_skin_root"),
            "world_from_skin_root": world_from_skin_root.tolist(),
            "actor_yaw_degrees": actor_yaw_degrees,
            "root_transform_reapplied_each_frame": True,
            "joint_pose_reapplied_each_frame": True,
        },
        "capture_contract": {
            "formal_capture": False,
            "qa_only_view_ids": list(_QA_VIEWS),
            "formal_m2_view_ids_remain": ["view0"],
            "modalities": ["rgb", "depth", "semantic"],
            "co_located_and_co_oriented": True,
            "advance_clock_between_frames": False,
            "advance_clock_between_modalities": False,
            "world_time_before": world_time_before,
            "world_time_after": world_time_after,
            "world_time_unchanged": world_time_before == world_time_after,
            "camera_framing": {
                "method": "idle_pose_world_bounding_sphere_v1",
                "visual_center": target.tolist(),
                "visual_minimum": visual_minimum.tolist(),
                "visual_maximum": visual_maximum.tolist(),
                "visual_radius": visual_radius,
                "camera_distance": camera_distance,
                "vertical_half_fov_fill_fraction": 0.78,
            },
        },
        "runs": runs,
        "notes": [
            "This report hash-binds local configuration and review outputs only.",
            "It is not a trusted attestation that Habitat executed those bytes.",
            "Runtime execution conclusions require a separately retained capture/audit.",
            "Mesh/skin/action alignment, anatomical plausibility, paw semantics, and contacts still require human review.",
            "The QA cameras are not additional formal dataset views.",
        ],
    }
    report_path = output / "review_report.json"
    _write_json(report_path, report)
    pending_review = {
        "schema": "avengine_m2_human_visual_review_v1",
        "status": "not_run",
        "qualification_claim": False,
        "bindings": {
            "visual_glb_sha256": document.sha256,
            "actions_npz_sha256": _sha256_file(actions_npz),
            "habitat_review_report_sha256": _sha256_file(report_path),
            "media": [_artifact_record(path, output) for path in emitted_artifacts],
        },
        "required_checks": {
            "mesh_and_skin_alignment": "not_run",
            "idle_animation_plausibility": "not_run",
            "walking_animation_plausibility": "not_run",
            "paw_identity_and_orientation": "not_run",
            "mouth_remains_visually_static": "not_run",
        },
        "reviewer_id": None,
        "decision_reason": (
            "Pending user inspection; this file must not be edited to pass without "
            "reviewing the hash-bound media."
        ),
    }
    _write_json(output / "human_review_pending.json", pending_review)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-glb", type=Path, required=True)
    parser.add_argument("--actions-npz", type=Path, required=True)
    parser.add_argument("--rebase-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--actor-yaw-degrees",
        type=float,
        default=0.0,
        help="Explicit world_from_actor yaw; +90 maps this candidate's +X head direction to -Z.",
    )
    parser.add_argument(
        "--shader-type",
        choices=("phong", "pbr"),
        default="phong",
        help="Explicit Habitat AO shader; formal M2 remains phong by default.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = render_review(
        visual_glb=args.visual_glb,
        actions_npz=args.actions_npz,
        rebase_report_path=args.rebase_report,
        output_path=args.output,
        actor_yaw_degrees=args.actor_yaw_degrees,
        shader_type=args.shader_type,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
