#!/usr/bin/env python3
"""Render the retained 270-frame ReplicaCAD route in an isolated SPEAR editor.

The isolated UE project must already contain the editor-imported/reloaded
``apt_0_comparison`` map.  SPEAR is launched through UnrealEditor ``-game`` so
uncooked assets can be exercised without writing to the legacy SPEAR checkout.
Habitat-native AVEngine remains authoritative for navigation, actor/source
centres, source events, binaural audio and the right-hand Topdown panel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY / "src"
TOOLS_ROOT = Path(__file__).resolve().parent
for value in (SOURCE_ROOT, TOOLS_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from avengine.optional_backends.spear_apartment import (  # noqa: E402
    ANIMATION_TOLERANCE_SECONDS,
    summarize_actor_bounds,
)
from avengine.optional_backends.spear_replicacad_execution import (  # noqa: E402
    M5_1_FPS,
    M5_1_FRAME_COUNT,
    M5_1_MAP_PATH,
    M5_1_ROOM_ID,
    M5_1_RUNTIME_SCHEMA,
    build_m5_1_replicacad_runtime_plan,
)
from run_spear_mp3d_canary import (  # noqa: E402
    _LuminanceAccumulator,
    _actor_bounds_readback,
    _actor_readback,
    _apply_actor_state,
    _apply_camera,
    _audio_packet_sha256,
    _cleanup_failed_constructor,
    _file_record,
    _grade_frame,
    _probe_media,
    _read_frame,
    _root_gate,
    _spawn_camera,
    _spawn_runtime_actors,
)


DEFAULT_SPEAR_ROOT = REPOSITORY.parent / "AVEngine/external/SPEAR"
DEFAULT_ROUTE = REPOSITORY / "examples/m5_1/replicacad_articulated_review/route_manifest.json"
DEFAULT_CAPTURE = REPOSITORY / "tmp/m5_1/replicacad_mixed_20260719_04/evidence.json"
DEFAULT_FRAME_READBACK = (
    REPOSITORY / "tmp/m5_1/replicacad_mixed_20260719_04/frame_readback.json"
)
DEFAULT_SOURCE_CENTER_GATE = (
    REPOSITORY
    / "tmp/m6x/replicacad_obstacle_review_20260719_02/source_center_gate.json"
)
DEFAULT_DELIVERY = REPOSITORY / "tmp/m5_1/replicacad_delivery_20260719_03"
DEFAULT_HABITAT_REVIEW = (
    REPOSITORY
    / "tmp/m6x/replicacad_obstacle_review_20260719_02/videos/"
    "replicacad_runtime_obstacles_diagnostic.mp4"
)
DEFAULT_REQUEST_ROOT = (
    REPOSITORY / "tmp/m6y/replicacad_apt0_spear_request_20260720_02"
)
EVIDENCE_SCHEMA = "avengine_optional_spear_replicacad_runtime_evidence_v1"
SMOKE_SCHEMA = "avengine_optional_spear_replicacad_runtime_smoke_v1"
INITIALIZE_CLIENT_MAX_TIME_SECONDS = 600.0
CLIENT_INTERNAL_TIMEOUT_SECONDS = 120.0


def _load_json(path: Path, *, owner: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load {owner}: {path}: {exc}") from exc


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_execution_plan(args: argparse.Namespace) -> dict[str, Any]:
    files = {
        "route_manifest": args.route_manifest,
        "capture_evidence": args.capture_evidence,
        "frame_readback": args.frame_readback,
        "source_center_gate": args.source_center_gate,
        "source_program": args.source_program,
        "emitter_trajectories": args.emitter_trajectories,
        "source_actor_bindings": args.source_actor_bindings,
        "execution_request": args.execution_request,
        "editor_import_result": args.editor_import_result,
        "editor_reload_result": args.editor_reload_result,
    }
    loaded = {
        name: _load_json(path.resolve(), owner=name) for name, path in files.items()
    }
    if not isinstance(loaded["frame_readback"], list):
        raise RuntimeError("ReplicaCAD frame readback root must be a list")
    for name, value in loaded.items():
        if name != "frame_readback" and not isinstance(value, Mapping):
            raise RuntimeError(f"ReplicaCAD {name} root must be an object")
    return build_m5_1_replicacad_runtime_plan(
        route_manifest=loaded["route_manifest"],
        capture_evidence=loaded["capture_evidence"],
        frame_readback=loaded["frame_readback"],
        source_center_gate=loaded["source_center_gate"],
        source_program=loaded["source_program"],
        emitter_trajectories=loaded["emitter_trajectories"],
        source_actor_bindings=loaded["source_actor_bindings"],
        execution_request=loaded["execution_request"],
        editor_import_result=loaded["editor_import_result"],
        editor_reload_result=loaded["editor_reload_result"],
        output_gain=args.fixed_output_gain,
    )


def _scene_readback(game: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    mesh_actors = game.unreal_service.find_actors_by_class(uclass="AStaticMeshActor")
    point_lights = game.unreal_service.find_actors_by_class(uclass="APointLight")
    if len(mesh_actors) != plan["scene"]["static_mesh_actor_count"]:
        raise RuntimeError(
            f"ReplicaCAD runtime mesh count {len(mesh_actors)} != "
            f"{plan['scene']['static_mesh_actor_count']}"
        )
    if len(point_lights) != plan["scene"]["runtime_positive_point_light_count"]:
        raise RuntimeError(
            f"ReplicaCAD runtime point-light count {len(point_lights)} != "
            f"{plan['scene']['runtime_positive_point_light_count']}"
        )
    tagged_meshes = sum(
        bool(actor.ActorHasTag(Tag="avengine_comparison_visual"))
        for actor in mesh_actors
    )
    tagged_lights = sum(
        bool(actor.ActorHasTag(Tag="avengine_dataset_light"))
        for actor in point_lights
    )
    if tagged_meshes != len(mesh_actors) or tagged_lights != len(point_lights):
        raise RuntimeError(
            "ReplicaCAD runtime actor tags do not close over imported map actors"
        )
    light_records = []
    for index, actor in enumerate(point_lights):
        component = game.unreal_service.get_component_by_class(
            actor=actor, uclass="UPointLightComponent"
        )
        light_records.append(
            {
                "runtime_index": index,
                "location_cm": _actor_readback(actor, 0)["location_cm"],
                "intensity": float(
                    component.get_property_value(property_name="Intensity")
                ),
                "attenuation_radius_cm": float(
                    component.get_property_value(property_name="AttenuationRadius")
                ),
                "cast_shadows": bool(
                    component.get_property_value(property_name="CastShadows")
                ),
            }
        )
    if not all(item["intensity"] > 0.0 and item["cast_shadows"] for item in light_records):
        raise RuntimeError("ReplicaCAD runtime dataset-light readback failed")
    return {
        "status": "pass",
        "map_path": plan["scene"]["map_path"],
        "static_mesh_actor_count": len(mesh_actors),
        "tagged_comparison_visual_actor_count": tagged_meshes,
        "positive_dataset_point_light_count": len(point_lights),
        "tagged_dataset_light_count": tagged_lights,
        "declared_dataset_light_count": 7,
        "recorded_negative_fill_count": 2,
        "review_light_added": False,
        "lights": sorted(light_records, key=lambda item: item["intensity"]),
    }


def _configure_instance(args: argparse.Namespace, plan: Mapping[str, Any]) -> tuple[Any, Path]:
    spear_root = args.spear_root.resolve()
    editor = args.unreal_editor.resolve()
    project = args.ue_project.resolve()
    if not editor.is_file() or not os.access(editor, os.X_OK):
        raise RuntimeError(f"UnrealEditor is missing or not executable: {editor}")
    if project.suffix != ".uproject" or not project.is_file():
        raise RuntimeError(f"isolated SPEAR uproject is missing: {project}")
    old_project = (
        spear_root / "cpp/unreal_projects/SpearSim/SpearSim.uproject"
    ).resolve()
    if project == old_project or old_project.parent in project.parents:
        raise RuntimeError("refusing to render ReplicaCAD through the old dirty SPEAR project")
    sys.path.insert(0, str(spear_root / "examples"))
    from render_in_apartment import parallel_instance_settings

    import spear

    settings = parallel_instance_settings(
        args.rpc_port, graphics_adapter=args.graphics_adapter
    )
    config = spear.get_config(user_config_files=[])
    config.defrost()
    config.SPEAR.LAUNCH_MODE = "editor"
    config.SPEAR.INSTANCE.EDITOR_EXECUTABLE = str(editor)
    config.SPEAR.INSTANCE.EDITOR_UPROJECT = str(project)
    config.SPEAR.INSTANCE.EDITOR_LAUNCH_MODE = "game"
    config.SPEAR.INSTANCE.INITIALIZE_CLIENT_MAX_TIME_SECONDS = (
        INITIALIZE_CLIENT_MAX_TIME_SECONDS
    )
    config.SPEAR.INSTANCE.CLIENT_INTERNAL_TIMEOUT_SECONDS = CLIENT_INTERNAL_TIMEOUT_SECONDS
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = M5_1_MAP_PATH
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.FIXED_DELTA_TIME = 1.0 / M5_1_FPS
    config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT = settings["rpc_port"]
    config.SPEAR.INSTANCE.TEMP_DIR = settings["temp_dir"]
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.log = settings["log"]
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.renderoffscreen = None
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.execcmds = ",".join(
        plan["exposure_and_lighting"]["console_commands"]
    )
    config.SP_CORE.SHARED_MEMORY_INITIAL_UNIQUE_ID = settings[
        "shared_memory_initial_unique_id"
    ]
    if settings["graphics_adapter"] is not None:
        config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.graphicsadapter = settings[
            "graphics_adapter"
        ]
    vulkan_icd = os.environ.get(
        "VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json"
    )
    if Path(vulkan_icd).is_file():
        config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = vulkan_icd
    config.freeze()
    spear.configure_system(config=config)
    try:
        instance = spear.Instance(config=config)
    except BaseException:
        _cleanup_failed_constructor(
            executable=editor, temporary_directory=Path(settings["temp_dir"])
        )
        raise
    return instance, spear_root


def _encode_media(
    *, frames_root: Path, habitat_review: Path, output_root: Path
) -> dict[str, Path]:
    visual = output_root / "replicacad_spear_visual_only.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(M5_1_FPS),
            "-i", str(frames_root / "frame_%04d.png"),
            "-frames:v", str(M5_1_FRAME_COUNT),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-movflags", "+faststart", str(visual),
        ],
        check=True,
    )
    clean = output_root / "replicacad_spear_clean_binaural.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(visual), "-i", str(habitat_review),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "copy", "-map_metadata", "-1",
            "-movflags", "+faststart", str(clean),
        ],
        check=True,
    )
    topdown = output_root / "replicacad_spear_topdown_binaural.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(visual), "-i", str(habitat_review),
            "-filter_complex",
            (
                "[0:v]scale=640:360:flags=lanczos,pad=640:480:0:60:black[ue];"
                "[1:v]crop=640:480:640:0[top];[ue][top]hstack=inputs=2[out]"
            ),
            "-map", "[out]", "-map", "1:a:0",
            "-frames:v", str(M5_1_FRAME_COUNT), "-r", str(M5_1_FPS),
            "-vsync", "cfr", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20", "-c:a", "copy", "-map_metadata", "-1",
            "-movflags", "+faststart", str(topdown),
        ],
        check=True,
    )
    triptych = output_root / "replicacad_spear_habitat_topdown_triptych_binaural.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(visual), "-i", str(habitat_review),
            "-filter_complex",
            (
                "[0:v]scale=640:360:flags=lanczos,pad=640:480:0:60:black[ue];"
                "[1:v]setsar=1[hab];[ue][hab]hstack=inputs=2[out]"
            ),
            "-map", "[out]", "-map", "1:a:0",
            "-frames:v", str(M5_1_FRAME_COUNT), "-r", str(M5_1_FPS),
            "-vsync", "cfr", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20", "-c:a", "copy", "-map_metadata", "-1",
            "-movflags", "+faststart", str(triptych),
        ],
        check=True,
    )
    return {"visual": visual, "clean": clean, "topdown": topdown, "triptych": triptych}


def run(args: argparse.Namespace) -> Path:
    output_root = args.output_dir.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to replace output directory: {output_root}")
    output_root.mkdir(parents=True)
    plan = build_execution_plan(args)
    if plan.get("schema") != M5_1_RUNTIME_SCHEMA:
        raise RuntimeError("compiled ReplicaCAD runtime plan schema changed")
    plan_path = output_root / "execution_plan.json"
    _write_json(plan_path, plan)
    if args.dry_run:
        print(f"SPEAR_REPLICACAD_DRY_RUN_OK plan={plan_path}", flush=True)
        return plan_path

    frames_root = output_root / "frames"
    frames_root.mkdir()
    instance, spear_root = _configure_instance(args, plan)
    game = instance.get_game()
    actor_readbacks = {"human0": [], "dog0": []}
    animation_readbacks = {"human0": [], "dog0": []}
    actor_bounds = {"human0": [], "dog0": []}
    camera_readbacks: list[dict[str, Any]] = []
    luminance = _LuminanceAccumulator(plan["exposure_and_lighting"])
    smoke_index = args.smoke_frame_index
    capture_indices = (
        [smoke_index] if smoke_index is not None else list(range(M5_1_FRAME_COUNT))
    )
    scene_readback: dict[str, Any]
    try:
        with instance.begin_frame():
            scene_readback = _scene_readback(game, plan)
            camera, capture = _spawn_camera(
                game=game,
                width=plan["render"]["width"],
                height=plan["render"]["height"],
                hfov=plan["camera"]["horizontal_fov_deg"],
            )
            runtimes = _spawn_runtime_actors(game, spear_root, plan)
            _apply_camera(camera, plan)
            first_index = smoke_index if smoke_index is not None else 0
            for state in plan["frames"][first_index]["actor_states"]:
                _apply_actor_state(runtimes[state["actor_id"]], state, first_index)
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                bPaused=False
            )
        with instance.end_frame():
            pass
        instance.step(num_frames=plan["render"]["streaming_warmup_frames"])
        with instance.begin_frame():
            _apply_camera(camera, plan)
        with instance.end_frame():
            pass
        instance.step(num_frames=plan["render"]["camera_warmup_frames"])

        import cv2

        for output_index, frame_index in enumerate(capture_indices):
            frame_plan = plan["frames"][frame_index]
            with instance.begin_frame():
                for state in frame_plan["actor_states"]:
                    actor_id = state["actor_id"]
                    root, animation = _apply_actor_state(
                        runtimes[actor_id], state, frame_index
                    )
                    actor_readbacks[actor_id].append(root)
                    animation_readbacks[actor_id].append(animation)
                _apply_camera(camera, plan)
                camera_readbacks.append(_actor_readback(camera, frame_index))
            with instance.end_frame():
                raw = _read_frame(capture)
                for actor_id, runtime in runtimes.items():
                    actor_bounds[actor_id].append(
                        _actor_bounds_readback(runtime["visual_actor"], frame_index)
                    )
                expected_shape = (
                    plan["render"]["height"], plan["render"]["width"], 3
                )
                if raw.shape != expected_shape:
                    raise RuntimeError(f"unexpected captured frame shape: {raw.shape}")
                frame = _grade_frame(
                    raw, plan["exposure_and_lighting"]["fixed_output_gain"]
                )
                luminance.add_bgr(frame)
                path = frames_root / f"frame_{output_index:04d}.png"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"could not write frame: {path}")
            if smoke_index is None and frame_index % M5_1_FPS == 0:
                print(
                    f"[spear-replicacad] frame {frame_index:03d}/"
                    f"{M5_1_FRAME_COUNT - 1}",
                    flush=True,
                )
    finally:
        instance.close(force=True)

    luminance_qa = luminance.result(plan["exposure_and_lighting"])
    if luminance_qa["status"] != "pass":
        raise RuntimeError(f"ReplicaCAD fixed-exposure QA failed: {luminance_qa}")
    captured_frames = [plan["frames"][index] for index in capture_indices]
    bounds_gate = summarize_actor_bounds(
        expected_frames=captured_frames,
        actor_declarations=plan["actors"],
        actor_bounds=actor_bounds,
    )
    hierarchy = {
        actor_id: {
            "asset_local_frame": runtime["component_frame"],
            "runtime_hierarchy": runtime["hierarchy"],
        }
        for actor_id, runtime in runtimes.items()
    }
    if smoke_index is not None:
        smoke = {
            "schema": SMOKE_SCHEMA,
            "status": "pass",
            "backend_role": "comparison_visual",
            "room_id": M5_1_ROOM_ID,
            "frame_index": smoke_index,
            "frame_path": str((frames_root / "frame_0000.png").resolve()),
            "scene_readback": scene_readback,
            "luminance_qa": luminance_qa,
            "visual_bounds_readback": bounds_gate,
            "runtime_actor_hierarchy": hierarchy,
            "clock": plan["clock"],
            "claim_boundary": "single-frame actual runtime smoke; not a 270-frame delivery",
        }
        smoke_path = output_root / "smoke_evidence.json"
        _write_json(smoke_path, smoke)
        print(
            f"SPEAR_REPLICACAD_SMOKE_OK frame={smoke['frame_path']} "
            f"evidence={smoke_path}",
            flush=True,
        )
        return smoke_path

    root_gate = _root_gate(plan["frames"], actor_readbacks, camera_readbacks, plan)
    animation_gate = {}
    for actor_id, records in animation_readbacks.items():
        maximum = max(item["absolute_error_seconds"] for item in records)
        if len(records) != M5_1_FRAME_COUNT or maximum > ANIMATION_TOLERANCE_SECONDS:
            raise RuntimeError(f"{actor_id} animation-phase closure failed")
        animation_gate[actor_id] = {
            "status": "pass",
            "frame_count": len(records),
            "maximum_absolute_error_seconds": maximum,
            "tolerance_seconds": ANIMATION_TOLERANCE_SECONDS,
        }

    habitat_review = args.habitat_review.resolve()
    media_paths = _encode_media(
        frames_root=frames_root,
        habitat_review=habitat_review,
        output_root=output_root,
    )
    media = {
        "ue_visual_only": _probe_media(
            media_paths["visual"], width=1280, height=720, expect_audio=False
        ),
        "ue_clean_binaural": _probe_media(
            media_paths["clean"], width=1280, height=720, expect_audio=True
        ),
        "ue_topdown_binaural": _probe_media(
            media_paths["topdown"], width=1280, height=480, expect_audio=True
        ),
        "ue_habitat_topdown_triptych_binaural": _probe_media(
            media_paths["triptych"], width=1920, height=480, expect_audio=True
        ),
    }
    source_audio_hash = _audio_packet_sha256(habitat_review)
    for name, record in media.items():
        if name != "ue_visual_only" and record["audio_packet_sha256"] != source_audio_hash:
            raise RuntimeError(f"{name} changed the authoritative binaural packets")

    input_paths = {
        "execution_plan": plan_path,
        "route_manifest": args.route_manifest.resolve(),
        "capture_evidence": args.capture_evidence.resolve(),
        "frame_readback": args.frame_readback.resolve(),
        "source_center_gate": args.source_center_gate.resolve(),
        "source_program": args.source_program.resolve(),
        "emitter_trajectories": args.emitter_trajectories.resolve(),
        "source_actor_bindings": args.source_actor_bindings.resolve(),
        "editor_import_result": args.editor_import_result.resolve(),
        "editor_reload_result": args.editor_reload_result.resolve(),
    }
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "status": "pass",
        "backend_role": "comparison_visual",
        "room_id": M5_1_ROOM_ID,
        "route_id": plan["route_id"],
        "clock": plan["clock"],
        "authority": plan["authority"],
        "runtime": {
            "launch_mode": "isolated_UnrealEditor_game",
            "map": M5_1_MAP_PATH,
            "rpc_port": args.rpc_port,
            "graphics_adapter": args.graphics_adapter,
            "resolution": [1280, 720],
            "frame_rate_hz": M5_1_FPS,
            "streaming_warmup_frames": plan["render"]["streaming_warmup_frames"],
            "camera_warmup_frames": plan["render"]["camera_warmup_frames"],
            "auto_exposure_console_commands_requested": plan[
                "exposure_and_lighting"
            ]["console_commands"],
            "fixed_output_gain": plan["exposure_and_lighting"]["fixed_output_gain"],
            "scene_and_lighting_readback": scene_readback,
        },
        "readback": {
            "root_and_camera": root_gate,
            "animation_phase": animation_gate,
            "visual_bounds": bounds_gate,
            "runtime_actor_hierarchy": hierarchy,
        },
        "source_center_gate": plan["source_logic"]["source_center_gate"],
        "exposure_qa": luminance_qa,
        "media": media,
        "audio_authority": {
            "source_video": _file_record(habitat_review),
            "audio_packet_sha256": source_audio_hash,
            "semantics": "Habitat-native two-channel binaural; no camera-FOV cutoff",
        },
        "inputs": {name: _file_record(path) for name, path in input_paths.items()},
        "claim_boundary": plan["claim_boundary"],
    }
    evidence_path = output_root / "evidence.json"
    _write_json(evidence_path, evidence)
    if not args.keep_frames:
        shutil.rmtree(frames_root)
    print(
        "SPEAR_REPLICACAD_CANARY_OK "
        f"video={media_paths['topdown']} evidence={evidence_path}",
        flush=True,
    )
    return evidence_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spear-root", type=Path, default=DEFAULT_SPEAR_ROOT)
    parser.add_argument("--unreal-editor", type=Path, required=True)
    parser.add_argument("--ue-project", type=Path, required=True)
    parser.add_argument("--route-manifest", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--capture-evidence", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--frame-readback", type=Path, default=DEFAULT_FRAME_READBACK)
    parser.add_argument(
        "--source-center-gate", type=Path, default=DEFAULT_SOURCE_CENTER_GATE
    )
    parser.add_argument(
        "--source-program", type=Path,
        default=DEFAULT_DELIVERY / "source_program_reuse.json",
    )
    parser.add_argument(
        "--emitter-trajectories", type=Path,
        default=DEFAULT_DELIVERY / "actual_emitter_trajectories.json",
    )
    parser.add_argument(
        "--source-actor-bindings", type=Path,
        default=DEFAULT_DELIVERY / "source_actor_bindings.json",
    )
    parser.add_argument(
        "--execution-request", type=Path,
        default=DEFAULT_REQUEST_ROOT / "execution_request.json",
    )
    parser.add_argument(
        "--editor-import-result", type=Path,
        default=DEFAULT_REQUEST_ROOT / "editor_import_result.json",
    )
    parser.add_argument(
        "--editor-reload-result", type=Path,
        default=DEFAULT_REQUEST_ROOT / "editor_reload_result.json",
    )
    parser.add_argument("--habitat-review", type=Path, default=DEFAULT_HABITAT_REVIEW)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rpc-port", type=int, default=39341)
    parser.add_argument("--graphics-adapter", type=int)
    parser.add_argument("--fixed-output-gain", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument(
        "--smoke-frame-index", type=int,
        help="Render one selected authority frame; this is not formal evidence.",
    )
    args = parser.parse_args(argv)
    if not 1024 <= args.rpc_port <= 65535:
        parser.error("--rpc-port must be in [1024,65535]")
    if args.graphics_adapter is not None and args.graphics_adapter < 0:
        parser.error("--graphics-adapter must be non-negative")
    if args.smoke_frame_index is not None and not (
        0 <= args.smoke_frame_index < M5_1_FRAME_COUNT
    ):
        parser.error("--smoke-frame-index must be in [0,269]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
