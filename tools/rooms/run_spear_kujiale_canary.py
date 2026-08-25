#!/usr/bin/env python3
"""Capture an external InteriorAgent/Kujiale room through SPEAR and UE.

This is a bounded four-view visual canary. It deliberately has no alternate
navigation, Timeline, source-program or acoustic implementation. Its host/game
client is AVEngine-owned; the external Unreal editor and project remain inputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "tools/rooms"))

from avengine.backends.spear_ue import client as spear_client  # noqa: E402
from avengine.backends.spear_ue.launch import parallel_instance_settings  # noqa: E402
from avengine.optional_backends.interioragent_kujiale import (  # noqa: E402
    build_kujiale_review_plan,
    load_profile,
    load_room_metadata,
)
from run_spear_mp3d_canary import _read_frame, _spawn_camera  # noqa: E402


def _write_montage(
    output: Path, records: list[dict[str, Any]], *, backend_role: str
) -> Path:
    panels = []
    for record in records:
        frame = cv2.imread(record["path"], cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"cannot reopen captured frame: {record['path']}")
        label = (
            f"{record['view_id']}  yaw={record['yaw_deg']:.0f} deg  "
            f"{backend_role}"
        )
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (20, 20, 20), -1)
        cv2.putText(
            frame,
            label,
            (18, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        panels.append(frame)
    while len(panels) < 4:
        panels.append(np.zeros_like(panels[0]))
    rows = [
        np.concatenate(panels[index : index + 2], axis=1)
        for index in range(0, len(panels), 2)
    ]
    montage = np.concatenate(rows, axis=0)
    path = output / "kujiale_four_view_montage.png"
    if not cv2.imwrite(str(path), montage):
        raise RuntimeError(f"cannot write montage: {path}")
    return path


def _spawn_review_lights(game: Any, plan: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for light in plan["review_lights"]:
        position = light["position_ue_cm"]
        actor = game.unreal_service.spawn_actor(
            uclass="APointLight",
            location={"X": position[0], "Y": position[1], "Z": position[2]},
        )
        if actor is None:
            raise RuntimeError(f"could not spawn review light {light['light_id']}")
        actor.K2_GetRootComponent().SetMobility(NewMobility="Movable")
        component = game.unreal_service.get_component_by_class(
            actor=actor, uclass="UPointLightComponent"
        )
        component.SetIntensity(NewIntensity=light["intensity_lumens"])
        component.SetAttenuationRadius(
            NewRadius=light["attenuation_radius_cm"]
        )
        component.SetCastShadows(bNewValue=True)
        component.set_property_value(
            property_name="SourceRadius",
            property_value=light["source_radius_cm"],
        )
        component.set_property_value(
            property_name="SoftSourceRadius",
            property_value=light["soft_source_radius_cm"],
        )
        component.set_property_value(
            property_name="bUseTemperature", property_value=True
        )
        component.set_property_value(
            property_name="Temperature",
            property_value=light["temperature_kelvin"],
        )
        records.append(
            {
                **light,
                "intensity_readback": float(
                    component.get_property_value(property_name="Intensity")
                ),
                "temperature_readback": float(
                    component.get_property_value(property_name="Temperature")
                ),
                "cast_shadows_readback": bool(
                    component.get_property_value(property_name="CastShadows")
                ),
            }
        )
    return records


def _build_plan(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.profile)
    if args.map_path is not None:
        profile["map_path"] = args.map_path
    rooms = load_room_metadata(args.rooms) if args.rooms else None
    return build_kujiale_review_plan(
        profile, source_stage=args.source_stage, rooms=rooms
    )


def _configure_spear(args: argparse.Namespace, plan: dict[str, Any]) -> Any:
    settings = parallel_instance_settings(
        args.rpc_port, graphics_adapter=args.graphics_adapter
    )
    config = spear_client.get_config(user_config_files=[])
    config.defrost()
    config.SPEAR.LAUNCH_MODE = "editor"
    config.SPEAR.INSTANCE.EDITOR_EXECUTABLE = str(
        args.unreal_editor.expanduser().resolve()
    )
    config.SPEAR.INSTANCE.EDITOR_UPROJECT = str(
        args.uproject.expanduser().resolve()
    )
    config.SPEAR.INSTANCE.EDITOR_LAUNCH_MODE = "game"
    config.SPEAR.INSTANCE.INITIALIZE_CLIENT_MAX_TIME_SECONDS = 900.0
    config.SPEAR.INSTANCE.CLIENT_INTERNAL_TIMEOUT_SECONDS = 180.0
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = plan["map_path"]
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.FIXED_DELTA_TIME = 1.0 / 15.0
    config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT = settings["rpc_port"]
    config.SPEAR.INSTANCE.TEMP_DIR = settings["temp_dir"]
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.log = settings["log"]
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.renderoffscreen = None
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.graphicsadapter = args.graphics_adapter
    # The persisted review map contains an AUsdStageActor which references the
    # local derived scene adapter.  SPEAR's stock project does not enable the
    # USD plugin by default, so make the dependency explicit for every run.
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.enableplugins = "USDImporter"
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.execcmds = ",".join(
        [
            "r.EyeAdaptationQuality 0",
            "r.DefaultFeature.AutoExposure 0",
            "r.Lumen.DiffuseIndirect.Allow 1",
            "r.Lumen.Reflections.Allow 1",
            "sg.ViewDistanceQuality 3",
            "sg.ShadowQuality 3",
            "sg.GlobalIlluminationQuality 3",
            "sg.ReflectionQuality 3",
        ]
    )
    config.SP_CORE.SHARED_MEMORY_INITIAL_UNIQUE_ID = settings[
        "shared_memory_initial_unique_id"
    ]
    vulkan_icd = os.environ.get(
        "VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json"
    )
    if Path(vulkan_icd).is_file():
        config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = vulkan_icd
    config.freeze()
    spear_client.configure_system(config=config)
    return spear_client.Instance(config=config)


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = _build_plan(args)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)

    instance = _configure_spear(args, plan)
    game = instance.get_game()
    records: list[dict[str, Any]] = []
    stage_actor_count = 0
    light_records: list[dict[str, Any]] = []
    try:
        with instance.begin_frame():
            camera, capture = _spawn_camera(
                game, width=args.width, height=args.height, hfov=args.hfov
            )
            light_records = _spawn_review_lights(game, plan)
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                bPaused=False
            )
        with instance.end_frame():
            pass
        instance.step(num_frames=args.streaming_warmup_frames)

        with instance.begin_frame():
            stage_actors = game.unreal_service.find_actors_by_class(
                uclass="AUsdStageActor"
            )
            stage_actor_count = len(stage_actors)
        with instance.end_frame():
            pass
        if stage_actor_count != 1:
            raise RuntimeError(f"expected one UsdStageActor, got {stage_actor_count}")

        for view in plan["camera_views"]:
            position = view["position_ue_cm"]
            with instance.begin_frame():
                camera.K2_SetActorLocation(
                    NewLocation={
                        "X": position[0],
                        "Y": position[1],
                        "Z": position[2],
                    },
                    bSweep=False,
                    bTeleport=True,
                )
                camera.K2_SetActorRotation(
                    NewRotation={
                        "Roll": 0.0,
                        "Pitch": 0.0,
                        "Yaw": view["yaw_deg"],
                    },
                    bTeleportPhysics=True,
                )
            with instance.end_frame():
                pass
            instance.step(num_frames=args.camera_warmup_frames)
            with instance.begin_frame():
                pass
            with instance.end_frame():
                frame = _read_frame(capture)
            path = output / f"{view['view_id']}.png"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"cannot write captured frame: {path}")
            records.append(
                {
                    **view,
                    "path": str(path),
                    "shape": list(frame.shape),
                    "mean_bgr": [
                        float(item) for item in frame.mean(axis=(0, 1))
                    ],
                    "minimum": int(frame.min()),
                    "maximum": int(frame.max()),
                }
            )
    finally:
        instance.close(force=True)

    montage = _write_montage(output, records, backend_role=plan["backend_role"])
    evidence = {
        "status": "pass",
        "schema_version": "avengine_optional_spear_kujiale_runtime_evidence_v1",
        "backend_role": plan["backend_role"],
        "plan": plan,
        "stage_actor_count": stage_actor_count,
        "runtime_review_lights": light_records,
        "frames": records,
        "montage": str(montage),
        "render": {
            "width": args.width,
            "height": args.height,
            "hfov_deg": args.hfov,
            "streaming_warmup_frames": args.streaming_warmup_frames,
            "camera_warmup_frames": args.camera_warmup_frames,
        },
        "claim_boundary": plan["claim_boundary"],
    }
    (output / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uproject", type=Path, required=True)
    parser.add_argument("--unreal-editor", type=Path, required=True)
    parser.add_argument("--source-stage", type=Path, required=True)
    parser.add_argument("--rooms", type=Path)
    parser.add_argument("--map-path")
    parser.add_argument(
        "--profile",
        type=Path,
        default=(
            REPOSITORY
            / "examples/m6z/interioragent_kujiale_0020_visual_profile.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rpc-port", type=int, default=39373)
    parser.add_argument("--graphics-adapter", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--streaming-warmup-frames", type=int, default=180)
    parser.add_argument("--camera-warmup-frames", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    evidence = run(parse_args())
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
