#!/usr/bin/env python3
"""Render motion-following binaural audio for a current Apartment UE capture.

The UE capture supplies the per-frame anchor poses (production RGB authority);
the fixed-apartment M1 review request supplies the camera-colocated listener
pose, which is cross-checked against the capture's own static camera. Audio
is rendered by the room-agnostic dynamic research-audio core (per-state RIRs
plus one AudioProgram routing variant). Research review only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-capture-dir", required=True, type=Path)
    parser.add_argument(
        "--m1-request",
        required=True,
        type=Path,
        help="fixed-apartment M1 review request (listener pose authority)",
    )
    parser.add_argument("--simulation-request", required=True, type=Path)
    parser.add_argument("--package-manifest", required=True, type=Path)
    parser.add_argument("--audio-program", required=True, type=Path)
    parser.add_argument("--source-endpoint-registry", required=True, type=Path)
    parser.add_argument("--sound-asset-registry", required=True, type=Path)
    parser.add_argument(
        "--beagle-audio",
        required=True,
        type=Path,
        help="external dry wav for dog_beagle_v2_scheduled_dry",
    )
    parser.add_argument("--hrtf", required=True, type=Path)
    parser.add_argument("--hrtf-license", type=Path)
    parser.add_argument("--runtime-prefix", required=True, type=Path)
    parser.add_argument("--rlr-sdk-root", required=True, type=Path)
    parser.add_argument("--magnum-python-site", type=Path)
    parser.add_argument("--rir-stride-frames", type=int, default=3)
    parser.add_argument("--variant", default="A")
    parser.add_argument(
        "--actor-selection",
        type=Path,
        help="executed actor selection; with --source-asset-registry it derives "
        "per-slot endpoints and emitter heights from the registries instead of "
        "the legacy human+beagle constants",
    )
    parser.add_argument(
        "--source-asset-registry",
        type=Path,
        help="source-asset runtime registry (required with --actor-selection)",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    os.environ["AVENGINE_HABITAT_RUNTIME_PREFIX"] = str(
        args.runtime_prefix.resolve()
    )
    os.environ["AVENGINE_RLR_SDK_ROOT"] = str(args.rlr_sdk_root.resolve())
    if args.magnum_python_site is not None:
        os.environ["AVENGINE_HABITAT_MAGNUM_PYTHON_SITE"] = str(
            args.magnum_python_site.resolve()
        )

    from avengine.contracts.json_io import sha256_file  # noqa: E402
    from avengine.timeline.current_mp3d_dynamic_audio import (  # noqa: E402
        CurrentMP3DDynamicAudioError,
        listener_pose_from_m1_request,
        render_dynamic_research_audio,
    )
    from avengine.m7.apartment_dynamic_audio import (  # noqa: E402
        captured_static_camera_world_m,
        derive_slot_bindings,
        load_ue_anchor_trajectories,
    )

    try:
        capture_dir = args.visual_capture_dir.resolve()
        if args.actor_selection is not None:
            if args.source_asset_registry is None:
                raise CurrentMP3DDynamicAudioError(
                    "--actor-selection requires --source-asset-registry"
                )
            slot_endpoints, emitter_heights = derive_slot_bindings(
                json.loads(args.actor_selection.resolve().read_text(encoding="utf-8")),
                json.loads(
                    args.source_asset_registry.resolve().read_text(encoding="utf-8")
                ),
                json.loads(
                    args.source_endpoint_registry.resolve().read_text(encoding="utf-8")
                ),
            )
            trajectories = load_ue_anchor_trajectories(
                capture_dir,
                slot_endpoints=slot_endpoints,
                emitter_heights_m=emitter_heights,
            )
        else:
            trajectories = load_ue_anchor_trajectories(capture_dir)
        camera_world, camera_ue_yaw = captured_static_camera_world_m(capture_dir)
        m1_request = json.loads(args.m1_request.resolve().read_text(encoding="utf-8"))
        listener_position, listener_wxyz = listener_pose_from_m1_request(m1_request)
        drift = max(
            abs(float(a) - float(b))
            for a, b in zip(camera_world, listener_position)
        )
        if drift > 1.0e-6:
            raise CurrentMP3DDynamicAudioError(
                "the capture camera does not match the M1 listener authority: "
                f"capture {camera_world} vs request {listener_position}"
            )
        receipt = render_dynamic_research_audio(
            source_trajectories_m=trajectories,
            listener_position_m=listener_position,
            listener_orientation_wxyz=listener_wxyz,
            simulation_request_path=args.simulation_request,
            package_manifest_path=args.package_manifest,
            audio_program_path=args.audio_program,
            source_endpoint_registry_path=args.source_endpoint_registry,
            sound_asset_registry_path=args.sound_asset_registry,
            external_sound_asset_paths={
                "dog_beagle_v2_scheduled_dry": args.beagle_audio
            },
            hrtf_file_path=args.hrtf,
            hrtf_license_path=args.hrtf_license,
            output_path=args.output,
            position_authority=(
                "current Apartment UE capture actor_anchor_poses (legacy "
                "glTF-import transform inverted; per-slot emitter heights "
                "from the fixed-apartment anchor library)"
            ),
            listener_authority=(
                "fixed-apartment M1 review request, cross-checked against "
                f"the capture camera (UE yaw {camera_ue_yaw} deg)"
            ),
            rir_stride_frames=args.rir_stride_frames,
            variant_id=args.variant,
            extra_inputs={
                "visual_capture_frame_records": {
                    "path": str(capture_dir / "frame_records.json"),
                    "sha256": sha256_file(capture_dir / "frame_records.json"),
                },
                "m1_request": {
                    "path": str(args.m1_request.resolve()),
                    "sha256": sha256_file(args.m1_request.resolve()),
                },
            },
        )
    except (CurrentMP3DDynamicAudioError, OSError, ValueError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}))
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "research_only": receipt["research_only"],
                "keyframe_count": receipt["rir"]["keyframe_count"],
                "event_count": receipt["audio_program"]["event_count"],
                "output": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
