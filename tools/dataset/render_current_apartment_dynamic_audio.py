#!/usr/bin/env python3
"""Render motion-following binaural audio for a current UE research capture.

The UE capture supplies the per-frame anchor poses (production RGB authority);
the matching per-point M1 request supplies the camera-colocated listener
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


def external_sound_paths(*, beagle_audio=None, assignments=(), mapping_path=None):
    """Resolve explicitly supplied sound-ID bindings, including legacy beagle input."""
    paths = {}
    if mapping_path is not None:
        mapping_path = Path(mapping_path).resolve()
        values = json.loads(mapping_path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("sound asset map must be an object mapping IDs to paths")
        for sound_id, value in values.items():
            if not isinstance(sound_id, str) or not sound_id or not isinstance(value, str) or not value:
                raise ValueError("sound asset map IDs and paths must be nonempty strings")
            path = Path(value).expanduser()
            paths[sound_id] = (mapping_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
    values = list(assignments)
    if beagle_audio is not None:
        values.append(f"dog_beagle_v2_scheduled_dry={beagle_audio}")
    for assignment in values:
        sound_id, separator, value = assignment.partition("=")
        if not separator or not sound_id or not value:
            raise ValueError("sound asset paths must use SOUND_ASSET_ID=PATH")
        if sound_id in paths:
            raise ValueError(f"duplicate sound asset path binding: {sound_id}")
        paths[sound_id] = Path(value).expanduser().resolve()
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-capture-dir", required=True, type=Path)
    parser.add_argument(
        "--m1-request",
        required=True,
        type=Path,
        help="matching per-point M1 listener-pose request",
    )
    parser.add_argument("--simulation-request", required=True, type=Path)
    parser.add_argument("--package-manifest", required=True, type=Path)
    parser.add_argument("--audio-program", required=True, type=Path)
    parser.add_argument("--source-endpoint-registry", required=True, type=Path)
    parser.add_argument("--sound-asset-registry", required=True, type=Path)
    parser.add_argument(
        "--beagle-audio",
        type=Path,
        help="legacy external dry wav binding for dog_beagle_v2_scheduled_dry",
    )
    parser.add_argument("--sound-asset-path", action="append", default=[],
                        help="explicit SOUND_ASSET_ID=PATH binding; repeat for each external sound")
    parser.add_argument("--sound-asset-map", type=Path,
                        help="JSON mapping of sound IDs to paths, relative to this JSON file")
    parser.add_argument("--hrtf", required=True, type=Path)
    parser.add_argument("--hrtf-license", type=Path)
    parser.add_argument("--runtime-prefix", required=True, type=Path)
    parser.add_argument("--rlr-sdk-root", required=True, type=Path)
    parser.add_argument("--magnum-python-site", type=Path)
    parser.add_argument("--rir-stride-frames", type=int, default=3)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--frame-rate-hz", type=float)
    parser.add_argument("--ticks-per-frame", type=int)
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
    parser.add_argument(
        "--canonical-emitter-height-m",
        type=float,
        help="optional QA counterfactual policy: use one world-space semantic "
        "emitter height for every selected actor while preserving registered "
        "endpoint identity",
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
        load_captured_render_clock,
        listener_pose_from_m1_request,
        render_dynamic_research_audio,
    )
    from avengine.dataset.apartment_dynamic_audio import (  # noqa: E402
        assert_listener_matches_capture_yaw,
        captured_static_camera_world_m,
        derive_slot_bindings,
        load_ue_anchor_trajectories,
    )

    try:
        sound_paths = external_sound_paths(
            beagle_audio=args.beagle_audio, assignments=args.sound_asset_path,
            mapping_path=args.sound_asset_map,
        )
        capture_dir = args.visual_capture_dir.resolve()
        clock = load_captured_render_clock(
            capture_dir,
            frame_count=args.frame_count,
            frame_rate_hz=args.frame_rate_hz,
            ticks_per_frame=args.ticks_per_frame,
        )
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
                canonical_emitter_height_m=args.canonical_emitter_height_m,
            )
            trajectories = load_ue_anchor_trajectories(
                capture_dir,
                slot_endpoints=slot_endpoints,
                emitter_heights_m=emitter_heights,
                frame_count=int(clock["frame_count"]),
                frame_rate_hz=clock["frame_rate_hz"],
                ticks_per_frame=int(clock["ticks_per_frame"]),
                canonical_emitter_height_m=args.canonical_emitter_height_m,
            )
        else:
            trajectories = load_ue_anchor_trajectories(
                capture_dir,
                frame_count=int(clock["frame_count"]),
                frame_rate_hz=clock["frame_rate_hz"],
                ticks_per_frame=int(clock["ticks_per_frame"]),
                canonical_emitter_height_m=args.canonical_emitter_height_m,
            )
        camera_world, camera_ue_yaw = captured_static_camera_world_m(
            capture_dir,
            frame_count=int(clock["frame_count"]),
            frame_rate_hz=clock["frame_rate_hz"],
            ticks_per_frame=int(clock["ticks_per_frame"]),
        )
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
        assert_listener_matches_capture_yaw(listener_wxyz, camera_ue_yaw)
        receipt = render_dynamic_research_audio(
            source_trajectories_m=trajectories,
            listener_position_m=listener_position,
            listener_orientation_wxyz=listener_wxyz,
            simulation_request_path=args.simulation_request,
            package_manifest_path=args.package_manifest,
            audio_program_path=args.audio_program,
            source_endpoint_registry_path=args.source_endpoint_registry,
            sound_asset_registry_path=args.sound_asset_registry,
            external_sound_asset_paths=sound_paths,
            hrtf_file_path=args.hrtf,
            hrtf_license_path=args.hrtf_license,
            output_path=args.output,
            position_authority=(
                "current UE source_emitter_poses (legacy glTF-import "
                "transform inverted; canonical QA Y override "
                f"{args.canonical_emitter_height_m} m)"
                if args.canonical_emitter_height_m is not None else
                "current UE source_emitter_poses when present (legacy "
                "glTF-import transform inverted; historical captures use "
                "per-slot height fallback from selected runtime profiles)"
            ),
            listener_authority=(
                "matching per-point M1 request, cross-checked against "
                f"the capture camera (UE yaw {camera_ue_yaw} deg)"
            ),
            rir_stride_frames=args.rir_stride_frames,
            variant_id=args.variant,
            visual_frame_count=int(clock["frame_count"]),
            visual_frame_rate_hz=clock["frame_rate_hz"],
            timeline_tick_rate_hz=int(clock["time_base_hz"]),
            ticks_per_frame=int(clock["ticks_per_frame"]),
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
