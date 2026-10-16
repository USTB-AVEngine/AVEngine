#!/usr/bin/env python3
"""Run the real ReplicaCAD Habitat capture with one shared lighting profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from avengine.capture.mixed_capture import M5_1_LIGHT_SETUP_KEY, MixedCaptureError
from avengine.capture.replicacad_capture import (
    ReplicaCADCaptureError,
    capture_replicacad_route,
)
from avengine.optional_backends.spear_replicacad_execution import (
    DATASET_LIGHTS_FAITHFUL_PROFILE_ID,
    ROOM_LOCAL_REVIEW_PROFILE_ID,
    ROUTE_CENTER_FILL_REVIEW_PROFILE_ID,
    ReplicaCADExecutionError,
    apply_replicacad_habitat_lighting_profile,
    compile_replicacad_lighting_profile,
    configure_replicacad_habitat_lighting_profile,
    load_replicacad_lighting_profiles,
    resolve_replicacad_route_center_fill,
    validate_replicacad_habitat_lighting_readback,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--room-manifest", type=Path, required=True)
    parser.add_argument("--m1-request", type=Path, required=True)
    parser.add_argument("--human-runtime-glb", type=Path, required=True)
    parser.add_argument("--beagle-manifest", type=Path, required=True)
    parser.add_argument("--beagle-m2-request", type=Path, required=True)
    parser.add_argument("--replicacad-root", type=Path, required=True)
    parser.add_argument("--execution-request", type=Path, required=True)
    parser.add_argument(
        "--lighting-profiles",
        type=Path,
        default=REPOSITORY / "examples/rooms/replicacad_apt0_lighting_profiles.json",
    )
    parser.add_argument(
        "--lighting-profile",
        choices=(
            DATASET_LIGHTS_FAITHFUL_PROFILE_ID,
            ROOM_LOCAL_REVIEW_PROFILE_ID,
            ROUTE_CENTER_FILL_REVIEW_PROFILE_ID,
        ),
        default=ROOM_LOCAL_REVIEW_PROFILE_ID,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--review-frame-index", type=int, default=135)
    parser.add_argument("--review-frame-output", type=Path, required=True)
    return parser


def _load_object(path: Path, *, owner: str) -> Mapping[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ReplicaCADExecutionError(f"{owner} root must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        profile = compile_replicacad_lighting_profile(
            execution_request=_load_object(
                args.execution_request, owner="ReplicaCAD execution request"
            ),
            profile_document=load_replicacad_lighting_profiles(args.lighting_profiles),
            profile_id=args.lighting_profile,
        )
        profile = resolve_replicacad_route_center_fill(
            profile,
            _load_object(args.route_manifest, owner="ReplicaCAD route manifest"),
        )

        def configuration_hook(
            *, configuration: Any, habitat_sim: Any
        ) -> Mapping[str, Any]:
            return configure_replicacad_habitat_lighting_profile(
                configuration=configuration,
                habitat_sim=habitat_sim,
                lighting_profile=profile,
            )

        def scene_hook(
            *, simulator: Any, habitat_sim: Any, **_: Any
        ) -> Mapping[str, Any]:
            return apply_replicacad_habitat_lighting_profile(
                simulator=simulator,
                habitat_sim=habitat_sim,
                lighting_profile=profile,
                actor_light_setup_key=M5_1_LIGHT_SETUP_KEY,
            )

        def readback_hook(
            *,
            simulator: Any,
            habitat_sim: Any,
            actor_light_setup_key: str,
            **_: Any,
        ) -> Mapping[str, Any]:
            return validate_replicacad_habitat_lighting_readback(
                simulator=simulator,
                lighting_profile=profile,
                habitat_sim=habitat_sim,
                actor_light_setup_key=actor_light_setup_key,
            )

        result = capture_replicacad_route(
            route_manifest_path=args.route_manifest,
            room_manifest_path=args.room_manifest,
            m1_request_path=args.m1_request,
            human_runtime_glb_path=args.human_runtime_glb,
            beagle_animal_manifest_path=args.beagle_manifest,
            beagle_m2_request_path=args.beagle_m2_request,
            output_dir=args.output,
            replicacad_root=args.replicacad_root,
            runtime_root=args.runtime_root,
            review_configuration_hook=configuration_hook,
            review_scene_hook=scene_hook,
            review_scene_readback_hook=readback_hook,
        )
        frame_index = int(args.review_frame_index)
        if not 0 <= frame_index < int(result.capture.rgb.shape[0]):
            raise ReplicaCADExecutionError("review frame index is out of range")
        frame_output = args.review_frame_output.expanduser().resolve()
        if frame_output.exists() or frame_output.is_symlink():
            raise ReplicaCADExecutionError(
                f"refusing to replace review frame: {frame_output}"
            )
        frame_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.capture.rgb[frame_index]).save(frame_output)
    except (
        ReplicaCADCaptureError,
        ReplicaCADExecutionError,
        MixedCaptureError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "status": "pass",
                "profile_id": profile["profile_id"],
                "habitat_intensity_scale": profile["habitat_intensity_scale"],
                "source_intensities_scaled": profile[
                    "habitat_source_intensities_scaled"
                ],
                "habitat_usage": profile["habitat_usage"],
                "habitat_maintained_default": profile["habitat_maintained_default"],
                "review_light_added": profile["review_light_added"],
                "generated_interior_fill": profile.get("generated_interior_fill"),
                "capture_output": str(result.capture.output_dir),
                "capture_evidence": str(result.capture.output_dir / "evidence.json"),
                "review_frame_index": frame_index,
                "review_frame_output": str(frame_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
