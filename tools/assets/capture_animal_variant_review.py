#!/usr/bin/env python3
"""Build and run one single-view animal-variant Habitat review capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.contracts.json_io import sha256_file
from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.assets.variant_review import (
    ROOM_PRESETS,
    VARIANT_REVIEW_EVIDENCE_FILENAME,
    VariantReviewError,
    build_variant_review_request,
    capture_variant_review,
    load_trajectory,
    load_variant_review_inputs,
    resolve_room_preset,
    write_json_exclusive,
)


_BAKE_HELP = """
This command never plays a GLB animation clock.  For an animated source GLB,
first run:

  python tools/assets/bake_actions.py --input-glb ANIMAL.glb \\
    --output-npz actions.npz --report action_bake_report.json

Then compile the baked poses, Habitat AO/URDF mapping, contacts, and QA into an
M2 animal package before invoking this review command.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=_BAKE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument(
        "--room-preset",
        choices=sorted(ROOM_PRESETS),
        required=True,
        help="Validated M1 custom or real Habitat MP3D room pair",
    )
    parser.add_argument(
        "--trajectory-json",
        type=Path,
        help=(
            "Absolute room-space root path override; recommended when the "
            "species cadence differs from the bounded Beagle review path"
        ),
    )
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--request-output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _preflight_outputs(request_output: Path, capture_output: Path) -> None:
    request = request_output.resolve()
    capture = capture_output.resolve()
    for label, raw, resolved in (
        ("request output", request_output, request),
        ("capture output", capture_output, capture),
    ):
        if raw.exists() or raw.is_symlink():
            raise VariantReviewError(f"refusing to replace {label}: {resolved}")
    if request == capture:
        raise VariantReviewError("request and capture outputs must differ")
    try:
        request.relative_to(capture)
    except ValueError:
        pass
    else:
        raise VariantReviewError(
            "request output must not be inside the not-yet-created capture output"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _preflight_outputs(args.request_output, args.output)
    room_manifest, room_request, preset_trajectory = resolve_room_preset(
        args.room_preset
    )
    room_inputs = load_m1_inputs(room_manifest, room_request)
    trajectory = (
        load_trajectory(args.trajectory_json)
        if args.trajectory_json is not None
        else preset_trajectory
    )
    request = build_variant_review_request(
        asset_manifest=args.asset_manifest,
        room_inputs=room_inputs,
        request_id=args.request_id,
        trajectory=trajectory,
    )
    request_path = write_json_exclusive(args.request_output, request)
    inputs = load_variant_review_inputs(args.asset_manifest, request_path)
    output = args.output.resolve()
    evidence = capture_variant_review(
        inputs,
        room_inputs,
        output,
        runtime_root=args.runtime_root,
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "review_only": evidence["review_only"],
                "qualification_claim": evidence["qualification_claim"],
                "asset_id": evidence["asset_id"],
                "asset_admission_state": evidence["asset_admission_state"],
                "room_preset": args.room_preset,
                "room_id": evidence["room_id"],
                "view_ids": evidence["view_contract"]["view_ids"],
                "camera_count": evidence["view_contract"]["camera_count"],
                "co_located_modalities": evidence["view_contract"][
                    "co_located_modalities"
                ],
                "modalities": evidence["view_contract"]["modalities"],
                "frame_count": evidence["timeline"]["frame_count"],
                "segments": evidence["timeline"]["segments"],
                "request": str(request_path),
                "request_sha256": sha256_file(request_path),
                "evidence": str(output / VARIANT_REVIEW_EVIDENCE_FILENAME),
                "evidence_content_sha256": evidence["evidence_content_sha256"],
                "rgb_review_video": {
                    **evidence["rgb_review_video"],
                    "absolute_path": str(output / evidence["rgb_review_video"]["path"]),
                },
                "array_artifacts": evidence["array_artifacts"],
                "world_time_seconds": evidence["world_time_seconds"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
