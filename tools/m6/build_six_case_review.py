#!/usr/bin/env python3
"""Validate, plan, or build the immutable M6 six-case human-review package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from avengine.m6.review import (
    build_six_case_review,
    load_six_case_review_request,
    plan_six_case_review,
    verify_six_case_review,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate six-case semantics")
    validate.add_argument("request", type=Path)

    plan = subparsers.add_parser(
        "plan", help="print exact FFmpeg commands without running them"
    )
    plan.add_argument("request", type=Path)
    plan.add_argument("--repository-root", type=Path, default=Path.cwd())
    plan.add_argument("--staging-directory", type=Path, required=True)
    plan.add_argument("--check-media", action="store_true")
    plan.add_argument("--ffmpeg", default="ffmpeg")

    build = subparsers.add_parser(
        "build", help="encode and atomically publish an immutable review package"
    )
    build.add_argument("request", type=Path)
    build.add_argument("--repository-root", type=Path, default=Path.cwd())
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--ffmpeg", default="ffmpeg")
    build.add_argument("--ffprobe", default="ffprobe")

    verify = subparsers.add_parser(
        "verify", help="re-open and verify a published review package"
    )
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--repository-root", type=Path, default=Path.cwd())
    verify.add_argument("--ffprobe", default="ffprobe")
    return parser


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "verify":
        report = verify_six_case_review(
            arguments.manifest,
            repository_root=arguments.repository_root,
            ffprobe=arguments.ffprobe,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["status"] == "pass" else 1

    request = load_six_case_review_request(arguments.request)
    if arguments.command == "validate":
        print(
            json.dumps(
                {
                    "status": "pass",
                    "review_id": request["review_id"],
                    "case_count": len(request["cases"]),
                    "room_lineage_count": len(
                        {
                            case["room_lineage_id"]
                            for case in request["cases"]
                            if case["is_room"]
                        }
                    ),
                },
                indent=2,
            )
        )
        return 0

    if arguments.command == "plan":
        plan = plan_six_case_review(
            request,
            repository_root=arguments.repository_root,
            staging_directory=arguments.staging_directory,
            ffmpeg=arguments.ffmpeg,
            check_media=arguments.check_media,
        )
        summary = {
            "status": "pass",
            "review_id": request["review_id"],
            "segments": [
                {
                    "case_id": segment["case"]["case_id"],
                    "title_lines": segment["title_lines"],
                    "command": segment["command"],
                }
                for segment in plan["segments"]
            ],
            "concat_command": plan["concat_command"],
        }
        print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))
        return 0

    manifest, video = build_six_case_review(
        request_path=arguments.request,
        output_directory=arguments.output,
        repository_root=arguments.repository_root,
        ffmpeg=arguments.ffmpeg,
        ffprobe=arguments.ffprobe,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "review_manifest": str(manifest.resolve()),
                "combined_video": str(video.resolve()),
                "segment_directory": str((video.parent / "segments").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
