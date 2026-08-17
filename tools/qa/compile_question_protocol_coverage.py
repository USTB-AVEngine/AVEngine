#!/usr/bin/env python3
"""Compile or independently validate the 12-type native QuestionSpec protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.question_protocol import (
    QuestionProtocolError,
    compile_question_protocol_coverage,
    validate_compiled_delivery,
)

DEFAULT_PROTOCOL = REPOSITORY / "examples/qa/question_spec_paper_protocol_v1.json"
DEFAULT_EPISODES = REPOSITORY / "examples/qa/native_question_episode_catalog_v1.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile", help="compile a no-clobber delivery")
    compile_parser.add_argument("--output", required=True, type=Path)
    compile_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    compile_parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    compile_parser.add_argument("--ffmpeg", default="ffmpeg")
    validate_parser = subparsers.add_parser("validate", help="validate compiled bytes")
    validate_parser.add_argument("--input", required=True, type=Path)
    validate_parser.add_argument(
        "--require-paper-ready",
        action="store_true",
        help="also fail while any answer-balance stratum remains missing",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "compile":
            result = compile_question_protocol_coverage(
                repository=REPOSITORY,
                protocol_path=args.protocol.resolve(),
                episode_catalog_path=args.episodes.resolve(),
                output=args.output.resolve(),
                ffmpeg=args.ffmpeg,
            )
        else:
            result = validate_compiled_delivery(
                args.input.resolve(), require_paper_ready=args.require_paper_ready
            )
    except QuestionProtocolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "minimum_protocol_status": result["minimum_protocol_status"],
                "visual_canary_status": result["visual_canary_status"],
                "paper_balance_status": result["paper_balance_status"],
                "episode_count": result["episode_count"],
                "candidate_case_count": result["candidate_case_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
