#!/usr/bin/env python3
"""Prepare or verify the two-commit AVEngine cross-repository release manifest.

``prepare`` must run on the clean implementation commit A.  It writes the
manifest with no-clobber semantics but deliberately does not commit or tag.
Commit only the manifest/allowlisted release metadata to create direct child B,
create an annotated tag on B, then run ``verify``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.release import (
    ReleaseManifestError,
    load_json_strict,
    prepare_release_manifest,
    verify_release_manifest,
)


def _artifact_roots(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        root_id, separator, raw_path = value.partition("=")
        if not separator or not root_id or not raw_path:
            raise argparse.ArgumentTypeError(
                f"artifact root must use ROOT_ID=/path syntax: {value!r}"
            )
        if root_id in roots:
            raise argparse.ArgumentTypeError(f"duplicate artifact root: {root_id}")
        roots[root_id] = Path(raw_path)
    return roots


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="atomically create the manifest while HEAD is clean commit A"
    )
    prepare.add_argument("--request", type=Path, required=True)
    prepare.add_argument("--avengine-root", type=Path, default=Path.cwd())
    prepare.add_argument("--habitat-runtime-root", type=Path, required=True)
    prepare.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        metavar="ROOT_ID=PATH",
        help="additional portable evidence root; may be repeated",
    )

    verify = commands.add_parser(
        "verify",
        help="verify direct-child commit B, annotated tag, hashes and environment",
    )
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--avengine-root", type=Path, default=Path.cwd())
    verify.add_argument("--habitat-runtime-root", type=Path, required=True)
    verify.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        metavar="ROOT_ID=PATH",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        roots = _artifact_roots(arguments.artifact_root)
        if arguments.command == "prepare":
            published = prepare_release_manifest(
                arguments.request,
                avengine_root=arguments.avengine_root,
                habitat_runtime_root=arguments.habitat_runtime_root,
                artifact_roots=roots,
            )
            manifest = load_json_strict(published)
            implementation = manifest["repositories"]["avengine"][
                "implementation_commit"
            ]
            result = {
                "schema": "avengine_release_prepare_result_v1",
                "status": "prepared",
                "implementation_commit_a": implementation,
                "manifest": str(published),
                "self_reference_avoided": True,
                "next_steps": [
                    "commit only the manifest and its declared allowlisted "
                    "release paths as direct child B",
                    "create an annotated release tag on B",
                    "run this tool's verify command from clean AVEngine and "
                    "Habitat worktrees",
                ],
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        report = verify_release_manifest(
            arguments.manifest,
            avengine_root=arguments.avengine_root,
            habitat_runtime_root=arguments.habitat_runtime_root,
            artifact_roots=roots,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "pass" else 1
    except (ReleaseManifestError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema": "avengine_release_tool_error_v1",
                    "status": "fail",
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
