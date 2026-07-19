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
import sys
from typing import Sequence

from avengine.release import (
    ReleaseManifestError,
    load_json_strict,
    prepare_release_manifest,
    verify_release_attestation,
    verify_release_manifest,
    write_release_attestation,
)
from avengine.release_receipt import (
    DEFAULT_RLR_SUBMODULE_PATH,
    execute_test_receipt,
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

    receipt = commands.add_parser(
        "receipt", help="write one immutable structured M6 test execution receipt"
    )
    receipt.add_argument("--output", type=Path, required=True)
    receipt.add_argument("--workspace-root", type=Path, default=Path.cwd())
    receipt.add_argument("--habitat-runtime-root", type=Path, required=True)
    receipt.add_argument("--receipt-id", required=True)
    receipt.add_argument(
        "--layer-id",
        required=True,
        choices=(
            "fast-unit",
            "slow-hermetic",
            "native-habitat",
            "rlr-audio",
            "blender-assets",
            "media-readback",
        ),
    )
    receipt.add_argument(
        "--junit-xml",
        required=True,
        help="fresh workspace-relative JUnit XML path the command must generate",
    )
    receipt.add_argument(
        "--rlr-submodule-path",
        default=DEFAULT_RLR_SUBMODULE_PATH,
    )
    receipt.add_argument(
        "execution_command",
        nargs=argparse.REMAINDER,
        help="exact executed argv, placed after --",
    )

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
    verify.add_argument(
        "--output",
        type=Path,
        help="write one immutable successful post-tag attestation",
    )

    verify_attestation = commands.add_parser(
        "verify-attestation",
        help="re-run the live release verifier against a retained attestation",
    )
    verify_attestation.add_argument("--attestation", type=Path, required=True)
    verify_attestation.add_argument(
        "--avengine-root", type=Path, default=Path.cwd()
    )
    verify_attestation.add_argument("--habitat-runtime-root", type=Path, required=True)
    verify_attestation.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        metavar="ROOT_ID=PATH",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    arguments = _parser().parse_args(raw_arguments)
    try:
        if arguments.command == "receipt":
            executed_command = list(arguments.execution_command)
            if executed_command[:1] == ["--"]:
                executed_command = executed_command[1:]
            if "--" not in raw_arguments:
                raise ValueError(
                    "receipt execution command must be placed after an explicit --"
                )
            separator = raw_arguments.index("--")
            if raw_arguments[separator + 1 :] != executed_command:
                raise ValueError(
                    "the first explicit -- must delimit the receipt execution command"
                )
            execution = execute_test_receipt(
                arguments.output,
                workspace_root=arguments.workspace_root,
                habitat_runtime_root=arguments.habitat_runtime_root,
                receipt_id=arguments.receipt_id,
                test_layer_id=arguments.layer_id,
                junit_xml=arguments.junit_xml,
                command=executed_command,
                rlr_submodule_path=arguments.rlr_submodule_path,
            )
            result = {
                "schema": "avengine_test_execution_receipt_write_result_v1",
                "status": execution.receipt["status"],
                "write_status": "written",
                "path": str(execution.path),
                "exit_code": execution.exit_code,
                "result_totals": execution.receipt["result_totals"],
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return execution.exit_code

        roots = _artifact_roots(arguments.artifact_root)
        if arguments.command == "verify-attestation":
            report = verify_release_attestation(
                arguments.attestation,
                avengine_root=arguments.avengine_root,
                habitat_runtime_root=arguments.habitat_runtime_root,
                artifact_roots=roots,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["status"] == "pass" else 1
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

        if arguments.output is None:
            report = verify_release_manifest(
                arguments.manifest,
                avengine_root=arguments.avengine_root,
                habitat_runtime_root=arguments.habitat_runtime_root,
                artifact_roots=roots,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["status"] == "pass" else 1
        command = [sys.executable, str(Path(__file__).resolve()), *raw_arguments]
        _, attestation = write_release_attestation(
            arguments.output,
            manifest_path=arguments.manifest,
            avengine_root=arguments.avengine_root,
            habitat_runtime_root=arguments.habitat_runtime_root,
            verification_command=command,
            artifact_roots=roots,
        )
        print(json.dumps(attestation, indent=2, sort_keys=True))
        return 0
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
