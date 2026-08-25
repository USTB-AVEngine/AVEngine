#!/usr/bin/env python3
"""Bake deterministic research-only local-translation action poses from one GLB."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.assets.glb import GlbError, load_glb
from avengine.assets.local_tr_actions import (
    LocalTRActionBakeError,
    bake_local_tr_actions,
    local_tr_actions_content_sha256,
    parse_local_tr_actions_npz,
    serialize_local_tr_actions_npz,
)


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        if created:
            path.unlink(missing_ok=True)
        raise


def _record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source animated GLB")
    parser.add_argument("--output", type=Path, required=True, help="Fresh local-TR NPZ")
    parser.add_argument(
        "--report", type=Path, required=True, help="Fresh bake report JSON"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    source = args.input.resolve()
    output = args.output.resolve()
    report_path = args.report.resolve()
    if len({source, output, report_path}) != 3:
        parser.error("input, output, and report paths must differ")
    if not source.is_file() or source.is_symlink():
        parser.error(f"input must be one regular non-symlink file: {source}")
    for owner, path in (("output", output), ("report", report_path)):
        if path.exists() or path.is_symlink():
            parser.error(f"refusing to replace {owner}: {path}")

    output_created = False
    try:
        document = load_glb(source)
        actions = bake_local_tr_actions(document)
        payload = serialize_local_tr_actions_npz(actions)
        expected_hash = local_tr_actions_content_sha256(actions)
        if expected_hash != hashlib.sha256(payload).hexdigest():
            raise LocalTRActionBakeError("canonical local-TR payload hash disagrees")
        _write_exclusive(output, payload)
        output_created = True
        parsed = parse_local_tr_actions_npz(output.read_bytes())
        if parsed != actions or sha256_file(output) != expected_hash:
            raise LocalTRActionBakeError("written local-TR NPZ failed strict readback")

        report: dict[str, Any] = {
            "schema": "avengine_m2_local_tr_actions_bake_report_v2",
            "status": "pass",
            "research_only": True,
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "source": _record(source),
            "output": _record(output),
            "action_contract": {
                "schema": "avengine_m2_local_tr_actions_v2",
                "sample_rate_hz": actions.sample_rate_hz,
                "time_base_hz": actions.time_base_hz,
                "runtime_joint_count": len(actions.runtime_joint_order),
                "runtime_joint_order": list(actions.runtime_joint_order),
                "translation_driven_joint_count": len(
                    actions.translation_driven_joint_ids
                ),
                "translation_driven_joint_ids": list(
                    actions.translation_driven_joint_ids
                ),
                "translation_semantics": "absolute_child_local_m",
                "rotation_semantics": "absolute_child_local_xyzw",
                "root_motion_policy": "static_root_actor_trajectory_external",
                "clips": [
                    {
                        "semantic_action_id": clip.semantic_action_id,
                        "source_action_name": clip.source_action_name,
                        "sample_count": clip.sample_count,
                        "loop_duration_ticks": clip.loop_duration_ticks,
                    }
                    for clip in actions.actions
                ],
            },
            "notes": [
                "This artifact is a non-qualifying local-TR v2 research input.",
                "It does not widen the formal rotation-only M2 action contract.",
            ],
        }
        report["report_content_sha256"] = canonical_json_sha256(report)
        report_payload = (
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _write_exclusive(report_path, report_payload)
    except (GlbError, LocalTRActionBakeError, OSError, ValueError) as exc:
        if output_created:
            output.unlink(missing_ok=True)
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "status": "pass",
                "qualification_claim": False,
                "output": str(output),
                "output_sha256": expected_hash,
                "report": str(report_path),
                "report_content_sha256": report["report_content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
