#!/usr/bin/env python3
"""Build one hash-bound formal M2 canary capture request."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from avengine.contracts.json_io import (
    load_json,
    resolve_declared_path,
    sha256_file,
)
from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.assets.actions import read_baked_actions_npz
from avengine.assets.contracts import (
    load_and_validate_inputs as load_m2_inputs,
    validate_animal_asset_package,
)
from avengine.assets.timeline import M2CanaryTrajectory, build_m2_capture_request


class CanaryRequestCliError(ValueError):
    """The declared package, room, trajectory, or output is not admissible."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--room-manifest", type=Path, required=True)
    parser.add_argument("--room-request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--world-contact-audit",
        type=Path,
        help="Optional passing world-contact audit that defines the root trajectory",
    )
    parser.add_argument("--request-id", default="m2_formal_canary_v1")
    return parser


def _records_by_role(asset: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    raw_records = asset.get("files")
    if not isinstance(raw_records, list):
        raise CanaryRequestCliError("asset files must be an array")
    for value in raw_records:
        if not isinstance(value, Mapping) or not isinstance(value.get("role"), str):
            raise CanaryRequestCliError("asset files must contain role-bound objects")
        role = str(value["role"])
        if role in records:
            raise CanaryRequestCliError(f"duplicate asset file role: {role}")
        records[role] = value
    return records


def _role_path(
    manifest: Path,
    records: Mapping[str, Mapping[str, Any]],
    role: str,
) -> Path:
    try:
        raw_path = records[role]["path"]
    except KeyError as exc:
        raise CanaryRequestCliError(f"asset package lacks role {role!r}") from exc
    if not isinstance(raw_path, str):
        raise CanaryRequestCliError(f"asset role {role!r} path must be a string")
    return resolve_declared_path(raw_path, manifest_dir=manifest.parent)


def _contact_phases(
    value: Mapping[str, Any],
) -> dict[str, tuple[tuple[bool, ...], ...]]:
    expected_order = [
        "paw_front_left",
        "paw_front_right",
        "paw_hind_left",
        "paw_hind_right",
    ]
    if value.get("contact_order") != expected_order:
        raise CanaryRequestCliError("contact report order differs from the M2 contract")
    actions = value.get("actions")
    if not isinstance(actions, list):
        raise CanaryRequestCliError("contact report actions must be an array")
    result: dict[str, tuple[tuple[bool, ...], ...]] = {}
    for action in actions:
        if not isinstance(action, Mapping):
            raise CanaryRequestCliError("contact report action must be an object")
        action_id = action.get("semantic_action_id")
        frames = action.get("frames")
        if (
            action_id not in {"idle", "walk"}
            or action_id in result
            or not isinstance(frames, list)
        ):
            raise CanaryRequestCliError("contact report action mapping is invalid")
        decoded: list[tuple[bool, ...]] = []
        for frame in frames:
            states = frame.get("contacts") if isinstance(frame, Mapping) else None
            if not isinstance(states, list) or len(states) != len(expected_order):
                raise CanaryRequestCliError(
                    "contact report frame state count is invalid"
                )
            frame_values: list[bool] = []
            for state, contact_id in zip(states, expected_order, strict=True):
                if (
                    not isinstance(state, Mapping)
                    or state.get("contact_id") != contact_id
                    or not isinstance(state.get("in_contact"), bool)
                ):
                    raise CanaryRequestCliError(
                        "contact report frame state order is invalid"
                    )
                frame_values.append(bool(state["in_contact"]))
            decoded.append(tuple(frame_values))
        result[str(action_id)] = tuple(decoded)
    if set(result) != {"idle", "walk"}:
        raise CanaryRequestCliError("contact report must contain exactly idle and walk")
    return result


def _trajectory_from_audit(
    audit_path: Path,
    *,
    records: Mapping[str, Mapping[str, Any]],
) -> M2CanaryTrajectory:
    declared = audit_path.absolute()
    if not declared.is_file() or declared.is_symlink():
        raise CanaryRequestCliError(
            f"world-contact audit must be a regular file: {declared}"
        )
    path = declared.resolve()
    audit = load_json(path)
    trajectory = audit.get("trajectory")
    gate = audit.get("gate")
    action_hash = audit.get("baked_actions_sha256")
    idle_action_hash = records["idle_poses"].get("sha256")
    walk_action_hash = records["walk_poses"].get("sha256")
    if (
        audit.get("schema") != "avengine_m2_world_contact_audit_v1"
        or audit.get("status") != "pass"
        or audit.get("qualification_claim") is not False
        or not isinstance(trajectory, Mapping)
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or trajectory.get("walk_frame_count") != 45
        or trajectory.get("sample_rate_hz") != 15
        or audit.get("source_glb_sha256") != records["visual"].get("sha256")
        or not isinstance(action_hash, str)
        or action_hash != idle_action_hash
        or action_hash != walk_action_hash
        or audit.get("contact_phases_sha256") != records["contact_phases"].get("sha256")
    ):
        raise CanaryRequestCliError(
            "world-contact audit must pass and bind package visual/actions/contacts"
        )
    try:
        return M2CanaryTrajectory(
            start_translation_m=tuple(trajectory["start_translation_m"]),
            end_translation_m=tuple(trajectory["end_translation_m"]),
            rotation_xyzw=tuple(trajectory["rotation_xyzw"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CanaryRequestCliError(
            "world-contact audit trajectory is invalid"
        ) from exc


def _exclusive_write_json(path: Path, value: Mapping[str, Any]) -> Path:
    declared = path.absolute()
    if declared.exists() or declared.is_symlink():
        raise CanaryRequestCliError(f"refusing to replace request output: {declared}")
    destination = declared.resolve()
    if destination.exists() or destination.is_symlink():
        raise CanaryRequestCliError(
            f"refusing to replace request output: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise CanaryRequestCliError(
            f"refusing to replace request output: {destination}"
        ) from exc
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    asset_path = args.asset_manifest.resolve()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise CanaryRequestCliError(f"refusing to replace request output: {output}")

    asset = load_json(asset_path)
    errors = validate_animal_asset_package(asset, manifest_path=asset_path)
    if errors:
        raise CanaryRequestCliError("invalid animal package: " + "; ".join(errors))
    if asset.get("admission_state") != "canary_qualified":
        raise CanaryRequestCliError(
            "formal request accepts only a canary_qualified package"
        )

    room_inputs = load_m1_inputs(args.room_manifest, args.room_request)
    records = _records_by_role(asset)
    actions = read_baked_actions_npz(_role_path(asset_path, records, "idle_poses"))
    contacts = _contact_phases(
        load_json(_role_path(asset_path, records, "contact_phases"))
    )
    trajectory = (
        _trajectory_from_audit(args.world_contact_audit, records=records)
        if args.world_contact_audit is not None
        else None
    )
    request = build_m2_capture_request(
        asset=asset,
        asset_manifest_sha256=sha256_file(asset_path),
        actions=actions,
        contact_phases=contacts,
        request_id=args.request_id,
        room_id=room_inputs.room["room_id"],
        seed=room_inputs.request["seed"],
        trajectory=trajectory,
    )
    output = _exclusive_write_json(output, request)
    validated = load_m2_inputs(asset_path, output)
    print(
        json.dumps(
            {
                "status": "pass",
                "review_only": False,
                "admission_state": validated.asset["admission_state"],
                "request": str(output),
                "request_sha256": sha256_file(output),
                "state_count": len(validated.request["states"]),
                "view_ids": validated.request["view_ids"],
                "modalities": validated.request["modalities"],
                "trajectory_source": (
                    str(args.world_contact_audit.resolve())
                    if args.world_contact_audit is not None
                    else "default_m2_canary_trajectory"
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
