#!/usr/bin/env python3
"""Build an exact 75-state M2 request for research-only human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from avengine.contracts.json_io import (
    load_json,
    resolve_declared_path,
    sha256_file,
    write_json,
)
from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.m2.actions import read_baked_actions_npz
from avengine.m2.contracts import validate_animal_asset_package
from avengine.m2.habitat_capture import load_research_review_inputs
from avengine.m2.timeline import build_m2_research_review_request


def _records_by_role(asset: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for value in asset.get("files", []):
        if not isinstance(value, Mapping) or not isinstance(value.get("role"), str):
            raise ValueError("asset files must contain role-bound objects")
        role = str(value["role"])
        if role in records:
            raise ValueError(f"duplicate asset file role: {role}")
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
        raise ValueError(f"asset package lacks role {role!r}") from exc
    if not isinstance(raw_path, str):
        raise ValueError(f"asset role {role!r} path must be a string")
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
    order = value.get("contact_order")
    if order != expected_order:
        raise ValueError("contact report order differs from the M2 contract")
    result: dict[str, tuple[tuple[bool, ...], ...]] = {}
    actions = value.get("actions")
    if not isinstance(actions, list):
        raise ValueError("contact report actions must be an array")
    for action in actions:
        if not isinstance(action, Mapping):
            raise ValueError("contact report action must be an object")
        action_id = action.get("semantic_action_id")
        frames = action.get("frames")
        if (
            action_id not in {"idle", "walk"}
            or action_id in result
            or not isinstance(frames, list)
        ):
            raise ValueError("contact report action mapping is invalid")
        decoded: list[tuple[bool, ...]] = []
        for frame in frames:
            states = frame.get("contacts") if isinstance(frame, Mapping) else None
            if not isinstance(states, list):
                raise ValueError("contact report frame states must be an array")
            if len(states) != len(expected_order):
                raise ValueError("contact report frame state count is invalid")
            frame_values: list[bool] = []
            for state, contact_id in zip(states, expected_order, strict=True):
                if (
                    not isinstance(state, Mapping)
                    or state.get("contact_id") != contact_id
                    or not isinstance(state.get("in_contact"), bool)
                ):
                    raise ValueError("contact report frame state order is invalid")
                frame_values.append(state["in_contact"])
            decoded.append(tuple(frame_values))
        result[str(action_id)] = tuple(decoded)
    if set(result) != {"idle", "walk"}:
        raise ValueError("contact report must contain exactly idle and walk")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--room-manifest", type=Path, required=True)
    parser.add_argument("--room-request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--request-id",
        default="rocketbox_beagle_m2_research_review_v3",
    )
    args = parser.parse_args()

    asset_path = args.asset_manifest.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"refusing to replace request output: {output}")
    asset = load_json(asset_path)
    errors = validate_animal_asset_package(asset, manifest_path=asset_path)
    if errors:
        raise ValueError("invalid animal package: " + "; ".join(errors))
    if asset.get("admission_state") != "research_candidate":
        raise ValueError("review request accepts only a research_candidate package")

    room_inputs = load_m1_inputs(args.room_manifest, args.room_request)
    records = _records_by_role(asset)
    actions = read_baked_actions_npz(_role_path(asset_path, records, "idle_poses"))
    contacts = _contact_phases(
        load_json(_role_path(asset_path, records, "contact_phases"))
    )
    request = build_m2_research_review_request(
        asset=asset,
        asset_manifest_sha256=sha256_file(asset_path),
        actions=actions,
        contact_phases=contacts,
        request_id=args.request_id,
        room_id=room_inputs.room["room_id"],
        seed=room_inputs.request["seed"],
    )
    write_json(output, request)
    load_research_review_inputs(asset_path, output)
    print(
        json.dumps(
            {
                "status": "pass",
                "review_only": True,
                "qualification_claim": False,
                "request": str(output),
                "request_sha256": sha256_file(output),
                "state_count": len(request["states"]),
                "view_ids": request["view_ids"],
                "modalities": request["modalities"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
