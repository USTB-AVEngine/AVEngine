#!/usr/bin/env python3
"""Fit M2 root cadence and emit hash-bound world-contact artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from avengine.contracts.json_io import load_json
from avengine.m2.actions import baked_actions_content_sha256, read_baked_actions_npz
from avengine.m2.glb import load_glb
from avengine.m2.habitat import build_habitat_asset_mapping
from avengine.m2.kinematics import CONTACT_ORDER, AnchorDefinition, RigidTransform
from avengine.m2.world_contact import derive_cadence_locked_contact_artifacts


class WorldContactCliError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(value: Path, *, suffix: str) -> Path:
    path = value.resolve()
    if path.suffix.lower() != suffix or not path.is_file() or path.is_symlink():
        raise WorldContactCliError(f"input must be a regular {suffix} file: {path}")
    return path


def _anchors(
    value: Mapping[str, Any], *, visual_sha256: str
) -> tuple[AnchorDefinition, ...]:
    if value.get("schema") != "avengine_m2_emitter_anchors_v1":
        raise WorldContactCliError("anchor profile schema is invalid")
    if value.get("source_visual_sha256") != visual_sha256:
        raise WorldContactCliError("anchor profile does not bind visual.glb")
    records = value.get("anchors")
    if not isinstance(records, list):
        raise WorldContactCliError("anchor profile anchors must be an array")
    by_id = {
        record.get("anchor_id"): record
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("anchor_id"), str)
    }
    result: list[AnchorDefinition] = []
    for contact_id in CONTACT_ORDER:
        record = by_id.get(contact_id)
        if not isinstance(record, Mapping):
            raise WorldContactCliError(f"missing contact anchor {contact_id!r}")
        transform = record.get("joint_from_anchor")
        if not isinstance(transform, Mapping):
            raise WorldContactCliError(f"anchor {contact_id!r} has no transform")
        result.append(
            AnchorDefinition(
                anchor_id=contact_id,
                joint_id=str(record.get("joint_id")),
                joint_from_anchor=RigidTransform(
                    tuple(transform.get("translation_m", ())),
                    tuple(transform.get("rotation_xyzw", ())),
                ),
            )
        )
    return tuple(result)


def _exclusive_write(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.resolve()
    if destination.exists() or destination.is_symlink():
        raise WorldContactCliError(f"refusing to replace output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with destination.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-glb", type=Path, required=True)
    parser.add_argument("--actions-npz", type=Path, required=True)
    parser.add_argument("--joint-mapping", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--contacts-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    visual = _regular_file(args.visual_glb, suffix=".glb")
    actions_path = _regular_file(args.actions_npz, suffix=".npz")
    mapping_path = _regular_file(args.joint_mapping, suffix=".json")
    anchors_path = _regular_file(args.anchors, suffix=".json")
    document = load_glb(visual)
    if document.sha256 != _sha256_file(visual):
        raise WorldContactCliError("parsed GLB hash differs from file bytes")
    mapping_value = load_json(mapping_path)
    if mapping_value.get("source_glb_sha256") != document.sha256:
        raise WorldContactCliError("joint mapping does not bind visual.glb")
    mapping = build_habitat_asset_mapping(
        document,
        actor_from_skin_root=mapping_value["actor_from_skin_root"],
        actor_from_skin_root_source=mapping_value["actor_from_skin_root_source"],
    )
    if mapping.joint_mapping_data() != mapping_value:
        raise WorldContactCliError(
            "joint mapping differs from reconstructed GLB mapping"
        )
    actions = read_baked_actions_npz(actions_path)
    if baked_actions_content_sha256(actions) != _sha256_file(actions_path):
        raise WorldContactCliError("baked-action content hash differs from NPZ bytes")
    anchors = _anchors(load_json(anchors_path), visual_sha256=document.sha256)
    contact_report, audit = derive_cadence_locked_contact_artifacts(
        mapping,
        actions,
        anchors,
    )
    _exclusive_write(args.contacts_output, contact_report)
    _exclusive_write(args.audit_output, audit)
    emitted_contact_sha = _sha256_file(args.contacts_output.resolve())
    if emitted_contact_sha != audit["contact_phases_sha256"]:
        raise WorldContactCliError("emitted contact report differs from audit binding")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "contacts": str(args.contacts_output.resolve()),
                "contacts_sha256": emitted_contact_sha,
                "audit": str(args.audit_output.resolve()),
                "audit_sha256": _sha256_file(args.audit_output.resolve()),
                "root_step_fit": audit["root_step_fit"],
                "trajectory": audit["trajectory"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
