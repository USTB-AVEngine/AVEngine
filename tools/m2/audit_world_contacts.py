#!/usr/bin/env python3
"""Fit M2 root cadence and emit hash-bound world-contact artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from avengine.contracts.json_io import load_json
from avengine.m2.actions import baked_actions_content_sha256, read_baked_actions_npz
from avengine.m2.contracts import validate_animal_asset_package
from avengine.m2.glb import load_glb
from avengine.m2.habitat import build_habitat_asset_mapping
from avengine.m2.kinematics import CONTACT_ORDER, AnchorDefinition, RigidTransform
from avengine.m2.world_contact import (
    derive_cadence_locked_contact_artifacts,
    infer_uniform_skin_linear_scale,
)


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


def _reference_scale(
    manifest_value: Path | None, *, target_document: Any
) -> tuple[float, dict[str, Any]]:
    if manifest_value is None:
        return 1.0, {
            "mode": "fixed_reference_unit_v1",
            "linear_scale": 1.0,
            "caller_supplied_linear_scale_allowed": False,
        }
    manifest_path = _regular_file(manifest_value, suffix=".json")
    manifest = load_json(manifest_path)
    errors = validate_animal_asset_package(
        manifest,
        manifest_path=manifest_path,
    )
    if errors:
        raise WorldContactCliError(
            "scale reference package is invalid: " + "; ".join(errors)
        )
    if manifest.get("admission_state") != "canary_qualified":
        raise WorldContactCliError("scale reference package must be canary_qualified")
    visual_records = [
        record
        for record in manifest.get("files", [])
        if isinstance(record, Mapping) and record.get("role") == "visual"
    ]
    if len(visual_records) != 1:
        raise WorldContactCliError(
            "scale reference package must contain exactly one visual role"
        )
    visual_record = visual_records[0]
    relative = visual_record.get("path")
    if not isinstance(relative, str) or not relative:
        raise WorldContactCliError("scale reference visual path is invalid")
    reference_visual = _regular_file(
        manifest_path.parent / relative,
        suffix=".glb",
    )
    reference_sha256 = _sha256_file(reference_visual)
    if (
        visual_record.get("sha256") != reference_sha256
        or visual_record.get("byte_size") != reference_visual.stat().st_size
    ):
        raise WorldContactCliError(
            "scale reference visual differs from its package binding"
        )
    reference_document = load_glb(reference_visual)
    measured = infer_uniform_skin_linear_scale(
        reference_document,
        target_document,
    )
    return measured.linear_scale, {
        "mode": "canary_package_skin_bone_ratio_v1",
        "caller_supplied_linear_scale_allowed": False,
        "reference_package_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "byte_size": manifest_path.stat().st_size,
            "admission_state": manifest["admission_state"],
        },
        "reference_visual_glb": {
            "path": str(reference_visual),
            "sha256": reference_sha256,
            "byte_size": reference_visual.stat().st_size,
        },
        "target_visual_sha256": target_document.sha256,
        "measurement": measured.to_json_data(),
    }


def _json_payload(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise WorldContactCliError(f"output is not canonical JSON: {error}") from error


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _preflight_output_pair(contacts_path: Path, audit_path: Path) -> tuple[Path, Path]:
    raw_contacts = _absolute_lexical(contacts_path)
    raw_audit = _absolute_lexical(audit_path)
    if raw_contacts == raw_audit:
        raise WorldContactCliError("contacts and audit outputs must be different paths")
    for destination in (raw_contacts, raw_audit):
        if destination.is_symlink():
            raise WorldContactCliError(
                f"output must not be a terminal symbolic link: {destination}"
            )
    for parent in {raw_contacts.parent, raw_audit.parent}:
        parent.mkdir(parents=True, exist_ok=True)
        if not parent.is_dir():
            raise WorldContactCliError(f"output parent is not a directory: {parent}")
    destinations = tuple(
        raw.parent.resolve(strict=True) / raw.name for raw in (raw_contacts, raw_audit)
    )
    if destinations[0] == destinations[1]:
        raise WorldContactCliError("contacts and audit outputs resolve to one path")
    for destination in destinations:
        if destination.is_symlink():
            raise WorldContactCliError(
                f"output must not be a terminal symbolic link: {destination}"
            )
        if destination.exists():
            raise WorldContactCliError(f"refusing to replace output: {destination}")
    return destinations


@dataclass(frozen=True)
class _CreatedOutput:
    path: Path
    device: int
    inode: int


def _unlink_if_owned(created: _CreatedOutput) -> None:
    try:
        current = created.path.lstat()
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(current.st_mode)
        and current.st_dev == created.device
        and current.st_ino == created.inode
    ):
        created.path.unlink()


def _exclusive_write_payload(path: Path, payload: bytes) -> _CreatedOutput:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as error:
        raise WorldContactCliError(
            f"refusing to create output {path}: {error}"
        ) from error
    opened = os.fstat(descriptor)
    created = _CreatedOutput(path=path, device=opened.st_dev, inode=opened.st_ino)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        _unlink_if_owned(created)
        raise
    return created


def _write_output_pair(
    contacts_path: Path,
    contact_report: Mapping[str, Any],
    audit_path: Path,
    audit: Mapping[str, Any],
) -> tuple[Path, Path]:
    # Serialize and preflight both destinations before creating either one.
    contact_payload = _json_payload(contact_report)
    audit_payload = _json_payload(audit)
    contacts_destination, audit_destination = _preflight_output_pair(
        contacts_path, audit_path
    )
    first = _exclusive_write_payload(contacts_destination, contact_payload)
    try:
        _exclusive_write_payload(audit_destination, audit_payload)
    except Exception:
        _unlink_if_owned(first)
        raise
    return contacts_destination, audit_destination


def _exclusive_write(path: Path, value: Mapping[str, Any]) -> None:
    """Backward-compatible single-output helper with hardened creation."""

    destination = _absolute_lexical(path)
    if destination.is_symlink():
        raise WorldContactCliError(
            f"output must not be a terminal symbolic link: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.parent.resolve(strict=True) / destination.name
    if destination.exists() or destination.is_symlink():
        raise WorldContactCliError(f"refusing to replace output: {destination}")
    payload = _json_payload(value)
    _exclusive_write_payload(destination, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-glb", type=Path, required=True)
    parser.add_argument("--actions-npz", type=Path, required=True)
    parser.add_argument("--joint-mapping", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--contacts-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument(
        "--reference-package-manifest",
        type=Path,
        help=(
            "Optional canary_qualified package used to derive, never supply, "
            "a uniform target/reference scale from exact skin bone lengths."
        ),
    )
    args = parser.parse_args()

    visual = _regular_file(args.visual_glb, suffix=".glb")
    actions_path = _regular_file(args.actions_npz, suffix=".npz")
    mapping_path = _regular_file(args.joint_mapping, suffix=".json")
    anchors_path = _regular_file(args.anchors, suffix=".json")
    document = load_glb(visual)
    if document.sha256 != _sha256_file(visual):
        raise WorldContactCliError("parsed GLB hash differs from file bytes")
    linear_scale, scale_reference = _reference_scale(
        args.reference_package_manifest,
        target_document=document,
    )
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
        linear_scale=linear_scale,
    )
    contact_report["scale_reference"] = scale_reference
    audit["scale_reference"] = scale_reference
    contact_payload = (
        json.dumps(
            contact_report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    audit["contact_phases_sha256"] = hashlib.sha256(contact_payload).hexdigest()
    contacts_output, audit_output = _write_output_pair(
        args.contacts_output,
        contact_report,
        args.audit_output,
        audit,
    )
    emitted_contact_sha = _sha256_file(contacts_output)
    if emitted_contact_sha != audit["contact_phases_sha256"]:
        raise WorldContactCliError("emitted contact report differs from audit binding")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "contacts": str(contacts_output),
                "contacts_sha256": emitted_contact_sha,
                "audit": str(audit_output),
                "audit_sha256": _sha256_file(audit_output),
                "root_step_fit": audit["root_step_fit"],
                "trajectory": audit["trajectory"],
                "uniform_linear_scale": audit["uniform_linear_scale"],
                "scale_reference": audit["scale_reference"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
