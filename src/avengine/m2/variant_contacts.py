"""Generic four-paw contact artifacts for explicit M2 variant specifications.

The variant spec owns taxonomy and joint names.  This module only joins its
declared anchors to one rebased GLB, one exact rebase report, and one canonical
baked-action NPZ.  It never guesses a Beagle joint or relaxes contact gates.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.m2.actions import (
    ActionBakeError,
    baked_actions_content_sha256,
    read_baked_actions_npz,
)
from avengine.m2.glb import GlbError, load_glb
from avengine.m2.habitat import (
    HabitatMappingError,
    build_habitat_asset_mapping_from_rebase_report,
)
from avengine.m2.kinematics import (
    CONTACT_ORDER,
    AnchorDefinition,
    KinematicsError,
    RigidTransform,
    derive_contact_phases,
)
from avengine.m2.variant_package import (
    VariantPackageError,
    VariantPackageSpec,
    load_variant_package_spec,
)


VARIANT_CONTACT_DERIVATION_SCHEMA = "avengine_m2_variant_contact_derivation_v1"
EMITTER_ANCHORS_SCHEMA = "avengine_m2_emitter_anchors_v1"
_CANONICAL_ANCHOR_ORDER = ("body", "head", "muzzle", *CONTACT_ORDER)


class VariantContactError(ValueError):
    """Variant contact inputs or exclusive outputs violate the strict join."""


def _absolute_without_symlinks(path: str | Path, *, owner: str) -> Path:
    absolute = Path(os.path.abspath(Path(path)))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise VariantContactError(
                f"{owner} path must not contain a symbolic link: {absolute}"
            )
    return absolute


def _regular_file(path: str | Path, *, owner: str, suffix: str) -> Path:
    resolved = _absolute_without_symlinks(path, owner=owner)
    if (
        resolved.suffix.lower() != suffix
        or not resolved.is_file()
        or resolved.stat().st_size <= 0
    ):
        raise VariantContactError(
            f"{owner} must be a non-empty {suffix} regular file: {resolved}"
        )
    return resolved


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VariantContactError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise VariantContactError(f"JSON contains non-finite number {value}")


def _load_json_file(path: Path, *, owner: str) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except OSError as exc:
        raise VariantContactError(f"unable to read {owner}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise VariantContactError(f"{owner} must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise VariantContactError(f"{owner} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise VariantContactError(f"{owner} must contain one JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _anchor_definition(record: Mapping[str, Any], *, owner: str) -> AnchorDefinition:
    if set(record) != {"anchor_id", "joint_id", "joint_from_anchor"}:
        raise VariantContactError(f"{owner} fields differ from the anchor contract")
    transform = record.get("joint_from_anchor")
    if not isinstance(transform, Mapping) or set(transform) != {
        "translation_m",
        "rotation_xyzw",
    }:
        raise VariantContactError(f"{owner}.joint_from_anchor is invalid")
    translation = transform["translation_m"]
    rotation = transform["rotation_xyzw"]
    if not isinstance(translation, list) or not isinstance(rotation, list):
        raise VariantContactError(f"{owner} transform values must be arrays")
    try:
        return AnchorDefinition(
            anchor_id=record["anchor_id"],
            joint_id=record["joint_id"],
            joint_from_anchor=RigidTransform(tuple(translation), tuple(rotation)),
        )
    except (KeyError, TypeError, KinematicsError) as exc:
        raise VariantContactError(f"{owner} is not canonical: {exc}") from exc


def _spec_anchors(
    spec: VariantPackageSpec, *, known_joint_ids: Sequence[str]
) -> tuple[tuple[AnchorDefinition, ...], tuple[AnchorDefinition, ...]]:
    if not isinstance(spec, VariantPackageSpec):
        raise VariantContactError("spec must come from load_variant_package_spec")
    by_id: dict[str, AnchorDefinition] = {}
    for index, record in enumerate(spec.anchors):
        anchor = _anchor_definition(record, owner=f"spec.anchors[{index}]")
        if anchor.anchor_id in by_id:
            raise VariantContactError(f"duplicate anchor ID {anchor.anchor_id!r}")
        by_id[anchor.anchor_id] = anchor
    missing = [
        anchor_id for anchor_id in _CANONICAL_ANCHOR_ORDER if anchor_id not in by_id
    ]
    if missing:
        raise VariantContactError(f"spec is missing required anchors: {missing}")
    known_joints = set(known_joint_ids)
    unknown = sorted(
        anchor.joint_id
        for anchor in by_id.values()
        if anchor.joint_id not in known_joints
    )
    if unknown:
        raise VariantContactError(
            f"spec anchors reference unknown visual joints: {unknown}"
        )

    extras = sorted(set(by_id) - set(_CANONICAL_ANCHOR_ORDER))
    all_anchors = tuple(
        by_id[anchor_id] for anchor_id in (*_CANONICAL_ANCHOR_ORDER, *extras)
    )
    paw_anchors = tuple(by_id[contact_id] for contact_id in CONTACT_ORDER)
    return all_anchors, paw_anchors


def _json_bytes(value: Mapping[str, Any]) -> bytes:
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


def _output_path(path: str | Path, *, owner: str) -> Path:
    output = _absolute_without_symlinks(path, owner=owner)
    if output.suffix.lower() != ".json":
        raise VariantContactError(f"{owner} must use the .json suffix")
    if output.exists() or output.is_symlink():
        raise VariantContactError(f"refusing to replace {owner}: {output}")
    return output


def _write_exclusive_pair(
    outputs: Sequence[tuple[Path, bytes, str]],
) -> None:
    """Reserve both outputs before writing and clean up this call's partial files."""

    try:
        for path, _, _ in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
        for path, _, owner in outputs:
            if _absolute_without_symlinks(path, owner=owner) != path:
                raise VariantContactError(f"{owner} path changed before emission")
    except OSError as exc:
        raise VariantContactError(
            f"unable to prepare output directories: {exc}"
        ) from exc

    streams: list[tuple[Path, BinaryIO]] = []
    try:
        for path, _, _ in outputs:
            streams.append((path, path.open("xb")))
        for (_, stream), (_, payload, _) in zip(streams, outputs, strict=True):
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        for _, stream in streams:
            stream.close()
        cleanup_errors: list[str] = []
        for path, _ in streams:
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_errors.append(f"{path}: {cleanup_exc}")
        suffix = f"; cleanup also failed: {cleanup_errors}" if cleanup_errors else ""
        raise VariantContactError(
            f"unable to create the exclusive output pair: {exc}{suffix}"
        ) from exc
    finally:
        for _, stream in streams:
            if not stream.closed:
                stream.close()


def derive_variant_contact_artifacts(
    *,
    spec_path: str | Path,
    visual_glb: str | Path,
    actions_npz: str | Path,
    rebase_report: str | Path,
    emitter_anchors_output: str | Path,
    contact_phases_output: str | Path,
) -> dict[str, Any]:
    """Derive package-ready emitter anchors and actor-space paw contacts."""

    try:
        spec = load_variant_package_spec(spec_path)
    except (OSError, ValueError, VariantPackageError) as exc:
        raise VariantContactError(f"invalid variant package spec: {exc}") from exc
    strict_spec_value, strict_spec_sha256 = _load_json_file(
        spec.path, owner="variant package spec"
    )
    if strict_spec_sha256 != spec.sha256 or sha256_file(spec.path) != spec.sha256:
        raise VariantContactError("variant package spec changed while being loaded")
    if strict_spec_value != spec.value:
        raise VariantContactError(
            "strict variant package spec differs from the validated spec"
        )

    visual_path = _regular_file(visual_glb, owner="rebased visual GLB", suffix=".glb")
    actions_path = _regular_file(actions_npz, owner="baked actions", suffix=".npz")
    rebase_path = _regular_file(rebase_report, owner="rebase report", suffix=".json")
    emitter_output = _output_path(emitter_anchors_output, owner="emitter anchor output")
    contacts_output = _output_path(contact_phases_output, owner="contact phase output")
    if emitter_output == contacts_output:
        raise VariantContactError("emitter and contact outputs must differ")
    input_paths = {spec.path, visual_path, actions_path, rebase_path}
    if emitter_output in input_paths or contacts_output in input_paths:
        raise VariantContactError("outputs must differ from every input")

    try:
        document = load_glb(visual_path)
    except (OSError, GlbError) as exc:
        raise VariantContactError(f"rebased visual GLB is invalid: {exc}") from exc
    visual_sha256 = sha256_file(visual_path)
    if document.sha256 != visual_sha256:
        raise VariantContactError("parsed visual hash differs from the GLB bytes")

    rebase_value, rebase_sha256 = _load_json_file(rebase_path, owner="rebase report")
    rebase_output = rebase_value.get("output")
    if (
        rebase_value.get("qualification_state") != "research_candidate"
        or rebase_value.get("qualification_claim") is not False
    ):
        raise VariantContactError(
            "rebase report must remain a non-qualifying research candidate"
        )
    if (
        not isinstance(rebase_output, Mapping)
        or rebase_output.get("byte_size") != document.byte_length
    ):
        raise VariantContactError(
            "rebase report output byte_size must match the rebased visual GLB"
        )
    try:
        mapping = build_habitat_asset_mapping_from_rebase_report(document, rebase_value)
    except HabitatMappingError as exc:
        raise VariantContactError(f"rebase/mapping validation failed: {exc}") from exc
    mapping_value = mapping.joint_mapping_data()

    try:
        actions = read_baked_actions_npz(actions_path)
    except (OSError, ActionBakeError) as exc:
        raise VariantContactError(f"baked actions are invalid: {exc}") from exc
    actions_file_sha256 = sha256_file(actions_path)
    actions_content_sha256 = baked_actions_content_sha256(actions)
    if actions_content_sha256 != actions_file_sha256:
        raise VariantContactError(
            "baked-action content hash differs from the exact NPZ bytes"
        )
    if actions.source_glb_sha256 != visual_sha256:
        raise VariantContactError("baked actions do not bind the rebased visual GLB")
    if actions.runtime_joint_order != mapping.runtime_joint_order:
        raise VariantContactError(
            "baked actions runtime joint order differs from the reconstructed mapping"
        )

    all_anchors, paw_anchors = _spec_anchors(spec, known_joint_ids=mapping.joint_order)
    emitter_value = {
        "schema": EMITTER_ANCHORS_SCHEMA,
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source_visual_sha256": visual_sha256,
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Y",
            "forward_axis": "-Z",
            "linear_unit": "meter",
            "quaternion_order": "xyzw",
        },
        "anchors": [anchor.to_json_data() for anchor in all_anchors],
    }
    try:
        contact_report = derive_contact_phases(mapping, actions, paw_anchors)
    except KinematicsError as exc:
        raise VariantContactError(
            f"actor-space contact derivation failed: {exc}"
        ) from exc
    emitter_payload = _json_bytes(emitter_value)
    contact_payload = contact_report.to_canonical_json().encode("utf-8")
    emitter_sha256 = hashlib.sha256(emitter_payload).hexdigest()
    contact_sha256 = hashlib.sha256(contact_payload).hexdigest()

    _write_exclusive_pair(
        (
            (emitter_output, emitter_payload, "emitter anchor output"),
            (contacts_output, contact_payload, "contact phase output"),
        )
    )
    emitted_anchors, emitted_anchors_sha256 = _load_json_file(
        emitter_output, owner="emitted anchor profile"
    )
    emitted_contacts, emitted_contacts_sha256 = _load_json_file(
        contacts_output, owner="emitted contact phases"
    )
    if emitted_anchors_sha256 != emitter_sha256:
        raise VariantContactError(
            "emitter anchor disk hash differs from the derived payload"
        )
    if emitted_contacts_sha256 != contact_sha256:
        raise VariantContactError(
            "contact phase disk hash differs from the derived payload"
        )
    if emitted_anchors != emitter_value:
        raise VariantContactError("emitter anchor readback differs from derived values")
    if emitted_contacts != contact_report.to_json_data():
        raise VariantContactError("contact phase readback differs from derived values")
    if contact_sha256 != contact_report.content_sha256():
        raise VariantContactError(
            "contact report canonical hash differs after emission"
        )

    return {
        "schema": VARIANT_CONTACT_DERIVATION_SCHEMA,
        "derivation_status": "completed",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "variant_package_spec": {
            "path": str(spec.path),
            "sha256": spec.sha256,
        },
        "visual_glb": {
            "path": str(visual_path),
            "sha256": visual_sha256,
        },
        "rebase_report": {
            "path": str(rebase_path),
            "sha256": rebase_sha256,
        },
        "baked_actions": {
            "path": str(actions_path),
            "sha256": actions_file_sha256,
        },
        "reconstructed_mapping_sha256": canonical_json_sha256(mapping_value),
        "emitter_anchors": {
            "path": str(emitter_output),
            "sha256": emitter_sha256,
            "anchor_count": len(all_anchors),
        },
        "contact_phases": {
            "path": str(contacts_output),
            "sha256": contact_sha256,
            "warning_count": len(contact_report.warnings),
            "warnings": [warning.to_json_data() for warning in contact_report.warnings],
            "thresholds": contact_report.thresholds.to_json_data(),
        },
    }


__all__ = [
    "EMITTER_ANCHORS_SCHEMA",
    "VARIANT_CONTACT_DERIVATION_SCHEMA",
    "VariantContactError",
    "derive_variant_contact_artifacts",
]
