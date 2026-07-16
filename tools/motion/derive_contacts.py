#!/usr/bin/env python3
"""Derive hash-bound M2 four-paw contact phases from explicit inputs.

The command intentionally does not infer contact joints from a retarget profile.
It accepts the emitted M2 anchor profile because that artifact binds explicit
``joint_id`` and ``joint_from_anchor`` values to the visual GLB hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from avengine.m2.actions import (  # noqa: E402
    baked_actions_content_sha256,
    read_baked_actions_npz,
)
from avengine.m2.glb import load_glb  # noqa: E402
from avengine.m2.habitat import build_habitat_asset_mapping  # noqa: E402
from avengine.m2.kinematics import (  # noqa: E402
    CONTACT_ORDER,
    AnchorDefinition,
    RigidTransform,
    derive_contact_phases,
)


ANCHOR_PROFILE_SCHEMA = "avengine_m2_emitter_anchors_v1"
JOINT_MAPPING_SCHEMA = "avengine_m2_habitat_joint_mapping_v1"


class ContactDerivationCliError(ValueError):
    """An input cannot be admitted to deterministic contact derivation."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _absolute_without_symlinks(path: Path, *, owner: str) -> Path:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ContactDerivationCliError(
                f"{owner} path must not contain a symbolic link: {absolute}"
            )
    return absolute


def _read_input(path: Path, *, owner: str, suffix: str) -> tuple[Path, bytes]:
    absolute = _absolute_without_symlinks(path, owner=owner)
    if (
        not absolute.is_file()
        or absolute.stat().st_size <= 0
        or absolute.suffix.lower() != suffix
    ):
        raise ContactDerivationCliError(
            f"{owner} must be a non-empty {suffix} regular file: {absolute}"
        )
    try:
        return absolute, absolute.read_bytes()
    except OSError as exc:
        raise ContactDerivationCliError(f"unable to read {owner}: {exc}") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContactDerivationCliError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ContactDerivationCliError(f"JSON contains non-finite number {value}")


def _json_object(payload: bytes, *, owner: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise ContactDerivationCliError(f"{owner} must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise ContactDerivationCliError(f"{owner} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ContactDerivationCliError(f"{owner} must contain one JSON object")
    return value


def _contact_anchors(
    profile: dict[str, Any], *, visual_sha256: str
) -> tuple[AnchorDefinition, ...]:
    if profile.get("schema") != ANCHOR_PROFILE_SCHEMA:
        raise ContactDerivationCliError(
            f"anchor profile schema must be {ANCHOR_PROFILE_SCHEMA!r}"
        )
    if profile.get("source_visual_sha256") != visual_sha256:
        raise ContactDerivationCliError(
            "anchor profile source_visual_sha256 must match visual.glb"
        )
    values = profile.get("anchors")
    if not isinstance(values, list):
        raise ContactDerivationCliError("anchor profile anchors must be an array")

    by_id: dict[str, AnchorDefinition] = {}
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ContactDerivationCliError(f"anchors[{index}] must be an object")
        if set(value) != {"anchor_id", "joint_id", "joint_from_anchor"}:
            raise ContactDerivationCliError(
                f"anchors[{index}] fields differ from the explicit anchor contract"
            )
        anchor_id = value["anchor_id"]
        if not isinstance(anchor_id, str) or not anchor_id:
            raise ContactDerivationCliError(
                f"anchors[{index}].anchor_id must be a non-empty string"
            )
        if anchor_id in by_id:
            raise ContactDerivationCliError(f"duplicate anchor_id {anchor_id!r}")
        transform = value["joint_from_anchor"]
        if not isinstance(transform, dict) or set(transform) != {
            "translation_m",
            "rotation_xyzw",
        }:
            raise ContactDerivationCliError(
                f"anchors[{index}].joint_from_anchor has invalid fields"
            )
        translation = transform["translation_m"]
        rotation = transform["rotation_xyzw"]
        if not isinstance(translation, list) or not isinstance(rotation, list):
            raise ContactDerivationCliError(
                f"anchors[{index}] transform values must be JSON arrays"
            )
        try:
            by_id[anchor_id] = AnchorDefinition(
                anchor_id=anchor_id,
                joint_id=value["joint_id"],
                joint_from_anchor=RigidTransform(tuple(translation), tuple(rotation)),
            )
        except (TypeError, ValueError) as exc:
            raise ContactDerivationCliError(
                f"anchors[{index}] is not a canonical anchor: {exc}"
            ) from exc

    missing = [contact_id for contact_id in CONTACT_ORDER if contact_id not in by_id]
    if missing:
        raise ContactDerivationCliError(
            f"anchor profile is missing fixed M2 contacts: {missing}"
        )
    declared_order = [
        value["anchor_id"] for value in values if value["anchor_id"] in CONTACT_ORDER
    ]
    if declared_order != list(CONTACT_ORDER):
        raise ContactDerivationCliError(
            f"anchor profile contacts must follow fixed M2 order {CONTACT_ORDER}"
        )
    return tuple(by_id[contact_id] for contact_id in CONTACT_ORDER)


def derive_contacts(
    *,
    visual_glb: Path,
    actions_npz: Path,
    joint_mapping_json: Path,
    anchor_profile_json: Path,
    output_json: Path,
) -> dict[str, Any]:
    """Derive and exclusively create one canonical contact report."""

    visual_path, visual_payload = _read_input(
        visual_glb, owner="visual GLB", suffix=".glb"
    )
    actions_path, actions_payload = _read_input(
        actions_npz, owner="baked actions", suffix=".npz"
    )
    _, mapping_payload = _read_input(
        joint_mapping_json, owner="Habitat joint mapping", suffix=".json"
    )
    _, anchor_payload = _read_input(
        anchor_profile_json, owner="anchor profile", suffix=".json"
    )
    output = _absolute_without_symlinks(output_json, owner="contact output")
    if output.exists() or output.is_symlink():
        raise ContactDerivationCliError(f"refusing to replace contact output: {output}")
    if output.suffix.lower() != ".json":
        raise ContactDerivationCliError("contact output must use the .json suffix")

    document = load_glb(visual_path)
    visual_sha256 = _sha256(visual_payload)
    if document.sha256 != visual_sha256:
        raise ContactDerivationCliError("parsed GLB hash differs from input bytes")

    mapping_value = _json_object(mapping_payload, owner="Habitat joint mapping")
    if mapping_value.get("schema") != JOINT_MAPPING_SCHEMA:
        raise ContactDerivationCliError(
            f"joint mapping schema must be {JOINT_MAPPING_SCHEMA!r}"
        )
    if mapping_value.get("source_glb_sha256") != visual_sha256:
        raise ContactDerivationCliError(
            "joint mapping source_glb_sha256 must match visual.glb"
        )
    try:
        mapping = build_habitat_asset_mapping(
            document,
            actor_from_skin_root=mapping_value["actor_from_skin_root"],
            actor_from_skin_root_source=mapping_value["actor_from_skin_root_source"],
        )
    except KeyError as exc:
        raise ContactDerivationCliError(
            f"joint mapping is missing required field {exc.args[0]!r}"
        ) from exc
    if mapping.joint_mapping_data() != mapping_value:
        raise ContactDerivationCliError(
            "joint mapping JSON differs from the mapping reconstructed from visual.glb"
        )

    actions = read_baked_actions_npz(actions_path)
    if baked_actions_content_sha256(actions) != _sha256(actions_payload):
        raise ContactDerivationCliError(
            "baked actions content hash differs from the admitted NPZ bytes"
        )
    anchors = _contact_anchors(
        _json_object(anchor_payload, owner="anchor profile"),
        visual_sha256=visual_sha256,
    )
    report = derive_contact_phases(mapping, actions, anchors)
    payload = report.to_canonical_json().encode("utf-8")
    if report.content_sha256() != _sha256(payload):
        raise ContactDerivationCliError("contact report canonical hash differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise ContactDerivationCliError(
            f"unable to create contact output: {exc}"
        ) from exc
    return {
        "status": "pass",
        "qualification_state": report.qualification_state,
        "qualification_claim": report.qualification_claim,
        "output": str(output),
        "output_sha256": _sha256(payload),
        "visual_glb_sha256": visual_sha256,
        "baked_actions_sha256": baked_actions_content_sha256(actions),
        "contact_order": list(CONTACT_ORDER),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-glb", type=Path, required=True)
    parser.add_argument("--actions-npz", type=Path, required=True)
    parser.add_argument("--joint-mapping", type=Path, required=True)
    parser.add_argument(
        "--anchors",
        "--anchor-profile",
        dest="anchor_profile",
        type=Path,
        required=True,
        help="Hash-bound avengine_m2_emitter_anchors_v1 JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = derive_contacts(
            visual_glb=args.visual_glb,
            actions_npz=args.actions_npz,
            joint_mapping_json=args.joint_mapping,
            anchor_profile_json=args.anchor_profile,
            output_json=args.output,
        )
    except (ContactDerivationCliError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
