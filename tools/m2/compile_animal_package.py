#!/usr/bin/env python3
"""Compile a pinned Rocketbox Beagle M2 research-candidate package.

This is intentionally a bounded, reproducible canary tool.  It derives the
contact evidence from explicit terminal-joint anchors, snapshots the pinned
upstream files and MIT notice, and invokes the strict package compiler.  It
never emits a human-review pass or promotes the asset to ``canary_qualified``.
Defaults preserve the historical v4 inputs; explicit paths and revisions are
required for replacement-motion candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from avengine.m2.actions import read_baked_actions_npz
from avengine.m2.glb import load_glb
from avengine.m2.habitat import build_habitat_asset_mapping_from_rebase_report
from avengine.m2.kinematics import (
    CONTACT_ORDER,
    AnchorDefinition,
    RigidTransform,
    derive_contact_phases,
)
from avengine.m2.package import (
    AnimalPackageIdentity,
    compile_research_candidate_animal_package,
)


_PINNED_ROCKETBOX_REVISION = "0943055db6ec570bcef9f2c8b41c9e5467c808f9"
_PINNED_SOURCE_HASHES = {
    "LICENSE.md": "17474e386e0b9e1a700cc3d06b2b0882a2c376d9c6b49c7f8274409b8f8d2352",
    "README.md": "bcf8013e3d5817a5dee5d1770b9943278189cd76b2d9b2abc243db99b703ecd6",
    "Assets/Animals/Dog_Beagle_01/Export/Dog_Beagle_01.fbx": (
        "db6bf29f8d568fc6d40c2fb0a9725854e6dc7ac90e310f7f3e8c49a431c45685"
    ),
    "Assets/Animals/Dog_Beagle_01/Textures/beagle_color.tga": (
        "f42e3816545f7ca39396a7ce5b2a6e0abac768d985a2f0b78f55fa5a8837d843"
    ),
    "Assets/Animals/Dog_Beagle_01/Textures/beagle_bump.tga": (
        "7d333ee8223ed3dc6865cb30d988f1c43facbc539f49d9c17bf463931897a302"
    ),
    "Assets/Animals/Dog_Beagle_01/Textures/beagle_specular.tga": (
        "64511ce6ef33cc8284006fb985e69a31d89bc66c16cf7aace35fb8137b054d9c"
    ),
}
_IDENTITY = (0.0, 0.0, 0.0, 1.0)
_ZERO = (0.0, 0.0, 0.0)
_SEMANTIC_JOINTS = {
    "body": "beagle Pelvis",
    "head": "beagle Head",
    "muzzle": "beagle Xtra Mouth",
    "paw_front_left": "beagle L Finger0",
    "paw_front_right": "beagle R Finger0",
    "paw_hind_left": "beagle L Toe0",
    "paw_hind_right": "beagle R Toe0",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: Path, *, owner: str) -> bytes:
    absolute = Path(path.absolute())
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{owner} must not contain a symbolic link: {path}")
    if not absolute.is_file():
        raise ValueError(f"{owner} is not a regular file: {path}")
    return absolute.read_bytes()


def _json_object(path: Path, *, owner: str) -> dict[str, Any]:
    value = json.loads(_read_regular(path, owner=owner).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a JSON object")
    return value


def _mapping(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _artifact(path: Path, *, root: Path, root_id: str) -> dict[str, Any]:
    payload = _read_regular(path, owner=f"{root_id} artifact")
    try:
        relative_path = path.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise ValueError(f"artifact is outside {root_id}: {path}") from exc
    return {
        "root_id": root_id,
        "path": relative_path.as_posix(),
        "byte_size": len(payload),
        "sha256": _sha256(payload),
    }


def _prepare_empty_directory(path: Path, *, owner: str) -> Path:
    absolute = path.absolute()
    if absolute.is_symlink():
        raise ValueError(f"{owner} must not be a symbolic link")
    if absolute.exists():
        if not absolute.is_dir() or any(absolute.iterdir()):
            raise ValueError(f"{owner} must be an empty directory: {absolute}")
    else:
        if not absolute.parent.is_dir():
            raise ValueError(f"{owner} parent does not exist: {absolute.parent}")
        absolute.mkdir()
    return absolute


def _repo_path(repo_root: Path, value: Path) -> Path:
    return value.absolute() if value.is_absolute() else repo_root / value


def _validate_motion_evidence(
    *,
    visual_sha256: str,
    actions_sha256: str,
    rebase_report: dict[str, Any],
    motion_profile: Path,
    retarget_report: Path,
    motion_qa_report: Path,
) -> None:
    profile_sha256 = _sha256(
        _read_regular(motion_profile, owner="motion retarget profile")
    )
    retarget = _json_object(retarget_report, owner="motion retarget report")
    motion_qa = _json_object(motion_qa_report, owner="motion QA report")
    if retarget.get("schema") != "avengine_motion_retarget_evidence_v1":
        raise ValueError("motion retarget report schema differs")
    if motion_qa.get("schema") != "avengine_motion_retarget_audit_v1":
        raise ValueError("motion QA report schema differs")
    for owner, value in (
        ("motion retarget report", retarget),
        ("motion QA report", motion_qa),
    ):
        if value.get("status") != "pass":
            raise ValueError(f"{owner} must pass")
        if value.get("qualification_state") != "research_candidate":
            raise ValueError(f"{owner} must remain a research candidate")
        if value.get("qualification_claim") is not False:
            raise ValueError(f"{owner} must not claim formal qualification")
        if value.get("formal_dataset_registration_authorized") is not False:
            raise ValueError(f"{owner} must not authorize dataset registration")
    motion_qa_payload = _mapping(motion_qa.get("qa"), owner="motion QA payload")
    if motion_qa_payload.get("status") != "pass":
        raise ValueError("motion QA report must pass")

    retarget_profile = _mapping(
        retarget.get("profile"), owner="motion retarget profile binding"
    )
    retarget_output = _mapping(
        retarget.get("output"), owner="motion retarget output binding"
    )
    bindings = _mapping(motion_qa.get("bindings"), owner="motion QA bindings")
    if retarget_profile.get("sha256") != profile_sha256:
        raise ValueError("motion profile hash differs from retarget evidence")
    if bindings.get("motion_profile_sha256") != profile_sha256:
        raise ValueError("motion profile hash differs from motion QA evidence")
    for field in (
        "profile_id",
        "adapter_id",
        "body_plan_id",
        "motion_family_id",
    ):
        if bindings.get(field) != retarget_profile.get(field):
            raise ValueError(f"motion evidence differs for {field}")
    rebase_source = _mapping(rebase_report.get("source"), owner="rebase source binding")
    if retarget_output.get("sha256") != rebase_source.get("sha256"):
        raise ValueError("retarget output hash differs from rebase input")
    if bindings.get("visual_glb_sha256") != visual_sha256:
        raise ValueError("motion QA visual hash differs from package visual")
    if bindings.get("baked_actions_sha256") != actions_sha256:
        raise ValueError("motion QA action hash differs from package actions")


def _anchors() -> tuple[AnchorDefinition, ...]:
    return tuple(
        AnchorDefinition(
            anchor_id=anchor_id,
            joint_id=joint_id,
            joint_from_anchor=RigidTransform(_ZERO, _IDENTITY),
        )
        for anchor_id, joint_id in _SEMANTIC_JOINTS.items()
    )


def _assert_pinned_source(rocketbox_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative_path, expected_hash in _PINNED_SOURCE_HASHES.items():
        path = rocketbox_root / relative_path
        record = _artifact(path, root=rocketbox_root, root_id="rocketbox_checkout")
        if record["sha256"] != expected_hash:
            raise ValueError(
                f"pinned Rocketbox source hash differs for {relative_path}: "
                f"{record['sha256']} != {expected_hash}"
            )
        records.append(record)
    return records


def main() -> int:
    repository_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repository_default)
    parser.add_argument(
        "--rocketbox-root",
        type=Path,
        default=Path("/data/datasets/rocketbox/Microsoft-Rocketbox"),
    )
    parser.add_argument(
        "--evidence-directory",
        type=Path,
        default=Path("tmp/m2/rocketbox_beagle_m2_package_inputs_v4"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/m2/rocketbox_beagle_m2_candidate_v4"),
    )
    parser.add_argument(
        "--visual-glb",
        type=Path,
        default=Path("tmp/m2/rocketbox_rebased_v3/visual.glb"),
    )
    parser.add_argument(
        "--rebase-report",
        type=Path,
        default=Path("tmp/m2/rocketbox_rebased_v3/rebase.json"),
    )
    parser.add_argument(
        "--rebase-deformation-report",
        type=Path,
        default=Path("tmp/m2/rocketbox_rebased_v3/deformation_verification.json"),
    )
    parser.add_argument(
        "--actions-npz",
        type=Path,
        default=Path("tmp/m2/rocketbox_actions_v1/actions.npz"),
    )
    parser.add_argument(
        "--action-report",
        type=Path,
        default=Path("tmp/m2/rocketbox_actions_v1/action_bake_report.json"),
    )
    parser.add_argument(
        "--habitat-static-probe",
        type=Path,
        default=Path("tmp/m2/rocketbox_rebased_v3_probe_optin/probe.json"),
    )
    parser.add_argument(
        "--habitat-animation-review",
        type=Path,
        default=Path("tmp/m2/rocketbox_habitat_review_v4/review_report.json"),
    )
    parser.add_argument(
        "--static-qa",
        type=Path,
        default=Path("tmp/m2/rocketbox_auto_qa_v1/static_geometry.json"),
    )
    parser.add_argument(
        "--deformation-qa",
        type=Path,
        default=Path("tmp/m2/rocketbox_auto_qa_v1/deformation.json"),
    )
    parser.add_argument(
        "--animation-qa",
        type=Path,
        default=Path("tmp/m2/rocketbox_auto_qa_v1/animation.json"),
    )
    parser.add_argument(
        "--normalization-report",
        type=Path,
        default=Path("tmp/m2/rocketbox_normalized_v2/normalization.json"),
    )
    parser.add_argument(
        "--retarget-report",
        type=Path,
        help="Optional hash-bound motion-retarget evidence added to source lineage",
    )
    parser.add_argument(
        "--motion-profile",
        type=Path,
        help="Profile bound to both retarget and motion-QA evidence",
    )
    parser.add_argument(
        "--motion-qa-report",
        type=Path,
        help="Hash-bound body-plan-neutral motion-QA evidence",
    )
    parser.add_argument(
        "--asset-id",
        default="rocketbox_dog_beagle_01_m2_v4_candidate",
    )
    parser.add_argument("--body-plan-id", default="quadruped_dog")
    parser.add_argument(
        "--skeleton-revision", default="rocketbox-beagle-skeleton-m2-v3"
    )
    parser.add_argument("--weights-revision", default="rocketbox-beagle-weights-m2-v3")
    parser.add_argument(
        "--action-revision", default="rocketbox-beagle-idle-walk-baked-v1"
    )
    args = parser.parse_args()
    repo_root = args.repo_root.absolute()
    rocketbox_root = args.rocketbox_root.absolute()
    evidence_directory = (
        args.evidence_directory
        if args.evidence_directory.is_absolute()
        else repo_root / args.evidence_directory
    )
    output = args.output if args.output.is_absolute() else repo_root / args.output
    evidence_directory = _prepare_empty_directory(
        evidence_directory, owner="evidence_directory"
    )

    visual = _repo_path(repo_root, args.visual_glb)
    rebase = _repo_path(repo_root, args.rebase_report)
    rebase_deformation = _repo_path(repo_root, args.rebase_deformation_report)
    actions_path = _repo_path(repo_root, args.actions_npz)
    action_report = _repo_path(repo_root, args.action_report)
    habitat_static_probe = _repo_path(repo_root, args.habitat_static_probe)
    habitat_animation_review = _repo_path(repo_root, args.habitat_animation_review)
    static_qa = _repo_path(repo_root, args.static_qa)
    deformation_qa = _repo_path(repo_root, args.deformation_qa)
    animation_qa = _repo_path(repo_root, args.animation_qa)
    normalization = _repo_path(repo_root, args.normalization_report)
    retarget_report = (
        _repo_path(repo_root, args.retarget_report)
        if args.retarget_report is not None
        else None
    )
    motion_profile = (
        _repo_path(repo_root, args.motion_profile)
        if args.motion_profile is not None
        else None
    )
    motion_qa_report = (
        _repo_path(repo_root, args.motion_qa_report)
        if args.motion_qa_report is not None
        else None
    )
    motion_evidence = (motion_profile, retarget_report, motion_qa_report)
    if any(path is not None for path in motion_evidence) and not all(
        path is not None for path in motion_evidence
    ):
        raise ValueError(
            "motion-profile, retarget-report and motion-qa-report must be supplied "
            "together"
        )

    document = load_glb(visual)
    actions = read_baked_actions_npz(actions_path)
    rebase_value = _json_object(rebase, owner="rebase report")
    if motion_profile is not None:
        assert retarget_report is not None
        assert motion_qa_report is not None
        _validate_motion_evidence(
            visual_sha256=document.sha256,
            actions_sha256=_sha256(_read_regular(actions_path, owner="baked actions")),
            rebase_report=rebase_value,
            motion_profile=motion_profile,
            retarget_report=retarget_report,
            motion_qa_report=motion_qa_report,
        )
    mapping = build_habitat_asset_mapping_from_rebase_report(document, rebase_value)
    all_anchors = _anchors()
    contact_anchors = tuple(
        anchor for anchor in all_anchors if anchor.anchor_id in CONTACT_ORDER
    )
    contacts = derive_contact_phases(mapping, actions, contact_anchors)
    contacts_path = evidence_directory / "contact_phases.json"
    contacts_path.write_text(contacts.to_canonical_json(), encoding="utf-8")

    source_records = _assert_pinned_source(rocketbox_root)
    lineage_paths = [
        normalization,
        rebase,
        rebase_deformation,
        action_report,
        static_qa,
        deformation_qa,
        animation_qa,
        habitat_static_probe,
        habitat_animation_review,
    ]
    if motion_profile is not None:
        assert retarget_report is not None
        assert motion_qa_report is not None
        lineage_paths.extend((motion_profile, retarget_report, motion_qa_report))
    source_manifest = {
        "schema": "avengine_m2_rocketbox_beagle_source_snapshot_v1",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "formal_dataset_registration_authorized": False,
        "source_repository": {
            "url": "https://github.com/microsoft/Microsoft-Rocketbox.git",
            "revision": _PINNED_ROCKETBOX_REVISION,
            "root_id": "rocketbox_checkout",
        },
        "source_artifacts": source_records,
        "m2_lineage_evidence": [
            _artifact(path, root=repo_root, root_id="avengine_habitat_native_repo")
            for path in lineage_paths
        ],
        "lineage_status": "legacy_conversion_assertion_review_required",
        "notes": [
            "Raw Beagle and license files are hash-pinned to the Rocketbox checkout.",
            "The historical FBX-to-GLB conversion itself lacks a complete hash-bound "
            "conversion manifest, so this snapshot cannot authorize formal registry "
            "promotion.",
        ],
    }
    source_manifest_path = evidence_directory / "source_manifest.json"
    _write_json(source_manifest_path, source_manifest)

    license_record = next(
        record for record in source_records if record["path"] == "LICENSE.md"
    )
    license_snapshot = {
        "schema": "avengine_m2_license_snapshot_v1",
        "license": "MIT",
        "allowed_use": "review_required",
        "redistribution": "review_required",
        "license_file": license_record,
        "source_revision": _PINNED_ROCKETBOX_REVISION,
        "qualification_claim": False,
        "decision_reason": (
            "The pinned Rocketbox checkout carries MIT, but package-level "
            "redistribution and the legacy conversion lineage remain review-required."
        ),
    }
    license_snapshot_path = evidence_directory / "license_snapshot.json"
    _write_json(license_snapshot_path, license_snapshot)

    identity = AnimalPackageIdentity(
        asset_id=args.asset_id,
        template_id="rocketbox_dog_beagle_01",
        body_plan_id=args.body_plan_id,
        morphotype_id="beagle",
        skeleton_revision=args.skeleton_revision,
        weights_revision=args.weights_revision,
        collision_revision="m2-kinematic-rest-bbox-proxy-v1",
        action_revision=args.action_revision,
        source="Microsoft Rocketbox Dog_Beagle_01",
        source_revision=_PINNED_ROCKETBOX_REVISION,
        license="MIT",
        allowed_use="review_required",
        redistribution="review_required",
        semantic_id=200,
    )
    manifest_path = compile_research_candidate_animal_package(
        output_directory=output,
        identity=identity,
        visual_glb=visual,
        rebase_report=rebase,
        rebase_deformation_report=rebase_deformation,
        action_report=action_report,
        static_qa=static_qa,
        deformation_qa=deformation_qa,
        animation_qa=animation_qa,
        habitat_static_probe=habitat_static_probe,
        habitat_animation_review=habitat_animation_review,
        baked_actions=actions_path,
        contacts=contacts_path,
        anchor_definitions=all_anchors,
        source_manifest=source_manifest_path,
        license_snapshot=license_snapshot_path,
    )
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload)
    print(
        json.dumps(
            {
                "status": "pass",
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_payload),
                "admission_state": manifest["admission_state"],
                "qualification": manifest["qualification"],
                "contact_report": str(contacts_path),
                "contact_report_sha256": _sha256(contacts_path.read_bytes()),
                "evidence_directory": str(evidence_directory),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
