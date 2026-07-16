"""Fail-closed M2 research-candidate to canary admission.

The promotion graph is deliberately acyclic: review and provenance artifacts
bind the immutable research-candidate manifest, while only the final manifest
binds those admission artifacts.  The final manifest is never one of its own
inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    write_json,
)
from avengine.m2.contracts import (
    FORMAL_MODALITIES,
    validate_animal_asset_package,
    validate_human_visual_review,
)
from avengine.m2.habitat_capture import (
    EVIDENCE_SCHEMA,
    load_research_review_inputs,
    verify_saved_capture_arrays,
)


ROCKETBOX_REPOSITORY = "https://github.com/microsoft/Microsoft-Rocketbox.git"
ROCKETBOX_REVISION = "0943055db6ec570bcef9f2c8b41c9e5467c808f9"
ROCKETBOX_LICENSE_SHA256 = (
    "17474e386e0b9e1a700cc3d06b2b0882a2c376d9c6b49c7f8274409b8f8d2352"
)
ROCKETBOX_README_SHA256 = (
    "bcf8013e3d5817a5dee5d1770b9943278189cd76b2d9b2abc243db99b703ecd6"
)
USER_DECISION_DATE = "2026-07-17"
USER_DECISION_STATEMENT = (
    "视频里面的后腿我觉得已经自然了，所以你可以继续完成M2没完成的地方并提交收尾"
)


class CanaryPromotionError(ValueError):
    """Raised when any admission input or output fails closed."""


@dataclass(frozen=True)
class ExpectedArtifact:
    path: Path
    sha256: str
    label: str | None = None


@dataclass(frozen=True)
class PromotionResult:
    manifest_path: Path
    manifest_sha256: str
    human_review_path: Path
    human_review_sha256: str
    provenance_path: Path
    provenance_sha256: str


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _regular_file(path: str | Path, *, owner: str) -> Path:
    absolute = Path(path).absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise CanaryPromotionError(
                f"{owner} must not contain a symbolic link: {path}"
            )
    if not absolute.is_file():
        raise CanaryPromotionError(f"{owner} is not a regular file: {path}")
    return absolute


def _safe_tree(path: str | Path, *, owner: str) -> Path:
    root = Path(path).absolute()
    if root.is_symlink() or not root.is_dir():
        raise CanaryPromotionError(f"{owner} is not a regular directory: {path}")
    for directory, names, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*names, *filenames]:
            if (base / name).is_symlink():
                raise CanaryPromotionError(
                    f"{owner} must not contain symbolic links: {base / name}"
                )
    return root


def _artifact(path: Path, *, relative_to: Path) -> dict[str, Any]:
    file_path = _regular_file(path, owner="admission artifact")
    try:
        relative_path = file_path.relative_to(relative_to.absolute())
    except ValueError as error:
        raise CanaryPromotionError(
            f"admission artifact is outside its package: {file_path}"
        ) from error
    return {
        "path": relative_path.as_posix(),
        "byte_size": file_path.stat().st_size,
        "sha256": sha256_file(file_path),
    }


def _package_file_record(
    path: Path, *, role: str, package_root: Path
) -> dict[str, Any]:
    record = _artifact(path, relative_to=package_root)
    return {"role": role, **record}


def _load_object(path: Path, *, owner: str) -> dict[str, Any]:
    file_path = _regular_file(path, owner=owner)
    try:
        return load_json(file_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CanaryPromotionError(f"{owner} is not a JSON object: {error}") from error


def _records_by_role(asset: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for record in asset.get("files", []):
        if not isinstance(record, Mapping) or not isinstance(record.get("role"), str):
            raise CanaryPromotionError("candidate files must be role-bound objects")
        role = str(record["role"])
        if role in records:
            raise CanaryPromotionError(f"candidate duplicates file role: {role}")
        records[role] = record
    return records


def _candidate_rocketbox_lineage(
    candidate: Mapping[str, Any], candidate_manifest: Path
) -> None:
    provenance = candidate.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CanaryPromotionError("candidate lacks provenance")
    if (
        provenance.get("source") != "Microsoft Rocketbox Dog_Beagle_01"
        or provenance.get("source_revision") != ROCKETBOX_REVISION
        or provenance.get("license") != "MIT"
    ):
        raise CanaryPromotionError("candidate is not the pinned MIT Rocketbox Beagle")

    record = _records_by_role(candidate).get("provenance_manifest")
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise CanaryPromotionError("candidate lacks a provenance_manifest role")
    lineage_path = _regular_file(
        candidate_manifest.parent / str(record["path"]),
        owner="candidate provenance manifest",
    )
    lineage = _load_object(lineage_path, owner="candidate provenance manifest")
    source_wrapper = lineage.get("source_manifest")
    source_snapshot = (
        source_wrapper.get("snapshot") if isinstance(source_wrapper, Mapping) else None
    )
    if not isinstance(source_snapshot, Mapping):
        raise CanaryPromotionError("candidate provenance lacks its source snapshot")
    repository = source_snapshot.get("source_repository")
    if not isinstance(repository, Mapping) or (
        repository.get("url") != ROCKETBOX_REPOSITORY
        or repository.get("revision") != ROCKETBOX_REVISION
    ):
        raise CanaryPromotionError("candidate Rocketbox repository pin differs")
    source_records = source_snapshot.get("source_artifacts")
    by_path = (
        {
            value.get("path"): value
            for value in source_records
            if isinstance(value, Mapping) and isinstance(value.get("path"), str)
        }
        if isinstance(source_records, list)
        else {}
    )
    expected = {
        "LICENSE.md": ROCKETBOX_LICENSE_SHA256,
        "README.md": ROCKETBOX_README_SHA256,
    }
    for path, digest in expected.items():
        value = by_path.get(path)
        if not isinstance(value, Mapping) or value.get("sha256") != digest:
            raise CanaryPromotionError(f"candidate source snapshot differs for {path}")


def _validate_candidate(candidate_manifest: Path) -> dict[str, Any]:
    candidate = _load_object(candidate_manifest, owner="candidate asset manifest")
    errors = validate_animal_asset_package(
        candidate,
        manifest_path=candidate_manifest,
    )
    if errors:
        raise CanaryPromotionError("invalid research candidate: " + "; ".join(errors))
    qualification = candidate.get("qualification")
    if (
        candidate.get("admission_state") != "research_candidate"
        or not isinstance(qualification, Mapping)
        or qualification.get("automatic_qa_status") != "pass"
        or qualification.get("human_visual_review_status") != "not_run"
        or qualification.get("human_review_binding_sha256") is not None
    ):
        raise CanaryPromotionError(
            "promotion requires an automatic-QA-pass, unreviewed research_candidate"
        )
    if "human_visual_review" in _records_by_role(candidate):
        raise CanaryPromotionError("candidate already contains a human review role")
    _candidate_rocketbox_lineage(candidate, candidate_manifest)
    return candidate


def _validate_expected_artifact(
    value: ExpectedArtifact, *, owner: str = "diagnostic video"
) -> Path:
    if not _is_sha256(value.sha256):
        raise CanaryPromotionError(f"{owner} expected SHA-256 must be lowercase")
    path = _regular_file(value.path, owner=owner)
    actual = sha256_file(path)
    if actual != value.sha256:
        raise CanaryPromotionError(
            f"{owner} hash mismatch for {path}: {actual} != {value.sha256}"
        )
    return path


def _evidence_input_path(raw_path: str, *, evidence_root: Path) -> Path:
    path = Path(raw_path)
    return path.absolute() if path.is_absolute() else (evidence_root / path).absolute()


def _verify_evidence_inputs(
    evidence: Mapping[str, Any],
    *,
    evidence_root: Path,
    candidate_manifest: Path,
    request_path: Path,
) -> None:
    inputs = evidence.get("inputs")
    if not isinstance(inputs, Mapping):
        raise CanaryPromotionError("capture evidence lacks hash-bound inputs")
    required = {
        "animal_asset_package",
        "m2_capture_request",
        "m1_room_manifest",
        "m1_camera_request",
    }
    if set(inputs) != required:
        raise CanaryPromotionError("capture evidence input set is incomplete")
    resolved: dict[str, Path] = {}
    for name in sorted(required):
        record = inputs[name]
        if not isinstance(record, Mapping):
            raise CanaryPromotionError(f"evidence input {name} is not an object")
        raw_path = record.get("path")
        expected_hash = record.get("sha256")
        if not isinstance(raw_path, str) or not _is_sha256(expected_hash):
            raise CanaryPromotionError(f"evidence input {name} lacks path/hash")
        path = _regular_file(
            _evidence_input_path(raw_path, evidence_root=evidence_root),
            owner=f"evidence input {name}",
        )
        if sha256_file(path) != expected_hash:
            raise CanaryPromotionError(f"evidence input hash mismatch: {name}")
        resolved[name] = path
    if resolved["animal_asset_package"] != candidate_manifest.absolute():
        raise CanaryPromotionError("evidence binds a different candidate path")
    if resolved["m2_capture_request"] != request_path.absolute():
        raise CanaryPromotionError("evidence binds a different review request path")


def _validate_evidence(
    evidence_path: Path,
    *,
    candidate: Mapping[str, Any],
    candidate_manifest: Path,
    request: Mapping[str, Any],
    request_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    evidence = _load_object(evidence_path, owner="75-state capture evidence")
    evidence_root = evidence_path.parent.absolute()
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise CanaryPromotionError("capture evidence schema differs from M2")
    declared_content_hash = evidence.get("evidence_content_sha256")
    without_content_hash = dict(evidence)
    without_content_hash.pop("evidence_content_sha256", None)
    if (
        not _is_sha256(declared_content_hash)
        or canonical_json_sha256(without_content_hash) != declared_content_hash
    ):
        raise CanaryPromotionError("capture evidence content hash differs")
    if (
        evidence.get("status") != "review_only"
        or evidence.get("review_only") is not True
        or evidence.get("qualification_claim") is not False
        or evidence.get("asset_admission_state") != "research_candidate"
        or evidence.get("formal_view_ids") != []
        or evidence.get("formal_modalities") != []
        or evidence.get("review_view_ids") != ["view0"]
        or evidence.get("review_modalities") != list(FORMAL_MODALITIES)
    ):
        raise CanaryPromotionError("capture evidence is not the bounded review run")
    if (
        evidence.get("asset_id") != candidate.get("asset_id")
        or evidence.get("request_id") != request.get("request_id")
        or evidence.get("room_id") != request.get("room_id")
    ):
        raise CanaryPromotionError("capture evidence identity differs from inputs")

    _verify_evidence_inputs(
        evidence,
        evidence_root=evidence_root,
        candidate_manifest=candidate_manifest,
        request_path=request_path,
    )
    frames = evidence.get("frames")
    states = request.get("states")
    if (
        not isinstance(frames, list)
        or not isinstance(states, list)
        or len(frames) != 75
    ):
        raise CanaryPromotionError("capture evidence must contain exactly 75 frames")
    for index, (frame, state) in enumerate(zip(frames, states, strict=True)):
        if not isinstance(frame, Mapping) or not isinstance(state, Mapping):
            raise CanaryPromotionError(f"capture frame {index} is not an object")
        hashes = frame.get("hashes")
        visibility = frame.get("animal_semantic_visibility")
        modalities = frame.get("modalities")
        if (
            frame.get("frame_index") != index
            or frame.get("pts_ticks") != state.get("pts_ticks")
            or frame.get("action_id") != state.get("action_id")
            or frame.get("action_time_ticks") != state.get("action_time_ticks")
            or not isinstance(hashes, Mapping)
            or hashes.get("declared_pose_hash") != state.get("pose_hash")
            or hashes.get("recomputed_pose_hash") != state.get("pose_hash")
            or hashes.get("declared_applied_state_hash")
            != state.get("applied_state_hash")
            or hashes.get("recomputed_applied_state_hash")
            != state.get("applied_state_hash")
            or not isinstance(visibility, Mapping)
            or visibility.get("visible") is not True
            or not isinstance(visibility.get("pixel_count"), int)
            or int(visibility["pixel_count"]) <= 0
            or not isinstance(modalities, Mapping)
            or set(modalities) != set(FORMAL_MODALITIES)
        ):
            raise CanaryPromotionError(f"capture frame {index} failed review closure")

    runtime = evidence.get("runtime_application")
    if not isinstance(runtime, Mapping) or runtime.get(
        "initial_world_time_seconds"
    ) != runtime.get("final_world_time_seconds"):
        raise CanaryPromotionError("capture advanced world time during review")
    array_errors = verify_saved_capture_arrays(evidence, evidence_root)
    if array_errors:
        raise CanaryPromotionError(
            "capture arrays failed hash closure: " + "; ".join(array_errors)
        )

    review_media = evidence.get("review_media")
    videos = review_media.get("videos") if isinstance(review_media, Mapping) else None
    if not isinstance(videos, Mapping) or set(videos) != set(FORMAL_MODALITIES):
        raise CanaryPromotionError("capture evidence lacks three review videos")
    media_paths: dict[str, Path] = {}
    for modality in FORMAL_MODALITIES:
        video = videos[modality]
        artifact = video.get("artifact") if isinstance(video, Mapping) else None
        if (
            not isinstance(video, Mapping)
            or video.get("frame_count") != 75
            or video.get("frame_rate_hz") != 15
            or video.get("view_id") != "view0"
            or video.get("review_only") is not True
            or video.get("qualification_claim") is not False
            or not isinstance(artifact, Mapping)
            or not isinstance(artifact.get("path"), str)
            or not _is_sha256(artifact.get("sha256"))
            or not isinstance(artifact.get("byte_size"), int)
        ):
            raise CanaryPromotionError(f"invalid {modality} review video record")
        media_path = _regular_file(
            evidence_root / str(artifact["path"]),
            owner=f"{modality} review video",
        )
        if (
            media_path.stat().st_size != artifact["byte_size"]
            or sha256_file(media_path) != artifact["sha256"]
        ):
            raise CanaryPromotionError(f"{modality} review video hash differs")
        media_paths[modality] = media_path
    return evidence, media_paths


def _validate_local_license(rocketbox_root: Path) -> tuple[Path, Path]:
    root = _safe_tree(rocketbox_root, owner="Rocketbox checkout")
    license_path = _regular_file(root / "LICENSE.md", owner="Rocketbox LICENSE.md")
    readme_path = _regular_file(root / "README.md", owner="Rocketbox README.md")
    if sha256_file(license_path) != ROCKETBOX_LICENSE_SHA256:
        raise CanaryPromotionError("local Rocketbox LICENSE.md hash differs")
    if sha256_file(readme_path) != ROCKETBOX_README_SHA256:
        raise CanaryPromotionError("local Rocketbox README.md hash differs")
    return license_path, readme_path


def _validate_world_contact_audit(
    audit_path: Path,
    *,
    candidate: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    audit = _load_object(audit_path, owner="world-contact audit")
    gate = audit.get("gate")
    if (
        audit.get("schema") != "avengine_m2_world_contact_audit_v1"
        or audit.get("status") != "pass"
        or audit.get("qualification_claim") is not False
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
    ):
        raise CanaryPromotionError("world-contact audit is not a non-claiming pass")
    maximum = gate.get("maximum_contact_horizontal_step_m")
    measured = gate.get("measured_maximum_contact_horizontal_step_m")
    if (
        not isinstance(maximum, (int, float))
        or isinstance(maximum, bool)
        or not isinstance(measured, (int, float))
        or isinstance(measured, bool)
        or float(measured) > float(maximum)
    ):
        raise CanaryPromotionError("world-contact audit exceeds its declared gate")

    records = _records_by_role(candidate)
    expected_hashes = {
        "source_glb_sha256": records["visual"]["sha256"],
        "baked_actions_sha256": records["walk_poses"]["sha256"],
        "contact_phases_sha256": records["contact_phases"]["sha256"],
    }
    for field, expected in expected_hashes.items():
        if audit.get(field) != expected:
            raise CanaryPromotionError(
                f"world-contact audit {field} differs from candidate"
            )
    walk_states = [
        state
        for state in request.get("states", [])
        if isinstance(state, Mapping) and state.get("action_id") == "walk"
    ]
    trajectory = audit.get("trajectory")
    if (
        len(walk_states) != 45
        or not isinstance(trajectory, Mapping)
        or trajectory.get("walk_frame_count") != 45
        or trajectory.get("sample_rate_hz") != 15
        or trajectory.get("start_translation_m")
        != walk_states[0].get("root_transform", {}).get("translation_m")
        or trajectory.get("end_translation_m")
        != walk_states[-1].get("root_transform", {}).get("translation_m")
    ):
        raise CanaryPromotionError(
            "world-contact trajectory differs from the 45-state walk request"
        )
    return audit


def _copy_diagnostics(
    diagnostics: Sequence[ExpectedArtifact], destination: Path
) -> list[tuple[str, Path]]:
    if not diagnostics:
        raise CanaryPromotionError(
            "at least one hash-pinned diagnostic video is required"
        )
    destination.mkdir(parents=True, exist_ok=False)
    copied: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for index, diagnostic in enumerate(diagnostics):
        source = _validate_expected_artifact(diagnostic)
        label = diagnostic.label or source.name
        if label in labels:
            raise CanaryPromotionError(f"diagnostic label is duplicated: {label}")
        labels.add(label)
        target = destination / f"{index:02d}_{source.name}"
        shutil.copy2(source, target)
        if sha256_file(target) != diagnostic.sha256:
            raise CanaryPromotionError(f"diagnostic copy hash differs: {label}")
        copied.append((label, target))
    return copied


def promote_research_candidate(
    *,
    candidate_manifest: str | Path,
    review_request: str | Path,
    capture_evidence: str | Path,
    world_contact_audit: ExpectedArtifact,
    diagnostic_videos: Sequence[ExpectedArtifact],
    rocketbox_root: str | Path,
    output_directory: str | Path,
    reviewer_id: str = "workspace_user",
) -> PromotionResult:
    """Copy and promote one immutable research candidate without overwriting."""

    candidate_path = _regular_file(candidate_manifest, owner="candidate asset manifest")
    request_path = _regular_file(review_request, owner="75-state review request")
    evidence_path = _regular_file(capture_evidence, owner="capture evidence")
    candidate_root = _safe_tree(candidate_path.parent, owner="candidate package")
    if candidate_path.parent != candidate_root:
        raise CanaryPromotionError("candidate manifest must be at the package root")
    output = Path(output_directory).absolute()
    if output.exists():
        raise CanaryPromotionError(f"refusing to overwrite promotion output: {output}")
    if not reviewer_id.strip():
        raise CanaryPromotionError("reviewer_id must be non-empty")

    candidate = _validate_candidate(candidate_path)
    try:
        inputs = load_research_review_inputs(candidate_path, request_path)
    except Exception as error:
        raise CanaryPromotionError(
            f"review request does not close over the candidate: {error}"
        ) from error
    request = inputs.request
    _validate_evidence(
        evidence_path,
        candidate=candidate,
        candidate_manifest=candidate_path,
        request=request,
        request_path=request_path,
    )
    world_contact_path = _validate_expected_artifact(
        world_contact_audit, owner="world-contact audit"
    )
    _validate_world_contact_audit(
        world_contact_path,
        candidate=candidate,
        request=request,
    )
    license_path, readme_path = _validate_local_license(Path(rocketbox_root))
    validated_diagnostics = [
        ExpectedArtifact(
            path=_validate_expected_artifact(value),
            sha256=value.sha256,
            label=value.label,
        )
        for value in diagnostic_videos
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid4().hex}"
    try:
        shutil.copytree(candidate_root, staging, symlinks=False)
        admission = staging / "admission"
        admission.mkdir()
        candidate_snapshot = admission / "candidate_asset_manifest.json"
        shutil.copy2(candidate_path, candidate_snapshot)
        request_snapshot = admission / "capture_request.json"
        shutil.copy2(request_path, request_snapshot)
        world_contact_snapshot = admission / "world_contact_audit.json"
        shutil.copy2(world_contact_path, world_contact_snapshot)
        if sha256_file(world_contact_snapshot) != world_contact_audit.sha256:
            raise CanaryPromotionError("world-contact audit copy hash differs")
        evidence_snapshot_root = admission / "capture_evidence"
        shutil.copytree(evidence_path.parent, evidence_snapshot_root, symlinks=False)
        evidence_snapshot = evidence_snapshot_root / evidence_path.name
        source_root = admission / "source"
        source_root.mkdir()
        copied_license = source_root / "LICENSE.md"
        copied_readme = source_root / "README.md"
        shutil.copy2(license_path, copied_license)
        shutil.copy2(readme_path, copied_readme)
        copied_diagnostics = _copy_diagnostics(
            validated_diagnostics,
            admission / "diagnostics",
        )

        copied_evidence = _load_object(
            evidence_snapshot, owner="copied capture evidence"
        )
        copied_media: dict[str, Path] = {}
        videos = copied_evidence["review_media"]["videos"]
        for modality in FORMAL_MODALITIES:
            copied_media[modality] = _regular_file(
                evidence_snapshot_root / videos[modality]["artifact"]["path"],
                owner=f"copied {modality} review video",
            )

        human_review = {
            "schema": "avengine_m2_human_visual_review_v1",
            "status": "pass",
            "scope": "m2_canary_admission",
            "qualification_claim": False,
            "formal_dataset_registration_authorized": False,
            "candidate": {
                "asset_id": candidate["asset_id"],
                "admission_state": "research_candidate",
                "asset_manifest": _artifact(candidate_snapshot, relative_to=admission),
            },
            "capture": {
                "request_id": request["request_id"],
                "state_count": 75,
                "review_only": True,
                "asset_manifest_sha256": sha256_file(candidate_snapshot),
                "request": _artifact(request_snapshot, relative_to=admission),
                "evidence": _artifact(evidence_snapshot, relative_to=admission),
            },
            "world_contact_audit": _artifact(
                world_contact_snapshot, relative_to=admission
            ),
            "review_media": {
                modality: _artifact(path, relative_to=admission)
                for modality, path in copied_media.items()
            },
            "diagnostic_media": [
                {
                    "label": label,
                    "artifact": _artifact(path, relative_to=admission),
                }
                for label, path in copied_diagnostics
            ],
            "source_license": {
                "source_repository": ROCKETBOX_REPOSITORY,
                "source_revision": ROCKETBOX_REVISION,
                "license": "MIT",
                "allowed_use": "research_canary",
                "redistribution": "allowed",
                "license_file": _artifact(copied_license, relative_to=admission),
                "readme_file": _artifact(copied_readme, relative_to=admission),
            },
            "reviewer_decision": {
                "reviewer_id": reviewer_id,
                "decision_date": USER_DECISION_DATE,
                "statement": USER_DECISION_STATEMENT,
                "overall_canary_visual_acceptance": "pass",
                "rear_leg_motion_naturalness": "pass",
            },
            "decision_reason": (
                "The user explicitly accepted the hash-bound rear-leg motion and "
                "authorized completion of the remaining M2 canary admission work."
            ),
        }
        review_path = admission / "human_visual_review.json"
        write_json(review_path, human_review)
        review_errors = validate_human_visual_review(
            human_review,
            review_path=review_path,
            expected_asset_id=str(candidate["asset_id"]),
        )
        if review_errors:
            raise CanaryPromotionError(
                "generated human review is invalid: " + "; ".join(review_errors)
            )

        old_provenance_record = _records_by_role(candidate)["provenance_manifest"]
        provenance = {
            "schema": "avengine_m2_canary_provenance_v1",
            "admission_state": "canary_qualified",
            "formal_dataset_registration_authorized": False,
            "candidate_asset_manifest": _artifact(
                candidate_snapshot, relative_to=admission
            ),
            "candidate_provenance_manifest": {
                "path": str(old_provenance_record["path"]),
                "byte_size": int(old_provenance_record["byte_size"]),
                "sha256": str(old_provenance_record["sha256"]),
            },
            "human_visual_review": _artifact(review_path, relative_to=admission),
            "world_contact_audit": _artifact(
                world_contact_snapshot, relative_to=admission
            ),
            "source_repository": {
                "url": ROCKETBOX_REPOSITORY,
                "revision": ROCKETBOX_REVISION,
            },
            "license_decision": {
                "license": "MIT",
                "allowed_use": "research_canary",
                "redistribution": "allowed",
                "license_file": _artifact(copied_license, relative_to=admission),
                "readme_file": _artifact(copied_readme, relative_to=admission),
                "decision_reason": (
                    "The pinned local Microsoft Rocketbox LICENSE.md grants MIT "
                    "use and redistribution; its notice is copied into this package."
                ),
            },
            "notes": [
                "Canary admission does not authorize M6 dataset registration.",
                "This provenance binds only the research-candidate manifest; it "
                "does not bind the final asset manifest and cannot form a hash cycle.",
            ],
        }
        provenance_path = admission / "canary_provenance.json"
        write_json(provenance_path, provenance)

        final_manifest = dict(candidate)
        final_manifest["admission_state"] = "canary_qualified"
        final_files: list[dict[str, Any]] = []
        for record in candidate["files"]:
            if record["role"] == "provenance_manifest":
                final_files.append(
                    _package_file_record(
                        provenance_path,
                        role="provenance_manifest",
                        package_root=staging,
                    )
                )
            else:
                final_files.append(dict(record))
        final_files.append(
            _package_file_record(
                review_path,
                role="human_visual_review",
                package_root=staging,
            )
        )
        final_manifest["files"] = final_files
        final_manifest["qualification"] = {
            "automatic_qa_status": "pass",
            "human_visual_review_status": "pass",
            "human_review_binding_sha256": sha256_file(review_path),
            "decision_reason": (
                "Hash-closed automatic QA and the 2026-07-17 user visual review "
                "both passed for bounded M2 research-canary use."
            ),
        }
        final_provenance = dict(candidate["provenance"])
        final_provenance.update(
            {
                "license": "MIT",
                "allowed_use": "research_canary",
                "redistribution": "allowed",
            }
        )
        final_manifest["provenance"] = final_provenance
        final_manifest_path = staging / "asset_manifest.json"
        write_json(final_manifest_path, final_manifest)
        final_errors = validate_animal_asset_package(
            final_manifest,
            manifest_path=final_manifest_path,
        )
        if final_errors:
            raise CanaryPromotionError(
                "generated canary package is invalid: " + "; ".join(final_errors)
            )
        if output.exists():
            raise CanaryPromotionError(
                f"refusing to overwrite promotion output: {output}"
            )
        staging.rename(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    manifest_path = output / "asset_manifest.json"
    final_value = load_json(manifest_path)
    final_errors = validate_animal_asset_package(
        final_value,
        manifest_path=manifest_path,
    )
    if final_errors:
        raise CanaryPromotionError(
            "installed canary package is invalid: " + "; ".join(final_errors)
        )
    human_review_path = output / "admission" / "human_visual_review.json"
    provenance_path = output / "admission" / "canary_provenance.json"
    return PromotionResult(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        human_review_path=human_review_path,
        human_review_sha256=sha256_file(human_review_path),
        provenance_path=provenance_path,
        provenance_sha256=sha256_file(provenance_path),
    )


__all__ = [
    "CanaryPromotionError",
    "ExpectedArtifact",
    "PromotionResult",
    "ROCKETBOX_LICENSE_SHA256",
    "ROCKETBOX_README_SHA256",
    "ROCKETBOX_REPOSITORY",
    "ROCKETBOX_REVISION",
    "USER_DECISION_DATE",
    "USER_DECISION_STATEMENT",
    "promote_research_candidate",
]
