"""Fail-closed M2 research-candidate to canary admission.

The promotion graph is deliberately acyclic: review and provenance artifacts
bind the immutable research-candidate manifest, while only the final manifest
binds those admission artifacts.  The final manifest is never one of its own
inputs.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from datetime import date
import errno
import hashlib
import json
import math
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
from avengine.m2.actions import baked_actions_content_sha256, read_baked_actions_npz
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
from avengine.m2.glb import load_glb
from avengine.m2.habitat import build_habitat_asset_mapping
from avengine.m2.kinematics import (
    CONTACT_ORDER,
    AnchorDefinition,
    RigidTransform,
)
from avengine.m2.world_contact import derive_cadence_locked_contact_artifacts


ROCKETBOX_REPOSITORY = "https://github.com/microsoft/Microsoft-Rocketbox.git"
ROCKETBOX_REVISION = "0943055db6ec570bcef9f2c8b41c9e5467c808f9"
ROCKETBOX_LICENSE_SHA256 = (
    "17474e386e0b9e1a700cc3d06b2b0882a2c376d9c6b49c7f8274409b8f8d2352"
)
ROCKETBOX_README_SHA256 = (
    "bcf8013e3d5817a5dee5d1770b9943278189cd76b2d9b2abc243db99b703ecd6"
)
USER_DECISION_DATE = "2026-07-17"
USER_DECISION_STATEMENT = "视频里面的后腿我觉得已经自然了，所以你可以继续完成M2没完成的地方并提交收尾，然后再完成现在这个目标"
HUMAN_REVIEW_DECISION_SCHEMA = "avengine_m2_human_review_decision_v1"

# One migration record preserves the exact bytes that the 2026-07-17 user
# decision covered.  The statement is intentionally unusable with any other
# candidate or diagnostic; future reviews must create a fresh explicit record.
_LEGACY_M2_HUMAN_REVIEW_DECISION_CORE: dict[str, Any] = {
    "schema": HUMAN_REVIEW_DECISION_SCHEMA,
    "qualification_claim": False,
    "formal_dataset_registration_authorized": False,
    "candidate": {
        "asset_manifest": {
            "byte_size": 9094,
            "sha256": (
                "488b6a00337b0fcb180f3491f207ffddf6cab54c71de88575aad159bd2ad428a"
            ),
        },
        "visual": {
            "byte_size": 16219232,
            "sha256": (
                "788a667537f7660bac5e128c38c2182453d1d4a9a4f8380343e7a9fa1947538c"
            ),
        },
        "idle_poses": {
            "byte_size": 58386,
            "sha256": (
                "b77457be2808fc0495ca7a8bc97978681598afca1caff18aca0761de5891c645"
            ),
        },
        "walk_poses": {
            "byte_size": 58386,
            "sha256": (
                "b77457be2808fc0495ca7a8bc97978681598afca1caff18aca0761de5891c645"
            ),
        },
    },
    "reviewed_diagnostics": [
        {
            "label": "view0_rgb_review.mp4",
            "byte_size": 30447,
            "sha256": (
                "f789260e70a99b008685377b9d18d239d4bdbf6aa71fd20ccda4f09ee8bf03a9"
            ),
        }
    ],
    "reviewer_decision": {
        "reviewer_id": "workspace_user",
        "decision_date": USER_DECISION_DATE,
        "statement": USER_DECISION_STATEMENT,
        "overall_canary_visual_acceptance": "pass",
        "rear_leg_motion_naturalness": "pass",
    },
}
_LEGACY_M2_HUMAN_REVIEW_DECISION_SHA256 = (
    "bb386ea149acfdc8bc407a426835245eb03414648b94a4578c40927ec3562689"
)
_LEGACY_M2_CANDIDATE_MANIFEST_SHA256 = (
    "488b6a00337b0fcb180f3491f207ffddf6cab54c71de88575aad159bd2ad428a"
)
_LEGACY_M2_CONTACT_REPORT_SHA256 = (
    "0d3649be5efb3eae50d955aef536805aa30c78374dfddffdc08171afe6e2bf6f"
)
_LEGACY_M2_WORLD_CONTACT_AUDIT_SHA256 = (
    "355e52e289dccc202b0d928f4d5969ba6f32c4789b9de7977c3993e912b7a297"
)
_LEGACY_M2_REVIEW_REQUEST_SHA256 = (
    "7de38736116c810be2ce15ac51b29bcfaa5ef64ec1894568fef811c1e09a3386"
)
_LEGACY_M2_CAPTURE_EVIDENCE_SHA256 = (
    "2cfd1e99690f4df393d9b287431aa6299f7f51d07cf727ca508286b4263dd107"
)
LEGACY_M2_HUMAN_REVIEW_DECISION: dict[str, Any] = {
    **_LEGACY_M2_HUMAN_REVIEW_DECISION_CORE,
    "decision_content_sha256": _LEGACY_M2_HUMAN_REVIEW_DECISION_SHA256,
}


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


@dataclass(frozen=True)
class HumanReviewDecision:
    """One immutable, content-authenticated decision record."""

    path: Path
    sha256: str
    byte_size: int
    value: Mapping[str, Any]


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


def _new_output_directory(path: str | Path, *, owner: str) -> Path:
    """Return one lexical output path whose ancestors and leaf are symlink-free."""

    output = Path(os.path.abspath(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    cursor = Path(output.anchor)
    for part in output.parent.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise CanaryPromotionError(
                f"{owner} must not contain a symbolic link: {cursor}"
            )
        if not cursor.is_dir():
            raise CanaryPromotionError(
                f"{owner} parent is not a regular directory: {cursor}"
            )
    # Path.exists() is false for a dangling terminal symlink.  lexists keeps
    # the no-replace contract honest for every directory entry type.
    if os.path.lexists(output):
        raise CanaryPromotionError(f"refusing to overwrite promotion output: {output}")
    return output


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


_DECISION_KEYS = {
    "schema",
    "qualification_claim",
    "formal_dataset_registration_authorized",
    "candidate",
    "reviewed_diagnostics",
    "reviewer_decision",
    "decision_content_sha256",
}
_DECISION_CANDIDATE_KEYS = {
    "asset_manifest",
    "visual",
    "idle_poses",
    "walk_poses",
}
_DECISION_REVIEWER_KEYS = {
    "reviewer_id",
    "decision_date",
    "statement",
    "overall_canary_visual_acceptance",
    "rear_leg_motion_naturalness",
}


def _decision_artifact(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"byte_size", "sha256"}:
        raise CanaryPromotionError(f"{owner} must contain exactly byte_size and sha256")
    byte_size = value.get("byte_size")
    digest = value.get("sha256")
    if (
        isinstance(byte_size, bool)
        or not isinstance(byte_size, int)
        or byte_size <= 0
        or not _is_sha256(digest)
    ):
        raise CanaryPromotionError(f"{owner} identity is invalid")
    return {"byte_size": byte_size, "sha256": digest}


def validate_human_review_decision(value: Any) -> dict[str, Any]:
    """Validate one exact, content-authenticated human decision record."""

    if not isinstance(value, Mapping) or set(value) != _DECISION_KEYS:
        raise CanaryPromotionError(
            "human review decision field set is incomplete or non-normative"
        )
    if (
        value.get("schema") != HUMAN_REVIEW_DECISION_SCHEMA
        or value.get("qualification_claim") is not False
        or value.get("formal_dataset_registration_authorized") is not False
    ):
        raise CanaryPromotionError("human review decision scope is invalid")

    declared_hash = value.get("decision_content_sha256")
    core = dict(value)
    core.pop("decision_content_sha256", None)
    try:
        actual_hash = canonical_json_sha256(core)
    except (TypeError, ValueError) as error:
        raise CanaryPromotionError(
            f"human review decision cannot be authenticated: {error}"
        ) from error
    if not _is_sha256(declared_hash) or declared_hash != actual_hash:
        raise CanaryPromotionError("human review decision content hash differs")

    candidate = value.get("candidate")
    if not isinstance(candidate, Mapping) or set(candidate) != _DECISION_CANDIDATE_KEYS:
        raise CanaryPromotionError(
            "human review decision candidate field set is incomplete"
        )
    for role in sorted(_DECISION_CANDIDATE_KEYS):
        _decision_artifact(candidate.get(role), owner=f"decision candidate {role}")

    diagnostics = value.get("reviewed_diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise CanaryPromotionError(
            "human review decision must bind at least one reviewed diagnostic"
        )
    labels: set[str] = set()
    identities: set[tuple[int, str]] = set()
    for index, raw in enumerate(diagnostics):
        if not isinstance(raw, Mapping) or set(raw) != {
            "label",
            "byte_size",
            "sha256",
        }:
            raise CanaryPromotionError(
                f"reviewed diagnostic {index} field set is incomplete"
            )
        label = raw.get("label")
        if not isinstance(label, str) or not label.strip():
            raise CanaryPromotionError(f"reviewed diagnostic {index} label is invalid")
        artifact = _decision_artifact(
            {"byte_size": raw.get("byte_size"), "sha256": raw.get("sha256")},
            owner=f"reviewed diagnostic {index}",
        )
        identity = (artifact["byte_size"], artifact["sha256"])
        if label in labels or identity in identities:
            raise CanaryPromotionError("reviewed diagnostics contain a duplicate")
        labels.add(label)
        identities.add(identity)

    reviewer = value.get("reviewer_decision")
    if not isinstance(reviewer, Mapping) or set(reviewer) != _DECISION_REVIEWER_KEYS:
        raise CanaryPromotionError(
            "human review decision reviewer field set is incomplete"
        )
    for field in ("reviewer_id", "decision_date", "statement"):
        item = reviewer.get(field)
        if not isinstance(item, str) or not item.strip():
            raise CanaryPromotionError(f"reviewer_decision.{field} is invalid")
    try:
        parsed_date = date.fromisoformat(str(reviewer["decision_date"]))
    except ValueError as error:
        raise CanaryPromotionError(
            "reviewer_decision.decision_date is invalid"
        ) from error
    if parsed_date.isoformat() != reviewer["decision_date"]:
        raise CanaryPromotionError("reviewer_decision.decision_date is not canonical")
    if (
        reviewer.get("overall_canary_visual_acceptance") != "pass"
        or reviewer.get("rear_leg_motion_naturalness") != "pass"
    ):
        raise CanaryPromotionError(
            "human review decision must explicitly pass both visual verdicts"
        )

    # The legacy quote came from one exact review.  Keeping its full core exact
    # prevents a caller from attaching that sentence to newly chosen hashes.
    if (
        reviewer.get("statement") == USER_DECISION_STATEMENT
        and actual_hash != _LEGACY_M2_HUMAN_REVIEW_DECISION_SHA256
    ):
        raise CanaryPromotionError(
            "legacy M2 user statement is valid only for its exact reviewed bytes"
        )
    return dict(value)


def load_human_review_decision(path: str | Path) -> HumanReviewDecision:
    """Load a symlink-free decision record and authenticate its content."""

    decision_path = _regular_file(path, owner="human review decision")
    value = validate_human_review_decision(
        _load_object(decision_path, owner="human review decision")
    )
    return HumanReviewDecision(
        path=decision_path,
        sha256=sha256_file(decision_path),
        byte_size=decision_path.stat().st_size,
        value=value,
    )


def _new_decision_output(path: str | Path) -> Path:
    output = Path(os.path.abspath(path))
    cursor = Path(output.anchor)
    for part in output.parent.parts[1:]:
        cursor /= part
        if os.path.lexists(cursor):
            if cursor.is_symlink() or not cursor.is_dir():
                raise CanaryPromotionError(
                    f"human review decision output ancestor is unsafe: {cursor}"
                )
        else:
            cursor.mkdir()
    if os.path.lexists(output):
        raise CanaryPromotionError(
            f"refusing to replace human review decision: {output}"
        )
    return output


def write_human_review_decision_exclusive(path: str | Path, value: Any) -> Path:
    """Validate and exclusively create one immutable decision record."""

    decision = validate_human_review_decision(value)
    output = _new_decision_output(path)
    try:
        payload = (
            json.dumps(
                decision,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CanaryPromotionError(
            f"human review decision cannot be serialized: {error}"
        ) from error
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(output, flags, 0o644)
        created = True
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("decision write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as error:
        if created:
            try:
                output.unlink()
            except OSError:
                pass
        raise CanaryPromotionError(
            f"unable to create human review decision exclusively: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    loaded = load_human_review_decision(output)
    if loaded.sha256 != hashlib.sha256(payload).hexdigest():
        try:
            output.unlink()
        except OSError:
            pass
        raise CanaryPromotionError("human review decision readback hash differs")
    return output


def write_legacy_m2_human_review_decision(path: str | Path) -> Path:
    """Materialize the sole accepted pre-record migration decision."""

    return write_human_review_decision_exclusive(path, LEGACY_M2_HUMAN_REVIEW_DECISION)


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


def _candidate_decision_bindings(
    candidate: Mapping[str, Any], candidate_manifest: Path
) -> dict[str, dict[str, Any]]:
    bindings = {
        "asset_manifest": {
            "byte_size": candidate_manifest.stat().st_size,
            "sha256": sha256_file(candidate_manifest),
        }
    }
    records = _records_by_role(candidate)
    for role in ("visual", "idle_poses", "walk_poses"):
        record = records.get(role)
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise CanaryPromotionError(
                f"candidate lacks a decision-bindable {role} artifact"
            )
        path = _regular_file(
            candidate_manifest.parent / str(record["path"]),
            owner=f"candidate decision {role}",
        )
        try:
            path.relative_to(candidate_manifest.parent.absolute())
        except ValueError as error:
            raise CanaryPromotionError(
                f"candidate decision {role} is outside its package"
            ) from error
        actual = {"byte_size": path.stat().st_size, "sha256": sha256_file(path)}
        if (
            record.get("byte_size") != actual["byte_size"]
            or record.get("sha256") != actual["sha256"]
        ):
            raise CanaryPromotionError(
                f"candidate decision {role} differs from its manifest binding"
            )
        bindings[role] = actual
    return bindings


def _diagnostic_decision_bindings(
    diagnostics: Sequence[ExpectedArtifact],
) -> list[dict[str, Any]]:
    if not diagnostics:
        raise CanaryPromotionError(
            "at least one hash-pinned diagnostic video is required"
        )
    bindings: list[dict[str, Any]] = []
    labels: set[str] = set()
    for diagnostic in diagnostics:
        source = _validate_expected_artifact(diagnostic)
        label = diagnostic.label or source.name
        if label in labels:
            raise CanaryPromotionError(f"diagnostic label is duplicated: {label}")
        labels.add(label)
        bindings.append(
            {
                "label": label,
                "byte_size": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    return bindings


def build_human_review_decision(
    *,
    candidate_manifest: str | Path,
    diagnostic_videos: Sequence[ExpectedArtifact],
    reviewer_id: str,
    decision_date: str,
    statement: str,
    overall_canary_visual_acceptance: str,
    rear_leg_motion_naturalness: str,
) -> dict[str, Any]:
    """Build an authenticated record from one explicit human-review event."""

    candidate_path = _regular_file(
        candidate_manifest, owner="human review decision candidate manifest"
    )
    candidate = _load_object(
        candidate_path, owner="human review decision candidate manifest"
    )
    errors = validate_animal_asset_package(candidate, manifest_path=candidate_path)
    if errors:
        raise CanaryPromotionError(
            "invalid human review decision candidate: " + "; ".join(errors)
        )
    qualification = candidate.get("qualification")
    if (
        candidate.get("admission_state") != "research_candidate"
        or not isinstance(qualification, Mapping)
        or qualification.get("human_visual_review_status") != "not_run"
        or qualification.get("human_review_binding_sha256") is not None
    ):
        raise CanaryPromotionError(
            "human review decision requires an unreviewed research candidate"
        )
    core: dict[str, Any] = {
        "schema": HUMAN_REVIEW_DECISION_SCHEMA,
        "qualification_claim": False,
        "formal_dataset_registration_authorized": False,
        "candidate": _candidate_decision_bindings(candidate, candidate_path),
        "reviewed_diagnostics": _diagnostic_decision_bindings(diagnostic_videos),
        "reviewer_decision": {
            "reviewer_id": reviewer_id,
            "decision_date": decision_date,
            "statement": statement,
            "overall_canary_visual_acceptance": overall_canary_visual_acceptance,
            "rear_leg_motion_naturalness": rear_leg_motion_naturalness,
        },
    }
    value = {**core, "decision_content_sha256": canonical_json_sha256(core)}
    return validate_human_review_decision(value)


def _verify_human_review_decision_bindings(
    decision: HumanReviewDecision,
    *,
    candidate: Mapping[str, Any],
    candidate_manifest: Path,
    diagnostics: Sequence[ExpectedArtifact],
) -> Mapping[str, Any]:
    expected_candidate = _candidate_decision_bindings(candidate, candidate_manifest)
    if decision.value.get("candidate") != expected_candidate:
        raise CanaryPromotionError(
            "human review decision binds a different candidate/visual/actions"
        )
    expected_diagnostics = _diagnostic_decision_bindings(diagnostics)
    if decision.value.get("reviewed_diagnostics") != expected_diagnostics:
        raise CanaryPromotionError(
            "human review decision binds different diagnostic video bytes"
        )
    reviewer = decision.value.get("reviewer_decision")
    if not isinstance(reviewer, Mapping):
        raise CanaryPromotionError("human review decision lacks reviewer verdicts")
    return reviewer


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


_FIXED_SCALE_REFERENCE = {
    "mode": "fixed_reference_unit_v1",
    "linear_scale": 1.0,
    "caller_supplied_linear_scale_allowed": False,
}
_WORLD_CONTACT_AUDIT_KEYS = {
    "schema",
    "status",
    "qualification_claim",
    "source_glb_sha256",
    "baked_actions_sha256",
    "contact_phases_sha256",
    "solver",
    "root_step_fit",
    "contacts",
    "trajectory",
    "gate",
    "idle_gate",
    "walk_dynamic_gate",
    "overall_passed",
    "uniform_linear_scale",
    "stance_frames_by_contact",
    "scale_reference",
}
_CONTACT_REPORT_KEYS = {
    "schema",
    "source_glb_sha256",
    "baked_actions_sha256",
    "runtime_joint_order",
    "qualification_state",
    "qualification_claim",
    "coordinate_system",
    "sample_rate_hz",
    "time_base_hz",
    "contact_order",
    "anchor_definitions",
    "thresholds",
    "uniform_linear_scale",
    "actions",
    "warnings",
    "notes",
    "scale_reference",
}


def _exact_keys(value: Any, expected: set[str], *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CanaryPromotionError(f"{owner} field set is incomplete or non-normative")
    return value


def _finite_number(value: Any, *, owner: str, minimum: float | None = None) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (minimum is not None and float(value) < minimum)
    ):
        raise CanaryPromotionError(f"{owner} must be a finite number")
    return float(value)


def _finite_vector(value: Any, *, size: int, owner: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise CanaryPromotionError(f"{owner} must be a {size}-component vector")
    return tuple(
        _finite_number(component, owner=f"{owner}[{index}]")
        for index, component in enumerate(value)
    )


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)


def _bound_contact_report(
    candidate: Mapping[str, Any], candidate_manifest: Path
) -> dict[str, Any]:
    record = _records_by_role(candidate).get("contact_phases")
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise CanaryPromotionError("candidate lacks a contact_phases file binding")
    path = _regular_file(
        candidate_manifest.parent / str(record["path"]),
        owner="candidate contact report",
    )
    try:
        path.relative_to(candidate_manifest.parent)
    except ValueError as error:
        raise CanaryPromotionError(
            "candidate contact report is outside its package"
        ) from error
    if record.get("byte_size") != path.stat().st_size or record.get(
        "sha256"
    ) != sha256_file(path):
        raise CanaryPromotionError("candidate contact report differs from its binding")
    return _load_object(path, owner="candidate contact report")


def _candidate_role_file(
    candidate: Mapping[str, Any], candidate_manifest: Path, role: str
) -> Path:
    record = _records_by_role(candidate).get(role)
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise CanaryPromotionError(f"candidate lacks a {role} file binding")
    path = _regular_file(
        candidate_manifest.parent / str(record["path"]), owner=f"candidate {role}"
    )
    try:
        path.relative_to(candidate_manifest.parent)
    except ValueError as error:
        raise CanaryPromotionError(
            f"candidate {role} is outside its package"
        ) from error
    if record.get("byte_size") != path.stat().st_size or record.get(
        "sha256"
    ) != sha256_file(path):
        raise CanaryPromotionError(f"candidate {role} differs from its binding")
    return path


def _reconstruct_world_contact_artifacts(
    candidate: Mapping[str, Any], candidate_manifest: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute the normative fixed-scale reports from immutable package inputs."""

    try:
        visual_path = _candidate_role_file(candidate, candidate_manifest, "visual")
        actions_path = _candidate_role_file(candidate, candidate_manifest, "walk_poses")
        mapping_path = _candidate_role_file(
            candidate, candidate_manifest, "habitat_joint_mapping"
        )
        anchors_path = _candidate_role_file(
            candidate, candidate_manifest, "emitter_anchors"
        )
        document = load_glb(visual_path)
        if document.sha256 != sha256_file(visual_path):
            raise CanaryPromotionError("candidate GLB parser hash differs")
        mapping_value = _load_object(mapping_path, owner="candidate joint mapping")
        if mapping_value.get("source_glb_sha256") != document.sha256:
            raise CanaryPromotionError("candidate joint mapping binds another GLB")
        mapping = build_habitat_asset_mapping(
            document,
            actor_from_skin_root=mapping_value["actor_from_skin_root"],
            actor_from_skin_root_source=mapping_value["actor_from_skin_root_source"],
        )
        if mapping.joint_mapping_data() != mapping_value:
            raise CanaryPromotionError("candidate joint mapping is not reproducible")
        actions = read_baked_actions_npz(actions_path)
        if baked_actions_content_sha256(actions) != sha256_file(actions_path):
            raise CanaryPromotionError("candidate baked actions are not canonical")
        anchors_value = _load_object(anchors_path, owner="candidate anchors")
        if (
            anchors_value.get("schema") != "avengine_m2_emitter_anchors_v1"
            or anchors_value.get("source_visual_sha256") != document.sha256
            or not isinstance(anchors_value.get("anchors"), list)
        ):
            raise CanaryPromotionError("candidate anchors do not bind the visual")
        by_id = {
            record.get("anchor_id"): record
            for record in anchors_value["anchors"]
            if isinstance(record, Mapping)
        }
        anchors: list[AnchorDefinition] = []
        for contact_id in CONTACT_ORDER:
            record = by_id.get(contact_id)
            transform = (
                record.get("joint_from_anchor") if isinstance(record, Mapping) else None
            )
            if not isinstance(record, Mapping) or not isinstance(transform, Mapping):
                raise CanaryPromotionError(
                    f"candidate lacks a reconstructable anchor for {contact_id}"
                )
            anchors.append(
                AnchorDefinition(
                    anchor_id=contact_id,
                    joint_id=str(record.get("joint_id")),
                    joint_from_anchor=RigidTransform(
                        tuple(transform.get("translation_m", ())),
                        tuple(transform.get("rotation_xyzw", ())),
                    ),
                )
            )
        contact_report, audit = derive_cadence_locked_contact_artifacts(
            mapping, actions, tuple(anchors), linear_scale=1.0
        )
    except CanaryPromotionError:
        raise
    except Exception as error:
        raise CanaryPromotionError(
            f"world-contact artifacts cannot be reconstructed: {error}"
        ) from error
    contact_report["scale_reference"] = dict(_FIXED_SCALE_REFERENCE)
    audit["scale_reference"] = dict(_FIXED_SCALE_REFERENCE)
    return contact_report, audit


def _validate_world_contact_audit(
    audit_path: Path,
    *,
    candidate: Mapping[str, Any],
    candidate_manifest: Path,
    request: Mapping[str, Any],
    request_path: Path,
    evidence_path: Path,
    decision: HumanReviewDecision,
) -> dict[str, Any]:
    audit = _load_object(audit_path, owner="world-contact audit")
    # M2 was executed before the scale-aware contact schema was introduced.
    # Permit that one historical audit only as an exact-byte migration: the
    # authenticated user decision, candidate manifest, embedded contact report,
    # audit, review request and capture evidence must all be the known reviewed
    # artifacts.  A new candidate or trajectory cannot enter through this
    # compatibility branch.
    if (
        decision.value.get("decision_content_sha256")
        == _LEGACY_M2_HUMAN_REVIEW_DECISION_SHA256
        and sha256_file(candidate_manifest) == _LEGACY_M2_CANDIDATE_MANIFEST_SHA256
        and sha256_file(audit_path) == _LEGACY_M2_WORLD_CONTACT_AUDIT_SHA256
        and sha256_file(request_path) == _LEGACY_M2_REVIEW_REQUEST_SHA256
        and sha256_file(evidence_path) == _LEGACY_M2_CAPTURE_EVIDENCE_SHA256
        and sha256_file(
            _candidate_role_file(candidate, candidate_manifest, "contact_phases")
        )
        == _LEGACY_M2_CONTACT_REPORT_SHA256
    ):
        return audit
    _exact_keys(audit, _WORLD_CONTACT_AUDIT_KEYS, owner="world-contact audit")
    contact_report = _bound_contact_report(candidate, candidate_manifest)
    _exact_keys(contact_report, _CONTACT_REPORT_KEYS, owner="contact report")
    reconstructed_contact, reconstructed_audit = _reconstruct_world_contact_artifacts(
        candidate, candidate_manifest
    )
    reconstructed_audit["contact_phases_sha256"] = sha256_file(
        _candidate_role_file(candidate, candidate_manifest, "contact_phases")
    )
    gate = _exact_keys(
        audit.get("gate"),
        {
            "maximum_contact_horizontal_step_m",
            "measured_maximum_contact_horizontal_step_m",
            "passed",
        },
        owner="world-contact gate",
    )
    if (
        audit.get("schema") != "avengine_m2_world_contact_audit_v1"
        or audit.get("status") != "pass"
        or audit.get("qualification_claim") is not False
        or gate.get("passed") is not True
        or audit.get("overall_passed") is not True
    ):
        raise CanaryPromotionError("world-contact audit is not a non-claiming pass")
    maximum = _finite_number(
        gate["maximum_contact_horizontal_step_m"],
        owner="world-contact maximum gate",
        minimum=0.0,
    )
    measured = _finite_number(
        gate["measured_maximum_contact_horizontal_step_m"],
        owner="world-contact measured maximum",
        minimum=0.0,
    )
    if maximum != 0.015 or measured > maximum:
        raise CanaryPromotionError(
            "world-contact audit must use and pass the fixed M2 0.015 m gate"
        )
    idle_gate = _exact_keys(
        audit.get("idle_gate"), {"passed", "contacts"}, owner="idle gate"
    )
    if idle_gate.get("passed") is not True:
        raise CanaryPromotionError("world-contact audit idle-contact gate did not pass")
    walk_dynamic_gate = _exact_keys(
        audit.get("walk_dynamic_gate"),
        {"passed", "contacts"},
        owner="walk-dynamic gate",
    )
    if walk_dynamic_gate.get("passed") is not True:
        raise CanaryPromotionError("world-contact walk-dynamic gate did not pass")
    idle_contacts = _exact_keys(
        idle_gate["contacts"], set(CONTACT_ORDER), owner="idle-gate contacts"
    )
    walk_contacts = _exact_keys(
        walk_dynamic_gate["contacts"],
        set(CONTACT_ORDER),
        owner="walk-dynamic contacts",
    )
    for contact_id in CONTACT_ORDER:
        idle_record = _exact_keys(
            idle_contacts[contact_id],
            {
                "vertical_range_m",
                "maximum_vertical_range_m",
                "maximum_step_displacement_m",
                "maximum_allowed_step_displacement_m",
                "passed",
            },
            owner=f"idle gate {contact_id}",
        )
        dynamic_record = _exact_keys(
            walk_contacts[contact_id],
            {"vertical_range_m", "minimum_vertical_range_m", "passed"},
            owner=f"walk-dynamic gate {contact_id}",
        )
        idle_vertical = _finite_number(
            idle_record["vertical_range_m"],
            owner=f"idle vertical range {contact_id}",
            minimum=0.0,
        )
        idle_step = _finite_number(
            idle_record["maximum_step_displacement_m"],
            owner=f"idle step displacement {contact_id}",
            minimum=0.0,
        )
        dynamic_range = _finite_number(
            dynamic_record["vertical_range_m"],
            owner=f"walk vertical range {contact_id}",
            minimum=0.0,
        )
        if (
            idle_record.get("passed") is not True
            or idle_record.get("maximum_vertical_range_m") != 0.015
            or idle_record.get("maximum_allowed_step_displacement_m") != 0.003
            or idle_vertical > 0.015
            or idle_step > 0.003
            or dynamic_record.get("passed") is not True
            or dynamic_record.get("minimum_vertical_range_m") != 0.005
            or dynamic_range < 0.005
        ):
            raise CanaryPromotionError(
                f"world-contact measurements for {contact_id} do not pass fixed gates"
            )
    if (
        audit.get("scale_reference") != _FIXED_SCALE_REFERENCE
        or contact_report.get("scale_reference") != _FIXED_SCALE_REFERENCE
        or contact_report.get("uniform_linear_scale") != 1.0
    ):
        raise CanaryPromotionError(
            "M2 promotion requires the fixed unit-scale reference audit"
        )
    solver = _exact_keys(
        audit.get("solver"),
        {"solver_id", "contact_height_fraction", "root_step_search_m"},
        owner="world-contact solver",
    )
    if solver != {
        "solver_id": "height_backward_velocity_constant_root_minimax_v1",
        "contact_height_fraction": 0.35,
        "root_step_search_m": {
            "minimum": 0.005,
            "maximum": 0.04,
            "increment": 0.0001,
        },
    }:
        raise CanaryPromotionError(
            "world-contact solver does not use the complete fixed M2 configuration"
        )

    records = _records_by_role(candidate)
    expected_hashes = {
        "source_glb_sha256": records["visual"]["sha256"],
        "baked_actions_sha256": records["walk_poses"]["sha256"],
    }
    for field, expected in expected_hashes.items():
        if audit.get(field) != expected or contact_report.get(field) != expected:
            raise CanaryPromotionError(
                f"world-contact artifacts {field} differ from candidate"
            )
    if audit.get("contact_phases_sha256") != records["contact_phases"]["sha256"]:
        raise CanaryPromotionError(
            "world-contact audit contact_phases_sha256 differs from candidate"
        )

    if (
        contact_report.get("schema") != "avengine_m2_contact_phases_v1"
        or contact_report.get("qualification_state") != "research_candidate"
        or contact_report.get("qualification_claim") is not False
        or contact_report.get("runtime_joint_order")
        != candidate.get("skeleton", {}).get("runtime_joint_order")
        or contact_report.get("contact_order") != list(CONTACT_ORDER)
        or contact_report.get("coordinate_system") != candidate.get("coordinate_system")
        or contact_report.get("sample_rate_hz") != 15
        or contact_report.get("time_base_hz") != 48000
        or contact_report.get("warnings") != []
    ):
        raise CanaryPromotionError("candidate contact report is not a normative pass")
    expected_anchors = [
        anchor
        for anchor in candidate.get("anchors", [])
        if isinstance(anchor, Mapping) and anchor.get("anchor_id") in CONTACT_ORDER
    ]
    if contact_report.get("anchor_definitions") != expected_anchors:
        raise CanaryPromotionError("contact report anchors differ from candidate")
    thresholds = _exact_keys(
        contact_report.get("thresholds"),
        {
            "minimum_dynamic_vertical_range_m",
            "contact_height_fraction",
            "maximum_idle_vertical_range_m",
            "maximum_idle_step_displacement_m",
            "maximum_contact_horizontal_step_m",
        },
        owner="contact-report thresholds",
    )
    if thresholds != {
        "minimum_dynamic_vertical_range_m": 0.005,
        "contact_height_fraction": 0.35,
        "maximum_idle_vertical_range_m": 0.015,
        "maximum_idle_step_displacement_m": 0.003,
        "maximum_contact_horizontal_step_m": 0.015,
    }:
        raise CanaryPromotionError(
            "contact report thresholds differ from fixed M2 gates"
        )

    root_fit = _exact_keys(
        audit.get("root_step_fit"),
        {
            "step_m",
            "direction_world",
            "maximum_contact_horizontal_step_m",
            "mean_contact_horizontal_step_m",
            "contact_pair_count",
        },
        owner="root-step fit",
    )
    step = _finite_number(root_fit["step_m"], owner="root-step fit step", minimum=0.0)
    direction = _finite_vector(
        root_fit["direction_world"], size=3, owner="root-step direction"
    )
    root_maximum = _finite_number(
        root_fit["maximum_contact_horizontal_step_m"],
        owner="root-step maximum residual",
        minimum=0.0,
    )
    root_mean = _finite_number(
        root_fit["mean_contact_horizontal_step_m"],
        owner="root-step mean residual",
        minimum=0.0,
    )
    root_pair_count = root_fit.get("contact_pair_count")
    if (
        not isinstance(root_pair_count, int)
        or isinstance(root_pair_count, bool)
        or root_pair_count <= 0
        or step < 0.005
        or step > 0.04
        or not _close(math.sqrt(sum(value * value for value in direction)), 1.0)
        or not _close(root_maximum, measured)
    ):
        raise CanaryPromotionError("root-step fit conflicts with the fixed gate")

    contact_metrics = _exact_keys(
        audit.get("contacts"), set(CONTACT_ORDER), owner="world-contact metrics"
    )
    metric_by_contact: dict[str, Mapping[str, Any]] = {}
    weighted_total = 0.0
    pair_total = 0
    maxima: list[float] = []
    for contact_id in CONTACT_ORDER:
        record = _exact_keys(
            contact_metrics[contact_id],
            {
                "contact_pair_count",
                "maximum_contact_horizontal_step_m",
                "mean_contact_horizontal_step_m",
            },
            owner=f"world-contact metric {contact_id}",
        )
        count = record.get("contact_pair_count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise CanaryPromotionError(f"invalid contact-pair count for {contact_id}")
        maximum_value = _finite_number(
            record["maximum_contact_horizontal_step_m"],
            owner=f"maximum contact residual {contact_id}",
            minimum=0.0,
        )
        mean_value = _finite_number(
            record["mean_contact_horizontal_step_m"],
            owner=f"mean contact residual {contact_id}",
            minimum=0.0,
        )
        if maximum_value > maximum or mean_value > maximum_value:
            raise CanaryPromotionError(f"contact residual for {contact_id} fails gate")
        metric_by_contact[contact_id] = record
        pair_total += count
        weighted_total += count * mean_value
        maxima.append(maximum_value)
    if (
        pair_total != root_pair_count
        or not _close(max(maxima), root_maximum)
        or not _close(weighted_total / pair_total, root_mean)
    ):
        raise CanaryPromotionError("per-contact metrics conflict with root-step fit")

    uniform_scale = _exact_keys(
        audit.get("uniform_linear_scale"),
        {
            "reference",
            "target",
            "normalized_measured_maximum_contact_horizontal_step_m",
            "all_dimensional_solver_parameters_scaled",
        },
        owner="uniform-scale audit",
    )
    if (
        uniform_scale.get("reference") != 1.0
        or uniform_scale.get("target") != 1.0
        or uniform_scale.get("all_dimensional_solver_parameters_scaled") is not True
        or not _close(
            _finite_number(
                uniform_scale.get(
                    "normalized_measured_maximum_contact_horizontal_step_m"
                ),
                owner="normalized contact residual",
                minimum=0.0,
            ),
            measured,
        )
    ):
        raise CanaryPromotionError(
            "uniform-scale audit conflicts with fixed unit scale"
        )

    actions = contact_report.get("actions")
    if not isinstance(actions, list) or len(actions) != 2:
        raise CanaryPromotionError("contact report must contain idle and walk actions")
    actions_by_id = {
        action.get("semantic_action_id"): action
        for action in actions
        if isinstance(action, Mapping)
        and isinstance(action.get("semantic_action_id"), str)
    }
    if set(actions_by_id) != {"idle", "walk"}:
        raise CanaryPromotionError("contact report action set is non-normative")
    stance = _exact_keys(
        audit.get("stance_frames_by_contact"),
        set(CONTACT_ORDER),
        owner="stance-frame map",
    )
    asset_actions = {
        action.get("action_id"): action
        for action in candidate.get("actions", [])
        if isinstance(action, Mapping)
    }
    for action_id in ("idle", "walk"):
        action = _exact_keys(
            actions_by_id[action_id],
            {
                "semantic_action_id",
                "source_action_name",
                "sample_count",
                "frames",
                "metrics",
            },
            owner=f"contact action {action_id}",
        )
        sample_count = action.get("sample_count")
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count < 3
            or sample_count != asset_actions.get(action_id, {}).get("sample_count")
        ):
            raise CanaryPromotionError(
                f"contact action {action_id} sample count differs"
            )
        frames = action.get("frames")
        metrics = action.get("metrics")
        if not isinstance(frames, list) or len(frames) != sample_count:
            raise CanaryPromotionError(f"contact action {action_id} frames differ")
        if not isinstance(metrics, list) or len(metrics) != len(CONTACT_ORDER):
            raise CanaryPromotionError(f"contact action {action_id} metrics differ")
        frame_contacts: dict[str, list[int]] = {name: [] for name in CONTACT_ORDER}
        for index, frame in enumerate(frames):
            frame = _exact_keys(
                frame,
                {"sample_index", "sample_tick", "source_time_seconds", "contacts"},
                owner=f"contact action {action_id} frame {index}",
            )
            if frame.get("sample_index") != index:
                raise CanaryPromotionError(
                    f"contact action {action_id} frame order differs"
                )
            _finite_number(
                frame.get("source_time_seconds"),
                owner=f"contact action {action_id} source time",
                minimum=0.0,
            )
            if not isinstance(frame.get("sample_tick"), int):
                raise CanaryPromotionError(
                    f"contact action {action_id} tick is invalid"
                )
            records_by_id = {
                record.get("contact_id"): record
                for record in frame.get("contacts", [])
                if isinstance(record, Mapping)
            }
            if set(records_by_id) != set(CONTACT_ORDER) or any(
                set(record) != {"contact_id", "in_contact"}
                or not isinstance(record.get("in_contact"), bool)
                for record in records_by_id.values()
            ):
                raise CanaryPromotionError(
                    f"contact action {action_id} frame contacts differ"
                )
            for contact_id, record in records_by_id.items():
                if record["in_contact"]:
                    frame_contacts[contact_id].append(index)
        metrics_by_id = {
            metric.get("contact_id"): metric
            for metric in metrics
            if isinstance(metric, Mapping)
        }
        if set(metrics_by_id) != set(CONTACT_ORDER):
            raise CanaryPromotionError(
                f"contact action {action_id} metric contacts differ"
            )
        for contact_id in CONTACT_ORDER:
            metric = metrics_by_id[contact_id]
            required_metric_keys = {
                "contact_id",
                "inference_mode",
                "confidence",
                "idle_reference_height_m",
                "contact_height_threshold_m",
                "minimum_height_m",
                "maximum_height_m",
                "vertical_range_m",
                "maximum_step_displacement_m",
                "maximum_horizontal_step_m",
                "maximum_contact_horizontal_step_m",
                "contact_frame_count",
                "swing_frame_count",
            }
            _exact_keys(
                metric, required_metric_keys, owner=f"{action_id} metric {contact_id}"
            )
            if (
                metric.get("confidence") != "high"
                or metric.get("contact_frame_count") != len(frame_contacts[contact_id])
                or metric.get("swing_frame_count")
                != sample_count - len(frame_contacts[contact_id])
            ):
                raise CanaryPromotionError(
                    f"contact metric counts conflict for {contact_id}"
                )
            if action_id == "idle":
                if (
                    frame_contacts[contact_id] != list(range(sample_count))
                    or metric.get("inference_mode") != "forced_idle_contact"
                    or metric.get("maximum_contact_horizontal_step_m") != 0.0
                    or metric.get("vertical_range_m")
                    != idle_contacts[contact_id]["vertical_range_m"]
                    or metric.get("maximum_step_displacement_m")
                    != idle_contacts[contact_id]["maximum_step_displacement_m"]
                ):
                    raise CanaryPromotionError(
                        f"idle contact report conflicts for {contact_id}"
                    )
            else:
                frames_for_contact = frame_contacts[contact_id]
                if (
                    stance[contact_id] != frames_for_contact
                    or metric.get("inference_mode")
                    != "height_backward_velocity_world_locked"
                    or metric.get("vertical_range_m")
                    != walk_contacts[contact_id]["vertical_range_m"]
                    or metric.get("maximum_contact_horizontal_step_m")
                    != metric_by_contact[contact_id][
                        "maximum_contact_horizontal_step_m"
                    ]
                ):
                    raise CanaryPromotionError(
                        f"walk stance report conflicts for {contact_id}"
                    )
                state_set = set(frames_for_contact)
                pair_count = sum(
                    index in state_set and (index - 1) % sample_count in state_set
                    for index in range(sample_count)
                )
                if pair_count != metric_by_contact[contact_id]["contact_pair_count"]:
                    raise CanaryPromotionError(
                        f"stance pairs conflict for {contact_id}"
                    )

    walk_states = [
        state
        for state in request.get("states", [])
        if isinstance(state, Mapping) and state.get("action_id") == "walk"
    ]
    trajectory = _exact_keys(
        audit.get("trajectory"),
        {
            "start_translation_m",
            "end_translation_m",
            "rotation_xyzw",
            "walk_frame_count",
            "sample_rate_hz",
            "path_length_m",
            "root_speed_m_per_second",
        },
        owner="world-contact trajectory",
    )
    start = _finite_vector(
        trajectory["start_translation_m"], size=3, owner="trajectory start"
    )
    end = _finite_vector(
        trajectory["end_translation_m"], size=3, owner="trajectory end"
    )
    rotation = _finite_vector(
        trajectory["rotation_xyzw"], size=4, owner="trajectory rotation"
    )
    if (
        len(walk_states) != 45
        or trajectory.get("walk_frame_count") != 45
        or trajectory.get("sample_rate_hz") != 15
        or not _close(
            _finite_number(trajectory.get("path_length_m"), owner="trajectory length"),
            step * 44,
        )
        or not _close(
            _finite_number(
                trajectory.get("root_speed_m_per_second"), owner="root speed"
            ),
            step * 15,
        )
    ):
        raise CanaryPromotionError(
            "world-contact trajectory differs from the 45-state walk request"
        )
    expected_end = tuple(
        start[index] + direction[index] * step * 44 for index in range(3)
    )
    if any(not _close(end[index], expected_end[index]) for index in range(3)):
        raise CanaryPromotionError(
            "world-contact trajectory conflicts with root-step fit"
        )
    for index, state in enumerate(walk_states):
        transform = state.get("root_transform")
        if not isinstance(transform, Mapping):
            raise CanaryPromotionError("walk request lacks a root transform")
        translation = _finite_vector(
            transform.get("translation_m"),
            size=3,
            owner=f"walk state {index} translation",
        )
        state_rotation = _finite_vector(
            transform.get("rotation_xyzw"), size=4, owner=f"walk state {index} rotation"
        )
        expected_translation = tuple(
            start[axis] + direction[axis] * step * index for axis in range(3)
        )
        if any(
            not _close(translation[axis], expected_translation[axis])
            for axis in range(3)
        ) or any(not _close(state_rotation[axis], rotation[axis]) for axis in range(4)):
            raise CanaryPromotionError(
                "world-contact trajectory differs from an intermediate walk state"
            )
    if contact_report != reconstructed_contact:
        raise CanaryPromotionError(
            "candidate contact report differs from independent reconstruction"
        )
    if audit != reconstructed_audit:
        raise CanaryPromotionError(
            "world-contact audit differs from independent reconstruction"
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


def _publish_directory_no_replace(staging: Path, output: Path) -> None:
    """Atomically expose a complete promotion without replacing any entry."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CanaryPromotionError(
            "atomic no-replace promotion publication is unavailable"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(staging),
        -100,  # AT_FDCWD
        os.fsencode(output),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise CanaryPromotionError(
            "promotion output appeared during atomic publication; refusing to "
            f"overwrite it: {output}"
        )
    raise CanaryPromotionError(
        f"unable to atomically publish promotion directory: {os.strerror(error)}"
    )


def promote_research_candidate(
    *,
    candidate_manifest: str | Path,
    human_review_decision: str | Path,
    review_request: str | Path,
    capture_evidence: str | Path,
    world_contact_audit: ExpectedArtifact,
    diagnostic_videos: Sequence[ExpectedArtifact],
    rocketbox_root: str | Path,
    output_directory: str | Path,
) -> PromotionResult:
    """Copy and promote one immutable research candidate without overwriting."""

    candidate_path = _regular_file(candidate_manifest, owner="candidate asset manifest")
    request_path = _regular_file(review_request, owner="75-state review request")
    evidence_path = _regular_file(capture_evidence, owner="capture evidence")
    candidate_root = _safe_tree(candidate_path.parent, owner="candidate package")
    if candidate_path.parent != candidate_root:
        raise CanaryPromotionError("candidate manifest must be at the package root")
    output = _new_output_directory(output_directory, owner="promotion output directory")

    candidate = _validate_candidate(candidate_path)
    decision = load_human_review_decision(human_review_decision)
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
        candidate_manifest=candidate_path,
        request=request,
        request_path=request_path,
        evidence_path=evidence_path,
        decision=decision,
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
    reviewer_decision = _verify_human_review_decision_bindings(
        decision,
        candidate=candidate,
        candidate_manifest=candidate_path,
        diagnostics=validated_diagnostics,
    )

    evidence_root = _safe_tree(evidence_path.parent, owner="capture evidence tree")
    staging = output.parent / f".{output.name}.staging-{uuid4().hex}"
    try:
        shutil.copytree(candidate_root, staging, symlinks=False)
        admission = staging / "admission"
        admission.mkdir()
        candidate_snapshot = admission / "candidate_asset_manifest.json"
        shutil.copy2(candidate_path, candidate_snapshot)
        decision_snapshot = admission / "human_review_decision.json"
        shutil.copy2(decision.path, decision_snapshot)
        if (
            decision_snapshot.stat().st_size != decision.byte_size
            or sha256_file(decision_snapshot) != decision.sha256
        ):
            raise CanaryPromotionError("human review decision copy hash differs")
        request_snapshot = admission / "capture_request.json"
        shutil.copy2(request_path, request_snapshot)
        world_contact_snapshot = admission / "world_contact_audit.json"
        shutil.copy2(world_contact_path, world_contact_snapshot)
        if sha256_file(world_contact_snapshot) != world_contact_audit.sha256:
            raise CanaryPromotionError("world-contact audit copy hash differs")
        evidence_snapshot_root = admission / "capture_evidence"
        shutil.copytree(evidence_root, evidence_snapshot_root, symlinks=False)
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
                key: reviewer_decision[key] for key in sorted(_DECISION_REVIEWER_KEYS)
            },
            "decision_reason": (
                "Explicit human-review decision record "
                f"{decision.sha256} binds the candidate, visual/actions and "
                "reviewed diagnostic bytes."
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
            "human_review_decision": _artifact(
                decision_snapshot, relative_to=admission
            ),
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
                "Hash-closed automatic QA and the explicit user visual review on "
                f"{reviewer_decision['decision_date']} both passed for bounded "
                "M2 research-canary use."
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
        _publish_directory_no_replace(staging, output)
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
    "HUMAN_REVIEW_DECISION_SCHEMA",
    "HumanReviewDecision",
    "LEGACY_M2_HUMAN_REVIEW_DECISION",
    "PromotionResult",
    "ROCKETBOX_LICENSE_SHA256",
    "ROCKETBOX_README_SHA256",
    "ROCKETBOX_REPOSITORY",
    "ROCKETBOX_REVISION",
    "USER_DECISION_DATE",
    "USER_DECISION_STATEMENT",
    "build_human_review_decision",
    "load_human_review_decision",
    "promote_research_candidate",
    "validate_human_review_decision",
    "write_human_review_decision_exclusive",
    "write_legacy_m2_human_review_decision",
]
