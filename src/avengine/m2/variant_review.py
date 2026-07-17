"""Single-view Habitat review capture for animal appearance variants.

This module is deliberately separate from formal M2 admission.  It reuses the
strict M2 baked-state request and fixed-state Habitat renderer, but the emitted
media is always review-only and never promotes or qualifies an asset.

The renderer consumes an M2 animal package.  A source GLB with ``Idle`` and
``Walking`` clips must first be converted to the canonical baked action NPZ by
``tools/m2/bake_actions.py`` and compiled into such a package; free-running GLB
animation is not an accepted input to this path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    resolve_declared_path,
    sha256_file,
)
from avengine.m1.contracts import ValidatedM1Inputs
from avengine.m2.actions import BakedActionSet, read_baked_actions_npz
from avengine.m2.contracts import (
    FORMAL_MODALITIES,
    FORMAL_VIEW_IDS,
    ValidatedM2Inputs,
    load_and_validate_inputs as load_formal_m2_inputs,
    validate_animal_asset_package,
)
from avengine.m2.habitat_capture import (
    EVIDENCE_SCHEMA as CORE_CAPTURE_EVIDENCE_SCHEMA,
    HabitatCaptureError,
    _capture_m2_states,
    load_research_review_inputs,
    validate_capture_context,
    validate_research_review_context,
)
from avengine.m2.timeline import (
    FRAME_COUNT,
    IDLE_LEAD_FRAME_COUNT,
    IDLE_TAIL_FRAME_COUNT,
    WALK_FRAME_COUNT,
    M2CanaryTrajectory,
    build_m2_capture_request,
    build_m2_research_review_request,
)


VARIANT_REVIEW_EVIDENCE_SCHEMA = "avengine_animal_variant_habitat_review_v1"
VARIANT_REVIEW_EVIDENCE_FILENAME = "variant_review_evidence.json"
ALLOWED_REVIEW_ADMISSION_STATES = {"research_candidate", "canary_qualified"}


class VariantReviewError(RuntimeError):
    """An animal variant cannot produce a bounded review-only capture."""


@dataclass(frozen=True)
class VariantReviewRoomPreset:
    """One validated M1 room pair and a visible actor-space review path."""

    preset_id: str
    room_manifest_relative: str
    room_request_relative: str
    trajectory: M2CanaryTrajectory


# Both paths cover 0.8712 m during the 45 walking states.  That is the bounded
# M2 Beagle canary distance.  Other species/cadences should supply an explicit
# trajectory JSON rather than silently inheriting this review default.
ROOM_PRESETS: Mapping[str, VariantReviewRoomPreset] = {
    "blender_custom": VariantReviewRoomPreset(
        preset_id="blender_custom",
        room_manifest_relative=("examples/m1/rooms/blender_custom/room_manifest.json"),
        room_request_relative=(
            "examples/m2/rooms/blender_custom_articulated_review/capture_request.json"
        ),
        # This review-only camera is 0.18 m below the formal M1 viewpoint so
        # the full animated paws remain in frame without lifting the actor off
        # its contact-bound ground trajectory.
        trajectory=M2CanaryTrajectory(
            start_translation_m=(-0.15, 0.02, 0.8),
            end_translation_m=(-0.15, 0.02, -0.0712),
            rotation_xyzw=(
                0.0,
                0.7071067811865475,
                0.0,
                0.7071067811865476,
            ),
        ),
    ),
    "habitat_mp3d_example": VariantReviewRoomPreset(
        preset_id="habitat_mp3d_example",
        room_manifest_relative=(
            "examples/m2/rooms/habitat_mp3d_articulated_review/room_manifest.json"
        ),
        room_request_relative=(
            "examples/m2/rooms/habitat_mp3d_articulated_review/capture_request.json"
        ),
        # The MP3D camera looks along world -Z.  The fixed -3.56 m depth keeps
        # the complete animal (including paws) away from the image boundary,
        # while the lateral +X path presents its profile instead of its rear.
        trajectory=M2CanaryTrajectory(
            start_translation_m=(-4.55, 0.072447, -3.56),
            end_translation_m=(-3.6788, 0.072447, -3.56),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
    ),
}


def repository_root() -> Path:
    """Return the source checkout that owns the packaged room presets."""

    return Path(__file__).resolve().parents[3]


def resolve_room_preset(
    preset_id: str,
    *,
    root: str | Path | None = None,
) -> tuple[Path, Path, M2CanaryTrajectory]:
    """Resolve one named M1 room without accepting an implicit fallback."""

    try:
        preset = ROOM_PRESETS[preset_id]
    except KeyError as exc:
        raise VariantReviewError(
            f"unknown room preset {preset_id!r}; expected one of {sorted(ROOM_PRESETS)}"
        ) from exc
    base = Path(root).resolve() if root is not None else repository_root()
    room_manifest = (base / preset.room_manifest_relative).resolve()
    room_request = (base / preset.room_request_relative).resolve()
    for label, path in (
        ("room manifest", room_manifest),
        ("room request", room_request),
    ):
        if path.is_symlink() or not path.is_file():
            raise VariantReviewError(f"{label} is unavailable for {preset_id}: {path}")
    return room_manifest, room_request, preset.trajectory


def load_trajectory(path: str | Path) -> M2CanaryTrajectory:
    """Load an explicit, absolute room-space trajectory with exact keys."""

    resolved = Path(path).resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise VariantReviewError(f"trajectory JSON is not a regular file: {resolved}")
    value = load_json(resolved)
    expected = {
        "start_translation_m",
        "end_translation_m",
        "rotation_xyzw",
    }
    if set(value) != expected:
        raise VariantReviewError(
            "trajectory JSON must contain exactly start_translation_m, "
            "end_translation_m, and rotation_xyzw"
        )
    try:
        return M2CanaryTrajectory(
            start_translation_m=tuple(value["start_translation_m"]),
            end_translation_m=tuple(value["end_translation_m"]),
            rotation_xyzw=tuple(value["rotation_xyzw"]),
        )
    except (TypeError, ValueError) as exc:
        raise VariantReviewError(f"invalid trajectory JSON: {exc}") from exc


def _records_by_role(asset: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for record in asset.get("files", []):
        if not isinstance(record, Mapping) or not isinstance(record.get("role"), str):
            raise VariantReviewError("asset files must contain role-bound objects")
        role = str(record["role"])
        if role in records:
            raise VariantReviewError(f"duplicate animal package role: {role}")
        records[role] = record
    return records


def _role_path(
    manifest_path: Path,
    records: Mapping[str, Mapping[str, Any]],
    role: str,
) -> Path:
    record = records.get(role)
    raw_path = record.get("path") if isinstance(record, Mapping) else None
    if not isinstance(raw_path, str) or not raw_path:
        raise VariantReviewError(f"animal package lacks a path for role {role!r}")
    try:
        path = resolve_declared_path(raw_path, manifest_dir=manifest_path.parent)
    except (OSError, TypeError, ValueError) as exc:
        raise VariantReviewError(
            f"unable to resolve package role {role!r}: {exc}"
        ) from exc
    if path.is_symlink() or not path.is_file():
        raise VariantReviewError(f"package role {role!r} is not a regular file")
    if path.stat().st_size != record.get("byte_size") or sha256_file(
        path
    ) != record.get("sha256"):
        raise VariantReviewError(f"package role {role!r} bytes changed")
    return path


def _merge_role_bound_actions(
    idle_actions: BakedActionSet,
    walk_actions: BakedActionSet,
) -> BakedActionSet:
    """Combine independently role-bound NPZs without accepting mixed sources."""

    if (
        idle_actions.source_glb_sha256 != walk_actions.source_glb_sha256
        or idle_actions.runtime_joint_order != walk_actions.runtime_joint_order
        or idle_actions.sample_rate_hz != walk_actions.sample_rate_hz
        or idle_actions.time_base_hz != walk_actions.time_base_hz
    ):
        raise VariantReviewError(
            "idle and walk baked actions have different identities"
        )
    return BakedActionSet(
        source_glb_sha256=idle_actions.source_glb_sha256,
        runtime_joint_order=idle_actions.runtime_joint_order,
        actions=(idle_actions.action("idle"), walk_actions.action("walk")),
        sample_rate_hz=idle_actions.sample_rate_hz,
        time_base_hz=idle_actions.time_base_hz,
    )


def _load_baked_actions(
    manifest_path: Path,
    records: Mapping[str, Mapping[str, Any]],
) -> BakedActionSet:
    idle_path = _role_path(manifest_path, records, "idle_poses")
    walk_path = _role_path(manifest_path, records, "walk_poses")
    idle_actions = read_baked_actions_npz(idle_path)
    walk_actions = (
        idle_actions if walk_path == idle_path else read_baked_actions_npz(walk_path)
    )
    return _merge_role_bound_actions(idle_actions, walk_actions)


def _load_contact_phases(
    path: Path,
    *,
    actions: BakedActionSet,
) -> dict[str, tuple[tuple[bool, ...], ...]]:
    value = load_json(path)
    expected_order = [
        "paw_front_left",
        "paw_front_right",
        "paw_hind_left",
        "paw_hind_right",
    ]
    if value.get("contact_order") != expected_order:
        raise VariantReviewError("contact report order differs from M2")
    raw_actions = value.get("actions")
    if not isinstance(raw_actions, list):
        raise VariantReviewError("contact report actions must be an array")
    phases: dict[str, tuple[tuple[bool, ...], ...]] = {}
    for raw_action in raw_actions:
        if not isinstance(raw_action, Mapping):
            raise VariantReviewError("contact report action must be an object")
        action_id = raw_action.get("semantic_action_id")
        frames = raw_action.get("frames")
        if action_id not in {"idle", "walk"} or not isinstance(frames, list):
            raise VariantReviewError("contact report action mapping is invalid")
        if action_id in phases:
            raise VariantReviewError(f"duplicate contact action: {action_id}")
        decoded: list[tuple[bool, ...]] = []
        for frame in frames:
            states = frame.get("contacts") if isinstance(frame, Mapping) else None
            if not isinstance(states, list) or len(states) != len(expected_order):
                raise VariantReviewError("contact frame state count is invalid")
            values: list[bool] = []
            for state, contact_id in zip(states, expected_order, strict=True):
                if (
                    not isinstance(state, Mapping)
                    or state.get("contact_id") != contact_id
                    or not isinstance(state.get("in_contact"), bool)
                ):
                    raise VariantReviewError("contact frame state order is invalid")
                values.append(bool(state["in_contact"]))
            decoded.append(tuple(values))
        phases[str(action_id)] = tuple(decoded)
    expected_counts = {
        action.semantic_action_id: action.sample_count for action in actions.actions
    }
    if set(phases) != set(expected_counts):
        raise VariantReviewError("contact report must contain exactly idle and walk")
    for action_id, count in expected_counts.items():
        if len(phases[action_id]) != count:
            raise VariantReviewError(
                f"contact report {action_id} frame count differs from baked actions"
            )
    return phases


def validate_variant_review_schedule(request: Mapping[str, Any]) -> list[str]:
    """Check the one-view 15/45/15 review contract independently."""

    errors: list[str] = []
    states = request.get("states")
    if not isinstance(states, list) or len(states) != FRAME_COUNT:
        errors.append(f"review request must contain exactly {FRAME_COUNT} states")
        states = []
    if request.get("view_ids") != FORMAL_VIEW_IDS:
        errors.append("review request view_ids must be exactly ['view0']")
    if request.get("modalities") != FORMAL_MODALITIES:
        errors.append("view0 must expose co-located rgb/depth/semantic modalities")
    expected_actions = (
        ["idle"] * IDLE_LEAD_FRAME_COUNT
        + ["walk"] * WALK_FRAME_COUNT
        + ["idle"] * IDLE_TAIL_FRAME_COUNT
    )
    if states:
        actual_indices = [state.get("frame_index") for state in states]
        if actual_indices != list(range(FRAME_COUNT)):
            errors.append("review frame indices must be contiguous 0..74")
        actual_actions = [state.get("action_id") for state in states]
        if actual_actions != expected_actions:
            errors.append("review action schedule must be Idle15/Walk45/Idle15")
    return errors


def build_variant_review_request(
    *,
    asset_manifest: str | Path,
    room_inputs: ValidatedM1Inputs,
    request_id: str,
    trajectory: M2CanaryTrajectory,
) -> dict[str, Any]:
    """Build one strict request from package-bound baked actions and contacts."""

    manifest_path = Path(asset_manifest).resolve()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise VariantReviewError(
            f"asset manifest is not a regular file: {manifest_path}"
        )
    asset = load_json(manifest_path)
    package_errors = validate_animal_asset_package(
        asset,
        manifest_path=manifest_path,
    )
    if package_errors:
        raise VariantReviewError("invalid animal package: " + "; ".join(package_errors))
    admission_state = asset.get("admission_state")
    if admission_state not in ALLOWED_REVIEW_ADMISSION_STATES:
        raise VariantReviewError(
            "variant review accepts only research_candidate or canary_qualified"
        )
    if not isinstance(room_inputs, ValidatedM1Inputs):
        raise VariantReviewError("room_inputs must come from the M1 contract loader")
    records = _records_by_role(asset)
    actions = _load_baked_actions(manifest_path, records)
    contacts = _load_contact_phases(
        _role_path(manifest_path, records, "contact_phases"),
        actions=actions,
    )
    builder = (
        build_m2_research_review_request
        if admission_state == "research_candidate"
        else build_m2_capture_request
    )
    try:
        request = builder(
            asset=asset,
            asset_manifest_sha256=sha256_file(manifest_path),
            actions=actions,
            contact_phases=contacts,
            request_id=request_id,
            room_id=room_inputs.room["room_id"],
            seed=room_inputs.request["seed"],
            trajectory=trajectory,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise VariantReviewError(
            f"unable to build variant review request: {exc}"
        ) from exc
    schedule_errors = validate_variant_review_schedule(request)
    if schedule_errors:
        raise VariantReviewError("; ".join(schedule_errors))
    return request


def write_json_exclusive(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Create deterministic JSON once and refuse files, directories or symlinks."""

    resolved = Path(path).resolve()
    if Path(path).exists() or Path(path).is_symlink():
        raise VariantReviewError(f"refusing to replace output: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        with resolved.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise VariantReviewError(f"refusing to replace output: {resolved}") from exc
    return resolved


def load_variant_review_inputs(
    asset_manifest: str | Path,
    request_path: str | Path,
) -> ValidatedM2Inputs:
    """Use the existing admission-specific M2 loaders without weakening either."""

    asset_path = Path(asset_manifest).resolve()
    asset = load_json(asset_path)
    admission_state = asset.get("admission_state")
    try:
        if admission_state == "research_candidate":
            result = load_research_review_inputs(asset_path, request_path)
        elif admission_state == "canary_qualified":
            result = load_formal_m2_inputs(asset_path, request_path)
        else:
            raise VariantReviewError(
                "variant review accepts only research_candidate or canary_qualified"
            )
    except VariantReviewError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise VariantReviewError(f"invalid variant review inputs: {exc}") from exc
    schedule_errors = validate_variant_review_schedule(result.request)
    if schedule_errors:
        raise VariantReviewError("; ".join(schedule_errors))
    return result


def _reload_variant_context(
    inputs: ValidatedM2Inputs,
    room_inputs: ValidatedM1Inputs,
) -> tuple[ValidatedM2Inputs, ValidatedM1Inputs]:
    admission_state = inputs.asset.get("admission_state")
    validator = (
        validate_research_review_context
        if admission_state == "research_candidate"
        else validate_capture_context
    )
    errors = validator(inputs, room_inputs)
    if errors:
        raise VariantReviewError("; ".join(errors))
    from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs

    reloaded_m2 = load_variant_review_inputs(inputs.asset_path, inputs.request_path)
    reloaded_m1 = load_m1_inputs(room_inputs.room_path, room_inputs.request_path)
    if reloaded_m2 != inputs:
        raise VariantReviewError("M2 inputs differ from current manifest bytes")
    if reloaded_m1 != room_inputs:
        raise VariantReviewError("M1 inputs differ from current manifest bytes")
    errors = validator(reloaded_m2, reloaded_m1)
    if errors:
        raise VariantReviewError("; ".join(errors))
    return reloaded_m2, reloaded_m1


def _assert_core_review_evidence(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema") != CORE_CAPTURE_EVIDENCE_SCHEMA
        or evidence.get("status") != "review_only"
        or evidence.get("review_only") is not True
        or evidence.get("qualification_claim") is not False
        or evidence.get("formal_view_ids") != []
        or evidence.get("review_view_ids") != ["view0"]
        or evidence.get("review_modalities") != FORMAL_MODALITIES
    ):
        raise VariantReviewError("fixed-state renderer emitted an invalid review claim")
    frames = evidence.get("frames")
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        raise VariantReviewError("fixed-state renderer did not emit 75 frames")
    review_media = evidence.get("review_media")
    videos = review_media.get("videos") if isinstance(review_media, Mapping) else None
    if not isinstance(videos, Mapping) or set(videos) != set(FORMAL_MODALITIES):
        raise VariantReviewError("fixed-state renderer did not emit review media")
    for modality in FORMAL_MODALITIES:
        record = videos[modality]
        if (
            not isinstance(record, Mapping)
            or record.get("view_id") != "view0"
            or record.get("frame_count") != FRAME_COUNT
            or record.get("frame_rate_hz") != 15
        ):
            raise VariantReviewError(f"invalid {modality} review media record")


def _wrapper_evidence(
    *,
    inputs: ValidatedM2Inputs,
    room_inputs: ValidatedM1Inputs,
    output: Path,
    core: Mapping[str, Any],
) -> dict[str, Any]:
    core_evidence_path = output / "evidence.json"
    videos = core["review_media"]["videos"]
    array_artifacts = core["array_artifacts"]
    value: dict[str, Any] = {
        "schema": VARIANT_REVIEW_EVIDENCE_SCHEMA,
        "status": "pass",
        "evidence_kind": "animal_variant_habitat_review",
        "review_only": True,
        "qualification_claim": False,
        "asset_id": inputs.asset["asset_id"],
        "asset_admission_state": inputs.asset["admission_state"],
        "request_id": inputs.request["request_id"],
        "room_id": room_inputs.room["room_id"],
        "view_contract": {
            "view_ids": ["view0"],
            "camera_rig_id": "camera_rig_0",
            "modalities": list(FORMAL_MODALITIES),
            "co_located_modalities": True,
            "camera_count": 1,
        },
        "timeline": {
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": 15,
            "segments": [
                {"action_id": "idle", "start_frame": 0, "frame_count": 15},
                {"action_id": "walk", "start_frame": 15, "frame_count": 45},
                {"action_id": "idle", "start_frame": 60, "frame_count": 15},
            ],
            "state_evaluation": "explicit_fixed_state",
            "free_running_animation": False,
        },
        "inputs": {
            "animal_asset_package": {
                "path": str(inputs.asset_path),
                "sha256": sha256_file(inputs.asset_path),
            },
            "capture_request": {
                "path": str(inputs.request_path),
                "sha256": sha256_file(inputs.request_path),
            },
            "room_manifest": {
                "path": str(room_inputs.room_path),
                "sha256": sha256_file(room_inputs.room_path),
            },
            "room_request": {
                "path": str(room_inputs.request_path),
                "sha256": sha256_file(room_inputs.request_path),
            },
        },
        "core_capture_evidence": {
            **file_record(core_evidence_path, relative_to=output),
            "evidence_content_sha256": core["evidence_content_sha256"],
        },
        "rgb_review_video": dict(videos["rgb"]["artifact"]),
        "review_videos": {
            modality: dict(videos[modality]["artifact"])
            for modality in FORMAL_MODALITIES
        },
        "array_artifacts": {
            modality: dict(array_artifacts[modality]["artifact"])
            for modality in FORMAL_MODALITIES
        },
        "runtime_identity": dict(core["runtime_identity"]),
        "world_time_seconds": [
            core["runtime_application"]["initial_world_time_seconds"],
            core["runtime_application"]["final_world_time_seconds"],
        ],
    }
    value["evidence_content_sha256"] = canonical_json_sha256(value)
    return value


def _path_without_symlinks(path: Path, *, owner: str) -> Path:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise VariantReviewError(f"{owner} path contains a symbolic link")
    return absolute


def _verified_record_path(
    record: Mapping[str, Any],
    *,
    owner: str,
    base: Path,
    confined: bool,
    include_byte_size: bool,
) -> Path:
    expected_fields = {"path", "sha256"}
    if include_byte_size:
        expected_fields.add("byte_size")
    if set(record) != expected_fields:
        raise VariantReviewError(f"{owner} file record fields are invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise VariantReviewError(f"{owner} has no artifact path")
    declared = Path(raw_path)
    if confined and declared.is_absolute():
        raise VariantReviewError(f"{owner} must use a relative capture path")
    if not confined and not declared.is_absolute():
        raise VariantReviewError(f"{owner} input path must be absolute")
    candidate = declared if declared.is_absolute() else base / declared
    artifact = _path_without_symlinks(candidate, owner=owner)
    if confined:
        try:
            artifact.relative_to(base)
        except ValueError as exc:
            raise VariantReviewError(f"{owner} escapes the capture output") from exc
    if not artifact.is_file():
        raise VariantReviewError(f"{owner} artifact is missing")
    try:
        byte_size = artifact.stat().st_size
        digest = sha256_file(artifact)
    except OSError as exc:
        raise VariantReviewError(f"{owner} artifact is unreadable: {exc}") from exc
    if (include_byte_size and record.get("byte_size") != byte_size) or record.get(
        "sha256"
    ) != digest:
        raise VariantReviewError(f"{owner} artifact bytes changed")
    return artifact


def _load_verified_input(
    record: Any, *, owner: str, output: Path
) -> tuple[Mapping[str, Any], Path] | None:
    if not isinstance(record, Mapping):
        raise VariantReviewError(f"{owner} input record is missing")
    input_path = _verified_record_path(
        record,
        owner=owner,
        base=output,
        confined=False,
        include_byte_size=False,
    )
    try:
        return load_json(input_path), input_path
    except (OSError, ValueError) as exc:
        raise VariantReviewError(f"{owner} input JSON is invalid: {exc}") from exc


def _expected_review_timeline() -> dict[str, Any]:
    return {
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": 15,
        "segments": [
            {"action_id": "idle", "start_frame": 0, "frame_count": 15},
            {"action_id": "walk", "start_frame": 15, "frame_count": 45},
            {"action_id": "idle", "start_frame": 60, "frame_count": 15},
        ],
        "state_evaluation": "explicit_fixed_state",
        "free_running_animation": False,
    }


def verify_variant_review_evidence(path: str | Path) -> list[str]:
    """Rehash and cross-bind the wrapper, core capture and every declared input."""

    raw_evidence_path = Path(path)
    try:
        evidence_path = _path_without_symlinks(
            raw_evidence_path, owner="variant review evidence"
        )
    except VariantReviewError as exc:
        return [str(exc)]
    if not evidence_path.is_file():
        return ["variant review evidence is not a regular file"]
    try:
        value = load_json(evidence_path)
    except (OSError, ValueError) as exc:
        return [f"variant review evidence is unreadable: {exc}"]
    errors: list[str] = []
    declared_hash = value.get("evidence_content_sha256")
    hash_payload = dict(value)
    hash_payload.pop("evidence_content_sha256", None)
    if declared_hash != canonical_json_sha256(hash_payload):
        errors.append("variant review evidence content hash differs")
    if (
        value.get("schema") != VARIANT_REVIEW_EVIDENCE_SCHEMA
        or value.get("status") != "pass"
        or value.get("evidence_kind") != "animal_variant_habitat_review"
        or value.get("review_only") is not True
        or value.get("qualification_claim") is not False
    ):
        errors.append("variant review evidence claim is invalid")

    output = evidence_path.parent
    core: Mapping[str, Any] | None = None
    core_record = value.get("core_capture_evidence")
    if not isinstance(core_record, Mapping):
        errors.append("core_capture_evidence record is missing")
    else:
        expected_core_fields = {
            "path",
            "byte_size",
            "sha256",
            "evidence_content_sha256",
        }
        if set(core_record) != expected_core_fields:
            errors.append("core_capture_evidence file record fields are invalid")
        if core_record.get("path") != "evidence.json":
            errors.append("core_capture_evidence path is not canonical")
        artifact_record = {
            key: core_record.get(key) for key in ("path", "byte_size", "sha256")
        }
        try:
            core_path = _verified_record_path(
                artifact_record,
                owner="core_capture_evidence",
                base=output,
                confined=True,
                include_byte_size=True,
            )
            core = load_json(core_path)
        except (OSError, ValueError, VariantReviewError) as exc:
            errors.append(f"core capture evidence is invalid: {exc}")
        if core is not None:
            core_hash = core.get("evidence_content_sha256")
            core_payload = dict(core)
            core_payload.pop("evidence_content_sha256", None)
            if core_hash != canonical_json_sha256(core_payload):
                errors.append("core capture evidence content hash differs")
            if core_record.get("evidence_content_sha256") != core_hash:
                errors.append("wrapper/core evidence content hashes differ")
            try:
                _assert_core_review_evidence(core)
            except VariantReviewError as exc:
                errors.append(f"core capture evidence is invalid: {exc}")

    artifact_collections: dict[str, Mapping[str, Any]] = {}
    for collection_name in ("review_videos", "array_artifacts"):
        collection = value.get(collection_name)
        if not isinstance(collection, Mapping) or set(collection) != set(
            FORMAL_MODALITIES
        ):
            errors.append(f"{collection_name} records are incomplete")
            continue
        artifact_collections[collection_name] = collection
        for modality in FORMAL_MODALITIES:
            record = collection[modality]
            if not isinstance(record, Mapping):
                errors.append(f"{collection_name}.{modality} record is invalid")
                continue
            try:
                _verified_record_path(
                    record,
                    owner=f"{collection_name}.{modality}",
                    base=output,
                    confined=True,
                    include_byte_size=True,
                )
            except VariantReviewError as exc:
                errors.append(str(exc))

    wrapper_inputs = value.get("inputs")
    expected_wrapper_inputs = {
        "animal_asset_package",
        "capture_request",
        "room_manifest",
        "room_request",
    }
    input_values: dict[str, Mapping[str, Any]] = {}
    if not isinstance(wrapper_inputs, Mapping) or set(wrapper_inputs) != (
        expected_wrapper_inputs
    ):
        errors.append("variant review input records are incomplete")
    else:
        for owner in expected_wrapper_inputs:
            try:
                loaded = _load_verified_input(
                    wrapper_inputs[owner], owner=owner, output=output
                )
                if loaded is not None:
                    input_values[owner] = loaded[0]
            except VariantReviewError as exc:
                errors.append(str(exc))

    if core is None:
        return errors

    core_inputs = core.get("inputs")
    input_name_map = {
        "animal_asset_package": "animal_asset_package",
        "capture_request": "m2_capture_request",
        "room_manifest": "m1_room_manifest",
        "room_request": "m1_camera_request",
    }
    if not isinstance(core_inputs, Mapping) or set(core_inputs) != set(
        input_name_map.values()
    ):
        errors.append("core capture input records are incomplete")
    elif isinstance(wrapper_inputs, Mapping):
        for wrapper_name, core_name in input_name_map.items():
            if wrapper_inputs.get(wrapper_name) != core_inputs.get(core_name):
                errors.append(f"wrapper/core {wrapper_name} input records differ")

    asset = input_values.get("animal_asset_package")
    request = input_values.get("capture_request")
    room = input_values.get("room_manifest")
    room_request = input_values.get("room_request")
    if asset is not None and request is not None and room is not None:
        identities = {
            value.get("asset_id"),
            core.get("asset_id"),
            asset.get("asset_id"),
            request.get("asset_id"),
        }
        if len(identities) != 1 or None in identities:
            errors.append("wrapper/core/request/package asset identities differ")
        admission_states = {
            value.get("asset_admission_state"),
            core.get("asset_admission_state"),
            asset.get("admission_state"),
        }
        if len(admission_states) != 1 or None in admission_states:
            errors.append("wrapper/core/package admission states differ")
        if isinstance(wrapper_inputs, Mapping) and request.get(
            "asset_manifest_sha256"
        ) != wrapper_inputs["animal_asset_package"].get("sha256"):
            errors.append("capture request does not bind the package bytes")
        request_ids = {
            value.get("request_id"),
            core.get("request_id"),
            request.get("request_id"),
        }
        if len(request_ids) != 1 or None in request_ids:
            errors.append("wrapper/core/request request_ids differ")
        room_ids = {
            value.get("room_id"),
            core.get("room_id"),
            request.get("room_id"),
            room.get("room_id"),
        }
        if room_request is not None:
            room_ids.add(room_request.get("room_id"))
        if len(room_ids) != 1 or None in room_ids:
            errors.append("wrapper/core/request/room room_ids differ")

    if request is not None:
        schedule_errors = validate_variant_review_schedule(request)
        errors.extend(f"capture request schedule: {error}" for error in schedule_errors)
        states = request.get("states")
        frames = core.get("frames")
        if isinstance(states, list) and isinstance(frames, list):
            for index, (state, frame) in enumerate(zip(states, frames, strict=False)):
                if not isinstance(state, Mapping) or not isinstance(frame, Mapping):
                    errors.append(f"core/request frame {index} is invalid")
                    break
                for field in (
                    "frame_index",
                    "action_id",
                    "pts_ticks",
                    "action_time_ticks",
                ):
                    if field in state and frame.get(field) != state.get(field):
                        errors.append(f"core/request frame {index} {field} differs")
                        break

    timeline = value.get("timeline")
    if timeline != _expected_review_timeline():
        errors.append("wrapper timeline contract differs")
    runtime_application = core.get("runtime_application")
    capture_policy = request.get("capture_policy") if request is not None else None
    if not isinstance(runtime_application, Mapping):
        errors.append("core runtime_application is invalid")
    else:
        if value.get("world_time_seconds") != [
            runtime_application.get("initial_world_time_seconds"),
            runtime_application.get("final_world_time_seconds"),
        ]:
            errors.append("wrapper/core world time differs")
        if isinstance(timeline, Mapping) and timeline.get(
            "state_evaluation"
        ) != runtime_application.get("state_evaluation"):
            errors.append("wrapper/core state evaluation differs")
    if isinstance(capture_policy, Mapping) and isinstance(timeline, Mapping):
        if timeline.get("state_evaluation") != capture_policy.get(
            "state_evaluation"
        ) or timeline.get("free_running_animation") != capture_policy.get(
            "free_running_animation"
        ):
            errors.append("wrapper/request timeline policy differs")
    if value.get("runtime_identity") != core.get("runtime_identity"):
        errors.append("wrapper/core runtime identity differs")

    view_contract = value.get("view_contract")
    expected_view_contract = {
        "view_ids": ["view0"],
        "camera_rig_id": "camera_rig_0",
        "modalities": list(FORMAL_MODALITIES),
        "co_located_modalities": True,
        "camera_count": 1,
    }
    if view_contract != expected_view_contract:
        errors.append("wrapper view contract differs")
    sensor = core.get("sensor_contract")
    if not isinstance(sensor, Mapping):
        errors.append("core sensor_contract is invalid")
    else:
        if (
            sensor.get("rig_id") != expected_view_contract["camera_rig_id"]
            or sensor.get("view_id") != "view0"
            or core.get("review_view_ids") != expected_view_contract["view_ids"]
            or core.get("review_modalities") != expected_view_contract["modalities"]
        ):
            errors.append("wrapper/core view contract differs")
        if request is not None and (
            request.get("camera_rig_id") != sensor.get("rig_id")
            or request.get("view_ids") != [sensor.get("view_id")]
            or request.get("modalities") != list(FORMAL_MODALITIES)
        ):
            errors.append("core/request sensor contract differs")
        primary_rig = (
            room_request.get("primary_camera_rig") if room_request is not None else None
        )
        if isinstance(primary_rig, Mapping):
            modality_records = primary_rig.get("modalities")
            modality_map = (
                {
                    item.get("modality"): item.get("sensor_uuid")
                    for item in modality_records
                    if isinstance(item, Mapping)
                }
                if isinstance(modality_records, list)
                else None
            )
            if (
                sensor.get("rig_id") != primary_rig.get("rig_id")
                or sensor.get("view_id") != primary_rig.get("view_id")
                or sensor.get("world_from_rig") != primary_rig.get("world_from_rig")
                or sensor.get("shared_calibration")
                != primary_rig.get("shared_calibration")
                or sensor.get("modality_to_sensor_uuid") != modality_map
            ):
                errors.append("core/room-request sensor contract differs")

    core_videos = core.get("review_media")
    core_videos = (
        core_videos.get("videos") if isinstance(core_videos, Mapping) else None
    )
    core_arrays = core.get("array_artifacts")
    review_videos = artifact_collections.get("review_videos")
    array_artifacts = artifact_collections.get("array_artifacts")
    if isinstance(review_videos, Mapping) and isinstance(core_videos, Mapping):
        for modality in FORMAL_MODALITIES:
            core_video = core_videos.get(modality)
            core_artifact = (
                core_video.get("artifact") if isinstance(core_video, Mapping) else None
            )
            if review_videos.get(modality) != core_artifact:
                errors.append(f"wrapper/core {modality} review video differs")
        if value.get("rgb_review_video") != review_videos.get("rgb"):
            errors.append("rgb_review_video differs from review_videos.rgb")
    if isinstance(array_artifacts, Mapping) and isinstance(core_arrays, Mapping):
        for modality in FORMAL_MODALITIES:
            core_array = core_arrays.get(modality)
            core_artifact = (
                core_array.get("artifact") if isinstance(core_array, Mapping) else None
            )
            if array_artifacts.get(modality) != core_artifact:
                errors.append(f"wrapper/core {modality} array artifact differs")
    return errors


def capture_variant_review(
    inputs: ValidatedM2Inputs,
    room_inputs: ValidatedM1Inputs,
    output_dir: str | Path,
    *,
    runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    """Capture one no-overwrite, review-only animal variant in Habitat."""

    output = Path(output_dir).resolve()
    raw_output = Path(output_dir)
    if raw_output.exists() or raw_output.is_symlink():
        raise VariantReviewError(f"refusing to replace capture output: {output}")
    inputs, room_inputs = _reload_variant_context(inputs, room_inputs)
    try:
        core = _capture_m2_states(
            inputs,
            room_inputs,
            output,
            runtime_root=runtime_root,
            review_only=True,
        )
    except HabitatCaptureError as exc:
        raise VariantReviewError(f"variant Habitat capture failed: {exc}") from exc
    _assert_core_review_evidence(core)
    wrapper = _wrapper_evidence(
        inputs=inputs,
        room_inputs=room_inputs,
        output=output,
        core=core,
    )
    evidence_path = write_json_exclusive(
        output / VARIANT_REVIEW_EVIDENCE_FILENAME,
        wrapper,
    )
    verification_errors = verify_variant_review_evidence(evidence_path)
    if verification_errors:
        raise VariantReviewError("; ".join(verification_errors))
    return wrapper


__all__ = [
    "ALLOWED_REVIEW_ADMISSION_STATES",
    "ROOM_PRESETS",
    "VARIANT_REVIEW_EVIDENCE_FILENAME",
    "VARIANT_REVIEW_EVIDENCE_SCHEMA",
    "VariantReviewError",
    "VariantReviewRoomPreset",
    "build_variant_review_request",
    "capture_variant_review",
    "load_trajectory",
    "load_variant_review_inputs",
    "resolve_room_preset",
    "validate_variant_review_schedule",
    "verify_variant_review_evidence",
    "write_json_exclusive",
]
