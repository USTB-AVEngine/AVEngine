"""Deterministic compiler for an M2 research-candidate animal package.

This module is deliberately downstream of the M2 GLB, action, Habitat and QA
boundaries.  It does not repair an asset, infer anchors, rerun QA, or promote a
candidate.  It verifies hash-bound evidence, emits the canonical package
layout, and always leaves human review unexecuted.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256
from avengine.assets.actions import (
    BakedActionSet,
    baked_actions_content_sha256,
    read_baked_actions_npz,
)
from avengine.assets.contracts import (
    ANIMAL_SCHEMA,
    CONTACT_ORDER,
    REQUIRED_FILE_ROLES,
    validate_animal_asset_package,
)
from avengine.assets.glb import GlbDocument, decode_accessor, load_glb
from avengine.assets.habitat import (
    HabitatAssetMapping,
    HabitatLinkJointBlock,
    bind_habitat_link_layout,
    build_habitat_ao_config_data,
    build_habitat_asset_mapping_from_rebase_report,
)


REBASE_REPORT_SCHEMA = "avengine_m2_skin_root_rebase_v1"
REBASE_DEFORMATION_REPORT_SCHEMA = "avengine_m2_rebase_deformation_verification_v1"
ACTION_REPORT_SCHEMA = "avengine_m2_action_bake_report_v1"
HABITAT_STATIC_PROBE_SCHEMA = "avengine_m2_habitat_skin_rest_probe_v1"
HABITAT_ANIMATION_REVIEW_SCHEMA = "avengine_m2_habitat_action_review_v1"
STATIC_QA_SCHEMA = "avengine_m2_static_geometry_qa_v1"
DEFORMATION_QA_SCHEMA = "avengine_m2_deformation_qa_v1"
ANIMATION_QA_SCHEMA = "avengine_m2_animation_qa_v1"
CONTACT_PHASES_SCHEMA = "avengine_m2_contact_phases_v1"

_JSON_CHUNK_TYPE = 0x4E4F534A
_BIN_CHUNK_TYPE = 0x004E4942
_IDENTITY_QUATERNION = [0.0, 0.0, 0.0, 1.0]
_ALLOWED_USE = {"research_canary", "research_only", "review_required"}
_REDISTRIBUTION = {"allowed", "prohibited", "review_required"}
_REQUIRED_STATIC_GATES = {
    "bootstrap_visible",
    "all_six_orbit_views_visible",
    "co_located_modalities",
    "runtime_joint_mapping_complete",
    "runtime_link_bind_alignment",
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_REPORT_PRODUCER_BY_OWNER = {
    "habitat_static_probe": _REPOSITORY_ROOT / "tools/assets/probe_habitat_skin_rest.py",
    "habitat_animation_review": _REPOSITORY_ROOT
    / "tools/assets/render_habitat_action_review.py",
}
_LOCAL_RUNTIME_EVIDENCE_SCOPE = {
    "local_report_claim": "artifact_integrity_only",
    "trusted_runtime_attestation": False,
    "runtime_execution_conclusion_source": "external_capture_audit_only",
}
_STATIC_PROBE_OBSERVATION_PATHS = (
    "qa_bootstrap_rgb.png",
    "qa_x_negative_rgb.png",
    "qa_x_positive_rgb.png",
    "qa_y_negative_rgb.png",
    "qa_y_positive_rgb.png",
    "qa_z_negative_rgb.png",
    "qa_z_positive_rgb.png",
    "rest_depth.png",
    "rest_rgb.png",
    "rest_semantic.png",
)

_ROLE_PATHS: tuple[tuple[str, str], ...] = (
    ("visual", "visual.glb"),
    ("collision_proxy", "collision_proxy.glb"),
    ("skeleton_manifest", "skeleton.json"),
    ("skinning_manifest", "skinning_manifest.json"),
    ("emitter_anchors", "emitter_anchors.json"),
    ("action_manifest", "actions/action_manifest.json"),
    ("idle_poses", "actions/idle.npz"),
    ("walk_poses", "actions/walk.npz"),
    ("contact_phases", "contacts/contact_phases.json"),
    ("static_geometry_qa", "qa/static_geometry.json"),
    ("deformation_qa", "qa/deformation.json"),
    ("animation_qa", "qa/animation.json"),
    ("provenance_manifest", "provenance_manifest.json"),
    ("habitat_urdf", "habitat/animal.urdf"),
    ("habitat_ao_config", "habitat/animal.ao_config.json"),
    ("habitat_joint_mapping", "habitat/joint_mapping.json"),
)


class PackageCompileError(ValueError):
    """An input or output violates the research package boundary."""


@dataclass(frozen=True)
class AnimalPackageIdentity:
    """Explicit stable identity and policy fields not inferred from geometry."""

    asset_id: str
    template_id: str
    body_plan_id: str
    morphotype_id: str
    skeleton_revision: str
    weights_revision: str
    collision_revision: str
    action_revision: str
    source: str
    source_revision: str
    license: str
    allowed_use: str
    redistribution: str
    semantic_id: int = 200

    def __post_init__(self) -> None:
        fields = (
            "asset_id",
            "template_id",
            "body_plan_id",
            "morphotype_id",
            "skeleton_revision",
            "weights_revision",
            "collision_revision",
            "action_revision",
            "source",
            "source_revision",
            "license",
        )
        for field_name in fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise PackageCompileError(f"identity.{field_name} must be non-empty")
        if self.allowed_use not in _ALLOWED_USE:
            raise PackageCompileError(
                f"identity.allowed_use must be one of {sorted(_ALLOWED_USE)}"
            )
        if self.redistribution not in _REDISTRIBUTION:
            raise PackageCompileError(
                f"identity.redistribution must be one of {sorted(_REDISTRIBUTION)}"
            )
        if (
            isinstance(self.semantic_id, bool)
            or not isinstance(self.semantic_id, int)
            or self.semantic_id < 0
        ):
            raise PackageCompileError("identity.semantic_id must be non-negative")


@dataclass(frozen=True)
class _JsonInput:
    label: str
    path: Path
    payload: bytes
    value: dict[str, Any]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def byte_size(self) -> int:
        return len(self.payload)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PackageCompileError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PackageCompileError(f"JSON contains non-finite number {value}")


def _absolute_without_symlinks(path: str | Path, *, owner: str) -> Path:
    raw = Path(path)
    absolute = Path(os.path.abspath(raw))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise PackageCompileError(f"{owner} must not contain a symbolic link")
    return absolute


def _read_regular(path: str | Path, *, owner: str) -> tuple[Path, bytes]:
    absolute = _absolute_without_symlinks(path, owner=owner)
    if not absolute.is_file():
        raise PackageCompileError(f"{owner} must be an existing regular file")
    try:
        payload = absolute.read_bytes()
    except OSError as exc:
        raise PackageCompileError(f"unable to read {owner}: {exc}") from exc
    return absolute, payload


def _read_json(path: str | Path, *, owner: str) -> _JsonInput:
    absolute, payload = _read_regular(path, owner=owner)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise PackageCompileError(f"{owner} must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise PackageCompileError(f"{owner} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PackageCompileError(f"{owner} must contain one JSON object")
    return _JsonInput(owner, absolute, payload, value)


def _json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PackageCompileError(f"unable to encode package JSON: {exc}") from exc


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PackageCompileError(f"{owner} must be an object")
    return value


def _sequence(value: Any, *, owner: str) -> list[Any]:
    if not isinstance(value, list):
        raise PackageCompileError(f"{owner} must be an array")
    return value


def _finite_number(value: Any, *, owner: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PackageCompileError(f"{owner} must be a finite number")
    return float(value)


def _positive_threshold(value: Any, *, owner: str) -> float:
    number = _finite_number(value, owner=owner)
    if number <= 0.0:
        raise PackageCompileError(f"{owner} must be positive")
    return number


def _nonnegative_number(value: Any, *, owner: str) -> float:
    number = _finite_number(value, owner=owner)
    if number < 0.0:
        raise PackageCompileError(f"{owner} must be non-negative")
    return number


def _require_reference(
    value: Any,
    *,
    owner: str,
    sha256: str,
    byte_size: int | None = None,
    require_byte_size: bool = True,
) -> Mapping[str, Any]:
    reference = _mapping(value, owner=owner)
    if reference.get("sha256") != sha256:
        raise PackageCompileError(f"{owner}.sha256 does not match its input")
    if byte_size is not None:
        if "byte_size" not in reference:
            if require_byte_size:
                raise PackageCompileError(f"{owner}.byte_size is required")
        elif reference.get("byte_size") != byte_size:
            raise PackageCompileError(f"{owner}.byte_size does not match its input")
    return reference


def _require_pass_report(report: _JsonInput, *, schema: str) -> None:
    value = report.value
    if value.get("schema") != schema:
        raise PackageCompileError(f"{report.label} schema must be {schema!r}")
    if value.get("status") != "pass":
        raise PackageCompileError(f"{report.label} status must be 'pass'")
    if value.get("qualification_claim") is not False:
        raise PackageCompileError(f"{report.label} must not claim asset qualification")
    state = value.get("qualification_state")
    if state is not None and state != "research_candidate":
        raise PackageCompileError(
            f"{report.label}.qualification_state must be research_candidate"
        )


def _validate_rebase_report(
    report: _JsonInput, *, visual_sha256: str, visual_size: int
) -> None:
    _require_pass_report(report, schema=REBASE_REPORT_SCHEMA)
    _require_reference(
        report.value.get("output"),
        owner="rebase_report.output",
        sha256=visual_sha256,
        byte_size=visual_size,
    )
    skin = _mapping(report.value.get("skin"), owner="rebase_report.skin")
    if not isinstance(skin.get("root_joint"), str) or not skin["root_joint"]:
        raise PackageCompileError("rebase_report.skin.root_joint must be non-empty")
    if "actor_from_canonical_root" not in skin:
        raise PackageCompileError(
            "rebase_report.skin.actor_from_canonical_root is required"
        )


def _validate_rebase_deformation_report(
    report: _JsonInput,
    *,
    visual_sha256: str,
    visual_size: int,
    rebase_report: _JsonInput,
) -> None:
    _require_pass_report(report, schema=REBASE_DEFORMATION_REPORT_SCHEMA)
    _require_reference(
        report.value.get("rebased"),
        owner="deformation_report.rebased",
        sha256=visual_sha256,
        byte_size=visual_size,
        require_byte_size=False,
    )
    _require_reference(
        report.value.get("rebase_report"),
        owner="deformation_report.rebase_report",
        sha256=rebase_report.sha256,
        byte_size=rebase_report.byte_size,
        require_byte_size=False,
    )
    error = report.value.get("maximum_vertex_error_m")
    threshold = report.value.get("threshold_maximum_vertex_error_m")
    if not (
        isinstance(error, (int, float))
        and not isinstance(error, bool)
        and math.isfinite(float(error))
        and isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and math.isfinite(float(threshold))
        and 0.0 <= float(error) <= float(threshold)
    ):
        raise PackageCompileError(
            "deformation_report maximum vertex error must meet its threshold"
        )
    samples = _sequence(report.value.get("samples"), owner="deformation_report.samples")
    semantics = {
        sample.get("semantic") for sample in samples if isinstance(sample, dict)
    }
    if semantics != {"idle", "walk"}:
        raise PackageCompileError(
            "deformation_report samples must cover exactly idle and walk"
        )


def _action_by_id(actions: BakedActionSet) -> dict[str, Any]:
    return {action.semantic_action_id: action for action in actions.actions}


def _validate_action_report(
    report: _JsonInput,
    *,
    visual_sha256: str,
    visual_size: int,
    actions_sha256: str,
    actions_size: int,
    actions: BakedActionSet,
) -> None:
    _require_pass_report(report, schema=ACTION_REPORT_SCHEMA)
    _require_reference(
        report.value.get("source_glb"),
        owner="action_report.source_glb",
        sha256=visual_sha256,
        byte_size=visual_size,
    )
    artifact = _require_reference(
        report.value.get("artifact"),
        owner="action_report.artifact",
        sha256=actions_sha256,
        byte_size=actions_size,
    )
    if (
        artifact.get("canonical_content_sha256") != actions_sha256
        or artifact.get("readback_equal") is not True
    ):
        raise PackageCompileError(
            "action_report must bind canonical/read-back-equal baked actions"
        )
    if report.value.get("runtime_joint_order") != list(actions.runtime_joint_order):
        raise PackageCompileError(
            "action_report runtime_joint_order differs from baked actions"
        )
    expected = _action_by_id(actions)
    reported = _sequence(report.value.get("actions"), owner="action_report.actions")
    if len(reported) != 2:
        raise PackageCompileError("action_report must contain exactly idle and walk")
    seen: set[str] = set()
    for index, item in enumerate(reported):
        record = _mapping(item, owner=f"action_report.actions[{index}]")
        action_id = record.get("semantic_action_id")
        if action_id not in expected or action_id in seen:
            raise PackageCompileError("action_report action mapping is not canonical")
        seen.add(action_id)
        clip = expected[action_id]
        if (
            record.get("source_action_name") != clip.source_action_name
            or record.get("sample_count") != clip.sample_count
            or record.get("loop_duration_ticks") != clip.loop_duration_ticks
        ):
            raise PackageCompileError(
                f"action_report {action_id} metadata differs from baked actions"
            )
    if seen != {"idle", "walk"}:
        raise PackageCompileError("action_report must cover idle and walk")


def _validate_habitat_static_probe(
    report: _JsonInput,
    *,
    visual_sha256: str,
    visual_size: int,
    shader_type: str,
    semantic_id: int,
    runtime_joint_order: Sequence[str],
) -> None:
    _require_pass_report(report, schema=HABITAT_STATIC_PROBE_SCHEMA)
    _validate_local_runtime_artifact_integrity(
        report,
        expected=shader_type,
        semantic_id=semantic_id,
        owner="habitat_static_probe",
        runtime_joint_order=runtime_joint_order,
    )
    _require_reference(
        report.value.get("input"),
        owner="static_qa.input",
        sha256=visual_sha256,
        byte_size=visual_size,
    )
    gates = _mapping(report.value.get("gates"), owner="static_qa.gates")
    if not _REQUIRED_STATIC_GATES.issubset(gates):
        missing = sorted(_REQUIRED_STATIC_GATES - set(gates))
        raise PackageCompileError(f"static_qa is missing gates: {missing}")
    if any(gates[name] is not True for name in _REQUIRED_STATIC_GATES):
        raise PackageCompileError("static_qa cannot report pass with a failed gate")


def _validate_habitat_animation_review(
    report: _JsonInput,
    *,
    visual_sha256: str,
    visual_size: int,
    actions_sha256: str,
    actions_size: int,
    rebase_report: _JsonInput,
    actions: BakedActionSet,
    shader_type: str,
    semantic_id: int,
) -> None:
    _require_pass_report(report, schema=HABITAT_ANIMATION_REVIEW_SCHEMA)
    _validate_local_runtime_artifact_integrity(
        report,
        expected=shader_type,
        semantic_id=semantic_id,
        owner="habitat_animation_review",
        runtime_joint_order=actions.runtime_joint_order,
    )
    source = _mapping(report.value.get("source"), owner="animation_qa.source")
    _require_reference(
        source.get("visual_glb"),
        owner="animation_qa.source.visual_glb",
        sha256=visual_sha256,
        byte_size=visual_size,
    )
    _require_reference(
        source.get("actions_npz"),
        owner="animation_qa.source.actions_npz",
        sha256=actions_sha256,
        byte_size=actions_size,
    )
    _require_reference(
        source.get("rebase_report"),
        owner="animation_qa.source.rebase_report",
        sha256=rebase_report.sha256,
        byte_size=rebase_report.byte_size,
    )
    runtime = _mapping(
        report.value.get("runtime_contract"), owner="animation_qa.runtime_contract"
    )
    if runtime.get("runtime_joint_order") != list(actions.runtime_joint_order):
        raise PackageCompileError(
            "animation_qa runtime_joint_order differs from baked actions"
        )
    capture = _mapping(
        report.value.get("capture_contract"), owner="animation_qa.capture_contract"
    )
    if (
        capture.get("formal_capture") is not False
        or capture.get("co_located_and_co_oriented") is not True
        or capture.get("world_time_unchanged") is not True
    ):
        raise PackageCompileError("animation_qa capture contract is not deterministic")
    expected = _action_by_id(actions)
    runs = _sequence(report.value.get("runs"), owner="animation_qa.runs")
    seen: set[str] = set()
    if not runs:
        raise PackageCompileError("animation_qa must contain review runs")
    for index, item in enumerate(runs):
        run = _mapping(item, owner=f"animation_qa.runs[{index}]")
        action_id = run.get("semantic_action_id")
        if action_id not in expected:
            raise PackageCompileError("animation_qa contains an unknown action")
        clip = expected[action_id]
        if (
            run.get("source_action_name") != clip.source_action_name
            or run.get("sample_count") != clip.sample_count
            or run.get("all_frames_visible") is not True
            or not isinstance(run.get("minimum_semantic_pixel_count"), int)
            or run["minimum_semantic_pixel_count"] <= 0
        ):
            raise PackageCompileError(
                f"animation_qa run {index} does not bind a visible complete clip"
            )
        seen.add(action_id)
    if seen != {"idle", "walk"}:
        raise PackageCompileError("animation_qa runs must cover idle and walk")


def _read_report_relative_artifact(
    report: _JsonInput,
    value: Any,
    *,
    owner: str,
    snapshot: bool,
) -> tuple[Mapping[str, Any], bytes, dict[str, Any] | None]:
    """Read and hash-close one real artifact below a report's directory."""

    record = _mapping(value, owner=owner)
    expected_fields = {"path", "byte_size", "sha256"}
    if snapshot:
        expected_fields.add("snapshot")
    if set(record) != expected_fields:
        raise PackageCompileError(f"{owner} fields are invalid")
    relative_text = record.get("path")
    if not isinstance(relative_text, str) or not relative_text or "\\" in relative_text:
        raise PackageCompileError(f"{owner}.path must be a POSIX relative path")
    relative = Path(relative_text)
    if (
        relative.is_absolute()
        or relative == Path(".")
        or ".." in relative.parts
        or relative.as_posix() != relative_text
    ):
        raise PackageCompileError(f"{owner}.path must be a canonical relative path")
    artifact_path = report.path.parent / relative
    _, payload = _read_regular(artifact_path, owner=owner)
    if (
        record.get("byte_size") != len(payload)
        or record.get("sha256") != hashlib.sha256(payload).hexdigest()
    ):
        raise PackageCompileError(f"{owner} does not hash-bind its real artifact")
    parsed: dict[str, Any] | None = None
    if snapshot:
        parsed_input = _read_json(artifact_path, owner=owner)
        parsed = parsed_input.value
        if record.get("snapshot") != parsed:
            raise PackageCompileError(
                f"{owner}.snapshot differs from its real JSON artifact"
            )
    return record, payload, parsed


def _validate_runtime_binding_snapshot(
    value: Mapping[str, Any],
    *,
    runtime_joint_order: Sequence[str],
    owner: str,
) -> None:
    links = _sequence(value.get("links"), owner=f"{owner}.links")
    blocks: list[HabitatLinkJointBlock] = []
    for index, item in enumerate(links):
        link = _mapping(item, owner=f"{owner}.links[{index}]")
        if set(link) != {
            "link_name",
            "joint_position_offset",
            "joint_position_count",
        }:
            raise PackageCompileError(f"{owner}.links[{index}] fields are invalid")
        blocks.append(
            HabitatLinkJointBlock(
                link_name=link.get("link_name"),
                joint_position_offset=link.get("joint_position_offset"),
                joint_position_count=link.get("joint_position_count"),
            )
        )
    joint_position_count = value.get("joint_position_count")
    try:
        binding = bind_habitat_link_layout(
            runtime_joint_order,
            blocks,
            joint_position_count=joint_position_count,
        )
    except (TypeError, ValueError) as exc:
        raise PackageCompileError(
            f"{owner} is not a valid Habitat binding: {exc}"
        ) from exc
    if binding.to_json_data() != value:
        raise PackageCompileError(
            f"{owner} differs from the canonical package runtime binding"
        )


def _validate_local_runtime_artifact_integrity(
    report: _JsonInput,
    *,
    expected: str,
    semantic_id: int,
    owner: str,
    runtime_joint_order: Sequence[str],
) -> None:
    """Validate local file integrity without treating it as runtime attestation.

    These fields prove that a report, its AO configuration, its measured binding,
    and its observation files form one hash-closed local bundle.  Because the
    bundle and report share the same untrusted filesystem, this deliberately
    makes no claim that Habitat executed those bytes.  Runtime conclusions come
    only from a separately retained capture/audit boundary.
    """

    configuration_value = report.value.get("render_configuration_integrity")
    if configuration_value is None and expected == "phong":
        # Pre-integrity-contract Phong reports remain valid for the formal M2
        # baseline.  They cannot be relabelled as PBR because PBR requires every
        # field and real artifact below.
        return
    scope = _mapping(
        report.value.get("evidence_scope"), owner=f"{owner}.evidence_scope"
    )
    if scope != _LOCAL_RUNTIME_EVIDENCE_SCOPE:
        raise PackageCompileError(
            f"{owner}.evidence_scope must declare local integrity without trusted "
            "runtime attestation"
        )
    configuration = _mapping(
        configuration_value,
        owner=f"{owner}.render_configuration_integrity",
    )
    if set(configuration) != {"configured_shader_type", "ao_config_artifact"}:
        raise PackageCompileError(
            f"{owner}.render_configuration_integrity fields are invalid"
        )
    if configuration.get("configured_shader_type") != expected:
        raise PackageCompileError(
            f"{owner} AO configuration must bind shader_type {expected!r}"
        )
    expected_config = build_habitat_ao_config_data(
        render_asset="visual.glb",
        urdf_filepath="animal.urdf",
        semantic_id=semantic_id,
        shader_type=expected,
    )
    _, _, config_snapshot = _read_report_relative_artifact(
        report,
        configuration.get("ao_config_artifact"),
        owner=f"{owner}.render_configuration_integrity.ao_config_artifact",
        snapshot=True,
    )
    if config_snapshot != expected_config:
        raise PackageCompileError(
            f"{owner} real AO config does not match the requested package shader"
        )
    producer = _mapping(
        report.value.get("producer_source_integrity"),
        owner=f"{owner}.producer_source_integrity",
    )
    if set(producer) != {"path", "byte_size", "sha256"}:
        raise PackageCompileError(f"{owner} producer source fields are invalid")
    producer_path = _REPORT_PRODUCER_BY_OWNER[owner]
    producer_payload = producer_path.read_bytes()
    reported_path = producer.get("path")
    if (
        not isinstance(reported_path, str)
        or not reported_path.replace("\\", "/").endswith(
            f"tools/assets/{producer_path.name}"
        )
        or producer.get("byte_size") != len(producer_payload)
        or producer.get("sha256") != hashlib.sha256(producer_payload).hexdigest()
    ):
        raise PackageCompileError(
            f"{owner} does not bind its current producer source bytes"
        )

    artifacts = _mapping(
        report.value.get("runtime_artifact_integrity"),
        owner=f"{owner}.runtime_artifact_integrity",
    )
    if set(artifacts) != {"runtime_binding_artifact", "observation_artifacts"}:
        raise PackageCompileError(f"{owner} runtime artifact fields are invalid")
    _, _, binding_snapshot = _read_report_relative_artifact(
        report,
        artifacts.get("runtime_binding_artifact"),
        owner=f"{owner}.runtime_artifact_integrity.runtime_binding_artifact",
        snapshot=True,
    )
    assert binding_snapshot is not None
    _validate_runtime_binding_snapshot(
        binding_snapshot,
        runtime_joint_order=runtime_joint_order,
        owner=f"{owner}.runtime_binding",
    )

    observation_values = _sequence(
        artifacts.get("observation_artifacts"),
        owner=f"{owner}.runtime_artifact_integrity.observation_artifacts",
    )
    if owner == "habitat_static_probe":
        observation_paths = [
            _mapping(value, owner=f"{owner}.observation_artifacts[{index}]").get("path")
            for index, value in enumerate(observation_values)
        ]
        observation_names = [
            Path(path).name if isinstance(path, str) else None
            for path in observation_paths
        ]
        if observation_names != list(_STATIC_PROBE_OBSERVATION_PATHS):
            raise PackageCompileError(
                f"{owner} must bind the complete deterministic observation set"
            )
    else:
        expected_observations: list[Any] = []
        for index, value in enumerate(
            _sequence(report.value.get("runs"), owner=f"{owner}.runs")
        ):
            run = _mapping(value, owner=f"{owner}.runs[{index}]")
            expected_observations.extend([run.get("video"), run.get("contact_sheet")])
        if observation_values != expected_observations:
            raise PackageCompileError(
                f"{owner} observation artifacts differ from its review runs"
            )
    seen_paths: set[str] = set()
    for index, value in enumerate(observation_values):
        record, _, _ = _read_report_relative_artifact(
            report,
            value,
            owner=f"{owner}.runtime_artifact_integrity.observation_artifacts[{index}]",
            snapshot=False,
        )
        path = record["path"]
        if path in seen_paths:
            raise PackageCompileError(f"{owner} observation artifact paths repeat")
        seen_paths.add(path)


def _validate_static_qa(
    report: _JsonInput,
    *,
    visual_sha256: str,
    topology_sha256: str,
    uv_sha256: str,
    weights_sha256: str,
    joint_count: int,
) -> None:
    _require_pass_report(report, schema=STATIC_QA_SCHEMA)
    value = report.value
    if value.get("source_glb_sha256") != visual_sha256:
        raise PackageCompileError("static_qa source GLB hash does not match visual")
    expected_hashes = {
        "topology_sha256": topology_sha256,
        "uv_sha256": uv_sha256,
        "weights_sha256": weights_sha256,
    }
    for field_name, expected_hash in expected_hashes.items():
        if value.get(field_name) != expected_hash:
            raise PackageCompileError(
                f"static_qa.{field_name} differs from independent GLB evidence"
            )
    if value.get("joint_count") != joint_count:
        raise PackageCompileError("static_qa joint_count differs from the GLB skin")

    thresholds = _mapping(value.get("thresholds"), owner="static_qa.thresholds")
    maximum_weight_sum_error = _positive_threshold(
        thresholds.get("maximum_weight_sum_error"),
        owner="static_qa.thresholds.maximum_weight_sum_error",
    )
    maximum_bind_closure_error = _positive_threshold(
        thresholds.get("maximum_bind_closure_error_m"),
        owner="static_qa.thresholds.maximum_bind_closure_error_m",
    )
    minimum_triangle_area = _positive_threshold(
        thresholds.get("minimum_triangle_area_m2_exclusive"),
        owner="static_qa.thresholds.minimum_triangle_area_m2_exclusive",
    )
    maximum_landmark_outside = _positive_threshold(
        thresholds.get("maximum_landmark_bbox_outside_distance_m"),
        owner=("static_qa.thresholds.maximum_landmark_bbox_outside_distance_m"),
    )
    if (
        _nonnegative_number(
            value.get("maximum_bind_closure_error"),
            owner="static_qa.maximum_bind_closure_error",
        )
        > maximum_bind_closure_error
    ):
        raise PackageCompileError("static_qa bind closure exceeds its threshold")
    if (
        _nonnegative_number(
            value.get("maximum_rest_landmark_bbox_outside_distance_m"),
            owner="static_qa.maximum_rest_landmark_bbox_outside_distance_m",
        )
        > maximum_landmark_outside
    ):
        raise PackageCompileError("static_qa rest landmarks exceed their threshold")

    primitives = _sequence(value.get("primitives"), owner="static_qa.primitives")
    if not primitives or value.get("primitive_count") != len(primitives):
        raise PackageCompileError("static_qa primitive_count is invalid")
    for index, primitive_value in enumerate(primitives):
        primitive = _mapping(primitive_value, owner=f"static_qa.primitives[{index}]")
        if (
            primitive.get("primitive_index") != index
            or isinstance(primitive.get("vertex_count"), bool)
            or not isinstance(primitive.get("vertex_count"), int)
            or primitive["vertex_count"] <= 0
            or isinstance(primitive.get("triangle_count"), bool)
            or not isinstance(primitive.get("triangle_count"), int)
            or primitive["triangle_count"] <= 0
        ):
            raise PackageCompileError(
                f"static_qa primitive {index} identity/counts are invalid"
            )
        if (
            _nonnegative_number(
                primitive.get("minimum_triangle_area_m2"),
                owner=f"static_qa.primitives[{index}].minimum_triangle_area_m2",
            )
            <= minimum_triangle_area
        ):
            raise PackageCompileError(
                f"static_qa primitive {index} contains collapsed geometry"
            )
        if (
            _nonnegative_number(
                primitive.get("maximum_weight_sum_error"),
                owner=f"static_qa.primitives[{index}].maximum_weight_sum_error",
            )
            > maximum_weight_sum_error
        ):
            raise PackageCompileError(
                f"static_qa primitive {index} weight sum exceeds its threshold"
            )
        if (
            _nonnegative_number(
                primitive.get("maximum_weighted_bind_vertex_error_m"),
                owner=(
                    f"static_qa.primitives[{index}]"
                    ".maximum_weighted_bind_vertex_error_m"
                ),
            )
            > maximum_bind_closure_error
        ):
            raise PackageCompileError(
                f"static_qa primitive {index} bind error exceeds its threshold"
            )


def _validate_deformation_qa(
    report: _JsonInput,
    *,
    visual_sha256: str,
    actions_sha256: str,
    actions: BakedActionSet,
) -> None:
    _require_pass_report(report, schema=DEFORMATION_QA_SCHEMA)
    value = report.value
    if value.get("source_glb_sha256") != visual_sha256:
        raise PackageCompileError(
            "deformation_qa source GLB hash does not match visual"
        )
    if value.get("baked_actions_sha256") != actions_sha256:
        raise PackageCompileError(
            "deformation_qa baked_actions_sha256 differs from canonical actions"
        )
    if baked_actions_content_sha256(actions) != actions_sha256:
        raise PackageCompileError("canonical baked action identity is inconsistent")

    thresholds = _mapping(value.get("thresholds"), owner="deformation_qa.thresholds")
    maximum_step_ratio = _positive_threshold(
        thresholds.get("maximum_vertex_step_rest_diagonal_ratio"),
        owner=("deformation_qa.thresholds.maximum_vertex_step_rest_diagonal_ratio"),
    )
    maximum_vertex_endpoint_error = _positive_threshold(
        thresholds.get("maximum_source_loop_endpoint_vertex_error_m"),
        owner=("deformation_qa.thresholds.maximum_source_loop_endpoint_vertex_error_m"),
    )
    maximum_translation_endpoint_error = _positive_threshold(
        thresholds.get("maximum_source_loop_endpoint_joint_translation_error_m"),
        owner=(
            "deformation_qa.thresholds."
            "maximum_source_loop_endpoint_joint_translation_error_m"
        ),
    )
    maximum_rotation_endpoint_error = _positive_threshold(
        thresholds.get("maximum_source_loop_endpoint_joint_rotation_error"),
        owner=(
            "deformation_qa.thresholds."
            "maximum_source_loop_endpoint_joint_rotation_error"
        ),
    )
    maximum_scale_endpoint_error = _positive_threshold(
        thresholds.get("maximum_source_loop_endpoint_joint_scale_error"),
        owner=(
            "deformation_qa.thresholds.maximum_source_loop_endpoint_joint_scale_error"
        ),
    )
    minimum_triangle_area = _positive_threshold(
        thresholds.get("minimum_triangle_area_m2_exclusive"),
        owner="deformation_qa.thresholds.minimum_triangle_area_m2_exclusive",
    )
    maximum_landmark_outside = _positive_threshold(
        thresholds.get("maximum_landmark_bbox_outside_distance_m"),
        owner=("deformation_qa.thresholds.maximum_landmark_bbox_outside_distance_m"),
    )
    rest_diagonal = _finite_number(
        value.get("rest_bbox_diagonal_m"),
        owner="deformation_qa.rest_bbox_diagonal_m",
    )
    if rest_diagonal <= 0.0:
        raise PackageCompileError("deformation_qa rest bbox diagonal must be positive")
    if (
        _nonnegative_number(
            value.get("maximum_source_loop_endpoint_vertex_error_m"),
            owner="deformation_qa.maximum_source_loop_endpoint_vertex_error_m",
        )
        > maximum_vertex_endpoint_error
    ):
        raise PackageCompileError(
            "deformation_qa true source endpoint closure exceeds its threshold"
        )
    maximum_vertex_step = _nonnegative_number(
        value.get("maximum_vertex_step_m"),
        owner="deformation_qa.maximum_vertex_step_m",
    )
    if maximum_vertex_step / rest_diagonal > maximum_step_ratio:
        raise PackageCompileError(
            "deformation_qa maximum vertex step exceeds its bbox-relative threshold"
        )
    if (
        _nonnegative_number(
            value.get("minimum_animated_triangle_area_m2"),
            owner="deformation_qa.minimum_animated_triangle_area_m2",
        )
        <= minimum_triangle_area
    ):
        raise PackageCompileError("deformation_qa contains collapsed geometry")
    if (
        _nonnegative_number(
            value.get("maximum_joint_landmark_bbox_outside_distance_m"),
            owner=("deformation_qa.maximum_joint_landmark_bbox_outside_distance_m"),
        )
        > maximum_landmark_outside
    ):
        raise PackageCompileError("deformation_qa landmarks exceed their threshold")

    expected = _action_by_id(actions)
    records = _sequence(value.get("actions"), owner="deformation_qa.actions")
    seen: set[str] = set()
    for index, record_value in enumerate(records):
        record = _mapping(record_value, owner=f"deformation_qa.actions[{index}]")
        action_id = record.get("semantic_action_id")
        if action_id not in expected or action_id in seen:
            raise PackageCompileError("deformation_qa action mapping is not canonical")
        seen.add(action_id)
        clip = expected[action_id]
        if (
            record.get("source_action_name") != clip.source_action_name
            or record.get("sample_count") != clip.sample_count
        ):
            raise PackageCompileError(
                f"deformation_qa {action_id} metadata differs from baked actions"
            )
        checks = (
            (
                "maximum_vertex_step_rest_diagonal_ratio",
                maximum_step_ratio,
            ),
            (
                "source_loop_endpoint_vertex_error_m",
                maximum_vertex_endpoint_error,
            ),
            (
                "source_loop_endpoint_maximum_joint_translation_error_m",
                maximum_translation_endpoint_error,
            ),
            (
                "source_loop_endpoint_maximum_joint_rotation_error",
                maximum_rotation_endpoint_error,
            ),
            (
                "source_loop_endpoint_maximum_joint_scale_error",
                maximum_scale_endpoint_error,
            ),
            (
                "maximum_joint_landmark_bbox_outside_distance_m",
                maximum_landmark_outside,
            ),
        )
        for field_name, threshold in checks:
            if (
                _nonnegative_number(
                    record.get(field_name),
                    owner=f"deformation_qa.actions[{index}].{field_name}",
                )
                > threshold
            ):
                raise PackageCompileError(
                    f"deformation_qa {action_id}.{field_name} exceeds its threshold"
                )
        if (
            _nonnegative_number(
                record.get("minimum_triangle_area_m2"),
                owner=(f"deformation_qa.actions[{index}].minimum_triangle_area_m2"),
            )
            <= minimum_triangle_area
        ):
            raise PackageCompileError(
                f"deformation_qa {action_id} contains collapsed geometry"
            )
    if seen != {"idle", "walk"}:
        raise PackageCompileError("deformation_qa must contain idle and walk")


def _validate_animation_qa(
    report: _JsonInput,
    *,
    visual_sha256: str,
    actions_sha256: str,
    actions: BakedActionSet,
    muzzle_joint_id: str,
) -> None:
    _require_pass_report(report, schema=ANIMATION_QA_SCHEMA)
    value = report.value
    if value.get("source_glb_sha256") != visual_sha256:
        raise PackageCompileError("animation_qa source GLB hash does not match visual")
    if value.get("baked_actions_sha256") != actions_sha256:
        raise PackageCompileError(
            "animation_qa baked_actions_sha256 differs from canonical actions"
        )
    if (
        value.get("sample_rate_hz") != actions.sample_rate_hz
        or value.get("time_base_hz") != actions.time_base_hz
        or value.get("runtime_joint_order") != list(actions.runtime_joint_order)
    ):
        raise PackageCompileError("animation_qa runtime/clock differs from actions")
    expected = _action_by_id(actions)
    records = _sequence(value.get("actions"), owner="animation_qa.actions")
    seen: set[str] = set()
    for index, record_value in enumerate(records):
        record = _mapping(record_value, owner=f"animation_qa.actions[{index}]")
        action_id = record.get("semantic_action_id")
        if action_id not in expected or action_id in seen:
            raise PackageCompileError("animation_qa action mapping is not canonical")
        seen.add(action_id)
        clip = expected[action_id]
        if (
            record.get("source_action_name") != clip.source_action_name
            or record.get("sample_count") != clip.sample_count
            or record.get("loop_duration_ticks") != clip.loop_duration_ticks
            or record.get("first_sample_tick") != clip.sample_ticks[0]
            or record.get("last_sample_tick") != clip.sample_ticks[-1]
        ):
            raise PackageCompileError(
                f"animation_qa {action_id} metadata differs from baked actions"
            )
    if seen != {"idle", "walk"}:
        raise PackageCompileError("animation_qa must contain idle and walk")

    mouth = _mapping(value.get("mouth"), owner="animation_qa.mouth")
    excursions = _mapping(
        mouth.get("rotation_excursion_degrees_by_action"),
        owner="animation_qa.mouth.rotation_excursion_degrees_by_action",
    )
    if (
        mouth.get("joint_id") != muzzle_joint_id
        or mouth.get("open_ratio_policy") != "exactly_zero"
        or set(excursions) != {"idle", "walk"}
        or any(
            _finite_number(
                excursions[action_id],
                owner=f"animation_qa.mouth.{action_id}",
            )
            != 0.0
            for action_id in ("idle", "walk")
        )
        or _finite_number(
            mouth.get("maximum_rotation_excursion_degrees"),
            owner="animation_qa.mouth.maximum_rotation_excursion_degrees",
        )
        != 0.0
        or _positive_threshold(
            mouth.get("threshold_degrees"),
            owner="animation_qa.mouth.threshold_degrees",
        )
        <= 0.0
    ):
        raise PackageCompileError("animation_qa does not prove the M2 mouth=0 policy")
    limitations = _sequence(
        value.get("known_limitations"), owner="animation_qa.known_limitations"
    )
    if (
        any(not isinstance(item, str) or not item.strip() for item in limitations)
        or value.get("human_visual_review_required") is not True
    ):
        raise PackageCompileError(
            "animation_qa must use valid known limitations and require human review"
        )
    terminal_motion = _mapping(
        value.get("semantic_terminal_motion"),
        owner="animation_qa.semantic_terminal_motion",
    )
    walking_summary = _mapping(
        terminal_motion.get("walking_summary"),
        owner="animation_qa.semantic_terminal_motion.walking_summary",
    )
    legacy_hind_gait_metric_triggered = walking_summary.get(
        "legacy_hind_gait_metric_triggered"
    )
    if not isinstance(legacy_hind_gait_metric_triggered, bool):
        raise PackageCompileError(
            "animation_qa legacy_hind_gait_metric_triggered must be boolean"
        )
    front_forward = _nonnegative_number(
        walking_summary.get("mean_front_paw_forward_range_m"),
        owner=(
            "animation_qa.semantic_terminal_motion.walking_summary."
            "mean_front_paw_forward_range_m"
        ),
    )
    hind_forward = _nonnegative_number(
        walking_summary.get("mean_hind_paw_forward_range_m"),
        owner=(
            "animation_qa.semantic_terminal_motion.walking_summary."
            "mean_hind_paw_forward_range_m"
        ),
    )
    hind_lateral = _nonnegative_number(
        walking_summary.get("mean_hind_paw_lateral_range_m"),
        owner=(
            "animation_qa.semantic_terminal_motion.walking_summary."
            "mean_hind_paw_lateral_range_m"
        ),
    )
    measured_hind_limitation = bool(
        hind_forward < 0.25 * front_forward and hind_lateral > hind_forward
    )
    if legacy_hind_gait_metric_triggered != measured_hind_limitation:
        raise PackageCompileError(
            "animation_qa legacy_hind_gait_metric_triggered does not match its "
            "hind-gait metrics"
        )
    if legacy_hind_gait_metric_triggered and not limitations:
        raise PackageCompileError(
            "animation_qa must retain the measured legacy hind-gait limitation"
        )
    if not legacy_hind_gait_metric_triggered and limitations:
        raise PackageCompileError(
            "animation_qa must not claim a legacy hind-gait limitation when the "
            "metric is not triggered"
        )


def _validate_contacts(
    report: _JsonInput,
    *,
    visual_sha256: str,
    actions_sha256: str,
    actions: BakedActionSet,
    anchors: Sequence[Mapping[str, Any]],
) -> None:
    value = report.value
    schema = value.get("schema", value.get("schema_version"))
    if schema != CONTACT_PHASES_SCHEMA:
        raise PackageCompileError(f"contacts schema must be {CONTACT_PHASES_SCHEMA!r}")
    if value.get("qualification_claim") is not False:
        raise PackageCompileError("contacts must not claim qualification")
    state = value.get("qualification_state")
    if state is not None and state != "research_candidate":
        raise PackageCompileError(
            "contacts qualification_state must be research_candidate"
        )
    source_sha = value.get("source_glb_sha256", value.get("source_sha256"))
    if source_sha != visual_sha256:
        raise PackageCompileError("contacts source GLB hash does not match visual")
    if value.get("baked_actions_sha256") != actions_sha256:
        raise PackageCompileError(
            "contacts baked_actions_sha256 does not match baked actions"
        )
    if value.get("runtime_joint_order") != list(actions.runtime_joint_order):
        raise PackageCompileError(
            "contacts runtime_joint_order differs from baked actions"
        )
    if (
        value.get("sample_rate_hz") != actions.sample_rate_hz
        or value.get("time_base_hz") != actions.time_base_hz
    ):
        raise PackageCompileError("contacts clock differs from baked actions")
    if value.get("contact_order") != CONTACT_ORDER:
        raise PackageCompileError(
            f"contacts contact_order must be exactly {CONTACT_ORDER}"
        )
    coordinate_system = _mapping(
        value.get("coordinate_system"), owner="contacts.coordinate_system"
    )
    if coordinate_system != {
        "handedness": "right",
        "up_axis": "+Y",
        "forward_axis": "-Z",
        "linear_unit": "meter",
        "quaternion_order": "xyzw",
    }:
        raise PackageCompileError("contacts coordinate_system is not canonical M2")
    expected_contact_anchors = [
        anchor
        for contact_id in CONTACT_ORDER
        for anchor in anchors
        if anchor.get("anchor_id") == contact_id
    ]
    reported_anchor_values = _sequence(
        value.get("anchor_definitions"), owner="contacts.anchor_definitions"
    )
    reported_contact_anchors = [
        _canonical_anchor(anchor, index=index)
        for index, anchor in enumerate(reported_anchor_values)
    ]
    if reported_contact_anchors != expected_contact_anchors:
        raise PackageCompileError(
            "contacts anchor_definitions differ from explicit package anchors"
        )
    _mapping(value.get("thresholds"), owner="contacts.thresholds")
    _sequence(value.get("warnings"), owner="contacts.warnings")
    expected = _action_by_id(actions)
    records = _sequence(value.get("actions"), owner="contacts.actions")
    seen: set[str] = set()
    for index, item in enumerate(records):
        record = _mapping(item, owner=f"contacts.actions[{index}]")
        action_id = record.get("semantic_action_id")
        if action_id not in expected or action_id in seen:
            raise PackageCompileError("contacts action mapping is not canonical")
        seen.add(action_id)
        clip = expected[action_id]
        if (
            record.get("source_action_name") != clip.source_action_name
            or record.get("sample_count") != clip.sample_count
        ):
            raise PackageCompileError(
                f"contacts {action_id} metadata differs from baked actions"
            )
        frames = _sequence(
            record.get("frames"), owner=f"contacts.actions[{index}].frames"
        )
        if len(frames) != clip.sample_count:
            raise PackageCompileError(
                f"contacts {action_id} frame count differs from baked actions"
            )
        for frame_index, frame_value in enumerate(frames):
            frame = _mapping(
                frame_value,
                owner=f"contacts.actions[{index}].frames[{frame_index}]",
            )
            source_time = frame.get("source_time_seconds")
            if (
                frame.get("sample_index") != frame_index
                or frame.get("sample_tick") != clip.sample_ticks[frame_index]
                or not isinstance(source_time, (int, float))
                or isinstance(source_time, bool)
                or not math.isfinite(float(source_time))
                or not math.isclose(
                    float(source_time),
                    clip.source_times_seconds[frame_index],
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise PackageCompileError(
                    f"contacts {action_id} frame {frame_index} clock is invalid"
                )
            states = _sequence(
                frame.get("contacts"),
                owner=(f"contacts.actions[{index}].frames[{frame_index}].contacts"),
            )
            if len(states) != len(CONTACT_ORDER):
                raise PackageCompileError(
                    f"contacts {action_id} frame {frame_index} state count is invalid"
                )
            for contact_index, (state_value, contact_id) in enumerate(
                zip(states, CONTACT_ORDER, strict=True)
            ):
                state = _mapping(
                    state_value,
                    owner=(
                        f"contacts.actions[{index}].frames[{frame_index}]"
                        f".contacts[{contact_index}]"
                    ),
                )
                if state.get("contact_id") != contact_id or not isinstance(
                    state.get("in_contact"), bool
                ):
                    raise PackageCompileError(
                        f"contacts {action_id} frame {frame_index} state order is invalid"
                    )
    if seen != {"idle", "walk"}:
        raise PackageCompileError("contacts must contain exactly idle and walk")


def _license_id(value: Mapping[str, Any]) -> str | None:
    direct = value.get("license", value.get("spdx_id"))
    if isinstance(direct, str):
        return direct
    if isinstance(direct, dict):
        nested = direct.get("spdx_id", direct.get("id"))
        if isinstance(nested, str):
            return nested
    return None


def _validate_source_and_license(
    source: _JsonInput,
    license_snapshot: _JsonInput,
    *,
    identity: AnimalPackageIdentity,
) -> None:
    if not isinstance(source.value.get("schema"), str) or not source.value["schema"]:
        raise PackageCompileError("source_manifest.schema must be non-empty")
    if source.value.get("formal_dataset_registration_authorized") is not False:
        raise PackageCompileError(
            "source_manifest.formal_dataset_registration_authorized must be "
            "exactly false"
        )
    if (
        not isinstance(license_snapshot.value.get("schema"), str)
        or not (license_snapshot.value["schema"])
    ):
        raise PackageCompileError("license_snapshot.schema must be non-empty")
    if _license_id(license_snapshot.value) != identity.license:
        raise PackageCompileError("license_snapshot does not match identity.license")
    if license_snapshot.value.get("allowed_use") != identity.allowed_use:
        raise PackageCompileError(
            "license_snapshot.allowed_use does not match package identity"
        )
    if license_snapshot.value.get("redistribution") != identity.redistribution:
        raise PackageCompileError(
            "license_snapshot.redistribution does not match package identity"
        )


def _component_format(component_type: int) -> tuple[str, int]:
    formats = {
        5120: ("b", 1),
        5121: ("B", 1),
        5122: ("h", 2),
        5123: ("H", 2),
        5125: ("I", 4),
        5126: ("f", 4),
    }
    try:
        return formats[component_type]
    except KeyError as exc:
        raise PackageCompileError(
            f"unsupported accessor component type {component_type}"
        ) from exc


def _decode_integer_scalar(
    document: GlbDocument, accessor_index: int
) -> tuple[int, ...]:
    root = document.json
    accessors = root.get("accessors")
    views = root.get("bufferViews")
    if not isinstance(accessors, list) or not isinstance(views, list):
        raise PackageCompileError("visual GLB lacks accessors/bufferViews")
    try:
        accessor = accessors[accessor_index]
    except (IndexError, TypeError) as exc:
        raise PackageCompileError("visual GLB index accessor is out of range") from exc
    if not isinstance(accessor, dict) or accessor.get("type") != "SCALAR":
        raise PackageCompileError("mesh indices must use a SCALAR accessor")
    if accessor.get("normalized", False) is not False or "sparse" in accessor:
        raise PackageCompileError("mesh indices must be non-sparse/non-normalized")
    component_type = accessor.get("componentType")
    if component_type not in {5121, 5123, 5125}:
        raise PackageCompileError("mesh indices must use an unsigned integer type")
    fmt, component_size = _component_format(component_type)
    count = accessor.get("count")
    view_index = accessor.get("bufferView")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(view_index, bool)
        or not isinstance(view_index, int)
    ):
        raise PackageCompileError("mesh index accessor metadata is invalid")
    try:
        view = views[view_index]
    except (IndexError, TypeError) as exc:
        raise PackageCompileError("mesh index bufferView is out of range") from exc
    if not isinstance(view, dict) or view.get("buffer") != 0:
        raise PackageCompileError("mesh index bufferView must use embedded buffer 0")
    stride = view.get("byteStride", component_size)
    if not isinstance(stride, int) or stride < component_size:
        raise PackageCompileError("mesh index byteStride is invalid")
    first = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    unpacker = struct.Struct("<" + fmt)
    try:
        return tuple(
            int(unpacker.unpack_from(document.binary, first + index * stride)[0])
            for index in range(count)
        )
    except struct.error as exc:
        raise PackageCompileError("mesh index accessor is truncated") from exc


def _decode_unsigned_vec4(
    document: GlbDocument, accessor_index: int
) -> tuple[tuple[int, int, int, int], ...]:
    root = document.json
    accessors = root.get("accessors")
    views = root.get("bufferViews")
    if not isinstance(accessors, list) or not isinstance(views, list):
        raise PackageCompileError("visual GLB lacks accessors/bufferViews")
    try:
        accessor = accessors[accessor_index]
    except (IndexError, TypeError) as exc:
        raise PackageCompileError(
            "visual GLB JOINTS_0 accessor is out of range"
        ) from exc
    if not isinstance(accessor, dict) or accessor.get("type") != "VEC4":
        raise PackageCompileError("visual GLB JOINTS_0 must use a VEC4 accessor")
    if accessor.get("normalized", False) is not False or "sparse" in accessor:
        raise PackageCompileError("visual GLB JOINTS_0 must be canonical unsigned data")
    component_type = accessor.get("componentType")
    if component_type not in {5121, 5123, 5125}:
        raise PackageCompileError("visual GLB JOINTS_0 must use unsigned integers")
    fmt, component_size = _component_format(component_type)
    count = accessor.get("count")
    view_index = accessor.get("bufferView")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(view_index, bool)
        or not isinstance(view_index, int)
    ):
        raise PackageCompileError("visual GLB JOINTS_0 metadata is invalid")
    try:
        view = views[view_index]
    except (IndexError, TypeError) as exc:
        raise PackageCompileError(
            "visual GLB JOINTS_0 bufferView is out of range"
        ) from exc
    if not isinstance(view, dict) or view.get("buffer") != 0:
        raise PackageCompileError("visual GLB JOINTS_0 must use embedded buffer 0")
    element_size = 4 * component_size
    stride = view.get("byteStride", element_size)
    if not isinstance(stride, int) or stride < element_size or stride % component_size:
        raise PackageCompileError("visual GLB JOINTS_0 byteStride is invalid")
    view_offset = view.get("byteOffset", 0)
    accessor_offset = accessor.get("byteOffset", 0)
    view_length = view.get("byteLength")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (view_offset, accessor_offset, view_length)
    ):
        raise PackageCompileError("visual GLB JOINTS_0 offsets are invalid")
    required = accessor_offset + (count - 1) * stride + element_size
    if required > view_length or view_offset + required > len(document.binary):
        raise PackageCompileError("visual GLB JOINTS_0 accessor is truncated")
    unpacker = struct.Struct("<" + fmt * 4)
    first = view_offset + accessor_offset
    return tuple(
        tuple(
            int(component)
            for component in unpacker.unpack_from(
                document.binary, first + index * stride
            )
        )
        for index in range(count)
    )  # type: ignore[return-value]


def _mesh_evidence(
    document: GlbDocument,
) -> tuple[str, str, str, tuple[np.ndarray, np.ndarray]]:
    root = document.json
    nodes = root.get("nodes")
    meshes = root.get("meshes")
    if not isinstance(nodes, list) or not isinstance(meshes, list):
        raise PackageCompileError("visual GLB lacks nodes/meshes")
    mesh_nodes = [
        node
        for node in nodes
        if isinstance(node, dict) and "mesh" in node and node.get("skin") == 0
    ]
    if len(mesh_nodes) != 1:
        raise PackageCompileError(
            "package compiler requires exactly one mesh node bound to skin 0"
        )
    mesh_index = mesh_nodes[0].get("mesh")
    if isinstance(mesh_index, bool) or not isinstance(mesh_index, int):
        raise PackageCompileError("visual GLB mesh index is invalid")
    try:
        mesh = meshes[mesh_index]
    except (IndexError, TypeError) as exc:
        raise PackageCompileError("visual GLB mesh index is out of range") from exc
    primitives = mesh.get("primitives") if isinstance(mesh, dict) else None
    if not isinstance(primitives, list) or not primitives:
        raise PackageCompileError("visual GLB mesh has no primitives")

    topology_payload: list[dict[str, Any]] = []
    uv_payload: list[dict[str, Any]] = []
    weights_payload: list[dict[str, Any]] = []
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    for primitive_index, primitive in enumerate(primitives):
        if not isinstance(primitive, dict) or primitive.get("mode", 4) != 4:
            raise PackageCompileError("visual GLB primitives must be TRIANGLES")
        attributes = primitive.get("attributes")
        if not isinstance(attributes, dict):
            raise PackageCompileError("visual GLB primitive lacks attributes")
        for name in (
            "POSITION",
            "NORMAL",
            "TEXCOORD_0",
            "JOINTS_0",
            "WEIGHTS_0",
        ):
            if not isinstance(attributes.get(name), int):
                raise PackageCompileError(f"visual GLB primitive lacks {name}")
        if not isinstance(primitive.get("indices"), int):
            raise PackageCompileError("visual GLB primitive must be indexed")
        positions = decode_accessor(document, attributes["POSITION"])
        normals = decode_accessor(document, attributes["NORMAL"])
        texcoords = decode_accessor(document, attributes["TEXCOORD_0"])
        weights = decode_accessor(document, attributes["WEIGHTS_0"])
        joints = _decode_unsigned_vec4(document, attributes["JOINTS_0"])
        if (
            positions.element_type != "VEC3"
            or normals.element_type != "VEC3"
            or texcoords.element_type != "VEC2"
        ):
            raise PackageCompileError(
                "visual GLB POSITION/NORMAL/TEXCOORD_0 types are invalid"
            )
        if weights.element_type != "VEC4" or not (
            positions.count
            == normals.count
            == texcoords.count
            == weights.count
            == len(joints)
        ):
            raise PackageCompileError("visual GLB vertex accessor counts differ")
        position_array = np.asarray(positions.values, dtype=np.float64)
        if not np.all(np.isfinite(position_array)):
            raise PackageCompileError("visual GLB positions must be finite")
        current_min = np.min(position_array, axis=0)
        current_max = np.max(position_array, axis=0)
        minimum = current_min if minimum is None else np.minimum(minimum, current_min)
        maximum = current_max if maximum is None else np.maximum(maximum, current_max)
        indices = _decode_integer_scalar(document, primitive["indices"])
        if len(indices) % 3 or min(indices) < 0 or max(indices) >= positions.count:
            raise PackageCompileError("visual GLB triangle indices are invalid")
        triangles = [
            list(indices[index : index + 3]) for index in range(0, len(indices), 3)
        ]
        topology_payload.append(
            {
                "primitive_index": primitive_index,
                "vertex_count": positions.count,
                "triangles": triangles,
            }
        )
        uv_payload.append(
            {
                "primitive_index": primitive_index,
                "texcoord_0": [list(value) for value in texcoords.values],
            }
        )
        weights_payload.append(
            {
                "primitive_index": primitive_index,
                "joint_ordinals": [list(value) for value in joints],
                "weights": [list(value) for value in weights.values],
            }
        )
    assert minimum is not None and maximum is not None
    if np.any(maximum <= minimum):
        raise PackageCompileError("visual GLB rest POSITION bounds are degenerate")
    topology_sha256 = canonical_json_sha256(
        {
            "schema": "avengine_m2_topology_identity_v1",
            "primitives": topology_payload,
        }
    )
    uv_sha256 = canonical_json_sha256(
        {"schema": "avengine_m2_uv_identity_v1", "primitives": uv_payload}
    )
    weights_sha256 = canonical_json_sha256(
        {"schema": "avengine_m2_weight_identity_v1", "primitives": weights_payload}
    )
    return topology_sha256, uv_sha256, weights_sha256, (minimum, maximum)


def _canonical_anchor(value: Any, *, index: int) -> dict[str, Any]:
    if isinstance(value, Mapping):
        anchor_id = value.get("anchor_id")
        joint_id = value.get("joint_id")
        transform_value = value.get("joint_from_anchor")
    else:
        anchor_id = getattr(value, "anchor_id", None)
        joint_id = getattr(value, "joint_id", None)
        transform_value = getattr(value, "joint_from_anchor", None)
    if not isinstance(anchor_id, str) or not anchor_id:
        raise PackageCompileError(f"anchor_definitions[{index}].anchor_id is invalid")
    if not isinstance(joint_id, str) or not joint_id:
        raise PackageCompileError(f"anchor_definitions[{index}].joint_id is invalid")
    if isinstance(transform_value, Mapping):
        translation = transform_value.get("translation_m")
        rotation = transform_value.get("rotation_xyzw")
    else:
        translation = getattr(transform_value, "translation_m", None)
        rotation = getattr(transform_value, "rotation_xyzw", None)
    try:
        translation_array = np.asarray(translation, dtype=np.float64)
        rotation_array = np.asarray(rotation, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PackageCompileError(
            f"anchor_definitions[{index}] has a non-numeric transform"
        ) from exc
    if (
        translation_array.shape != (3,)
        or rotation_array.shape != (4,)
        or not np.all(np.isfinite(translation_array))
        or not np.all(np.isfinite(rotation_array))
    ):
        raise PackageCompileError(
            f"anchor_definitions[{index}] transform must be finite vec3/quat"
        )
    norm = float(np.linalg.norm(rotation_array))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise PackageCompileError(
            f"anchor_definitions[{index}] quaternion must already be unit"
        )
    scalar = float(rotation_array[3])
    sign_component = scalar
    if math.isclose(scalar, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
        sign_component = next(
            (
                float(component)
                for component in rotation_array[:3]
                if not math.isclose(float(component), 0.0, rel_tol=0.0, abs_tol=1.0e-15)
            ),
            0.0,
        )
    if sign_component < 0.0:
        raise PackageCompileError(
            f"anchor_definitions[{index}] quaternion is not canonical"
        )
    return {
        "anchor_id": anchor_id,
        "joint_id": joint_id,
        "joint_from_anchor": {
            "translation_m": [
                0.0 if float(component) == 0.0 else float(component)
                for component in translation_array
            ],
            "rotation_xyzw": [
                0.0 if float(component) == 0.0 else float(component)
                for component in rotation_array
            ],
        },
    }


def _anchors(
    definitions: Sequence[Any], *, mapping: HabitatAssetMapping
) -> list[dict[str, Any]]:
    if isinstance(definitions, (str, bytes)):
        raise PackageCompileError("anchor_definitions must be an explicit sequence")
    values = [
        _canonical_anchor(value, index=index) for index, value in enumerate(definitions)
    ]
    ids = [value["anchor_id"] for value in values]
    if len(ids) != len(set(ids)):
        raise PackageCompileError("anchor_definitions contains duplicate anchor IDs")
    required = {"body", "head", "muzzle", *CONTACT_ORDER}
    missing = sorted(required - set(ids))
    if missing:
        raise PackageCompileError(f"anchor_definitions is missing {missing}")
    known = set(mapping.joint_order)
    unknown = sorted(
        value["joint_id"] for value in values if value["joint_id"] not in known
    )
    if unknown:
        raise PackageCompileError(f"anchor_definitions uses unknown joints: {unknown}")
    order = {
        name: index
        for index, name in enumerate(("body", "head", "muzzle", *CONTACT_ORDER))
    }
    return sorted(
        values,
        key=lambda item: (
            order.get(item["anchor_id"], len(order)),
            item["anchor_id"],
        ),
    )


def _pad(payload: bytes, fill: bytes) -> bytes:
    return payload + fill * ((-len(payload)) % 4)


def _collision_proxy_glb(
    *, minimum: np.ndarray, maximum: np.ndarray, visual_sha256: str
) -> bytes:
    extent = maximum - minimum
    margin = 0.01 + 0.02 * extent
    low = minimum - margin
    high = maximum + margin
    vertices = (
        (low[0], low[1], low[2]),
        (high[0], low[1], low[2]),
        (high[0], high[1], low[2]),
        (low[0], high[1], low[2]),
        (low[0], low[1], high[2]),
        (high[0], low[1], high[2]),
        (high[0], high[1], high[2]),
        (low[0], high[1], high[2]),
    )
    indices = (
        0,
        2,
        1,
        0,
        3,
        2,
        4,
        5,
        6,
        4,
        6,
        7,
        0,
        1,
        5,
        0,
        5,
        4,
        3,
        7,
        6,
        3,
        6,
        2,
        0,
        4,
        7,
        0,
        7,
        3,
        1,
        2,
        6,
        1,
        6,
        5,
    )
    positions = b"".join(struct.pack("<3f", *vertex) for vertex in vertices)
    index_bytes = b"".join(struct.pack("<H", index) for index in indices)
    binary = positions + index_bytes
    extras = {
        "schema": "avengine_m2_kinematic_canary_collision_proxy_v1",
        "source_visual_sha256": visual_sha256,
        "coordinate_frame": "canonical_skin_root",
        "construction": "rest_position_aabb_with_deterministic_margin",
        "rest_position_minimum_m": [float(value) for value in minimum],
        "rest_position_maximum_m": [float(value) for value in maximum],
        "proxy_minimum_m": [float(value) for value in low],
        "proxy_maximum_m": [float(value) for value in high],
        "kinematic_canary_only": True,
        "used_for_physics": False,
        "used_for_contact_inference": False,
        "qualification_claim": False,
    }
    document = {
        "asset": {
            "version": "2.0",
            "generator": "avengine.m2.package collision proxy v1",
            "extras": extras,
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "m2_kinematic_canary_rest_bbox", "mesh": 0}],
        "meshes": [
            {
                "name": "m2_kinematic_canary_rest_bbox",
                "extras": extras,
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}
                ],
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(positions),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": len(positions),
                "byteLength": len(index_bytes),
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 8,
                "type": "VEC3",
                "min": [float(value) for value in low],
                "max": [float(value) for value in high],
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": len(indices),
                "type": "SCALAR",
            },
        ],
    }
    json_payload = _pad(
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
        b" ",
    )
    bin_payload = _pad(binary, b"\0")
    chunks = (
        struct.pack("<II", len(json_payload), _JSON_CHUNK_TYPE)
        + json_payload
        + struct.pack("<II", len(bin_payload), _BIN_CHUNK_TYPE)
        + bin_payload
    )
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def _file_record(role: str, path: str, payload: bytes) -> dict[str, Any]:
    return {
        "role": role,
        "path": path,
        "byte_size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _input_record(value: _JsonInput) -> dict[str, Any]:
    return {
        "evidence_id": value.label,
        "byte_size": value.byte_size,
        "sha256": value.sha256,
        "schema": value.value.get("schema", value.value.get("schema_version")),
    }


def _skeleton_manifest(
    *, mapping: HabitatAssetMapping, visual_sha256: str
) -> dict[str, Any]:
    joint_data = mapping.joint_mapping_data()
    return {
        "schema": "avengine_m2_skeleton_manifest_v1",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source_visual_sha256": visual_sha256,
        "root_joint_id": mapping.root_joint_id,
        "joint_order": list(mapping.joint_order),
        "runtime_joint_order": list(mapping.runtime_joint_order),
        "joint_pose_encoding": "ordered_local_rotation_xyzw_float64",
        "actor_from_skin_root": [list(row) for row in mapping.actor_from_skin_root],
        "actor_from_skin_root_source": mapping.actor_from_skin_root_source,
        "joints": joint_data["joints"],
    }


def _action_manifest(
    *,
    actions: BakedActionSet,
    actions_sha256: str,
    action_report: _JsonInput,
    action_revision: str,
) -> dict[str, Any]:
    return {
        "schema": "avengine_m2_action_manifest_v1",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source_visual_sha256": actions.source_glb_sha256,
        "action_revision": action_revision,
        "container_policy": {
            "idle_poses_and_walk_poses_are_identical_complete_baked_action_sets": True,
            "complete_baked_action_set_sha256": actions_sha256,
            "selection_is_manifest_driven": True,
        },
        "actions": [
            {
                "action_id": clip.semantic_action_id,
                "poses_file_role": f"{clip.semantic_action_id}_poses",
                "container_sha256": actions_sha256,
                "selected_member": {
                    "semantic_action_id": clip.semantic_action_id,
                    "source_action_name": clip.source_action_name,
                    "sample_count": clip.sample_count,
                    "loop_duration_ticks": clip.loop_duration_ticks,
                    "sample_ticks": list(clip.sample_ticks),
                },
                "container_members": ["idle", "walk"],
            }
            for clip in actions.actions
        ],
        "action_bake_report": {
            **_input_record(action_report),
            "snapshot": action_report.value,
        },
    }


def _prepare_output_directory(path: str | Path) -> Path:
    output = _absolute_without_symlinks(path, owner="output_directory")
    if os.path.lexists(output):
        raise PackageCompileError("output_directory must not already exist")
    parent = output.parent
    if not parent.is_dir():
        raise PackageCompileError("output_directory parent must exist")
    return output


def _publish_directory_no_replace(staging: Path, output: Path) -> None:
    """Atomically expose one complete directory without replacing a name."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise PackageCompileError(
            "atomic no-replace directory publication is unavailable"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(staging),
        at_fdcwd,
        os.fsencode(output),
        rename_noreplace,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise PackageCompileError(
            "output_directory appeared during atomic publication; refusing to replace it"
        )
    raise PackageCompileError(
        f"unable to atomically publish package directory: {os.strerror(error)}"
    )


def _write_file_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    failed = True
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive package write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        failed = False
    finally:
        os.close(descriptor)
        if failed:
            try:
                path.unlink()
            except OSError:
                pass


def _output_tree_files(output: Path) -> set[str]:
    files: set[str] = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise PackageCompileError("compiled package tree contains a symbolic link")
        if path.is_file():
            files.add(path.relative_to(output).as_posix())
        elif not path.is_dir():
            raise PackageCompileError("compiled package tree contains a special file")
    return files


def _write_payloads(
    output: Path,
    payloads: Mapping[str, bytes],
    *,
    validate: Callable[[Path], None],
) -> None:
    """Publish one package with exclusive leaves and rollback on any failure."""

    created_output = False
    created_directories: list[Path] = []
    created_files: list[Path] = []
    try:
        try:
            output.mkdir(parents=False, exist_ok=False)
            created_output = True
        except FileExistsError:
            if output.is_symlink() or not output.is_dir():
                raise PackageCompileError("output_directory must be a real directory")
            if any(output.iterdir()):
                raise PackageCompileError("output_directory must be empty")

        for relative_path, payload in payloads.items():
            relative = Path(relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise PackageCompileError(
                    f"package path is not a safe relative path: {relative_path!r}"
                )
            destination = output / relative
            current = output
            for part in relative.parts[:-1]:
                current /= part
                try:
                    current.mkdir(exist_ok=False)
                    created_directories.append(current)
                except FileExistsError:
                    if current.is_symlink() or not current.is_dir():
                        raise PackageCompileError(
                            f"package directory is unsafe: {current.relative_to(output)!s}"
                        )
            try:
                _write_file_exclusive(destination, payload)
            except FileExistsError as exc:
                raise PackageCompileError(
                    f"refusing to replace package path {relative_path!r}"
                ) from exc
            except OSError as exc:
                raise PackageCompileError(
                    f"unable to publish package path {relative_path!r}: {exc}"
                ) from exc
            created_files.append(destination)

        expected = set(payloads)
        actual = _output_tree_files(output)
        if actual != expected:
            raise PackageCompileError(
                "compiled package tree differs from the exact payload set: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        validate(output / "asset_manifest.json")
    except Exception:
        for path in reversed(created_files):
            try:
                path.unlink()
            except OSError:
                pass
        for path in reversed(created_directories):
            try:
                path.rmdir()
            except OSError:
                pass
        if created_output:
            try:
                output.rmdir()
            except OSError:
                pass
        raise


def compile_research_candidate_animal_package(
    *,
    output_directory: str | Path,
    identity: AnimalPackageIdentity,
    visual_glb: str | Path,
    rebase_report: str | Path,
    rebase_deformation_report: str | Path,
    action_report: str | Path,
    static_qa: str | Path,
    deformation_qa: str | Path,
    animation_qa: str | Path,
    habitat_static_probe: str | Path,
    habitat_animation_review: str | Path,
    baked_actions: str | Path,
    contacts: str | Path,
    anchor_definitions: Sequence[Any],
    source_manifest: str | Path,
    license_snapshot: str | Path,
    shader_type: str = "phong",
) -> Path:
    """Compile a complete, hash-closed research-candidate animal package.

    The destination must be absent. No symbolic-link component is accepted.
    The complete package is validated in a same-filesystem staging directory
    and becomes visible with one atomic no-replace directory publication.
    """

    if not isinstance(identity, AnimalPackageIdentity):
        raise PackageCompileError("identity must be AnimalPackageIdentity")
    if not isinstance(shader_type, str) or shader_type not in {"phong", "pbr"}:
        raise PackageCompileError("shader_type must be exactly 'phong' or 'pbr'")
    output = _prepare_output_directory(output_directory)

    visual_path, visual_payload = _read_regular(visual_glb, owner="visual_glb")
    visual_sha256 = hashlib.sha256(visual_payload).hexdigest()
    try:
        document = load_glb(visual_path)
    except (OSError, ValueError) as exc:
        raise PackageCompileError(f"visual_glb is invalid: {exc}") from exc
    if document.sha256 != visual_sha256:
        raise PackageCompileError("visual_glb parser identity mismatch")

    rebase = _read_json(rebase_report, owner="rebase_report")
    rebase_deformation = _read_json(
        rebase_deformation_report, owner="rebase_deformation_report"
    )
    action = _read_json(action_report, owner="action_report")
    static = _read_json(static_qa, owner="static_qa")
    deformation = _read_json(deformation_qa, owner="deformation_qa")
    animation = _read_json(animation_qa, owner="animation_qa")
    static_probe = _read_json(habitat_static_probe, owner="habitat_static_probe")
    animation_review = _read_json(
        habitat_animation_review, owner="habitat_animation_review"
    )
    contact = _read_json(contacts, owner="contacts")
    source = _read_json(source_manifest, owner="source_manifest")
    license_value = _read_json(license_snapshot, owner="license_snapshot")
    actions_path, actions_payload = _read_regular(baked_actions, owner="baked_actions")
    actions_sha256 = hashlib.sha256(actions_payload).hexdigest()
    try:
        action_set = read_baked_actions_npz(actions_path)
    except (OSError, ValueError) as exc:
        raise PackageCompileError(f"baked_actions is invalid: {exc}") from exc
    if action_set.source_glb_sha256 != visual_sha256:
        raise PackageCompileError("baked_actions source GLB hash does not match visual")

    _validate_rebase_report(
        rebase, visual_sha256=visual_sha256, visual_size=len(visual_payload)
    )
    try:
        habitat_mapping = build_habitat_asset_mapping_from_rebase_report(
            document, rebase.value
        )
    except ValueError as exc:
        raise PackageCompileError(f"unable to build Habitat mapping: {exc}") from exc
    if tuple(action_set.runtime_joint_order) != habitat_mapping.runtime_joint_order:
        raise PackageCompileError(
            "baked_actions runtime_joint_order differs from the GLB skeleton"
        )
    anchors = _anchors(anchor_definitions, mapping=habitat_mapping)
    _validate_rebase_deformation_report(
        rebase_deformation,
        visual_sha256=visual_sha256,
        visual_size=len(visual_payload),
        rebase_report=rebase,
    )
    _validate_action_report(
        action,
        visual_sha256=visual_sha256,
        visual_size=len(visual_payload),
        actions_sha256=actions_sha256,
        actions_size=len(actions_payload),
        actions=action_set,
    )
    _validate_habitat_static_probe(
        static_probe,
        visual_sha256=visual_sha256,
        visual_size=len(visual_payload),
        shader_type=shader_type,
        semantic_id=identity.semantic_id,
        runtime_joint_order=action_set.runtime_joint_order,
    )
    _validate_habitat_animation_review(
        animation_review,
        visual_sha256=visual_sha256,
        visual_size=len(visual_payload),
        actions_sha256=actions_sha256,
        actions_size=len(actions_payload),
        rebase_report=rebase,
        actions=action_set,
        shader_type=shader_type,
        semantic_id=identity.semantic_id,
    )
    topology_sha256, uv_sha256, weights_sha256, (minimum, maximum) = _mesh_evidence(
        document
    )
    _validate_static_qa(
        static,
        visual_sha256=visual_sha256,
        topology_sha256=topology_sha256,
        uv_sha256=uv_sha256,
        weights_sha256=weights_sha256,
        joint_count=len(habitat_mapping.joint_order),
    )
    _validate_deformation_qa(
        deformation,
        visual_sha256=visual_sha256,
        actions_sha256=actions_sha256,
        actions=action_set,
    )
    muzzle_joint_id = next(
        anchor["joint_id"] for anchor in anchors if anchor["anchor_id"] == "muzzle"
    )
    _validate_animation_qa(
        animation,
        visual_sha256=visual_sha256,
        actions_sha256=actions_sha256,
        actions=action_set,
        muzzle_joint_id=muzzle_joint_id,
    )
    _validate_contacts(
        contact,
        visual_sha256=visual_sha256,
        actions_sha256=actions_sha256,
        actions=action_set,
        anchors=anchors,
    )
    _validate_source_and_license(source, license_value, identity=identity)

    collision_payload = _collision_proxy_glb(
        minimum=minimum, maximum=maximum, visual_sha256=visual_sha256
    )
    if hashlib.sha256(collision_payload).hexdigest() == visual_sha256:
        raise PackageCompileError("collision proxy must not masquerade as visual GLB")

    skeleton_value = _skeleton_manifest(
        mapping=habitat_mapping, visual_sha256=visual_sha256
    )
    skinning_value = {
        "schema": "avengine_m2_skinning_manifest_v1",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source_visual_sha256": visual_sha256,
        "root_joint_id": habitat_mapping.root_joint_id,
        "joint_order": list(habitat_mapping.joint_order),
        "weights_revision": identity.weights_revision,
        "bind_evidence": {
            **_input_record(rebase),
            "snapshot": rebase.value,
        },
        "rebase_deformation_evidence_sha256": rebase_deformation.sha256,
        "automatic_deformation_qa_sha256": deformation.sha256,
        "weights_sha256": weights_sha256,
        "collision_proxy": {
            "schema": "avengine_m2_kinematic_canary_collision_proxy_v1",
            "construction": "rest_position_aabb_with_deterministic_margin",
            "kinematic_canary_only": True,
            "used_for_physics": False,
            "used_for_contact_inference": False,
            "decision_reason": (
                "The conservative proxy is package completeness evidence only; "
                "M2 contact phases come from explicit kinematic anchors."
            ),
        },
    }
    anchor_value = {
        "schema": "avengine_m2_emitter_anchors_v1",
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
        "anchors": anchors,
    }
    action_value = _action_manifest(
        actions=action_set,
        actions_sha256=actions_sha256,
        action_report=action,
        action_revision=identity.action_revision,
    )
    ao_config = build_habitat_ao_config_data(
        render_asset="../visual.glb",
        urdf_filepath="animal.urdf",
        semantic_id=identity.semantic_id,
        shader_type=shader_type,
    )

    payloads: dict[str, bytes] = {
        "visual.glb": visual_payload,
        "collision_proxy.glb": collision_payload,
        "skeleton.json": _json_bytes(skeleton_value),
        "skinning_manifest.json": _json_bytes(skinning_value),
        "emitter_anchors.json": _json_bytes(anchor_value),
        "actions/action_manifest.json": _json_bytes(action_value),
        "actions/idle.npz": actions_payload,
        "actions/walk.npz": actions_payload,
        "contacts/contact_phases.json": contact.payload,
        "qa/static_geometry.json": static.payload,
        "qa/deformation.json": deformation.payload,
        "qa/animation.json": animation.payload,
        "habitat/animal.urdf": habitat_mapping.render_urdf().encode("utf-8"),
        "habitat/animal.ao_config.json": _json_bytes(ao_config),
        "habitat/joint_mapping.json": _json_bytes(habitat_mapping.joint_mapping_data()),
    }
    role_by_path = {path: role for role, path in _ROLE_PATHS}
    non_provenance_records = [
        _file_record(role_by_path[path], path, payload)
        for path, payload in payloads.items()
    ]
    evidence = [
        rebase,
        rebase_deformation,
        static_probe,
        animation_review,
        deformation,
        action,
        animation,
        static,
        contact,
        source,
        license_value,
    ]
    provenance_value = {
        "schema": "avengine_m2_provenance_manifest_v1",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "formal_registry_promotion": False,
        "source_visual_sha256": visual_sha256,
        "source_manifest": {
            **_input_record(source),
            "snapshot": source.value,
        },
        "license_snapshot": {
            **_input_record(license_value),
            "snapshot": license_value.value,
        },
        "input_evidence": [_input_record(value) for value in evidence],
        "package_payloads_excluding_this_manifest": non_provenance_records,
        "lineage": {
            "rebase_report_sha256": rebase.sha256,
            "rebased_visual_sha256": visual_sha256,
            "rebase_deformation_report_sha256": rebase_deformation.sha256,
            "baked_actions_sha256": actions_sha256,
            "action_report_sha256": action.sha256,
            "contact_phases_sha256": contact.sha256,
            "static_qa_sha256": static.sha256,
            "deformation_qa_sha256": deformation.sha256,
            "animation_qa_sha256": animation.sha256,
            "habitat_static_probe_sha256": static_probe.sha256,
            "habitat_animation_review_sha256": animation_review.sha256,
        },
        "license_decision": {
            "license": identity.license,
            "allowed_use": identity.allowed_use,
            "redistribution": identity.redistribution,
        },
    }
    provenance_payload = _json_bytes(provenance_value)
    payloads["provenance_manifest.json"] = provenance_payload

    records = [_file_record(role, path, payloads[path]) for role, path in _ROLE_PATHS]
    if {record["role"] for record in records} != REQUIRED_FILE_ROLES:
        raise PackageCompileError(
            "compiler role table differs from REQUIRED_FILE_ROLES"
        )
    clips = _action_by_id(action_set)
    asset = {
        "schema": ANIMAL_SCHEMA,
        "asset_id": identity.asset_id,
        "template_id": identity.template_id,
        "body_plan_id": identity.body_plan_id,
        "morphotype_id": identity.morphotype_id,
        "admission_state": "research_candidate",
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Y",
            "forward_axis": "-Z",
            "linear_unit": "meter",
            "quaternion_order": "xyzw",
        },
        "revisions": {
            "topology_sha256": topology_sha256,
            "uv_sha256": uv_sha256,
            "skeleton_revision": identity.skeleton_revision,
            "weights_revision": identity.weights_revision,
            "collision_revision": identity.collision_revision,
            "action_revision": identity.action_revision,
        },
        "skeleton": {
            "root_joint_id": habitat_mapping.root_joint_id,
            "joint_order": list(habitat_mapping.joint_order),
            "runtime_joint_order": list(habitat_mapping.runtime_joint_order),
            "joint_pose_encoding": "ordered_local_rotation_xyzw_float64",
        },
        "contacts": {"contact_order": CONTACT_ORDER},
        "anchors": anchors,
        "actions": [
            {
                "action_id": action_id,
                "poses_file_role": f"{action_id}_poses",
                "source_action_revision": (
                    f"{identity.action_revision}:{clips[action_id].source_action_name}"
                ),
                "sample_count": clips[action_id].sample_count,
            }
            for action_id in ("idle", "walk")
        ],
        "files": records,
        "qualification": {
            "automatic_qa_status": "pass",
            "human_visual_review_status": "not_run",
            "human_review_binding_sha256": None,
            "decision_reason": (
                "Hash-closed local technical artifacts passed structural checks; "
                "the probe/review reports are not trusted runtime attestations. "
                "External capture/audit and human visual review have not established "
                "formal qualification, so this package remains a research candidate."
            ),
        },
        "provenance": {
            "source": identity.source,
            "source_revision": identity.source_revision,
            "source_sha256": source.sha256,
            "license": identity.license,
            "allowed_use": identity.allowed_use,
            "redistribution": identity.redistribution,
        },
    }
    manifest_payload = _json_bytes(asset)
    payloads["asset_manifest.json"] = manifest_payload

    def validate_compiled(manifest_path: Path) -> None:
        errors = validate_animal_asset_package(asset, manifest_path=manifest_path)
        if errors:
            raise PackageCompileError(
                "compiled package failed its own contract: " + "; ".join(errors)
            )

    # Validate the complete byte set in a same-filesystem staging directory
    # before exposing the destination name at all.
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        _write_payloads(staging, payloads, validate=validate_compiled)
        _publish_directory_no_replace(staging, output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return output / "asset_manifest.json"


__all__ = [
    "AnimalPackageIdentity",
    "PackageCompileError",
    "compile_research_candidate_animal_package",
]
