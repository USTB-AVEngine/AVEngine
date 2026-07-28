"""Fail-closed research registration for generated static sound sources.

The SPEAR static-object route is deliberately separate from articulated
source runtime profiles.  This module authenticates one sealed admission
batch, its selected job and stage-receipt closure, and the separate human
marker approval before appending one ``rigid_object`` record to the existing
M6 entity registry.  Marker review approves only the measured emitter
placement; it is not formal dataset-owner approval.
"""

from __future__ import annotations

import ast
import base64
import binascii
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
import json
import math
from numbers import Real
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

from avengine.contracts.json_io import canonical_json_sha256
from avengine.m6.entities import (
    STATIC_OBJECT_EVIDENCE_KINDS,
    resolve_entity_asset,
    validate_entity_asset_registry,
)
from avengine.m6.registry import (
    STATIC_OBJECT_MARKER_VISUAL_APPROVAL_SCHEMA,
    bind_content_hash,
    is_sha256,
    is_stable_id,
    json_schema_errors,
)
from avengine.optional_backends.spear_replicacad_glb import (
    ReplicaCADGLBError,
    _load_glb as _load_glb_document,
    _node_world_matrices as _glb_node_world_matrices,
    _transform_point as _transform_glb_point,
)
from avengine.security.path_policy import (
    PathPolicyError,
    WorkspacePathPolicy,
)


STATIC_DECISION_SCHEMA = "avengine_controlled_static_object_decision_v1"
STATIC_DECISION_BATCH_SCHEMA = (
    "avengine_controlled_static_object_decision_batch_v1"
)
STATIC_REVIEW_BATCH_SCHEMA = "avengine_controlled_static_object_review_batch_v1"
STATIC_REVIEW_SCHEMA = "avengine_controlled_static_object_review_v1"
STATIC_PIXAL_BATCH_SCHEMA = "avengine_controlled_animal_pixal_batch_v1"
STATIC_PIXAL_INPUT_SCHEMA = "avengine_controlled_animal_pixal_inputs_v1"
ISNET_EXECUTION_RECEIPT_SCHEMA = (
    "avengine_controlled_isnet_execution_receipt_v1"
)
ISNET_JOBS_SCHEMA = "avengine_controlled_animal_isnet_jobs_v1"
ISNET_STATUS_SCHEMA = "avengine_controlled_animal_isnet_status_v1"
ISNET_MODEL_FILENAME = "isnet-general-use.onnx"
ISNET_WORKER_FILENAME = "controlled_animal_isnet_worker.py"
COMBINED_PIXAL_INPUT_SCHEMA = "avengine_controlled_pixal_inputs_combined_v1"
STATIC_2D_REVIEW_BATCH_SCHEMA = (
    "avengine_controlled_static_object_2d_review_batch_v1"
)
STATIC_2D_REVIEW_SCHEMA = "avengine_controlled_static_object_2d_review_v1"
FLUX_BATCH_SCHEMA = "avengine_controlled_animal_flux2_batch_v1"
FLUX_CANDIDATE_SCHEMA = "avengine_controlled_animal_flux2_candidate_v1"
EXECUTION_PREFLIGHT_SCHEMA = "avengine_controlled_execution_preflight_v1"
ONE_SHOT_POLICY_RECORD_SCHEMA = (
    "avengine_controlled_animal_one_shot_policy_record_v1"
)
ONE_SHOT_POLICY_SCHEMA = "avengine_controlled_animal_one_shot_policy_v1"
ONE_SHOT_POLICY_ID = "animal_one_shot_no_seed_lottery_v1"
COMBINED_UPSTREAM_FLUX_EVIDENCE_SCHEMA = (
    "avengine_combined_upstream_flux_one_shot_evidence_v1"
)
STATIC_ADMISSION_PLAN_SCHEMA = (
    "avengine_controlled_static_object_admission_plan_v1"
)
STATIC_ADMISSION_BATCH_SCHEMA = (
    "avengine_controlled_static_object_admission_batch_v1"
)
STATIC_ADMISSION_JOB_RECEIPT_SCHEMA = (
    "avengine_controlled_static_object_admission_receipt_v1"
)
STATIC_ADMISSION_STAGE_RECEIPT_SCHEMA = (
    "avengine_controlled_static_object_stage_receipt_v1"
)
STATIC_ADMISSION_COMMAND_INPUT_MANIFEST_SCHEMA = (
    "avengine_controlled_static_object_command_input_manifest_v1"
)
STATIC_FINALIZATION_SCHEMA = "avengine_generated_static_object_finalization_v1"
EMITTER_MEASUREMENT_SCHEMA = "avengine_asset_emitter_measurement_v2"
WATERTIGHT_MANIFEST_SCHEMA = "avengine_watertight_textured_runtime_proxy_v1"
HEADING_EVIDENCE_SCHEMA = "avengine_static_heading_review_v1"
ANCHOR_AUTHORITY_SCHEMA = "avengine_static_emitter_anchor_authority_v1"
ANCHOR_SPEC_SCHEMA = "avengine_static_emitter_anchor_spec_v1"
MARKER_VISUAL_APPROVAL_SCHEMA = (
    "avengine_m6_static_object_marker_visual_approval_v1"
)
STATIC_ADMISSION_EVIDENCE_SCHEMA = (
    "avengine_m6_static_object_admission_evidence_v1"
)
STATIC_ROUTE = "flux2_pixal3d_static_v1"
STATIC_GENERATION_PLAN_SCHEMA = "flux2_pixal3d_static_generation_plan_v1"

CANONICAL_COORDINATE_SYSTEM = {
    "id": "avengine_local_x_forward_y_up_z_right_m",
    "handedness": "right_handed",
    "forward_axis": [1.0, 0.0, 0.0],
    "up_axis": [0.0, 1.0, 0.0],
    "right_axis": [0.0, 0.0, 1.0],
    "right_rule": "right_equals_forward_cross_up",
}
_STATIC_BASE_ACQUISITION_POLICY = {
    "policy_id": "static_object_per_request_one_shot_v1",
    "acquisition_unit": "one_frozen_asset_per_request",
    "sampled_domains_must_be_singleton": False,
    "downstream_instance_route": STATIC_ROUTE,
    "profile_validation": (
        "all_predeclared_requests_count_zero_hidden_failures"
    ),
}
_STATIC_EXECUTION_GATE = {
    "before_flux2": "authenticated_preflight_passed",
    "before_pixal3d": "approved_2d_review_for_exact_candidate_sha256",
    "before_source_asset_v2": "all_required_static_ue_audio_qa_passed",
}

_STATIC_REVIEW_CHECKS = {
    "silhouette_and_category_identity",
    "emitter_feature_visible",
    "material_and_declared_attributes",
    "physically_plausible_construction",
    "no_disconnected_or_floating_parts",
}
_STAGE_PYTHON_TOOLS = {
    "watertight": "blender_create_watertight_textured_proxy_mesh.py",
    "finalization": "blender_finalize_generated_static_object.py",
    "emitter_measurement": "blender_measure_generated_static_emitter.py",
}
_STAGE_PYTHON_DEPENDENCIES = {
    "watertight": frozenset(),
    "finalization": frozenset({"generated_asset_emitter_contract.py"}),
    "emitter_measurement": frozenset({"generated_asset_emitter_contract.py"}),
}

_STATIC_DECISION_FIELDS = {
    "schema",
    "instance_id",
    "review_sha256",
    "decision",
    "checks",
    "attribute_evidence",
    "caveats",
    "notes",
    "asset_class",
    "route",
    "request_sha256",
    "profile_sha256",
    "target_physical_profile",
    "pixal_output",
    "review",
    "physical_scale",
    "canonical_heading",
    "state_classification",
    "formal_dataset_registration_authorized",
    "next_gate",
    "decision_sha256",
}
_STATIC_REVIEW_FIELDS = {
    "schema",
    "instance_id",
    "execution_job_id",
    "request_sha256",
    "profile_schema_id",
    "profile_sha256",
    "asset_class",
    "route",
    "sampled_attributes",
    "target_physical_profile",
    "physical_scale",
    "orientation",
    "pixal_output",
    "mesh_readback",
    "reference_rgba",
    "raw_pbr_render_manifest",
    "raw_pbr_views",
    "raw_pbr_blender_log",
    "clay_geometry",
    "contact_sheet",
    "state_classification",
    "formal_dataset_registration_authorized",
    "automatic_checks",
    "visual_qa",
    "next_gate",
    "review_sha256",
}
_STAGE_NAMES = ("watertight", "finalization", "emitter_measurement")
_WATERTIGHT_PARAMETER_FIELDS = {
    "voxel_resolution",
    "target_faces",
    "smooth_iterations",
    "shrinkwrap_strength",
    "post_shrinkwrap_smooth_iterations",
    "torso_fold_repair_iterations",
    "attribute_transfer_backend",
    "bake_resolution",
    "base_color_encoding_policy",
    "base_color_gain",
    "double_sided",
}
_REVIEW_VIEW_MAPPING = {
    "orbit_anchor": "front",
    "orbit_opposite": "back",
    "orbit_right": "side",
    "orbit_top": "top",
    "orbit_quarter": "quarter",
}
_REVIEW_VIEW_KEYS = set(_REVIEW_VIEW_MAPPING)
_STATIC_ORIENTATION_CONTRACT = {
    "reference_facing": {
        "role": "appearance_category_and_emitter_reference",
        "source_view": "generated_three_quarter_product_view",
        "canonical_heading_authority": False,
    },
    "review_orbit": {
        "frame": "source_glb_axes_review_only",
        "anchor_axis": "negative-y",
        "up_axis": "positive-z",
        "canonical_heading_authority": False,
        "view_mapping": _REVIEW_VIEW_MAPPING,
    },
    "canonical_heading": {
        "status": "deferred_to_static_finalization",
        "axis": None,
        "derived_from_reference_facing": False,
    },
}
_CONTROLLED_REQUEST_FIELDS = {
    "execution_job_id",
    "instance_id",
    "request_sha256",
    "generation_seed",
    "profile_schema_id",
    "profile_sha256",
    "asset_class",
    "route",
    "sampled_attributes",
    "target_physical_profile",
    "rig_profile",
}


class StaticObjectRegistrationError(ValueError):
    """Static-object evidence or registry publication failed validation."""


@dataclass(frozen=True)
class AuthenticatedArtifact:
    path: Path
    sha256: str
    size_bytes: int

    def record(self, kind: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
        if kind is not None:
            result = {"kind": kind, **result}
        return result


@dataclass(frozen=True)
class _GLBSurface:
    name: str
    vertices: tuple[tuple[float, float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class _GLBReadback:
    surfaces: tuple[_GLBSurface, ...]
    vertex_count: int
    triangle_count: int
    material_names: tuple[str, ...]
    image_count: int
    minimum_m: tuple[float, float, float]
    maximum_m: tuple[float, float, float]
    has_complete_uvs: bool
    has_complete_materials: bool

    @property
    def mesh_count(self) -> int:
        return len(self.surfaces)


@dataclass(frozen=True)
class ValidatedStaticObjectAdmission:
    """Authenticated inputs needed to build one research registry row."""

    instance_id: str
    request_sha256: str
    profile_sha256: str
    decision: Mapping[str, Any]
    finalization: Mapping[str, Any]
    measurement: Mapping[str, Any]
    marker_approval: Mapping[str, Any]
    admission_batch_artifact: AuthenticatedArtifact
    job_receipt_artifact: AuthenticatedArtifact
    watertight_stage_receipt_artifact: AuthenticatedArtifact
    finalization_stage_receipt_artifact: AuthenticatedArtifact
    emitter_stage_receipt_artifact: AuthenticatedArtifact
    decision_artifact: AuthenticatedArtifact
    review_artifact: AuthenticatedArtifact
    pixal_glb: AuthenticatedArtifact
    watertight_glb: AuthenticatedArtifact
    watertight_manifest_artifact: AuthenticatedArtifact
    finalization_artifact: AuthenticatedArtifact
    measurement_artifact: AuthenticatedArtifact
    marker_approval_artifact: AuthenticatedArtifact
    finalized_glb: AuthenticatedArtifact
    marker_glb: AuthenticatedArtifact
    anchor_id: str
    semantic_role: str
    emitter_offset_m: tuple[float, float, float]
    target_height_m: float
    readback_height_m: float

    @property
    def entity_revision(self) -> str:
        return f"spear_static_{self.admission_evidence()['evidence_content_sha256']}"

    def admission_evidence(self) -> dict[str, Any]:
        by_kind = {
            "emitter_marker_glb": self.marker_glb,
            "marker_visual_approval": self.marker_approval_artifact,
            "spear_static_admission_batch": self.admission_batch_artifact,
            "spear_static_admission_job_receipt": self.job_receipt_artifact,
            "spear_static_emitter_stage_receipt": self.emitter_stage_receipt_artifact,
            "spear_static_finalization_stage_receipt": (
                self.finalization_stage_receipt_artifact
            ),
            "spear_static_watertight_stage_receipt": (
                self.watertight_stage_receipt_artifact
            ),
            "visual_asset_glb": self.finalized_glb,
        }
        evidence: dict[str, Any] = {
            "schema": STATIC_ADMISSION_EVIDENCE_SCHEMA,
            "pipeline": f"spear_{STATIC_ROUTE}",
            "status": "passed_research_registration",
            "identity": {
                "instance_id": self.instance_id,
                "request_sha256": self.request_sha256,
                "profile_sha256": self.profile_sha256,
            },
            "coordinate_system_id": CANONICAL_COORDINATE_SYSTEM["id"],
            "emitter_anchor_id": self.anchor_id,
            "artifacts": [
                by_kind[kind].record(kind) for kind in STATIC_OBJECT_EVIDENCE_KINDS
            ],
            "formal_dataset_registration_authorized": False,
        }
        evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
        return evidence


def _fail(message: str) -> None:
    raise StaticObjectRegistrationError(message)


def _require_mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{owner} must be a JSON object")
    return value


def _require_stable_id(value: Any, *, owner: str) -> str:
    if not is_stable_id(value):
        _fail(f"{owner} must be a stable identifier")
    return str(value)


def _require_sha256(value: Any, *, owner: str) -> str:
    if not is_sha256(value):
        _fail(f"{owner} must be a lowercase SHA-256")
    return str(value)


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _fail(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{owner} must be a finite number")
    return result


def _is_exact_int(value: Any, expected: int) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value == expected
    )


def _is_exact_finite_number(value: Any, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
        and float(value) == expected
    )


def _finite_vector(
    value: Any, length: int, *, owner: str
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"{owner} must contain exactly {length} finite numbers")
    if len(value) != length:
        _fail(f"{owner} must contain exactly {length} finite numbers")
    return tuple(
        _finite_number(item, owner=f"{owner}[{index}]")
        for index, item in enumerate(value)
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _artifact_for_path(
    policy: WorkspacePathPolicy,
    raw: str | Path,
    *,
    owner: str,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    allow_empty: bool = False,
) -> AuthenticatedArtifact:
    try:
        unresolved = Path(raw).expanduser()
        if not unresolved.is_absolute():
            _fail(f"{owner} path must be absolute")
        if unresolved.is_symlink():
            _fail(f"{owner} must not be a symbolic link")
        path = policy.resolve_input(
            unresolved,
            owner=owner,
            kind="file",
            expected_sha256=expected_sha256,
            allow_empty=allow_empty,
        )
    except (OSError, PathPolicyError) as error:
        raise StaticObjectRegistrationError(str(error)) from error
    size = path.stat().st_size
    if expected_size is not None and (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < (0 if allow_empty else 1)
        or size != expected_size
    ):
        _fail(f"{owner} size changed")
    if expected_sha256 is None:
        try:
            snapshot = policy.snapshot_file(path, owner=owner)
        except (OSError, PathPolicyError) as error:
            raise StaticObjectRegistrationError(str(error)) from error
        digest = snapshot.sha256
    else:
        digest = expected_sha256
    return AuthenticatedArtifact(path=path, sha256=digest, size_bytes=size)


def _file_record(
    value: Any,
    policy: WorkspacePathPolicy,
    *,
    owner: str,
    expected: AuthenticatedArtifact | None = None,
    base: Path | None = None,
    require_within: Path | None = None,
    extra_fields: frozenset[str] = frozenset(),
    allow_empty: bool = False,
) -> AuthenticatedArtifact:
    record = _require_mapping(value, owner=f"{owner} record")
    if set(record) != {"path", "sha256", "size_bytes"} | set(extra_fields):
        _fail(f"{owner} file record fields are invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        _fail(f"{owner} path must be a non-empty string")
    unresolved = Path(raw_path).expanduser()
    if not unresolved.is_absolute():
        if base is None:
            _fail(f"{owner} relative path has no declared base")
        unresolved = base / unresolved
    digest = _require_sha256(record.get("sha256"), owner=f"{owner} sha256")
    artifact = _artifact_for_path(
        policy,
        unresolved,
        owner=owner,
        expected_sha256=digest,
        expected_size=record.get("size_bytes"),
        allow_empty=allow_empty,
    )
    if require_within is not None:
        root = require_within.resolve(strict=True)
        try:
            artifact.path.relative_to(root)
        except ValueError:
            _fail(f"{owner} escaped its immutable root")
    if expected is not None and artifact != expected:
        _fail(f"{owner} does not match its upstream authority")
    return artifact


def _json_file_record(
    value: Any,
    policy: WorkspacePathPolicy,
    *,
    owner: str,
    expected: AuthenticatedArtifact | None = None,
    base: Path | None = None,
    require_within: Path | None = None,
    extra_fields: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], AuthenticatedArtifact]:
    artifact = _file_record(
        value,
        policy,
        owner=owner,
        expected=expected,
        base=base,
        require_within=require_within,
        extra_fields=extra_fields,
    )
    return _json_artifact(
        policy,
        artifact.path,
        owner=owner,
        expected_sha256=artifact.sha256,
    )


def _require_same_file_content(
    actual: AuthenticatedArtifact,
    expected: AuthenticatedArtifact,
    *,
    owner: str,
) -> None:
    if (
        actual.sha256 != expected.sha256
        or actual.size_bytes != expected.size_bytes
    ):
        _fail(f"{owner} bytes differ from upstream authority")


def _validate_png_artifact(
    artifact: AuthenticatedArtifact,
    *,
    owner: str,
    expected_size: tuple[int, int] | None = None,
    expected_mode: str | None = None,
    expected_extrema: tuple[int, int] | None = None,
) -> None:
    try:
        with Image.open(artifact.path) as image:
            image.load()
            image_format = image.format
            image_size = image.size
            image_mode = image.mode
            image_extrema = image.getextrema()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise StaticObjectRegistrationError(
            f"{owner} is not a decodable PNG: {error}"
        ) from error
    if image_format != "PNG":
        _fail(f"{owner} must be PNG")
    if expected_size is not None and image_size != expected_size:
        _fail(f"{owner} dimensions changed")
    if expected_mode is not None and image_mode != expected_mode:
        _fail(f"{owner} pixel mode changed")
    if expected_extrema is not None and image_extrema != expected_extrema:
        _fail(f"{owner} pixel extrema changed")


def _validate_rgba_alpha_binding(
    rgba_artifact: AuthenticatedArtifact,
    alpha_artifact: AuthenticatedArtifact,
    *,
    owner: str,
) -> None:
    try:
        with Image.open(rgba_artifact.path) as rgba_image:
            rgba_image.load()
            alpha_channel = rgba_image.getchannel("A").tobytes()
        with Image.open(alpha_artifact.path) as alpha_image:
            alpha_image.load()
            alpha_pixels = alpha_image.tobytes()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise StaticObjectRegistrationError(
            f"{owner} alpha binding is unreadable: {error}"
        ) from error
    if alpha_channel != alpha_pixels:
        _fail(f"{owner} alpha channel differs from its ISNet mask")


def _validate_rgba_source_rgb_binding(
    source_artifact: AuthenticatedArtifact,
    rgba_artifact: AuthenticatedArtifact,
    *,
    owner: str,
) -> None:
    try:
        with Image.open(source_artifact.path) as source_image:
            source_image.load()
            source_pixels = source_image.tobytes()
        with Image.open(rgba_artifact.path) as rgba_image:
            rgba_image.load()
            rgba_pixels = rgba_image.convert("RGB").tobytes()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise StaticObjectRegistrationError(
            f"{owner} RGB binding is unreadable: {error}"
        ) from error
    if source_pixels != rgba_pixels:
        _fail(f"{owner} RGB channels differ from the approved FLUX candidate")


def _validate_alpha_foreground_readback(
    alpha_artifact: AuthenticatedArtifact,
    *,
    declared_fraction: float,
    declared_bbox: Sequence[int],
    owner: str,
) -> None:
    try:
        with Image.open(alpha_artifact.path) as alpha_image:
            alpha_image.load()
            histogram = alpha_image.histogram()
            pixel_count = alpha_image.width * alpha_image.height
            foreground_count = sum(histogram[128:])
            derived_fraction = foreground_count / pixel_count
            thresholded = alpha_image.point(
                lambda item: 255 if item >= 128 else 0
            )
            derived_bbox = thresholded.getbbox()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise StaticObjectRegistrationError(
            f"{owner} readback is unreadable: {error}"
        ) from error
    if (
        derived_bbox is None
        or not math.isclose(
            declared_fraction,
            derived_fraction,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or list(derived_bbox) != list(declared_bbox)
    ):
        _fail(f"{owner} fraction/bbox differs from its alpha pixels")


def _glb_accessor_values(
    document: Mapping[str, Any],
    binary: bytes | bytearray,
    accessor_index: Any,
    *,
    owner: str,
    expected_type: str,
    allowed_component_types: frozenset[int],
    require_declared_bounds: bool = False,
) -> tuple[Any, ...]:
    accessors = document.get("accessors")
    views = document.get("bufferViews")
    if not isinstance(accessors, list) or not isinstance(views, list):
        _fail(f"{owner} GLB lacks accessor/bufferView arrays")
    if (
        isinstance(accessor_index, bool)
        or not isinstance(accessor_index, int)
        or not 0 <= accessor_index < len(accessors)
    ):
        _fail(f"{owner} GLB references an invalid accessor")
    accessor = _require_mapping(
        accessors[accessor_index], owner=f"{owner} GLB accessor"
    )
    if (
        accessor.get("sparse") is not None
        or accessor.get("extensions") is not None
    ):
        _fail(f"{owner} GLB accessor extensions/sparse data are unsupported")
    component_type = accessor.get("componentType")
    if (
        component_type not in allowed_component_types
        or accessor.get("type") != expected_type
        or accessor.get("normalized") not in {None, False}
    ):
        _fail(f"{owner} GLB accessor layout changed")
    view_index = accessor.get("bufferView")
    if (
        isinstance(view_index, bool)
        or not isinstance(view_index, int)
        or not 0 <= view_index < len(views)
    ):
        _fail(f"{owner} GLB accessor has no embedded bufferView")
    view = _require_mapping(views[view_index], owner=f"{owner} GLB bufferView")
    if view.get("buffer") != 0 or view.get("extensions") is not None:
        _fail(f"{owner} GLB bufferView is compressed or externally backed")

    component_layout = {
        5121: ("B", 1),
        5123: ("H", 2),
        5125: ("I", 4),
        5126: ("f", 4),
    }
    component_format, component_size = component_layout[component_type]
    component_count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[expected_type]
    element_size = component_size * component_count
    count = accessor.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        _fail(f"{owner} GLB accessor count is invalid")
    accessor_offset = accessor.get("byteOffset", 0)
    view_offset = view.get("byteOffset", 0)
    view_length = view.get("byteLength")
    stride = view.get("byteStride", element_size)
    integer_fields = (accessor_offset, view_offset, view_length, stride)
    if (
        any(isinstance(item, bool) or not isinstance(item, int) for item in integer_fields)
        or accessor_offset < 0
        or view_offset < 0
        or view_length < 0
        or stride < element_size
        or stride % component_size != 0
    ):
        _fail(f"{owner} GLB accessor offsets/stride are invalid")
    first = view_offset + accessor_offset
    final = first + (count - 1) * stride + element_size
    if (
        accessor_offset + (count - 1) * stride + element_size > view_length
        or final > len(binary)
    ):
        _fail(f"{owner} GLB accessor exceeds its embedded buffer")

    layout = "<" + component_format * component_count
    result: list[Any] = []
    for item_index in range(count):
        values = struct.unpack_from(layout, binary, first + item_index * stride)
        if component_type == 5126 and not all(
            math.isfinite(float(item)) for item in values
        ):
            _fail(f"{owner} GLB accessor contains a non-finite value")
        result.append(values[0] if component_count == 1 else tuple(values))
    if require_declared_bounds:
        declared_minimum = _finite_vector(
            accessor.get("min"),
            component_count,
            owner=f"{owner} GLB accessor minimum",
        )
        declared_maximum = _finite_vector(
            accessor.get("max"),
            component_count,
            owner=f"{owner} GLB accessor maximum",
        )
        derived_minimum = tuple(
            min(
                float(item[axis])
                if isinstance(item, tuple)
                else float(item)
                for item in result
            )
            for axis in range(component_count)
        )
        derived_maximum = tuple(
            max(
                float(item[axis])
                if isinstance(item, tuple)
                else float(item)
                for item in result
            )
            for axis in range(component_count)
        )
        if any(
            not math.isclose(
                declared,
                derived,
                rel_tol=1.0e-7,
                abs_tol=1.0e-7,
            )
            for declared_values, derived_values in (
                (declared_minimum, derived_minimum),
                (declared_maximum, derived_maximum),
            )
            for declared, derived in zip(declared_values, derived_values)
        ):
            _fail(f"{owner} GLB accessor bounds differ from its values")
    return tuple(result)


def _glb_default_scene_nodes(
    document: Mapping[str, Any],
    nodes: Sequence[Any],
    *,
    owner: str,
) -> tuple[int, ...]:
    scenes = document.get("scenes")
    default_scene = document.get("scene")
    if (
        not isinstance(scenes, list)
        or len(scenes) != 1
        or isinstance(default_scene, bool)
        or default_scene != 0
    ):
        _fail(f"{owner} GLB must have exactly one explicit default scene")
    scene = _require_mapping(scenes[0], owner=f"{owner} GLB default scene")
    if scene.get("extensions") is not None:
        _fail(f"{owner} GLB default-scene extensions are unsupported")
    roots = scene.get("nodes")
    if not isinstance(roots, list) or not roots:
        _fail(f"{owner} GLB default scene has no root nodes")

    parents: list[int | None] = [None] * len(nodes)
    children_by_node: list[tuple[int, ...]] = []
    for node_index, raw_node in enumerate(nodes):
        node = _require_mapping(raw_node, owner=f"{owner} GLB node")
        if node.get("extensions") is not None:
            _fail(f"{owner} GLB node extensions are unsupported")
        children = node.get("children", [])
        if not isinstance(children, list):
            _fail(f"{owner} GLB node children are invalid")
        strict_children: list[int] = []
        for child in children:
            if (
                isinstance(child, bool)
                or not isinstance(child, int)
                or not 0 <= child < len(nodes)
                or child in strict_children
                or parents[child] is not None
            ):
                _fail(f"{owner} GLB node graph is invalid or multiply parented")
            parents[child] = node_index
            strict_children.append(child)
        children_by_node.append(tuple(strict_children))

    strict_roots: list[int] = []
    for root in roots:
        if (
            isinstance(root, bool)
            or not isinstance(root, int)
            or not 0 <= root < len(nodes)
            or root in strict_roots
            or parents[root] is not None
        ):
            _fail(f"{owner} GLB default-scene root is invalid")
        strict_roots.append(root)

    reachable: set[int] = set()
    visiting: set[int] = set()

    def visit(node_index: int) -> None:
        if node_index in visiting:
            _fail(f"{owner} GLB node graph contains a cycle")
        if node_index in reachable:
            _fail(f"{owner} GLB default scene repeats a node")
        visiting.add(node_index)
        for child in children_by_node[node_index]:
            visit(child)
        visiting.remove(node_index)
        reachable.add(node_index)

    for root in strict_roots:
        visit(root)
    if reachable != set(range(len(nodes))):
        _fail(f"{owner} GLB contains nodes outside its default scene")
    return tuple(sorted(reachable))


def _read_glb_geometry(
    artifact: AuthenticatedArtifact,
    *,
    owner: str,
) -> _GLBReadback:
    try:
        document, binary = _load_glb_document(artifact.path)
    except (OSError, ReplicaCADGLBError, TypeError, ValueError) as error:
        raise StaticObjectRegistrationError(
            f"{owner} is not a supported complete GLB 2.0: {error}"
        ) from error

    required_extensions = document.get("extensionsRequired", [])
    used_extensions = document.get("extensionsUsed", [])
    if (
        not isinstance(required_extensions, list)
        or required_extensions
        or not isinstance(used_extensions, list)
        or used_extensions
        or document.get("extensions") is not None
    ):
        _fail(f"{owner} uses an unsupported GLB extension")
    skins = document.get("skins", [])
    animations = document.get("animations", [])
    if (
        not isinstance(skins, list)
        or skins
        or not isinstance(animations, list)
        or animations
    ):
        _fail(f"{owner} is not a rigid, animation-free GLB")

    nodes = document.get("nodes", [])
    meshes = document.get("meshes", [])
    materials = document.get("materials", [])
    textures = document.get("textures", [])
    samplers = document.get("samplers", [])
    images = document.get("images", [])
    views = document.get("bufferViews", [])
    if (
        not isinstance(nodes, list)
        or not isinstance(meshes, list)
        or not isinstance(materials, list)
        or not isinstance(textures, list)
        or not isinstance(samplers, list)
        or not isinstance(images, list)
        or not isinstance(views, list)
    ):
        _fail(f"{owner} GLB scene arrays are invalid")
    reachable_nodes = _glb_default_scene_nodes(document, nodes, owner=owner)
    try:
        world_matrices = _glb_node_world_matrices(document)
    except (ReplicaCADGLBError, TypeError, ValueError) as error:
        raise StaticObjectRegistrationError(
            f"{owner} is not a supported complete GLB 2.0: {error}"
        ) from error

    image_formats = {"image/png": "PNG", "image/jpeg": "JPEG"}
    for raw_image in images:
        image = _require_mapping(raw_image, owner=f"{owner} GLB image")
        if image.get("extensions") is not None:
            _fail(f"{owner} GLB image extensions are unsupported")
        image_uri = image.get("uri")
        view_index = image.get("bufferView")
        declared_mime = image.get("mimeType")
        image_mime = declared_mime
        payload: bytes
        if image_uri is not None:
            if (
                view_index is not None
                or not isinstance(image_uri, str)
                or not image_uri.startswith("data:image/")
                or ";base64," not in image_uri
            ):
                _fail(f"{owner} GLB image is not self-contained")
            header, encoded = image_uri.split(",", 1)
            image_mime = header[5:].removesuffix(";base64")
            if declared_mime is not None and declared_mime != image_mime:
                _fail(f"{owner} GLB image MIME declarations differ")
            try:
                payload = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as error:
                raise StaticObjectRegistrationError(
                    f"{owner} GLB image data URI is invalid: {error}"
                ) from error
        else:
            if (
                isinstance(view_index, bool)
                or not isinstance(view_index, int)
                or not 0 <= view_index < len(views)
            ):
                _fail(f"{owner} GLB image has no embedded bufferView")
            view = _require_mapping(
                views[view_index], owner=f"{owner} GLB image bufferView"
            )
            offset = view.get("byteOffset", 0)
            length = view.get("byteLength")
            if (
                view.get("buffer") != 0
                or view.get("extensions") is not None
                or isinstance(offset, bool)
                or not isinstance(offset, int)
                or isinstance(length, bool)
                or not isinstance(length, int)
                or offset < 0
                or length <= 0
                or offset + length > len(binary)
            ):
                _fail(f"{owner} GLB image bufferView is invalid")
            payload = bytes(binary[offset : offset + length])
        if image_mime not in image_formats:
            _fail(f"{owner} GLB image MIME type is unsupported")
        try:
            with Image.open(BytesIO(payload)) as decoded_image:
                decoded_image.load()
                if decoded_image.width <= 0 or decoded_image.height <= 0:
                    _fail(f"{owner} GLB image dimensions are invalid")
                if decoded_image.format != image_formats[image_mime]:
                    _fail(f"{owner} GLB image MIME differs from decoded bytes")
        except (
            OSError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
        ) as error:
            raise StaticObjectRegistrationError(
                f"{owner} GLB image is not decodable: {error}"
            ) from error

    for sampler_index, raw_sampler in enumerate(samplers):
        sampler = _require_mapping(
            raw_sampler, owner=f"{owner} GLB sampler {sampler_index}"
        )
        if sampler.get("extensions") is not None:
            _fail(f"{owner} GLB sampler extensions are unsupported")
        if sampler.get("magFilter", 9729) not in {9728, 9729}:
            _fail(f"{owner} GLB sampler magFilter is invalid")
        if sampler.get("minFilter", 9987) not in {
            9728,
            9729,
            9984,
            9985,
            9986,
            9987,
        }:
            _fail(f"{owner} GLB sampler minFilter is invalid")
        for field in ("wrapS", "wrapT"):
            if sampler.get(field, 10497) not in {33071, 33648, 10497}:
                _fail(f"{owner} GLB sampler wrap mode is invalid")

    texture_sources: list[int] = []
    for texture_index, raw_texture in enumerate(textures):
        texture = _require_mapping(
            raw_texture, owner=f"{owner} GLB texture {texture_index}"
        )
        source_index = texture.get("source")
        sampler_index = texture.get("sampler")
        if (
            texture.get("extensions") is not None
            or isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not 0 <= source_index < len(images)
            or (
                sampler_index is not None
                and (
                    isinstance(sampler_index, bool)
                    or not isinstance(sampler_index, int)
                    or not 0 <= sampler_index < len(samplers)
                )
            )
        ):
            _fail(f"{owner} GLB texture linkage is invalid")
        texture_sources.append(source_index)

    material_names: list[str] = []
    for index, raw_material in enumerate(materials):
        material = _require_mapping(
            raw_material, owner=f"{owner} GLB material"
        )
        name = material.get("name", f"material_{index}")
        if not isinstance(name, str) or not name:
            _fail(f"{owner} GLB material name is invalid")
        material_names.append(name)

    surfaces: list[_GLBSurface] = []
    complete_uvs = True
    complete_materials = True
    referenced_meshes: set[int] = set()
    referenced_materials: set[int] = set()
    for node_index in reachable_nodes:
        raw_node = nodes[node_index]
        node = _require_mapping(raw_node, owner=f"{owner} GLB node")
        if node.get("skin") is not None or node.get("weights") is not None:
            _fail(f"{owner} GLB mesh node has rig or morph weights")
        mesh_index = node.get("mesh")
        if mesh_index is None:
            continue
        if (
            isinstance(mesh_index, bool)
            or not isinstance(mesh_index, int)
            or not 0 <= mesh_index < len(meshes)
        ):
            _fail(f"{owner} GLB node references an invalid mesh")
        referenced_meshes.add(mesh_index)
        mesh = _require_mapping(meshes[mesh_index], owner=f"{owner} GLB mesh")
        if (
            mesh.get("weights") is not None
            or mesh.get("extensions") is not None
        ):
            _fail(f"{owner} GLB mesh has morph weights or extensions")
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list) or not primitives:
            _fail(f"{owner} GLB mesh has no primitives")
        surface_name = node.get("name", mesh.get("name", f"mesh_{mesh_index}"))
        if not isinstance(surface_name, str) or not surface_name:
            _fail(f"{owner} GLB mesh node has no stable name")
        if any(item.name == surface_name for item in surfaces):
            _fail(f"{owner} GLB repeats a mesh node name")

        surface_vertices: list[tuple[float, float, float]] = []
        surface_triangles: list[tuple[int, int, int]] = []
        matrix = world_matrices[node_index]
        for raw_primitive in primitives:
            primitive = _require_mapping(
                raw_primitive, owner=f"{owner} GLB primitive"
            )
            if primitive.get("targets") is not None:
                _fail(f"{owner} GLB primitive has morph targets")
            if primitive.get("extensions") is not None:
                _fail(f"{owner} GLB primitive extensions are unsupported")
            if primitive.get("mode", 4) != 4:
                _fail(f"{owner} GLB primitive is not a triangle list")
            attributes = _require_mapping(
                primitive.get("attributes"),
                owner=f"{owner} GLB primitive attributes",
            )
            if set(attributes) != {"POSITION", "NORMAL", "TEXCOORD_0"}:
                _fail(f"{owner} GLB primitive attributes are not rigid core PBR")
            material_index = primitive.get("material")
            if material_index is None:
                complete_materials = False
            elif (
                isinstance(material_index, bool)
                or not isinstance(material_index, int)
                or not 0 <= material_index < len(materials)
            ):
                _fail(f"{owner} GLB primitive references an invalid material")
            else:
                referenced_materials.add(material_index)

            positions = _glb_accessor_values(
                document,
                binary,
                attributes.get("POSITION"),
                owner=owner,
                expected_type="VEC3",
                allowed_component_types=frozenset({5126}),
                require_declared_bounds=True,
            )
            normals = _glb_accessor_values(
                document,
                binary,
                attributes["NORMAL"],
                owner=owner,
                expected_type="VEC3",
                allowed_component_types=frozenset({5126}),
            )
            texcoords = _glb_accessor_values(
                document,
                binary,
                attributes["TEXCOORD_0"],
                owner=owner,
                expected_type="VEC2",
                allowed_component_types=frozenset({5126}),
            )
            if (
                len(normals) != len(positions)
                or len(texcoords) != len(positions)
                or any(
                    not math.isclose(
                        math.sqrt(sum(float(axis) ** 2 for axis in normal)),
                        1.0,
                        rel_tol=1.0e-4,
                        abs_tol=1.0e-4,
                    )
                    for normal in normals
                )
            ):
                _fail(f"{owner} GLB NORMAL/UV/POSITION values differ")
            if primitive.get("indices") is None:
                indices = tuple(range(len(positions)))
            else:
                indices = _glb_accessor_values(
                    document,
                    binary,
                    primitive["indices"],
                    owner=owner,
                    expected_type="SCALAR",
                    allowed_component_types=frozenset({5121, 5123, 5125}),
                )
            if len(indices) % 3 != 0 or any(
                not 0 <= int(item) < len(positions) for item in indices
            ):
                _fail(f"{owner} GLB triangle indices are invalid")
            vertex_base = len(surface_vertices)
            surface_vertices.extend(
                tuple(float(item) for item in _transform_glb_point(matrix, point))
                for point in positions
            )
            surface_triangles.extend(
                (
                    vertex_base + int(indices[index]),
                    vertex_base + int(indices[index + 1]),
                    vertex_base + int(indices[index + 2]),
                )
                for index in range(0, len(indices), 3)
            )
        surfaces.append(
            _GLBSurface(
                name=surface_name,
                vertices=tuple(surface_vertices),
                triangles=tuple(surface_triangles),
            )
        )

    if (
        not surfaces
        or referenced_meshes != set(range(len(meshes)))
        or referenced_materials != set(range(len(materials)))
    ):
        _fail(f"{owner} GLB contains missing or unreachable mesh geometry")

    used_textures: set[int] = set()
    texture_info_fields = (
        "normalTexture",
        "occlusionTexture",
        "emissiveTexture",
    )

    def validate_texture_info(
        value: Any,
        *,
        field_owner: str,
        scalar_field: str | None = None,
        scalar_maximum: float | None = None,
    ) -> None:
        texture_info = _require_mapping(value, owner=field_owner)
        texture_index = texture_info.get("index")
        texcoord = texture_info.get("texCoord", 0)
        if (
            texture_info.get("extensions") is not None
            or isinstance(texture_index, bool)
            or not isinstance(texture_index, int)
            or not 0 <= texture_index < len(textures)
            or isinstance(texcoord, bool)
            or not isinstance(texcoord, int)
            or texcoord != 0
        ):
            _fail(f"{field_owner} linkage is invalid")
        if scalar_field is not None:
            scalar = _finite_number(
                texture_info.get(scalar_field, 1.0),
                owner=f"{field_owner} {scalar_field}",
            )
            if scalar < 0.0 or (
                scalar_maximum is not None and scalar > scalar_maximum
            ):
                _fail(f"{field_owner} {scalar_field} is out of range")
        used_textures.add(texture_index)

    for material_index in sorted(referenced_materials):
        material = _require_mapping(
            materials[material_index],
            owner=f"{owner} GLB material {material_index}",
        )
        if material.get("extensions") is not None:
            _fail(f"{owner} GLB material extensions are unsupported")
        pbr = _require_mapping(
            material.get("pbrMetallicRoughness"),
            owner=f"{owner} GLB material base PBR",
        )
        if pbr.get("extensions") is not None:
            _fail(f"{owner} GLB base PBR extensions are unsupported")
        base_color_factor = _finite_vector(
            pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]),
            4,
            owner=f"{owner} GLB material baseColorFactor",
        )
        emissive_factor = _finite_vector(
            material.get("emissiveFactor", [0.0, 0.0, 0.0]),
            3,
            owner=f"{owner} GLB material emissiveFactor",
        )
        metallic = _finite_number(
            pbr.get("metallicFactor", 1.0),
            owner=f"{owner} GLB material metallicFactor",
        )
        roughness = _finite_number(
            pbr.get("roughnessFactor", 1.0),
            owner=f"{owner} GLB material roughnessFactor",
        )
        alpha_cutoff = _finite_number(
            material.get("alphaCutoff", 0.5),
            owner=f"{owner} GLB material alphaCutoff",
        )
        if (
            any(not 0.0 <= item <= 1.0 for item in base_color_factor)
            or any(not 0.0 <= item <= 1.0 for item in emissive_factor)
            or not 0.0 <= metallic <= 1.0
            or not 0.0 <= roughness <= 1.0
            or not 0.0 <= alpha_cutoff <= 1.0
            or material.get("alphaMode", "OPAQUE")
            not in {"OPAQUE", "MASK", "BLEND"}
            or not isinstance(material.get("doubleSided", False), bool)
        ):
            _fail(f"{owner} GLB material PBR values are invalid")
        validate_texture_info(
            pbr.get("baseColorTexture"),
            field_owner=f"{owner} GLB material baseColorTexture",
        )
        if pbr.get("metallicRoughnessTexture") is not None:
            validate_texture_info(
                pbr["metallicRoughnessTexture"],
                field_owner=f"{owner} GLB material metallicRoughnessTexture",
            )
        for field in texture_info_fields:
            if material.get(field) is not None:
                validate_texture_info(
                    material[field],
                    field_owner=f"{owner} GLB material {field}",
                    scalar_field=(
                        "scale"
                        if field == "normalTexture"
                        else "strength"
                        if field == "occlusionTexture"
                        else None
                    ),
                    scalar_maximum=(
                        1.0 if field == "occlusionTexture" else None
                    ),
                )

    used_images = {texture_sources[index] for index in used_textures}
    if (
        used_textures != set(range(len(textures)))
        or used_images != set(range(len(images)))
    ):
        _fail(f"{owner} GLB contains unused or unbound texture/image data")

    all_points = [
        point for surface in surfaces for point in surface.vertices
    ]
    minimum = tuple(min(point[axis] for point in all_points) for axis in range(3))
    maximum = tuple(max(point[axis] for point in all_points) for axis in range(3))
    if any(maximum[axis] <= minimum[axis] for axis in range(3)):
        _fail(f"{owner} GLB has degenerate bounds")
    return _GLBReadback(
        surfaces=tuple(surfaces),
        vertex_count=sum(len(item.vertices) for item in surfaces),
        triangle_count=sum(len(item.triangles) for item in surfaces),
        material_names=tuple(material_names),
        image_count=len(images),
        minimum_m=minimum,
        maximum_m=maximum,
        has_complete_uvs=complete_uvs,
        has_complete_materials=complete_materials,
    )


def _glb_welded_topology(
    readback: _GLBReadback,
    *,
    owner: str,
) -> dict[str, int]:
    vertex_ids: dict[tuple[float, float, float], int] = {}
    edge_uses: dict[tuple[int, int], list[tuple[int, int]]] = {}
    face_count = 0
    for surface in readback.surfaces:
        welded = []
        for point in surface.vertices:
            welded.append(vertex_ids.setdefault(point, len(vertex_ids)))
        for triangle in surface.triangles:
            indices = tuple(welded[index] for index in triangle)
            if len(set(indices)) != 3:
                _fail(f"{owner} GLB contains a degenerate triangle")
            first, second, third = (
                surface.vertices[index] for index in triangle
            )
            cross = (
                (second[1] - first[1]) * (third[2] - first[2])
                - (second[2] - first[2]) * (third[1] - first[1]),
                (second[2] - first[2]) * (third[0] - first[0])
                - (second[0] - first[0]) * (third[2] - first[2]),
                (second[0] - first[0]) * (third[1] - first[1])
                - (second[1] - first[1]) * (third[0] - first[0]),
            )
            if sum(item * item for item in cross) <= 1.0e-24:
                _fail(f"{owner} GLB contains a zero-area triangle")
            for left, right in (
                (indices[0], indices[1]),
                (indices[1], indices[2]),
                (indices[2], indices[0]),
            ):
                key = (min(left, right), max(left, right))
                edge_uses.setdefault(key, []).append((left, right))
            face_count += 1
    boundary = sum(len(uses) == 1 for uses in edge_uses.values())
    nonmanifold = sum(len(uses) > 2 for uses in edge_uses.values())
    noncontiguous = sum(
        len(uses) == 2 and uses[0] == uses[1]
        for uses in edge_uses.values()
    )
    return {
        "vertices": len(vertex_ids),
        "edges": len(edge_uses),
        "triangles": face_count,
        "boundary_edges": boundary,
        "wire_edges": 0,
        "nonmanifold_edges_over_two_faces": nonmanifold,
        "noncontiguous_two_face_edges": noncontiguous,
    }


def _closest_point_on_triangle(
    point: Sequence[float],
    first: Sequence[float],
    second: Sequence[float],
    third: Sequence[float],
) -> tuple[float, float, float]:
    # Ericson's region tests, expressed without a geometry dependency.
    ab = tuple(second[index] - first[index] for index in range(3))
    ac = tuple(third[index] - first[index] for index in range(3))
    ap = tuple(point[index] - first[index] for index in range(3))
    d1 = sum(ab[index] * ap[index] for index in range(3))
    d2 = sum(ac[index] * ap[index] for index in range(3))
    if d1 <= 0.0 and d2 <= 0.0:
        return tuple(float(item) for item in first)
    bp = tuple(point[index] - second[index] for index in range(3))
    d3 = sum(ab[index] * bp[index] for index in range(3))
    d4 = sum(ac[index] * bp[index] for index in range(3))
    if d3 >= 0.0 and d4 <= d3:
        return tuple(float(item) for item in second)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        fraction = d1 / (d1 - d3)
        return tuple(first[index] + fraction * ab[index] for index in range(3))
    cp = tuple(point[index] - third[index] for index in range(3))
    d5 = sum(ab[index] * cp[index] for index in range(3))
    d6 = sum(ac[index] * cp[index] for index in range(3))
    if d6 >= 0.0 and d5 <= d6:
        return tuple(float(item) for item in third)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        fraction = d2 / (d2 - d6)
        return tuple(first[index] + fraction * ac[index] for index in range(3))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        edge = tuple(third[index] - second[index] for index in range(3))
        fraction = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return tuple(second[index] + fraction * edge[index] for index in range(3))
    denominator = 1.0 / (va + vb + vc)
    second_weight = vb * denominator
    third_weight = vc * denominator
    return tuple(
        first[index]
        + ab[index] * second_weight
        + ac[index] * third_weight
        for index in range(3)
    )


def _json_path_hash_record(
    value: Any,
    policy: WorkspacePathPolicy,
    *,
    owner: str,
    content_hash_field: str,
    base: Path | None = None,
    require_within: Path | None = None,
) -> tuple[dict[str, Any], AuthenticatedArtifact]:
    record = _require_mapping(value, owner=f"{owner} record")
    if set(record) != {"path", "sha256", content_hash_field}:
        _fail(f"{owner} path/hash record fields are invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        _fail(f"{owner} path must be a non-empty string")
    unresolved = Path(raw_path).expanduser()
    if not unresolved.is_absolute():
        if base is None:
            _fail(f"{owner} relative path has no declared base")
        unresolved = base / unresolved
    artifact = _artifact_for_path(
        policy,
        unresolved,
        owner=owner,
        expected_sha256=_require_sha256(
            record.get("sha256"), owner=f"{owner} sha256"
        ),
    )
    if require_within is not None:
        root = require_within.resolve(strict=True)
        try:
            artifact.path.relative_to(root)
        except ValueError:
            _fail(f"{owner} escaped its immutable root")
    payload, confirmed = _json_artifact(
        policy,
        artifact.path,
        owner=owner,
        expected_sha256=artifact.sha256,
    )
    inner_hash = _require_sha256(
        record.get(content_hash_field),
        owner=f"{owner}.{content_hash_field}",
    )
    if payload.get(content_hash_field) != inner_hash:
        _fail(f"{owner} inner content hash differs from its record")
    return payload, confirmed


def _require_content_hash(
    value: Mapping[str, Any], *, field: str, owner: str
) -> str:
    declared = _require_sha256(value.get(field), owner=f"{owner}.{field}")
    payload = {key: item for key, item in value.items() if key != field}
    if declared != canonical_json_sha256(payload):
        _fail(f"{owner} canonical content hash changed")
    return declared


def _require_nonnegative_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{owner} must be a nonnegative integer")
    return value


def _json_artifact(
    policy: WorkspacePathPolicy,
    raw: str | Path,
    *,
    owner: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], AuthenticatedArtifact]:
    artifact = _artifact_for_path(
        policy,
        raw,
        owner=owner,
        expected_sha256=expected_sha256,
    )
    try:
        value = json.loads(artifact.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StaticObjectRegistrationError(f"{owner} is invalid JSON") from error
    if not isinstance(value, dict):
        _fail(f"{owner} must be a JSON object")
    # Authenticate again after parsing so a trusted-workspace concurrent edit
    # cannot silently turn parsed bytes into a different published reference.
    confirmed = _artifact_for_path(
        policy,
        artifact.path,
        owner=owner,
        expected_sha256=artifact.sha256,
        expected_size=artifact.size_bytes,
    )
    return value, confirmed


def _require_identity(
    value: Mapping[str, Any],
    *,
    owner: str,
    expected: tuple[str, str, str] | None = None,
) -> tuple[str, str, str]:
    identity = (
        _require_stable_id(value.get("instance_id"), owner=f"{owner}.instance_id"),
        _require_sha256(
            value.get("request_sha256"), owner=f"{owner}.request_sha256"
        ),
        _require_sha256(
            value.get("profile_sha256"), owner=f"{owner}.profile_sha256"
        ),
    )
    if expected is not None and identity != expected:
        _fail(f"{owner} identity does not match upstream evidence")
    return identity


def _validate_coordinate_system(value: Any, *, owner: str) -> None:
    if value != CANONICAL_COORDINATE_SYSTEM:
        _fail(
            f"{owner} must be +X forward, +Y up, +Z right in the canonical "
            "right-handed frame"
        )
    forward = CANONICAL_COORDINATE_SYSTEM["forward_axis"]
    up = CANONICAL_COORDINATE_SYSTEM["up_axis"]
    cross = [
        forward[1] * up[2] - forward[2] * up[1],
        forward[2] * up[0] - forward[0] * up[2],
        forward[0] * up[1] - forward[1] * up[0],
    ]
    if cross != CANONICAL_COORDINATE_SYSTEM["right_axis"]:
        _fail(f"{owner} violates forward cross up = right")


def _validate_anchor_selection(value: Any, *, owner: str) -> str:
    selection = _require_mapping(value, owner=owner)
    method = selection.get("method")
    samples = selection.get("samples")
    if (
        method
        not in {
            "mesh_surface_barycentric_samples_v1",
            "reviewed_bbox_fraction_nearest_surface_v1",
        }
        or selection.get("aggregation") != "weighted_centroid"
        or not isinstance(samples, list)
        or not samples
    ):
        _fail(f"{owner} aggregation/samples are invalid")
    total_weight = 0.0
    if method == "mesh_surface_barycentric_samples_v1":
        if set(selection) != {"method", "samples", "aggregation"}:
            _fail(f"{owner} barycentric fields are invalid")
        for index, raw_sample in enumerate(samples):
            sample = _require_mapping(
                raw_sample, owner=f"{owner}.samples[{index}]"
            )
            if set(sample) != {
                "mesh_name",
                "triangle_index",
                "barycentric",
                "weight",
            }:
                _fail(f"{owner} barycentric sample fields are invalid")
            if not isinstance(sample.get("mesh_name"), str) or not sample["mesh_name"]:
                _fail(f"{owner} mesh_name is invalid")
            triangle = sample.get("triangle_index")
            if (
                isinstance(triangle, bool)
                or not isinstance(triangle, int)
                or triangle < 0
            ):
                _fail(f"{owner} triangle_index is invalid")
            barycentric = _finite_vector(
                sample.get("barycentric"),
                3,
                owner=f"{owner} barycentric",
            )
            if (
                any(item < 0.0 or item > 1.0 for item in barycentric)
                or not math.isclose(sum(barycentric), 1.0, abs_tol=1.0e-6)
            ):
                _fail(f"{owner} barycentric weights are invalid")
            weight = _finite_number(
                sample.get("weight"), owner=f"{owner} sample weight"
            )
            if weight <= 0.0:
                _fail(f"{owner} sample weight must be positive")
            total_weight += weight
    else:
        if set(selection) != {
            "method",
            "samples",
            "aggregation",
            "maximum_search_distance_fraction",
        }:
            _fail(f"{owner} reviewed-bbox fields are invalid")
        maximum = _finite_number(
            selection.get("maximum_search_distance_fraction"),
            owner=f"{owner} maximum search distance fraction",
        )
        if not 0.0 < maximum <= 1.0:
            _fail(f"{owner} maximum search distance fraction is outside (0,1]")
        for index, raw_sample in enumerate(samples):
            sample = _require_mapping(
                raw_sample, owner=f"{owner}.samples[{index}]"
            )
            if set(sample) != {"target_fraction_xyz", "weight"}:
                _fail(f"{owner} reviewed-bbox sample fields are invalid")
            fraction = _finite_vector(
                sample.get("target_fraction_xyz"),
                3,
                owner=f"{owner} target fraction",
            )
            if any(item < 0.0 or item > 1.0 for item in fraction):
                _fail(f"{owner} target fractions are outside [0,1]")
            weight = _finite_number(
                sample.get("weight"), owner=f"{owner} sample weight"
            )
            if weight <= 0.0:
                _fail(f"{owner} sample weight must be positive")
            total_weight += weight
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        _fail(f"{owner} sample weights have zero mass")
    return str(method)


def _validate_watertight_parameters(value: Any) -> dict[str, Any]:
    parameters = _require_mapping(value, owner="watertight parameters")
    if set(parameters) != _WATERTIGHT_PARAMETER_FIELDS:
        _fail("watertight parameter fields are invalid")
    result = deepcopy(dict(parameters))
    integer_ranges = {
        "voxel_resolution": (96, 512),
        "target_faces": (10000, 1000000),
        "smooth_iterations": (0, 8),
        "post_shrinkwrap_smooth_iterations": (0, 8),
        "torso_fold_repair_iterations": (0, 20),
    }
    for field, (minimum, maximum) in integer_ranges.items():
        item = result[field]
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or not minimum <= item <= maximum
        ):
            _fail(f"watertight {field} must be in [{minimum}, {maximum}]")
    strength = _finite_number(
        result["shrinkwrap_strength"], owner="watertight shrinkwrap_strength"
    )
    if not 0.0 <= strength <= 1.0:
        _fail("watertight shrinkwrap_strength must be in [0,1]")
    result["shrinkwrap_strength"] = strength
    if result["attribute_transfer_backend"] not in {
        "bake",
        "bvh",
        "data-transfer",
    }:
        _fail("unsupported watertight attribute transfer backend")
    if result["bake_resolution"] not in {512, 1024, 2048, 4096}:
        _fail("unsupported watertight bake resolution")
    if result["base_color_encoding_policy"] not in {
        "preserve-bake",
        "srgb-to-linear",
    }:
        _fail("unsupported base-color encoding policy")
    gain = _finite_vector(
        result["base_color_gain"], 3, owner="watertight base_color_gain"
    )
    if any(not 0.0 < item <= 2.0 for item in gain):
        _fail("watertight base_color_gain values must be in (0,2]")
    result["base_color_gain"] = list(gain)
    if not isinstance(result["double_sided"], bool):
        _fail("watertight double_sided must be boolean")
    return result


def _validate_controlled_request(
    value: Any,
    *,
    instance_id: str,
) -> Mapping[str, Any]:
    controlled = _require_mapping(value, owner="Pixal controlled request")
    if (
        set(controlled) != _CONTROLLED_REQUEST_FIELDS
        or controlled.get("instance_id") != instance_id
        or controlled.get("asset_class") != "static_object"
        or controlled.get("route") != STATIC_ROUTE
        or controlled.get("rig_profile") is not None
        or not isinstance(controlled.get("sampled_attributes"), Mapping)
        or not controlled["sampled_attributes"]
        or not isinstance(controlled.get("target_physical_profile"), Mapping)
    ):
        _fail("Pixal controlled request contract is invalid")
    request_sha256 = _require_sha256(
        controlled.get("request_sha256"),
        owner="Pixal controlled request.request_sha256",
    )
    _require_sha256(
        controlled.get("profile_sha256"),
        owner="Pixal controlled request.profile_sha256",
    )
    if (
        not isinstance(controlled.get("profile_schema_id"), str)
        or not controlled["profile_schema_id"].strip()
    ):
        _fail("Pixal controlled request.profile_schema_id is invalid")
    seed = controlled.get("generation_seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < (1 << 63)
        or controlled.get("execution_job_id")
        != f"static_{request_sha256[:16]}"
    ):
        _fail("Pixal controlled request seed/execution identity is invalid")
    return controlled


def _validate_one_shot_policy_record(
    value: Any,
    workspace_policy: WorkspacePathPolicy,
    *,
    owner: str,
) -> Mapping[str, Any]:
    policy_record = _require_mapping(value, owner=owner)
    if (
        set(policy_record)
        != {"schema", "policy_id", "policy_schema", "path", "sha256"}
        or policy_record.get("schema") != ONE_SHOT_POLICY_RECORD_SCHEMA
        or policy_record.get("policy_id") != ONE_SHOT_POLICY_ID
        or policy_record.get("policy_schema") != ONE_SHOT_POLICY_SCHEMA
        or not isinstance(policy_record.get("path"), str)
        or not policy_record["path"]
        or not Path(policy_record["path"]).is_absolute()
    ):
        _fail(f"{owner} identity is invalid")
    policy_sha256 = _require_sha256(
        policy_record.get("sha256"), owner=f"{owner} sha256"
    )
    policy_payload, policy_artifact = _json_artifact(
        workspace_policy,
        policy_record["path"],
        owner=f"{owner} file",
        expected_sha256=policy_sha256,
    )
    request_freeze = _require_mapping(
        policy_payload.get("request_freeze"),
        owner=f"{owner} request freeze",
    )
    cardinality = _require_mapping(
        policy_payload.get("per_request_cardinality"),
        owner=f"{owner} cardinality",
    )
    failure = _require_mapping(
        policy_payload.get("failure_policy"),
        owner=f"{owner} failure policy",
    )
    qualification = _require_mapping(
        policy_payload.get("profile_qualification"),
        owner=f"{owner} profile qualification",
    )
    production = _require_mapping(
        policy_payload.get("production_instance_policy"),
        owner=f"{owner} production policy",
    )
    if (
        policy_artifact.path.name
        != "animal_one_shot_no_seed_lottery_v1.json"
        or policy_payload.get("schema") != ONE_SHOT_POLICY_SCHEMA
        or policy_payload.get("policy_id") != ONE_SHOT_POLICY_ID
        or request_freeze.get(
            "seed_override_after_generation_started_allowed"
        )
        is not False
        or request_freeze.get(
            "request_replacement_after_observing_output_allowed"
        )
        is not False
        or not _is_exact_int(cardinality.get("flux_invocations"), 1)
        or not _is_exact_int(
            cardinality.get("flux_images_per_invocation"), 1
        )
        or not _is_exact_int(cardinality.get("pixal3d_invocations"), 1)
        or cardinality.get("seed_retry_allowed") is not False
        or cardinality.get("candidate_ranking_or_best_of_n_allowed")
        is not False
        or failure.get("failed_output_may_be_hidden_from_profile_metrics")
        is not False
        or qualification.get("all_predeclared_requests_count") is not True
        or not _is_exact_finite_number(
            qualification.get("required_pass_fraction"), 1.0
        )
        or production.get(
            "rerun_flux_or_pixal_for_each_color_or_size_instance"
        )
        is not False
        or policy_payload.get("formal_dataset_registration_authorized")
        is not False
    ):
        _fail(f"{owner} file weakens the one-shot admission policy")
    return policy_record


def _validate_one_shot_stage_record(
    value: Any,
    workspace_policy: WorkspacePathPolicy,
    *,
    stage: str,
    owner: str,
) -> Mapping[str, Any]:
    record = _require_mapping(value, owner=owner)
    if (
        set(record)
        != {
            "policy",
            "stage",
            "invocation_ordinal",
            "invocations_allowed",
            "seed_retry_allowed",
            "candidate_ranking_allowed",
            "failure_action",
        }
        or record.get("stage") != stage
        or not _is_exact_int(record.get("invocation_ordinal"), 0)
        or not _is_exact_int(record.get("invocations_allowed"), 1)
        or record.get("seed_retry_allowed") is not False
        or record.get("candidate_ranking_allowed") is not False
        or record.get("failure_action")
        != "preserve_evidence_and_reject_instance"
    ):
        _fail(f"{owner} one-shot execution contract changed")
    _validate_one_shot_policy_record(
        record.get("policy"),
        workspace_policy,
        owner=f"{owner} policy",
    )
    return record


def _validate_upstream_flux_one_shot_evidence(
    value: Any,
    workspace_policy: WorkspacePathPolicy,
    *,
    expected_policy: Mapping[str, Any],
) -> str:
    evidence = _require_mapping(value, owner="upstream FLUX one-shot evidence")
    policy = _validate_one_shot_policy_record(
        evidence.get("policy"),
        workspace_policy,
        owner="upstream FLUX one-shot policy",
    )
    if dict(policy) != dict(expected_policy):
        _fail("FLUX and Pixal one-shot policies differ")
    flux_batch_sha256 = _require_sha256(
        evidence.get("flux_batch_sha256"),
        owner="upstream FLUX batch hash",
    )
    mode = evidence.get("mode")
    if mode == "native_policy_enforced_before_inference":
        if (
            set(evidence)
            != {
                "mode",
                "policy",
                "flux_batch_sha256",
                "profile_qualification_authorized",
            }
            or evidence.get("profile_qualification_authorized") is not True
        ):
            _fail("native upstream FLUX one-shot evidence changed")
    elif mode == "legacy_sealed_manifest_attestation":
        if (
            set(evidence)
            != {
                "mode",
                "policy",
                "flux_batch_sha256",
                "recorded_flux_invocations_per_candidate",
                "recorded_candidates_per_request",
                "cross_batch_seed_lottery_exclusion_proven",
                "profile_qualification_authorized",
            }
            or not _is_exact_int(
                evidence.get("recorded_flux_invocations_per_candidate"), 1
            )
            or not _is_exact_int(
                evidence.get("recorded_candidates_per_request"), 1
            )
            or evidence.get("cross_batch_seed_lottery_exclusion_proven") is not False
            or evidence.get("profile_qualification_authorized") is not False
        ):
            _fail("legacy upstream FLUX one-shot evidence changed")
    else:
        _fail("unsupported upstream FLUX one-shot evidence mode")
    return flux_batch_sha256


def _validate_selected_static_preflight_job(
    value: Mapping[str, Any],
    *,
    instance_id: str,
    controlled: Mapping[str, Any],
) -> Mapping[str, Any]:
    if (
        set(value)
        != {
            "schema",
            "source_bundle",
            "artifact_roots",
            "profile_artifact_authentication",
            "routes",
            "execution_summary",
            "automatic_checks",
            "preflight_sha256",
        }
        or value.get("schema") != EXECUTION_PREFLIGHT_SCHEMA
        or value.get("automatic_checks", {}).get("overall") != "passed"
    ):
        _fail("upstream FLUX execution preflight contract changed")
    routes = _require_mapping(
        value.get("routes"), owner="upstream FLUX execution routes"
    )
    route_names = {
        "flux2_pixal3d_animal_v1",
        STATIC_ROUTE,
        "stable_animal_template_v1",
        "rocketbox_material_v1",
    }
    if set(routes) != route_names or any(
        not isinstance(routes[name], list) for name in route_names
    ):
        _fail("upstream FLUX execution route coverage changed")
    matches = []
    for raw_job in routes[STATIC_ROUTE]:
        job = _require_mapping(
            raw_job, owner="upstream static FLUX execution job"
        )
        consumers = job.get("consumer_requests")
        if (
            isinstance(consumers, list)
            and len(consumers) == 1
            and isinstance(consumers[0], Mapping)
            and consumers[0].get("instance_id") == instance_id
        ):
            matches.append(job)
    if len(matches) != 1:
        _fail("upstream FLUX preflight lacks one selected static job")
    job = matches[0]
    if (
        set(job)
        != {
            "execution_job_id",
            "profile_schema_id",
            "profile_sha256",
            "lineage_group_id",
            "state_classification",
            "taxonomy",
            "fixed_attributes",
            "sampled_attributes",
            "consumer_requests",
            "generation_plan",
            "target_physical_profile",
            "rig_profile",
            "acoustic_profile",
            "execution_gate",
        }
        or job.get("state_classification") != "research_candidate"
        or job.get("rig_profile") is not None
        or job.get("execution_gate") != _STATIC_EXECUTION_GATE
        or not isinstance(job.get("lineage_group_id"), str)
        or not job["lineage_group_id"]
        or not isinstance(job.get("taxonomy"), Mapping)
        or not isinstance(job.get("fixed_attributes"), Mapping)
        or not isinstance(job.get("sampled_attributes"), Mapping)
        or not isinstance(job.get("target_physical_profile"), Mapping)
        or not isinstance(job.get("acoustic_profile"), Mapping)
    ):
        _fail("selected static FLUX preflight job contract changed")
    consumer = _require_mapping(
        job["consumer_requests"][0],
        owner="selected static FLUX preflight consumer",
    )
    if set(consumer) != {"instance_id", "request_sha256"}:
        _fail("selected static FLUX preflight consumer fields changed")
    request_sha256 = _require_sha256(
        consumer.get("request_sha256"),
        owner="selected static FLUX preflight request hash",
    )
    if (
        consumer.get("instance_id") != instance_id
        or job.get("execution_job_id") != f"static_{request_sha256[:16]}"
    ):
        _fail("selected static FLUX preflight execution identity changed")
    _require_sha256(
        job.get("profile_sha256"),
        owner="selected static FLUX preflight profile hash",
    )
    if (
        not isinstance(job.get("profile_schema_id"), str)
        or not job["profile_schema_id"]
    ):
        _fail("selected static FLUX preflight profile identity changed")

    plan = _require_mapping(
        job.get("generation_plan"),
        owner="selected static FLUX generation plan",
    )
    if (
        set(plan)
        != {
            "schema",
            "route",
            "prompt_template_id",
            "base_template",
            "prompt",
            "negative_prompt",
            "generation_seed",
            "flux_invocations",
            "model_revisions",
            "base_acquisition_policy",
        }
        or plan.get("schema") != STATIC_GENERATION_PLAN_SCHEMA
        or plan.get("route") != STATIC_ROUTE
        or not _is_exact_int(plan.get("flux_invocations"), 1)
        or plan.get("base_acquisition_policy")
        != _STATIC_BASE_ACQUISITION_POLICY
        or not isinstance(plan.get("prompt_template_id"), str)
        or not plan["prompt_template_id"]
        or not isinstance(plan.get("prompt"), str)
        or not plan["prompt"].strip()
        or not isinstance(plan.get("negative_prompt"), str)
        or not plan["negative_prompt"].strip()
        or not isinstance(plan.get("model_revisions"), Mapping)
    ):
        _fail("selected static FLUX generation plan changed")
    base_template = _require_mapping(
        plan.get("base_template"),
        owner="selected static FLUX base template",
    )
    if (
        set(base_template)
        != {
            "template_id",
            "kind",
            "artifact",
            "provenance_status",
            "usage_scope",
        }
        or base_template.get("kind") != "text_prompt_only"
        or base_template.get("artifact") is not None
    ):
        _fail("selected static FLUX base-template contract changed")
    seed = plan.get("generation_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        _fail("selected static FLUX generation seed is invalid")

    expected_controlled = {
        "execution_job_id": job["execution_job_id"],
        "instance_id": instance_id,
        "request_sha256": request_sha256,
        "generation_seed": seed,
        "profile_schema_id": job["profile_schema_id"],
        "profile_sha256": job["profile_sha256"],
        "asset_class": "static_object",
        "route": STATIC_ROUTE,
        "sampled_attributes": job["sampled_attributes"],
        "target_physical_profile": job["target_physical_profile"],
        "rig_profile": None,
    }
    if dict(controlled) != expected_controlled:
        _fail("Pixal controlled request differs from its static preflight job")
    return job


def _validate_selected_flux_candidate(
    raw_batch_record: Any,
    policy: WorkspacePathPolicy,
    *,
    instance_id: str,
    controlled: Mapping[str, Any],
    expected_flux_batch_sha256: str,
    expected_one_shot_policy: Mapping[str, Any],
) -> tuple[
    AuthenticatedArtifact,
    AuthenticatedArtifact,
    frozenset[str],
]:
    batch, batch_artifact = _json_path_hash_record(
        raw_batch_record,
        policy,
        owner="upstream FLUX batch",
        content_hash_field="batch_sha256",
    )
    batch_sha256 = _require_content_hash(
        batch, field="batch_sha256", owner="upstream FLUX batch"
    )
    if (
        raw_batch_record.get("batch_sha256") != batch_sha256
        or batch_sha256 != expected_flux_batch_sha256
        or batch.get("schema") != FLUX_BATCH_SCHEMA
        or batch.get("status") != "pending_2d_review"
        or batch.get("state_classification") != "research_candidate"
        or batch.get("formal_dataset_registration_authorized") is not False
        or batch.get("selection", {}).get("route") != STATIC_ROUTE
        or batch.get("automatic_checks", {}).get("overall")
        != "pending_2d_review"
    ):
        _fail("upstream FLUX batch contract/hash changed")
    flux_one_shot = _validate_one_shot_stage_record(
        batch.get("one_shot_execution"),
        policy,
        stage="flux2",
        owner="upstream FLUX batch",
    )
    if flux_one_shot.get("policy") != expected_one_shot_policy:
        _fail("upstream FLUX batch one-shot policy changed")

    preflight, _ = _json_path_hash_record(
        batch.get("execution_preflight"),
        policy,
        owner="upstream FLUX execution preflight",
        content_hash_field="preflight_sha256",
    )
    preflight_sha256 = _require_content_hash(
        preflight,
        field="preflight_sha256",
        owner="upstream FLUX execution preflight",
    )
    if batch["execution_preflight"].get("preflight_sha256") != preflight_sha256:
        _fail("upstream FLUX execution preflight content changed")
    preflight_job = _validate_selected_static_preflight_job(
        preflight,
        instance_id=instance_id,
        controlled=controlled,
    )

    candidates = batch.get("candidates")
    if (
        not isinstance(candidates, list)
        or batch.get("candidate_count") != len(candidates)
        or not candidates
    ):
        _fail("upstream FLUX candidate index is invalid")
    candidate_ids = [
        item.get("instance_id") if isinstance(item, Mapping) else None
        for item in candidates
    ]
    if (
        any(not isinstance(item, str) or not item for item in candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        _fail("upstream FLUX candidate identities are invalid")
    matches = [
        item
        for item in candidates
        if isinstance(item, Mapping) and item.get("instance_id") == instance_id
    ]
    if len(matches) != 1:
        _fail("upstream FLUX batch lacks exactly one selected candidate")
    selected = _require_mapping(matches[0], owner="selected upstream FLUX candidate")
    if (
        set(selected)
        != {
            "execution_job_id",
            "instance_id",
            "profile_schema_id",
            "sampled_attributes",
            "status",
            "candidate",
            "candidate_manifest",
        }
        or selected.get("execution_job_id") != controlled["execution_job_id"]
        or selected.get("profile_schema_id") != controlled["profile_schema_id"]
        or selected.get("sampled_attributes") != controlled["sampled_attributes"]
        or selected.get("status") != "pending_2d_review"
    ):
        _fail("selected upstream FLUX candidate index changed")

    flux_root = batch_artifact.path.parent
    candidate = _file_record(
        selected.get("candidate"),
        policy,
        owner="selected upstream FLUX candidate image",
        base=flux_root,
        require_within=flux_root,
    )
    _validate_png_artifact(
        candidate,
        owner="selected upstream FLUX candidate image",
        expected_size=(1024, 1024),
        expected_mode="RGB",
    )
    manifest, manifest_artifact = _json_file_record(
        selected.get("candidate_manifest"),
        policy,
        owner="selected upstream FLUX candidate manifest",
        base=flux_root,
        require_within=flux_root,
    )
    manifest_fields = {
        "schema",
        "status",
        "state_classification",
        "formal_dataset_registration_authorized",
        "execution_preflight_sha256",
        "execution_job_id",
        "instance_id",
        "request_sha256",
        "profile_schema_id",
        "profile_sha256",
        "lineage_group_id",
        "taxonomy",
        "fixed_attributes",
        "sampled_attributes",
        "input",
        "output",
        "generation",
        "one_shot_execution",
        "downstream_gate",
        "timings",
        "automatic_checks",
        "manifest_sha256",
    }
    if (
        set(manifest) != manifest_fields
        or manifest.get("schema") != FLUX_CANDIDATE_SCHEMA
        or manifest.get("status") != "pending_2d_review"
        or manifest.get("state_classification") != "research_candidate"
        or manifest.get("formal_dataset_registration_authorized") is not False
        or manifest.get("execution_preflight_sha256") != preflight_sha256
        or manifest.get("execution_job_id") != controlled["execution_job_id"]
        or manifest.get("instance_id") != instance_id
        or manifest.get("request_sha256") != controlled["request_sha256"]
        or manifest.get("profile_schema_id") != controlled["profile_schema_id"]
        or manifest.get("profile_sha256") != controlled["profile_sha256"]
        or manifest.get("lineage_group_id")
        != preflight_job["lineage_group_id"]
        or manifest.get("taxonomy") != preflight_job["taxonomy"]
        or manifest.get("fixed_attributes")
        != preflight_job["fixed_attributes"]
        or manifest.get("sampled_attributes") != controlled["sampled_attributes"]
        or manifest.get("input") is not None
        or manifest.get("downstream_gate")
        != {
            "status": "blocked_pending_2d_review",
            "required_review": "approved_for_exact_candidate_sha256",
            "next_stage": "foreground_segmentation_then_pixal3d",
        }
        or manifest.get("automatic_checks", {}).get("overall")
        != "pending_2d_review"
    ):
        _fail("selected upstream FLUX candidate manifest changed")
    _require_content_hash(
        manifest,
        field="manifest_sha256",
        owner="selected upstream FLUX candidate manifest",
    )
    _file_record(
        manifest.get("output"),
        policy,
        owner="selected upstream FLUX candidate output",
        expected=candidate,
        base=flux_root,
        require_within=flux_root,
    )
    candidate_one_shot = _validate_one_shot_stage_record(
        manifest.get("one_shot_execution"),
        policy,
        stage="flux2",
        owner="selected upstream FLUX candidate",
    )
    if candidate_one_shot.get("policy") != expected_one_shot_policy:
        _fail("selected upstream FLUX candidate one-shot policy changed")
    generation = _require_mapping(
        manifest.get("generation"), owner="selected upstream FLUX generation"
    )
    if (
        generation.get("seed") != controlled["generation_seed"]
        or not _is_exact_int(generation.get("flux_invocations"), 1)
    ):
        _fail("selected upstream FLUX generation violates one-shot request")
    return candidate, manifest_artifact, frozenset(candidate_ids)


def _validate_static_2d_review_lineage(
    raw_review_batch_record: Any,
    policy: WorkspacePathPolicy,
    *,
    instance_id: str,
    controlled: Mapping[str, Any],
    source: AuthenticatedArtifact,
    expected_flux_batch_sha256: str,
    expected_one_shot_policy: Mapping[str, Any],
) -> None:
    review_batch, review_batch_artifact = _json_path_hash_record(
        raw_review_batch_record,
        policy,
        owner="upstream static 2D review batch",
        content_hash_field="review_batch_sha256",
    )
    review_batch_sha256 = _require_content_hash(
        review_batch,
        field="review_batch_sha256",
        owner="upstream static 2D review batch",
    )
    review_batch_fields = {
        "schema",
        "status",
        "state_classification",
        "formal_dataset_registration_authorized",
        "review_domain",
        "flux2_batch",
        "decisions_input",
        "candidate_count",
        "approved_count",
        "rejected_count",
        "reviews",
        "qa_pair_eligibility",
        "automatic_checks",
        "review_batch_sha256",
    }
    reviews = review_batch.get("reviews")
    if (
        set(review_batch) != review_batch_fields
        or raw_review_batch_record.get("review_batch_sha256")
        != review_batch_sha256
        or review_batch.get("schema") != STATIC_2D_REVIEW_BATCH_SCHEMA
        or review_batch.get("status") not in {"passed", "completed_with_rejections"}
        or review_batch.get("state_classification") != "research_candidate"
        or review_batch.get("formal_dataset_registration_authorized") is not False
        or review_batch.get("review_domain") != "static_object"
        or review_batch.get("automatic_checks", {}).get("overall") != "passed"
        or not isinstance(reviews, list)
        or not isinstance(review_batch.get("qa_pair_eligibility"), list)
    ):
        _fail("upstream static 2D review batch contract changed")
    candidate_count = _require_nonnegative_int(
        review_batch.get("candidate_count"),
        owner="upstream static 2D review candidate count",
    )
    approved_count = _require_nonnegative_int(
        review_batch.get("approved_count"),
        owner="upstream static 2D review approved count",
    )
    rejected_count = _require_nonnegative_int(
        review_batch.get("rejected_count"),
        owner="upstream static 2D review rejected count",
    )
    if (
        candidate_count != len(reviews)
        or approved_count + rejected_count != candidate_count
        or approved_count <= 0
    ):
        _fail("upstream static 2D review counts changed")
    static_check_fields = {
        "category_identity",
        "construction",
        "stable_product_pose",
        "background",
    }
    hard_gate_fields = {
        "single_subject",
        "photorealistic_pbr_style",
        "category_distinctive_features",
        "emitter_feature_visible",
        "physically_connected_construction",
        "complete_object",
        "stable_rest_or_mount",
        "target_attribute_only",
    }
    decisions_payload, _ = _json_file_record(
        review_batch.get("decisions_input"),
        policy,
        owner="upstream static 2D review decisions",
    )
    raw_decisions = decisions_payload.get("decisions")
    if (
        set(decisions_payload)
        != {"schema", "flux2_batch_sha256", "reviewer", "decisions"}
        or decisions_payload.get("schema")
        != "avengine_controlled_static_object_2d_review_decisions_v1"
        or decisions_payload.get("flux2_batch_sha256")
        != expected_flux_batch_sha256
        or not isinstance(decisions_payload.get("reviewer"), str)
        or not decisions_payload["reviewer"].strip()
        or not isinstance(raw_decisions, list)
    ):
        _fail("upstream static 2D review decision authority changed")
    decisions: dict[str, Mapping[str, Any]] = {}
    decision_fields = {
        "instance_id",
        "candidate_sha256",
        "decision",
        "sampled_attribute_checks",
        "notes",
        "hard_gates",
    } | static_check_fields
    for raw_decision in raw_decisions:
        decision = _require_mapping(
            raw_decision, owner="upstream static 2D review decision"
        )
        if set(decision) != decision_fields:
            _fail("upstream static 2D review decision fields changed")
        decision_id = _require_stable_id(
            decision.get("instance_id"),
            owner="upstream static 2D review decision instance_id",
        )
        if decision_id in decisions:
            _fail("upstream static 2D review repeats a decision")
        _require_sha256(
            decision.get("candidate_sha256"),
            owner="upstream static 2D review candidate hash",
        )
        if (
            decision.get("decision") not in {"approved_for_pixal3d", "rejected"}
            or any(
                decision.get(field) not in {"passed", "rejected"}
                for field in static_check_fields
            )
            or not isinstance(decision.get("notes"), str)
        ):
            _fail("upstream static 2D review decision value changed")
        attribute_checks = _require_mapping(
            decision.get("sampled_attribute_checks"),
            owner="upstream static sampled-attribute decision checks",
        )
        if any(
            status
            not in {
                "passed",
                "deferred_to_3d_physical_scale",
                "rejected",
            }
            or (
                status == "deferred_to_3d_physical_scale"
                and attribute != "size"
            )
            for attribute, status in attribute_checks.items()
        ):
            _fail("upstream static sampled-attribute decision changed")
        hard_gates = _require_mapping(
            decision.get("hard_gates"),
            owner="upstream static 2D hard-gate decision",
        )
        if set(hard_gates) != hard_gate_fields or any(
            status not in {"passed", "rejected"}
            for status in hard_gates.values()
        ):
            _fail("upstream static 2D hard-gate decision changed")
        rejected = (
            any(decision[field] == "rejected" for field in static_check_fields)
            or "rejected" in attribute_checks.values()
            or "rejected" in hard_gates.values()
        )
        if (decision["decision"] == "rejected") != rejected:
            _fail("upstream static 2D decision disagrees with its checks")
        decisions[decision_id] = decision

    candidate, candidate_manifest, flux_candidate_ids = (
        _validate_selected_flux_candidate(
        review_batch.get("flux2_batch"),
        policy,
        instance_id=instance_id,
        controlled=controlled,
        expected_flux_batch_sha256=expected_flux_batch_sha256,
        expected_one_shot_policy=expected_one_shot_policy,
    )
    )
    if candidate != source:
        _fail("Pixal source differs from approved upstream FLUX candidate")

    review_ids = [
        item.get("instance_id") if isinstance(item, Mapping) else None
        for item in reviews
    ]
    if (
        any(not isinstance(item, str) or not item for item in review_ids)
        or len(review_ids) != len(set(review_ids))
        or set(review_ids) != set(decisions)
        or set(review_ids) != set(flux_candidate_ids)
    ):
        _fail("upstream static 2D review/decision candidate coverage changed")
    actual_approved = sum(
        decision["decision"] == "approved_for_pixal3d"
        for decision in decisions.values()
    )
    if (
        actual_approved != approved_count
        or len(decisions) - actual_approved != rejected_count
        or review_batch.get("status")
        != ("passed" if rejected_count == 0 else "completed_with_rejections")
    ):
        _fail("upstream static 2D decision counts/status changed")
    selected_decision = decisions[instance_id]
    if (
        selected_decision.get("decision") != "approved_for_pixal3d"
        or selected_decision.get("candidate_sha256") != candidate.sha256
        or set(selected_decision["sampled_attribute_checks"])
        != set(controlled["sampled_attributes"])
    ):
        _fail("selected upstream static 2D decision authority changed")
    matches = [
        item
        for item in reviews
        if isinstance(item, Mapping) and item.get("instance_id") == instance_id
    ]
    if len(matches) != 1:
        _fail("upstream static 2D review lacks exactly one selected instance")
    selected = _require_mapping(
        matches[0], owner="selected upstream static 2D review index"
    )
    if (
        set(selected)
        != {
            "instance_id",
            "profile_schema_id",
            "decision",
            "candidate_sha256",
            "review",
        }
        or selected.get("profile_schema_id") != controlled["profile_schema_id"]
        or selected.get("decision") != selected_decision["decision"]
        or selected.get("candidate_sha256")
        != selected_decision["candidate_sha256"]
    ):
        _fail("selected upstream static 2D review index changed")
    review_root = review_batch_artifact.path.parent
    review, _ = _json_file_record(
        selected.get("review"),
        policy,
        owner="selected upstream static 2D review",
        base=review_root,
        require_within=review_root,
    )
    review_fields = {
        "schema",
        "instance_id",
        "request_sha256",
        "profile_schema_id",
        "sampled_attributes",
        "candidate",
        "candidate_manifest",
        "reviewer",
        "decision",
        "checks",
        "notes",
        "downstream_gate",
        "review_sha256",
    }
    if (
        set(review) != review_fields
        or review.get("schema") != STATIC_2D_REVIEW_SCHEMA
        or review.get("instance_id") != instance_id
        or review.get("request_sha256") != controlled["request_sha256"]
        or review.get("profile_schema_id") != controlled["profile_schema_id"]
        or review.get("sampled_attributes") != controlled["sampled_attributes"]
        or review.get("reviewer") != decisions_payload["reviewer"]
        or review.get("decision") != selected_decision["decision"]
        or review.get("downstream_gate")
        != "approved_for_segmentation_and_pixal3d"
        or review.get("notes") != selected_decision["notes"]
    ):
        _fail("selected upstream static 2D review changed")
    _require_content_hash(
        review, field="review_sha256", owner="selected upstream static 2D review"
    )
    _file_record(
        review.get("candidate"),
        policy,
        owner="selected upstream reviewed candidate",
        expected=candidate,
    )
    _file_record(
        review.get("candidate_manifest"),
        policy,
        owner="selected upstream reviewed candidate manifest",
        expected=candidate_manifest,
    )
    checks = _require_mapping(
        review.get("checks"), owner="selected upstream static 2D review checks"
    )
    expected_checks = {
        **{
            field: selected_decision[field]
            for field in static_check_fields
        },
        "sampled_attributes": selected_decision[
            "sampled_attribute_checks"
        ],
        "hard_gates": selected_decision["hard_gates"],
    }
    if (
        checks != expected_checks
        or any(checks.get(field) != "passed" for field in static_check_fields)
    ):
        _fail("published static 2D review differs from decision authority")
    attribute_checks = _require_mapping(
        checks.get("sampled_attributes"),
        owner="selected upstream sampled-attribute checks",
    )
    if set(attribute_checks) != set(controlled["sampled_attributes"]) or any(
        status not in {"passed", "deferred_to_3d_physical_scale"}
        or (status == "deferred_to_3d_physical_scale" and attribute != "size")
        for attribute, status in attribute_checks.items()
    ):
        _fail("selected upstream sampled-attribute checks changed")
    hard_gates = _require_mapping(
        checks.get("hard_gates"), owner="selected upstream static hard gates"
    )
    if set(hard_gates) != hard_gate_fields or any(
        status != "passed" for status in hard_gates.values()
    ):
        _fail("selected upstream static hard gates failed")


def _path_is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _configured_executable_record(
    value: Any,
    policy: WorkspacePathPolicy,
    *,
    owner: str,
) -> tuple[Path, AuthenticatedArtifact]:
    record = _require_mapping(value, owner=f"{owner} record")
    if set(record) != {"path", "sha256", "size_bytes"}:
        _fail(f"{owner} file record fields are invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        _fail(f"{owner} path must be a non-empty string")
    configured_path = Path(raw_path).expanduser()
    canonical_path = Path(os.path.abspath(configured_path))
    if (
        not configured_path.is_absolute()
        or configured_path != canonical_path
        or not configured_path.is_file()
        or not os.access(configured_path, os.X_OK)
    ):
        _fail(f"{owner} configured executable path is invalid")
    try:
        configured_parent = configured_path.parent.resolve(strict=True)
    except OSError as error:
        raise StaticObjectRegistrationError(
            f"{owner} configured executable parent is invalid: {error}"
        ) from error
    if not any(
        _path_is_within(configured_parent, root) for root in policy.roots
    ):
        _fail(f"{owner} configured executable escapes workspace roots")
    digest = _require_sha256(record.get("sha256"), owner=f"{owner} sha256")
    try:
        resolved = policy.resolve_input(
            configured_path,
            owner=owner,
            kind="file",
            expected_sha256=digest,
        )
    except (OSError, PathPolicyError) as error:
        raise StaticObjectRegistrationError(str(error)) from error
    size = configured_path.stat().st_size
    if (
        isinstance(record.get("size_bytes"), bool)
        or record.get("size_bytes") != size
        or size <= 0
    ):
        _fail(f"{owner} size changed")
    return (
        configured_path,
        AuthenticatedArtifact(path=resolved, sha256=digest, size_bytes=size),
    )


def _rebind_isnet_command_root(
    command: Sequence[str],
    *,
    published_root: Path,
    staging_root: Path,
) -> list[str]:
    published = str(published_root)
    staging = str(staging_root)
    prefix = published + os.sep
    return [
        staging + argument[len(published) :]
        if argument == published or argument.startswith(prefix)
        else argument
        for argument in command
    ]


def _isnet_worker_pins(
    artifact: AuthenticatedArtifact,
) -> tuple[Path, str, str, str]:
    try:
        module = ast.parse(
            artifact.path.read_text(encoding="utf-8"),
            filename=str(artifact.path),
        )
    except (OSError, UnicodeError, SyntaxError) as error:
        raise StaticObjectRegistrationError(
            f"base Pixal frozen ISNet worker cannot be inspected: {error}"
        ) from error
    values: dict[str, list[ast.expr]] = {
        "MODEL_PATH": [],
        "MODEL_SHA256": [],
        "JOBS_SCHEMA": [],
        "STATUS_SCHEMA": [],
    }
    for statement in module.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id in values
        ):
            values[statement.targets[0].id].append(statement.value)
    if any(len(items) != 1 for items in values.values()):
        _fail("base Pixal frozen ISNet worker pins are incomplete")
    model_expression = values["MODEL_PATH"][0]
    if (
        not isinstance(model_expression, ast.Call)
        or not isinstance(model_expression.func, ast.Name)
        or model_expression.func.id != "Path"
        or len(model_expression.args) != 1
        or model_expression.keywords
        or not isinstance(model_expression.args[0], ast.Constant)
        or not isinstance(model_expression.args[0].value, str)
    ):
        _fail("base Pixal frozen ISNet worker model path pin changed")
    strings: dict[str, str] = {}
    for name in ("MODEL_SHA256", "JOBS_SCHEMA", "STATUS_SCHEMA"):
        expression = values[name][0]
        if (
            not isinstance(expression, ast.Constant)
            or not isinstance(expression.value, str)
        ):
            _fail(f"base Pixal frozen ISNet worker {name} pin changed")
        strings[name] = expression.value
    return (
        Path(model_expression.args[0].value),
        strings["MODEL_SHA256"],
        strings["JOBS_SCHEMA"],
        strings["STATUS_SCHEMA"],
    )


def _validate_isnet_execution_receipt(
    value: Any,
    policy: WorkspacePathPolicy,
    *,
    artifact: AuthenticatedArtifact,
    pixal_jobs: Sequence[Any],
    segmentations: Sequence[Any],
) -> None:
    receipt = _require_mapping(value, owner="base Pixal ISNet receipt")
    if (
        set(receipt)
        != {
            "schema",
            "model",
            "python",
            "worker",
            "jobs",
            "working_directory",
            "command",
            "command_sha256",
            "executed_command",
            "executed_command_sha256",
            "path_rebinding",
            "status",
            "log",
        }
        or receipt.get("schema") != ISNET_EXECUTION_RECEIPT_SCHEMA
    ):
        _fail("base Pixal ISNet execution receipt changed")

    published_root = artifact.path.parent
    rebinding = _require_mapping(
        receipt.get("path_rebinding"),
        owner="base Pixal ISNet path rebinding",
    )
    if set(rebinding) != {"staging_root", "published_root"}:
        _fail("base Pixal ISNet path rebinding fields changed")
    published_value = rebinding.get("published_root")
    staging_value = rebinding.get("staging_root")
    if (
        not isinstance(published_value, str)
        or published_value != str(published_root)
        or not isinstance(staging_value, str)
    ):
        _fail("base Pixal ISNet published/staging roots changed")
    staging_root = Path(staging_value)
    expected_staging_prefix = f".{published_root.name}."
    if (
        not staging_root.is_absolute()
        or staging_root != Path(os.path.abspath(staging_root))
        or staging_root.parent != published_root.parent
        or not staging_root.name.startswith(expected_staging_prefix)
        or not staging_root.name.endswith(".staging")
        or staging_root.exists()
        or staging_root.is_symlink()
    ):
        _fail("base Pixal ISNet staging-root provenance changed")

    working_value = receipt.get("working_directory")
    if not isinstance(working_value, str) or not Path(working_value).is_absolute():
        _fail("base Pixal ISNet working directory changed")
    try:
        working_directory = policy.resolve_input(
            working_value,
            owner="base Pixal ISNet working directory",
            kind="directory",
        )
    except (OSError, PathPolicyError) as error:
        raise StaticObjectRegistrationError(str(error)) from error
    if Path(working_value).is_symlink() or str(working_directory) != working_value:
        _fail("base Pixal ISNet working directory is not canonical")

    model = _file_record(
        receipt.get("model"),
        policy,
        owner="base Pixal pinned ISNet model",
    )
    if model.path.name != ISNET_MODEL_FILENAME:
        _fail("base Pixal pinned ISNet model identity changed")

    python_record = _require_mapping(
        receipt.get("python"), owner="base Pixal ISNet Python"
    )
    if set(python_record) != {"configured", "resolved"}:
        _fail("base Pixal ISNet Python fields changed")
    configured_path, configured = _configured_executable_record(
        python_record.get("configured"),
        policy,
        owner="base Pixal configured ISNet Python",
    )
    resolved = _file_record(
        python_record.get("resolved"),
        policy,
        owner="base Pixal resolved ISNet Python",
    )
    if (
        not os.access(resolved.path, os.X_OK)
        or configured_path.resolve(strict=True) != resolved.path
        or configured.sha256 != resolved.sha256
        or configured.size_bytes != resolved.size_bytes
    ):
        _fail("base Pixal configured/resolved ISNet Python identity changed")

    worker_record = _require_mapping(
        receipt.get("worker"), owner="base Pixal ISNet worker"
    )
    if set(worker_record) != {"source", "executed"}:
        _fail("base Pixal ISNet worker fields changed")
    worker_source = _file_record(
        worker_record.get("source"),
        policy,
        owner="base Pixal ISNet worker source",
    )
    worker_executed = _file_record(
        worker_record.get("executed"),
        policy,
        owner="base Pixal frozen ISNet worker",
        require_within=published_root,
    )
    if (
        worker_source.path
        != working_directory / "tools" / ISNET_WORKER_FILENAME
        or worker_executed.path
        != published_root / ".runtime_commands" / ISNET_WORKER_FILENAME
        or worker_executed.path.parent.is_symlink()
        or worker_executed.path.parent.stat().st_mode & 0o777 != 0o555
        or worker_executed.path.stat().st_mode & 0o777 != 0o444
    ):
        _fail("base Pixal ISNet worker paths changed")
    _require_same_file_content(
        worker_executed,
        worker_source,
        owner="base Pixal frozen ISNet worker",
    )
    (
        worker_model_path,
        worker_model_sha256,
        worker_jobs_schema,
        worker_status_schema,
    ) = _isnet_worker_pins(worker_executed)
    if (
        not worker_model_path.is_absolute()
        or worker_model_path != model.path
        or worker_model_sha256 != model.sha256
        or worker_jobs_schema != ISNET_JOBS_SCHEMA
        or worker_status_schema != ISNET_STATUS_SCHEMA
    ):
        _fail("base Pixal frozen ISNet worker pins differ from its receipt")

    jobs_payload, jobs_artifact = _json_file_record(
        receipt.get("jobs"),
        policy,
        owner="base Pixal ISNet jobs",
        require_within=published_root,
    )
    if jobs_artifact.path != published_root / "isnet_jobs.json":
        _fail("base Pixal ISNet jobs path changed")
    status_payload, status_artifact = _json_file_record(
        receipt.get("status"),
        policy,
        owner="base Pixal ISNet status",
        base=published_root,
        require_within=published_root,
    )
    if status_artifact.path != published_root / "isnet_status.json":
        _fail("base Pixal ISNet status path changed")
    log_artifact = _file_record(
        receipt.get("log"),
        policy,
        owner="base Pixal ISNet log",
        base=published_root,
        require_within=published_root,
        allow_empty=True,
    )
    if log_artifact.path != published_root / "isnet.log":
        _fail("base Pixal ISNet log path changed")

    command = receipt.get("command")
    executed_command = receipt.get("executed_command")
    expected_command = [
        str(resolved.path),
        str(worker_executed.path),
        "--jobs",
        str(jobs_artifact.path),
        "--status",
        str(status_artifact.path),
    ]
    if (
        not isinstance(command, list)
        or any(not isinstance(item, str) for item in command)
        or command != expected_command
        or receipt.get("command_sha256") != canonical_json_sha256(command)
        or not isinstance(executed_command, list)
        or any(not isinstance(item, str) for item in executed_command)
        or executed_command
        != _rebind_isnet_command_root(
            command,
            published_root=published_root,
            staging_root=staging_root,
        )
        or receipt.get("executed_command_sha256")
        != canonical_json_sha256(executed_command)
    ):
        _fail("base Pixal ISNet command execution receipt changed")

    job_items = jobs_payload.get("jobs")
    if (
        set(jobs_payload) != {"schema", "jobs"}
        or jobs_payload.get("schema") != ISNET_JOBS_SCHEMA
        or not isinstance(job_items, list)
        or len(job_items) != len(pixal_jobs)
    ):
        _fail("base Pixal ISNet jobs payload changed")
    outer_jobs_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_job in pixal_jobs:
        outer_job = _require_mapping(raw_job, owner="base Pixal input job")
        controlled = _require_mapping(
            outer_job.get("controlled_request"),
            owner="base Pixal controlled request",
        )
        identifier = _require_stable_id(
            controlled.get("instance_id"),
            owner="base Pixal controlled request instance_id",
        )
        outer_jobs_by_id[identifier] = outer_job
    segmentations_by_id = {
        item.get("instance_id"): item
        for item in segmentations
        if isinstance(item, Mapping)
    }
    job_ids = [
        item.get("instance_id") if isinstance(item, Mapping) else None
        for item in job_items
    ]
    if (
        job_ids != sorted(outer_jobs_by_id)
        or set(job_ids) != set(outer_jobs_by_id)
        or set(segmentations_by_id) != set(outer_jobs_by_id)
    ):
        _fail("base Pixal ISNet jobs do not cover every Pixal input")

    output_artifacts: dict[str, tuple[AuthenticatedArtifact, AuthenticatedArtifact]] = {}
    jobs_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_job in job_items:
        isnet_job = _require_mapping(raw_job, owner="base Pixal ISNet job")
        if set(isnet_job) != {
            "instance_id",
            "candidate_path",
            "candidate_sha256",
            "alpha_path",
            "rgba_path",
        }:
            _fail("base Pixal ISNet job fields changed")
        identifier = str(isnet_job["instance_id"])
        outer_job = outer_jobs_by_id[identifier]
        reference = _require_mapping(
            outer_job.get("reference"),
            owner="base Pixal ISNet-bound reference",
        )
        source = _file_record(
            reference.get("source"),
            policy,
            owner=f"base Pixal ISNet source {identifier}",
        )
        expected_staging_alpha = (
            staging_root / "segmentation" / identifier / "alpha_isnet.png"
        )
        expected_staging_rgba = (
            staging_root
            / "segmentation"
            / identifier
            / "input_rgba_isnet.png"
        )
        if (
            isnet_job.get("candidate_path") != str(source.path)
            or isnet_job.get("candidate_sha256") != source.sha256
            or isnet_job.get("alpha_path") != str(expected_staging_alpha)
            or isnet_job.get("rgba_path") != str(expected_staging_rgba)
        ):
            _fail("base Pixal ISNet job differs from its sealed Pixal input")
        segmentation = _require_mapping(
            segmentations_by_id[identifier],
            owner=f"base Pixal segmentation {identifier}",
        )
        alpha = _file_record(
            segmentation.get("alpha"),
            policy,
            owner=f"base Pixal ISNet alpha {identifier}",
            base=published_root,
            require_within=published_root,
        )
        rgba = _file_record(
            segmentation.get("rgba"),
            policy,
            owner=f"base Pixal ISNet RGBA {identifier}",
            base=published_root,
            require_within=published_root,
        )
        if (
            alpha.path
            != published_root
            / "segmentation"
            / identifier
            / "alpha_isnet.png"
            or rgba.path
            != published_root
            / "segmentation"
            / identifier
            / "input_rgba_isnet.png"
        ):
            _fail("base Pixal ISNet output paths changed")
        output_artifacts[identifier] = (alpha, rgba)
        jobs_by_id[identifier] = isnet_job

    status_jobs = status_payload.get("jobs")
    model_load_seconds = status_payload.get("model_load_seconds")
    if (
        set(status_payload)
        != {
            "schema",
            "status",
            "model",
            "model_load_seconds",
            "jobs",
            "passed_count",
            "failed_count",
        }
        or status_payload.get("schema") != ISNET_STATUS_SCHEMA
        or status_payload.get("status") != "passed"
        or status_payload.get("model")
        != {
            "path": str(model.path),
            "sha256": model.sha256,
            "name": "isnet-general-use",
        }
        or not isinstance(status_jobs, list)
        or len(status_jobs) != len(job_items)
        or not _is_exact_int(
            status_payload.get("passed_count"), len(job_items)
        )
        or not _is_exact_int(status_payload.get("failed_count"), 0)
        or _finite_number(
            model_load_seconds,
            owner="base Pixal ISNet model load seconds",
        )
        < 0.0
    ):
        _fail("base Pixal ISNet status summary changed")
    status_ids = [
        item.get("instance_id") if isinstance(item, Mapping) else None
        for item in status_jobs
    ]
    if status_ids != job_ids:
        _fail("base Pixal ISNet status job order changed")
    for raw_status in status_jobs:
        status_job = _require_mapping(
            raw_status, owner="base Pixal ISNet status job"
        )
        if set(status_job) != {
            "instance_id",
            "status",
            "alpha_path",
            "alpha_sha256",
            "rgba_path",
            "rgba_sha256",
            "foreground_fraction_at_128",
            "foreground_bbox_xyxy",
            "alpha_extrema",
            "wall_seconds",
        }:
            _fail("base Pixal ISNet status job fields changed")
        identifier = str(status_job["instance_id"])
        segmentation = segmentations_by_id[identifier]
        alpha, rgba = output_artifacts[identifier]
        alpha_extrema = status_job.get("alpha_extrema")
        if (
            status_job.get("status") != "passed"
            or status_job.get("alpha_path") != jobs_by_id[identifier]["alpha_path"]
            or status_job.get("alpha_sha256") != alpha.sha256
            or status_job.get("rgba_path") != jobs_by_id[identifier]["rgba_path"]
            or status_job.get("rgba_sha256") != rgba.sha256
            or status_job.get("foreground_fraction_at_128")
            != segmentation.get("foreground_fraction_at_128")
            or status_job.get("foreground_bbox_xyxy")
            != segmentation.get("foreground_bbox_xyxy")
            or not isinstance(alpha_extrema, list)
            or len(alpha_extrema) != 2
            or not _is_exact_int(alpha_extrema[0], 0)
            or not _is_exact_int(alpha_extrema[1], 255)
            or _finite_number(
                status_job.get("wall_seconds"),
                owner=f"base Pixal ISNet wall seconds {identifier}",
            )
            < 0.0
        ):
            _fail("base Pixal ISNet status job differs from sealed outputs")


def _validate_base_pixal_input_manifest(
    value: Mapping[str, Any],
    artifact: AuthenticatedArtifact,
    policy: WorkspacePathPolicy,
    *,
    selected_instance_id: str | None,
) -> Mapping[str, Any] | None:
    expected_fields = {
        "schema",
        "status",
        "state_classification",
        "formal_dataset_registration_authorized",
        "asset_class",
        "route",
        "one_shot_execution",
        "upstream_flux_one_shot_evidence",
        "review_batch",
        "isnet",
        "pixal_output_root",
        "job_count",
        "jobs",
        "segmentations",
        "automatic_checks",
        "manifest_sha256",
    }
    jobs = value.get("jobs")
    if (
        set(value) != expected_fields
        or value.get("schema") != STATIC_PIXAL_INPUT_SCHEMA
        or value.get("status") != "ready_for_pixal3d"
        or value.get("state_classification") != "research_candidate"
        or value.get("formal_dataset_registration_authorized") is not False
        or value.get("asset_class") != "static_object"
        or value.get("route") != STATIC_ROUTE
        or value.get("automatic_checks", {}).get("overall") != "passed"
        or value.get("automatic_checks", {}).get(
            "static_jobs_have_no_rig_or_animation_binding"
        )
        is not True
        or not isinstance(jobs, list)
        or not jobs
        or value.get("job_count") != len(jobs)
    ):
        _fail("base Pixal input manifest contract is invalid")
    _require_content_hash(
        value,
        field="manifest_sha256",
        owner="base Pixal input manifest",
    )
    pixal_one_shot = _validate_one_shot_stage_record(
        value.get("one_shot_execution"),
        policy,
        stage="pixal3d",
        owner="base Pixal input manifest",
    )
    flux_batch_sha256 = _validate_upstream_flux_one_shot_evidence(
        value.get("upstream_flux_one_shot_evidence"),
        policy,
        expected_policy=_require_mapping(
            pixal_one_shot.get("policy"), owner="Pixal one-shot policy"
        ),
    )
    identifiers: list[str] = []
    for raw in jobs:
        job = _require_mapping(raw, owner="base Pixal input job")
        controlled = _require_mapping(
            job.get("controlled_request"),
            owner="base Pixal controlled request",
        )
        identifiers.append(
            _require_stable_id(
                controlled.get("instance_id"),
                owner="base Pixal input job instance_id",
            )
        )
    if len(identifiers) != len(set(identifiers)):
        _fail("base Pixal input manifest repeats an instance")
    if selected_instance_id is None:
        return None
    matches = [
        job
        for job, identifier in zip(jobs, identifiers)
        if identifier == selected_instance_id
    ]
    if len(matches) != 1:
        return None
    job = _require_mapping(matches[0], owner="selected base Pixal input job")
    expected_job_fields = {
        "legacy_tag",
        "candidate_tag",
        "asset_class",
        "route",
        "seed",
        "attempt_ordinal",
        "one_shot_execution",
        "reference",
        "output",
        "manifest",
        "controlled_request",
        "model_revisions",
        "parameters",
    }
    if (
        set(job) != expected_job_fields
        or job.get("legacy_tag") != selected_instance_id
        or job.get("asset_class") != "static_object"
        or job.get("route") != STATIC_ROUTE
        or job.get("attempt_ordinal") != 0
        or job.get("parameters")
        != {
            "resolution": 1024,
            "manual_fov": 0.2,
            "low_vram": False,
        }
        or not isinstance(job.get("model_revisions"), Mapping)
    ):
        _fail("selected base Pixal input job contract is invalid")
    controlled = _validate_controlled_request(
        job.get("controlled_request"),
        instance_id=selected_instance_id,
    )
    if job.get("seed") != controlled.get("generation_seed"):
        _fail("selected base Pixal job seed differs from controlled request")
    reference = _require_mapping(
        job.get("reference"), owner="selected base Pixal reference"
    )
    if (
        set(reference) != {"source", "pixal_input", "normalization"}
        or reference.get("normalization")
        != "pinned_isnet_general_use_alpha_v1"
    ):
        _fail("selected base Pixal reference contract is invalid")
    source = _file_record(
        reference.get("source"),
        policy,
        owner="selected base Pixal source image",
    )
    _validate_png_artifact(
        source,
        owner="selected base Pixal source image",
        expected_size=(1024, 1024),
        expected_mode="RGB",
    )
    pixal_input = _file_record(
        reference.get("pixal_input"),
        policy,
        owner="selected base Pixal RGBA",
        require_within=artifact.path.parent,
    )
    _validate_png_artifact(
        pixal_input,
        owner="selected base Pixal RGBA",
        expected_size=(1024, 1024),
        expected_mode="RGBA",
    )

    segmentation_items = value.get("segmentations")
    if (
        not isinstance(segmentation_items, list)
        or len(segmentation_items) != len(jobs)
    ):
        _fail("base Pixal segmentation index is invalid")
    segmentation_ids = [
        item.get("instance_id") if isinstance(item, Mapping) else None
        for item in segmentation_items
    ]
    if set(segmentation_ids) != set(identifiers) or len(segmentation_ids) != len(
        set(segmentation_ids)
    ):
        _fail("base Pixal segmentations do not cover every job")
    segmentation = next(
        item for item in segmentation_items if item.get("instance_id") == selected_instance_id
    )
    if (
        set(segmentation)
        != {
            "instance_id",
            "candidate_sha256",
            "alpha",
            "rgba",
            "foreground_fraction_at_128",
            "foreground_bbox_xyxy",
            "status",
        }
        or segmentation.get("candidate_sha256") != source.sha256
        or segmentation.get("status") != "passed"
    ):
        _fail("selected base Pixal segmentation contract changed")
    alpha = _file_record(
        segmentation.get("alpha"),
        policy,
        owner="selected base Pixal alpha mask",
        base=artifact.path.parent,
        require_within=artifact.path.parent,
    )
    _validate_png_artifact(
        alpha,
        owner="selected base Pixal alpha mask",
        expected_size=(1024, 1024),
        expected_mode="L",
        expected_extrema=(0, 255),
    )
    rgba = _file_record(
        segmentation.get("rgba"),
        policy,
        owner="selected base Pixal segmented RGBA",
        base=artifact.path.parent,
        require_within=artifact.path.parent,
    )
    _require_same_file_content(
        rgba,
        pixal_input,
        owner="selected base Pixal segmented RGBA",
    )
    _validate_rgba_alpha_binding(
        rgba,
        alpha,
        owner="selected base Pixal segmented RGBA",
    )
    _validate_rgba_source_rgb_binding(
        source,
        rgba,
        owner="selected base Pixal segmented RGBA",
    )
    fraction = _finite_number(
        segmentation.get("foreground_fraction_at_128"),
        owner="selected base Pixal foreground fraction",
    )
    bbox = segmentation.get("foreground_bbox_xyxy")
    if (
        not 0.05 <= fraction <= 0.85
        or not isinstance(bbox, list)
        or len(bbox) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 <= item <= 1024
            for item in bbox
        )
        or bbox[0] >= bbox[2]
        or bbox[1] >= bbox[3]
    ):
        _fail("selected base Pixal foreground readback is invalid")
    _validate_alpha_foreground_readback(
        alpha,
        declared_fraction=fraction,
        declared_bbox=bbox,
        owner="selected base Pixal foreground",
    )

    _validate_isnet_execution_receipt(
        value.get("isnet"),
        policy,
        artifact=artifact,
        pixal_jobs=jobs,
        segmentations=segmentation_items,
    )

    _validate_static_2d_review_lineage(
        value.get("review_batch"),
        policy,
        instance_id=selected_instance_id,
        controlled=controlled,
        source=source,
        expected_flux_batch_sha256=flux_batch_sha256,
        expected_one_shot_policy=_require_mapping(
            pixal_one_shot.get("policy"), owner="Pixal one-shot policy"
        ),
    )
    return job


def _validate_pixal_input_lineage(
    batch: Mapping[str, Any],
    policy: WorkspacePathPolicy,
    *,
    instance_id: str,
    attempt: Mapping[str, Any],
    pixal_glb: AuthenticatedArtifact,
    attempt_manifest: AuthenticatedArtifact,
) -> Mapping[str, Any]:
    inputs, inputs_artifact = _json_path_hash_record(
        batch.get("pixal_inputs"),
        policy,
        owner="Pixal input manifest",
        content_hash_field="manifest_sha256",
    )
    _require_content_hash(
        inputs,
        field="manifest_sha256",
        owner="Pixal input manifest",
    )
    schema = inputs.get("schema")
    if (
        batch.get("one_shot_execution") != inputs.get("one_shot_execution")
        or batch.get("upstream_flux_one_shot_evidence")
        != inputs.get("upstream_flux_one_shot_evidence")
    ):
        _fail("Pixal batch changed its input one-shot authority")
    selected_job: Mapping[str, Any] | None
    if schema == STATIC_PIXAL_INPUT_SCHEMA:
        selected_job = _validate_base_pixal_input_manifest(
            inputs,
            inputs_artifact,
            policy,
            selected_instance_id=instance_id,
        )
    elif schema == COMBINED_PIXAL_INPUT_SCHEMA:
        expected_fields = {
            "schema",
            "status",
            "state_classification",
            "formal_dataset_registration_authorized",
            "asset_class",
            "route",
            "one_shot_execution",
            "upstream_flux_one_shot_evidence",
            "model_revisions",
            "parameters",
            "combined_input_root",
            "parent_count",
            "parents",
            "pixal_output_root",
            "job_count",
            "jobs",
            "input_copies",
            "automatic_checks",
            "manifest_sha256",
        }
        jobs = inputs.get("jobs")
        copies = inputs.get("input_copies")
        parents = inputs.get("parents")
        if (
            set(inputs) != expected_fields
            or inputs.get("status") != "ready_for_pixal3d"
            or inputs.get("state_classification") != "research_candidate"
            or inputs.get("formal_dataset_registration_authorized") is not False
            or inputs.get("asset_class") != "static_object"
            or inputs.get("route") != STATIC_ROUTE
            or inputs.get("automatic_checks", {}).get("overall") != "passed"
            or inputs.get("automatic_checks", {}).get(
                "static_jobs_have_no_rig_or_animation_binding"
            )
            is not True
            or not isinstance(jobs, list)
            or not isinstance(copies, list)
            or not isinstance(parents, list)
            or len(parents) < 2
            or inputs.get("parent_count") != len(parents)
            or inputs.get("job_count") != len(jobs)
            or len(copies) != len(jobs)
        ):
            _fail("combined Pixal input manifest contract is invalid")
        combined_one_shot = _validate_one_shot_stage_record(
            inputs.get("one_shot_execution"),
            policy,
            stage="pixal3d",
            owner="combined Pixal input manifest",
        )
        combined_policy = _require_mapping(
            combined_one_shot.get("policy"),
            owner="combined Pixal one-shot policy",
        )
        combined_upstream = _require_mapping(
            inputs.get("upstream_flux_one_shot_evidence"),
            owner="combined upstream FLUX one-shot evidence",
        )
        upstream_parents = combined_upstream.get("parents")
        if (
            set(combined_upstream)
            != {"schema", "policy", "parent_count", "parents"}
            or combined_upstream.get("schema")
            != COMBINED_UPSTREAM_FLUX_EVIDENCE_SCHEMA
            or combined_upstream.get("policy") != combined_policy
            or combined_upstream.get("parent_count") != len(parents)
            or not isinstance(upstream_parents, list)
            or len(upstream_parents) != len(parents)
        ):
            _fail("combined upstream FLUX one-shot evidence changed")
        combined_matches = [
            item
            for item in jobs
            if isinstance(item, Mapping)
            and item.get("controlled_request", {}).get("instance_id")
            == instance_id
        ]
        copy_matches = [
            item
            for item in copies
            if isinstance(item, Mapping)
            and item.get("instance_id") == instance_id
        ]
        if len(combined_matches) != 1 or len(copy_matches) != 1:
            _fail("combined Pixal input lacks one selected job/copy")
        selected_job = _require_mapping(
            combined_matches[0], owner="selected combined Pixal job"
        )
        copy_binding = _require_mapping(
            copy_matches[0], owner="selected combined Pixal copy"
        )
        copy_fields = {
            "instance_id",
            "parent_ordinal",
            "parent_content_sha256",
            "parent_job_sha256",
            "source",
            "parent_pixal_input",
            "copied_pixal_input",
        }
        if set(copy_binding) != copy_fields:
            _fail("selected combined Pixal copy fields are invalid")
        parent_ordinal = copy_binding.get("parent_ordinal")
        if (
            isinstance(parent_ordinal, bool)
            or not isinstance(parent_ordinal, int)
            or not 0 <= parent_ordinal < len(parents)
        ):
            _fail("selected combined Pixal parent ordinal is invalid")
        parent_receipt = _require_mapping(
            parents[parent_ordinal],
            owner="selected combined Pixal parent receipt",
        )
        parent_fields = {
            "ordinal",
            "path",
            "sha256",
            "schema",
            "content_sha256",
            "asset_class",
            "route",
            "job_count",
        }
        if (
            set(parent_receipt) != parent_fields
            or parent_receipt.get("ordinal") != parent_ordinal
            or parent_receipt.get("schema") != STATIC_PIXAL_INPUT_SCHEMA
            or parent_receipt.get("asset_class") != "static_object"
            or parent_receipt.get("route") != STATIC_ROUTE
        ):
            _fail("selected combined Pixal parent receipt is invalid")
        parent, parent_artifact = _json_artifact(
            policy,
            parent_receipt.get("path"),
            owner="selected combined Pixal parent manifest",
            expected_sha256=_require_sha256(
                parent_receipt.get("sha256"),
                owner="selected combined Pixal parent file hash",
            ),
        )
        if (
            parent.get("manifest_sha256")
            != parent_receipt.get("content_sha256")
            or parent.get("job_count") != parent_receipt.get("job_count")
        ):
            _fail("selected combined Pixal parent receipt/content changed")
        parent_job = _validate_base_pixal_input_manifest(
            parent,
            parent_artifact,
            policy,
            selected_instance_id=instance_id,
        )
        if parent_job is None:
            _fail("selected combined Pixal parent lacks the selected job")
        selected_upstream = _require_mapping(
            upstream_parents[parent_ordinal],
            owner="selected combined upstream FLUX evidence",
        )
        if (
            set(selected_upstream) != {"parent_content_sha256", "evidence"}
            or selected_upstream.get("parent_content_sha256")
            != parent_receipt["content_sha256"]
            or selected_upstream.get("evidence")
            != parent.get("upstream_flux_one_shot_evidence")
        ):
            _fail("selected combined upstream FLUX evidence changed")
        _validate_upstream_flux_one_shot_evidence(
            selected_upstream.get("evidence"),
            policy,
            expected_policy=combined_policy,
        )
        parent_reference = _require_mapping(
            parent_job.get("reference"),
            owner="selected combined Pixal parent reference",
        )
        if (
            copy_binding.get("parent_content_sha256")
            != parent["manifest_sha256"]
            or copy_binding.get("parent_job_sha256")
            != canonical_json_sha256(parent_job)
            or copy_binding.get("source") != parent_reference.get("source")
            or copy_binding.get("parent_pixal_input")
            != parent_reference.get("pixal_input")
        ):
            _fail("selected combined Pixal copy differs from parent authority")
        copied_rgba = _file_record(
            copy_binding.get("copied_pixal_input"),
            policy,
            owner="selected combined Pixal copied RGBA",
            require_within=inputs_artifact.path.parent,
        )
        parent_rgba = _file_record(
            parent_reference.get("pixal_input"),
            policy,
            owner="selected combined Pixal parent RGBA",
            require_within=parent_artifact.path.parent,
        )
        _require_same_file_content(
            copied_rgba,
            parent_rgba,
            owner="selected combined Pixal RGBA copy",
        )
        expected_combined = deepcopy(dict(parent_job))
        expected_combined["reference"]["pixal_input"] = dict(
            copy_binding["copied_pixal_input"]
        )
        expected_combined["output"] = selected_job.get("output")
        expected_combined["manifest"] = selected_job.get("manifest")
        if dict(selected_job) != expected_combined:
            _fail("selected combined Pixal job differs from parent authority")
    else:
        _fail("unsupported Pixal input manifest schema")

    if selected_job is None:
        _fail("Pixal input manifest lacks the selected job")
    controlled = _validate_controlled_request(
        selected_job.get("controlled_request"),
        instance_id=instance_id,
    )
    reference = _require_mapping(
        selected_job.get("reference"), owner="selected Pixal input reference"
    )
    models = _require_mapping(
        selected_job.get("model_revisions"),
        owner="selected Pixal model revisions",
    )
    batch_models = _require_mapping(
        batch.get("models"), owner="Pixal batch model revisions"
    )
    if (
        set(models) != {"pixal3d", "dino"}
        or batch_models
        != {
            "pixal3d_revision": models["pixal3d"],
            "dino_revision": models["dino"],
        }
        or selected_job.get("parameters") != batch.get("parameters")
    ):
        _fail("selected Pixal model/parameter binding changed")
    raw_output = selected_job.get("output")
    raw_manifest = selected_job.get("manifest")
    if (
        not isinstance(raw_output, str)
        or not raw_output
        or Path(raw_output).expanduser().resolve() != pixal_glb.path
        or not isinstance(raw_manifest, str)
        or not raw_manifest
        or Path(raw_manifest).expanduser().resolve() != attempt_manifest.path
        or selected_job.get("seed") != attempt.get("seed")
        or selected_job.get("attempt_ordinal") != attempt.get("attempt_ordinal")
        or selected_job.get("one_shot_execution")
        != attempt.get("one_shot_execution")
        or reference.get("pixal_input") != attempt.get("pixal_input")
        or controlled.get("execution_job_id")
        != attempt.get("execution_job_id")
        or controlled.get("request_sha256") != attempt.get("request_sha256")
        or controlled.get("profile_schema_id")
        != attempt.get("profile_schema_id")
        or controlled.get("sampled_attributes")
        != attempt.get("sampled_attributes")
        or controlled.get("target_physical_profile")
        != attempt.get("target_physical_profile")
    ):
        _fail("selected Pixal attempt differs from authenticated input job")
    attempt_payload, _ = _json_artifact(
        policy,
        attempt_manifest.path,
        owner="selected Pixal attempt manifest",
        expected_sha256=attempt_manifest.sha256,
    )
    if (
        attempt_payload.get("backend") != "pixal3d"
        or attempt_payload.get("controlled_request") != controlled
        or attempt_payload.get("model", {}).get("revision")
        != models["pixal3d"]
        or attempt_payload.get("dino", {}).get("revision") != models["dino"]
        or attempt_payload.get("parameters")
        != {
            "low_vram": False,
            "manual_fov": 0.2,
            "resolution": 1024,
            "seed": selected_job["seed"],
        }
        or attempt_payload.get("one_shot_execution")
        != selected_job.get("one_shot_execution")
    ):
        _fail("selected Pixal attempt manifest/request binding changed")
    _file_record(
        attempt_payload.get("output"),
        policy,
        owner="selected Pixal attempt manifest output",
        expected=pixal_glb,
    )
    return controlled


def _validate_pixal_batch(
    value: Mapping[str, Any],
    artifact: AuthenticatedArtifact,
    policy: WorkspacePathPolicy,
    *,
    instance_id: str,
) -> tuple[Mapping[str, Any], AuthenticatedArtifact, Mapping[str, Any]]:
    expected_fields = {
        "schema",
        "status",
        "state_classification",
        "formal_dataset_registration_authorized",
        "one_shot_execution",
        "upstream_flux_one_shot_evidence",
        "started_at",
        "finished_at",
        "pixal_inputs",
        "models",
        "parameters",
        "gpus",
        "job_count",
        "passed_count",
        "failed_count",
        "attempts",
        "workers",
        "scheduling",
        "automatic_checks",
        "batch_sha256",
    }
    attempts = value.get("attempts")
    if (
        set(value) != expected_fields
        or value.get("schema") != STATIC_PIXAL_BATCH_SCHEMA
        or value.get("status") != "passed_generation_and_glb_readback"
        or value.get("state_classification") != "research_candidate"
        or value.get("formal_dataset_registration_authorized") is not False
        or not isinstance(attempts, list)
    ):
        _fail("Pixal batch contract is invalid")
    _require_content_hash(value, field="batch_sha256", owner="Pixal batch")
    job_count = _require_nonnegative_int(
        value.get("job_count"), owner="Pixal batch.job_count"
    )
    passed = _require_nonnegative_int(
        value.get("passed_count"), owner="Pixal batch.passed_count"
    )
    failed = _require_nonnegative_int(
        value.get("failed_count"), owner="Pixal batch.failed_count"
    )
    if job_count != len(attempts) or passed != job_count or failed != 0:
        _fail("Pixal batch counts are not an all-pass closure")
    matches = [
        item
        for item in attempts
        if isinstance(item, Mapping) and item.get("instance_id") == instance_id
    ]
    if len(matches) != 1:
        _fail("Pixal batch does not contain exactly one selected instance")
    attempt = matches[0]
    expected_attempt_fields = {
        "instance_id",
        "execution_job_id",
        "request_sha256",
        "profile_schema_id",
        "sampled_attributes",
        "target_physical_profile",
        "gpu",
        "seed",
        "attempt_ordinal",
        "one_shot_execution",
        "pixal_input",
        "output",
        "attempt_manifest",
        "mesh_readback",
        "timings",
        "status",
        "next_gate",
    }
    if (
        set(attempt) != expected_attempt_fields
        or attempt.get("status") != "passed_generation_and_glb_readback"
        or attempt.get("next_gate") != "static_visual_qa"
        or not isinstance(attempt.get("sampled_attributes"), Mapping)
        or not attempt["sampled_attributes"]
    ):
        _fail("selected Pixal attempt is not review-ready")
    _require_sha256(
        attempt.get("request_sha256"), owner="Pixal attempt.request_sha256"
    )
    mesh = _require_mapping(
        attempt.get("mesh_readback"), owner="Pixal attempt.mesh_readback"
    )
    if (
        mesh.get("skins") != 0
        or mesh.get("animations") != 0
        or mesh.get("vertices", 0) <= 0
        or mesh.get("triangles", 0) <= 0
        or mesh.get("materials", 0) <= 0
        or mesh.get("textures", 0) <= 0
    ):
        _fail("selected Pixal attempt is not a textured rigid mesh")
    output = _file_record(
        attempt.get("output"),
        policy,
        owner="selected Pixal GLB",
        base=artifact.path.parent,
        require_within=artifact.path.parent,
    )
    attempt_manifest = _file_record(
        attempt.get("attempt_manifest"),
        policy,
        owner="selected Pixal attempt manifest",
        base=artifact.path.parent,
        require_within=artifact.path.parent,
    )
    controlled = _validate_pixal_input_lineage(
        value,
        policy,
        instance_id=instance_id,
        attempt=attempt,
        pixal_glb=output,
        attempt_manifest=attempt_manifest,
    )
    return attempt, output, controlled


def _validate_static_render_manifest(
    value: Mapping[str, Any],
    *,
    pixal_glb: AuthenticatedArtifact,
    expected_mode: str,
    owner: str,
) -> None:
    raw_input = value.get("input")
    if (
        not isinstance(raw_input, str)
        or Path(raw_input).expanduser().resolve() != pixal_glb.path
        or value.get("front_axis") != "negative-y"
        or value.get("resolution") != [480, 480]
        or value.get("material_preview", {}).get("mode") != expected_mode
    ):
        _fail(f"{owner} camera/material contract changed")
    views = _require_mapping(value.get("views"), owner=f"{owner} views")
    if set(views) != {"front", "back", "side", "top", "quarter"}:
        _fail(f"{owner} five-view orbit is incomplete")
    for name, position in views.items():
        _finite_vector(position, 3, owner=f"{owner} view {name}")
    minimum = _finite_vector(
        value.get("bbox_min"), 3, owner=f"{owner} bounds minimum"
    )
    maximum = _finite_vector(
        value.get("bbox_max"), 3, owner=f"{owner} bounds maximum"
    )
    extent = _finite_vector(
        value.get("extent"), 3, owner=f"{owner} bounds extent"
    )
    if any(
        extent[index] <= 0.0
        or not math.isclose(
            maximum[index] - minimum[index],
            extent[index],
            abs_tol=1.0e-8,
        )
        for index in range(3)
    ):
        _fail(f"{owner} bounds are inconsistent")
    samples = value.get("samples")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        _fail(f"{owner} sample count is invalid")


def _validate_review_payload(
    value: Mapping[str, Any],
    artifact: AuthenticatedArtifact,
    policy: WorkspacePathPolicy,
    *,
    identity: tuple[str, str, str],
    pixal_attempt: Mapping[str, Any],
    controlled_request: Mapping[str, Any],
    pixal_glb: AuthenticatedArtifact,
    review_root: Path,
) -> None:
    if (
        set(value) != _STATIC_REVIEW_FIELDS
        or value.get("schema") != STATIC_REVIEW_SCHEMA
        or value.get("instance_id") != identity[0]
        or value.get("request_sha256") != identity[1]
        or value.get("profile_sha256") != identity[2]
        or value.get("asset_class") != "static_object"
        or value.get("route") != STATIC_ROUTE
        or value.get("state_classification") != "research_candidate"
        or value.get("formal_dataset_registration_authorized") is not False
        or value.get("visual_qa") != "pending"
        or value.get("next_gate") != "static_object_visual_decision"
    ):
        _fail("static visual review contract/identity is invalid")
    _require_content_hash(
        value, field="review_sha256", owner="static visual review"
    )
    if (
        value.get("execution_job_id") != pixal_attempt.get("execution_job_id")
        or controlled_request.get("request_sha256") != identity[1]
        or controlled_request.get("profile_sha256") != identity[2]
        or value.get("profile_schema_id") != pixal_attempt.get("profile_schema_id")
        or value.get("sampled_attributes")
        != pixal_attempt.get("sampled_attributes")
        or value.get("target_physical_profile")
        != pixal_attempt.get("target_physical_profile")
        or value.get("mesh_readback") != pixal_attempt.get("mesh_readback")
        or value.get("pixal_output") != pixal_attempt.get("output")
        or value.get("reference_rgba") != pixal_attempt.get("pixal_input")
    ):
        _fail("static review differs from its selected Pixal attempt")
    expected_physical_scale = {
        "status": "deferred_to_finalization",
        "control_attribute": value["target_physical_profile"].get(
            "control_attribute"
        ),
        "measurement": "height_cm",
        "target_physical_profile": value["target_physical_profile"],
    }
    if (
        value.get("physical_scale") != expected_physical_scale
        or value.get("orientation") != _STATIC_ORIENTATION_CONTRACT
    ):
        _fail("static review scale/orientation contract is invalid")
    _file_record(
        value.get("reference_rgba"),
        policy,
        owner="static review reference RGBA",
        base=review_root,
    )
    raw_manifest, _ = _json_file_record(
        value.get("raw_pbr_render_manifest"),
        policy,
        owner="static review raw PBR render manifest",
        base=review_root,
        require_within=review_root,
    )
    _validate_static_render_manifest(
        raw_manifest,
        pixal_glb=pixal_glb,
        expected_mode="raw_glb_material",
        owner="static review raw PBR render manifest",
    )
    _file_record(
        value.get("raw_pbr_blender_log"),
        policy,
        owner="static review raw_pbr_blender_log",
        base=review_root,
        require_within=review_root,
    )
    contact_sheet = _file_record(
        value.get("contact_sheet"),
        policy,
        owner="static review contact_sheet",
        base=review_root,
        require_within=review_root,
    )
    _validate_png_artifact(contact_sheet, owner="static review contact sheet")
    views = _require_mapping(
        value.get("raw_pbr_views"), owner="static review raw PBR views"
    )
    if set(views) != _REVIEW_VIEW_KEYS:
        _fail("static review raw PBR five-view closure is incomplete")
    raw_view_paths: set[Path] = set()
    for name, record in views.items():
        view = _file_record(
            record,
            policy,
            owner=f"static review raw PBR {name}",
            base=review_root,
            require_within=review_root,
        )
        if view.path in raw_view_paths:
            _fail("static review raw PBR views reuse one artifact")
        raw_view_paths.add(view.path)
        _validate_png_artifact(
            view,
            owner=f"static review raw PBR {name}",
            expected_size=(480, 480),
        )
    clay = _require_mapping(
        value.get("clay_geometry"), owner="static review clay geometry"
    )
    if clay.get("status") == "not_requested":
        if set(clay) != {"status"}:
            _fail("unrequested clay review must not contain artifacts")
    elif clay.get("status") == "included":
        if set(clay) != {"status", "render_manifest", "views", "blender_log"}:
            _fail("included clay review fields are invalid")
        clay_manifest, _ = _json_file_record(
            clay.get("render_manifest"),
            policy,
            owner="static review clay render manifest",
            base=review_root,
            require_within=review_root,
        )
        _validate_static_render_manifest(
            clay_manifest,
            pixal_glb=pixal_glb,
            expected_mode="neutral_clay_geometry_qa_v1",
            owner="static review clay render manifest",
        )
        _file_record(
            clay.get("blender_log"),
            policy,
            owner="static review clay Blender log",
            base=review_root,
            require_within=review_root,
        )
        clay_views = _require_mapping(
            clay.get("views"), owner="static review clay views"
        )
        if set(clay_views) != _REVIEW_VIEW_KEYS:
            _fail("static review clay five-view closure is incomplete")
        clay_view_paths: set[Path] = set()
        for name, record in clay_views.items():
            view = _file_record(
                record,
                policy,
                owner=f"static review clay {name}",
                base=review_root,
                require_within=review_root,
            )
            if view.path in clay_view_paths:
                _fail("static review clay views reuse one artifact")
            clay_view_paths.add(view.path)
            _validate_png_artifact(
                view,
                owner=f"static review clay {name}",
                expected_size=(480, 480),
            )
    else:
        _fail("static review clay status is invalid")
    mesh = _require_mapping(
        value.get("mesh_readback"), owner="static review mesh readback"
    )
    if mesh.get("skins") != 0 or mesh.get("animations") != 0:
        _fail("static review contains a skin or animation")
    if value.get("automatic_checks", {}).get("overall") != "passed":
        _fail("static review automatic checks are not passed")
    if artifact.sha256 == pixal_glb.sha256:
        _fail("static review manifest cannot masquerade as its Pixal GLB")


def _validate_review_batch(
    value: Mapping[str, Any],
    artifact: AuthenticatedArtifact,
    policy: WorkspacePathPolicy,
    *,
    instance_id: str,
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "status",
        "state_classification",
        "formal_dataset_registration_authorized",
        "asset_class",
        "route",
        "pixal_batch",
        "orientation",
        "render_contract",
        "review_count",
        "reviews",
        "automatic_checks",
        "review_batch_sha256",
    }
    reviews = value.get("reviews")
    if (
        set(value) != expected_fields
        or value.get("schema") != STATIC_REVIEW_BATCH_SCHEMA
        or value.get("status") != "rendered_pending_visual_qa"
        or value.get("state_classification") != "research_candidate"
        or value.get("formal_dataset_registration_authorized") is not False
        or value.get("asset_class") != "static_object"
        or value.get("route") != STATIC_ROUTE
        or not isinstance(reviews, list)
        or value.get("automatic_checks", {}).get("overall") != "passed"
    ):
        _fail("static review batch contract is invalid")
    _require_content_hash(
        value, field="review_batch_sha256", owner="static review batch"
    )
    if (
        _require_nonnegative_int(
            value.get("review_count"), owner="static review batch.review_count"
        )
        != len(reviews)
    ):
        _fail("static review batch count changed")
    review_index_fields = {
        "instance_id",
        "request_sha256",
        "review",
        "review_sha256",
        "contact_sheet",
        "status",
    }
    review_ids: list[str] = []
    for raw_index in reviews:
        index = _require_mapping(
            raw_index, owner="static review batch index"
        )
        if (
            set(index) != review_index_fields
            or index.get("status") != "rendered_pending_visual_qa"
        ):
            _fail("static review batch index contract is invalid")
        review_ids.append(
            _require_stable_id(
                index.get("instance_id"),
                owner="static review batch index instance_id",
            )
        )
        _require_sha256(
            index.get("request_sha256"),
            owner="static review batch index request_sha256",
        )
        _require_sha256(
            index.get("review_sha256"),
            owner="static review batch index review_sha256",
        )
    if len(review_ids) != len(set(review_ids)):
        _fail("static review batch repeats an instance")
    pixal_batch, pixal_batch_artifact = _json_path_hash_record(
        value.get("pixal_batch"),
        policy,
        owner="static review upstream Pixal batch",
        content_hash_field="batch_sha256",
    )
    _require_content_hash(
        pixal_batch, field="batch_sha256", owner="upstream Pixal batch"
    )
    pixal_attempt, pixal_glb, controlled_request = _validate_pixal_batch(
        pixal_batch,
        pixal_batch_artifact,
        policy,
        instance_id=instance_id,
    )
    indexes = [
        item
        for item in reviews
        if isinstance(item, Mapping) and item.get("instance_id") == instance_id
    ]
    if len(indexes) != 1:
        _fail("static review batch lacks exactly one selected review")
    index = indexes[0]
    review, review_artifact = _json_file_record(
        index.get("review"),
        policy,
        owner="selected static review",
        base=artifact.path.parent,
        require_within=artifact.path.parent,
    )
    review_hash = _require_sha256(
        index.get("review_sha256"), owner="static review index.review_sha256"
    )
    identity = (
        _require_stable_id(instance_id, owner="static review instance_id"),
        _require_sha256(
            review.get("request_sha256"), owner="static review request_sha256"
        ),
        _require_sha256(
            review.get("profile_sha256"), owner="static review profile_sha256"
        ),
    )
    if (
        review.get("review_sha256") != review_hash
        or index.get("request_sha256") != identity[1]
        or index.get("contact_sheet") != review.get("contact_sheet")
    ):
        _fail("static review index/content hash differs")
    _validate_review_payload(
        review,
        review_artifact,
        policy,
        identity=identity,
        pixal_attempt=pixal_attempt,
        controlled_request=controlled_request,
        pixal_glb=pixal_glb,
        review_root=artifact.path.parent,
    )
    expected_render_contract = {
        "reference_rgba_included": True,
        "raw_pbr_views": list(_REVIEW_VIEW_MAPPING),
        "clay_geometry": review["clay_geometry"]["status"],
    }
    if (
        value.get("orientation") != _STATIC_ORIENTATION_CONTRACT
        or value.get("render_contract") != expected_render_contract
    ):
        _fail("static review batch render/orientation contract changed")
    return {
        "identity": identity,
        "review": review,
        "review_artifact": review_artifact,
        "pixal_attempt": pixal_attempt,
        "controlled_request": controlled_request,
        "pixal_glb": pixal_glb,
        "pixal_batch_artifact": pixal_batch_artifact,
    }


def _validate_decision_batch(
    value: Mapping[str, Any],
    artifact: AuthenticatedArtifact,
    policy: WorkspacePathPolicy,
    *,
    instance_id: str,
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "status",
        "state_classification",
        "formal_dataset_registration_authorized",
        "asset_class",
        "route",
        "static_object_review_batch",
        "decision_input",
        "decision_count",
        "approved_count",
        "rejected_count",
        "decisions",
        "automatic_checks",
        "decision_batch_sha256",
    }
    indexes = value.get("decisions")
    if (
        set(value) != expected_fields
        or value.get("schema") != STATIC_DECISION_BATCH_SCHEMA
        or value.get("status") != "completed"
        or value.get("state_classification") != "research_candidate"
        or value.get("formal_dataset_registration_authorized") is not False
        or value.get("asset_class") != "static_object"
        or value.get("route") != STATIC_ROUTE
        or not isinstance(indexes, list)
        or value.get("automatic_checks", {}).get("overall") != "passed"
    ):
        _fail("static decision batch contract is invalid")
    _require_content_hash(
        value, field="decision_batch_sha256", owner="static decision batch"
    )
    decision_count = _require_nonnegative_int(
        value.get("decision_count"), owner="static decision batch.decision_count"
    )
    approved_count = _require_nonnegative_int(
        value.get("approved_count"), owner="static decision batch.approved_count"
    )
    rejected_count = _require_nonnegative_int(
        value.get("rejected_count"), owner="static decision batch.rejected_count"
    )
    if (
        decision_count != len(indexes)
        or approved_count + rejected_count != decision_count
    ):
        _fail("static decision batch counts changed")
    decision_index_fields = {
        "instance_id",
        "request_sha256",
        "profile_sha256",
        "pixal_output_sha256",
        "decision",
        "decision_sha256",
        "record",
    }
    decision_ids: list[str] = []
    actual_approved = 0
    for raw_index in indexes:
        decision_index = _require_mapping(
            raw_index, owner="static decision batch index"
        )
        if (
            set(decision_index) != decision_index_fields
            or decision_index.get("decision")
            not in {
                "approved_for_watertight_finalization",
                "rejected",
            }
        ):
            _fail("static decision batch index contract is invalid")
        decision_ids.append(
            _require_stable_id(
                decision_index.get("instance_id"),
                owner="static decision batch index instance_id",
            )
        )
        for field in (
            "request_sha256",
            "profile_sha256",
            "pixal_output_sha256",
            "decision_sha256",
        ):
            _require_sha256(
                decision_index.get(field),
                owner=f"static decision batch index {field}",
            )
        actual_approved += (
            decision_index["decision"]
            == "approved_for_watertight_finalization"
        )
    if (
        len(decision_ids) != len(set(decision_ids))
        or actual_approved != approved_count
        or len(indexes) - actual_approved != rejected_count
    ):
        _fail("static decision batch identity/status counts changed")
    review_batch, review_batch_artifact = _json_path_hash_record(
        value.get("static_object_review_batch"),
        policy,
        owner="static decision upstream review batch",
        content_hash_field="review_batch_sha256",
    )
    _require_content_hash(
        review_batch,
        field="review_batch_sha256",
        owner="upstream static review batch",
    )
    review_closure = _validate_review_batch(
        review_batch,
        review_batch_artifact,
        policy,
        instance_id=instance_id,
    )
    decision_input, _ = _json_file_record(
        value.get("decision_input"),
        policy,
        owner="static manual decision input",
    )
    if (
        set(decision_input)
        != {"schema", "static_object_review_batch_sha256", "decisions"}
        or decision_input.get("schema")
        != "avengine_controlled_static_object_review_decisions_v1"
        or decision_input.get("static_object_review_batch_sha256")
        != review_batch["review_batch_sha256"]
        or not isinstance(decision_input.get("decisions"), list)
    ):
        _fail("static manual decision input contract is invalid")
    matches = [
        item
        for item in indexes
        if isinstance(item, Mapping) and item.get("instance_id") == instance_id
    ]
    if len(matches) != 1 or matches[0].get("decision") != (
        "approved_for_watertight_finalization"
    ):
        _fail("static decision batch has no unique approved selected decision")
    index = matches[0]
    decision, decision_artifact = _json_file_record(
        index.get("record"),
        policy,
        owner="selected static decision",
        base=artifact.path.parent,
        require_within=artifact.path.parent,
    )
    manual_matches = [
        item
        for item in decision_input["decisions"]
        if isinstance(item, Mapping) and item.get("instance_id") == instance_id
    ]
    manual_fields = {
        "instance_id",
        "review_sha256",
        "decision",
        "checks",
        "attribute_evidence",
        "caveats",
        "notes",
    }
    if (
        len(manual_matches) != 1
        or set(manual_matches[0]) != manual_fields
        or any(
            decision.get(field) != manual_matches[0].get(field)
            for field in manual_fields
        )
    ):
        _fail("published static decision differs from manual decision authority")
    if set(decision.get("attribute_evidence", {})) != set(
        review_closure["review"].get("sampled_attributes", {})
    ):
        _fail("static decision attribute evidence differs from reviewed attributes")
    review_payload = review_closure["review"]
    if (
        decision.get("physical_scale") != review_payload.get("physical_scale")
        or decision.get("canonical_heading")
        != review_payload.get("orientation", {}).get("canonical_heading")
    ):
        _fail("static decision changed reviewed scale/heading deferral")
    identity, target_height_m, tolerance_m = _validate_decision(
        decision,
        policy,
        expected_review=review_closure["review_artifact"],
        expected_pixal=review_closure["pixal_glb"],
        expected_identity=review_closure["identity"],
        expected_review_sha256=review_closure["review"]["review_sha256"],
    )
    if (
        decision.get("decision_sha256") != index.get("decision_sha256")
        or decision.get("request_sha256") != index.get("request_sha256")
        or decision.get("profile_sha256") != index.get("profile_sha256")
        or decision["pixal_output"]["sha256"]
        != index.get("pixal_output_sha256")
    ):
        _fail("static decision index differs from selected decision")
    return {
        **review_closure,
        "identity": identity,
        "target_height_m": target_height_m,
        "tolerance_m": tolerance_m,
        "decision": decision,
        "decision_artifact": decision_artifact,
        "decision_batch_artifact": artifact,
        "approved_ids": {
            str(item["instance_id"])
            for item in indexes
            if isinstance(item, Mapping)
            and item.get("decision") == "approved_for_watertight_finalization"
        },
        "approved_identities": {
            str(item["instance_id"]): (
                str(item["instance_id"]),
                str(item["request_sha256"]),
                str(item["profile_sha256"]),
            )
            for item in indexes
            if isinstance(item, Mapping)
            and item.get("decision")
            == "approved_for_watertight_finalization"
        },
        "approved_indexes": {
            str(item["instance_id"]): item
            for item in indexes
            if isinstance(item, Mapping)
            and item.get("decision")
            == "approved_for_watertight_finalization"
        },
    }


def _validate_decision(
    value: Mapping[str, Any],
    policy: WorkspacePathPolicy,
    *,
    expected_review: AuthenticatedArtifact,
    expected_pixal: AuthenticatedArtifact,
    expected_identity: tuple[str, str, str],
    expected_review_sha256: str,
) -> tuple[tuple[str, str, str], float, float]:
    if (
        set(value) != _STATIC_DECISION_FIELDS
        or value.get("schema") != STATIC_DECISION_SCHEMA
        or value.get("asset_class") != "static_object"
        or value.get("route") != STATIC_ROUTE
        or value.get("decision") != "approved_for_watertight_finalization"
        or value.get("next_gate") != "watertight_then_static_finalization"
        or value.get("state_classification") != "research_candidate"
        or value.get("formal_dataset_registration_authorized") is not False
    ):
        _fail("static decision is not an approved research static-object decision")
    _require_content_hash(
        value, field="decision_sha256", owner="static decision"
    )
    identity = _require_identity(
        value, owner="static decision", expected=expected_identity
    )

    checks = _require_mapping(value.get("checks"), owner="static decision.checks")
    if set(checks) != _STATIC_REVIEW_CHECKS or any(
        item is not True for item in checks.values()
    ):
        _fail("static decision does not pass every required visual check")
    attribute_evidence = _require_mapping(
        value.get("attribute_evidence"),
        owner="static decision.attribute_evidence",
    )
    if not attribute_evidence or any(
        item != "passed_raw_pbr_visual" for item in attribute_evidence.values()
    ):
        _fail("static decision declared-attribute visual evidence is incomplete")
    caveats = value.get("caveats")
    notes = value.get("notes")
    if (
        not isinstance(caveats, list)
        or len(caveats) != len(set(caveats))
        or any(not isinstance(item, str) or not item.strip() for item in caveats)
        or not isinstance(notes, str)
        or not notes.strip()
    ):
        _fail("static decision notes/caveats are invalid")

    _file_record(
        value.get("pixal_output"),
        policy,
        owner="static decision approved Pixal GLB",
        expected=expected_pixal,
    )
    _file_record(
        value.get("review"),
        policy,
        owner="static decision source visual review",
        expected=expected_review,
    )
    if (
        _require_sha256(
            value.get("review_sha256"), owner="static decision.review_sha256"
        )
        != expected_review_sha256
    ):
        _fail("static decision review hash differs from reviewed content")
    physical = _require_mapping(
        value.get("target_physical_profile"),
        owner="static decision.target_physical_profile",
    )
    if (
        physical.get("control_attribute") is not None
        or physical.get("measurement") != "height_cm"
    ):
        _fail("static decision must declare an absolute physical height")
    target_cm = _finite_number(
        physical.get("target_value_cm"), owner="static decision target height"
    )
    tolerance_cm = _finite_number(
        physical.get("tolerance_cm"), owner="static decision height tolerance"
    )
    if target_cm <= 0.0 or tolerance_cm <= 0.0:
        _fail("static decision physical height range is invalid")
    return identity, target_cm / 100.0, tolerance_cm / 100.0


def _direct_json_artifact(
    raw: Any,
    policy: WorkspacePathPolicy,
    *,
    owner: str,
    base: Path,
) -> tuple[dict[str, Any], AuthenticatedArtifact]:
    if not isinstance(raw, str) or not raw:
        _fail(f"{owner} path must be a non-empty string")
    unresolved = Path(raw).expanduser()
    if not unresolved.is_absolute():
        unresolved = base / unresolved
    return _json_artifact(policy, unresolved, owner=owner)


def _validate_heading_authority(
    value: Mapping[str, Any],
    policy: WorkspacePathPolicy,
    *,
    identity: tuple[str, str, str],
    input_glb: AuthenticatedArtifact,
    owner: str,
    review_require_within: Path | None = None,
) -> AuthenticatedArtifact:
    required = {
        "schema",
        "instance_id",
        "request_sha256",
        "profile_sha256",
        "input_glb_sha256",
        "review_artifact",
        "reviewed_source_front_yaw_deg",
        "target_front_axis",
        "decision",
        "formal_dataset_registration_authorized",
    }
    if (
        set(value) != required
        or value.get("schema") != HEADING_EVIDENCE_SCHEMA
        or value.get("target_front_axis") != "positive-x"
        or value.get("decision") != "approved_for_positive_x_normalization"
        or value.get("formal_dataset_registration_authorized") is not False
    ):
        _fail(f"{owner} contract is invalid")
    _require_identity(value, owner=owner, expected=identity)
    if value.get("input_glb_sha256") != input_glb.sha256:
        _fail(f"{owner} input GLB hash changed")
    review_artifact = _file_record(
        value.get("review_artifact"),
        policy,
        owner=f"{owner} review artifact",
        require_within=review_require_within,
    )
    yaw = _finite_number(
        value.get("reviewed_source_front_yaw_deg"),
        owner=f"{owner} reviewed source yaw",
    )
    if not -180.0 <= yaw <= 180.0:
        _fail(f"{owner} reviewed source yaw is outside [-180,180]")
    return review_artifact


def _validate_anchor_authority(
    value: Mapping[str, Any],
    policy: WorkspacePathPolicy,
    *,
    identity: tuple[str, str, str],
    input_glb: AuthenticatedArtifact,
) -> AuthenticatedArtifact:
    required = {
        "schema",
        "instance_id",
        "request_sha256",
        "profile_sha256",
        "input_glb_sha256",
        "anchor_id",
        "anchor_type",
        "semantic_role",
        "selection",
        "review_evidence",
        "formal_dataset_registration_authorized",
    }
    if (
        set(value) != required
        or value.get("schema") != ANCHOR_AUTHORITY_SCHEMA
        or value.get("anchor_type") != "object_speaker"
        or value.get("formal_dataset_registration_authorized") is not False
    ):
        _fail("static emitter anchor authority contract is invalid")
    _require_identity(value, owner="static anchor authority", expected=identity)
    if value.get("input_glb_sha256") != input_glb.sha256:
        _fail("static anchor authority must bind the approved Pixal GLB")
    _require_stable_id(value.get("anchor_id"), owner="anchor authority anchor_id")
    _require_stable_id(
        value.get("semantic_role"), owner="anchor authority semantic_role"
    )
    _validate_anchor_selection(
        value.get("selection"), owner="static anchor authority selection"
    )
    return _file_record(
        value.get("review_evidence"),
        policy,
        owner="static anchor authority review evidence",
    )


def _validate_admission_plan(
    value: Mapping[str, Any],
    artifact: AuthenticatedArtifact,
    policy: WorkspacePathPolicy,
    *,
    decision_batch_sha256: str,
    approved_ids: set[str],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        set(value)
        != {
            "schema",
            "decision_batch_sha256",
            "instances",
            "formal_dataset_registration_authorized",
            "plan_sha256",
        }
        or value.get("schema") != STATIC_ADMISSION_PLAN_SCHEMA
        or value.get("decision_batch_sha256") != decision_batch_sha256
        or value.get("formal_dataset_registration_authorized") is not False
        or not isinstance(value.get("instances"), list)
        or not value["instances"]
    ):
        _fail("static admission plan contract is invalid")
    _require_content_hash(
        value, field="plan_sha256", owner="static admission plan"
    )
    jobs: dict[str, Mapping[str, Any]] = {}
    for raw in value["instances"]:
        item = _require_mapping(raw, owner="static admission plan instance")
        if set(item) != {
            "instance_id",
            "heading_evidence_path",
            "anchor_spec_path",
            "watertight_parameters",
        }:
            _fail("static admission plan instance fields are invalid")
        instance_id = _require_stable_id(
            item.get("instance_id"), owner="static admission plan instance_id"
        )
        if instance_id in jobs:
            _fail("static admission plan repeats an instance")
        _validate_watertight_parameters(item.get("watertight_parameters"))
        jobs[instance_id] = item
    if set(jobs) != approved_ids:
        _fail("static admission plan does not cover every approved decision once")
    instance_id = str(selected["identity"][0])
    item = jobs[instance_id]
    heading, heading_artifact = _direct_json_artifact(
        item.get("heading_evidence_path"),
        policy,
        owner="static heading authority",
        base=artifact.path.parent,
    )
    heading_review_artifact = _validate_heading_authority(
        heading,
        policy,
        identity=selected["identity"],
        input_glb=selected["pixal_glb"],
        owner="static heading authority",
    )
    anchor, anchor_artifact = _direct_json_artifact(
        item.get("anchor_spec_path"),
        policy,
        owner="static emitter anchor authority",
        base=artifact.path.parent,
    )
    anchor_review_artifact = _validate_anchor_authority(
        anchor,
        policy,
        identity=selected["identity"],
        input_glb=selected["pixal_glb"],
    )
    return {
        "all_jobs": jobs,
        "plan_artifact": artifact,
        "parameters": _validate_watertight_parameters(
            item["watertight_parameters"]
        ),
        "heading": heading,
        "heading_artifact": heading_artifact,
        "heading_review_artifact": heading_review_artifact,
        "anchor": anchor,
        "anchor_artifact": anchor_artifact,
        "anchor_review_artifact": anchor_review_artifact,
    }


def _validate_scene_readback(
    value: Any,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    scene = _require_mapping(value, owner="static finalization.scene_readback")
    if set(scene) != {
        "before",
        "after",
        "bounds_minimum_blender_xyz_m",
        "bounds_maximum_blender_xyz_m",
        "protected_scene_counts_preserved",
        "no_rig_or_animation",
    }:
        _fail("static finalization scene readback fields are invalid")
    if (
        scene.get("protected_scene_counts_preserved") is not True
        or scene.get("no_rig_or_animation") is not True
    ):
        _fail("static finalization scene readback did not preserve a rigid object")
    after = _require_mapping(
        scene.get("after"), owner="static finalization.scene_readback.after"
    )
    before = _require_mapping(
        scene.get("before"), owner="static finalization.scene_readback.before"
    )
    count_fields = {
        "mesh_count",
        "skin_count",
        "armature_count",
        "animation_count",
        "vertex_count",
        "face_count",
        "material_count",
        "image_count",
    }
    if set(before) != count_fields or set(after) != count_fields:
        _fail("static finalization scene count fields are invalid")
    if before != after:
        _fail("static finalization changed protected scene counts")
    if any(
        isinstance(after.get(field), bool)
        or not isinstance(after.get(field), int)
        or after[field] < 0
        for field in count_fields
    ):
        _fail("static finalization scene counts are invalid")
    for field in ("skin_count", "armature_count", "animation_count"):
        if after.get(field) != 0:
            _fail(f"static finalization scene readback has nonzero {field}")
    if any(
        after[field] < 1
        for field in (
            "mesh_count",
            "vertex_count",
            "face_count",
            "material_count",
            "image_count",
        )
    ):
        _fail("static finalization scene readback lacks rigid PBR geometry")
    minimum = _finite_vector(
        scene.get("bounds_minimum_blender_xyz_m"),
        3,
        owner="static finalization bounds minimum",
    )
    maximum = _finite_vector(
        scene.get("bounds_maximum_blender_xyz_m"),
        3,
        owner="static finalization bounds maximum",
    )
    if any(maximum[index] <= minimum[index] for index in range(3)):
        _fail("static finalization scene bounds are degenerate")
    return minimum, maximum


def _validate_finalization(
    value: Mapping[str, Any],
    artifact: AuthenticatedArtifact,
    policy: WorkspacePathPolicy,
    *,
    identity: tuple[str, str, str],
    target_height_m: float,
    tolerance_m: float,
    expected_watertight: AuthenticatedArtifact,
    expected_watertight_readback: _GLBReadback,
    expected_heading: AuthenticatedArtifact,
) -> tuple[AuthenticatedArtifact, float, _GLBReadback]:
    if (
        set(value)
        != {
            "schema",
            "created_at",
            "status",
            "asset_class",
            "instance_id",
            "request_sha256",
            "profile_sha256",
            "input",
            "output",
            "coordinate_system",
            "heading",
            "physical_scale",
            "grounding",
            "scene_readback",
            "formal_dataset_registration_authorized",
        }
        or value.get("schema") != STATIC_FINALIZATION_SCHEMA
        or value.get("status") != "passed_final_scaled_grounded_canonical_glb"
        or value.get("asset_class") != "static_object"
        or value.get("formal_dataset_registration_authorized") is not False
    ):
        _fail("static finalization is not a passed research static-object result")
    _require_identity(value, owner="static finalization", expected=identity)
    _validate_coordinate_system(
        value.get("coordinate_system"), owner="static finalization coordinate system"
    )
    _file_record(
        value.get("input"),
        policy,
        owner="static finalization watertight input",
        expected=expected_watertight,
    )
    finalized_glb = _file_record(
        value.get("output"), policy, owner="static finalization output GLB"
    )

    heading = _require_mapping(
        value.get("heading"), owner="static finalization.heading"
    )
    if (
        set(heading)
        != {
            "passed",
            "reviewed_source_front_yaw_deg",
            "target_front_axis",
            "applied_world_z_yaw_deg",
            "evidence",
        }
        or heading.get("passed") is not True
        or heading.get("target_front_axis") != "positive-x"
    ):
        _fail("static finalization +X-forward heading gate is not passed")
    reviewed_yaw = _finite_number(
        heading.get("reviewed_source_front_yaw_deg"),
        owner="static finalization reviewed source yaw",
    )
    applied_yaw = _finite_number(
        heading.get("applied_world_z_yaw_deg"),
        owner="static finalization applied yaw",
    )
    if not math.isclose(
        applied_yaw,
        -reviewed_yaw,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        _fail("static finalization applied yaw differs from reviewed heading")
    _file_record(
        heading.get("evidence"),
        policy,
        owner="static heading review evidence",
        expected=expected_heading,
    )

    physical = _require_mapping(
        value.get("physical_scale"), owner="static finalization.physical_scale"
    )
    if (
        set(physical)
        != {
            "passed",
            "measurement",
            "height_before_m",
            "target_height_m",
            "tolerance_m",
            "uniform_scale",
            "readback_height_m",
            "absolute_error_m",
        }
        or physical.get("passed") is not True
        or physical.get("measurement") != "height_m"
    ):
        _fail("static finalization physical-scale gate is not passed")
    height_before = _finite_number(
        physical.get("height_before_m"), owner="finalized source height"
    )
    declared_target = _finite_number(
        physical.get("target_height_m"), owner="finalized target height"
    )
    declared_tolerance = _finite_number(
        physical.get("tolerance_m"), owner="finalized height tolerance"
    )
    readback = _finite_number(
        physical.get("readback_height_m"), owner="finalized readback height"
    )
    absolute_error = _finite_number(
        physical.get("absolute_error_m"), owner="finalized height error"
    )
    uniform_scale = _finite_number(
        physical.get("uniform_scale"), owner="finalized uniform scale"
    )
    if (
        not math.isclose(declared_target, target_height_m, abs_tol=1.0e-9)
        or not math.isclose(declared_tolerance, tolerance_m, abs_tol=1.0e-9)
        or height_before <= 0.0
        or uniform_scale <= 0.0
        or not math.isclose(
            uniform_scale,
            declared_target / height_before,
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        )
        or not math.isclose(
            absolute_error,
            abs(readback - declared_target),
            abs_tol=1.0e-9,
        )
        or absolute_error > declared_tolerance
    ):
        _fail("static finalization scale/readback differs from approved physical profile")

    grounding = _require_mapping(
        value.get("grounding"), owner="static finalization.grounding"
    )
    if set(grounding) != {
        "passed",
        "method",
        "minimum_up_before_translation_m",
        "minimum_up_after_export_readback_m",
        "tolerance_m",
    }:
        _fail("static finalization grounding fields are invalid")
    _finite_number(
        grounding.get("minimum_up_before_translation_m"),
        owner="finalized pre-ground minimum",
    )
    ground_minimum = _finite_number(
        grounding.get("minimum_up_after_export_readback_m"),
        owner="finalized ground minimum",
    )
    ground_tolerance = _finite_number(
        grounding.get("tolerance_m"), owner="finalized ground tolerance"
    )
    if (
        grounding.get("passed") is not True
        or grounding.get("method") != "mesh_minimum_up_to_asset_root_zero_v1"
        or ground_tolerance < 0.0
        or abs(ground_minimum) > ground_tolerance
    ):
        _fail("static finalization grounding readback is not passed")
    scene_minimum, scene_maximum = _validate_scene_readback(
        value.get("scene_readback")
    )
    direct_readback = _read_glb_geometry(
        finalized_glb,
        owner="static finalization output",
    )
    scene_after = _require_mapping(
        value["scene_readback"]["after"],
        owner="static finalization direct scene readback",
    )
    direct_blender_minimum = (
        direct_readback.minimum_m[0],
        -direct_readback.maximum_m[2],
        direct_readback.minimum_m[1],
    )
    direct_blender_maximum = (
        direct_readback.maximum_m[0],
        -direct_readback.minimum_m[2],
        direct_readback.maximum_m[1],
    )
    watertight_height = (
        expected_watertight_readback.maximum_m[1]
        - expected_watertight_readback.minimum_m[1]
    )
    if (
        direct_readback.mesh_count != scene_after["mesh_count"]
        or direct_readback.vertex_count != scene_after["vertex_count"]
        or direct_readback.triangle_count != scene_after["face_count"]
        or len(direct_readback.material_names)
        != scene_after["material_count"]
        or direct_readback.image_count != scene_after["image_count"]
        or not direct_readback.has_complete_uvs
        or not direct_readback.has_complete_materials
        or expected_watertight_readback.mesh_count
        != direct_readback.mesh_count
        or expected_watertight_readback.vertex_count
        != direct_readback.vertex_count
        or expected_watertight_readback.triangle_count
        != direct_readback.triangle_count
        or expected_watertight_readback.material_names
        != direct_readback.material_names
        or expected_watertight_readback.image_count
        != direct_readback.image_count
        or not math.isclose(
            watertight_height,
            height_before,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        )
        or any(
            not math.isclose(
                direct_blender_minimum[index],
                scene_minimum[index],
                rel_tol=1.0e-6,
                abs_tol=1.0e-6,
            )
            or not math.isclose(
                direct_blender_maximum[index],
                scene_maximum[index],
                rel_tol=1.0e-6,
                abs_tol=1.0e-6,
            )
            for index in range(3)
        )
    ):
        _fail("static finalization manifest differs from direct GLB readback")
    if (
        not math.isclose(
            scene_maximum[2] - scene_minimum[2],
            readback,
            abs_tol=1.0e-8,
        )
        or not math.isclose(
            scene_minimum[2],
            ground_minimum,
            abs_tol=1.0e-8,
        )
    ):
        _fail("static finalization scene bounds differ from scale/ground readback")
    if artifact.sha256 == finalized_glb.sha256:
        _fail("static finalization manifest cannot masquerade as its GLB output")
    return finalized_glb, readback, direct_readback


def _validate_anchor_spec(
    value: Mapping[str, Any],
    policy: WorkspacePathPolicy,
    *,
    identity: tuple[str, str, str],
    finalized_glb: AuthenticatedArtifact,
    finalization: AuthenticatedArtifact,
    expected_anchor_id: str,
    expected_semantic_role: str,
    expected_method: str,
    authority: Mapping[str, Any],
    authority_review: AuthenticatedArtifact,
    review_require_within: Path,
) -> None:
    if (
        set(value)
        != {
            "schema",
            "instance_id",
            "request_sha256",
            "profile_sha256",
            "finalized_glb_sha256",
            "finalization_manifest_sha256",
            "anchor_id",
            "anchor_type",
            "semantic_role",
            "selection",
            "review_evidence",
            "formal_dataset_registration_authorized",
        }
        or value.get("schema") != ANCHOR_SPEC_SCHEMA
        or value.get("anchor_type") != "object_speaker"
        or value.get("formal_dataset_registration_authorized") is not False
    ):
        _fail("static emitter anchor spec is invalid")
    _require_identity(value, owner="static emitter anchor spec", expected=identity)
    if (
        value.get("finalized_glb_sha256") != finalized_glb.sha256
        or value.get("finalization_manifest_sha256") != finalization.sha256
        or value.get("anchor_id") != expected_anchor_id
        or value.get("semantic_role") != expected_semantic_role
    ):
        _fail("static emitter anchor spec lineage or anchor identity changed")
    _require_stable_id(value.get("semantic_role"), owner="anchor semantic_role")
    method = _validate_anchor_selection(
        value.get("selection"), owner="static emitter anchor selection"
    )
    if method != expected_method:
        _fail("static emitter anchor selection method changed")
    bound_review = _file_record(
        value.get("review_evidence"),
        policy,
        owner="static emitter anchor review evidence",
        require_within=review_require_within,
    )
    _require_same_file_content(
        bound_review,
        authority_review,
        owner="bound static anchor review evidence",
    )
    preserved = {
        key: item
        for key, item in value.items()
        if key
        not in {
            "schema",
            "finalized_glb_sha256",
            "finalization_manifest_sha256",
            "review_evidence",
        }
    }
    authority_preserved = {
        key: item
        for key, item in authority.items()
        if key not in {"schema", "input_glb_sha256", "review_evidence"}
    }
    if preserved != authority_preserved:
        _fail("bound static anchor spec changed reviewed semantic authority")


def _validate_bounds(value: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    bounds = _require_mapping(value, owner="emitter measurement.asset_bounds")
    if set(bounds) != {"minimum_m", "maximum_m", "extent_m"}:
        _fail("emitter measurement asset bounds fields are invalid")
    minimum = _finite_vector(bounds["minimum_m"], 3, owner="asset bounds minimum")
    maximum = _finite_vector(bounds["maximum_m"], 3, owner="asset bounds maximum")
    extent = _finite_vector(bounds["extent_m"], 3, owner="asset bounds extent")
    for index in range(3):
        if (
            extent[index] <= 0.0
            or not math.isclose(
                maximum[index] - minimum[index], extent[index], abs_tol=1.0e-8
            )
        ):
            _fail("emitter measurement asset bounds are inconsistent")
    return minimum, maximum


def _validate_resolved_surface_samples(
    value: Any,
    *,
    selection: Mapping[str, Any],
    method: str,
    minimum: Sequence[float],
    maximum: Sequence[float],
    offset: Sequence[float],
) -> None:
    resolved_samples = value
    selected_samples = selection.get("samples")
    if (
        selection.get("method") != method
        or not isinstance(resolved_samples, list)
        or not resolved_samples
        or not isinstance(selected_samples, list)
        or len(resolved_samples) != len(selected_samples)
    ):
        _fail("emitter resolved samples do not match reviewed anchor selection")

    extent = [
        float(maximum[index]) - float(minimum[index]) for index in range(3)
    ]
    diagonal = math.sqrt(sum(item * item for item in extent))
    bound_tolerance = max(diagonal * 1.0e-6, 1.0e-8)
    weighted_points: list[tuple[tuple[float, ...], float]] = []

    for index, (raw_resolved, raw_selected) in enumerate(
        zip(resolved_samples, selected_samples, strict=True)
    ):
        resolved = _require_mapping(
            raw_resolved, owner=f"emitter resolved sample[{index}]"
        )
        selected = _require_mapping(
            raw_selected, owner=f"reviewed anchor sample[{index}]"
        )
        common_fields = {
            "mesh_name",
            "triangle_index",
            "weight",
            "surface_point_m",
            "reviewed_target_m",
            "target_to_surface_distance_m",
        }
        method_field = (
            "barycentric"
            if method == "mesh_surface_barycentric_samples_v1"
            else "target_fraction_xyz"
        )
        if set(resolved) != common_fields | {method_field}:
            _fail("emitter resolved sample fields are invalid")

        mesh_name = resolved.get("mesh_name")
        triangle_index = resolved.get("triangle_index")
        if (
            not isinstance(mesh_name, str)
            or not mesh_name
            or isinstance(triangle_index, bool)
            or not isinstance(triangle_index, int)
            or triangle_index < 0
        ):
            _fail("emitter resolved sample mesh reference is invalid")
        weight = _finite_number(
            resolved.get("weight"), owner="emitter resolved sample weight"
        )
        expected_weight = _finite_number(
            selected.get("weight"), owner="reviewed anchor sample weight"
        )
        if weight <= 0.0 or not math.isclose(
            weight, expected_weight, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            _fail("emitter resolved sample weight changed")

        surface_point = _finite_vector(
            resolved.get("surface_point_m"),
            3,
            owner="emitter resolved surface point",
        )
        if any(
            surface_point[axis] < minimum[axis] - bound_tolerance
            or surface_point[axis] > maximum[axis] + bound_tolerance
            for axis in range(3)
        ):
            _fail("emitter resolved surface point lies outside asset bounds")

        distance = _finite_number(
            resolved.get("target_to_surface_distance_m"),
            owner="emitter resolved target-to-surface distance",
        )
        if distance < 0.0:
            _fail("emitter resolved target-to-surface distance is negative")

        if method == "mesh_surface_barycentric_samples_v1":
            if (
                mesh_name != selected.get("mesh_name")
                or triangle_index != selected.get("triangle_index")
            ):
                _fail("emitter resolved barycentric mesh reference changed")
            barycentric = _finite_vector(
                resolved.get("barycentric"),
                3,
                owner="emitter resolved barycentric coordinates",
            )
            expected_barycentric = _finite_vector(
                selected.get("barycentric"),
                3,
                owner="reviewed anchor barycentric coordinates",
            )
            if (
                any(
                    not math.isclose(
                        barycentric[axis],
                        expected_barycentric[axis],
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                    for axis in range(3)
                )
                or resolved.get("reviewed_target_m") is not None
                or not math.isclose(distance, 0.0, abs_tol=1.0e-12)
            ):
                _fail("emitter resolved barycentric sample changed")
        else:
            target_fraction = _finite_vector(
                resolved.get("target_fraction_xyz"),
                3,
                owner="emitter resolved target fraction",
            )
            expected_fraction = _finite_vector(
                selected.get("target_fraction_xyz"),
                3,
                owner="reviewed anchor target fraction",
            )
            if any(
                not math.isclose(
                    target_fraction[axis],
                    expected_fraction[axis],
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
                for axis in range(3)
            ):
                _fail("emitter resolved bbox target fraction changed")
            reviewed_target = _finite_vector(
                resolved.get("reviewed_target_m"),
                3,
                owner="emitter resolved reviewed target",
            )
            derived_target = tuple(
                minimum[axis] + extent[axis] * expected_fraction[axis]
                for axis in range(3)
            )
            derived_distance = math.sqrt(
                sum(
                    (surface_point[axis] - derived_target[axis]) ** 2
                    for axis in range(3)
                )
            )
            maximum_distance = diagonal * _finite_number(
                selection.get("maximum_search_distance_fraction"),
                owner="reviewed anchor maximum search distance fraction",
            )
            if (
                any(
                    not math.isclose(
                        reviewed_target[axis],
                        derived_target[axis],
                        rel_tol=1.0e-9,
                        abs_tol=1.0e-9,
                    )
                    for axis in range(3)
                )
                or distance > maximum_distance + bound_tolerance
                or not math.isclose(
                    distance,
                    derived_distance,
                    rel_tol=1.0e-7,
                    abs_tol=1.0e-8,
                )
            ):
                _fail("emitter resolved bbox nearest-surface result is inconsistent")
        weighted_points.append((surface_point, weight))

    total_weight = sum(weight for _point, weight in weighted_points)
    derived_offset = tuple(
        sum(point[axis] * weight for point, weight in weighted_points)
        / total_weight
        for axis in range(3)
    )
    if any(
        not math.isclose(
            float(offset[axis]),
            derived_offset[axis],
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        )
        for axis in range(3)
    ):
        _fail("emitter offset is not the weighted centroid of resolved samples")


def _validate_glb_emitter_samples(
    readback: _GLBReadback,
    *,
    resolved_samples: Sequence[Mapping[str, Any]],
    method: str,
    minimum: Sequence[float],
    maximum: Sequence[float],
) -> None:
    diagonal = math.sqrt(
        sum(
            (readback.maximum_m[index] - readback.minimum_m[index]) ** 2
            for index in range(3)
        )
    )
    tolerance = max(diagonal * 2.0e-6, 2.0e-7)
    if any(
        not math.isclose(
            readback.minimum_m[index],
            float(minimum[index]),
            rel_tol=1.0e-6,
            abs_tol=tolerance,
        )
        or not math.isclose(
            readback.maximum_m[index],
            float(maximum[index]),
            rel_tol=1.0e-6,
            abs_tol=tolerance,
        )
        for index in range(3)
    ):
        _fail("emitter asset bounds differ from direct finalized GLB readback")
    surfaces = {surface.name: surface for surface in readback.surfaces}
    for index, sample in enumerate(resolved_samples):
        mesh_name = str(sample["mesh_name"])
        surface = surfaces.get(mesh_name)
        triangle_index = int(sample["triangle_index"])
        if surface is None or not 0 <= triangle_index < len(surface.triangles):
            _fail(
                f"emitter resolved sample[{index}] is absent from finalized GLB"
            )
        triangle = surface.triangles[triangle_index]
        vertices = tuple(surface.vertices[item] for item in triangle)
        declared_point = tuple(
            float(item) for item in sample["surface_point_m"]
        )
        if method == "mesh_surface_barycentric_samples_v1":
            barycentric = tuple(float(item) for item in sample["barycentric"])
            direct_point = tuple(
                sum(
                    vertices[vertex][axis] * barycentric[vertex]
                    for vertex in range(3)
                )
                for axis in range(3)
            )
            direct_distance = 0.0
        else:
            target = tuple(float(item) for item in sample["reviewed_target_m"])
            direct_point = _closest_point_on_triangle(
                target,
                vertices[0],
                vertices[1],
                vertices[2],
            )
            direct_distance = math.sqrt(
                sum(
                    (direct_point[axis] - target[axis]) ** 2
                    for axis in range(3)
                )
            )
            global_distance = min(
                math.sqrt(
                    sum(
                        (
                            _closest_point_on_triangle(
                                target,
                                candidate.vertices[candidate_triangle[0]],
                                candidate.vertices[candidate_triangle[1]],
                                candidate.vertices[candidate_triangle[2]],
                            )[axis]
                            - target[axis]
                        )
                        ** 2
                        for axis in range(3)
                    )
                )
                for candidate in readback.surfaces
                for candidate_triangle in candidate.triangles
            )
            if not math.isclose(
                direct_distance,
                global_distance,
                rel_tol=1.0e-6,
                abs_tol=tolerance,
            ):
                _fail(
                    "emitter declared triangle is not nearest on finalized GLB"
                )
        declared_distance = float(sample["target_to_surface_distance_m"])
        if (
            any(
                not math.isclose(
                    direct_point[axis],
                    declared_point[axis],
                    rel_tol=1.0e-6,
                    abs_tol=tolerance,
                )
                for axis in range(3)
            )
            or not math.isclose(
                direct_distance,
                declared_distance,
                rel_tol=1.0e-6,
                abs_tol=tolerance,
            )
        ):
            _fail("emitter resolved sample differs from finalized GLB geometry")


def _validate_measurement(
    value: Mapping[str, Any],
    artifact: AuthenticatedArtifact,
    policy: WorkspacePathPolicy,
    *,
    identity: tuple[str, str, str],
    finalized_glb: AuthenticatedArtifact,
    finalized_glb_readback: _GLBReadback,
    finalization: AuthenticatedArtifact,
    anchor_authority: Mapping[str, Any],
    anchor_authority_review: AuthenticatedArtifact,
    expected_anchor_spec: AuthenticatedArtifact,
    expected_marker_glb: AuthenticatedArtifact,
    admission_root: Path,
) -> tuple[str, str, tuple[float, float, float], AuthenticatedArtifact]:
    if (
        set(value)
        != {
            "schema",
            "created_at",
            "status",
            "asset_class",
            "instance_id",
            "request_sha256",
            "profile_sha256",
            "input",
            "finalization_manifest",
            "anchor_spec",
            "coordinate_system",
            "asset_bounds",
            "emitter_anchor",
            "marker_review",
            "formal_dataset_registration_authorized",
        }
        or value.get("schema") != EMITTER_MEASUREMENT_SCHEMA
        or value.get("status") != "measured_pending_marker_visual_review"
        or value.get("asset_class") != "static_object"
        or value.get("formal_dataset_registration_authorized") is not False
    ):
        _fail("emitter measurement is not a pending-review static-object result")
    _require_identity(value, owner="emitter measurement", expected=identity)
    _validate_coordinate_system(
        value.get("coordinate_system"), owner="emitter measurement coordinate system"
    )
    _file_record(
        value.get("input"),
        policy,
        owner="emitter measurement finalized GLB",
        expected=finalized_glb,
    )
    _file_record(
        value.get("finalization_manifest"),
        policy,
        owner="emitter measurement finalization manifest",
        expected=finalization,
    )
    anchor_spec_artifact = _file_record(
        value.get("anchor_spec"),
        policy,
        owner="emitter measurement anchor spec",
        expected=expected_anchor_spec,
    )

    minimum, maximum = _validate_bounds(value.get("asset_bounds"))
    anchor = _require_mapping(
        value.get("emitter_anchor"), owner="emitter measurement.emitter_anchor"
    )
    anchor_id = _require_stable_id(
        anchor.get("anchor_id"), owner="emitter anchor_id"
    )
    semantic_role = _require_stable_id(
        anchor.get("semantic_role"), owner="emitter semantic_role"
    )
    offset = _finite_vector(anchor.get("offset_m"), 3, owner="emitter offset_m")
    method = anchor.get("method")
    resolved_samples = anchor.get("resolved_surface_samples")
    if (
        set(anchor)
        != {
            "anchor_id",
            "anchor_type",
            "semantic_role",
            "offset_m",
            "offset_space",
            "method",
            "aggregation",
            "resolved_surface_samples",
            "asset_specific_not_class_template",
            "animation_required",
        }
        or anchor.get("anchor_type") != "object_speaker"
        or anchor.get("offset_space") != "final_scaled_asset_root"
        or method
        not in {
            "mesh_surface_barycentric_samples_v1",
            "reviewed_bbox_fraction_nearest_surface_v1",
        }
        or anchor.get("aggregation") != "weighted_centroid"
        or not isinstance(resolved_samples, list)
        or not resolved_samples
        or anchor.get("asset_specific_not_class_template") is not True
        or anchor.get("animation_required") is not False
    ):
        _fail("emitter measurement is not a static object_speaker root offset")
    reviewed_selection = _require_mapping(
        anchor_authority.get("selection"),
        owner="reviewed static emitter anchor selection",
    )
    _validate_resolved_surface_samples(
        resolved_samples,
        selection=reviewed_selection,
        method=str(method),
        minimum=minimum,
        maximum=maximum,
        offset=offset,
    )
    _validate_glb_emitter_samples(
        finalized_glb_readback,
        resolved_samples=resolved_samples,
        method=str(method),
        minimum=minimum,
        maximum=maximum,
    )
    diagonal = math.sqrt(
        sum((maximum[index] - minimum[index]) ** 2 for index in range(3))
    )
    bound_tolerance = max(diagonal * 1.0e-6, 1.0e-8)
    if any(
        offset[index] < minimum[index] - bound_tolerance
        or offset[index] > maximum[index] + bound_tolerance
        for index in range(3)
    ):
        _fail("emitter offset lies outside finalized asset bounds")

    marker_review = _require_mapping(
        value.get("marker_review"), owner="emitter measurement.marker_review"
    )
    if (
        set(marker_review) != {"marker_glb", "marker_radius_m", "visual_review"}
        or marker_review.get("visual_review") != "pending"
    ):
        _fail("source emitter measurement must remain pending separate marker review")
    radius = _finite_number(
        marker_review.get("marker_radius_m"), owner="emitter marker radius"
    )
    if radius <= 0.0:
        _fail("emitter marker radius must be positive")
    marker_glb = _file_record(
        marker_review.get("marker_glb"),
        policy,
        owner="emitter marker GLB",
        expected=expected_marker_glb,
    )

    anchor_spec, _ = _json_artifact(
        policy,
        anchor_spec_artifact.path,
        owner="emitter anchor spec",
        expected_sha256=anchor_spec_artifact.sha256,
    )
    _validate_anchor_spec(
        anchor_spec,
        policy,
        identity=identity,
        finalized_glb=finalized_glb,
        finalization=finalization,
        expected_anchor_id=anchor_id,
        expected_semantic_role=semantic_role,
        expected_method=str(method),
        authority=anchor_authority,
        authority_review=anchor_authority_review,
        review_require_within=admission_root,
    )
    if artifact.sha256 == marker_glb.sha256:
        _fail("emitter measurement cannot masquerade as its marker GLB")
    return (
        anchor_id,
        semantic_role,
        (float(offset[0]), float(offset[1]), float(offset[2])),
        marker_glb,
    )


def _validate_marker_approval(
    value: Mapping[str, Any],
    policy: WorkspacePathPolicy,
    *,
    identity: tuple[str, str, str],
    finalized_glb: AuthenticatedArtifact,
    measurement: AuthenticatedArtifact,
    marker_glb: AuthenticatedArtifact,
) -> None:
    errors = json_schema_errors(
        value, STATIC_OBJECT_MARKER_VISUAL_APPROVAL_SCHEMA
    )
    if errors:
        _fail("marker visual approval schema failed: " + "; ".join(errors))
    if value.get("schema") != MARKER_VISUAL_APPROVAL_SCHEMA:
        _fail("marker visual approval schema identity changed")
    _require_identity(value, owner="marker visual approval", expected=identity)
    declared = _require_sha256(
        value.get("approval_content_sha256"),
        owner="marker visual approval content hash",
    )
    content = {
        key: item
        for key, item in value.items()
        if key != "approval_content_sha256"
    }
    if declared != canonical_json_sha256(content):
        _fail("marker visual approval canonical content hash changed")
    _file_record(
        value.get("finalized_glb"),
        policy,
        owner="marker approval finalized GLB",
        expected=finalized_glb,
    )
    _file_record(
        value.get("emitter_measurement"),
        policy,
        owner="marker approval source measurement",
        expected=measurement,
    )
    _file_record(
        value.get("marker_glb"),
        policy,
        owner="marker approval marker GLB",
        expected=marker_glb,
    )


def _validate_watertight_manifest(
    value: Mapping[str, Any],
    policy: WorkspacePathPolicy,
    *,
    pixal_glb: AuthenticatedArtifact,
    watertight_glb: AuthenticatedArtifact,
    parameters: Mapping[str, Any],
    admission_root: Path,
) -> _GLBReadback:
    if (
        set(value)
        != {
            "schema",
            "created_at",
            "input",
            "attribute_input",
            "output",
            "parameters",
            "topology",
            "surface_attributes",
            "torso_fold_repair",
            "authority_contract",
            "actual_faces",
            "status",
            "formal_dataset_registration_authorized",
        }
        or value.get("schema") != WATERTIGHT_MANIFEST_SCHEMA
        or value.get("status")
        != "research_candidate_pending_static_and_animation_qa"
        or value.get("formal_dataset_registration_authorized") is not False
    ):
        _fail("watertight manifest contract is invalid")
    frozen_input = _file_record(
        value.get("input"),
        policy,
        owner="watertight raw Pixal input",
        require_within=admission_root,
    )
    _require_same_file_content(
        frozen_input,
        pixal_glb,
        owner="watertight frozen Pixal input",
    )
    frozen_attribute_input = _file_record(
        value.get("attribute_input"),
        policy,
        owner="watertight attribute input",
        expected=frozen_input,
        require_within=admission_root,
        extra_fields=frozenset({"same_as_geometry_input"}),
    )
    _require_same_file_content(
        frozen_attribute_input,
        pixal_glb,
        owner="watertight frozen attribute input",
    )
    if value["attribute_input"].get("same_as_geometry_input") is not True:
        _fail("watertight attribute input must be the approved Pixal GLB")
    _file_record(
        value.get("output"),
        policy,
        owner="watertight output",
        expected=watertight_glb,
    )
    actual_parameters = _require_mapping(
        value.get("parameters"), owner="watertight parameter readback"
    )
    if set(actual_parameters) != _WATERTIGHT_PARAMETER_FIELDS | {"voxel_size"}:
        _fail("watertight parameter readback fields are invalid")
    for field, expected in parameters.items():
        if actual_parameters.get(field) != expected:
            _fail(f"watertight parameter readback changed: {field}")
    voxel_size = _finite_number(
        actual_parameters.get("voxel_size"), owner="watertight voxel size"
    )
    if voxel_size <= 0.0:
        _fail("watertight voxel size must be positive")
    topology = _require_mapping(
        value.get("topology"), owner="watertight topology"
    )
    if set(topology) != {"source", "after_voxel_remesh", "final"}:
        _fail("watertight topology readback fields are invalid")
    topology_fields = {
        "vertices",
        "edges",
        "faces",
        "boundary_edges",
        "wire_edges",
        "nonmanifold_edges_over_two_faces",
        "noncontiguous_two_face_edges",
    }
    topology_stages: dict[str, Mapping[str, Any]] = {}
    for stage_name in ("source", "after_voxel_remesh", "final"):
        stage_topology = _require_mapping(
            topology.get(stage_name), owner=f"watertight {stage_name} topology"
        )
        if set(stage_topology) != topology_fields:
            _fail(f"watertight {stage_name} topology fields are invalid")
        for field in topology_fields:
            item = stage_topology.get(field)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                _fail(f"watertight {stage_name} topology count is invalid")
        if any(stage_topology[field] <= 0 for field in ("vertices", "edges", "faces")):
            _fail(f"watertight {stage_name} topology is empty")
        topology_stages[stage_name] = stage_topology
    final = topology_stages["final"]
    if any(
        final.get(field) != 0
        for field in (
            "boundary_edges",
            "wire_edges",
            "nonmanifold_edges_over_two_faces",
            "noncontiguous_two_face_edges",
        )
    ):
        _fail("watertight boundary/nonmanifold gate is not passed")
    authority = _require_mapping(
        value.get("authority_contract"), owner="watertight authority contract"
    )
    backend = parameters["attribute_transfer_backend"]
    expected_authority = {
        "attribute_source_pbr_material_reused": backend != "bake",
        "attribute_source_uvs_transferred_by_nearest_surface": backend
        in {"bvh", "data-transfer"},
        "attribute_source_pbr_baked_to_new_uv_atlas": backend == "bake",
        "full_resolution_source_remains_geometry_authority": True,
        "source_geometry_replaced": True,
        "approved_skeleton_or_animation_touched": False,
    }
    if dict(authority) != expected_authority:
        _fail("watertight geometry/material authority contract changed")
    actual_faces = value.get("actual_faces")
    if (
        isinstance(actual_faces, bool)
        or not isinstance(actual_faces, int)
        or actual_faces <= 0
        or final["faces"] != actual_faces
    ):
        _fail("watertight actual face count is invalid")
    surface_attributes = _require_mapping(
        value.get("surface_attributes"),
        owner="watertight surface attributes",
    )
    if surface_attributes.get("backend") != backend:
        _fail("watertight surface-attribute backend changed")
    list_fields = {"uv_layers", "color_attributes", "material_slots"}
    for field in list_fields:
        items = surface_attributes.get(field)
        if (
            not isinstance(items, list)
            or (field != "color_attributes" and not items)
            or any(
                item is not None
                and (not isinstance(item, str) or not item)
                for item in items
            )
        ):
            _fail(f"watertight surface-attribute {field} is invalid")
    if backend == "data-transfer":
        if set(surface_attributes) != {"backend"} | list_fields:
            _fail("watertight data-transfer telemetry fields changed")
    elif backend == "bvh":
        if (
            set(surface_attributes)
            != {
                "backend",
                "bvh_query_count",
                "query_domain",
                "outward_ray_hit_count",
                "nearest_fallback_count",
                "outward_ray_offset",
                "source_triangle_count",
                "uv_layers",
                "color_attributes",
                "skipped_non_corner_color_attributes",
                "material_slots",
            }
            or surface_attributes.get("query_domain") != "face_corner"
        ):
            _fail("watertight BVH telemetry fields changed")
        query_count = _require_nonnegative_int(
            surface_attributes.get("bvh_query_count"),
            owner="watertight BVH query count",
        )
        ray_hits = _require_nonnegative_int(
            surface_attributes.get("outward_ray_hit_count"),
            owner="watertight BVH outward-ray hit count",
        )
        fallbacks = _require_nonnegative_int(
            surface_attributes.get("nearest_fallback_count"),
            owner="watertight BVH nearest fallback count",
        )
        source_triangles = _require_nonnegative_int(
            surface_attributes.get("source_triangle_count"),
            owner="watertight BVH source triangle count",
        )
        ray_offset = _finite_number(
            surface_attributes.get("outward_ray_offset"),
            owner="watertight BVH outward-ray offset",
        )
        skipped = surface_attributes.get(
            "skipped_non_corner_color_attributes"
        )
        if (
            query_count <= 0
            or ray_hits + fallbacks != query_count
            or source_triangles <= 0
            or ray_offset <= 0.0
            or not isinstance(skipped, list)
            or any(not isinstance(item, str) or not item for item in skipped)
        ):
            _fail("watertight BVH telemetry values changed")
    else:
        if (
            set(surface_attributes)
            != {
                "backend",
                "bake_resolution",
                "bake_device",
                "ray_distance",
                "cage_extrusion",
                "uv_layers",
                "baked_images",
                "base_color_bake_type",
                "color_attributes",
                "material_slots",
                "metallic_policy",
                "base_color_encoding_policy",
                "base_color_gain",
            }
            or surface_attributes.get("bake_resolution")
            != parameters["bake_resolution"]
            or surface_attributes.get("bake_device") != "CPU"
            or surface_attributes.get("base_color_bake_type")
            != "EMIT_FROM_PRINCIPLED_BASE_COLOR"
            or surface_attributes.get("metallic_policy")
            != "constant_zero_for_nonmetallic_animal_surface"
            or surface_attributes.get("base_color_encoding_policy")
            != parameters["base_color_encoding_policy"]
            or surface_attributes.get("base_color_gain")
            != parameters["base_color_gain"]
            or not isinstance(surface_attributes.get("baked_images"), list)
            or len(surface_attributes["baked_images"]) != 2
            or any(
                not isinstance(item, str) or not item
                for item in surface_attributes["baked_images"]
            )
            or _finite_number(
                surface_attributes.get("ray_distance"),
                owner="watertight bake ray distance",
            )
            <= 0.0
            or _finite_number(
                surface_attributes.get("cage_extrusion"),
                owner="watertight bake cage extrusion",
            )
            <= 0.0
        ):
            _fail("watertight bake telemetry changed")

    glb_readback = _read_glb_geometry(
        watertight_glb,
        owner="watertight output",
    )
    if (
        glb_readback.mesh_count != 1
        or not glb_readback.has_complete_uvs
        or not glb_readback.has_complete_materials
        or not glb_readback.material_names
        or glb_readback.image_count <= 0
        or tuple(surface_attributes["material_slots"])
        != glb_readback.material_names
    ):
        _fail("watertight output GLB lacks its declared rigid PBR surface")
    direct_topology = _glb_welded_topology(
        glb_readback,
        owner="watertight output",
    )
    if (
        direct_topology["vertices"] != final["vertices"]
        or direct_topology["triangles"] < final["faces"]
        or direct_topology["triangles"] > parameters["target_faces"] * 2
        or direct_topology["edges"]
        != (
            final["edges"]
            + direct_topology["triangles"]
            - final["faces"]
        )
        or any(
            direct_topology[field] != final[field]
            for field in (
                "boundary_edges",
                "wire_edges",
                "nonmanifold_edges_over_two_faces",
                "noncontiguous_two_face_edges",
            )
        )
    ):
        _fail("watertight manifest differs from direct output GLB topology")
    torso_fold_repair = _require_mapping(
        value.get("torso_fold_repair"),
        owner="watertight torso-fold repair",
    )
    repair_iterations = parameters["torso_fold_repair_iterations"]
    if repair_iterations == 0:
        if dict(torso_fold_repair) != {
            "iterations": 0,
            "selected_vertices": 0,
            "policy": "disabled",
        }:
            _fail("disabled watertight torso-fold repair readback changed")
    else:
        if (
            set(torso_fold_repair)
            != {
                "iterations",
                "selected_vertices",
                "longitudinal_axis",
                "normalized_longitudinal_range",
                "normalized_vertical_range",
                "fade",
                "lambda_factor",
                "policy",
            }
            or isinstance(torso_fold_repair.get("iterations"), bool)
            or torso_fold_repair.get("iterations") != repair_iterations
            or isinstance(torso_fold_repair.get("selected_vertices"), bool)
            or not isinstance(torso_fold_repair.get("selected_vertices"), int)
            or torso_fold_repair["selected_vertices"] <= 0
            or isinstance(torso_fold_repair.get("longitudinal_axis"), bool)
            or torso_fold_repair.get("longitudinal_axis") not in {0, 1}
            or torso_fold_repair.get("normalized_longitudinal_range")
            != [0.25, 0.70]
            or torso_fold_repair.get("normalized_vertical_range")
            != [0.34, 0.72]
            or torso_fold_repair.get("fade") != 0.08
            or torso_fold_repair.get("lambda_factor") != 0.18
            or torso_fold_repair.get("policy")
            != "weighted_mid_torso_only_preserve_volume"
        ):
            _fail("watertight torso-fold repair readback changed")
    return glb_readback


def _validate_stage_command_inputs(
    raw_manifest_record: Any,
    policy: WorkspacePathPolicy,
    *,
    instance_id: str,
    stage: str,
    command: Sequence[str],
    command_sha256: str,
    admission_root: Path,
    stage_receipt: AuthenticatedArtifact,
) -> Mapping[str, Any]:
    manifest, manifest_artifact = _json_file_record(
        raw_manifest_record,
        policy,
        owner=f"{stage} command-input manifest",
        base=admission_root,
        require_within=admission_root,
    )
    expected_manifest_path = (
        stage_receipt.path.parent / "command_input_manifest.json"
    ).resolve()
    if manifest_artifact.path != expected_manifest_path:
        _fail(f"{stage} command-input manifest path changed")
    if (
        set(manifest)
        != {
            "schema",
            "instance_id",
            "stage",
            "command",
            "command_sha256",
            "blender",
            "python_tool",
            "python_dependencies",
            "formal_dataset_registration_authorized",
            "manifest_sha256",
        }
        or manifest.get("schema")
        != STATIC_ADMISSION_COMMAND_INPUT_MANIFEST_SCHEMA
        or manifest.get("instance_id") != instance_id
        or manifest.get("stage") != stage
        or manifest.get("command") != list(command)
        or manifest.get("command_sha256") != command_sha256
        or manifest.get("formal_dataset_registration_authorized") is not False
    ):
        _fail(f"{stage} command-input manifest contract changed")
    _require_content_hash(
        manifest,
        field="manifest_sha256",
        owner=f"{stage} command-input manifest",
    )

    blender = _require_mapping(
        manifest.get("blender"), owner=f"{stage} Blender command identity"
    )
    if (
        set(blender) != {"configured_path", "resolved_path", "record"}
        or not isinstance(blender.get("configured_path"), str)
        or not blender["configured_path"]
        or not isinstance(blender.get("resolved_path"), str)
        or not blender["resolved_path"]
    ):
        _fail(f"{stage} Blender command identity changed")
    configured_path = Path(blender["configured_path"]).expanduser()
    resolved_path = Path(blender["resolved_path"]).expanduser()
    if not configured_path.is_absolute() or not resolved_path.is_absolute():
        _fail(f"{stage} Blender paths must be absolute")
    blender_artifact = _file_record(
        blender.get("record"),
        policy,
        owner=f"{stage} Blender binary",
    )
    try:
        if (
            configured_path.resolve(strict=True) != blender_artifact.path
            or resolved_path.resolve(strict=True) != blender_artifact.path
        ):
            _fail(f"{stage} Blender configured/resolved identity changed")
    except OSError as error:
        raise StaticObjectRegistrationError(
            f"{stage} Blender configured/resolved identity is unavailable: {error}"
        ) from error

    expected_tool_path = (
        admission_root
        / ".runtime_commands"
        / "tools"
        / _STAGE_PYTHON_TOOLS[stage]
    ).resolve()
    python_tool = _file_record(
        manifest.get("python_tool"),
        policy,
        owner=f"{stage} frozen Python tool",
        require_within=admission_root,
    )
    if python_tool.path != expected_tool_path:
        _fail(f"{stage} frozen Python tool path changed")

    dependencies = _require_mapping(
        manifest.get("python_dependencies"),
        owner=f"{stage} frozen Python dependencies",
    )
    expected_dependencies = _STAGE_PYTHON_DEPENDENCIES[stage]
    if set(dependencies) != expected_dependencies:
        _fail(f"{stage} frozen Python dependency coverage changed")
    for name in expected_dependencies:
        dependency = _file_record(
            dependencies[name],
            policy,
            owner=f"{stage} frozen Python dependency {name}",
            require_within=admission_root,
        )
        expected_dependency_path = (
            admission_root / ".runtime_commands" / "tools" / name
        ).resolve()
        if dependency.path != expected_dependency_path:
            _fail(f"{stage} frozen Python dependency path changed: {name}")

    python_indexes = [
        index for index, item in enumerate(command) if item == "--python"
    ]
    if (
        not command
        or command[0] != blender["record"]["path"]
        or len(python_indexes) != 1
        or python_indexes[0] + 1 >= len(command)
        or command[python_indexes[0] + 1] != manifest["python_tool"]["path"]
    ):
        _fail(f"{stage} command does not use its sealed execution inputs")
    return manifest


def _validate_stage_receipt(
    value: Mapping[str, Any],
    artifact: AuthenticatedArtifact,
    policy: WorkspacePathPolicy,
    *,
    identity: tuple[str, str, str],
    stage: str,
    admission_root: Path,
) -> Mapping[str, Any]:
    if (
        set(value)
        != {
            "schema",
            "instance_id",
            "stage",
            "status",
            "command",
            "command_sha256",
            "command_input_manifest",
            "execution",
            "log",
            "inputs",
            "outputs",
            "validation",
            "formal_dataset_registration_authorized",
            "receipt_sha256",
        }
        or value.get("schema") != STATIC_ADMISSION_STAGE_RECEIPT_SCHEMA
        or value.get("instance_id") != identity[0]
        or value.get("stage") != stage
        or value.get("status") != "passed"
        or value.get("formal_dataset_registration_authorized") is not False
    ):
        _fail(f"{stage} stage receipt contract is invalid")
    _require_content_hash(
        value, field="receipt_sha256", owner=f"{stage} stage receipt"
    )
    command = value.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
        or value.get("command_sha256") != canonical_json_sha256(command)
    ):
        _fail(f"{stage} stage command hash is invalid")
    command_manifest = _validate_stage_command_inputs(
        value.get("command_input_manifest"),
        policy,
        instance_id=identity[0],
        stage=stage,
        command=command,
        command_sha256=value["command_sha256"],
        admission_root=admission_root,
        stage_receipt=artifact,
    )
    execution = _require_mapping(
        value.get("execution"), owner=f"{stage} stage execution"
    )
    if (
        set(execution)
        != {
            "started_at",
            "finished_at",
            "wall_seconds",
            "returncode",
            "timeout_seconds",
            "error",
        }
        or execution.get("returncode") != 0
        or execution.get("error") is not None
    ):
        _fail(f"{stage} stage execution is not a clean pass")
    if (
        _finite_number(
            execution.get("wall_seconds"), owner=f"{stage} stage wall_seconds"
        )
        < 0.0
        or _require_nonnegative_int(
            execution.get("timeout_seconds"),
            owner=f"{stage} stage timeout_seconds",
        )
        <= 0
    ):
        _fail(f"{stage} stage execution timing is invalid")
    _file_record(
        value.get("log"),
        policy,
        owner=f"{stage} stage log",
        base=admission_root,
        require_within=admission_root,
    )
    _require_mapping(value.get("inputs"), owner=f"{stage} stage inputs")
    _require_mapping(value.get("outputs"), owner=f"{stage} stage outputs")
    validation = _require_mapping(
        value.get("validation"), owner=f"{stage} stage validation"
    )
    expected_validation: Mapping[str, Any]
    if stage == "watertight":
        expected_validation = {
            "boundary_edges": 0,
            "nonmanifold_edges_over_two_faces": 0,
            "no_rig_or_animation": True,
        }
    elif stage == "finalization":
        expected_validation = {
            "heading_passed": True,
            "physical_scale_passed": True,
            "grounding_passed": True,
            "no_rig_or_animation": True,
        }
    else:
        expected_validation = {"marker_visual_review": "pending"}
    if validation != expected_validation:
        _fail(f"{stage} stage validation summary is invalid")
    for field in ("started_at", "finished_at"):
        if (
            not isinstance(execution.get(field), str)
            or not execution[field].strip()
        ):
            _fail(f"{stage} stage execution {field} is invalid")
    if artifact.sha256 == value.get("command_sha256"):
        _fail(f"{stage} stage receipt cannot masquerade as its command hash")
    return command_manifest


def _validate_exact_stage_command(
    value: Mapping[str, Any],
    command_manifest: Mapping[str, Any],
    *,
    stage: str,
    arguments: Sequence[str],
) -> None:
    expected = [
        command_manifest["blender"]["record"]["path"],
        "-b",
        "--python-exit-code",
        "2",
        "--python",
        command_manifest["python_tool"]["path"],
        "--",
        *arguments,
    ]
    if value.get("command") != expected:
        _fail(f"{stage} command argv differs from its sealed stage I/O")


def _validate_indexed_admission_job_closure(
    job_index: Mapping[str, Any],
    policy: WorkspacePathPolicy,
    *,
    admission_root: Path,
    decision_index: Mapping[str, Any],
    decision_batch_artifact: AuthenticatedArtifact,
    plan_item: Mapping[str, Any],
    plan_artifact: AuthenticatedArtifact,
) -> tuple[Mapping[str, Any], AuthenticatedArtifact]:
    job_receipt, job_artifact = _json_file_record(
        job_index.get("job_receipt"),
        policy,
        owner="indexed static admission job receipt",
        base=admission_root,
        require_within=admission_root,
    )
    job_fields = {
        "schema",
        "status",
        "asset_class",
        "route",
        "instance_id",
        "request_sha256",
        "profile_sha256",
        "decision",
        "pixal_output",
        "heading_authority",
        "anchor_authority",
        "bound_heading_evidence",
        "bound_anchor_spec",
        "watertight_glb",
        "finalized_glb",
        "emitter_measurement",
        "emitter_marker_glb",
        "stage_receipts",
        "path_rebinding",
        "marker_review",
        "next_gate",
        "formal_dataset_registration_authorized",
        "receipt_sha256",
    }
    identity = (
        job_index.get("instance_id"),
        job_index.get("request_sha256"),
        job_index.get("profile_sha256"),
    )
    if (
        set(job_receipt) != job_fields
        or job_receipt.get("schema")
        != STATIC_ADMISSION_JOB_RECEIPT_SCHEMA
        or job_receipt.get("status")
        != "passed_pending_emitter_marker_review"
        or job_receipt.get("asset_class") != "static_object"
        or job_receipt.get("route") != STATIC_ROUTE
        or job_receipt.get("marker_review") != "pending"
        or job_receipt.get("next_gate") != "emitter_marker_visual_review"
        or job_receipt.get("formal_dataset_registration_authorized") is not False
        or job_receipt.get("receipt_sha256")
        != job_index.get("receipt_sha256")
    ):
        _fail("indexed static admission job receipt is invalid")
    _require_content_hash(
        job_receipt,
        field="receipt_sha256",
        owner="indexed static admission job receipt",
    )
    _require_identity(
        job_receipt,
        owner="indexed static admission job receipt",
        expected=identity,
    )

    leaf_artifacts = {
        field: _file_record(
            job_receipt.get(field),
            policy,
            owner=f"indexed static admission job {field}",
        )
        for field in (
            "decision",
            "pixal_output",
            "heading_authority",
            "anchor_authority",
            "bound_heading_evidence",
            "bound_anchor_spec",
            "watertight_glb",
            "finalized_glb",
            "emitter_measurement",
            "emitter_marker_glb",
        )
    }
    indexed_decision_artifact = _file_record(
        decision_index.get("record"),
        policy,
        owner="indexed approved static decision",
        base=decision_batch_artifact.path.parent,
        require_within=decision_batch_artifact.path.parent,
    )
    if indexed_decision_artifact != leaf_artifacts["decision"]:
        _fail("indexed admission job changed its approved decision record")
    decision, _ = _json_artifact(
        policy,
        leaf_artifacts["decision"].path,
        owner="indexed static admission decision",
        expected_sha256=leaf_artifacts["decision"].sha256,
    )
    if (
        set(decision) != _STATIC_DECISION_FIELDS
        or decision.get("schema") != STATIC_DECISION_SCHEMA
        or decision.get("asset_class") != "static_object"
        or decision.get("route") != STATIC_ROUTE
        or decision.get("decision")
        != "approved_for_watertight_finalization"
        or decision.get("next_gate")
        != "watertight_then_static_finalization"
        or decision.get("state_classification") != "research_candidate"
        or decision.get("formal_dataset_registration_authorized") is not False
        or decision.get("decision_sha256")
        != decision_index.get("decision_sha256")
        or decision["pixal_output"].get("sha256")
        != decision_index.get("pixal_output_sha256")
    ):
        _fail("indexed static admission decision authority changed")
    _require_content_hash(
        decision,
        field="decision_sha256",
        owner="indexed static admission decision",
    )
    _require_identity(
        decision,
        owner="indexed static admission decision",
        expected=identity,
    )
    _file_record(
        decision.get("pixal_output"),
        policy,
        owner="indexed static decision Pixal GLB",
        expected=leaf_artifacts["pixal_output"],
    )
    physical = _require_mapping(
        decision.get("target_physical_profile"),
        owner="indexed static decision physical profile",
    )
    if (
        set(physical)
        != {
            "control_attribute",
            "measurement",
            "target_value_cm",
            "tolerance_cm",
        }
        or physical.get("control_attribute") is not None
        or physical.get("measurement") != "height_cm"
    ):
        _fail("indexed static decision physical profile changed")
    target_height_m = _finite_number(
        physical.get("target_value_cm"),
        owner="indexed static decision target height",
    ) / 100.0
    tolerance_m = _finite_number(
        physical.get("tolerance_cm"),
        owner="indexed static decision height tolerance",
    ) / 100.0
    if target_height_m <= 0.0 or tolerance_m <= 0.0:
        _fail("indexed static decision physical range is invalid")

    if plan_item.get("instance_id") != identity[0]:
        _fail("indexed static admission plan identity changed")
    plan_parameters = _validate_watertight_parameters(
        plan_item.get("watertight_parameters")
    )
    heading_authority, heading_artifact = _direct_json_artifact(
        plan_item.get("heading_evidence_path"),
        policy,
        owner="indexed static heading authority",
        base=plan_artifact.path.parent,
    )
    heading_review = _validate_heading_authority(
        heading_authority,
        policy,
        identity=identity,
        input_glb=leaf_artifacts["pixal_output"],
        owner="indexed static heading authority",
    )
    if heading_artifact != leaf_artifacts["heading_authority"]:
        _fail("indexed static job changed its planned heading authority")
    anchor_authority, anchor_artifact = _direct_json_artifact(
        plan_item.get("anchor_spec_path"),
        policy,
        owner="indexed static anchor authority",
        base=plan_artifact.path.parent,
    )
    anchor_review = _validate_anchor_authority(
        anchor_authority,
        policy,
        identity=identity,
        input_glb=leaf_artifacts["pixal_output"],
    )
    if anchor_artifact != leaf_artifacts["anchor_authority"]:
        _fail("indexed static job changed its planned anchor authority")
    stage_records = _require_mapping(
        job_receipt.get("stage_receipts"),
        owner="indexed static admission stage receipt closure",
    )
    if set(stage_records) != set(_STAGE_NAMES):
        _fail("indexed static admission stage receipt closure is incomplete")
    stages: dict[str, Mapping[str, Any]] = {}
    command_manifests: dict[str, Mapping[str, Any]] = {}
    for stage in _STAGE_NAMES:
        payload, artifact = _json_file_record(
            stage_records[stage],
            policy,
            owner=f"indexed {stage} stage receipt",
            base=admission_root,
            require_within=admission_root,
        )
        command_manifests[stage] = _validate_stage_receipt(
            payload,
            artifact,
            policy,
            identity=identity,
            stage=stage,
            admission_root=admission_root,
        )
        stages[stage] = payload

    watertight = stages["watertight"]
    if (
        set(watertight["inputs"])
        != {"decision", "pixal_output", "watertight_parameters"}
        or set(watertight["outputs"])
        != {"watertight_glb", "watertight_manifest"}
    ):
        _fail("indexed watertight stage I/O fields are invalid")
    _file_record(
        watertight["inputs"]["decision"],
        policy,
        owner="indexed watertight decision",
        expected=leaf_artifacts["decision"],
    )
    _file_record(
        watertight["inputs"]["pixal_output"],
        policy,
        owner="indexed watertight Pixal output",
        expected=leaf_artifacts["pixal_output"],
    )
    parameters = _validate_watertight_parameters(
        watertight["inputs"]["watertight_parameters"]
    )
    if parameters != plan_parameters:
        _fail("indexed watertight parameters differ from admission plan")
    _file_record(
        watertight["outputs"]["watertight_glb"],
        policy,
        owner="indexed watertight GLB",
        expected=leaf_artifacts["watertight_glb"],
    )
    watertight_manifest, _ = _json_file_record(
        watertight["outputs"]["watertight_manifest"],
        policy,
        owner="indexed watertight manifest",
    )
    _file_record(
        watertight_manifest.get("input"),
        policy,
        owner="indexed watertight frozen source",
        require_within=admission_root,
    )
    watertight_readback = _validate_watertight_manifest(
        watertight_manifest,
        policy,
        pixal_glb=leaf_artifacts["pixal_output"],
        watertight_glb=leaf_artifacts["watertight_glb"],
        parameters=parameters,
        admission_root=admission_root,
    )
    watertight_arguments = [
        "--source",
        watertight_manifest["input"]["path"],
        "--output",
        watertight["outputs"]["watertight_glb"]["path"],
        "--manifest",
        watertight["outputs"]["watertight_manifest"]["path"],
        "--voxel-resolution",
        str(parameters["voxel_resolution"]),
        "--target-faces",
        str(parameters["target_faces"]),
        "--smooth-iterations",
        str(parameters["smooth_iterations"]),
        "--shrinkwrap-strength",
        str(parameters["shrinkwrap_strength"]),
        "--post-shrinkwrap-smooth-iterations",
        str(parameters["post_shrinkwrap_smooth_iterations"]),
        "--torso-fold-repair-iterations",
        str(parameters["torso_fold_repair_iterations"]),
        "--attribute-transfer-backend",
        str(parameters["attribute_transfer_backend"]),
        "--bake-resolution",
        str(parameters["bake_resolution"]),
        "--base-color-encoding-policy",
        str(parameters["base_color_encoding_policy"]),
        "--base-color-gain",
        *(str(item) for item in parameters["base_color_gain"]),
    ]
    if parameters["double_sided"]:
        watertight_arguments.append("--double-sided")
    _validate_exact_stage_command(
        watertight,
        command_manifests["watertight"],
        stage="watertight",
        arguments=watertight_arguments,
    )

    finalization = stages["finalization"]
    if (
        set(finalization["inputs"])
        != {
            "watertight_glb",
            "watertight_manifest",
            "heading_authority",
            "bound_heading_evidence",
        }
        or set(finalization["outputs"])
        != {"finalized_glb", "finalization_manifest"}
    ):
        _fail("indexed finalization stage I/O fields are invalid")
    for stage_field, leaf_field in (
        ("watertight_glb", "watertight_glb"),
        ("heading_authority", "heading_authority"),
        ("bound_heading_evidence", "bound_heading_evidence"),
    ):
        _file_record(
            finalization["inputs"][stage_field],
            policy,
            owner=f"indexed finalization {stage_field}",
            expected=leaf_artifacts[leaf_field],
        )
    _file_record(
        finalization["inputs"]["watertight_manifest"],
        policy,
        owner="indexed finalization watertight manifest",
    )
    _file_record(
        finalization["outputs"]["finalized_glb"],
        policy,
        owner="indexed finalization GLB",
        expected=leaf_artifacts["finalized_glb"],
    )
    finalization_manifest, finalization_artifact = _json_file_record(
        finalization["outputs"]["finalization_manifest"],
        policy,
        owner="indexed finalization manifest",
    )
    bound_heading, _ = _json_artifact(
        policy,
        leaf_artifacts["bound_heading_evidence"].path,
        owner="indexed bound heading evidence",
        expected_sha256=leaf_artifacts["bound_heading_evidence"].sha256,
    )
    bound_heading_review = _validate_heading_authority(
        bound_heading,
        policy,
        identity=identity,
        input_glb=leaf_artifacts["watertight_glb"],
        owner="indexed bound heading evidence",
        review_require_within=admission_root,
    )
    if {
        key: item
        for key, item in bound_heading.items()
        if key not in {"input_glb_sha256", "review_artifact"}
    } != {
        key: item
        for key, item in heading_authority.items()
        if key not in {"input_glb_sha256", "review_artifact"}
    }:
        _fail("indexed bound heading changed reviewed semantic authority")
    _require_same_file_content(
        bound_heading_review,
        heading_review,
        owner="indexed bound heading review evidence",
    )
    final_from_manifest, _, finalized_readback = _validate_finalization(
        finalization_manifest,
        finalization_artifact,
        policy,
        identity=identity,
        target_height_m=target_height_m,
        tolerance_m=tolerance_m,
        expected_watertight=leaf_artifacts["watertight_glb"],
        expected_watertight_readback=watertight_readback,
        expected_heading=leaf_artifacts["bound_heading_evidence"],
    )
    if final_from_manifest != leaf_artifacts["finalized_glb"]:
        _fail("indexed finalization manifest changed its stage output")
    _validate_exact_stage_command(
        finalization,
        command_manifests["finalization"],
        stage="finalization",
        arguments=[
            "--input-glb",
            finalization["inputs"]["watertight_glb"]["path"],
            "--watertight-manifest",
            finalization["inputs"]["watertight_manifest"]["path"],
            "--static-decision",
            str(
                (
                    admission_root
                    / ".runtime_inputs"
                    / str(identity[0])
                    / "decision.json"
                ).resolve()
            ),
            "--heading-evidence",
            finalization["inputs"]["bound_heading_evidence"]["path"],
            "--output",
            finalization["outputs"]["finalized_glb"]["path"],
            "--manifest",
            finalization["outputs"]["finalization_manifest"]["path"],
        ],
    )

    emitter = stages["emitter_measurement"]
    if (
        set(emitter["inputs"])
        != {
            "finalized_glb",
            "finalization_manifest",
            "anchor_authority",
            "bound_anchor_spec",
        }
        or set(emitter["outputs"])
        != {"emitter_measurement", "marker_glb"}
    ):
        _fail("indexed emitter stage I/O fields are invalid")
    for stage_field, leaf_field in (
        ("finalized_glb", "finalized_glb"),
        ("anchor_authority", "anchor_authority"),
        ("bound_anchor_spec", "bound_anchor_spec"),
    ):
        _file_record(
            emitter["inputs"][stage_field],
            policy,
            owner=f"indexed emitter {stage_field}",
            expected=leaf_artifacts[leaf_field],
        )
    _file_record(
        emitter["inputs"]["finalization_manifest"],
        policy,
        owner="indexed emitter finalization manifest",
        expected=finalization_artifact,
    )
    measurement, measurement_artifact = _json_file_record(
        emitter["outputs"]["emitter_measurement"],
        policy,
        owner="indexed emitter measurement",
        expected=leaf_artifacts["emitter_measurement"],
    )
    _file_record(
        emitter["outputs"]["marker_glb"],
        policy,
        owner="indexed emitter marker GLB",
        expected=leaf_artifacts["emitter_marker_glb"],
    )
    _validate_exact_stage_command(
        emitter,
        command_manifests["emitter_measurement"],
        stage="emitter_measurement",
        arguments=[
            "--input-glb",
            emitter["inputs"]["finalized_glb"]["path"],
            "--finalization-manifest",
            emitter["inputs"]["finalization_manifest"]["path"],
            "--anchor-spec",
            emitter["inputs"]["bound_anchor_spec"]["path"],
            "--output",
            emitter["outputs"]["emitter_measurement"]["path"],
            "--marker-glb",
            emitter["outputs"]["marker_glb"]["path"],
        ],
    )
    _, _, _, measured_marker = _validate_measurement(
        measurement,
        measurement_artifact,
        policy,
        identity=identity,
        finalized_glb=leaf_artifacts["finalized_glb"],
        finalized_glb_readback=finalized_readback,
        finalization=finalization_artifact,
        anchor_authority=anchor_authority,
        anchor_authority_review=anchor_review,
        expected_anchor_spec=leaf_artifacts["bound_anchor_spec"],
        expected_marker_glb=leaf_artifacts["emitter_marker_glb"],
        admission_root=admission_root,
    )
    if measured_marker != leaf_artifacts["emitter_marker_glb"]:
        _fail("indexed emitter measurement changed its marker output")
    return job_receipt, job_artifact


def _validate_admission_receipts(
    admission_batch: Mapping[str, Any],
    admission_batch_artifact: AuthenticatedArtifact,
    policy: WorkspacePathPolicy,
    *,
    instance_id: str,
    marker_visual_approval_path: str | Path,
    expected_evidence_sha256: Mapping[str, str],
) -> ValidatedStaticObjectAdmission:
    expected_fields = {
        "schema",
        "status",
        "state_classification",
        "formal_dataset_registration_authorized",
        "asset_class",
        "route",
        "decision_batch",
        "plan",
        "job_count",
        "passed_count",
        "failed_count",
        "jobs",
        "marker_review",
        "automatic_checks",
        "batch_sha256",
    }
    jobs = admission_batch.get("jobs")
    if (
        set(admission_batch) != expected_fields
        or admission_batch.get("schema") != STATIC_ADMISSION_BATCH_SCHEMA
        or admission_batch.get("status")
        != "passed_all_instances_pending_emitter_marker_review"
        or admission_batch.get("state_classification") != "research_candidate"
        or admission_batch.get("formal_dataset_registration_authorized")
        is not False
        or admission_batch.get("asset_class") != "static_object"
        or admission_batch.get("route") != STATIC_ROUTE
        or admission_batch.get("marker_review")
        != {
            "status": "pending",
            "next_gate": "emitter_marker_visual_review",
        }
        or not isinstance(jobs, list)
        or admission_batch.get("automatic_checks", {}).get("overall")
        != "passed"
    ):
        _fail("static admission batch contract is invalid")
    _require_content_hash(
        admission_batch,
        field="batch_sha256",
        owner="static admission batch",
    )
    count = _require_nonnegative_int(
        admission_batch.get("job_count"), owner="static admission batch.job_count"
    )
    passed = _require_nonnegative_int(
        admission_batch.get("passed_count"),
        owner="static admission batch.passed_count",
    )
    failed = _require_nonnegative_int(
        admission_batch.get("failed_count"),
        owner="static admission batch.failed_count",
    )
    if count != len(jobs) or passed != count or failed != 0:
        _fail("static admission batch counts are not an all-pass closure")
    admission_root = admission_batch_artifact.path.parent

    decision_batch, decision_batch_artifact = _json_file_record(
        admission_batch.get("decision_batch"),
        policy,
        owner="static admission decision batch",
        extra_fields=frozenset({"decision_batch_sha256"}),
    )
    decision_batch_hash = _require_content_hash(
        decision_batch,
        field="decision_batch_sha256",
        owner="static admission decision batch",
    )
    if (
        admission_batch["decision_batch"].get("decision_batch_sha256")
        != decision_batch_hash
    ):
        _fail("admission/decision batch inner hash changed")
    selected = _validate_decision_batch(
        decision_batch,
        decision_batch_artifact,
        policy,
        instance_id=instance_id,
    )
    plan, plan_artifact = _json_file_record(
        admission_batch.get("plan"),
        policy,
        owner="static admission plan",
        extra_fields=frozenset({"plan_sha256"}),
    )
    plan_hash = _require_content_hash(
        plan, field="plan_sha256", owner="static admission plan"
    )
    if admission_batch["plan"].get("plan_sha256") != plan_hash:
        _fail("admission/plan inner hash changed")
    plan_closure = _validate_admission_plan(
        plan,
        plan_artifact,
        policy,
        decision_batch_sha256=decision_batch_hash,
        approved_ids=selected["approved_ids"],
        selected=selected,
    )

    job_index_fields = {
        "instance_id",
        "request_sha256",
        "profile_sha256",
        "status",
        "receipt_sha256",
        "job_receipt",
        "marker_review",
    }
    batch_job_ids: list[str] = []
    indexed_job_closures: dict[
        str,
        tuple[Mapping[str, Any], AuthenticatedArtifact],
    ] = {}
    for raw_job_index in jobs:
        batch_job_index = _require_mapping(
            raw_job_index, owner="static admission batch job index"
        )
        if (
            set(batch_job_index) != job_index_fields
            or batch_job_index.get("status")
            != "passed_pending_emitter_marker_review"
            or batch_job_index.get("marker_review") != "pending"
        ):
            _fail("static admission batch job index contract is invalid")
        batch_job_id = _require_stable_id(
            batch_job_index.get("instance_id"),
            owner="static admission batch job instance_id",
        )
        batch_job_ids.append(batch_job_id)
        for field in ("request_sha256", "profile_sha256", "receipt_sha256"):
            _require_sha256(
                batch_job_index.get(field),
                owner=f"static admission batch job {field}",
            )
        expected_identity = selected["approved_identities"].get(batch_job_id)
        if expected_identity is None or (
            batch_job_index.get("instance_id"),
            batch_job_index.get("request_sha256"),
            batch_job_index.get("profile_sha256"),
        ) != expected_identity:
            _fail("static admission batch job differs from approved decision")
        indexed_job_closures[batch_job_id] = (
            _validate_indexed_admission_job_closure(
                batch_job_index,
                policy,
                admission_root=admission_root,
                decision_index=selected["approved_indexes"][batch_job_id],
                decision_batch_artifact=decision_batch_artifact,
                plan_item=plan_closure["all_jobs"][batch_job_id],
                plan_artifact=plan_closure["plan_artifact"],
            )
        )
    if len(batch_job_ids) != len(set(batch_job_ids)):
        _fail("static admission batch repeats a job instance")
    if set(batch_job_ids) != selected["approved_ids"]:
        _fail("static admission batch jobs do not cover every approved instance")
    job_indexes = [
        item
        for item in jobs
        if isinstance(item, Mapping) and item.get("instance_id") == instance_id
    ]
    if len(job_indexes) != 1:
        _fail("static admission batch lacks exactly one selected job")
    job_index = job_indexes[0]
    if set(job_index) != job_index_fields:
        _fail("static admission selected job index fields are invalid")
    job_receipt, job_receipt_artifact = indexed_job_closures[instance_id]
    job_receipt_hash = _require_sha256(
        job_index.get("receipt_sha256"),
        owner="static admission job index.receipt_sha256",
    )
    if job_receipt.get("receipt_sha256") != job_receipt_hash:
        _fail("static admission job index/receipt hash differs")
    identity = selected["identity"]
    if (
        job_index.get("request_sha256") != identity[1]
        or job_index.get("profile_sha256") != identity[2]
        or job_index.get("status")
        != "passed_pending_emitter_marker_review"
        or job_index.get("marker_review") != "pending"
    ):
        _fail("static admission job index identity/status changed")
    job_fields = {
        "schema",
        "status",
        "asset_class",
        "route",
        "instance_id",
        "request_sha256",
        "profile_sha256",
        "decision",
        "pixal_output",
        "heading_authority",
        "anchor_authority",
        "bound_heading_evidence",
        "bound_anchor_spec",
        "watertight_glb",
        "finalized_glb",
        "emitter_measurement",
        "emitter_marker_glb",
        "stage_receipts",
        "path_rebinding",
        "marker_review",
        "next_gate",
        "formal_dataset_registration_authorized",
        "receipt_sha256",
    }
    if (
        set(job_receipt) != job_fields
        or job_receipt.get("schema") != STATIC_ADMISSION_JOB_RECEIPT_SCHEMA
        or job_receipt.get("status")
        != "passed_pending_emitter_marker_review"
        or job_receipt.get("asset_class") != "static_object"
        or job_receipt.get("route") != STATIC_ROUTE
        or job_receipt.get("marker_review") != "pending"
        or job_receipt.get("next_gate") != "emitter_marker_visual_review"
        or job_receipt.get("formal_dataset_registration_authorized") is not False
    ):
        _fail("static admission job receipt contract is invalid")
    _require_content_hash(
        job_receipt,
        field="receipt_sha256",
        owner="static admission job receipt",
    )
    _require_identity(
        job_receipt, owner="static admission job receipt", expected=identity
    )
    rebinding = _require_mapping(
        job_receipt.get("path_rebinding"),
        owner="static admission path rebinding",
    )
    if (
        set(rebinding)
        != {
            "policy",
            "staging_root",
            "public_root",
            "hashes_before",
            "hashes_after",
            "semantic_authority_fields_changed",
        }
        or rebinding.get("policy")
        != "staging_to_atomic_public_root_paths_only_v1"
        or rebinding.get("semantic_authority_fields_changed") is not False
        or not isinstance(rebinding.get("staging_root"), str)
        or not rebinding["staging_root"]
        or not isinstance(rebinding.get("public_root"), str)
        or not rebinding["public_root"]
    ):
        _fail("static admission path-rebinding contract is invalid")
    try:
        public_root = Path(rebinding["public_root"]).expanduser().resolve(
            strict=True
        )
    except (OSError, RuntimeError) as error:
        raise StaticObjectRegistrationError(
            "static admission public root is invalid"
        ) from error
    if public_root != admission_root:
        _fail("static admission path rebinding names a different public root")
    rebinding_fields = {
        "watertight_manifest",
        "finalization_manifest",
        "bound_anchor",
        "emitter_measurement",
    }
    for phase in ("hashes_before", "hashes_after"):
        hashes = _require_mapping(
            rebinding.get(phase),
            owner=f"static admission path rebinding {phase}",
        )
        if set(hashes) != rebinding_fields:
            _fail(f"static admission path rebinding {phase} is incomplete")
        for kind, digest in hashes.items():
            _require_sha256(
                digest,
                owner=f"static admission path rebinding {phase}.{kind}",
            )
    decision_artifact = _file_record(
        job_receipt.get("decision"),
        policy,
        owner="job receipt static decision",
        expected=selected["decision_artifact"],
    )
    pixal_glb = _file_record(
        job_receipt.get("pixal_output"),
        policy,
        owner="job receipt Pixal GLB",
        expected=selected["pixal_glb"],
    )
    heading_artifact = _file_record(
        job_receipt.get("heading_authority"),
        policy,
        owner="job receipt heading authority",
        expected=plan_closure["heading_artifact"],
    )
    anchor_authority_artifact = _file_record(
        job_receipt.get("anchor_authority"),
        policy,
        owner="job receipt anchor authority",
        expected=plan_closure["anchor_artifact"],
    )
    bound_heading, bound_heading_artifact = _json_file_record(
        job_receipt.get("bound_heading_evidence"),
        policy,
        owner="job receipt bound heading evidence",
    )
    bound_anchor, bound_anchor_artifact = _json_file_record(
        job_receipt.get("bound_anchor_spec"),
        policy,
        owner="job receipt bound anchor spec",
    )
    watertight_glb = _file_record(
        job_receipt.get("watertight_glb"),
        policy,
        owner="job receipt watertight GLB",
    )
    finalized_glb = _file_record(
        job_receipt.get("finalized_glb"),
        policy,
        owner="job receipt finalized GLB",
    )
    measurement_artifact = _file_record(
        job_receipt.get("emitter_measurement"),
        policy,
        owner="job receipt emitter measurement",
    )
    marker_glb = _file_record(
        job_receipt.get("emitter_marker_glb"),
        policy,
        owner="job receipt emitter marker GLB",
    )

    stage_records = _require_mapping(
        job_receipt.get("stage_receipts"),
        owner="static admission stage receipt index",
    )
    if set(stage_records) != set(_STAGE_NAMES):
        _fail("static admission stage receipt closure is incomplete")
    stage_values: dict[str, Mapping[str, Any]] = {}
    stage_artifacts: dict[str, AuthenticatedArtifact] = {}
    stage_command_manifests: dict[str, Mapping[str, Any]] = {}
    for stage in _STAGE_NAMES:
        payload, receipt_artifact = _json_file_record(
            stage_records[stage],
            policy,
            owner=f"{stage} stage receipt",
            base=admission_root,
            require_within=admission_root,
        )
        stage_command_manifests[stage] = _validate_stage_receipt(
            payload,
            receipt_artifact,
            policy,
            identity=identity,
            stage=stage,
            admission_root=admission_root,
        )
        stage_values[stage] = payload
        stage_artifacts[stage] = receipt_artifact

    watertight_stage = stage_values["watertight"]
    if set(watertight_stage["inputs"]) != {
        "decision",
        "pixal_output",
        "watertight_parameters",
    } or set(watertight_stage["outputs"]) != {
        "watertight_glb",
        "watertight_manifest",
    }:
        _fail("watertight stage receipt I/O fields are invalid")
    _file_record(
        watertight_stage["inputs"]["decision"],
        policy,
        owner="watertight stage decision input",
        expected=decision_artifact,
    )
    _file_record(
        watertight_stage["inputs"]["pixal_output"],
        policy,
        owner="watertight stage Pixal input",
        expected=pixal_glb,
    )
    if (
        _validate_watertight_parameters(
            watertight_stage["inputs"]["watertight_parameters"]
        )
        != plan_closure["parameters"]
    ):
        _fail("watertight stage parameters differ from admission plan")
    _file_record(
        watertight_stage["outputs"]["watertight_glb"],
        policy,
        owner="watertight stage GLB output",
        expected=watertight_glb,
    )
    watertight_manifest, watertight_manifest_artifact = _json_file_record(
        watertight_stage["outputs"]["watertight_manifest"],
        policy,
        owner="watertight stage manifest",
    )
    watertight_readback = _validate_watertight_manifest(
        watertight_manifest,
        policy,
        pixal_glb=pixal_glb,
        watertight_glb=watertight_glb,
        parameters=plan_closure["parameters"],
        admission_root=admission_root,
    )
    watertight_parameters = plan_closure["parameters"]
    watertight_arguments = [
        "--source",
        watertight_manifest["input"]["path"],
        "--output",
        watertight_stage["outputs"]["watertight_glb"]["path"],
        "--manifest",
        watertight_stage["outputs"]["watertight_manifest"]["path"],
        "--voxel-resolution",
        str(watertight_parameters["voxel_resolution"]),
        "--target-faces",
        str(watertight_parameters["target_faces"]),
        "--smooth-iterations",
        str(watertight_parameters["smooth_iterations"]),
        "--shrinkwrap-strength",
        str(watertight_parameters["shrinkwrap_strength"]),
        "--post-shrinkwrap-smooth-iterations",
        str(watertight_parameters["post_shrinkwrap_smooth_iterations"]),
        "--torso-fold-repair-iterations",
        str(watertight_parameters["torso_fold_repair_iterations"]),
        "--attribute-transfer-backend",
        str(watertight_parameters["attribute_transfer_backend"]),
        "--bake-resolution",
        str(watertight_parameters["bake_resolution"]),
        "--base-color-encoding-policy",
        str(watertight_parameters["base_color_encoding_policy"]),
        "--base-color-gain",
        *(
            str(item)
            for item in watertight_parameters["base_color_gain"]
        ),
    ]
    if watertight_parameters["double_sided"]:
        watertight_arguments.append("--double-sided")
    _validate_exact_stage_command(
        watertight_stage,
        stage_command_manifests["watertight"],
        stage="watertight",
        arguments=watertight_arguments,
    )

    finalization_stage = stage_values["finalization"]
    if set(finalization_stage["inputs"]) != {
        "watertight_glb",
        "watertight_manifest",
        "heading_authority",
        "bound_heading_evidence",
    } or set(finalization_stage["outputs"]) != {
        "finalized_glb",
        "finalization_manifest",
    }:
        _fail("finalization stage receipt I/O fields are invalid")
    _file_record(
        finalization_stage["inputs"]["watertight_glb"],
        policy,
        owner="finalization stage watertight GLB",
        expected=watertight_glb,
    )
    _file_record(
        finalization_stage["inputs"]["watertight_manifest"],
        policy,
        owner="finalization stage watertight manifest",
        expected=watertight_manifest_artifact,
    )
    _file_record(
        finalization_stage["inputs"]["heading_authority"],
        policy,
        owner="finalization stage heading authority",
        expected=heading_artifact,
    )
    _file_record(
        finalization_stage["inputs"]["bound_heading_evidence"],
        policy,
        owner="finalization stage bound heading",
        expected=bound_heading_artifact,
    )
    _file_record(
        finalization_stage["outputs"]["finalized_glb"],
        policy,
        owner="finalization stage GLB output",
        expected=finalized_glb,
    )
    bound_heading_review = _validate_heading_authority(
        bound_heading,
        policy,
        identity=identity,
        input_glb=watertight_glb,
        owner="bound static heading evidence",
        review_require_within=admission_root,
    )
    if {
        key: item
        for key, item in bound_heading.items()
        if key not in {"input_glb_sha256", "review_artifact"}
    } != {
        key: item
        for key, item in plan_closure["heading"].items()
        if key not in {"input_glb_sha256", "review_artifact"}
    }:
        _fail("bound heading changed reviewed semantic authority")
    _require_same_file_content(
        bound_heading_review,
        plan_closure["heading_review_artifact"],
        owner="bound static heading review evidence",
    )
    finalization, finalization_artifact = _json_file_record(
        finalization_stage["outputs"]["finalization_manifest"],
        policy,
        owner="finalization stage manifest",
    )
    (
        final_glb_from_manifest,
        readback_height_m,
        finalized_glb_readback,
    ) = _validate_finalization(
        finalization,
        finalization_artifact,
        policy,
        identity=identity,
        target_height_m=selected["target_height_m"],
        tolerance_m=selected["tolerance_m"],
        expected_watertight=watertight_glb,
        expected_watertight_readback=watertight_readback,
        expected_heading=bound_heading_artifact,
    )
    _validate_exact_stage_command(
        finalization_stage,
        stage_command_manifests["finalization"],
        stage="finalization",
        arguments=[
            "--input-glb",
            finalization_stage["inputs"]["watertight_glb"]["path"],
            "--watertight-manifest",
            finalization_stage["inputs"]["watertight_manifest"]["path"],
            "--static-decision",
            str(
                (
                    admission_root
                    / ".runtime_inputs"
                    / identity[0]
                    / "decision.json"
                ).resolve()
            ),
            "--heading-evidence",
            finalization_stage["inputs"]["bound_heading_evidence"]["path"],
            "--output",
            finalization_stage["outputs"]["finalized_glb"]["path"],
            "--manifest",
            finalization_stage["outputs"]["finalization_manifest"]["path"],
        ],
    )
    finalized_heading = _require_mapping(
        finalization.get("heading"),
        owner="static finalization heading derivation",
    )
    reviewed_yaw = _finite_number(
        bound_heading.get("reviewed_source_front_yaw_deg"),
        owner="bound static heading reviewed yaw",
    )
    if (
        not math.isclose(
            _finite_number(
                finalized_heading.get("reviewed_source_front_yaw_deg"),
                owner="finalized reviewed source yaw",
            ),
            reviewed_yaw,
            abs_tol=1.0e-9,
        )
        or not math.isclose(
            _finite_number(
                finalized_heading.get("applied_world_z_yaw_deg"),
                owner="finalized applied yaw",
            ),
            -reviewed_yaw,
            abs_tol=1.0e-9,
        )
    ):
        _fail("static finalization heading differs from bound review authority")
    if final_glb_from_manifest != finalized_glb:
        _fail("finalization manifest output differs from stage receipt")

    emitter_stage = stage_values["emitter_measurement"]
    if set(emitter_stage["inputs"]) != {
        "finalized_glb",
        "finalization_manifest",
        "anchor_authority",
        "bound_anchor_spec",
    } or set(emitter_stage["outputs"]) != {
        "emitter_measurement",
        "marker_glb",
    }:
        _fail("emitter stage receipt I/O fields are invalid")
    _file_record(
        emitter_stage["inputs"]["finalized_glb"],
        policy,
        owner="emitter stage finalized GLB",
        expected=finalized_glb,
    )
    _file_record(
        emitter_stage["inputs"]["finalization_manifest"],
        policy,
        owner="emitter stage finalization manifest",
        expected=finalization_artifact,
    )
    _file_record(
        emitter_stage["inputs"]["anchor_authority"],
        policy,
        owner="emitter stage anchor authority",
        expected=anchor_authority_artifact,
    )
    _file_record(
        emitter_stage["inputs"]["bound_anchor_spec"],
        policy,
        owner="emitter stage bound anchor spec",
        expected=bound_anchor_artifact,
    )
    _file_record(
        emitter_stage["outputs"]["emitter_measurement"],
        policy,
        owner="emitter stage measurement output",
        expected=measurement_artifact,
    )
    _file_record(
        emitter_stage["outputs"]["marker_glb"],
        policy,
        owner="emitter stage marker output",
        expected=marker_glb,
    )
    _validate_exact_stage_command(
        emitter_stage,
        stage_command_manifests["emitter_measurement"],
        stage="emitter_measurement",
        arguments=[
            "--input-glb",
            emitter_stage["inputs"]["finalized_glb"]["path"],
            "--finalization-manifest",
            emitter_stage["inputs"]["finalization_manifest"]["path"],
            "--anchor-spec",
            emitter_stage["inputs"]["bound_anchor_spec"]["path"],
            "--output",
            emitter_stage["outputs"]["emitter_measurement"]["path"],
            "--marker-glb",
            emitter_stage["outputs"]["marker_glb"]["path"],
        ],
    )
    measurement, confirmed_measurement = _json_artifact(
        policy,
        measurement_artifact.path,
        owner="SPEAR static emitter measurement",
        expected_sha256=measurement_artifact.sha256,
    )
    anchor_id, semantic_role, emitter_offset, confirmed_marker = (
        _validate_measurement(
            measurement,
            confirmed_measurement,
            policy,
            identity=identity,
            finalized_glb=finalized_glb,
            finalized_glb_readback=finalized_glb_readback,
            finalization=finalization_artifact,
            anchor_authority=plan_closure["anchor"],
            anchor_authority_review=plan_closure[
                "anchor_review_artifact"
            ],
            expected_anchor_spec=bound_anchor_artifact,
            expected_marker_glb=marker_glb,
            admission_root=admission_root,
        )
    )
    if confirmed_marker != marker_glb:
        _fail("emitter measurement marker differs from stage receipt")
    expected_rebased_hashes = {
        "watertight_manifest": watertight_manifest_artifact.sha256,
        "finalization_manifest": finalization_artifact.sha256,
        "bound_anchor": bound_anchor_artifact.sha256,
        "emitter_measurement": measurement_artifact.sha256,
    }
    if rebinding["hashes_after"] != expected_rebased_hashes:
        _fail("static admission post-rebinding hashes changed")

    marker_approval, marker_approval_artifact = _json_artifact(
        policy,
        marker_visual_approval_path,
        owner="static marker visual approval",
        expected_sha256=expected_evidence_sha256.get("marker_visual_approval"),
    )
    _validate_marker_approval(
        marker_approval,
        policy,
        identity=identity,
        finalized_glb=finalized_glb,
        measurement=measurement_artifact,
        marker_glb=marker_glb,
    )

    evidence_artifacts = {
        "emitter_marker_glb": marker_glb,
        "marker_visual_approval": marker_approval_artifact,
        "spear_static_admission_batch": admission_batch_artifact,
        "spear_static_admission_job_receipt": job_receipt_artifact,
        "spear_static_emitter_stage_receipt": stage_artifacts[
            "emitter_measurement"
        ],
        "spear_static_finalization_stage_receipt": stage_artifacts[
            "finalization"
        ],
        "spear_static_watertight_stage_receipt": stage_artifacts["watertight"],
        "visual_asset_glb": finalized_glb,
    }
    unknown_expected = set(expected_evidence_sha256) - set(evidence_artifacts)
    if unknown_expected:
        _fail(
            "unknown expected static evidence kinds: "
            + ", ".join(sorted(unknown_expected))
        )
    for kind, expected_hash in expected_evidence_sha256.items():
        _require_sha256(expected_hash, owner=f"expected {kind} hash")
        if evidence_artifacts[kind].sha256 != expected_hash:
            _fail(f"{kind} does not match published admission evidence")

    return ValidatedStaticObjectAdmission(
        instance_id=identity[0],
        request_sha256=identity[1],
        profile_sha256=identity[2],
        decision=selected["decision"],
        finalization=finalization,
        measurement=measurement,
        marker_approval=marker_approval,
        admission_batch_artifact=admission_batch_artifact,
        job_receipt_artifact=job_receipt_artifact,
        watertight_stage_receipt_artifact=stage_artifacts["watertight"],
        finalization_stage_receipt_artifact=stage_artifacts["finalization"],
        emitter_stage_receipt_artifact=stage_artifacts["emitter_measurement"],
        decision_artifact=selected["decision_artifact"],
        review_artifact=selected["review_artifact"],
        pixal_glb=pixal_glb,
        watertight_glb=watertight_glb,
        watertight_manifest_artifact=watertight_manifest_artifact,
        finalization_artifact=finalization_artifact,
        measurement_artifact=measurement_artifact,
        marker_approval_artifact=marker_approval_artifact,
        finalized_glb=finalized_glb,
        marker_glb=marker_glb,
        anchor_id=anchor_id,
        semantic_role=semantic_role,
        emitter_offset_m=emitter_offset,
        target_height_m=selected["target_height_m"],
        readback_height_m=readback_height_m,
    )


def validate_static_object_admission(
    *,
    admission_batch_path: str | Path,
    instance_id: str,
    marker_visual_approval_path: str | Path,
    workspace_roots: Iterable[str | Path],
    expected_evidence_sha256: Mapping[str, str] | None = None,
) -> ValidatedStaticObjectAdmission:
    """Authenticate one selected job from a sealed SPEAR admission batch."""

    selected_instance_id = _require_stable_id(instance_id, owner="instance_id")
    try:
        policy = WorkspacePathPolicy.from_roots(workspace_roots)
    except (OSError, PathPolicyError) as error:
        raise StaticObjectRegistrationError(str(error)) from error
    expected = dict(expected_evidence_sha256 or {})

    admission_batch, admission_batch_artifact = _json_artifact(
        policy,
        admission_batch_path,
        owner="SPEAR static admission batch",
        expected_sha256=expected.get("spear_static_admission_batch"),
    )
    return _validate_admission_receipts(
        admission_batch,
        admission_batch_artifact,
        policy,
        instance_id=selected_instance_id,
        marker_visual_approval_path=marker_visual_approval_path,
        expected_evidence_sha256=expected,
    )


def _entity_record(
    admission: ValidatedStaticObjectAdmission,
) -> dict[str, Any]:
    evidence = admission.admission_evidence()
    return {
        "entity_asset_id": admission.instance_id,
        "revision": admission.entity_revision,
        "entity_class": "rigid_object",
        "visual_asset": {
            "uri": str(admission.finalized_glb.path),
            "sha256": admission.finalized_glb.sha256,
        },
        "realized_visual_attributes": {
            "asset_class": "static_object",
            "source_instance_id": admission.instance_id,
            "target_height_m": admission.target_height_m,
            "readback_height_m": admission.readback_height_m,
            "coordinate_system_id": CANONICAL_COORDINATE_SYSTEM["id"],
            "emitter_semantic_role": admission.semantic_role,
            "emitter_offset_space": "final_scaled_asset_root",
            "formal_dataset_registration_authorized": False,
        },
        "capabilities": {
            "articulated": False,
            "skeleton_revision": None,
            "skeleton_sha256": None,
            "action_ids": [],
        },
        "emitter_anchors": [
            {
                "anchor_id": admission.anchor_id,
                "anchor_type": "object_speaker",
                "joint_id": None,
                "local_position_m": list(admission.emitter_offset_m),
            }
        ],
        "admission_evidence": evidence,
        "provenance": {
            "source": "SPEAR generated static-object research admission",
            "source_revision": admission.decision["decision_sha256"],
            "license": None,
            "rights_status": "review_required",
            "evidence_sha256": evidence["evidence_content_sha256"],
        },
        "admission_state": "research",
    }


def _validate_static_entity(
    entity: Mapping[str, Any],
    *,
    admission: ValidatedStaticObjectAdmission | None = None,
) -> None:
    capabilities = _require_mapping(
        entity.get("capabilities"), owner="registered static capabilities"
    )
    if (
        entity.get("entity_class") != "rigid_object"
        or entity.get("admission_state") != "research"
        or capabilities.get("articulated") is not False
        or capabilities.get("skeleton_revision") is not None
        or capabilities.get("skeleton_sha256") is not None
        or capabilities.get("action_ids") != []
    ):
        _fail("registered static object must be non-articulated research data")
    anchors = entity.get("emitter_anchors")
    if not isinstance(anchors, list) or len(anchors) != 1:
        _fail("registered static object must have exactly one emitter anchor")
    anchor = _require_mapping(anchors[0], owner="registered static emitter anchor")
    if (
        anchor.get("anchor_type") != "object_speaker"
        or anchor.get("joint_id") is not None
    ):
        _fail("registered static emitter must be a joint-free object_speaker")
    evidence = _require_mapping(
        entity.get("admission_evidence"),
        owner="registered static admission evidence",
    )
    if evidence.get("formal_dataset_registration_authorized") is not False:
        _fail("static registration cannot claim formal dataset authorization")
    attributes = _require_mapping(
        entity.get("realized_visual_attributes"),
        owner="registered static realized attributes",
    )
    if attributes.get("formal_dataset_registration_authorized") is not False:
        _fail("static realized attributes cannot claim formal authorization")
    if admission is not None:
        expected = _entity_record(admission)
        if dict(entity) != expected:
            _fail("registered static entity differs from authenticated evidence")


def _load_registry(
    policy: WorkspacePathPolicy,
    path: str | Path,
    *,
    owner: str,
) -> tuple[dict[str, Any], AuthenticatedArtifact]:
    registry, artifact = _json_artifact(policy, path, owner=owner)
    errors = validate_entity_asset_registry(registry)
    if errors:
        _fail(f"{owner} is invalid: " + "; ".join(errors))
    return registry, artifact


def publish_static_object_entity_registry(
    *,
    base_registry_path: str | Path,
    admission_batch_path: str | Path,
    instance_id: str,
    marker_visual_approval_path: str | Path,
    output_path: str | Path,
    registry_revision: str,
    workspace_roots: Iterable[str | Path],
) -> Path:
    """Append one authenticated static object and publish without replacement."""

    _require_stable_id(registry_revision, owner="registry_revision")
    roots = tuple(workspace_roots)
    try:
        policy = WorkspacePathPolicy.from_roots(roots)
    except (OSError, PathPolicyError) as error:
        raise StaticObjectRegistrationError(str(error)) from error
    base, _ = _load_registry(policy, base_registry_path, owner="base entity registry")
    if base["revision"] == registry_revision:
        _fail("registry_revision must advance beyond the base registry revision")
    admission = validate_static_object_admission(
        admission_batch_path=admission_batch_path,
        instance_id=instance_id,
        marker_visual_approval_path=marker_visual_approval_path,
        workspace_roots=roots,
    )
    entity = _entity_record(admission)
    _validate_static_entity(entity, admission=admission)

    records = deepcopy(base["entities"])
    new_key = (entity["entity_asset_id"], entity["revision"])
    if any(
        (item["entity_asset_id"], item["revision"]) == new_key for item in records
    ):
        _fail(
            f"entity registry already contains "
            f"{entity['entity_asset_id']}@{entity['revision']}"
        )
    records.append(entity)
    records.sort(key=lambda item: (item["entity_asset_id"], item["revision"]))
    result = {
        **{key: deepcopy(item) for key, item in base.items() if key != "entities"},
        "revision": registry_revision,
        "entities": records,
    }
    result = bind_content_hash(result)
    errors = validate_entity_asset_registry(result)
    if errors:
        _fail("published entity registry is invalid: " + "; ".join(errors))

    destination: Path | None = None
    descriptor = -1
    temporary: Path | None = None
    try:
        destination = policy.resolve_output(
            output_path,
            owner="evidence file",
            create_parent=True,
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".verify.json",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o644)
        view = memoryview(_json_bytes(result))
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("temporary registry write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1

        # Verify the exact sibling inode before it can become the immutable
        # destination.  A failed verification therefore leaves no final file.
        verify_static_object_entity_registry(
            registry_path=temporary,
            entity_asset_id=admission.instance_id,
            entity_revision=admission.entity_revision,
            workspace_roots=roots,
        )
        confirmed_destination = policy.resolve_output(
            destination,
            owner="evidence file",
        )
        if confirmed_destination != destination:
            _fail("registry destination changed during pre-publication verification")
        os.link(temporary, destination, follow_symlinks=False)
    except (OSError, PathPolicyError) as error:
        if isinstance(error, FileExistsError):
            raise FileExistsError(
                f"refusing to replace immutable evidence: {output_path}"
            ) from error
        raise StaticObjectRegistrationError(str(error)) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    assert destination is not None
    return destination


def verify_static_object_entity_registry(
    *,
    registry_path: str | Path,
    entity_asset_id: str,
    entity_revision: str,
    workspace_roots: Iterable[str | Path],
) -> Mapping[str, Any]:
    """Re-open a published registry and every referenced admission artifact."""

    roots = tuple(workspace_roots)
    try:
        policy = WorkspacePathPolicy.from_roots(roots)
    except (OSError, PathPolicyError) as error:
        raise StaticObjectRegistrationError(str(error)) from error
    registry, _ = _load_registry(
        policy, registry_path, owner="published entity registry"
    )
    try:
        entity = resolve_entity_asset(registry, entity_asset_id, entity_revision)
    except KeyError as error:
        raise StaticObjectRegistrationError(str(error)) from error
    _validate_static_entity(entity)
    evidence = _require_mapping(
        entity.get("admission_evidence"), owner="published admission evidence"
    )
    raw_artifacts = evidence.get("artifacts")
    if not isinstance(raw_artifacts, list):
        _fail("published admission evidence artifacts are missing")
    artifacts = {
        item.get("kind"): item
        for item in raw_artifacts
        if isinstance(item, Mapping)
    }
    if tuple(item.get("kind") for item in raw_artifacts) != STATIC_OBJECT_EVIDENCE_KINDS:
        _fail("published admission evidence closure is not canonical")
    hashes = {
        kind: _require_sha256(
            artifacts[kind].get("sha256"), owner=f"published {kind} hash"
        )
        for kind in STATIC_OBJECT_EVIDENCE_KINDS
    }
    admission = validate_static_object_admission(
        admission_batch_path=artifacts["spear_static_admission_batch"]["path"],
        instance_id=entity_asset_id,
        marker_visual_approval_path=artifacts["marker_visual_approval"]["path"],
        workspace_roots=roots,
        expected_evidence_sha256=hashes,
    )
    _validate_static_entity(entity, admission=admission)
    return entity


def resolve_static_object_emitter_world(
    registry: Mapping[str, Any],
    *,
    entity_asset_id: str,
    entity_revision: str,
    world_from_asset: Any,
    anchor_id: str | None = None,
) -> tuple[float, float, float]:
    """Resolve a joint-free static emitter through one rigid world transform."""

    errors = validate_entity_asset_registry(registry)
    if errors:
        _fail("entity registry is invalid: " + "; ".join(errors))
    try:
        entity = resolve_entity_asset(registry, entity_asset_id, entity_revision)
    except KeyError as error:
        raise StaticObjectRegistrationError(str(error)) from error
    capabilities = _require_mapping(
        entity.get("capabilities"), owner="static emitter capabilities"
    )
    if (
        entity.get("entity_class") != "rigid_object"
        or capabilities.get("articulated") is not False
        or capabilities.get("action_ids") != []
    ):
        _fail("world emitter resolution requires a non-articulated rigid object")
    anchors = [
        item
        for item in entity["emitter_anchors"]
        if item["anchor_type"] == "object_speaker"
        and (anchor_id is None or item["anchor_id"] == anchor_id)
    ]
    if len(anchors) != 1 or anchors[0]["joint_id"] is not None:
        _fail("world emitter resolution requires one joint-free object_speaker")

    if isinstance(world_from_asset, (str, bytes)):
        _fail("world_from_asset must be a finite rigid 4x4 matrix")
    try:
        rows = tuple(tuple(row) for row in world_from_asset)
    except TypeError:
        _fail("world_from_asset must be a finite rigid 4x4 matrix")
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        _fail("world_from_asset must be a finite rigid 4x4 matrix")
    matrix = tuple(
        tuple(
            _finite_number(item, owner=f"world_from_asset[{row_index}][{column}]")
            for column, item in enumerate(row)
        )
        for row_index, row in enumerate(rows)
    )
    if any(
        not math.isclose(matrix[3][column], expected, abs_tol=1.0e-9)
        for column, expected in enumerate((0.0, 0.0, 0.0, 1.0))
    ):
        _fail("world_from_asset must have homogeneous bottom row [0,0,0,1]")
    rotation = tuple(row[:3] for row in matrix[:3])
    for row_index, row in enumerate(rotation):
        norm = sum(value * value for value in row)
        if not math.isclose(norm, 1.0, abs_tol=1.0e-8):
            _fail(f"world_from_asset rotation row {row_index} is not unit length")
    for left in range(3):
        for right in range(left + 1, 3):
            dot = sum(
                rotation[left][axis] * rotation[right][axis]
                for axis in range(3)
            )
            if not math.isclose(dot, 0.0, abs_tol=1.0e-8):
                _fail("world_from_asset rotation is not orthogonal")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if not math.isclose(determinant, 1.0, abs_tol=1.0e-8):
        _fail("world_from_asset rotation must be right-handed")

    local = _finite_vector(
        anchors[0]["local_position_m"], 3, owner="static emitter local position"
    )
    return tuple(
        sum(matrix[row][column] * local[column] for column in range(3))
        + matrix[row][3]
        for row in range(3)
    )


__all__ = [
    "CANONICAL_COORDINATE_SYSTEM",
    "EMITTER_MEASUREMENT_SCHEMA",
    "MARKER_VISUAL_APPROVAL_SCHEMA",
    "STATIC_ADMISSION_EVIDENCE_SCHEMA",
    "STATIC_DECISION_SCHEMA",
    "STATIC_FINALIZATION_SCHEMA",
    "STATIC_ROUTE",
    "StaticObjectRegistrationError",
    "ValidatedStaticObjectAdmission",
    "publish_static_object_entity_registry",
    "resolve_static_object_emitter_world",
    "validate_static_object_admission",
    "verify_static_object_entity_registry",
]
