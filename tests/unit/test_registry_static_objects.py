from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any, Callable

import numpy as np
from PIL import Image
import pytest

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    write_json,
)
from avengine.registry.entities import (
    STATIC_OBJECT_EVIDENCE_KINDS,
    load_entity_asset_registry,
    validate_entity_asset_registry,
)
from avengine.registry.registry import bind_content_hash
import avengine.registry.static_objects as static_objects
from avengine.registry.static_objects import (
    ANCHOR_AUTHORITY_SCHEMA,
    ANCHOR_SPEC_SCHEMA,
    CANONICAL_COORDINATE_SYSTEM,
    EMITTER_MEASUREMENT_SCHEMA,
    HEADING_EVIDENCE_SCHEMA,
    MARKER_VISUAL_APPROVAL_SCHEMA,
    STATIC_ADMISSION_BATCH_SCHEMA,
    STATIC_ADMISSION_JOB_RECEIPT_SCHEMA,
    STATIC_ADMISSION_PLAN_SCHEMA,
    STATIC_ADMISSION_STAGE_RECEIPT_SCHEMA,
    STATIC_DECISION_BATCH_SCHEMA,
    STATIC_DECISION_SCHEMA,
    STATIC_FINALIZATION_SCHEMA,
    STATIC_PIXAL_INPUT_SCHEMA,
    STATIC_PIXAL_BATCH_SCHEMA,
    STATIC_REVIEW_BATCH_SCHEMA,
    STATIC_REVIEW_SCHEMA,
    STATIC_ROUTE,
    WATERTIGHT_MANIFEST_SCHEMA,
    StaticObjectRegistrationError,
    publish_static_object_entity_registry,
    resolve_static_object_emitter_world,
    validate_static_object_admission,
    verify_static_object_entity_registry,
)


REQUEST_SHA256 = "1" * 64
PROFILE_SHA256 = "2" * 64
INSTANCE_ID = "fixture_static_source_0001"
REVIEW_VIEW_KEYS = (
    "orbit_anchor",
    "orbit_opposite",
    "orbit_quarter",
    "orbit_right",
    "orbit_top",
)
WATERTIGHT_PARAMETERS = {
    "voxel_resolution": 128,
    "target_faces": 20000,
    "smooth_iterations": 2,
    "shrinkwrap_strength": 0.5,
    "post_shrinkwrap_smooth_iterations": 1,
    "torso_fold_repair_iterations": 0,
    "attribute_transfer_backend": "bvh",
    "bake_resolution": 1024,
    "base_color_encoding_policy": "preserve-bake",
    "base_color_gain": [1.0, 1.0, 1.0],
    "double_sided": False,
}
FIXTURE_ISNET_MODEL_BYTES = b"fixture pinned ISNet model bytes\n"
FIXTURE_ISNET_MODEL_SHA256 = hashlib.sha256(
    FIXTURE_ISNET_MODEL_BYTES
).hexdigest()


def _write_binary(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _record(path)


def _write_png(
    path: Path,
    *,
    size: tuple[int, int] = (480, 480),
    mode: str = "RGB",
    color: int | tuple[int, ...] = (80, 120, 160),
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path, format="PNG")
    return _record(path)


def _write_box_glb(
    path: Path,
    *,
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    image_uri: str | None = None,
    mutate_document: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    minimum_x, minimum_y, minimum_z = minimum
    maximum_x, maximum_y, maximum_z = maximum
    positions = (
        (minimum_x, minimum_y, minimum_z),
        (maximum_x, minimum_y, minimum_z),
        (maximum_x, maximum_y, minimum_z),
        (minimum_x, maximum_y, minimum_z),
        (minimum_x, minimum_y, maximum_z),
        (maximum_x, minimum_y, maximum_z),
        (maximum_x, maximum_y, maximum_z),
        (minimum_x, maximum_y, maximum_z),
    )
    normals = ((0.0, 1.0, 0.0),) * len(positions)
    triangles = (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
        (0, 1, 5),
        (0, 5, 4),
        (3, 7, 6),
        (3, 6, 2),
    )
    texcoords = (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )
    binary = bytearray()

    def append_blob(payload: bytes) -> tuple[int, int]:
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        binary.extend(payload)
        return offset, len(payload)

    position_offset, position_length = append_blob(
        b"".join(struct.pack("<3f", *item) for item in positions)
    )
    normal_offset, normal_length = append_blob(
        b"".join(struct.pack("<3f", *item) for item in normals)
    )
    texcoord_offset, texcoord_length = append_blob(
        b"".join(struct.pack("<2f", *item) for item in texcoords)
    )
    index_offset, index_length = append_blob(
        b"".join(
            struct.pack("<H", vertex)
            for triangle in triangles
            for vertex in triangle
        )
    )
    document = {
        "asset": {"version": "2.0", "generator": "AVEngine test fixture"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "fixture", "mesh": 0}],
        "meshes": [
            {
                "name": "fixture",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TEXCOORD_0": 2,
                        },
                        "indices": 3,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": list(minimum),
                "max": list(maximum),
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": len(normals),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": len(texcoords),
                "type": "VEC2",
            },
            {
                "bufferView": 3,
                "componentType": 5123,
                "count": len(triangles) * 3,
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": position_offset,
                "byteLength": position_length,
            },
            {
                "buffer": 0,
                "byteOffset": normal_offset,
                "byteLength": normal_length,
            },
            {
                "buffer": 0,
                "byteOffset": texcoord_offset,
                "byteLength": texcoord_length,
            },
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": index_length,
            },
        ],
        "buffers": [{"byteLength": len(binary)}],
        "materials": [
            {
                "name": "FixtureMaterial",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.5,
                },
            }
        ],
        "textures": [{"source": 0}],
        "images": [
            {
                "name": "FixtureImage",
                "uri": image_uri
                or (
                    "data:image/png;base64,"
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
                    "AAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                ),
            }
        ],
    }
    if mutate_document is not None:
        mutate_document(document)
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    padded_binary = bytes(binary) + b"\x00" * ((-len(binary)) % 4)
    total_length = 12 + 8 + len(encoded) + 8 + len(padded_binary)
    payload = b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total_length),
            struct.pack("<II", len(encoded), 0x4E4F534A),
            encoded,
            struct.pack("<II", len(padded_binary), 0x004E4942),
            padded_binary,
        )
    )
    return _write_binary(path, payload)


def _record(
    path: Path,
    *,
    relative_to: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload = path.read_bytes()
    recorded_path = (
        str(path.resolve().relative_to(relative_to.resolve()))
        if relative_to is not None
        else str(path.resolve())
    )
    return {
        "path": recorded_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        **extra,
    }


def _json_record(
    path: Path,
    value: dict[str, Any],
    *,
    relative_to: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)
    return _record(path, relative_to=relative_to, **extra)


def _seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = canonical_json_sha256(value)
    return value


def _path_hash_record(
    path: Path,
    *,
    content_hash_field: str,
    content_hash: str,
) -> dict[str, Any]:
    record = _record(path)
    return {
        "path": record["path"],
        "sha256": record["sha256"],
        content_hash_field: content_hash,
    }


def _base_registry(path: Path) -> None:
    registry = {
        "schema": "avengine_m6_entity_asset_registry_v1",
        "registry_id": "fixture_entity_registry",
        "revision": "base_v1",
        "entities": [
            {
                "entity_asset_id": "existing_rigid_source",
                "revision": "v1",
                "entity_class": "rigid_object",
                "visual_asset": {
                    "uri": "artifact://fixture/existing.glb",
                    "sha256": "a" * 64,
                },
                "realized_visual_attributes": {"fixture": True},
                "capabilities": {
                    "articulated": False,
                    "skeleton_revision": None,
                    "skeleton_sha256": None,
                    "action_ids": [],
                },
                "emitter_anchors": [
                    {
                        "anchor_id": "speaker",
                        "anchor_type": "object_speaker",
                        "joint_id": None,
                        "local_position_m": [0.0, 0.0, 0.0],
                    }
                ],
                "provenance": {
                    "source": "fixture",
                    "source_revision": "v1",
                    "license": None,
                    "rights_status": "research_use_only",
                    "evidence_sha256": "b" * 64,
                },
                "admission_state": "research",
            }
        ],
    }
    write_json(path, bind_content_hash(registry))


def _admission_fixture(
    root: Path,
    *,
    combined_pixal_inputs: bool = False,
) -> dict[str, Any]:
    root.mkdir(parents=True)
    pixal_input_root = root / "pixal_inputs"
    pixal_root = root / "pixal"
    flux_root = root / "flux"
    flux_review_root = root / "flux_review"
    review_root = root / "review"
    decision_root = root / "decision"
    authority_root = root / "authority"
    admission_root = root / "admission"
    for directory in (
        pixal_root,
        pixal_input_root,
        flux_root,
        flux_review_root,
        review_root,
        decision_root,
        authority_root,
        admission_root,
    ):
        directory.mkdir()

    pixal_glb_path = pixal_root / "fixture.glb"
    _write_binary(pixal_glb_path, b"pixal rigid textured mesh fixture\n")
    pixal_record = _record(pixal_glb_path)
    sampled_attributes = {
        "body_color": "warm_white",
        "material": "painted_metal",
    }
    target_physical_profile = {
        "control_attribute": None,
        "measurement": "height_cm",
        "target_value_cm": 100.0,
        "tolerance_cm": 1.0,
    }
    mesh_readback = {
        "vertices": 128,
        "triangles": 240,
        "materials": 1,
        "textures": 1,
        "skins": 0,
        "animations": 0,
    }
    execution_job_id = f"static_{REQUEST_SHA256[:16]}"
    one_shot_policy_path = (
        root / "contracts" / "animal_one_shot_no_seed_lottery_v1.json"
    )
    one_shot_policy_file = _json_record(
        one_shot_policy_path,
        {
            "schema": "avengine_controlled_animal_one_shot_policy_v1",
            "policy_id": "animal_one_shot_no_seed_lottery_v1",
            "request_freeze": {
                "seed_override_after_generation_started_allowed": False,
                "request_replacement_after_observing_output_allowed": False,
            },
            "per_request_cardinality": {
                "flux_invocations": 1,
                "flux_images_per_invocation": 1,
                "pixal3d_invocations": 1,
                "seed_retry_allowed": False,
                "candidate_ranking_or_best_of_n_allowed": False,
            },
            "failure_policy": {
                "failed_output_may_be_hidden_from_profile_metrics": False,
            },
            "profile_qualification": {
                "all_predeclared_requests_count": True,
                "required_pass_fraction": 1.0,
            },
            "production_instance_policy": {
                "rerun_flux_or_pixal_for_each_color_or_size_instance": False,
            },
            "formal_dataset_registration_authorized": False,
        },
    )
    one_shot_policy = {
        "schema": "avengine_controlled_animal_one_shot_policy_record_v1",
        "policy_id": "animal_one_shot_no_seed_lottery_v1",
        "policy_schema": "avengine_controlled_animal_one_shot_policy_v1",
        "path": one_shot_policy_file["path"],
        "sha256": one_shot_policy_file["sha256"],
    }

    def one_shot_execution(stage: str) -> dict[str, Any]:
        return {
            "policy": deepcopy(one_shot_policy),
            "stage": stage,
            "invocation_ordinal": 0,
            "invocations_allowed": 1,
            "seed_retry_allowed": False,
            "candidate_ranking_allowed": False,
            "failure_action": "preserve_evidence_and_reject_instance",
        }

    pixal_one_shot_execution = one_shot_execution("pixal3d")
    source_path = (
        flux_root / "candidates" / execution_job_id / "candidate.png"
    )
    rgba_path = (
        pixal_input_root
        / "segmentation"
        / INSTANCE_ID
        / "input_rgba_isnet.png"
    )
    alpha_path = (
        pixal_input_root / "segmentation" / INSTANCE_ID / "alpha_isnet.png"
    )
    _write_png(source_path, size=(1024, 1024), color=(170, 150, 120))
    alpha_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_image = Image.new("L", (1024, 1024), 0)
    alpha_image.paste(255, (100, 80, 900, 940))
    alpha_image.save(alpha_path, format="PNG")
    rgba_image = Image.new("RGBA", (1024, 1024), (170, 150, 120, 255))
    rgba_image.putalpha(alpha_image)
    rgba_image.save(rgba_path, format="PNG")
    source_record = _record(source_path)
    rgba_record = _record(rgba_path)
    controlled_request = {
        "execution_job_id": execution_job_id,
        "instance_id": INSTANCE_ID,
        "request_sha256": REQUEST_SHA256,
        "generation_seed": 42,
        "profile_schema_id": "fixture_static_profile_v1",
        "profile_sha256": PROFILE_SHA256,
        "asset_class": "static_object",
        "route": STATIC_ROUTE,
        "sampled_attributes": sampled_attributes,
        "target_physical_profile": target_physical_profile,
        "rig_profile": None,
    }

    static_generation_plan = {
        "schema": "flux2_pixal3d_static_generation_plan_v1",
        "route": STATIC_ROUTE,
        "prompt_template_id": "fixture_static_prompt_v1",
        "base_template": {
            "template_id": "static_object_text_prompt_only_v1",
            "kind": "text_prompt_only",
            "artifact": None,
            "provenance_status": "verified",
            "usage_scope": "research_candidate",
        },
        "prompt": "one fixture static object",
        "negative_prompt": "duplicate disconnected parts",
        "generation_seed": 42,
        "flux_invocations": 1,
        "model_revisions": {"flux2": "fixture_flux_revision"},
        "base_acquisition_policy": {
            "policy_id": "static_object_per_request_one_shot_v1",
            "acquisition_unit": "one_frozen_asset_per_request",
            "sampled_domains_must_be_singleton": False,
            "downstream_instance_route": STATIC_ROUTE,
            "profile_validation": (
                "all_predeclared_requests_count_zero_hidden_failures"
            ),
        },
    }
    preflight_job = {
        "execution_job_id": execution_job_id,
        "profile_schema_id": "fixture_static_profile_v1",
        "profile_sha256": PROFILE_SHA256,
        "lineage_group_id": "fixture_static_lineage",
        "state_classification": "research_candidate",
        "taxonomy": {"category": "fixture_static"},
        "fixed_attributes": {},
        "sampled_attributes": sampled_attributes,
        "consumer_requests": [
            {
                "instance_id": INSTANCE_ID,
                "request_sha256": REQUEST_SHA256,
            }
        ],
        "generation_plan": static_generation_plan,
        "target_physical_profile": target_physical_profile,
        "rig_profile": None,
        "acoustic_profile": {"source_role": "fixture"},
        "execution_gate": {
            "before_flux2": "authenticated_preflight_passed",
            "before_pixal3d": (
                "approved_2d_review_for_exact_candidate_sha256"
            ),
            "before_source_asset_v2": (
                "all_required_static_ue_audio_qa_passed"
            ),
        },
    }
    preflight = _seal(
        {
            "schema": "avengine_controlled_execution_preflight_v1",
            "source_bundle": {"fixture": True},
            "artifact_roots": {},
            "profile_artifact_authentication": {},
            "routes": {
                "flux2_pixal3d_animal_v1": [],
                STATIC_ROUTE: [preflight_job],
                "stable_animal_template_v1": [],
                "rocketbox_material_v1": [],
            },
            "execution_summary": {"fixture": True},
            "automatic_checks": {"overall": "passed"},
        },
        "preflight_sha256",
    )
    preflight_path = root / "flux_execution_preflight.json"
    _json_record(preflight_path, preflight)
    flux_candidate_manifest = _seal(
        {
            "schema": "avengine_controlled_animal_flux2_candidate_v1",
            "status": "pending_2d_review",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "execution_preflight_sha256": preflight["preflight_sha256"],
            "execution_job_id": execution_job_id,
            "instance_id": INSTANCE_ID,
            "request_sha256": REQUEST_SHA256,
            "profile_schema_id": "fixture_static_profile_v1",
            "profile_sha256": PROFILE_SHA256,
            "lineage_group_id": "fixture_static_lineage",
            "taxonomy": {"category": "fixture_static"},
            "fixed_attributes": {},
            "sampled_attributes": sampled_attributes,
            "input": None,
            "output": _record(source_path, relative_to=flux_root),
            "generation": {
                "prompt": "one fixture static object",
                "negative_prompt": "duplicate disconnected parts",
                "effective_prompt": "one fixture static object",
                "seed": 42,
                "flux_invocations": 1,
                "model": {"revision": "fixture_flux_revision"},
                "parameters": {"width": 1024, "height": 1024},
            },
            "one_shot_execution": one_shot_execution("flux2"),
            "downstream_gate": {
                "status": "blocked_pending_2d_review",
                "required_review": "approved_for_exact_candidate_sha256",
                "next_stage": "foreground_segmentation_then_pixal3d",
            },
            "timings": {
                "persistent_worker_model_load_seconds": 1.0,
                "inference_and_publish_seconds": 2.0,
                "model_reused": True,
            },
            "automatic_checks": {
                "reference_hash_before_after_stable": True,
                "one_flux_invocation": True,
                "one_flux_image": True,
                "seed_retry_forbidden": True,
                "candidate_ranking_forbidden": True,
                "canvas_1024_rgb": True,
                "visual_attributes_verified": False,
                "overall": "pending_2d_review",
            },
        },
        "manifest_sha256",
    )
    flux_candidate_manifest_path = source_path.parent / "candidate_manifest.json"
    _json_record(flux_candidate_manifest_path, flux_candidate_manifest)
    flux_batch = _seal(
        {
            "schema": "avengine_controlled_animal_flux2_batch_v1",
            "status": "pending_2d_review",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "one_shot_execution": one_shot_execution("flux2"),
            "execution_preflight": _path_hash_record(
                preflight_path,
                content_hash_field="preflight_sha256",
                content_hash=preflight["preflight_sha256"],
            ),
            "selection": {
                "semantics": "predeclared_request_subset_only_not_output_ranking",
                "route": STATIC_ROUTE,
                "profile_ids": ["fixture_static_profile_v1"],
                "execution_job_ids": [execution_job_id],
                "qa_pair_canary": False,
                "planned_qa_pairs": [],
            },
            "model": {"revision": "fixture_flux_revision"},
            "parameters": {"width": 1024, "height": 1024},
            "candidate_count": 1,
            "candidates": [
                {
                    "execution_job_id": execution_job_id,
                    "instance_id": INSTANCE_ID,
                    "profile_schema_id": "fixture_static_profile_v1",
                    "sampled_attributes": sampled_attributes,
                    "status": "pending_2d_review",
                    "candidate": _record(source_path, relative_to=flux_root),
                    "candidate_manifest": _record(
                        flux_candidate_manifest_path,
                        relative_to=flux_root,
                    ),
                }
            ],
            "workers": [{"gpu": 0, "returncode": 0}],
            "automatic_checks": {"overall": "pending_2d_review"},
        },
        "batch_sha256",
    )
    flux_batch_path = flux_root / "flux2_batch_manifest.json"
    _json_record(flux_batch_path, flux_batch)

    flux_review_decisions_path = flux_review_root / "decisions.json"
    upstream_checks = {
        "category_identity": "passed",
        "construction": "passed",
        "stable_product_pose": "passed",
        "background": "passed",
        "sampled_attributes": {
            attribute: "passed" for attribute in sampled_attributes
        },
        "hard_gates": {
            "single_subject": "passed",
            "photorealistic_pbr_style": "passed",
            "category_distinctive_features": "passed",
            "emitter_feature_visible": "passed",
            "physically_connected_construction": "passed",
            "complete_object": "passed",
            "stable_rest_or_mount": "passed",
            "target_attribute_only": "passed",
        },
    }
    _json_record(
        flux_review_decisions_path,
        {
            "schema": "avengine_controlled_static_object_2d_review_decisions_v1",
            "flux2_batch_sha256": flux_batch["batch_sha256"],
            "reviewer": "fixture_reviewer",
            "decisions": [
                {
                    "instance_id": INSTANCE_ID,
                    "candidate_sha256": source_record["sha256"],
                    "decision": "approved_for_pixal3d",
                    "sampled_attribute_checks": upstream_checks[
                        "sampled_attributes"
                    ],
                    "notes": (
                        "Fixture passes every objective static hard gate."
                    ),
                    "category_identity": "passed",
                    "construction": "passed",
                    "stable_product_pose": "passed",
                    "background": "passed",
                    "hard_gates": upstream_checks["hard_gates"],
                }
            ],
        },
    )
    flux_review = _seal(
        {
            "schema": "avengine_controlled_static_object_2d_review_v1",
            "instance_id": INSTANCE_ID,
            "request_sha256": REQUEST_SHA256,
            "profile_schema_id": "fixture_static_profile_v1",
            "sampled_attributes": sampled_attributes,
            "candidate": source_record,
            "candidate_manifest": _record(flux_candidate_manifest_path),
            "reviewer": "fixture_reviewer",
            "decision": "approved_for_pixal3d",
            "checks": upstream_checks,
            "notes": "Fixture passes every objective static hard gate.",
            "downstream_gate": "approved_for_segmentation_and_pixal3d",
        },
        "review_sha256",
    )
    flux_review_path = flux_review_root / "reviews" / f"{INSTANCE_ID}.json"
    flux_review_record = _json_record(
        flux_review_path,
        flux_review,
        relative_to=flux_review_root,
    )
    flux_review_batch = _seal(
        {
            "schema": "avengine_controlled_static_object_2d_review_batch_v1",
            "status": "passed",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "review_domain": "static_object",
            "flux2_batch": _path_hash_record(
                flux_batch_path,
                content_hash_field="batch_sha256",
                content_hash=flux_batch["batch_sha256"],
            ),
            "decisions_input": _record(flux_review_decisions_path),
            "candidate_count": 1,
            "approved_count": 1,
            "rejected_count": 0,
            "reviews": [
                {
                    "instance_id": INSTANCE_ID,
                    "profile_schema_id": "fixture_static_profile_v1",
                    "decision": "approved_for_pixal3d",
                    "candidate_sha256": source_record["sha256"],
                    "review": flux_review_record,
                }
            ],
            "qa_pair_eligibility": [],
            "automatic_checks": {"overall": "passed"},
        },
        "review_batch_sha256",
    )
    flux_review_batch_path = flux_review_root / "review_batch_manifest.json"
    _json_record(flux_review_batch_path, flux_review_batch)

    attempt_manifest_path = pixal_root / "fixture.manifest.json"
    pixal_input_job = {
        "legacy_tag": INSTANCE_ID,
        "candidate_tag": f"{INSTANCE_ID}_pixal_v1",
        "asset_class": "static_object",
        "route": STATIC_ROUTE,
        "seed": 42,
        "attempt_ordinal": 0,
        "one_shot_execution": pixal_one_shot_execution,
        "reference": {
            "source": source_record,
            "pixal_input": rgba_record,
            "normalization": "pinned_isnet_general_use_alpha_v1",
        },
        "output": str(pixal_glb_path.resolve()),
        "manifest": str(attempt_manifest_path.resolve()),
        "controlled_request": controlled_request,
        "model_revisions": {
            "pixal3d": "fixture_pixal_revision",
            "dino": "fixture_dino_revision",
        },
        "parameters": {
            "resolution": 1024,
            "manual_fov": 0.2,
            "low_vram": False,
        },
    }
    model_path = root / "models" / "isnet-general-use.onnx"
    _write_binary(model_path, FIXTURE_ISNET_MODEL_BYTES)
    runtime_root = root / "runtime"
    resolved_python_path = runtime_root / "python3.10"
    _write_binary(resolved_python_path, b"#!/bin/sh\nexit 0\n")
    resolved_python_path.chmod(0o755)
    configured_python_path = runtime_root / "python"
    configured_python_path.symlink_to(resolved_python_path.name)
    resolved_python_record = _record(resolved_python_path)
    configured_python_record = {
        **resolved_python_record,
        "path": str(configured_python_path.absolute()),
    }

    spear_root = root / "spear"
    worker_source_path = (
        spear_root / "tools" / "controlled_animal_isnet_worker.py"
    )
    worker_payload = (
        "from pathlib import Path\n"
        f"MODEL_PATH = Path({str(model_path.resolve())!r})\n"
        f"MODEL_SHA256 = {FIXTURE_ISNET_MODEL_SHA256!r}\n"
        "JOBS_SCHEMA = 'avengine_controlled_animal_isnet_jobs_v1'\n"
        "STATUS_SCHEMA = 'avengine_controlled_animal_isnet_status_v1'\n"
    ).encode("utf-8")
    _write_binary(worker_source_path, worker_payload)
    worker_executed_path = (
        pixal_input_root
        / ".runtime_commands"
        / "controlled_animal_isnet_worker.py"
    )
    _write_binary(worker_executed_path, worker_payload)
    worker_executed_path.chmod(0o444)
    worker_executed_path.parent.chmod(0o555)

    staging_root = root / f".{pixal_input_root.name}.fixture.staging"
    isnet_jobs_path = pixal_input_root / "isnet_jobs.json"
    _json_record(
        isnet_jobs_path,
        {
            "schema": "avengine_controlled_animal_isnet_jobs_v1",
            "jobs": [
                {
                    "instance_id": INSTANCE_ID,
                    "candidate_path": source_record["path"],
                    "candidate_sha256": source_record["sha256"],
                    "alpha_path": str(
                        staging_root
                        / "segmentation"
                        / INSTANCE_ID
                        / "alpha_isnet.png"
                    ),
                    "rgba_path": str(
                        staging_root
                        / "segmentation"
                        / INSTANCE_ID
                        / "input_rgba_isnet.png"
                    ),
                }
            ],
        },
    )
    isnet_status_path = pixal_input_root / "isnet_status.json"
    _json_record(
        isnet_status_path,
        {
            "schema": "avengine_controlled_animal_isnet_status_v1",
            "status": "passed",
            "model": {
                "path": str(model_path.resolve()),
                "sha256": FIXTURE_ISNET_MODEL_SHA256,
                "name": "isnet-general-use",
            },
            "model_load_seconds": 0.1,
            "jobs": [
                {
                    "instance_id": INSTANCE_ID,
                    "status": "passed",
                    "alpha_path": str(
                        staging_root
                        / "segmentation"
                        / INSTANCE_ID
                        / "alpha_isnet.png"
                    ),
                    "alpha_sha256": _record(alpha_path)["sha256"],
                    "rgba_path": str(
                        staging_root
                        / "segmentation"
                        / INSTANCE_ID
                        / "input_rgba_isnet.png"
                    ),
                    "rgba_sha256": rgba_record["sha256"],
                    "foreground_fraction_at_128": 0.6561279296875,
                    "foreground_bbox_xyxy": [100, 80, 900, 940],
                    "alpha_extrema": [0, 255],
                    "wall_seconds": 0.2,
                }
            ],
            "passed_count": 1,
            "failed_count": 0,
        },
    )
    isnet_log_path = pixal_input_root / "isnet.log"
    _write_binary(isnet_log_path, b"")
    isnet_command = [
        str(resolved_python_path.resolve()),
        str(worker_executed_path.resolve()),
        "--jobs",
        str(isnet_jobs_path.resolve()),
        "--status",
        str(isnet_status_path.resolve()),
    ]
    executed_isnet_command = [
        (
            str(staging_root)
            + item[len(str(pixal_input_root.resolve())) :]
        )
        if item == str(pixal_input_root.resolve())
        or item.startswith(str(pixal_input_root.resolve()) + os.sep)
        else item
        for item in isnet_command
    ]
    pixal_inputs = _seal(
        {
            "schema": STATIC_PIXAL_INPUT_SCHEMA,
            "status": "ready_for_pixal3d",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "asset_class": "static_object",
            "route": STATIC_ROUTE,
            "one_shot_execution": pixal_one_shot_execution,
            "upstream_flux_one_shot_evidence": {
                "mode": "native_policy_enforced_before_inference",
                "policy": one_shot_policy,
                "flux_batch_sha256": flux_batch["batch_sha256"],
                "profile_qualification_authorized": True,
            },
            "review_batch": _path_hash_record(
                flux_review_batch_path,
                content_hash_field="review_batch_sha256",
                content_hash=flux_review_batch["review_batch_sha256"],
            ),
            "isnet": {
                "schema": "avengine_controlled_isnet_execution_receipt_v1",
                "model": _record(model_path),
                "python": {
                    "configured": configured_python_record,
                    "resolved": resolved_python_record,
                },
                "worker": {
                    "source": _record(worker_source_path),
                    "executed": _record(worker_executed_path),
                },
                "jobs": _record(isnet_jobs_path),
                "working_directory": str(spear_root.resolve()),
                "command": isnet_command,
                "command_sha256": canonical_json_sha256(isnet_command),
                "executed_command": executed_isnet_command,
                "executed_command_sha256": canonical_json_sha256(
                    executed_isnet_command
                ),
                "path_rebinding": {
                    "staging_root": str(staging_root),
                    "published_root": str(pixal_input_root.resolve()),
                },
                "status": _record(
                    isnet_status_path,
                    relative_to=pixal_input_root,
                ),
                "log": _record(
                    isnet_log_path,
                    relative_to=pixal_input_root,
                ),
            },
            "pixal_output_root": str(pixal_root.resolve()),
            "job_count": 1,
            "jobs": [pixal_input_job],
            "segmentations": [
                {
                    "instance_id": INSTANCE_ID,
                    "candidate_sha256": source_record["sha256"],
                    "alpha": _record(
                        alpha_path,
                        relative_to=pixal_input_root,
                    ),
                    "rgba": _record(
                        rgba_path,
                        relative_to=pixal_input_root,
                    ),
                    "foreground_fraction_at_128": 0.6561279296875,
                    "foreground_bbox_xyxy": [100, 80, 900, 940],
                    "status": "passed",
                }
            ],
            "automatic_checks": {
                "static_jobs_have_no_rig_or_animation_binding": True,
                "overall": "passed",
            },
        },
        "manifest_sha256",
    )
    pixal_inputs_path = pixal_input_root / "pixal_inputs_manifest.json"
    _json_record(pixal_inputs_path, pixal_inputs)
    selected_pixal_inputs = pixal_inputs
    selected_pixal_inputs_path = pixal_inputs_path
    selected_rgba_record = rgba_record
    if combined_pixal_inputs:
        second_root = root / "pixal_inputs_second"
        second_root.mkdir()
        second_source_path = second_root / "source.png"
        second_rgba_path = (
            second_root / "segmentation/fixture_static_source_0002/input.png"
        )
        _write_binary(second_source_path, b"second source pixels\n")
        _write_binary(second_rgba_path, b"second rgba pixels\n")
        second_request_sha256 = "3" * 64
        second_controlled = {
            **controlled_request,
            "execution_job_id": f"static_{second_request_sha256[:16]}",
            "instance_id": "fixture_static_source_0002",
            "request_sha256": second_request_sha256,
        }
        second_job = {
            **deepcopy(pixal_input_job),
            "legacy_tag": "fixture_static_source_0002",
            "candidate_tag": "fixture_static_source_0002_pixal_v1",
            "reference": {
                "source": _record(second_source_path),
                "pixal_input": _record(second_rgba_path),
                "normalization": "pinned_isnet_general_use_alpha_v1",
            },
            "output": str(
                (root / "unused_pixal/fixture_static_source_0002/output.glb")
                .resolve()
            ),
            "manifest": str(
                (
                    root
                    / "unused_pixal/fixture_static_source_0002/output.manifest.json"
                ).resolve()
            ),
            "controlled_request": second_controlled,
        }
        second_inputs = deepcopy(pixal_inputs)
        second_inputs["pixal_output_root"] = str(
            (root / "unused_pixal").resolve()
        )
        second_inputs["jobs"] = [second_job]
        second_inputs["segmentations"] = [
            {
                "instance_id": "fixture_static_source_0002",
                "status": "passed",
            }
        ]
        second_inputs.pop("manifest_sha256")
        _seal(second_inputs, "manifest_sha256")
        second_inputs_path = second_root / "pixal_inputs_manifest.json"
        _json_record(second_inputs_path, second_inputs)

        combined_root = root / "pixal_inputs_combined"
        combined_root.mkdir()
        copied_rgba_path = (
            combined_root / "segmentation" / INSTANCE_ID / "input.png"
        )
        copied_second_rgba_path = (
            combined_root
            / "segmentation"
            / "fixture_static_source_0002"
            / "input.png"
        )
        _write_binary(copied_rgba_path, rgba_path.read_bytes())
        _write_binary(
            copied_second_rgba_path, second_rgba_path.read_bytes()
        )
        copied_rgba_record = _record(copied_rgba_path)
        copied_second_rgba_record = _record(copied_second_rgba_path)
        combined_job = deepcopy(pixal_input_job)
        combined_job["reference"]["pixal_input"] = copied_rgba_record
        combined_second_job = deepcopy(second_job)
        combined_second_job["reference"][
            "pixal_input"
        ] = copied_second_rgba_record
        combined_second_job["output"] = str(
            (
                pixal_root / "fixture_static_source_0002/pixal_raw_1024.glb"
            ).resolve()
        )
        combined_second_job["manifest"] = str(
            Path(combined_second_job["output"]).with_suffix(
                ".manifest.json"
            )
        )

        def parent_receipt(
            ordinal: int,
            path: Path,
            payload: dict[str, Any],
        ) -> dict[str, Any]:
            record = _record(path)
            return {
                "ordinal": ordinal,
                "path": record["path"],
                "sha256": record["sha256"],
                "schema": STATIC_PIXAL_INPUT_SCHEMA,
                "content_sha256": payload["manifest_sha256"],
                "asset_class": "static_object",
                "route": STATIC_ROUTE,
                "job_count": 1,
            }

        parents = [
            parent_receipt(0, pixal_inputs_path, pixal_inputs),
            parent_receipt(1, second_inputs_path, second_inputs),
        ]
        combined = _seal(
            {
                "schema": "avengine_controlled_pixal_inputs_combined_v1",
                "status": "ready_for_pixal3d",
                "state_classification": "research_candidate",
                "formal_dataset_registration_authorized": False,
                "asset_class": "static_object",
                "route": STATIC_ROUTE,
                "one_shot_execution": pixal_one_shot_execution,
                "upstream_flux_one_shot_evidence": {
                    "schema": (
                        "avengine_combined_upstream_flux_"
                        "one_shot_evidence_v1"
                    ),
                    "policy": one_shot_policy,
                    "parent_count": 2,
                    "parents": [
                        {
                            "parent_content_sha256": pixal_inputs[
                                "manifest_sha256"
                            ],
                            "evidence": deepcopy(
                                pixal_inputs[
                                    "upstream_flux_one_shot_evidence"
                                ]
                            ),
                        },
                        {
                            "parent_content_sha256": second_inputs[
                                "manifest_sha256"
                            ],
                            "evidence": deepcopy(
                                second_inputs[
                                    "upstream_flux_one_shot_evidence"
                                ]
                            ),
                        },
                    ],
                },
                "model_revisions": pixal_input_job["model_revisions"],
                "parameters": pixal_input_job["parameters"],
                "combined_input_root": str(combined_root.resolve()),
                "parent_count": 2,
                "parents": parents,
                "pixal_output_root": str(pixal_root.resolve()),
                "job_count": 2,
                "jobs": [combined_job, combined_second_job],
                "input_copies": [
                    {
                        "instance_id": INSTANCE_ID,
                        "parent_ordinal": 0,
                        "parent_content_sha256": pixal_inputs[
                            "manifest_sha256"
                        ],
                        "parent_job_sha256": canonical_json_sha256(
                            pixal_input_job
                        ),
                        "source": source_record,
                        "parent_pixal_input": rgba_record,
                        "copied_pixal_input": copied_rgba_record,
                    },
                    {
                        "instance_id": "fixture_static_source_0002",
                        "parent_ordinal": 1,
                        "parent_content_sha256": second_inputs[
                            "manifest_sha256"
                        ],
                        "parent_job_sha256": canonical_json_sha256(
                            second_job
                        ),
                        "source": second_job["reference"]["source"],
                        "parent_pixal_input": second_job["reference"][
                            "pixal_input"
                        ],
                        "copied_pixal_input": copied_second_rgba_record,
                    },
                ],
                "automatic_checks": {
                    "static_jobs_have_no_rig_or_animation_binding": True,
                    "overall": "passed",
                },
            },
            "manifest_sha256",
        )
        selected_pixal_inputs_path = (
            combined_root / "pixal_inputs_manifest.json"
        )
        _json_record(selected_pixal_inputs_path, combined)
        selected_pixal_inputs = combined
        selected_rgba_record = copied_rgba_record
    attempt_manifest = {
        "backend": "pixal3d",
        "controlled_request": controlled_request,
        "model": {"revision": "fixture_pixal_revision"},
        "dino": {"revision": "fixture_dino_revision"},
        "parameters": {
            "low_vram": False,
            "manual_fov": 0.2,
            "resolution": 1024,
            "seed": 42,
        },
        "one_shot_execution": pixal_one_shot_execution,
        "output": pixal_record,
    }
    _json_record(attempt_manifest_path, attempt_manifest)
    pixal_attempt = {
        "instance_id": INSTANCE_ID,
        "execution_job_id": execution_job_id,
        "request_sha256": REQUEST_SHA256,
        "profile_schema_id": "fixture_static_profile_v1",
        "sampled_attributes": sampled_attributes,
        "target_physical_profile": target_physical_profile,
        "gpu": 0,
        "seed": 42,
        "attempt_ordinal": 0,
        "one_shot_execution": pixal_one_shot_execution,
        "pixal_input": selected_rgba_record,
        "mesh_readback": mesh_readback,
        "output": _record(pixal_glb_path, relative_to=pixal_root),
        "attempt_manifest": _record(
            attempt_manifest_path, relative_to=pixal_root
        ),
        "timings": {
            "model_load_seconds": 1.0,
            "inference_and_export_seconds": 2.0,
            "model_reused": False,
        },
        "status": "passed_generation_and_glb_readback",
        "next_gate": "static_visual_qa",
    }
    pixal_batch = _seal(
        {
            "schema": STATIC_PIXAL_BATCH_SCHEMA,
            "status": "passed_generation_and_glb_readback",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "one_shot_execution": pixal_one_shot_execution,
            "upstream_flux_one_shot_evidence": deepcopy(
                selected_pixal_inputs["upstream_flux_one_shot_evidence"]
            ),
            "started_at": "2026-07-27T00:00:00+00:00",
            "finished_at": "2026-07-27T00:00:01+00:00",
            "pixal_inputs": _path_hash_record(
                selected_pixal_inputs_path,
                content_hash_field="manifest_sha256",
                content_hash=selected_pixal_inputs["manifest_sha256"],
            ),
            "models": {
                "pixal3d_revision": "fixture_pixal_revision",
                "dino_revision": "fixture_dino_revision",
            },
            "parameters": {
                "resolution": 1024,
                "manual_fov": 0.2,
                "low_vram": False,
            },
            "gpus": [0],
            "job_count": 1,
            "passed_count": 1,
            "failed_count": 0,
            "attempts": [pixal_attempt],
            "workers": [{"fixture": True}],
            "scheduling": {"fixture": True},
            "automatic_checks": {"overall": "passed"},
        },
        "batch_sha256",
    )
    pixal_batch_path = pixal_root / "batch.json"
    _json_record(pixal_batch_path, pixal_batch)

    render_manifest_path = review_root / "raw_pbr_manifest.json"
    blender_log_path = review_root / "raw_pbr.log"
    contact_sheet_path = review_root / "contact_sheet.png"
    _json_record(
        render_manifest_path,
        {
            "input": str(pixal_glb_path.resolve()),
            "bbox_min": [-0.5, 0.0, -0.5],
            "bbox_max": [0.5, 1.0, 0.5],
            "extent": [1.0, 1.0, 1.0],
            "front_axis": "negative-y",
            "views": {
                "front": [0.0, -2.0, 0.5],
                "back": [0.0, 2.0, 0.5],
                "side": [2.0, 0.0, 0.5],
                "top": [0.0, 0.0, 2.5],
                "quarter": [1.5, -1.5, 0.5],
            },
            "resolution": [480, 480],
            "samples": 64,
            "material_preview": {"mode": "raw_glb_material"},
            "lighting": {"fixture": True},
        },
    )
    _write_binary(blender_log_path, b"blender review passed\n")
    _write_png(contact_sheet_path, size=(1200, 720))
    view_records: dict[str, Any] = {}
    for index, name in enumerate(REVIEW_VIEW_KEYS):
        path = review_root / f"{name}.png"
        _write_png(
            path,
            color=(40 + index * 20, 80 + index * 10, 120 + index * 5),
        )
        view_records[name] = _record(path, relative_to=review_root)

    orientation_contract = {
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
            "view_mapping": {
                "orbit_anchor": "front",
                "orbit_opposite": "back",
                "orbit_right": "side",
                "orbit_top": "top",
                "orbit_quarter": "quarter",
            },
        },
        "canonical_heading": {
            "status": "deferred_to_static_finalization",
            "axis": None,
            "derived_from_reference_facing": False,
        },
    }
    physical_scale_contract = {
        "status": "deferred_to_finalization",
        "control_attribute": None,
        "measurement": "height_cm",
        "target_physical_profile": target_physical_profile,
    }
    review = _seal(
        {
            "schema": STATIC_REVIEW_SCHEMA,
            "instance_id": INSTANCE_ID,
            "execution_job_id": execution_job_id,
            "request_sha256": REQUEST_SHA256,
            "profile_schema_id": "fixture_static_profile_v1",
            "profile_sha256": PROFILE_SHA256,
            "asset_class": "static_object",
            "route": STATIC_ROUTE,
            "sampled_attributes": sampled_attributes,
            "target_physical_profile": target_physical_profile,
            "physical_scale": physical_scale_contract,
            "orientation": orientation_contract,
            "pixal_output": pixal_attempt["output"],
            "mesh_readback": mesh_readback,
            "reference_rgba": selected_rgba_record,
            "raw_pbr_render_manifest": _record(
                render_manifest_path, relative_to=review_root
            ),
            "raw_pbr_views": view_records,
            "raw_pbr_blender_log": _record(
                blender_log_path, relative_to=review_root
            ),
            "clay_geometry": {"status": "not_requested"},
            "contact_sheet": _record(
                contact_sheet_path, relative_to=review_root
            ),
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "automatic_checks": {"overall": "passed"},
            "visual_qa": "pending",
            "next_gate": "static_object_visual_decision",
        },
        "review_sha256",
    )
    review_path = review_root / "selected_review.json"
    review_record = _json_record(
        review_path, review, relative_to=review_root
    )
    review_batch = _seal(
        {
            "schema": STATIC_REVIEW_BATCH_SCHEMA,
            "status": "rendered_pending_visual_qa",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "asset_class": "static_object",
            "route": STATIC_ROUTE,
            "pixal_batch": _path_hash_record(
                pixal_batch_path,
                content_hash_field="batch_sha256",
                content_hash=pixal_batch["batch_sha256"],
            ),
            "orientation": orientation_contract,
            "render_contract": {
                "reference_rgba_included": True,
                "raw_pbr_views": [
                    "orbit_anchor",
                    "orbit_opposite",
                    "orbit_right",
                    "orbit_top",
                    "orbit_quarter",
                ],
                "clay_geometry": "not_requested",
            },
            "review_count": 1,
            "reviews": [
                {
                    "instance_id": INSTANCE_ID,
                    "request_sha256": REQUEST_SHA256,
                    "review_sha256": review["review_sha256"],
                    "review": review_record,
                    "contact_sheet": _record(
                        contact_sheet_path, relative_to=review_root
                    ),
                    "status": "rendered_pending_visual_qa",
                }
            ],
            "automatic_checks": {"overall": "passed"},
        },
        "review_batch_sha256",
    )
    review_batch_path = review_root / "review_batch.json"
    _json_record(review_batch_path, review_batch)

    manual_decision = {
        "instance_id": INSTANCE_ID,
        "review_sha256": review["review_sha256"],
        "decision": "approved_for_watertight_finalization",
        "checks": {
            "silhouette_and_category_identity": True,
            "emitter_feature_visible": True,
            "material_and_declared_attributes": True,
            "physically_plausible_construction": True,
            "no_disconnected_or_floating_parts": True,
        },
        "attribute_evidence": {
            field: "passed_raw_pbr_visual" for field in sampled_attributes
        },
        "caveats": [],
        "notes": "Fixture object passes bounded human visual review.",
    }
    decision_input = {
        "schema": "avengine_controlled_static_object_review_decisions_v1",
        "static_object_review_batch_sha256": review_batch[
            "review_batch_sha256"
        ],
        "decisions": [manual_decision],
    }
    decision_input_path = decision_root / "manual_decisions.json"
    decision_input_record = _json_record(
        decision_input_path, decision_input
    )
    decision = _seal(
        {
            "schema": STATIC_DECISION_SCHEMA,
            **manual_decision,
            "asset_class": "static_object",
            "route": STATIC_ROUTE,
            "request_sha256": REQUEST_SHA256,
            "profile_sha256": PROFILE_SHA256,
            "target_physical_profile": target_physical_profile,
            "pixal_output": pixal_record,
            "review": _record(review_path),
            "physical_scale": physical_scale_contract,
            "canonical_heading": orientation_contract[
                "canonical_heading"
            ],
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "next_gate": "watertight_then_static_finalization",
        },
        "decision_sha256",
    )
    decision_path = decision_root / "selected_decision.json"
    decision_record = _json_record(
        decision_path, decision, relative_to=decision_root
    )
    decision_batch = _seal(
        {
            "schema": STATIC_DECISION_BATCH_SCHEMA,
            "status": "completed",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "asset_class": "static_object",
            "route": STATIC_ROUTE,
            "static_object_review_batch": _path_hash_record(
                review_batch_path,
                content_hash_field="review_batch_sha256",
                content_hash=review_batch["review_batch_sha256"],
            ),
            "decision_input": decision_input_record,
            "decision_count": 1,
            "approved_count": 1,
            "rejected_count": 0,
            "decisions": [
                {
                    "instance_id": INSTANCE_ID,
                    "request_sha256": REQUEST_SHA256,
                    "profile_sha256": PROFILE_SHA256,
                    "decision": "approved_for_watertight_finalization",
                    "decision_sha256": decision["decision_sha256"],
                    "pixal_output_sha256": pixal_record["sha256"],
                    "record": decision_record,
                }
            ],
            "automatic_checks": {"overall": "passed"},
        },
        "decision_batch_sha256",
    )
    decision_batch_path = decision_root / "decision_batch.json"
    decision_batch_record = _json_record(
        decision_batch_path,
        decision_batch,
        decision_batch_sha256=decision_batch["decision_batch_sha256"],
    )

    heading_review_path = authority_root / "heading_review.png"
    anchor_review_path = authority_root / "anchor_review.png"
    _write_binary(heading_review_path, b"reviewed positive x heading\n")
    _write_binary(anchor_review_path, b"reviewed speaker surface\n")
    heading_authority = {
        "schema": HEADING_EVIDENCE_SCHEMA,
        "instance_id": INSTANCE_ID,
        "request_sha256": REQUEST_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "input_glb_sha256": pixal_record["sha256"],
        "review_artifact": _record(heading_review_path),
        "reviewed_source_front_yaw_deg": 0.0,
        "target_front_axis": "positive-x",
        "decision": "approved_for_positive_x_normalization",
        "formal_dataset_registration_authorized": False,
    }
    heading_authority_path = authority_root / "heading_authority.json"
    heading_authority_record = _json_record(
        heading_authority_path, heading_authority
    )
    selection = {
        "method": "reviewed_bbox_fraction_nearest_surface_v1",
        "samples": [
            {"target_fraction_xyz": [0.8, 0.6, 0.55], "weight": 1.0}
        ],
        "aggregation": "weighted_centroid",
        "maximum_search_distance_fraction": 0.25,
    }
    anchor_authority = {
        "schema": ANCHOR_AUTHORITY_SCHEMA,
        "instance_id": INSTANCE_ID,
        "request_sha256": REQUEST_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "input_glb_sha256": pixal_record["sha256"],
        "anchor_id": "speaker_surface",
        "anchor_type": "object_speaker",
        "semantic_role": "reviewed_speaker_surface",
        "selection": selection,
        "review_evidence": _record(anchor_review_path),
        "formal_dataset_registration_authorized": False,
    }
    anchor_authority_path = authority_root / "anchor_authority.json"
    anchor_authority_record = _json_record(
        anchor_authority_path, anchor_authority
    )
    plan = _seal(
        {
            "schema": STATIC_ADMISSION_PLAN_SCHEMA,
            "decision_batch_sha256": decision_batch[
                "decision_batch_sha256"
            ],
            "instances": [
                {
                    "instance_id": INSTANCE_ID,
                    "heading_evidence_path": str(
                        heading_authority_path.resolve()
                    ),
                    "anchor_spec_path": str(anchor_authority_path.resolve()),
                    "watertight_parameters": WATERTIGHT_PARAMETERS,
                }
            ],
            "formal_dataset_registration_authorized": False,
        },
        "plan_sha256",
    )
    plan_path = authority_root / "admission_plan.json"
    plan_record = _json_record(
        plan_path, plan, plan_sha256=plan["plan_sha256"]
    )

    job_root = admission_root / "jobs" / INSTANCE_ID
    watertight_root = job_root / "01_watertight"
    finalization_root = job_root / "02_finalization"
    emitter_root = job_root / "03_emitter"
    for directory in (watertight_root, finalization_root, emitter_root):
        directory.mkdir(parents=True)
    frozen_root = (
        admission_root / "input_snapshots" / "instances" / INSTANCE_ID
    )
    frozen_pixal_path = frozen_root / "raw_pixal.glb"
    frozen_heading_review_path = frozen_root / "heading_review.bin"
    frozen_anchor_review_path = frozen_root / "anchor_review.bin"
    _write_binary(frozen_pixal_path, pixal_glb_path.read_bytes())
    _write_binary(
        frozen_heading_review_path, heading_review_path.read_bytes()
    )
    _write_binary(frozen_anchor_review_path, anchor_review_path.read_bytes())
    frozen_pixal_record = _record(frozen_pixal_path)

    watertight_glb_path = watertight_root / "watertight.glb"
    _write_box_glb(
        watertight_glb_path,
        minimum=(-0.4, 0.0, -0.4),
        maximum=(0.4, 0.8, 0.4),
    )
    watertight_glb_record = _record(watertight_glb_path)
    watertight_manifest = {
        "schema": WATERTIGHT_MANIFEST_SCHEMA,
        "created_at": "2026-07-27T00:01:00+00:00",
        "input": frozen_pixal_record,
        "attribute_input": {
            **frozen_pixal_record,
            "same_as_geometry_input": True,
        },
        "output": watertight_glb_record,
        "parameters": {
            **WATERTIGHT_PARAMETERS,
            "voxel_size": 0.01,
        },
        "topology": {
            "source": {
                "vertices": 128,
                "edges": 360,
                "faces": 240,
                "boundary_edges": 0,
                "wire_edges": 0,
                "nonmanifold_edges_over_two_faces": 0,
                "noncontiguous_two_face_edges": 0,
            },
            "after_voxel_remesh": {
                "vertices": 8,
                "edges": 18,
                "faces": 12,
                "boundary_edges": 0,
                "wire_edges": 0,
                "nonmanifold_edges_over_two_faces": 0,
                "noncontiguous_two_face_edges": 0,
            },
            "final": {
                "vertices": 8,
                "edges": 18,
                "faces": 12,
                "boundary_edges": 0,
                "wire_edges": 0,
                "nonmanifold_edges_over_two_faces": 0,
                "noncontiguous_two_face_edges": 0,
            },
        },
        "surface_attributes": {
            "backend": "bvh",
            "bvh_query_count": 36,
            "query_domain": "face_corner",
            "outward_ray_hit_count": 36,
            "nearest_fallback_count": 0,
            "outward_ray_offset": 0.01,
            "source_triangle_count": 240,
            "uv_layers": ["UVMap"],
            "color_attributes": [],
            "skipped_non_corner_color_attributes": [],
            "material_slots": ["FixtureMaterial"],
        },
        "torso_fold_repair": {
            "iterations": 0,
            "selected_vertices": 0,
            "policy": "disabled",
        },
        "authority_contract": {
            "attribute_source_pbr_material_reused": True,
            "attribute_source_uvs_transferred_by_nearest_surface": True,
            "attribute_source_pbr_baked_to_new_uv_atlas": False,
            "full_resolution_source_remains_geometry_authority": True,
            "source_geometry_replaced": True,
            "approved_skeleton_or_animation_touched": False,
        },
        "actual_faces": 12,
        "status": "research_candidate_pending_static_and_animation_qa",
        "formal_dataset_registration_authorized": False,
    }
    watertight_manifest_path = watertight_root / "watertight_manifest.json"
    watertight_manifest_record = _json_record(
        watertight_manifest_path, watertight_manifest
    )
    bound_heading = {
        **heading_authority,
        "input_glb_sha256": watertight_glb_record["sha256"],
        "review_artifact": _record(frozen_heading_review_path),
    }
    bound_heading_path = finalization_root / "bound_heading.json"
    bound_heading_record = _json_record(bound_heading_path, bound_heading)

    final_glb_path = finalization_root / "final.glb"
    _write_box_glb(
        final_glb_path,
        minimum=(-0.5, 0.0, -0.5),
        maximum=(0.5, 1.0, 0.5),
    )
    final_glb_record = _record(final_glb_path)
    counts = {
        "mesh_count": 1,
        "skin_count": 0,
        "armature_count": 0,
        "animation_count": 0,
        "vertex_count": 8,
        "face_count": 12,
        "material_count": 1,
        "image_count": 1,
    }
    finalization = {
        "schema": STATIC_FINALIZATION_SCHEMA,
        "created_at": "2026-07-27T00:02:00+00:00",
        "status": "passed_final_scaled_grounded_canonical_glb",
        "asset_class": "static_object",
        "instance_id": INSTANCE_ID,
        "request_sha256": REQUEST_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "input": watertight_glb_record,
        "output": final_glb_record,
        "coordinate_system": CANONICAL_COORDINATE_SYSTEM,
        "heading": {
            "passed": True,
            "reviewed_source_front_yaw_deg": 0.0,
            "target_front_axis": "positive-x",
            "applied_world_z_yaw_deg": 0.0,
            "evidence": bound_heading_record,
        },
        "physical_scale": {
            "passed": True,
            "measurement": "height_m",
            "height_before_m": 0.8,
            "target_height_m": 1.0,
            "tolerance_m": 0.01,
            "uniform_scale": 1.25,
            "readback_height_m": 1.0,
            "absolute_error_m": 0.0,
        },
        "grounding": {
            "passed": True,
            "method": "mesh_minimum_up_to_asset_root_zero_v1",
            "minimum_up_before_translation_m": 0.2,
            "minimum_up_after_export_readback_m": 0.0,
            "tolerance_m": 0.00001,
        },
        "scene_readback": {
            "before": counts,
            "after": counts,
            "bounds_minimum_blender_xyz_m": [-0.5, -0.5, 0.0],
            "bounds_maximum_blender_xyz_m": [0.5, 0.5, 1.0],
            "protected_scene_counts_preserved": True,
            "no_rig_or_animation": True,
        },
        "formal_dataset_registration_authorized": False,
    }
    finalization_path = finalization_root / "finalization.json"
    finalization_record = _json_record(finalization_path, finalization)

    bound_anchor = {
        "schema": ANCHOR_SPEC_SCHEMA,
        **{
            key: value
            for key, value in anchor_authority.items()
            if key not in {"schema", "input_glb_sha256"}
        },
        "review_evidence": _record(frozen_anchor_review_path),
        "finalized_glb_sha256": final_glb_record["sha256"],
        "finalization_manifest_sha256": finalization_record["sha256"],
    }
    bound_anchor_path = emitter_root / "bound_anchor.json"
    bound_anchor_record = _json_record(bound_anchor_path, bound_anchor)
    marker_glb_path = emitter_root / "marker.glb"
    _write_binary(marker_glb_path, b"emitter marker fixture glb\n")
    marker_glb_record = _record(marker_glb_path)
    measurement = {
        "schema": EMITTER_MEASUREMENT_SCHEMA,
        "created_at": "2026-07-27T00:03:00+00:00",
        "status": "measured_pending_marker_visual_review",
        "asset_class": "static_object",
        "instance_id": INSTANCE_ID,
        "request_sha256": REQUEST_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "input": final_glb_record,
        "finalization_manifest": finalization_record,
        "anchor_spec": bound_anchor_record,
        "coordinate_system": CANONICAL_COORDINATE_SYSTEM,
        "asset_bounds": {
            "minimum_m": [-0.5, 0.0, -0.5],
            "maximum_m": [0.5, 1.0, 0.5],
            "extent_m": [1.0, 1.0, 1.0],
        },
        "emitter_anchor": {
            "anchor_id": "speaker_surface",
            "anchor_type": "object_speaker",
            "semantic_role": "reviewed_speaker_surface",
            "offset_m": [0.5, 0.6, 0.05],
            "offset_space": "final_scaled_asset_root",
            "method": "reviewed_bbox_fraction_nearest_surface_v1",
            "aggregation": "weighted_centroid",
            "resolved_surface_samples": [
                {
                    "mesh_name": "fixture",
                    "triangle_index": 6,
                    "target_fraction_xyz": [0.8, 0.6, 0.55],
                    "weight": 1.0,
                    "surface_point_m": [0.5, 0.6, 0.05],
                    "reviewed_target_m": [0.3, 0.6, 0.05],
                    "target_to_surface_distance_m": 0.2,
                }
            ],
            "asset_specific_not_class_template": True,
            "animation_required": False,
        },
        "marker_review": {
            "marker_glb": marker_glb_record,
            "marker_radius_m": 0.02,
            "visual_review": "pending",
        },
        "formal_dataset_registration_authorized": False,
    }
    measurement_path = emitter_root / "measurement.json"
    measurement_record = _json_record(measurement_path, measurement)

    leaf_records = {
        "decision": _record(decision_path),
        "pixal_output": pixal_record,
        "heading_authority": heading_authority_record,
        "anchor_authority": anchor_authority_record,
        "bound_heading_evidence": bound_heading_record,
        "bound_anchor_spec": bound_anchor_record,
        "watertight_glb": watertight_glb_record,
        "finalized_glb": final_glb_record,
        "emitter_measurement": measurement_record,
        "emitter_marker_glb": marker_glb_record,
    }
    blender_path = root / "bin" / "blender"
    _write_binary(blender_path, b"fixture Blender executable bytes\n")
    command_tool_root = admission_root / ".runtime_commands" / "tools"
    stage_tool_paths = {
        "watertight": (
            command_tool_root
            / "blender_create_watertight_textured_proxy_mesh.py"
        ),
        "finalization": (
            command_tool_root
            / "blender_finalize_generated_static_object.py"
        ),
        "emitter_measurement": (
            command_tool_root
            / "blender_measure_generated_static_emitter.py"
        ),
    }
    for stage_name, tool_path in stage_tool_paths.items():
        _write_binary(
            tool_path,
            f"# frozen {stage_name} tool fixture\n".encode(),
        )
    emitter_contract_path = (
        command_tool_root / "generated_asset_emitter_contract.py"
    )
    _write_binary(
        emitter_contract_path,
        b"# frozen emitter contract fixture\n",
    )

    def stage_receipt(
        stage: str,
        directory: Path,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        validation: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        log_path = directory / f"{stage}.log"
        _write_binary(log_path, f"{stage} passed\n".encode())
        command = [
            str(blender_path.resolve()),
            "-b",
            "--python-exit-code",
            "2",
            "--python",
            str(stage_tool_paths[stage].resolve()),
            "--",
        ]
        if stage == "watertight":
            command.extend(
                [
                    "--source",
                    frozen_pixal_record["path"],
                    "--output",
                    outputs["watertight_glb"]["path"],
                    "--manifest",
                    outputs["watertight_manifest"]["path"],
                    "--voxel-resolution",
                    str(WATERTIGHT_PARAMETERS["voxel_resolution"]),
                    "--target-faces",
                    str(WATERTIGHT_PARAMETERS["target_faces"]),
                    "--smooth-iterations",
                    str(WATERTIGHT_PARAMETERS["smooth_iterations"]),
                    "--shrinkwrap-strength",
                    str(WATERTIGHT_PARAMETERS["shrinkwrap_strength"]),
                    "--post-shrinkwrap-smooth-iterations",
                    str(
                        WATERTIGHT_PARAMETERS[
                            "post_shrinkwrap_smooth_iterations"
                        ]
                    ),
                    "--torso-fold-repair-iterations",
                    str(
                        WATERTIGHT_PARAMETERS[
                            "torso_fold_repair_iterations"
                        ]
                    ),
                    "--attribute-transfer-backend",
                    str(
                        WATERTIGHT_PARAMETERS[
                            "attribute_transfer_backend"
                        ]
                    ),
                    "--bake-resolution",
                    str(WATERTIGHT_PARAMETERS["bake_resolution"]),
                    "--base-color-encoding-policy",
                    str(
                        WATERTIGHT_PARAMETERS[
                            "base_color_encoding_policy"
                        ]
                    ),
                    "--base-color-gain",
                    *(
                        str(item)
                        for item in WATERTIGHT_PARAMETERS["base_color_gain"]
                    ),
                ]
            )
        elif stage == "finalization":
            command.extend(
                [
                    "--input-glb",
                    inputs["watertight_glb"]["path"],
                    "--watertight-manifest",
                    inputs["watertight_manifest"]["path"],
                    "--static-decision",
                    str(
                        (
                            admission_root
                            / ".runtime_inputs"
                            / INSTANCE_ID
                            / "decision.json"
                        ).resolve()
                    ),
                    "--heading-evidence",
                    inputs["bound_heading_evidence"]["path"],
                    "--output",
                    outputs["finalized_glb"]["path"],
                    "--manifest",
                    outputs["finalization_manifest"]["path"],
                ]
            )
        else:
            command.extend(
                [
                    "--input-glb",
                    inputs["finalized_glb"]["path"],
                    "--finalization-manifest",
                    inputs["finalization_manifest"]["path"],
                    "--anchor-spec",
                    inputs["bound_anchor_spec"]["path"],
                    "--output",
                    outputs["emitter_measurement"]["path"],
                    "--marker-glb",
                    outputs["marker_glb"]["path"],
                ]
            )
        command_dependencies = (
            {}
            if stage == "watertight"
            else {
                "generated_asset_emitter_contract.py": _record(
                    emitter_contract_path
                )
            }
        )
        command_input_manifest = _seal(
            {
                "schema": (
                    "avengine_controlled_static_object_"
                    "command_input_manifest_v1"
                ),
                "instance_id": INSTANCE_ID,
                "stage": stage,
                "command": command,
                "command_sha256": canonical_json_sha256(command),
                "blender": {
                    "configured_path": str(blender_path.resolve()),
                    "resolved_path": str(blender_path.resolve()),
                    "record": _record(blender_path),
                },
                "python_tool": _record(stage_tool_paths[stage]),
                "python_dependencies": command_dependencies,
                "formal_dataset_registration_authorized": False,
            },
            "manifest_sha256",
        )
        command_input_manifest_path = (
            directory / "command_input_manifest.json"
        )
        command_input_manifest_record = _json_record(
            command_input_manifest_path,
            command_input_manifest,
        )
        value = _seal(
            {
                "schema": STATIC_ADMISSION_STAGE_RECEIPT_SCHEMA,
                "instance_id": INSTANCE_ID,
                "stage": stage,
                "status": "passed",
                "command": command,
                "command_sha256": canonical_json_sha256(command),
                "command_input_manifest": command_input_manifest_record,
                "execution": {
                    "started_at": "2026-07-27T00:00:00+00:00",
                    "finished_at": "2026-07-27T00:00:01+00:00",
                    "wall_seconds": 1.0,
                    "returncode": 0,
                    "timeout_seconds": 300,
                    "error": None,
                },
                "log": _record(log_path, relative_to=admission_root),
                "inputs": inputs,
                "outputs": outputs,
                "validation": validation,
                "formal_dataset_registration_authorized": False,
            },
            "receipt_sha256",
        )
        path = directory / "stage_receipt.json"
        return path, _json_record(path, value, relative_to=admission_root)

    watertight_stage_path, watertight_stage_record = stage_receipt(
        "watertight",
        watertight_root,
        {
            "decision": leaf_records["decision"],
            "pixal_output": pixal_record,
            "watertight_parameters": WATERTIGHT_PARAMETERS,
        },
        {
            "watertight_glb": watertight_glb_record,
            "watertight_manifest": watertight_manifest_record,
        },
        {
            "boundary_edges": 0,
            "nonmanifold_edges_over_two_faces": 0,
            "no_rig_or_animation": True,
        },
    )
    finalization_stage_path, finalization_stage_record = stage_receipt(
        "finalization",
        finalization_root,
        {
            "watertight_glb": watertight_glb_record,
            "watertight_manifest": watertight_manifest_record,
            "heading_authority": heading_authority_record,
            "bound_heading_evidence": bound_heading_record,
        },
        {
            "finalized_glb": final_glb_record,
            "finalization_manifest": finalization_record,
        },
        {
            "heading_passed": True,
            "physical_scale_passed": True,
            "grounding_passed": True,
            "no_rig_or_animation": True,
        },
    )
    emitter_stage_path, emitter_stage_record = stage_receipt(
        "emitter_measurement",
        emitter_root,
        {
            "finalized_glb": final_glb_record,
            "finalization_manifest": finalization_record,
            "anchor_authority": anchor_authority_record,
            "bound_anchor_spec": bound_anchor_record,
        },
        {
            "emitter_measurement": measurement_record,
            "marker_glb": marker_glb_record,
        },
        {"marker_visual_review": "pending"},
    )

    job_receipt = _seal(
        {
            "schema": STATIC_ADMISSION_JOB_RECEIPT_SCHEMA,
            "status": "passed_pending_emitter_marker_review",
            "asset_class": "static_object",
            "route": STATIC_ROUTE,
            "instance_id": INSTANCE_ID,
            "request_sha256": REQUEST_SHA256,
            "profile_sha256": PROFILE_SHA256,
            **leaf_records,
            "stage_receipts": {
                "watertight": watertight_stage_record,
                "finalization": finalization_stage_record,
                "emitter_measurement": emitter_stage_record,
            },
            "path_rebinding": {
                "policy": "staging_to_atomic_public_root_paths_only_v1",
                "staging_root": "/fixture/deleted/staging",
                "public_root": str(admission_root.resolve()),
                "hashes_before": {
                    "watertight_manifest": "8" * 64,
                    "finalization_manifest": "8" * 64,
                    "bound_anchor": "8" * 64,
                    "emitter_measurement": "8" * 64,
                },
                "hashes_after": {
                    "watertight_manifest": watertight_manifest_record["sha256"],
                    "finalization_manifest": finalization_record["sha256"],
                    "bound_anchor": bound_anchor_record["sha256"],
                    "emitter_measurement": measurement_record["sha256"],
                },
                "semantic_authority_fields_changed": False,
            },
            "marker_review": "pending",
            "next_gate": "emitter_marker_visual_review",
            "formal_dataset_registration_authorized": False,
        },
        "receipt_sha256",
    )
    job_receipt_path = job_root / "job_receipt.json"
    job_receipt_record = _json_record(
        job_receipt_path, job_receipt, relative_to=admission_root
    )
    admission_batch = _seal(
        {
            "schema": STATIC_ADMISSION_BATCH_SCHEMA,
            "status": "passed_all_instances_pending_emitter_marker_review",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "asset_class": "static_object",
            "route": STATIC_ROUTE,
            "decision_batch": decision_batch_record,
            "plan": plan_record,
            "job_count": 1,
            "passed_count": 1,
            "failed_count": 0,
            "jobs": [
                {
                    "instance_id": INSTANCE_ID,
                    "request_sha256": REQUEST_SHA256,
                    "profile_sha256": PROFILE_SHA256,
                    "status": "passed_pending_emitter_marker_review",
                    "receipt_sha256": job_receipt["receipt_sha256"],
                    "job_receipt": job_receipt_record,
                    "marker_review": "pending",
                }
            ],
            "marker_review": {
                "status": "pending",
                "next_gate": "emitter_marker_visual_review",
            },
            "automatic_checks": {
                "all_decisions_and_plan_authorities_reauthenticated": True,
                "all_watertight_boundary_and_nonmanifold_gates_passed": True,
                "all_heading_scale_and_grounding_gates_passed": True,
                "all_static_emitter_measurements_hash_bound": True,
                "all_marker_reviews_pending": True,
                "fail_first_no_partial_batch_published": True,
                "no_formal_registration_authorized": True,
                "overall": "passed",
            },
        },
        "batch_sha256",
    )
    admission_batch_path = (
        admission_root / "static_object_admission_batch_manifest.json"
    )
    _json_record(admission_batch_path, admission_batch)

    approval = _seal(
        {
            "schema": MARKER_VISUAL_APPROVAL_SCHEMA,
            "status": "completed",
            "scope": "emitter_marker_placement_only",
            "asset_class": "static_object",
            "instance_id": INSTANCE_ID,
            "request_sha256": REQUEST_SHA256,
            "profile_sha256": PROFILE_SHA256,
            "finalized_glb": final_glb_record,
            "emitter_measurement": measurement_record,
            "marker_glb": marker_glb_record,
            "reviewer_kind": "human",
            "review_mode": "visual",
            "verdict": "approved",
            "reviewer_id": "fixture_reviewer",
            "reviewed_at": "2026-07-27T00:04:00+00:00",
            "notes": "Marker is visibly on the reviewed speaker surface.",
            "formal_dataset_registration_authorized": False,
        },
        "approval_content_sha256",
    )
    approval_path = root / "marker_visual_approval.json"
    _json_record(approval_path, approval)

    base_registry = root / "base_registry.json"
    _base_registry(base_registry)
    return {
        "root": root,
        "admission_root": admission_root,
        "base_registry": base_registry,
        "batch": admission_batch_path,
        "approval": approval_path,
        "pixal_glb": pixal_glb_path,
        "pixal_batch": pixal_batch_path,
        "pixal_inputs": selected_pixal_inputs_path,
        "base_pixal_inputs": pixal_inputs_path,
        "source_png": source_path,
        "rgba_png": rgba_path,
        "alpha_png": alpha_path,
        "one_shot_policy": one_shot_policy_path,
        "isnet_model": model_path,
        "isnet_configured_python": configured_python_path,
        "isnet_resolved_python": resolved_python_path,
        "isnet_worker_source": worker_source_path,
        "isnet_worker_executed": worker_executed_path,
        "isnet_jobs": isnet_jobs_path,
        "isnet_status": isnet_status_path,
        "isnet_log": isnet_log_path,
        "flux_preflight": preflight_path,
        "flux_candidate_manifest": flux_candidate_manifest_path,
        "flux_review_decisions": flux_review_decisions_path,
        "flux_batch": flux_batch_path,
        "flux_review_batch": flux_review_batch_path,
        "review": review_path,
        "review_batch": review_batch_path,
        "decision": decision_path,
        "decision_batch": decision_batch_path,
        "watertight_glb": watertight_glb_path,
        "watertight_manifest": watertight_manifest_path,
        "final_glb": final_glb_path,
        "finalization_manifest": finalization_path,
        "measurement": measurement_path,
        "marker_glb": marker_glb_path,
        "watertight_stage": watertight_stage_path,
        "finalization_stage": finalization_stage_path,
        "emitter_stage": emitter_stage_path,
        "blender_binary": blender_path,
        "watertight_command_tool": stage_tool_paths["watertight"],
        "emitter_contract_tool": emitter_contract_path,
        "job_receipt": job_receipt_path,
    }


def _validate(paths: dict[str, Any]):
    return validate_static_object_admission(
        admission_batch_path=paths["batch"],
        instance_id=INSTANCE_ID,
        marker_visual_approval_path=paths["approval"],
        workspace_roots=[paths["root"]],
    )


def _validate_isnet_receipt(
    paths: dict[str, Any],
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    manifest = load_json(paths["base_pixal_inputs"])
    if mutate is not None:
        mutate(manifest["isnet"])
    policy = static_objects.WorkspacePathPolicy.from_roots([paths["root"]])
    artifact = static_objects._file_record(
        _record(paths["base_pixal_inputs"]),
        policy,
        owner="test base Pixal input manifest",
    )
    static_objects._validate_isnet_execution_receipt(
        manifest["isnet"],
        policy,
        artifact=artifact,
        pixal_jobs=manifest["jobs"],
        segmentations=manifest["segmentations"],
    )


def _publish(
    paths: dict[str, Any],
    output: Path,
    revision: str = "run_v1",
) -> Path:
    return publish_static_object_entity_registry(
        base_registry_path=paths["base_registry"],
        admission_batch_path=paths["batch"],
        instance_id=INSTANCE_ID,
        marker_visual_approval_path=paths["approval"],
        output_path=output,
        registry_revision=revision,
        workspace_roots=[paths["root"]],
    )


def _rewrite_measurement_closure(
    paths: dict[str, Any],
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    measurement = load_json(paths["measurement"])
    mutate(measurement)
    measurement_record = _json_record(paths["measurement"], measurement)

    stage = load_json(paths["emitter_stage"])
    stage["outputs"]["emitter_measurement"] = measurement_record
    stage.pop("receipt_sha256")
    _seal(stage, "receipt_sha256")
    stage_record = _json_record(
        paths["emitter_stage"],
        stage,
        relative_to=paths["admission_root"],
    )

    job = load_json(paths["job_receipt"])
    job["emitter_measurement"] = measurement_record
    job["stage_receipts"]["emitter_measurement"] = stage_record
    job["path_rebinding"]["hashes_after"][
        "emitter_measurement"
    ] = measurement_record["sha256"]
    job.pop("receipt_sha256")
    _seal(job, "receipt_sha256")
    job_record = _json_record(
        paths["job_receipt"],
        job,
        relative_to=paths["admission_root"],
    )

    batch = load_json(paths["batch"])
    batch["jobs"][0]["receipt_sha256"] = job["receipt_sha256"]
    batch["jobs"][0]["job_receipt"] = job_record
    batch.pop("batch_sha256")
    _seal(batch, "batch_sha256")
    _json_record(paths["batch"], batch)

    approval = load_json(paths["approval"])
    approval["emitter_measurement"] = measurement_record
    approval.pop("approval_content_sha256")
    _seal(approval, "approval_content_sha256")
    _json_record(paths["approval"], approval)


def _rewrite_stage_command_closure(
    paths: dict[str, Any],
    *,
    stage_path_key: str,
    stage_name: str,
    mutate: Callable[[list[str]], None],
) -> None:
    stage_path = Path(paths[stage_path_key])
    stage = load_json(stage_path)
    manifest_path = Path(stage["command_input_manifest"]["path"])
    manifest = load_json(manifest_path)
    command = list(stage["command"])
    mutate(command)

    manifest["command"] = command
    manifest["command_sha256"] = canonical_json_sha256(command)
    manifest.pop("manifest_sha256")
    _seal(manifest, "manifest_sha256")
    manifest_record = _json_record(manifest_path, manifest)

    stage["command"] = command
    stage["command_sha256"] = canonical_json_sha256(command)
    stage["command_input_manifest"] = manifest_record
    stage.pop("receipt_sha256")
    _seal(stage, "receipt_sha256")
    stage_record = _json_record(
        stage_path,
        stage,
        relative_to=paths["admission_root"],
    )

    job = load_json(paths["job_receipt"])
    job["stage_receipts"][stage_name] = stage_record
    job.pop("receipt_sha256")
    _seal(job, "receipt_sha256")
    job_record = _json_record(
        paths["job_receipt"],
        job,
        relative_to=paths["admission_root"],
    )

    batch = load_json(paths["batch"])
    batch["jobs"][0]["receipt_sha256"] = job["receipt_sha256"]
    batch["jobs"][0]["job_receipt"] = job_record
    batch.pop("batch_sha256")
    _seal(batch, "batch_sha256")
    _json_record(paths["batch"], batch)


def _published_entity(paths: dict[str, Any], output: Path) -> dict[str, Any]:
    registry = load_entity_asset_registry(_publish(paths, output))
    return next(
        item
        for item in registry["entities"]
        if item["entity_asset_id"] == INSTANCE_ID
    )


def test_complete_receipt_chain_publishes_and_resolves_list_and_ndarray(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "workspace")
    output = Path(paths["root"]) / "published.json"
    entity = _published_entity(paths, output)
    assert entity["entity_class"] == "rigid_object"
    assert entity["admission_state"] == "research"
    assert entity["capabilities"] == {
        "articulated": False,
        "skeleton_revision": None,
        "skeleton_sha256": None,
        "action_ids": [],
    }
    assert tuple(
        item["kind"] for item in entity["admission_evidence"]["artifacts"]
    ) == STATIC_OBJECT_EVIDENCE_KINDS
    assert entity["revision"] == (
        "spear_static_"
        + entity["admission_evidence"]["evidence_content_sha256"]
    )

    registry = load_entity_asset_registry(output)
    matrix = [
        [0.0, 0.0, 1.0, 10.0],
        [0.0, 1.0, 0.0, 2.0],
        [-1.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    expected = (10.05, 2.6, 2.5)
    assert resolve_static_object_emitter_world(
        registry,
        entity_asset_id=INSTANCE_ID,
        entity_revision=entity["revision"],
        world_from_asset=matrix,
    ) == pytest.approx(expected)
    assert resolve_static_object_emitter_world(
        registry,
        entity_asset_id=INSTANCE_ID,
        entity_revision=entity["revision"],
        anchor_id="speaker_surface",
        world_from_asset=np.asarray(matrix, dtype=np.float64),
    ) == pytest.approx(expected)
    verified = verify_static_object_entity_registry(
        registry_path=output,
        entity_asset_id=INSTANCE_ID,
        entity_revision=entity["revision"],
        workspace_roots=[paths["root"]],
    )
    assert verified == entity


def test_combined_pixal_input_reauthenticates_selected_parent_lineage(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(
        tmp_path / "combined",
        combined_pixal_inputs=True,
    )
    admission = _validate(paths)
    assert admission.instance_id == INSTANCE_ID

    Path(paths["base_pixal_inputs"]).write_bytes(
        Path(paths["base_pixal_inputs"]).read_bytes() + b"\ntampered\n"
    )
    with pytest.raises(StaticObjectRegistrationError):
        _validate(paths)


def test_static_preflight_job_is_the_controlled_request_authority(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "preflight")
    preflight = load_json(paths["flux_preflight"])
    controlled = load_json(paths["base_pixal_inputs"])["jobs"][0][
        "controlled_request"
    ]
    preflight["routes"][STATIC_ROUTE][0]["generation_plan"][
        "generation_seed"
    ] = 43
    preflight.pop("preflight_sha256")
    _seal(preflight, "preflight_sha256")
    with pytest.raises(
        StaticObjectRegistrationError,
        match="controlled request differs",
    ):
        static_objects._validate_selected_static_preflight_job(
            preflight,
            instance_id=INSTANCE_ID,
            controlled=controlled,
        )


def test_resealed_2d_review_cannot_override_rejected_manual_decision(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "decisions")
    decisions = load_json(paths["flux_review_decisions"])
    decisions["decisions"][0]["category_identity"] = "rejected"
    decisions["decisions"][0]["decision"] = "rejected"
    decisions_record = _json_record(
        paths["flux_review_decisions"],
        decisions,
    )

    review_batch = load_json(paths["flux_review_batch"])
    review_batch["decisions_input"] = decisions_record
    review_batch.pop("review_batch_sha256")
    _seal(review_batch, "review_batch_sha256")
    _json_record(paths["flux_review_batch"], review_batch)
    review_batch_record = _path_hash_record(
        paths["flux_review_batch"],
        content_hash_field="review_batch_sha256",
        content_hash=review_batch["review_batch_sha256"],
    )

    base_inputs = load_json(paths["base_pixal_inputs"])
    selected_job = base_inputs["jobs"][0]
    policy = static_objects.WorkspacePathPolicy.from_roots([paths["root"]])
    source = static_objects._file_record(
        selected_job["reference"]["source"],
        policy,
        owner="test approved FLUX source",
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="decision counts/status changed",
    ):
        static_objects._validate_static_2d_review_lineage(
            review_batch_record,
            policy,
            instance_id=INSTANCE_ID,
            controlled=selected_job["controlled_request"],
            source=source,
            expected_flux_batch_sha256=base_inputs[
                "upstream_flux_one_shot_evidence"
            ]["flux_batch_sha256"],
            expected_one_shot_policy=base_inputs["one_shot_execution"][
                "policy"
            ],
        )


def test_isnet_rgba_rgb_must_equal_approved_flux_candidate(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "rgba")
    with Image.open(paths["alpha_png"]) as alpha:
        alpha.load()
        changed = Image.new("RGBA", alpha.size, (10, 20, 30, 255))
        changed.putalpha(alpha)
        changed.save(paths["rgba_png"], format="PNG")
    policy = static_objects.WorkspacePathPolicy.from_roots([paths["root"]])
    source = static_objects._file_record(
        _record(paths["source_png"]),
        policy,
        owner="test FLUX source",
    )
    rgba = static_objects._file_record(
        _record(paths["rgba_png"]),
        policy,
        owner="test ISNet RGBA",
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="RGB channels differ",
    ):
        static_objects._validate_rgba_source_rgb_binding(
            source,
            rgba,
            owner="test ISNet RGBA",
        )


def test_one_shot_policy_record_rehashes_policy_file(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "policy")
    Path(paths["one_shot_policy"]).write_bytes(b"replaced policy bytes\n")
    with pytest.raises(
        StaticObjectRegistrationError,
        match="policy file SHA-256 mismatch",
    ):
        _validate(paths)


def test_resealed_weakened_one_shot_policy_is_rejected(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "weakened_policy")
    payload = load_json(paths["one_shot_policy"])
    payload["per_request_cardinality"]["seed_retry_allowed"] = True
    _json_record(paths["one_shot_policy"], payload)
    manifest = load_json(paths["base_pixal_inputs"])
    policy_record = manifest["one_shot_execution"]["policy"]
    policy_record["sha256"] = _record(paths["one_shot_policy"])["sha256"]
    workspace_policy = static_objects.WorkspacePathPolicy.from_roots(
        [paths["root"]]
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="weakens the one-shot admission policy",
    ):
        static_objects._validate_one_shot_policy_record(
            policy_record,
            workspace_policy,
            owner="test one-shot policy",
        )


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("per_request_cardinality", "flux_invocations"),
        ("profile_qualification", "required_pass_fraction"),
    ),
)
def test_one_shot_policy_rejects_boolean_numeric_fields(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    paths = _admission_fixture(tmp_path / f"boolean_{field}")
    payload = load_json(paths["one_shot_policy"])
    payload[section][field] = True
    _json_record(paths["one_shot_policy"], payload)
    manifest = load_json(paths["base_pixal_inputs"])
    policy_record = manifest["one_shot_execution"]["policy"]
    policy_record["sha256"] = _record(paths["one_shot_policy"])["sha256"]
    workspace_policy = static_objects.WorkspacePathPolicy.from_roots(
        [paths["root"]]
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="weakens the one-shot admission policy",
    ):
        static_objects._validate_one_shot_policy_record(
            policy_record,
            workspace_policy,
            owner="test boolean one-shot policy",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("invocation_ordinal", False),
        ("invocations_allowed", True),
    ),
)
def test_one_shot_stage_rejects_boolean_numeric_fields(
    tmp_path: Path,
    field: str,
    value: bool,
) -> None:
    paths = _admission_fixture(tmp_path / f"boolean_{field}")
    manifest = load_json(paths["base_pixal_inputs"])
    record = manifest["one_shot_execution"]
    record[field] = value
    workspace_policy = static_objects.WorkspacePathPolicy.from_roots(
        [paths["root"]]
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="one-shot execution contract changed",
    ):
        static_objects._validate_one_shot_stage_record(
            record,
            workspace_policy,
            stage="pixal3d",
            owner="test boolean one-shot stage",
        )


def test_legacy_isnet_string_receipt_is_rejected(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "legacy_isnet")

    def use_legacy_receipt(receipt: dict[str, Any]) -> None:
        receipt.clear()
        receipt.update(
            {
                "model_path": "/fixture/models/isnet-general-use.onnx",
                "model_sha256": "6" * 64,
                "worker": "/fixture/worker.py",
                "python": "/fixture/python",
                "status": {"fixture": True},
                "log": {"fixture": True},
            }
        )

    with pytest.raises(
        StaticObjectRegistrationError,
        match="execution receipt changed",
    ):
        _validate_isnet_receipt(paths, use_legacy_receipt)


@pytest.mark.parametrize(
    "artifact_key",
    (
        "isnet_model",
        "isnet_resolved_python",
        "isnet_worker_source",
        "isnet_jobs",
    ),
)
def test_isnet_runtime_inputs_are_rehashed(
    tmp_path: Path,
    artifact_key: str,
) -> None:
    paths = _admission_fixture(tmp_path / artifact_key)
    path = Path(paths[artifact_key])
    path.write_bytes(path.read_bytes() + b"tampered\n")
    with pytest.raises(
        StaticObjectRegistrationError,
        match="SHA-256 mismatch",
    ):
        _validate_isnet_receipt(paths)


def test_isnet_configured_python_symlink_target_is_bound(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "python_symlink")
    configured = Path(paths["isnet_configured_python"])
    resolved = Path(paths["isnet_resolved_python"])
    replacement = resolved.with_name("python3.10.same-bytes")
    replacement.write_bytes(resolved.read_bytes())
    replacement.chmod(0o755)
    configured.unlink()
    configured.symlink_to(replacement.name)
    with pytest.raises(
        StaticObjectRegistrationError,
        match="configured/resolved ISNet Python identity changed",
    ):
        _validate_isnet_receipt(paths)


def test_resealed_isnet_command_must_match_exact_runtime_inputs(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "isnet_command")

    def append_argument(receipt: dict[str, Any]) -> None:
        receipt["command"].append("--changed")
        receipt["command_sha256"] = canonical_json_sha256(receipt["command"])

    with pytest.raises(
        StaticObjectRegistrationError,
        match="command execution receipt changed",
    ):
        _validate_isnet_receipt(paths, append_argument)


def test_resealed_isnet_status_is_semantically_revalidated(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "isnet_status")
    status = load_json(paths["isnet_status"])
    status["passed_count"] = 2
    _json_record(paths["isnet_status"], status)

    def update_status_record(receipt: dict[str, Any]) -> None:
        receipt["status"] = _record(
            paths["isnet_status"],
            relative_to=Path(paths["base_pixal_inputs"]).parent,
        )

    with pytest.raises(
        StaticObjectRegistrationError,
        match="status summary changed",
    ):
        _validate_isnet_receipt(paths, update_status_record)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("passed_count", "status summary changed"),
        ("alpha_minimum", "status job differs from sealed outputs"),
    ),
)
def test_isnet_status_rejects_boolean_numeric_fields(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    paths = _admission_fixture(tmp_path / f"isnet_boolean_{field}")
    status = load_json(paths["isnet_status"])
    if field == "passed_count":
        status["passed_count"] = True
    else:
        status["jobs"][0]["alpha_extrema"][0] = False
    _json_record(paths["isnet_status"], status)

    def update_status_record(receipt: dict[str, Any]) -> None:
        receipt["status"] = _record(
            paths["isnet_status"],
            relative_to=Path(paths["base_pixal_inputs"]).parent,
        )

    with pytest.raises(StaticObjectRegistrationError, match=message):
        _validate_isnet_receipt(paths, update_status_record)


@pytest.mark.parametrize(
    ("gate", "message"),
    (
        ("normal", "primitive attributes are not rigid core PBR"),
        ("position_bounds", "accessor bounds differ from its values"),
    ),
)
def test_direct_glb_readback_rejects_unbound_vertex_semantics(
    tmp_path: Path,
    gate: str,
    message: str,
) -> None:
    root = tmp_path / "glb"
    root.mkdir()
    glb_path = root / f"invalid_{gate}.glb"

    def break_vertex_semantics(document: dict[str, Any]) -> None:
        if gate == "normal":
            del document["meshes"][0]["primitives"][0]["attributes"]["NORMAL"]
        else:
            document["accessors"][0]["min"][0] = -0.25

    _write_box_glb(
        glb_path,
        minimum=(-0.5, 0.0, -0.5),
        maximum=(0.5, 1.0, 0.5),
        mutate_document=break_vertex_semantics,
    )
    policy = static_objects.WorkspacePathPolicy.from_roots([root])
    artifact = static_objects._file_record(
        _record(glb_path),
        policy,
        owner=f"test invalid-{gate} GLB",
    )
    with pytest.raises(StaticObjectRegistrationError, match=message):
        static_objects._read_glb_geometry(
            artifact,
            owner=f"test invalid-{gate} GLB",
        )


def test_direct_glb_readback_rejects_external_texture_uri(
    tmp_path: Path,
) -> None:
    root = tmp_path / "glb"
    root.mkdir()
    glb_path = root / "external_texture.glb"
    _write_box_glb(
        glb_path,
        minimum=(-0.5, 0.0, -0.5),
        maximum=(0.5, 1.0, 0.5),
        image_uri="external_texture.png",
    )
    policy = static_objects.WorkspacePathPolicy.from_roots([root])
    artifact = static_objects._file_record(
        _record(glb_path),
        policy,
        owner="test external-texture GLB",
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="not self-contained",
    ):
        static_objects._read_glb_geometry(
            artifact,
            owner="test external-texture GLB",
        )


def test_direct_glb_readback_rejects_nodes_outside_default_scene(
    tmp_path: Path,
) -> None:
    root = tmp_path / "glb"
    root.mkdir()
    glb_path = root / "off_scene.glb"

    def add_off_scene_mesh(document: dict[str, Any]) -> None:
        document["nodes"].append(
            {
                "name": "hidden_emitter_surface",
                "mesh": 0,
                "translation": [100.0, 0.0, 0.0],
            }
        )

    _write_box_glb(
        glb_path,
        minimum=(-0.5, 0.0, -0.5),
        maximum=(0.5, 1.0, 0.5),
        mutate_document=add_off_scene_mesh,
    )
    policy = static_objects.WorkspacePathPolicy.from_roots([root])
    artifact = static_objects._file_record(
        _record(glb_path),
        policy,
        owner="test off-scene GLB",
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="outside its default scene",
    ):
        static_objects._read_glb_geometry(
            artifact,
            owner="test off-scene GLB",
        )


def test_direct_glb_readback_rejects_non_integer_child_index(
    tmp_path: Path,
) -> None:
    root = tmp_path / "glb"
    root.mkdir()
    glb_path = root / "boolean_child.glb"

    def use_boolean_child(document: dict[str, Any]) -> None:
        mesh_node = document["nodes"][0]
        document["nodes"] = [
            {"name": "root", "children": [True]},
            mesh_node,
        ]
        document["scenes"][0]["nodes"] = [0]

    _write_box_glb(
        glb_path,
        minimum=(-0.5, 0.0, -0.5),
        maximum=(0.5, 1.0, 0.5),
        mutate_document=use_boolean_child,
    )
    policy = static_objects.WorkspacePathPolicy.from_roots([root])
    artifact = static_objects._file_record(
        _record(glb_path),
        policy,
        owner="test boolean-child GLB",
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="node graph is invalid",
    ):
        static_objects._read_glb_geometry(
            artifact,
            owner="test boolean-child GLB",
        )


@pytest.mark.parametrize(
    ("morph_site", "message"),
    (
        ("node", "mesh node has rig or morph weights"),
        ("mesh", "mesh has morph weights"),
        ("primitive", "primitive has morph targets"),
    ),
)
def test_direct_glb_readback_rejects_default_morph_geometry(
    tmp_path: Path,
    morph_site: str,
    message: str,
) -> None:
    root = tmp_path / "glb"
    root.mkdir()
    glb_path = root / "morphed.glb"

    def add_default_morph(document: dict[str, Any]) -> None:
        if morph_site == "node":
            document["nodes"][0]["weights"] = [1.0]
        elif morph_site == "mesh":
            document["meshes"][0]["weights"] = [1.0]
        else:
            document["meshes"][0]["primitives"][0]["targets"] = [
                {"POSITION": 0}
            ]

    _write_box_glb(
        glb_path,
        minimum=(-0.5, 0.0, -0.5),
        maximum=(0.5, 1.0, 0.5),
        mutate_document=add_default_morph,
    )
    policy = static_objects.WorkspacePathPolicy.from_roots([root])
    artifact = static_objects._file_record(
        _record(glb_path),
        policy,
        owner="test morphed GLB",
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match=message,
    ):
        static_objects._read_glb_geometry(
            artifact,
            owner="test morphed GLB",
        )


@pytest.mark.parametrize(
    ("extension_site", "message"),
    (
        ("root", "uses an unsupported GLB extension"),
        ("scene", "default-scene extensions are unsupported"),
        ("node", "node extensions are unsupported"),
        ("mesh", "mesh has morph weights or extensions"),
        ("primitive", "primitive extensions are unsupported"),
        ("image", "image extensions are unsupported"),
        ("accessor", "accessor extensions/sparse data are unsupported"),
        ("pbr", "base PBR extensions are unsupported"),
    ),
)
def test_direct_glb_readback_rejects_all_extension_payloads(
    tmp_path: Path,
    extension_site: str,
    message: str,
) -> None:
    root = tmp_path / "glb"
    root.mkdir()
    glb_path = root / f"{extension_site}_extension.glb"

    def add_extension(document: dict[str, Any]) -> None:
        extension = {"EXT_fixture_semantics": {"enabled": True}}
        if extension_site == "root":
            document["extensionsUsed"] = ["EXT_fixture_semantics"]
        elif extension_site == "scene":
            document["scenes"][0]["extensions"] = extension
        elif extension_site == "node":
            document["nodes"][0]["extensions"] = extension
        elif extension_site == "mesh":
            document["meshes"][0]["extensions"] = extension
        elif extension_site == "primitive":
            document["meshes"][0]["primitives"][0]["extensions"] = extension
        elif extension_site == "image":
            document["images"][0]["extensions"] = extension
        elif extension_site == "accessor":
            document["accessors"][0]["extensions"] = extension
        else:
            document["materials"][0]["pbrMetallicRoughness"][
                "extensions"
            ] = extension

    _write_box_glb(
        glb_path,
        minimum=(-0.5, 0.0, -0.5),
        maximum=(0.5, 1.0, 0.5),
        mutate_document=add_extension,
    )
    policy = static_objects.WorkspacePathPolicy.from_roots([root])
    artifact = static_objects._file_record(
        _record(glb_path),
        policy,
        owner=f"test {extension_site}-extension GLB",
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match=message,
    ):
        static_objects._read_glb_geometry(
            artifact,
            owner=f"test {extension_site}-extension GLB",
        )


def test_direct_glb_readback_requires_material_texture_image_chain(
    tmp_path: Path,
) -> None:
    root = tmp_path / "glb"
    root.mkdir()
    glb_path = root / "flat_material.glb"

    def remove_base_color_texture(document: dict[str, Any]) -> None:
        del document["materials"][0]["pbrMetallicRoughness"][
            "baseColorTexture"
        ]

    _write_box_glb(
        glb_path,
        minimum=(-0.5, 0.0, -0.5),
        maximum=(0.5, 1.0, 0.5),
        mutate_document=remove_base_color_texture,
    )
    policy = static_objects.WorkspacePathPolicy.from_roots([root])
    artifact = static_objects._file_record(
        _record(glb_path),
        policy,
        owner="test flat-material GLB",
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="baseColorTexture must be a JSON object",
    ):
        static_objects._read_glb_geometry(
            artifact,
            owner="test flat-material GLB",
        )


def test_direct_glb_readback_rejects_nonfinite_pbr_values(
    tmp_path: Path,
) -> None:
    root = tmp_path / "glb"
    root.mkdir()
    glb_path = root / "nonfinite_pbr.glb"

    def inject_nan_metallic(document: dict[str, Any]) -> None:
        document["materials"][0]["pbrMetallicRoughness"][
            "metallicFactor"
        ] = float("nan")

    _write_box_glb(
        glb_path,
        minimum=(-0.5, 0.0, -0.5),
        maximum=(0.5, 1.0, 0.5),
        mutate_document=inject_nan_metallic,
    )
    policy = static_objects.WorkspacePathPolicy.from_roots([root])
    artifact = static_objects._file_record(
        _record(glb_path),
        policy,
        owner="test nonfinite-PBR GLB",
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="metallicFactor must be a finite number",
    ):
        static_objects._read_glb_geometry(
            artifact,
            owner="test nonfinite-PBR GLB",
        )


def test_direct_glb_readback_rejects_out_of_range_pbr_values(
    tmp_path: Path,
) -> None:
    root = tmp_path / "glb"
    root.mkdir()
    glb_path = root / "out_of_range_pbr.glb"

    def inject_invalid_alpha(document: dict[str, Any]) -> None:
        document["materials"][0]["alphaMode"] = "EXECUTE"

    _write_box_glb(
        glb_path,
        minimum=(-0.5, 0.0, -0.5),
        maximum=(0.5, 1.0, 0.5),
        mutate_document=inject_invalid_alpha,
    )
    policy = static_objects.WorkspacePathPolicy.from_roots([root])
    artifact = static_objects._file_record(
        _record(glb_path),
        policy,
        owner="test invalid-PBR GLB",
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="material PBR values are invalid",
    ):
        static_objects._read_glb_geometry(
            artifact,
            owner="test invalid-PBR GLB",
        )


def test_direct_glb_readback_rejects_declared_mime_byte_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "glb"
    root.mkdir()
    glb_path = root / "wrong_mime.glb"
    png_payload = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
        "AAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    _write_box_glb(
        glb_path,
        minimum=(-0.5, 0.0, -0.5),
        maximum=(0.5, 1.0, 0.5),
        image_uri=f"data:image/jpeg;base64,{png_payload}",
    )
    policy = static_objects.WorkspacePathPolicy.from_roots([root])
    artifact = static_objects._file_record(
        _record(glb_path),
        policy,
        owner="test wrong-MIME GLB",
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="MIME differs from decoded bytes",
    ):
        static_objects._read_glb_geometry(
            artifact,
            owner="test wrong-MIME GLB",
        )


def test_direct_glb_readback_rejects_extensionless_webp(
    tmp_path: Path,
) -> None:
    root = tmp_path / "glb"
    root.mkdir()
    glb_path = root / "extensionless_webp.glb"
    _write_box_glb(
        glb_path,
        minimum=(-0.5, 0.0, -0.5),
        maximum=(0.5, 1.0, 0.5),
        image_uri="data:image/webp;base64,AAAA",
    )
    policy = static_objects.WorkspacePathPolicy.from_roots([root])
    artifact = static_objects._file_record(
        _record(glb_path),
        policy,
        owner="test extensionless-WebP GLB",
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="MIME type is unsupported",
    ):
        static_objects._read_glb_geometry(
            artifact,
            owner="test extensionless-WebP GLB",
        )


def test_resealed_stage_command_must_match_exact_stage_io(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "command")

    def change_target_faces(command: list[str]) -> None:
        index = command.index("--target-faces")
        command[index + 1] = "10001"

    _rewrite_stage_command_closure(
        paths,
        stage_path_key="watertight_stage",
        stage_name="watertight",
        mutate=change_target_faces,
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="command argv differs",
    ):
        _validate(paths)


@pytest.mark.parametrize(
    "artifact_key",
    (
        "pixal_glb",
        "flux_batch",
        "flux_review_batch",
        "review",
        "watertight_glb",
        "blender_binary",
        "watertight_command_tool",
        "emitter_contract_tool",
        "watertight_stage",
        "job_receipt",
        "batch",
        "approval",
    ),
)
def test_every_receipt_layer_is_hash_authenticated(
    tmp_path: Path,
    artifact_key: str,
) -> None:
    paths = _admission_fixture(tmp_path / artifact_key)
    Path(paths[artifact_key]).write_bytes(
        Path(paths[artifact_key]).read_bytes() + b"\ntampered\n"
    )
    with pytest.raises(StaticObjectRegistrationError):
        _validate(paths)


def test_stage_summary_is_revalidated_after_resealing_entire_chain(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "workspace")
    stage = load_json(paths["watertight_stage"])
    stage["validation"]["no_rig_or_animation"] = False
    stage.pop("receipt_sha256")
    _seal(stage, "receipt_sha256")
    stage_record = _json_record(
        paths["watertight_stage"],
        stage,
        relative_to=paths["admission_root"],
    )

    job = load_json(paths["job_receipt"])
    job["stage_receipts"]["watertight"] = stage_record
    job.pop("receipt_sha256")
    _seal(job, "receipt_sha256")
    job_record = _json_record(
        paths["job_receipt"],
        job,
        relative_to=paths["admission_root"],
    )
    batch = load_json(paths["batch"])
    batch["jobs"][0]["receipt_sha256"] = job["receipt_sha256"]
    batch["jobs"][0]["job_receipt"] = job_record
    batch.pop("batch_sha256")
    _seal(batch, "batch_sha256")
    _json_record(paths["batch"], batch)

    with pytest.raises(
        StaticObjectRegistrationError,
        match="stage validation summary",
    ):
        _validate(paths)


def test_admission_batch_jobs_must_cover_exactly_the_approved_instances(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "workspace")
    batch = load_json(paths["batch"])
    unexpected = deepcopy(batch["jobs"][0])
    unexpected["instance_id"] = "unexpected_static_source_0002"
    unexpected["request_sha256"] = "3" * 64
    unexpected["profile_sha256"] = "4" * 64
    batch["jobs"].append(unexpected)
    batch["job_count"] = 2
    batch["passed_count"] = 2
    batch.pop("batch_sha256")
    _seal(batch, "batch_sha256")
    _json_record(paths["batch"], batch)

    with pytest.raises(
        StaticObjectRegistrationError,
        match="differs from approved decision",
    ):
        _validate(paths)


def test_marker_gate_requires_a_separate_human_visual_approval(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "workspace")
    approval = load_json(paths["approval"])
    approval["reviewer_kind"] = "agent"
    approval.pop("approval_content_sha256")
    _seal(approval, "approval_content_sha256")
    _json_record(paths["approval"], approval)
    with pytest.raises(
        StaticObjectRegistrationError,
        match="marker visual approval schema failed",
    ):
        _validate(paths)


def test_resealed_emitter_surface_cannot_depart_from_final_glb(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "workspace")

    def move_resolved_surface(value: dict[str, Any]) -> None:
        anchor = value["emitter_anchor"]
        anchor["offset_m"] = [0.4, 0.6, 0.05]
        sample = anchor["resolved_surface_samples"][0]
        sample["surface_point_m"] = [0.4, 0.6, 0.05]
        sample["target_to_surface_distance_m"] = 0.1

    _rewrite_measurement_closure(
        paths,
        move_resolved_surface,
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="finalized GLB geometry",
    ):
        _validate(paths)


def test_emitter_offset_must_be_derived_from_resolved_samples(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "workspace")
    _rewrite_measurement_closure(
        paths,
        lambda value: value["emitter_anchor"].update(
            {"offset_m": [0.49, 0.6, 0.05]}
        ),
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="weighted centroid",
    ):
        _validate(paths)


def test_coordinate_contract_rejects_legacy_z_left_even_when_resealed(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "workspace")

    def mutate(value: dict[str, Any]) -> None:
        value["coordinate_system"] = {
            **CANONICAL_COORDINATE_SYSTEM,
            "id": "avengine_local_x_forward_y_up_z_left_m",
        }

    _rewrite_measurement_closure(paths, mutate)
    with pytest.raises(StaticObjectRegistrationError, match=r"\+Z right"):
        _validate(paths)


def test_generic_registry_validator_rejects_rebound_semantic_mismatches(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "workspace")
    output = Path(paths["root"]) / "published.json"
    _publish(paths, output)
    original = load_json(output)
    index = next(
        index
        for index, item in enumerate(original["entities"])
        if item["entity_asset_id"] == INSTANCE_ID
    )

    formal = load_json(output)
    formal_entity = formal["entities"][index]
    formal_evidence = formal_entity["admission_evidence"]
    formal_evidence["formal_dataset_registration_authorized"] = True
    formal_evidence["evidence_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in formal_evidence.items()
            if key != "evidence_content_sha256"
        }
    )
    formal_entity["revision"] = (
        "spear_static_" + formal_evidence["evidence_content_sha256"]
    )
    formal_entity["provenance"]["evidence_sha256"] = formal_evidence[
        "evidence_content_sha256"
    ]
    formal = bind_content_hash(
        {
            key: value
            for key, value in formal.items()
            if key != "registry_content_sha256"
        }
    )
    assert any(
        "formal" in error
        for error in validate_entity_asset_registry(formal)
    )

    identity = load_json(output)
    entity = identity["entities"][index]
    evidence = entity["admission_evidence"]
    evidence["identity"]["instance_id"] = "different_static_source"
    evidence["evidence_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in evidence.items()
            if key != "evidence_content_sha256"
        }
    )
    entity["revision"] = "spear_static_" + evidence["evidence_content_sha256"]
    entity["provenance"]["evidence_sha256"] = evidence[
        "evidence_content_sha256"
    ]
    identity = bind_content_hash(
        {
            key: value
            for key, value in identity.items()
            if key != "registry_content_sha256"
        }
    )
    assert any(
        "identity.instance_id" in error
        for error in validate_entity_asset_registry(identity)
    )

    visual = load_json(output)
    visual["entities"][index]["visual_asset"]["sha256"] = "c" * 64
    visual = bind_content_hash(
        {
            key: value
            for key, value in visual.items()
            if key != "registry_content_sha256"
        }
    )
    assert any(
        "visual_asset" in error
        for error in validate_entity_asset_registry(visual)
    )

    anchor = load_json(output)
    anchor["entities"][index]["emitter_anchors"][0][
        "anchor_id"
    ] = "different_speaker"
    anchor = bind_content_hash(
        {
            key: value
            for key, value in anchor.items()
            if key != "registry_content_sha256"
        }
    )
    assert any(
        "speaker named by admission_evidence" in error
        for error in validate_entity_asset_registry(anchor)
    )


def test_failed_prepublication_verify_leaves_no_immutable_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _admission_fixture(tmp_path / "workspace")
    output = Path(paths["root"]) / "must_not_exist.json"

    def fail_verify(**_: Any) -> None:
        raise StaticObjectRegistrationError("injected staging verify failure")

    monkeypatch.setattr(
        static_objects,
        "verify_static_object_entity_registry",
        fail_verify,
    )
    with pytest.raises(
        StaticObjectRegistrationError,
        match="injected staging verify failure",
    ):
        _publish(paths, output)
    assert not output.exists()
    assert not tuple(output.parent.glob(f".{output.name}.*.verify.json"))


def test_publication_never_replaces_existing_registry(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "workspace")
    output = Path(paths["root"]) / "published.json"
    _publish(paths, output)
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        _publish(paths, output, revision="run_v2")
    assert output.read_bytes() == before


def test_loose_leaf_api_is_removed_and_cli_requires_sealed_batch(
    tmp_path: Path,
) -> None:
    paths = _admission_fixture(tmp_path / "workspace")
    with pytest.raises(TypeError):
        validate_static_object_admission(
            static_decision_path=paths["decision"],
            finalization_manifest_path=paths["final_glb"],
            emitter_measurement_path=paths["measurement"],
            marker_visual_approval_path=paths["approval"],
            workspace_roots=[paths["root"]],
        )

    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "tools/registry/publish_static_object_registry.py"),
            "publish",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--admission-batch" in completed.stdout
    assert "--instance-id" in completed.stdout
    assert "--static-decision" not in completed.stdout
    assert "--finalization-manifest" not in completed.stdout
    assert "--emitter-measurement" not in completed.stdout

    output = Path(paths["root"]) / "cli_published.json"
    published = subprocess.run(
        [
            sys.executable,
            str(repository / "tools/registry/publish_static_object_registry.py"),
            "publish",
            "--base-registry",
            str(paths["base_registry"]),
            "--admission-batch",
            str(paths["batch"]),
            "--instance-id",
            INSTANCE_ID,
            "--marker-visual-approval",
            str(paths["approval"]),
            "--registry-revision",
            "cli_run_v1",
            "--output",
            str(output),
            "--workspace-root",
            str(paths["root"]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert published.returncode == 0, published.stderr
    assert output.is_file()
    assert f'"entity_asset_id": "{INSTANCE_ID}"' in published.stdout
