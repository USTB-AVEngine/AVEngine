from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable
from unittest.mock import patch

import pytest

from avengine.appearance import generate_l9_batch
from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.assets import variant_package
from avengine.assets.glb import load_glb
from avengine.assets.glb_write import build_glb
from avengine.assets.materials import normalize_glb_materials
from avengine.assets.variant_package import VariantPackageError
from tools.assets import build_appearance_variant_inputs as builder


REPOSITORY = Path(__file__).resolve().parents[2]
REQUEST = REPOSITORY / "examples/assets/appearance/beagle_l9_request_v1.json"
TEMPLATE = (
    REPOSITORY / "examples/assets/variant_packages/rocketbox_beagle_review_spec_v1.json"
)
REALIZER = REPOSITORY / "tools/blender/realize_animal_appearance.py"
MATERIAL_NORMALIZER = REPOSITORY / "src/avengine/assets/materials.py"
IDENTITY = [0.0, 0.0, 0.0, 1.0]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _pretty_payload(value: Any) -> bytes:
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


def _file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _json_binding(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        **_file_binding(path),
        "canonical_content_sha256": canonical_json_sha256(value),
        "snapshot": value,
    }


def _rehash_lineage(lineage: dict[str, Any]) -> None:
    lineage["lineage_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in lineage.items()
            if key != "lineage_content_sha256"
        }
    )


def _rehash_report(report: dict[str, Any]) -> None:
    report["report_content_sha256"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_content_sha256"}
    )


def _glb_document() -> dict[str, Any]:
    names = [
        "beagle Pelvis",
        "beagle Head",
        "beagle Xtra Mouth",
        "beagle L Finger0",
        "beagle R Finger0",
        "beagle L Toe0",
        "beagle R Toe0",
    ]
    nodes = [
        {
            "name": names[0],
            "children": list(range(1, len(names))),
            "rotation": IDENTITY,
        }
    ]
    nodes.extend({"name": name, "rotation": IDENTITY} for name in names[1:])
    return {
        "asset": {"version": "2.0", "generator": "appearance-input-test"},
        "extensionsUsed": ["KHR_materials_specular"],
        "nodes": nodes,
        "skins": [{"skeleton": 0, "joints": list(range(len(names)))}],
        "meshes": [
            {
                "name": "animal",
                "primitives": [{"attributes": {}, "material": 0}],
            }
        ],
        "materials": [
            {
                "name": "animal_pbr",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.72,
                },
                "alphaMode": "OPAQUE",
                "emissiveFactor": [0.0, 0.0, 0.0],
                "extensions": {
                    "KHR_materials_specular": {
                        "specularFactor": 0.25,
                        "specularColorFactor": [1.0, 1.0, 1.0],
                    }
                },
            }
        ],
        "images": [
            {"name": "base_color"},
            {"name": "normal"},
            {"name": "specular"},
        ],
        "animations": [{"name": "Idle"}, {"name": "Walking"}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }


def _parameters(request: dict[str, Any], attribute: str) -> dict[str, Any]:
    return next(
        item["parameters"]
        for item in request["realization_operations"]
        if item["attribute"] == attribute
    )


def _report(
    *,
    batch: dict[str, Any],
    batch_path: Path,
    ordinal: int,
    source: Path,
    visual: Path,
    texture: Path,
) -> dict[str, Any]:
    request = batch["requests"][ordinal - 1]
    size = _parameters(request, "size")
    body = _parameters(request, "body_build")
    coat = _parameters(request, "coat_profile")
    life = _parameters(request, "life_stage")
    head_ratio = 1.0 + (life["head_scale"] - 1.0) * 0.98
    torso_ratio = 1.0 + (body["torso_girth_scale"] - 1.0) * 0.35
    texture_sha = sha256_file(texture)
    normal_sha = "22" * 32
    specular_sha = "33" * 32
    action_records = [
        {
            "action": action,
            "channel_count": 7,
            "maximum_errors": {
                "rotation": 0.0,
                "scale": 0.0,
                "timestamps": 0.0,
                "translation": 0.0,
            },
        }
        for action in ("Idle", "Walking")
    ]
    report = {
        "schema": "avengine_animal_appearance_realization_v1",
        "status": "pass",
        "state_classification": "research_candidate",
        "qualification_claim": False,
        "formal_dataset_registration_authorized": False,
        "created_at": "2026-07-17T00:00:00Z",
        "limitations": ["synthetic unit fixture"],
        "batch": {
            "path": str(batch_path.resolve()),
            "sha256": sha256_file(batch_path),
            "batch_id": batch["batch_id"],
            "batch_content_sha256": batch["batch_content_sha256"],
        },
        "instance_request": {
            "ordinal": request["ordinal"],
            "instance_request_id": request["instance_request_id"],
            "request_sha256": request["request_sha256"],
            "taxonomy": deepcopy(request["taxonomy"]),
            "attributes": deepcopy(request["attributes"]),
            "realization_operations": deepcopy(request["realization_operations"]),
        },
        "source": {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
            "byte_size": source.stat().st_size,
        },
        "tool_identity": {
            "path": str(REALIZER.resolve()),
            "sha256": sha256_file(REALIZER),
            "material_normalizer": {
                "path": str(MATERIAL_NORMALIZER.resolve()),
                "sha256": sha256_file(MATERIAL_NORMALIZER),
            },
            "blender_version": "4.2.1 LTS",
            "export_profile": {
                "animation_mode": "ACTIONS",
                "force_sampling": True,
                "format": "GLB",
                "image_format": "AUTO",
                "normals": True,
                "skins": True,
                "texcoords": True,
            },
            "output_readback_float_tolerance": 5.0e-5,
        },
        "realization": {
            "topology_uv_skin_sha256_before": "10" * 32,
            "topology_uv_skin_sha256_after": "10" * 32,
            "topology_uv_skin_unchanged": True,
            "action_curve_sha256_before": "11" * 32,
            "action_curve_sha256_after": "11" * 32,
            "in_memory_authored_action_curves_unchanged": True,
            "uniform_size": {
                "strategy": "armature_data_and_mesh_data_matrix_bake_v1",
                "ancestor_scale_node_created": False,
                "scale_ratio": size["scale_ratio"],
                "maximum_mesh_scale_error": 0.0,
            },
            "shape": {
                "requested_torso_girth_scale": body["torso_girth_scale"],
                "requested_head_scale": life["head_scale"],
                "head_group_names": ["beagle Head"],
                "torso_group_names": ["beagle Pelvis"],
                "head_selected_vertices": 10,
                "torso_selected_vertices": 20,
                "head_weighted_radius_rms_before": 1.0,
                "head_weighted_radius_rms_after": head_ratio,
                "head_weighted_radius_rms_ratio": head_ratio,
                "torso_weighted_yz_rms_before": 1.0,
                "torso_weighted_yz_rms_after": torso_ratio,
                "torso_weighted_yz_rms_ratio": torso_ratio,
            },
            "texture": {
                "luminance_gain": coat["luminance_gain"],
                "coat_desaturation": life["coat_desaturation"],
                "muzzle_gray_mix": life["muzzle_gray_mix"],
                "muzzle_gray_target": life["muzzle_gray_target"],
                "output_texture": str(texture.resolve()),
                "resolution": [8, 8],
                "pigmented_pixel_count": 32,
                "muzzle_mask_nonzero_pixels": 8,
                "mean_pigmented_luminance_before": 0.4,
                "mean_pigmented_luminance_after": 0.4,
                "muzzle_mask_max": 1.0,
                "muzzle_forward_quantile": 0.58,
                "uv_minimum": 0.0,
                "uv_maximum": 1.0,
                "uv_addressing_assumption": "non_tiled_clamp_0_1",
                "source_image": "synthetic",
                "preserve_pattern": "tricolor",
                "pattern_audit": {
                    "status": "pass",
                    "registered_pattern": "tricolor",
                    "spatial_region_mask_sha256": "12" * 32,
                    "coat_gain_and_desaturation_preserve_region_membership": True,
                    "white_pixel_count": 32,
                    "pigmented_pixel_count": 32,
                    "dark_pixel_count": 32,
                    "warm_pixel_count": 32,
                },
            },
            "materials": [
                {
                    "material": "animal_pbr",
                    "before": {"metallic": 0.0, "roughness": 0.72},
                    "after": {
                        "metallic": 0.0,
                        "roughness": 0.72,
                        "roughness_texture_driven": False,
                    },
                }
            ],
        },
        "output": {
            "glb": {
                "path": str(visual.resolve()),
                "sha256": sha256_file(visual),
                "byte_size": visual.stat().st_size,
            },
            "base_color_texture": {
                "path": str(texture.resolve()),
                "sha256": texture_sha,
                "byte_size": texture.stat().st_size,
            },
            "readback_audit": {
                "skin_count": 1,
                "skin_joint_count": 7,
                "skin_root_name": "beagle Pelvis",
                "mesh_count": 1,
                "material_count": 1,
                "image_count": 3,
                "animation_names": ["Idle", "Walking"],
                "maximum_skin_ancestor_scale_error": 0.0,
                "skin_ancestors": [],
                "mesh_invariants": {
                    "indices_exact": True,
                    "joints_0_exact": True,
                    "vertex_count": 10,
                    "index_count": 30,
                    "maximum_expected_position_error_m": 0.0,
                    "maximum_output_normal_norm_error": 0.0,
                    "maximum_texcoord_0_error": 0.0,
                    "maximum_weights_0_error": 0.0,
                    "head_weight_sum": 1.0,
                    "torso_weight_sum": 1.0,
                    "geometry_frame": {
                        "basis_formula": "blender=(gltf.x,-gltf.z,gltf.y)",
                        "blender_import": {
                            "forward": "positive_x",
                            "lateral": "positive_y",
                            "up": "positive_z",
                        },
                        "source": "gltf_positive_y_up",
                    },
                },
                "skin_invariants": {
                    "joint_order_unchanged": True,
                    "joint_count": 7,
                    "tolerance": 5.0e-5,
                    "maximum_rest_rotation_error": 0.0,
                    "maximum_rest_scale_error": 0.0,
                    "maximum_scaled_inverse_bind_matrix_error": 0.0,
                    "maximum_scaled_rest_translation_error_m": 0.0,
                },
                "action_invariants": {
                    "channel_targets_unchanged": True,
                    "translations_scaled_by_size": True,
                    "tolerance": 5.0e-5,
                    "maximum_error": 0.0,
                    "actions": action_records,
                },
                "material_invariants": {
                    "material_count": 1,
                    "metallic_factor": 0.0,
                    "roughness_factor": 0.72,
                    "metallic_roughness_texture_present": False,
                    "alpha_mode": "OPAQUE",
                    "base_color_factor": [1.0, 1.0, 1.0, 1.0],
                    "emissive_factor": [0.0, 0.0, 0.0],
                    "emissive_texture_present": False,
                    "effective_khr_materials_specular_factor": 0.25,
                    "effective_khr_materials_specular_color_factor": [
                        1.0,
                        1.0,
                        1.0,
                    ],
                    "maximum_effective_khr_materials_specular_channel": 0.25,
                    "allowed_material_extensions": ["KHR_materials_specular"],
                    "texture_images": {
                        "base_color": {
                            "mime_type": "image/png",
                            "source_sha256": texture_sha,
                            "output_sha256": texture_sha,
                            "standalone_sha256": texture_sha,
                            "embedded_matches_standalone": True,
                            "unchanged": False,
                        },
                        "normal": {
                            "mime_type": "image/png",
                            "source_sha256": normal_sha,
                            "output_sha256": normal_sha,
                            "unchanged": True,
                        },
                        "specular": {
                            "mime_type": "image/png",
                            "source_sha256": specular_sha,
                            "output_sha256": specular_sha,
                            "unchanged": True,
                        },
                    },
                },
            },
        },
    }
    report["report_content_sha256"] = canonical_json_sha256(report)
    return report


def _fixture(tmp_path: Path) -> dict[str, Any]:
    source = tmp_path / "source.glb"
    visual = tmp_path / "appearance.glb"
    source_document = _glb_document()
    source.write_bytes(build_glb(source_document, b""))
    visual_document = deepcopy(source_document)
    visual_document["asset"]["generator"] = "appearance-input-test-realized"
    visual.write_bytes(build_glb(visual_document, b""))
    texture = tmp_path / "appearance.base_color.png"
    texture.write_bytes(b"synthetic-png-bytes")

    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    request["source_asset"]["expected_sha256"] = sha256_file(source)
    request_path = tmp_path / "request.json"
    _write_json(request_path, request)
    batch = generate_l9_batch(request_path, source)
    batch_path = tmp_path / "batch.json"
    _write_json(batch_path, batch)

    report_path = tmp_path / "appearance_report.json"
    report = _report(
        batch=batch,
        batch_path=batch_path,
        ordinal=1,
        source=source,
        visual=visual,
        texture=texture,
    )
    _write_json(report_path, report)

    template_path = tmp_path / "template.json"
    _write_json(template_path, json.loads(TEMPLATE.read_text(encoding="utf-8")))
    template = json.loads(template_path.read_text(encoding="utf-8"))
    upstream_path = tmp_path / "upstream.json"
    upstream = {
        "schema": "avengine_m2_test_source_snapshot_v1",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "formal_dataset_registration_authorized": False,
        "source_repository": {
            "url": "https://example.invalid/source.git",
            "revision": template["identity"]["source_revision"],
        },
        "source_artifacts": [
            {
                "path": "source.asset",
                "root_id": "test_source",
                "byte_size": 1,
                "sha256": "44" * 32,
            }
        ],
    }
    _write_json(upstream_path, upstream)
    return {
        "source": source,
        "visual": visual,
        "texture": texture,
        "batch": batch,
        "batch_path": batch_path,
        "report": report,
        "report_path": report_path,
        "template": template,
        "template_path": template_path,
        "upstream": upstream,
        "upstream_path": upstream_path,
    }


def _build(fixture: dict[str, Any], ordinal: int = 1) -> tuple[dict, dict]:
    if ordinal != 1:
        report = _report(
            batch=fixture["batch"],
            batch_path=fixture["batch_path"],
            ordinal=ordinal,
            source=fixture["source"],
            visual=fixture["visual"],
            texture=fixture["texture"],
        )
        report_path = fixture["report_path"].with_name(f"report_{ordinal:02d}.json")
        _write_json(report_path, report)
    else:
        report_path = fixture["report_path"]
    report_value = json.loads(report_path.read_text(encoding="utf-8"))
    with patch.object(
        builder,
        "verify_appearance_realization",
        return_value=_independent_audit_from_report(report_value),
    ):
        return builder.build_inputs(
            batch_path=fixture["batch_path"],
            ordinal=ordinal,
            appearance_report_path=report_path,
            template_spec_path=fixture["template_path"],
            upstream_source_manifest_path=fixture["upstream_path"],
        )


def _independent_audit_from_report(report: dict[str, Any]) -> dict[str, Any]:
    """Keep legacy contract fixtures focused on report/file closure.

    Byte-realization behavior has dedicated tests below; this long-standing
    fixture intentionally contains a metadata-only GLB and cannot represent a
    Blender realization.
    """

    shape = deepcopy(report["realization"]["shape"])
    mesh = report["output"]["readback_audit"]["mesh_invariants"]
    shape.update(
        {
            "maximum_expected_position_error_m": mesh[
                "maximum_expected_position_error_m"
            ],
            "maximum_output_normal_norm_error": mesh[
                "maximum_output_normal_norm_error"
            ],
            "maximum_raw_weights_0_error": mesh["maximum_weights_0_error"],
            "head_weight_sum": mesh["head_weight_sum"],
            "torso_weight_sum": mesh["torso_weight_sum"],
        }
    )
    texture = deepcopy(report["realization"]["texture"])
    pattern = texture["pattern_audit"]
    texture.update(
        {
            "preserve_pattern": "tricolor",
            "white_pixel_count": pattern["white_pixel_count"],
            "dark_pixel_count": pattern["dark_pixel_count"],
            "warm_pixel_count": pattern["warm_pixel_count"],
        }
    )
    readback = report["output"]["readback_audit"]
    skin = readback["skin_invariants"]
    actions = readback["action_invariants"]
    return {
        "shape": shape,
        "texture": texture,
        "compatibility": {
            "mesh": {
                "vertex_count": mesh["vertex_count"],
                "index_count": mesh["index_count"],
                "maximum_texcoord_0_error": mesh["maximum_texcoord_0_error"],
                "maximum_weights_0_error": mesh["maximum_weights_0_error"],
            },
            "skin": {
                "joint_count": skin["joint_count"],
                "maximum_scaled_rest_translation_error_m": skin[
                    "maximum_scaled_rest_translation_error_m"
                ],
                "maximum_rest_rotation_error": skin["maximum_rest_rotation_error"],
                "maximum_rest_scale_error": skin["maximum_rest_scale_error"],
                "maximum_scaled_inverse_bind_matrix_error": skin[
                    "maximum_scaled_inverse_bind_matrix_error"
                ],
            },
            "actions": {
                "maximum_errors": {
                    "timestamps": actions["maximum_error"],
                    "translation": 0.0,
                    "rotation": 0.0,
                    "scale": 0.0,
                }
            },
        },
    }


def _validate_metadata_only_fixture_lineage(
    spec: dict[str, Any],
    lineage: dict[str, Any],
    *,
    serialized_spec: bytes | None = None,
    audit_lineage: dict[str, Any] | None = None,
) -> None:
    """Validate an old contract fixture without pretending its GLB is real.

    The production validator must inspect actual GLB/PNG bytes.  This test file
    predates that gate and deliberately uses a metadata-only GLB so its many
    lineage/atomic-write cases stay small.  Patch only the independent byte
    audit at each explicit legacy-fixture boundary; dedicated verifier tests
    exercise the unpatched implementation.
    """

    audit_source = lineage if audit_lineage is None else audit_lineage
    report = audit_source["inputs"]["appearance_realization_report"]["snapshot"]
    with patch.object(
        builder,
        "verify_appearance_realization",
        return_value=_independent_audit_from_report(report),
    ):
        builder.validate_spec_lineage_binding(
            spec,
            lineage,
            serialized_spec=serialized_spec,
        )


def _write_metadata_only_fixture_pair(
    *,
    spec_output: Path,
    lineage_output: Path,
    spec: dict[str, Any],
    lineage: dict[str, Any],
) -> tuple[Path, Path]:
    """Call the atomic writer for the explicit metadata-only fixture."""

    report = lineage["inputs"]["appearance_realization_report"]["snapshot"]
    with patch.object(
        builder,
        "verify_appearance_realization",
        return_value=_independent_audit_from_report(report),
    ):
        return builder._write_output_pair(
            spec_output=spec_output,
            lineage_output=lineage_output,
            spec=spec,
            lineage=lineage,
        )


def _rebind_realization_report(
    lineage: dict[str, Any], report_path: Path, report: dict[str, Any]
) -> None:
    _rehash_report(report)
    _write_json(report_path, report)
    lineage["inputs"]["appearance_realization_report"] = _json_binding(report_path)
    _rehash_lineage(lineage)


def _run_strict_assembler_lineage_validation(
    *,
    tmp_path: Path,
    fixture: dict[str, Any],
    spec: dict[str, Any],
    lineage: dict[str, Any],
) -> None:
    spec_path = tmp_path / "derived_spec_for_assembler.json"
    spec_path.write_bytes(_pretty_payload(spec))
    lineage_path = tmp_path / "lineage_for_assembler.json"
    _write_json(lineage_path, lineage)
    parsed_spec = variant_package.load_variant_package_spec(spec_path)
    unused = fixture["upstream_path"]
    evidence = variant_package.VariantPackageEvidence(
        visual_glb=fixture["visual"],
        rebase_report=unused,
        rebase_deformation_report=unused,
        action_report=unused,
        static_qa=unused,
        deformation_qa=unused,
        animation_qa=unused,
        habitat_static_probe=unused,
        habitat_animation_review=unused,
        baked_actions=fixture["visual"],
        contacts=unused,
        source_manifest=fixture["upstream_path"],
        license_snapshot=unused,
        appearance_lineage=lineage_path,
        material_normalization_report=unused,
    )
    visual_binding = _file_binding(fixture["visual"])
    report = lineage["inputs"]["appearance_realization_report"]["snapshot"]
    try:
        independent_audit = _independent_audit_from_report(report)
    except KeyError:
        independent_audit = {}
    with (
        patch.object(
            builder,
            "verify_appearance_realization",
            return_value=independent_audit,
        ),
        patch.object(
            variant_package,
            "_load_l9_producer_validator",
            return_value=(
                builder.AppearanceVariantInputError,
                builder.validate_spec_lineage_binding,
            ),
        ),
    ):
        variant_package._validate_l9_appearance_lineage(
            spec=parsed_spec,
            evidence=evidence,
            visual=fixture["visual"],
            rebase_report={
                "source": visual_binding,
                "output": visual_binding,
            },
            lineage=lineage,
        )


def test_assembler_runs_complete_producer_contract_for_valid_ordinal_two(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    spec, lineage = _build(fixture, ordinal=2)

    _run_strict_assembler_lineage_validation(
        tmp_path=tmp_path,
        fixture=fixture,
        spec=spec,
        lineage=lineage,
    )


def test_raw_cli_subprocess_bootstraps_repo_src_without_pythonpath(
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "tools/assets/assemble_variant_package.py"),
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_real_cli_subprocess_assembles_after_full_l9_validation_with_startup_hook(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    spec, lineage = _build(fixture, ordinal=2)
    spec_path = tmp_path / "cli_spec.json"
    spec_path.write_bytes(_pretty_payload(spec))
    lineage_path = tmp_path / "cli_lineage.json"
    _write_json(lineage_path, lineage)

    final_visual = tmp_path / "cli_final_visual.glb"
    normalization = normalize_glb_materials(
        fixture["visual"],
        final_visual,
        force_opaque=True,
    )
    normalization_path = tmp_path / "cli_material_normalization.json"
    _write_json(normalization_path, normalization)
    rebase_path = tmp_path / "cli_rebase.json"
    _write_json(
        rebase_path,
        {
            "schema": "avengine_m2_skin_root_rebase_v1",
            "status": "pass",
            "qualification_claim": False,
            "source": _file_binding(fixture["visual"]),
            "output": _file_binding(final_visual),
        },
    )

    evidence_schemas = {
        "rebase_deformation": "avengine_m2_rebase_deformation_verification_v1",
        "action_report": "avengine_m2_action_bake_report_v1",
        "static_qa": "avengine_m2_static_geometry_qa_v1",
        "deformation_qa": "avengine_m2_deformation_qa_v1",
        "animation_qa": "avengine_m2_animation_qa_v1",
        "habitat_static": "avengine_m2_habitat_skin_rest_probe_v1",
        "habitat_animation": "avengine_m2_habitat_action_review_v1",
    }
    reports: dict[str, Path] = {}
    for name, schema in evidence_schemas.items():
        path = tmp_path / f"cli_{name}.json"
        _write_json(
            path,
            {
                "schema": schema,
                "status": "pass",
                "qualification_claim": False,
            },
        )
        reports[name] = path
    actions = tmp_path / "cli_actions.npz"
    actions.write_bytes(b"compiler-isolated-cli-smoke")
    contacts = tmp_path / "cli_contacts.json"
    _write_json(contacts, {"schema": "avengine_m2_contact_phases_v1"})
    license_snapshot = tmp_path / "cli_license.json"
    _write_json(license_snapshot, {"schema": "test_license_snapshot_v1"})

    # The subprocess must execute the real CLI and every strict evidence gate.
    # Only the downstream package compiler is isolated, because this fixture is
    # intentionally small and the regression target is the canonical L9 loader.
    hook_directory = tmp_path / "python_hook"
    hook_directory.mkdir()
    subprocess_audit = _independent_audit_from_report(
        lineage["inputs"]["appearance_realization_report"]["snapshot"]
    )
    (hook_directory / "sitecustomize.py").write_text(
        """from __future__ import annotations
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path("""
        + repr(str(REPOSITORY / "src"))
        + """)
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import avengine.assets.appearance_realization as appearance_realization
import avengine.assets.contracts as contracts
import avengine.assets.variant_package as variant_package

def compile_smoke(**kwargs):
    output = Path(kwargs["output_directory"])
    output.mkdir()
    identity = kwargs["identity"]
    manifest = output / "asset_manifest.json"
    manifest.write_text(json.dumps({
        "admission_state": "research_candidate",
        "asset_id": identity.asset_id,
        "body_plan_id": identity.body_plan_id,
        "morphotype_id": identity.morphotype_id,
        "qualification": {"automatic_qa_status": "pass"},
    }) + "\\n", encoding="utf-8")
    return manifest

variant_package.compile_research_candidate_animal_package = compile_smoke
contracts.validate_animal_asset_package = lambda *args, **kwargs: []
"""
        + "\n_INDEPENDENT_AUDIT = json.loads("
        + repr(json.dumps(subprocess_audit, sort_keys=True))
        + ")\nappearance_realization.verify_appearance_realization = "
        + "lambda **kwargs: _INDEPENDENT_AUDIT\n",
        encoding="utf-8",
    )
    output = tmp_path / "cli_package"
    command = [
        sys.executable,
        str(REPOSITORY / "tools/assets/assemble_variant_package.py"),
        "--spec",
        str(spec_path),
        "--visual-glb",
        str(final_visual),
        "--actions-npz",
        str(actions),
        "--rebase-report",
        str(rebase_path),
        "--rebase-deformation-report",
        str(reports["rebase_deformation"]),
        "--action-report",
        str(reports["action_report"]),
        "--static-qa",
        str(reports["static_qa"]),
        "--deformation-qa",
        str(reports["deformation_qa"]),
        "--animation-qa",
        str(reports["animation_qa"]),
        "--habitat-static-probe",
        str(reports["habitat_static"]),
        "--habitat-animation-review",
        str(reports["habitat_animation"]),
        "--contact-phases",
        str(contacts),
        "--appearance-lineage",
        str(lineage_path),
        "--material-normalization-report",
        str(normalization_path),
        "--source-manifest",
        str(fixture["upstream_path"]),
        "--license-snapshot",
        str(license_snapshot),
        "--output",
        str(output),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(hook_directory)
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert json.loads(result.stdout)["status"] == "pass"
    assert (output / "asset_manifest.json").is_file()


def test_assembler_rejects_ordinal_two_unchanged_source_output(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    spec, lineage = _build(fixture, ordinal=2)
    report_path = Path(lineage["inputs"]["appearance_realization_report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["output"]["glb"] = _file_binding(fixture["source"])
    _rebind_realization_report(lineage, report_path, report)

    with pytest.raises(
        VariantPackageError,
        match="L9 producer contract failed full validation.*byte-identical",
    ):
        _run_strict_assembler_lineage_validation(
            tmp_path=tmp_path,
            fixture=fixture,
            spec=spec,
            lineage=lineage,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report.pop("realization"), "appearance report realization"),
        (
            lambda report: report["output"].pop("readback_audit"),
            "output.readback_audit",
        ),
    ],
)
def test_assembler_rejects_missing_producer_operations_or_readback(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    spec, lineage = _build(fixture, ordinal=2)
    report_path = Path(lineage["inputs"]["appearance_realization_report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mutation(report)
    _rebind_realization_report(lineage, report_path, report)

    with pytest.raises(
        VariantPackageError,
        match=f"L9 producer contract failed full validation.*{message}",
    ):
        _run_strict_assembler_lineage_validation(
            tmp_path=tmp_path,
            fixture=fixture,
            spec=spec,
            lineage=lineage,
        )


def _mutate_visual_and_rebind_report(
    fixture: dict[str, Any], mutation: Callable[[dict[str, Any]], None]
) -> None:
    visual = fixture["visual"]
    source = load_glb(visual)
    document = deepcopy(source.json)
    mutation(document)
    visual.write_bytes(build_glb(document, source.binary))
    report = deepcopy(fixture["report"])
    glb_record = report["output"]["glb"]
    glb_record["sha256"] = sha256_file(visual)
    glb_record["byte_size"] = visual.stat().st_size
    _rehash_report(report)
    _write_json(fixture["report_path"], report)
    fixture["report"] = report


def test_build_binds_template_and_exact_derived_spec(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    spec, lineage = _build(fixture)
    payload = _pretty_payload(spec)

    _validate_metadata_only_fixture_lineage(
        spec,
        lineage,
        serialized_spec=payload,
    )
    template_binding = lineage["inputs"]["template_variant_spec"]
    assert template_binding["sha256"] == sha256_file(fixture["template_path"])
    assert template_binding["canonical_content_sha256"] == canonical_json_sha256(
        fixture["template"]
    )
    assert template_binding["snapshot"] == fixture["template"]
    derived = lineage["derivative"]["derived_variant_spec"]
    assert derived == {
        "schema": "avengine_m2_variant_package_spec_v1",
        "asset_id": spec["identity"]["asset_id"],
        "byte_size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "canonical_content_sha256": canonical_json_sha256(spec),
    }
    assert lineage["lineage_content_sha256"] == canonical_json_sha256(
        {
            key: value
            for key, value in lineage.items()
            if key != "lineage_content_sha256"
        }
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda material: material.__setitem__("alphaMode", "BLEND"),
            "opaque matte policy",
        ),
        (
            lambda material: material["pbrMetallicRoughness"].__setitem__(
                "baseColorFactor", [1.0, 1.0, 1.0, 0.0]
            ),
            "opaque matte policy",
        ),
        (
            lambda material: material.__setitem__("emissiveFactor", [0.0, 0.2, 0.0]),
            "opaque matte policy",
        ),
        (
            lambda material: material.__setitem__("emissiveTexture", {"index": 0}),
            "opaque matte policy",
        ),
        (
            lambda material: material["extensions"][
                "KHR_materials_specular"
            ].__setitem__("specularColorFactor", [4.0, 1.0, 1.0]),
            "specularColorFactor",
        ),
    ],
)
def test_builder_rejects_material_bytes_that_hide_non_matte_contributions(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    _mutate_visual_and_rebind_report(
        fixture, lambda document: mutation(document["materials"][0])
    )

    with pytest.raises(builder.AppearanceVariantInputError, match=message):
        _build(fixture)


def test_builder_accepts_equivalent_gltf_default_matte_values(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def use_gltf_defaults(document: dict[str, Any]) -> None:
        material = document["materials"][0]
        material.pop("alphaMode")
        material.pop("emissiveFactor")
        material["pbrMetallicRoughness"].pop("baseColorFactor")

    _mutate_visual_and_rebind_report(fixture, use_gltf_defaults)

    spec, lineage = _build(fixture)
    assert spec["identity"]["asset_id"]
    assert lineage["qualification_claim"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.pop("extensionsUsed"),
        lambda document: document.__setitem__(
            "extensionsUsed", ["KHR_materials_specular", "KHR_unreviewed_extension"]
        ),
    ],
)
def test_builder_rejects_missing_or_tampered_root_material_extension_declaration(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], Any],
) -> None:
    fixture = _fixture(tmp_path)
    _mutate_visual_and_rebind_report(fixture, mutation)

    with pytest.raises(
        builder.AppearanceVariantInputError,
        match="root extensionsUsed must be exactly",
    ):
        _build(fixture)


def test_cross_ordinal_spec_lineage_pair_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    spec_one, _lineage_one = _build(fixture, 1)
    _spec_two, lineage_two = _build(fixture, 2)

    with pytest.raises(builder.AppearanceVariantInputError, match="exact derived"):
        _validate_metadata_only_fixture_lineage(spec_one, lineage_two)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["inputs"]["template_variant_spec"].__setitem__(
                "sha256", "00" * 32
            ),
            "exact file bytes",
        ),
        (
            lambda value: value["inputs"]["template_variant_spec"].__setitem__(
                "canonical_content_sha256", "00" * 32
            ),
            "exact snapshot",
        ),
        (
            lambda value: value["inputs"]["template_variant_spec"]["snapshot"][
                "identity"
            ].__setitem__("asset_id", "different"),
            "exact snapshot",
        ),
    ],
)
def test_template_raw_canonical_and_snapshot_bindings_are_enforced(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    spec, lineage = _build(fixture)
    mutation(lineage)
    _rehash_lineage(lineage)

    with pytest.raises(builder.AppearanceVariantInputError, match=message):
        _validate_metadata_only_fixture_lineage(spec, lineage)


@pytest.mark.parametrize("field", ["asset_id", "sha256", "canonical_content_sha256"])
def test_derived_spec_raw_canonical_and_asset_id_bindings_are_enforced(
    tmp_path: Path, field: str
) -> None:
    fixture = _fixture(tmp_path)
    spec, lineage = _build(fixture)
    lineage["derivative"]["derived_variant_spec"][field] = (
        "different" if field == "asset_id" else "00" * 32
    )
    _rehash_lineage(lineage)

    with pytest.raises(builder.AppearanceVariantInputError, match="exact derived"):
        _validate_metadata_only_fixture_lineage(spec, lineage)


def test_lineage_inputs_and_serialized_spec_cannot_be_detached(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    spec, lineage = _build(fixture)
    detached = deepcopy(lineage)
    detached.pop("inputs")
    _rehash_lineage(detached)
    with pytest.raises(builder.AppearanceVariantInputError, match="lineage.inputs"):
        _validate_metadata_only_fixture_lineage(
            spec,
            detached,
            audit_lineage=lineage,
        )

    with pytest.raises(builder.AppearanceVariantInputError, match="serialized"):
        _validate_metadata_only_fixture_lineage(
            spec, lineage, serialized_spec=_pretty_payload({"different": True})
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["inputs"].pop("appearance_batch"),
            "input fields",
        ),
        (
            lambda value: value["derivative"].pop("tool_identity"),
            "derivative fields",
        ),
        (
            lambda value: value["inputs"]["appearance_realization_report"].__setitem__(
                "sha256", "00" * 32
            ),
            "exact file bytes",
        ),
        (
            lambda value: value["derivative"]["pre_rebase_visual_glb"].__setitem__(
                "sha256", "00" * 32
            ),
            "exact file bytes",
        ),
        (
            lambda value: value["derivative"].__setitem__(
                "tool_identity", {"detached": True}
            ),
            "tool identity differs",
        ),
    ],
)
def test_all_lineage_inputs_and_derivatives_are_required_and_bound(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    spec, lineage = _build(fixture)
    mutation(lineage)
    _rehash_lineage(lineage)

    with pytest.raises(builder.AppearanceVariantInputError, match=message):
        _validate_metadata_only_fixture_lineage(spec, lineage)


def test_spec_must_be_the_exact_template_request_derivation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    spec, lineage = _build(fixture)
    spec["identity"]["morphotype_id"] = "detached"
    payload = _pretty_payload(spec)
    binding = lineage["derivative"]["derived_variant_spec"]
    binding["byte_size"] = len(payload)
    binding["sha256"] = hashlib.sha256(payload).hexdigest()
    binding["canonical_content_sha256"] = canonical_json_sha256(spec)
    _rehash_lineage(lineage)

    with pytest.raises(builder.AppearanceVariantInputError, match="exact derived"):
        _validate_metadata_only_fixture_lineage(
            spec,
            lineage,
            serialized_spec=payload,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report["batch"].pop("path"), "batch.path"),
        (lambda report: report.pop("tool_identity"), "tool_identity"),
        (
            lambda report: report["tool_identity"].__setitem__("sha256", "00" * 32),
            "approved realizer",
        ),
        (
            lambda report: report["source"].__setitem__("sha256", "00" * 32),
            "exact file bytes",
        ),
        (
            lambda report: report["output"]["readback_audit"][
                "action_invariants"
            ].__setitem__("maximum_error", 5.1e-5),
            "maximum_error",
        ),
        (
            lambda report: report["realization"].__setitem__(
                "topology_uv_skin_unchanged", False
            ),
            "topology/UV/skin",
        ),
        (
            lambda report: report["realization"]["shape"].__setitem__(
                "head_weighted_radius_rms_ratio", 999.0
            ),
            "head_weighted_radius_rms_ratio",
        ),
        (
            lambda report: report["output"]["readback_audit"]["material_invariants"][
                "texture_images"
            ].pop("normal"),
            "normal and specular",
        ),
    ],
)
def test_report_tool_source_and_full_gates_fail_closed(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    report = deepcopy(fixture["report"])
    mutation(report)
    _rehash_report(report)
    _write_json(fixture["report_path"], report)

    with pytest.raises(builder.AppearanceVariantInputError, match=message):
        _build(fixture)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("schema", None), "schema"),
        (lambda value: value.__setitem__("schema", "unrelated_v1"), "schema"),
        (
            lambda value: value.__setitem__("qualification_claim", True),
            "non-qualifying",
        ),
        (
            lambda value: value["source_repository"].__setitem__(
                "revision", "different"
            ),
            "source revision",
        ),
        (lambda value: value.__setitem__("source_artifacts", []), "cannot be empty"),
    ],
)
def test_upstream_source_schema_qualification_and_revision_are_strict(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    upstream = deepcopy(fixture["upstream"])
    mutation(upstream)
    _write_json(fixture["upstream_path"], upstream)

    with pytest.raises(builder.AppearanceVariantInputError, match=message):
        _build(fixture)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["taxonomy"].__setitem__("species_id", "felis_catus"),
            "taxonomy pair",
        ),
        (
            lambda value: value["taxonomy"].__setitem__("breed_id", "other"),
            "taxonomy pair",
        ),
        (
            lambda value: value["identity"].__setitem__(
                "body_plan_id", "quadruped_mammal_felid_v1"
            ),
            "identity.body_plan_id",
        ),
        (
            lambda value: value["anchors"][0].__setitem__("joint_id", "unknown_joint"),
            "unknown visual joint",
        ),
    ],
)
def test_template_taxonomy_body_plan_and_anchor_joints_are_strict(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    template = deepcopy(fixture["template"])
    mutation(template)
    _write_json(fixture["template_path"], template)

    with pytest.raises(builder.AppearanceVariantInputError, match=message):
        _build(fixture)


@pytest.mark.parametrize("kind", ["duplicate", "nonfinite"])
def test_strict_json_rejects_duplicate_keys_and_nonfinite(
    tmp_path: Path, kind: str
) -> None:
    fixture = _fixture(tmp_path)
    original = fixture["upstream_path"].read_text(encoding="utf-8")
    prefix = '"schema":"ignored",' if kind == "duplicate" else '"bad":NaN,'
    fixture["upstream_path"].write_text(
        "{" + prefix + original.lstrip()[1:], encoding="utf-8"
    )

    with pytest.raises(
        builder.AppearanceVariantInputError,
        match="duplicate key|non-finite",
    ):
        _build(fixture)


def test_input_and_output_ancestor_symlinks_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    linked_inputs = tmp_path / "linked-inputs"
    linked_inputs.symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(builder.AppearanceVariantInputError, match="symbolic link"):
        builder.build_inputs(
            batch_path=linked_inputs / "batch.json",
            ordinal=1,
            appearance_report_path=fixture["report_path"],
            template_spec_path=fixture["template_path"],
            upstream_source_manifest_path=fixture["upstream_path"],
        )

    spec, lineage = _build(fixture)
    real_outputs = tmp_path / "real-outputs"
    real_outputs.mkdir()
    linked_outputs = tmp_path / "linked-outputs"
    linked_outputs.symlink_to(real_outputs, target_is_directory=True)
    with pytest.raises(builder.AppearanceVariantInputError, match="symbolic link"):
        _write_metadata_only_fixture_pair(
            spec_output=linked_outputs / "spec.json",
            lineage_output=tmp_path / "lineage.json",
            spec=spec,
            lineage=lineage,
        )
    assert list(real_outputs.iterdir()) == []


def test_input_ancestor_swap_cannot_redirect_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    safe = tmp_path / "safe"
    evil = tmp_path / "evil"
    moved = tmp_path / "moved-safe"
    safe.mkdir()
    evil.mkdir()
    (safe / "value.json").write_text('{"value":"safe"}\n', encoding="utf-8")
    (evil / "value.json").write_text('{"value":"evil"}\n', encoding="utf-8")
    original = builder._absolute_without_symlinks
    swapped = False

    def swap_after_check(path: str | Path, *, owner: str) -> Path:
        nonlocal swapped
        result = original(path, owner=owner)
        if owner == "race input" and not swapped:
            safe.rename(moved)
            safe.symlink_to(evil, target_is_directory=True)
            swapped = True
        return result

    monkeypatch.setattr(builder, "_absolute_without_symlinks", swap_after_check)
    with pytest.raises(builder.AppearanceVariantInputError, match="without symbolic"):
        builder._snapshot_file(safe / "value.json", "race input")


def test_pair_preflight_never_leaves_half_or_replaces_existing(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    spec, lineage = _build(fixture)
    spec_path = tmp_path / "outputs" / "spec.json"
    lineage_path = tmp_path / "outputs" / "lineage.json"
    lineage_path.parent.mkdir()
    lineage_path.write_text("caller-owned\n", encoding="utf-8")

    with pytest.raises(
        builder.AppearanceVariantInputError, match="refusing to replace"
    ):
        _write_metadata_only_fixture_pair(
            spec_output=spec_path,
            lineage_output=lineage_path,
            spec=spec,
            lineage=lineage,
        )
    assert not spec_path.exists()
    assert lineage_path.read_text(encoding="utf-8") == "caller-owned\n"

    same = tmp_path / "same.json"
    with pytest.raises(builder.AppearanceVariantInputError, match="must differ"):
        _write_metadata_only_fixture_pair(
            spec_output=same,
            lineage_output=same,
            spec=spec,
            lineage=lineage,
        )
    assert not same.exists()


def test_pair_rolls_back_first_file_when_second_reservation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    spec, lineage = _build(fixture)
    spec_path = tmp_path / "spec.json"
    lineage_path = tmp_path / "lineage.json"
    original_reserve = builder._reserve_output

    def fail_second(path: Path, *, owner: str):
        if path == lineage_path:
            raise OSError("synthetic second reservation failure")
        return original_reserve(path, owner=owner)

    monkeypatch.setattr(builder, "_reserve_output", fail_second)
    with pytest.raises(builder.AppearanceVariantInputError, match="second reservation"):
        _write_metadata_only_fixture_pair(
            spec_output=spec_path,
            lineage_output=lineage_path,
            spec=spec,
            lineage=lineage,
        )
    assert not spec_path.exists()
    assert not lineage_path.exists()


def test_pair_rolls_back_when_reservation_fstat_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    spec, lineage = _build(fixture)
    spec_path = tmp_path / "spec.json"
    lineage_path = tmp_path / "lineage.json"
    original_fstat = builder.os.fstat

    def fail_spec(descriptor: int):
        target = Path(f"/proc/self/fd/{descriptor}").resolve()
        if target == spec_path:
            raise OSError("synthetic reservation fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(builder.os, "fstat", fail_spec)
    with pytest.raises(builder.AppearanceVariantInputError, match="fstat failure"):
        _write_metadata_only_fixture_pair(
            spec_output=spec_path,
            lineage_output=lineage_path,
            spec=spec,
            lineage=lineage,
        )
    assert not spec_path.exists()
    assert not lineage_path.exists()


def test_pair_rolls_back_both_files_when_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    spec, lineage = _build(fixture)
    spec_path = tmp_path / "spec.json"
    lineage_path = tmp_path / "lineage.json"
    original_fsync = builder.os.fsync
    calls = 0

    def fail_second(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(builder.os, "fsync", fail_second)
    with pytest.raises(builder.AppearanceVariantInputError, match="fsync failure"):
        _write_metadata_only_fixture_pair(
            spec_output=spec_path,
            lineage_output=lineage_path,
            spec=spec,
            lineage=lineage,
        )
    assert not spec_path.exists()
    assert not lineage_path.exists()


def test_pair_rolls_back_when_final_parser_rejects_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    spec, lineage = _build(fixture)
    spec_path = tmp_path / "spec.json"
    lineage_path = tmp_path / "lineage.json"

    def reject(_path: Path) -> None:
        raise VariantPackageError("synthetic final parser rejection")

    monkeypatch.setattr(builder, "load_variant_package_spec", reject)
    with pytest.raises(builder.AppearanceVariantInputError, match="final parser"):
        _write_metadata_only_fixture_pair(
            spec_output=spec_path,
            lineage_output=lineage_path,
            spec=spec,
            lineage=lineage,
        )
    assert not spec_path.exists()
    assert not lineage_path.exists()


def test_pair_success_readback_and_rerun_are_exclusive(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    spec, lineage = _build(fixture)
    spec_path = tmp_path / "out" / "spec.json"
    lineage_path = tmp_path / "out" / "lineage.json"

    emitted_spec, emitted_lineage = _write_metadata_only_fixture_pair(
        spec_output=spec_path,
        lineage_output=lineage_path,
        spec=spec,
        lineage=lineage,
    )
    spec_bytes = emitted_spec.read_bytes()
    lineage_bytes = emitted_lineage.read_bytes()
    _validate_metadata_only_fixture_lineage(
        json.loads(spec_bytes), json.loads(lineage_bytes), serialized_spec=spec_bytes
    )

    with pytest.raises(
        builder.AppearanceVariantInputError, match="refusing to replace"
    ):
        _write_metadata_only_fixture_pair(
            spec_output=spec_path,
            lineage_output=lineage_path,
            spec=spec,
            lineage=lineage,
        )
    assert spec_path.read_bytes() == spec_bytes
    assert lineage_path.read_bytes() == lineage_bytes
