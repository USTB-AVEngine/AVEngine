from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any

import pytest

import avengine.assets.package as package_module
from avengine.assets.actions import (
    TICKS_PER_SAMPLE,
    TIME_BASE_HZ,
    BakedActionClip,
    BakedActionSet,
    serialize_baked_actions_npz,
)
from avengine.assets.contracts import (
    CONTACT_ORDER,
    REQUIRED_FILE_ROLES,
    validate_animal_asset_package,
)
from avengine.assets.glb import load_glb
from avengine.assets.package import (
    AnimalPackageIdentity,
    PackageCompileError,
    compile_research_candidate_animal_package,
)


_JSON_CHUNK_TYPE = 0x4E4F534A
_BIN_CHUNK_TYPE = 0x004E4942
_IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _pad(payload: bytes, fill: bytes) -> bytes:
    return payload + fill * ((-len(payload)) % 4)


def _build_visual_glb() -> bytes:
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "m2-package-unit-test"},
        "scene": 0,
        "scenes": [{"nodes": [0, 2]}],
        "nodes": [
            {
                "name": "root",
                "children": [1],
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            {
                "name": "paw",
                "translation": [0.0, -0.5, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            {"name": "skinned_mesh", "mesh": 0, "skin": 0},
        ],
        "skins": [{"name": "animal_skin", "skeleton": 0, "joints": [0, 1]}],
    }
    binary = bytearray()

    def append_accessor(
        *,
        component_type: int,
        element_type: str,
        values: list[tuple[int | float, ...]],
        fmt: str,
        target: int | None = None,
    ) -> int:
        component_size = struct.calcsize("<" + fmt[0])
        while len(binary) % component_size:
            binary.append(0)
        offset = len(binary)
        packer = struct.Struct("<" + fmt)
        for value in values:
            binary.extend(packer.pack(*value))
        view_index = len(document.setdefault("bufferViews", []))
        view: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(binary) - offset,
        }
        if target is not None:
            view["target"] = target
        document["bufferViews"].append(view)
        accessor_index = len(document.setdefault("accessors", []))
        document["accessors"].append(
            {
                "bufferView": view_index,
                "componentType": component_type,
                "count": len(values),
                "type": element_type,
            }
        )
        return accessor_index

    positions = [(-0.5, 0.0, -0.2), (0.5, 0.0, 0.2), (0.0, 0.8, -0.1)]
    position = append_accessor(
        component_type=5126,
        element_type="VEC3",
        values=positions,
        fmt="fff",
        target=34962,
    )
    normal = append_accessor(
        component_type=5126,
        element_type="VEC3",
        values=[(0.0, 1.0, 0.0)] * 3,
        fmt="fff",
        target=34962,
    )
    texcoord = append_accessor(
        component_type=5126,
        element_type="VEC2",
        values=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
        fmt="ff",
        target=34962,
    )
    joints = append_accessor(
        component_type=5121,
        element_type="VEC4",
        values=[(0, 1, 0, 0)] * 3,
        fmt="BBBB",
        target=34962,
    )
    weights = append_accessor(
        component_type=5126,
        element_type="VEC4",
        values=[(0.75, 0.25, 0.0, 0.0)] * 3,
        fmt="ffff",
        target=34962,
    )
    indices = append_accessor(
        component_type=5123,
        element_type="SCALAR",
        values=[(0,), (1,), (2,)],
        fmt="H",
        target=34963,
    )
    document["accessors"][position]["min"] = [-0.5, 0.0, -0.2]
    document["accessors"][position]["max"] = [0.5, 0.8, 0.2]
    document["meshes"] = [
        {
            "name": "animal_mesh",
            "primitives": [
                {
                    "attributes": {
                        "POSITION": position,
                        "NORMAL": normal,
                        "TEXCOORD_0": texcoord,
                        "JOINTS_0": joints,
                        "WEIGHTS_0": weights,
                    },
                    "indices": indices,
                    "mode": 4,
                }
            ],
        }
    ]
    document["buffers"] = [{"byteLength": len(binary)}]
    json_payload = _pad(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        b" ",
    )
    bin_payload = _pad(bytes(binary), b"\0")
    chunks = (
        struct.pack("<II", len(json_payload), _JSON_CHUNK_TYPE)
        + json_payload
        + struct.pack("<II", len(bin_payload), _BIN_CHUNK_TYPE)
        + bin_payload
    )
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def _action_clip(action_id: str, source_name: str) -> BakedActionClip:
    sample_count = 20
    duration_ticks = sample_count * TICKS_PER_SAMPLE
    ticks = tuple(range(0, duration_ticks, TICKS_PER_SAMPLE))
    return BakedActionClip(
        semantic_action_id=action_id,
        source_action_name=source_name,
        clip_start_seconds=0.0,
        clip_end_seconds=duration_ticks / TIME_BASE_HZ,
        loop_duration_ticks=duration_ticks,
        sample_ticks=ticks,
        source_times_seconds=tuple(tick / TIME_BASE_HZ for tick in ticks),
        rotations_xyzw=tuple((_IDENTITY,) for _ in ticks),
    )


def _actions(visual_sha256: str) -> BakedActionSet:
    return BakedActionSet(
        source_glb_sha256=visual_sha256,
        runtime_joint_order=("paw",),
        actions=(
            _action_clip("idle", "Idle"),
            _action_clip("walk", "Walking"),
        ),
    )


def _reference(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "byte_size": len(payload),
        "sha256": _sha256(payload),
    }


def _anchors() -> list[dict[str, Any]]:
    joints = {
        "body": "root",
        "head": "paw",
        "muzzle": "paw",
        **{contact_id: "paw" for contact_id in CONTACT_ORDER},
    }
    return [
        {
            "anchor_id": anchor_id,
            "joint_id": joint_id,
            "joint_from_anchor": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
        for anchor_id, joint_id in joints.items()
    ]


@dataclass(frozen=True)
class _Fixture:
    root: Path
    identity: AnimalPackageIdentity
    visual: Path
    rebase: Path
    rebase_deformation: Path
    action_report: Path
    static_qa: Path
    deformation_qa: Path
    animation_qa: Path
    habitat_static_probe: Path
    habitat_animation_review: Path
    actions: Path
    contacts: Path
    source: Path
    license_snapshot: Path

    def arguments(self, output: Path) -> dict[str, Any]:
        return {
            "output_directory": output,
            "identity": self.identity,
            "visual_glb": self.visual,
            "rebase_report": self.rebase,
            "rebase_deformation_report": self.rebase_deformation,
            "action_report": self.action_report,
            "static_qa": self.static_qa,
            "deformation_qa": self.deformation_qa,
            "animation_qa": self.animation_qa,
            "habitat_static_probe": self.habitat_static_probe,
            "habitat_animation_review": self.habitat_animation_review,
            "baked_actions": self.actions,
            "contacts": self.contacts,
            "anchor_definitions": _anchors(),
            "source_manifest": self.source,
            "license_snapshot": self.license_snapshot,
        }


def _fixture(root: Path) -> _Fixture:
    root.mkdir()
    visual = root / "visual.glb"
    visual.write_bytes(_build_visual_glb())
    visual_reference = _reference(visual)

    rebase = root / "rebase.json"
    _write_json(
        rebase,
        {
            "schema": "avengine_m2_skin_root_rebase_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "output": visual_reference,
            "skin": {
                "root_joint": "root",
                "actor_from_canonical_root": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
        },
    )
    rebase_reference = _reference(rebase)

    rebase_deformation = root / "rebase_deformation.json"
    _write_json(
        rebase_deformation,
        {
            "schema": "avengine_m2_rebase_deformation_verification_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "rebased": visual_reference,
            "rebase_report": rebase_reference,
            "maximum_vertex_error_m": 0.0,
            "threshold_maximum_vertex_error_m": 0.0001,
            "samples": [{"semantic": "idle"}, {"semantic": "walk"}],
        },
    )

    action_set = _actions(visual_reference["sha256"])
    actions = root / "actions.npz"
    actions.write_bytes(serialize_baked_actions_npz(action_set))
    actions_reference = _reference(actions)
    action_report = root / "action_report.json"
    _write_json(
        action_report,
        {
            "schema": "avengine_m2_action_bake_report_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "source_glb": visual_reference,
            "artifact": {
                **actions_reference,
                "canonical_content_sha256": actions_reference["sha256"],
                "readback_equal": True,
            },
            "runtime_joint_order": ["paw"],
            "actions": [
                {
                    "semantic_action_id": clip.semantic_action_id,
                    "source_action_name": clip.source_action_name,
                    "sample_count": clip.sample_count,
                    "loop_duration_ticks": clip.loop_duration_ticks,
                }
                for clip in action_set.actions
            ],
        },
    )

    habitat_static_probe = root / "habitat_static_probe.json"
    _write_json(
        habitat_static_probe,
        {
            "schema": "avengine_m2_habitat_skin_rest_probe_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "input": visual_reference,
            "gates": {
                "bootstrap_visible": True,
                "all_six_orbit_views_visible": True,
                "co_located_modalities": True,
                "runtime_joint_mapping_complete": True,
                "runtime_link_bind_alignment": True,
            },
        },
    )

    habitat_animation_review = root / "habitat_animation_review.json"
    _write_json(
        habitat_animation_review,
        {
            "schema": "avengine_m2_habitat_action_review_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "source": {
                "visual_glb": visual_reference,
                "actions_npz": actions_reference,
                "rebase_report": rebase_reference,
            },
            "runtime_contract": {"runtime_joint_order": ["paw"]},
            "capture_contract": {
                "formal_capture": False,
                "co_located_and_co_oriented": True,
                "world_time_unchanged": True,
            },
            "runs": [
                {
                    "semantic_action_id": clip.semantic_action_id,
                    "source_action_name": clip.source_action_name,
                    "sample_count": clip.sample_count,
                    "all_frames_visible": True,
                    "minimum_semantic_pixel_count": 32,
                }
                for clip in action_set.actions
            ],
        },
    )

    topology_sha256, uv_sha256, weights_sha256, _ = package_module._mesh_evidence(
        load_glb(visual)
    )
    static_qa = root / "static_qa.json"
    _write_json(
        static_qa,
        {
            "schema": "avengine_m2_static_geometry_qa_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "source_glb_sha256": visual_reference["sha256"],
            "joint_count": 2,
            "primitive_count": 1,
            "primitives": [
                {
                    "primitive_index": 0,
                    "vertex_count": 3,
                    "triangle_count": 1,
                    "minimum_triangle_area_m2": 0.1,
                    "maximum_weight_sum_error": 0.0,
                    "maximum_weighted_bind_vertex_error_m": 0.0,
                }
            ],
            "maximum_bind_closure_error": 0.0,
            "maximum_rest_landmark_bbox_outside_distance_m": 0.0,
            "topology_sha256": topology_sha256,
            "uv_sha256": uv_sha256,
            "weights_sha256": weights_sha256,
            "thresholds": {
                "maximum_weight_sum_error": 1.0e-5,
                "maximum_bind_closure_error_m": 1.0e-4,
                "minimum_triangle_area_m2_exclusive": 1.0e-12,
                "maximum_landmark_bbox_outside_distance_m": 0.02,
            },
        },
    )
    deformation_qa = root / "deformation_qa.json"
    _write_json(
        deformation_qa,
        {
            "schema": "avengine_m2_deformation_qa_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "source_glb_sha256": visual_reference["sha256"],
            "baked_actions_sha256": actions_reference["sha256"],
            "rest_bbox_diagonal_m": 1.0,
            "maximum_vertex_step_m": 0.01,
            "maximum_source_loop_endpoint_vertex_error_m": 0.0,
            "minimum_animated_triangle_area_m2": 0.1,
            "maximum_joint_landmark_bbox_outside_distance_m": 0.0,
            "actions": [
                {
                    "semantic_action_id": clip.semantic_action_id,
                    "source_action_name": clip.source_action_name,
                    "sample_count": clip.sample_count,
                    "minimum_triangle_area_m2": 0.1,
                    "maximum_joint_landmark_bbox_outside_distance_m": 0.0,
                    "maximum_vertex_step_rest_diagonal_ratio": 0.01,
                    "source_loop_endpoint_vertex_error_m": 0.0,
                    "source_loop_endpoint_maximum_joint_rotation_error": 0.0,
                    "source_loop_endpoint_maximum_joint_translation_error_m": 0.0,
                    "source_loop_endpoint_maximum_joint_scale_error": 0.0,
                }
                for clip in action_set.actions
            ],
            "thresholds": {
                "maximum_vertex_step_rest_diagonal_ratio": 0.1,
                "maximum_source_loop_endpoint_vertex_error_m": 1.0e-4,
                "maximum_source_loop_endpoint_joint_translation_error_m": 1.0e-4,
                "maximum_source_loop_endpoint_joint_rotation_error": 1.0e-5,
                "maximum_source_loop_endpoint_joint_scale_error": 1.0e-5,
                "minimum_triangle_area_m2_exclusive": 1.0e-12,
                "maximum_landmark_bbox_outside_distance_m": 0.02,
            },
        },
    )
    animation_qa = root / "animation_qa.json"
    _write_json(
        animation_qa,
        {
            "schema": "avengine_m2_animation_qa_v1",
            "status": "pass",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "source_glb_sha256": visual_reference["sha256"],
            "baked_actions_sha256": actions_reference["sha256"],
            "sample_rate_hz": action_set.sample_rate_hz,
            "time_base_hz": action_set.time_base_hz,
            "runtime_joint_order": ["paw"],
            "actions": [
                {
                    "semantic_action_id": clip.semantic_action_id,
                    "source_action_name": clip.source_action_name,
                    "sample_count": clip.sample_count,
                    "loop_duration_ticks": clip.loop_duration_ticks,
                    "first_sample_tick": clip.sample_ticks[0],
                    "last_sample_tick": clip.sample_ticks[-1],
                }
                for clip in action_set.actions
            ],
            "mouth": {
                "joint_id": "paw",
                "open_ratio_policy": "exactly_zero",
                "rotation_excursion_degrees_by_action": {
                    "idle": 0.0,
                    "walk": 0.0,
                },
                "maximum_rotation_excursion_degrees": 0.0,
                "threshold_degrees": 1.0e-6,
            },
            "semantic_terminal_motion": {
                "walking_summary": {
                    "legacy_hind_gait_metric_triggered": True,
                    "mean_front_paw_forward_range_m": 0.2,
                    "mean_hind_paw_forward_range_m": 0.01,
                    "mean_hind_paw_lateral_range_m": 0.1,
                }
            },
            "known_limitations": ["Fixture retains a known hind-gait limitation."],
            "human_visual_review_required": True,
        },
    )

    contacts = root / "contacts.json"
    _write_json(
        contacts,
        {
            "schema": "avengine_m2_contact_phases_v1",
            "qualification_state": "research_candidate",
            "qualification_claim": False,
            "source_glb_sha256": visual_reference["sha256"],
            "baked_actions_sha256": actions_reference["sha256"],
            "runtime_joint_order": ["paw"],
            "coordinate_system": {
                "handedness": "right",
                "up_axis": "+Y",
                "forward_axis": "-Z",
                "linear_unit": "meter",
                "quaternion_order": "xyzw",
            },
            "sample_rate_hz": action_set.sample_rate_hz,
            "time_base_hz": action_set.time_base_hz,
            "contact_order": CONTACT_ORDER,
            "anchor_definitions": [
                anchor
                for contact_id in CONTACT_ORDER
                for anchor in _anchors()
                if anchor["anchor_id"] == contact_id
            ],
            "thresholds": {},
            "actions": [
                {
                    "semantic_action_id": clip.semantic_action_id,
                    "source_action_name": clip.source_action_name,
                    "sample_count": clip.sample_count,
                    "frames": [
                        {
                            "sample_index": index,
                            "sample_tick": clip.sample_ticks[index],
                            "source_time_seconds": clip.source_times_seconds[index],
                            "contacts": [
                                {"contact_id": contact_id, "in_contact": True}
                                for contact_id in CONTACT_ORDER
                            ],
                        }
                        for index in range(20)
                    ],
                }
                for clip in action_set.actions
            ],
            "warnings": [],
        },
    )

    source = root / "source_manifest.json"
    _write_json(
        source,
        {
            "schema": "avengine_m2_source_snapshot_v1",
            "formal_dataset_registration_authorized": False,
            "source_visual": visual_reference,
        },
    )
    license_snapshot = root / "license_snapshot.json"
    _write_json(
        license_snapshot,
        {
            "schema": "avengine_m2_license_snapshot_v1",
            "license": "MIT",
            "allowed_use": "research_canary",
            "redistribution": "allowed",
        },
    )
    identity = AnimalPackageIdentity(
        asset_id="fixture_dog_v1",
        template_id="fixture_template_v1",
        body_plan_id="quadruped_dog",
        morphotype_id="fixture",
        skeleton_revision="skeleton-fixture-v1",
        weights_revision="weights-fixture-v1",
        collision_revision="bbox-canary-v1",
        action_revision="actions-fixture-v1",
        source="unit-test fixture",
        source_revision="fixture-v1",
        license="MIT",
        allowed_use="research_canary",
        redistribution="allowed",
    )
    return _Fixture(
        root=root,
        identity=identity,
        visual=visual,
        rebase=rebase,
        rebase_deformation=rebase_deformation,
        action_report=action_report,
        static_qa=static_qa,
        deformation_qa=deformation_qa,
        animation_qa=animation_qa,
        habitat_static_probe=habitat_static_probe,
        habitat_animation_review=habitat_animation_review,
        actions=actions,
        contacts=contacts,
        source=source,
        license_snapshot=license_snapshot,
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _report_artifact_record(
    path: Path,
    *,
    report_path: Path,
    snapshot: Any | None = None,
) -> dict[str, Any]:
    record = _reference(path)
    record["path"] = path.relative_to(report_path.parent).as_posix()
    if snapshot is not None:
        record["snapshot"] = snapshot
    return record


def _runtime_binding_fixture() -> dict[str, Any]:
    return {
        "runtime_joint_order": ["paw"],
        "joint_position_count": 4,
        "quaternion_order": "xyzw",
        "links": [
            {
                "link_name": "paw",
                "joint_position_offset": 0,
                "joint_position_count": 4,
            }
        ],
    }


def _bind_local_runtime_artifact_integrity(
    fixture: _Fixture, *, shader_type: str
) -> None:
    """Build local integrity fixtures; this does not emulate trusted execution."""

    config = package_module.build_habitat_ao_config_data(
        render_asset="visual.glb",
        urdf_filepath="animal.urdf",
        semantic_id=fixture.identity.semantic_id,
        shader_type=shader_type,
    )
    repository = Path(__file__).resolve().parents[2]
    bindings = (
        (
            fixture.habitat_static_probe,
            repository / "tools/assets/probe_habitat_skin_rest.py",
            "static",
        ),
        (
            fixture.habitat_animation_review,
            repository / "tools/assets/render_habitat_action_review.py",
            "animation",
        ),
    )
    for report_path, producer_path, report_kind in bindings:
        artifact_root = report_path.parent / f"{report_path.stem}_artifacts"
        artifact_root.mkdir()
        config_path = artifact_root / "animal.ao_config.json"
        _write_json(config_path, config)
        runtime_binding = _runtime_binding_fixture()
        runtime_binding_path = artifact_root / "habitat_runtime_binding.json"
        _write_json(runtime_binding_path, runtime_binding)
        report = _load_json(report_path)
        report["evidence_scope"] = {
            "local_report_claim": "artifact_integrity_only",
            "trusted_runtime_attestation": False,
            "runtime_execution_conclusion_source": "external_capture_audit_only",
        }
        report["producer_source_integrity"] = _reference(producer_path)
        report["render_configuration_integrity"] = {
            "configured_shader_type": shader_type,
            "ao_config_artifact": _report_artifact_record(
                config_path,
                report_path=report_path,
                snapshot=config,
            ),
        }
        if report_kind == "static":
            observation_names = package_module._STATIC_PROBE_OBSERVATION_PATHS
            observation_paths = [artifact_root / name for name in observation_names]
            for path in observation_paths:
                path.write_bytes(f"fixture observation: {path.name}\n".encode())
            report["runtime"] = {"joint_position_count": 4}
        else:
            report["runtime_contract"]["joint_position_count"] = 4
            observation_paths = []
            for index, run in enumerate(report["runs"]):
                video = artifact_root / f"run_{index}_review.mp4"
                contact_sheet = artifact_root / f"run_{index}_contact_sheet.png"
                video.write_bytes(f"fixture video {index}\n".encode())
                contact_sheet.write_bytes(f"fixture sheet {index}\n".encode())
                run["video"] = _report_artifact_record(video, report_path=report_path)
                run["contact_sheet"] = _report_artifact_record(
                    contact_sheet, report_path=report_path
                )
                observation_paths.extend([video, contact_sheet])
        report["runtime_artifact_integrity"] = {
            "runtime_binding_artifact": _report_artifact_record(
                runtime_binding_path,
                report_path=report_path,
                snapshot=runtime_binding,
            ),
            "observation_artifacts": [
                _report_artifact_record(path, report_path=report_path)
                for path in observation_paths
            ],
        }
        _write_json(report_path, report)


def _offline_relabel_legacy_reports_as_pbr(fixture: _Fixture) -> None:
    """Reproduce the unsafe old helper: patch report JSON without execution."""

    config = package_module.build_habitat_ao_config_data(
        render_asset="visual.glb",
        urdf_filepath="animal.urdf",
        semantic_id=fixture.identity.semantic_id,
        shader_type="pbr",
    )
    repository = Path(__file__).resolve().parents[2]
    for report_path, producer_path in (
        (
            fixture.habitat_static_probe,
            repository / "tools/assets/probe_habitat_skin_rest.py",
        ),
        (
            fixture.habitat_animation_review,
            repository / "tools/assets/render_habitat_action_review.py",
        ),
    ):
        config_path = report_path.parent / f"{report_path.stem}_relabel_config.json"
        _write_json(config_path, config)
        report = _load_json(report_path)
        report["producer"] = _reference(producer_path)
        report["rendering"] = {
            "shader_type": "pbr",
            "ao_config": {
                **_reference(config_path),
                "snapshot": config,
            },
        }
        _write_json(report_path, report)


def test_compiler_emits_complete_candidate_without_implicit_promotion(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    manifest_path = compile_research_candidate_animal_package(
        **fixture.arguments(tmp_path / "package")
    )

    manifest = _load_json(manifest_path)
    assert validate_animal_asset_package(manifest, manifest_path=manifest_path) == []
    assert manifest["admission_state"] == "research_candidate"
    assert manifest["qualification"] == {
        "automatic_qa_status": "pass",
        "human_visual_review_status": "not_run",
        "human_review_binding_sha256": None,
        "decision_reason": (
            "Hash-closed local technical artifacts passed structural checks; the "
            "probe/review reports are not trusted runtime attestations. External "
            "capture/audit and human visual review have not established formal "
            "qualification, so this package remains a research candidate."
        ),
    }
    records = {record["role"]: record for record in manifest["files"]}
    assert set(records) == REQUIRED_FILE_ROLES
    assert (manifest_path.parent / records["idle_poses"]["path"]).read_bytes() == (
        manifest_path.parent / records["walk_poses"]["path"]
    ).read_bytes()
    action_manifest = _load_json(manifest_path.parent / "actions/action_manifest.json")
    assert [
        action["selected_member"]["semantic_action_id"]
        for action in action_manifest["actions"]
    ] == ["idle", "walk"]
    assert (
        _load_json(manifest_path.parent / "qa/static_geometry.json")["schema"]
        == "avengine_m2_static_geometry_qa_v1"
    )
    assert (
        _load_json(manifest_path.parent / "qa/deformation.json")["schema"]
        == "avengine_m2_deformation_qa_v1"
    )
    assert (
        _load_json(manifest_path.parent / "qa/animation.json")["schema"]
        == "avengine_m2_animation_qa_v1"
    )

    collision = load_glb(manifest_path.parent / "collision_proxy.glb")
    extras = collision.json["asset"]["extras"]
    assert extras["kinematic_canary_only"] is True
    assert extras["used_for_physics"] is False
    assert extras["used_for_contact_inference"] is False
    assert collision.sha256 != _sha256(fixture.visual.read_bytes())
    assert (
        _load_json(manifest_path.parent / "habitat/animal.ao_config.json")[
            "shader_type"
        ]
        == "phong"
    )


def test_compiler_emits_explicit_pbr_from_hash_closed_local_artifacts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    _bind_local_runtime_artifact_integrity(fixture, shader_type="pbr")

    arguments = fixture.arguments(tmp_path / "package")
    arguments["shader_type"] = "pbr"
    manifest_path = compile_research_candidate_animal_package(**arguments)

    assert (
        _load_json(manifest_path.parent / "habitat/animal.ao_config.json")[
            "shader_type"
        ]
        == "pbr"
    )
    qualification = _load_json(manifest_path)["qualification"]
    assert "not trusted runtime attestations" in qualification["decision_reason"]
    assert "External capture/audit" in qualification["decision_reason"]


def test_compiler_rejects_pbr_with_legacy_phong_runtime_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    arguments = fixture.arguments(tmp_path / "package")
    arguments["shader_type"] = "pbr"

    with pytest.raises(PackageCompileError, match="evidence_scope"):
        compile_research_candidate_animal_package(**arguments)


def test_compiler_rejects_offline_relabel_of_legacy_phong_reports(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    _offline_relabel_legacy_reports_as_pbr(fixture)
    arguments = fixture.arguments(tmp_path / "package")
    arguments["shader_type"] = "pbr"

    with pytest.raises(PackageCompileError, match="evidence_scope"):
        compile_research_candidate_animal_package(**arguments)


@pytest.mark.parametrize(
    ("report_field", "section", "field", "value", "message"),
    [
        (
            "habitat_static_probe",
            "producer_source_integrity",
            "sha256",
            "00" * 32,
            "current producer source",
        ),
        (
            "habitat_animation_review",
            "ao_config_artifact",
            "sha256",
            "00" * 32,
            "real artifact",
        ),
    ],
)
def test_compiler_rejects_tampered_pbr_runtime_binding(
    tmp_path: Path,
    report_field: str,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    _bind_local_runtime_artifact_integrity(fixture, shader_type="pbr")
    report_path = getattr(fixture, report_field)
    report = _load_json(report_path)
    target = (
        report["producer_source_integrity"]
        if section == "producer_source_integrity"
        else report["render_configuration_integrity"]["ao_config_artifact"]
    )
    target[field] = value
    _write_json(report_path, report)
    arguments = fixture.arguments(tmp_path / "package")
    arguments["shader_type"] = "pbr"

    with pytest.raises(PackageCompileError, match=message):
        compile_research_candidate_animal_package(**arguments)


@pytest.mark.parametrize(
    "artifact_kind", ["ao_config", "runtime_binding", "observation"]
)
def test_compiler_reads_real_pbr_artifacts_instead_of_trusting_report_json(
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    _bind_local_runtime_artifact_integrity(fixture, shader_type="pbr")
    report_path = fixture.habitat_animation_review
    report = _load_json(report_path)
    if artifact_kind == "ao_config":
        record = report["render_configuration_integrity"]["ao_config_artifact"]
    elif artifact_kind == "runtime_binding":
        record = report["runtime_artifact_integrity"]["runtime_binding_artifact"]
    else:
        record = report["runtime_artifact_integrity"]["observation_artifacts"][0]
    artifact_path = report_path.parent / record["path"]
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")
    arguments = fixture.arguments(tmp_path / "package")
    arguments["shader_type"] = "pbr"

    with pytest.raises(PackageCompileError, match="real artifact"):
        compile_research_candidate_animal_package(**arguments)


def test_compiler_accepts_untriggered_hind_metric_without_legacy_claim(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    animation = _load_json(fixture.animation_qa)
    summary = animation["semantic_terminal_motion"]["walking_summary"]
    summary.update(
        {
            "legacy_hind_gait_metric_triggered": False,
            "mean_front_paw_forward_range_m": 0.2,
            "mean_hind_paw_forward_range_m": 0.1,
            "mean_hind_paw_lateral_range_m": 0.05,
        }
    )
    animation["known_limitations"] = []
    _write_json(fixture.animation_qa, animation)

    manifest_path = compile_research_candidate_animal_package(
        **fixture.arguments(tmp_path / "package")
    )

    emitted = _load_json(manifest_path.parent / "qa/animation.json")
    emitted_summary = emitted["semantic_terminal_motion"]["walking_summary"]
    assert emitted_summary["legacy_hind_gait_metric_triggered"] is False
    assert emitted["known_limitations"] == []


def test_compiler_rejects_legacy_claim_when_hind_metric_is_not_triggered(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    animation = _load_json(fixture.animation_qa)
    summary = animation["semantic_terminal_motion"]["walking_summary"]
    summary.update(
        {
            "legacy_hind_gait_metric_triggered": False,
            "mean_front_paw_forward_range_m": 0.2,
            "mean_hind_paw_forward_range_m": 0.1,
            "mean_hind_paw_lateral_range_m": 0.05,
        }
    )
    _write_json(fixture.animation_qa, animation)

    with pytest.raises(
        PackageCompileError,
        match="must not claim a legacy hind-gait limitation",
    ):
        compile_research_candidate_animal_package(
            **fixture.arguments(tmp_path / "package")
        )


def test_compiler_is_byte_deterministic_across_output_directories(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    first = compile_research_candidate_animal_package(
        **fixture.arguments(tmp_path / "first")
    ).parent
    second = compile_research_candidate_animal_package(
        **fixture.arguments(tmp_path / "second")
    ).parent

    first_files = {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


@pytest.mark.parametrize(
    ("report_name", "mutation", "message"),
    [
        (
            "habitat_static_probe",
            lambda value: value["input"].__setitem__("sha256", "00" * 32),
            "static_qa.input.sha256",
        ),
        (
            "action_report",
            lambda value: value["artifact"].__setitem__("readback_equal", False),
            "canonical/read-back-equal",
        ),
        (
            "rebase_deformation",
            lambda value: value.__setitem__("maximum_vertex_error_m", 1.0),
            "maximum vertex error",
        ),
        (
            "rebase_deformation",
            lambda value: value["rebased"].__setitem__("byte_size", 1),
            "deformation_report.rebased.byte_size",
        ),
        (
            "animation_qa",
            lambda value: value.__setitem__("qualification_claim", True),
            "must not claim asset qualification",
        ),
        (
            "static_qa",
            lambda value: value.__setitem__("topology_sha256", "00" * 32),
            "independent GLB evidence",
        ),
        (
            "deformation_qa",
            lambda value: value.__setitem__("baked_actions_sha256", "00" * 32),
            "canonical actions",
        ),
        (
            "deformation_qa",
            lambda value: value["actions"][0].__setitem__(
                "source_loop_endpoint_vertex_error_m", 1.0
            ),
            "source_loop_endpoint_vertex_error_m.*exceeds",
        ),
        (
            "animation_qa",
            lambda value: value["mouth"].__setitem__(
                "maximum_rotation_excursion_degrees", 1.0
            ),
            "mouth=0",
        ),
        (
            "deformation_qa",
            lambda value: value["actions"][0].__setitem__(
                "source_loop_endpoint_maximum_joint_rotation_error", -1.0
            ),
            "must be non-negative",
        ),
        (
            "animation_qa",
            lambda value: value.__setitem__("human_visual_review_required", False),
            "require human review",
        ),
    ],
)
def test_compiler_rejects_false_pass_and_unbound_evidence(
    tmp_path: Path,
    report_name: str,
    mutation,
    message: str,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    report_path = getattr(fixture, report_name)
    report = _load_json(report_path)
    mutation(report)
    _write_json(report_path, report)

    with pytest.raises(PackageCompileError, match=message):
        compile_research_candidate_animal_package(
            **fixture.arguments(tmp_path / "package")
        )


def test_compiler_requires_absent_and_rejects_repeated_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "inputs")
    output = tmp_path / "package"
    output.mkdir()
    (output / "unexpected.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(PackageCompileError, match="must not already exist"):
        compile_research_candidate_animal_package(**fixture.arguments(output))

    (output / "unexpected.txt").unlink()
    output.rmdir()
    compile_research_candidate_animal_package(**fixture.arguments(output))
    with pytest.raises(PackageCompileError, match="must not already exist"):
        compile_research_candidate_animal_package(**fixture.arguments(output))


def test_compiler_publishes_complete_directory_in_one_atomic_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    output = tmp_path / "package"
    observed: dict[str, object] = {}
    original = package_module._publish_directory_no_replace

    def observe(staging: Path, destination: Path) -> None:
        observed["destination_absent"] = not os.path.lexists(destination)
        observed["staging_files"] = package_module._output_tree_files(staging)
        original(staging, destination)

    monkeypatch.setattr(package_module, "_publish_directory_no_replace", observe)
    manifest = compile_research_candidate_animal_package(**fixture.arguments(output))

    assert observed["destination_absent"] is True
    assert observed["staging_files"] == {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert manifest.is_file()


def test_atomic_package_publication_never_replaces_racing_empty_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    output = tmp_path / "package"
    original = package_module._publish_directory_no_replace
    raced_inode: int | None = None

    def race(staging: Path, destination: Path) -> None:
        nonlocal raced_inode
        destination.mkdir()
        raced_inode = destination.stat().st_ino
        original(staging, destination)

    monkeypatch.setattr(package_module, "_publish_directory_no_replace", race)
    with pytest.raises(PackageCompileError, match="refusing to replace"):
        compile_research_candidate_animal_package(**fixture.arguments(output))

    assert raced_inode is not None
    assert output.is_dir()
    assert output.stat().st_ino == raced_inode
    assert list(output.iterdir()) == []
    assert not any(tmp_path.glob(".package.staging-*"))


def test_atomic_package_publication_fails_closed_when_renameat2_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    output = tmp_path / "package"
    monkeypatch.setattr(
        package_module.ctypes, "CDLL", lambda *_args, **_kwargs: object()
    )

    with pytest.raises(PackageCompileError, match="no-replace.*unavailable"):
        compile_research_candidate_animal_package(**fixture.arguments(output))

    assert not output.exists()
    assert not any(tmp_path.glob(".package.staging-*"))


def test_compiler_rejects_symlinked_input_and_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "inputs")
    visual_link = tmp_path / "visual-link.glb"
    visual_link.symlink_to(fixture.visual)
    arguments = fixture.arguments(tmp_path / "package")
    arguments["visual_glb"] = visual_link
    with pytest.raises(PackageCompileError, match="symbolic link"):
        compile_research_candidate_animal_package(**arguments)

    real_output = tmp_path / "real-output"
    real_output.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(real_output, target_is_directory=True)
    with pytest.raises(PackageCompileError, match="symbolic link"):
        compile_research_candidate_animal_package(**fixture.arguments(output_link))


def test_payload_publication_rolls_back_after_later_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "package"
    original = package_module._write_file_exclusive
    calls = 0

    def fail_second(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        original(path, payload)

    monkeypatch.setattr(package_module, "_write_file_exclusive", fail_second)
    with pytest.raises(PackageCompileError, match="injected write failure"):
        package_module._write_payloads(
            output,
            {"first.bin": b"first", "nested/second.bin": b"second"},
            validate=lambda path: None,
        )
    assert not output.exists()


def test_payload_publication_rolls_back_after_final_validation_failure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "package"
    output.mkdir()

    def fail_validation(path: Path) -> None:
        raise PackageCompileError("injected final validation failure")

    with pytest.raises(PackageCompileError, match="final validation failure"):
        package_module._write_payloads(
            output,
            {"asset_manifest.json": b"{}\n", "qa/check.json": b"{}\n"},
            validate=fail_validation,
        )
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_payload_publication_refuses_racing_leaf_and_removes_earlier_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "package"
    output.mkdir()
    original = package_module._write_file_exclusive
    calls = 0

    def inject_racing_leaf(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            path.write_bytes(b"foreign")
        original(path, payload)

    monkeypatch.setattr(
        package_module,
        "_write_file_exclusive",
        inject_racing_leaf,
    )

    with pytest.raises(PackageCompileError, match="refusing to replace"):
        package_module._write_payloads(
            output,
            {"first.bin": b"first", "second.bin": b"second"},
            validate=lambda path: None,
        )
    assert not (output / "first.bin").exists()
    assert (output / "second.bin").read_bytes() == b"foreign"


def test_emitted_package_detects_payload_tampering(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "inputs")
    manifest_path = compile_research_candidate_animal_package(
        **fixture.arguments(tmp_path / "package")
    )
    manifest = _load_json(manifest_path)
    (manifest_path.parent / "qa/static_geometry.json").write_bytes(b"{}\n")

    errors = validate_animal_asset_package(manifest, manifest_path=manifest_path)
    assert any(
        "byte_size does not match qa/static_geometry.json" in error for error in errors
    )
    assert any(
        "sha256 does not match qa/static_geometry.json" in error for error in errors
    )


def test_source_and_license_policy_are_explicitly_bound(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "inputs")
    source = _load_json(fixture.source)
    source["formal_dataset_registration_authorized"] = True
    _write_json(fixture.source, source)
    with pytest.raises(PackageCompileError, match="must be exactly false"):
        compile_research_candidate_animal_package(
            **fixture.arguments(tmp_path / "source-package")
        )

    source["formal_dataset_registration_authorized"] = False
    _write_json(fixture.source, source)
    license_snapshot = _load_json(fixture.license_snapshot)
    license_snapshot["license"] = "different"
    _write_json(fixture.license_snapshot, license_snapshot)
    with pytest.raises(PackageCompileError, match="identity.license"):
        compile_research_candidate_animal_package(
            **fixture.arguments(tmp_path / "license-package")
        )


@pytest.mark.parametrize("invalid_value", [1, "false"])
def test_source_registration_authorization_requires_boolean_false(
    tmp_path: Path, invalid_value: object
) -> None:
    fixture = _fixture(tmp_path / "inputs")
    source = _load_json(fixture.source)
    source["formal_dataset_registration_authorized"] = invalid_value
    _write_json(fixture.source, source)

    with pytest.raises(PackageCompileError, match="must be exactly false"):
        compile_research_candidate_animal_package(
            **fixture.arguments(tmp_path / "package")
        )


def test_source_registration_authorization_is_required(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "inputs")
    source = _load_json(fixture.source)
    source.pop("formal_dataset_registration_authorized")
    _write_json(fixture.source, source)

    with pytest.raises(PackageCompileError, match="must be exactly false"):
        compile_research_candidate_animal_package(
            **fixture.arguments(tmp_path / "package")
        )
