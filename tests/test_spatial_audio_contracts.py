from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from avengine.contracts.json_io import canonical_json_sha256
from avengine.spatial_audio.contracts import (
    AUDIO_BUNDLE_SCHEMA,
    CURRENT_INSTALLED_EVIDENCE_SCHEMA,
    EVIDENCE_SCHEMA,
    FOA_CONTRACT,
    IDENTITY_SCHEMA,
    REQUEST_SCHEMA,
    M4ContractError,
    canonical_source_ids,
    json_schema_errors,
    load_and_validate_multi_source_canary_request,
    validate_audio_bundle,
    validate_current_installed_multi_source_canary_evidence,
    validate_multi_source_canary_evidence,
    validate_multi_source_canary_request,
    validate_source_identity_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples/spatial_audio/blender_custom"
REQUEST_PATH = EXAMPLE_ROOT / "multi_source_canary_request.json"
IDENTITY_PATH = EXAMPLE_ROOT / "source_identity_manifest.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rehash(value: dict[str, Any], field: str) -> None:
    value[field] = canonical_json_sha256(
        {key: item for key, item in value.items() if key != field}
    )


def _record(path: Path, *, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "byte_size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_checked_in_request_and_identity_pass_schema_and_semantics() -> None:
    request = _load(REQUEST_PATH)
    identity = _load(IDENTITY_PATH)

    assert json_schema_errors(request, REQUEST_SCHEMA) == []
    assert json_schema_errors(identity, IDENTITY_SCHEMA) == []
    assert validate_multi_source_canary_request(
        request, request_path=REQUEST_PATH
    ) == []
    assert validate_source_identity_manifest(
        identity, manifest_path=IDENTITY_PATH
    ) == []

    validated = load_and_validate_multi_source_canary_request(REQUEST_PATH)
    assert validated.canonical_source_ids == ("source0", "source1")
    assert validated.listener["listener_id"] == "listener0"
    assert [source["source_id"] for source in validated.sources] == [
        "source0",
        "source1",
    ]


def test_checked_in_example_is_explicitly_not_m2_qualified() -> None:
    identity = _load(IDENTITY_PATH)

    assert identity["qualification_claim"] is False
    assert {
        source["m2_anchor_evidence"]["status"] for source in identity["sources"]
    } == {"not_run"}
    assert all(
        source["m2_anchor_evidence"]["qualification_claim"] is False
        for source in identity["sources"]
    )
    assert all(
        source["position_authority"] == "m1_capture_request_source_pose"
        for source in identity["sources"]
    )


def test_request_requires_exactly_one_listener_and_at_least_two_sources() -> None:
    request = _load(REQUEST_PATH)
    request["listeners"].append(copy.deepcopy(request["listeners"][0]))
    request["sources"] = request["sources"][:1]
    _rehash(request, "request_content_sha256")

    schema_errors = json_schema_errors(request, REQUEST_SCHEMA)
    semantic_errors = validate_multi_source_canary_request(
        request, request_path=REQUEST_PATH
    )

    assert any("listeners" in error for error in schema_errors)
    assert any("sources" in error for error in schema_errors)
    assert any("exactly one listener" in error for error in semantic_errors)
    assert any("at least two" in error for error in semantic_errors)


def test_identity_ids_and_dry_signals_are_one_to_one() -> None:
    identity = _load(IDENTITY_PATH)
    second = identity["sources"][1]
    first = identity["sources"][0]
    for field in ("actor_id", "event_id", "anchor_id", "dry_audio_id"):
        second[field] = first[field]
    second["deterministic_signal"] = copy.deepcopy(first["deterministic_signal"])
    _rehash(identity, "manifest_content_sha256")

    errors = validate_source_identity_manifest(
        identity, manifest_path=IDENTITY_PATH
    )

    for field in ("actor_id", "event_id", "anchor_id", "dry_audio_id"):
        assert any(field in error and "one-to-one" in error for error in errors)
    assert any("deterministic_signal" in error for error in errors)


def test_source_ids_use_portable_canonical_byte_order() -> None:
    assert canonical_source_ids(["source10", "source2", "Source0"]) == (
        "Source0",
        "source10",
        "source2",
    )
    try:
        canonical_source_ids(["source0", "声源1"])
    except M4ContractError as exc:
        assert any("portable ASCII" in error for error in exc.errors)
    else:  # pragma: no cover - contract must fail closed
        raise AssertionError("non-ASCII source ID was accepted")

    request = _load(REQUEST_PATH)
    request["sources"].reverse()
    _rehash(request, "request_content_sha256")
    errors = validate_multi_source_canary_request(
        request, request_path=REQUEST_PATH
    )
    assert any("sources must use canonical" in error for error in errors)


def test_stable_id_length_limit_is_uniform_across_m4_contracts(
    tmp_path: Path,
) -> None:
    accepted = "s" * 128
    rejected = "s" * 129
    assert canonical_source_ids([accepted]) == (accepted,)
    with pytest.raises(M4ContractError):
        canonical_source_ids([rejected])

    request = _load(REQUEST_PATH)
    request["request_id"] = rejected
    _rehash(request, "request_content_sha256")
    assert any(
        "request_id" in error
        for error in json_schema_errors(request, REQUEST_SCHEMA)
    )

    identity = _load(IDENTITY_PATH)
    identity["manifest_id"] = rejected
    _rehash(identity, "manifest_content_sha256")
    assert any(
        "manifest_id" in error
        for error in json_schema_errors(identity, IDENTITY_SCHEMA)
    )

    _, bundle = _audio_bundle(tmp_path)
    bundle["bundle_id"] = rejected
    _rehash(bundle, "bundle_content_sha256")
    assert any(
        "bundle_id" in error
        for error in json_schema_errors(bundle, AUDIO_BUNDLE_SCHEMA)
    )

    _, evidence = _evidence(tmp_path)
    evidence["runtime"]["foa_endpoint_receipts"]["sources"][0][
        "source_id"
    ] = rejected
    _rehash(evidence, "evidence_content_sha256")
    assert any(
        "source_id" in error
        for error in json_schema_errors(evidence, EVIDENCE_SCHEMA)
    )


def test_registration_orders_must_include_exact_canonical_and_reverse() -> None:
    request = _load(REQUEST_PATH)
    request["registration_orders"][1]["source_ids"] = ["source0", "source1"]
    _rehash(request, "request_content_sha256")

    errors = validate_multi_source_canary_request(
        request, request_path=REQUEST_PATH
    )

    assert any("distinct permutations" in error for error in errors)
    assert any("reversed source order" in error for error in errors)


def test_foa_contract_is_exact_raw_rlr_acn_n3d_world() -> None:
    request = _load(REQUEST_PATH)

    assert request["spatial_audio"] == FOA_CONTRACT
    assert request["simulation"]["channel_layout"] == {
        "type": "ambisonics",
        "channel_count": 4,
    }

    request["spatial_audio"]["raw_channel_order"] = ["W", "X", "Y", "Z"]
    request["spatial_audio"]["normalization"] = "SN3D"
    request["spatial_audio"]["axes"]["forward"] = "+Z"
    _rehash(request, "request_content_sha256")
    schema_errors = json_schema_errors(request, REQUEST_SCHEMA)
    semantic_errors = validate_multi_source_canary_request(
        request, request_path=REQUEST_PATH
    )

    assert schema_errors
    assert any("ACN/N3D world FOA" in error for error in semantic_errors)


def test_listener_position_and_orientation_are_bound_to_m1_camera_rig() -> None:
    request = _load(REQUEST_PATH)
    request["listeners"][0]["position_m"][0] += 0.01
    request["listeners"][0]["orientation_wxyz"] = [1.0, 0.0, 0.0, 0.0]
    _rehash(request, "request_content_sha256")

    errors = validate_multi_source_canary_request(
        request, request_path=REQUEST_PATH
    )

    assert any("listener position differs" in error for error in errors)
    assert any("listener orientation differs" in error for error in errors)


def test_listener_orientation_must_be_unit_even_if_rotation_is_equivalent() -> None:
    request = _load(REQUEST_PATH)
    request["listeners"][0]["orientation_wxyz"] = [
        component * 2.0
        for component in request["listeners"][0]["orientation_wxyz"]
    ]
    _rehash(request, "request_content_sha256")

    errors = validate_multi_source_canary_request(
        request, request_path=REQUEST_PATH
    )

    assert any("unit normalized" in error for error in errors)


def test_source_positions_bind_both_m1_and_identity_manifest() -> None:
    request = _load(REQUEST_PATH)
    request["sources"][0]["position_m"][2] += 0.25
    _rehash(request, "request_content_sha256")

    errors = validate_multi_source_canary_request(
        request, request_path=REQUEST_PATH
    )

    assert any("differs from M1 source pose" in error for error in errors)
    assert any("differs from source identity manifest" in error for error in errors)


def test_identity_position_tamper_is_rejected_after_rehash() -> None:
    identity = _load(IDENTITY_PATH)
    identity["sources"][1]["position_m"][0] += 0.5
    _rehash(identity, "manifest_content_sha256")

    errors = validate_source_identity_manifest(
        identity, manifest_path=IDENTITY_PATH
    )

    assert any("differs from its M1 source pose" in error for error in errors)


def test_input_file_record_hash_and_path_confinement_are_enforced() -> None:
    request = _load(REQUEST_PATH)
    request["inputs"]["m1_capture_request"]["sha256"] = "0" * 64
    request["inputs"]["m3_acoustic_canary_request"]["path"] = "../escape.json"
    _rehash(request, "request_content_sha256")

    schema_errors = json_schema_errors(request, REQUEST_SCHEMA)
    semantic_errors = validate_multi_source_canary_request(
        request, request_path=REQUEST_PATH
    )

    assert any("path" in error for error in schema_errors)
    assert any("sha256 does not match" in error for error in semantic_errors)
    assert any("confined POSIX relative path" in error for error in semantic_errors)


def test_document_content_hash_detects_tamper() -> None:
    request = _load(REQUEST_PATH)
    request["request_id"] = "tampered_request"

    errors = validate_multi_source_canary_request(
        request, request_path=REQUEST_PATH
    )

    assert any("request_content_sha256" in error for error in errors)


def _audio_bundle(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    identity_copy = tmp_path / "source_identity_manifest.json"
    identity_copy.write_bytes(IDENTITY_PATH.read_bytes())
    pairs: list[dict[str, Any]] = []
    for index, source_id in enumerate(("source0", "source1")):
        rir = tmp_path / f"{source_id}.rir.npy"
        stem = tmp_path / f"{source_id}.stem.wav"
        rir.write_bytes(f"rir-{source_id}".encode("ascii"))
        stem.write_bytes(f"stem-{source_id}".encode("ascii"))
        pairs.append(
            {
                "listener_id": "listener0",
                "source_id": source_id,
                "dry": {
                    "dry_audio_id": f"dry{index}",
                    "sample_sha256": hashlib.sha256(
                        f"dry-{source_id}".encode("ascii")
                    ).hexdigest(),
                    "sample_count": 4,
                    "channel_count": 1,
                    "dtype": "float64_le",
                },
                "rir": {
                    "status": "available",
                    "storage": "npy_channel_major",
                    "file": _record(rir, root=tmp_path),
                    "sample_sha256": hashlib.sha256(
                        f"samples-rir-{source_id}".encode("ascii")
                    ).hexdigest(),
                    "sample_count": 10,
                    "channel_count": 4,
                    "dtype": "float32_le",
                },
                "stem": {
                    "status": "available",
                    "storage": "wav_float32_interleaved",
                    "file": _record(stem, root=tmp_path),
                    "sample_sha256": hashlib.sha256(
                        f"samples-stem-{source_id}".encode("ascii")
                    ).hexdigest(),
                    "sample_count": 13,
                    "channel_count": 4,
                    "dtype": "float32_le",
                },
            }
        )
    bundle: dict[str, Any] = {
        "schema": AUDIO_BUNDLE_SCHEMA,
        "bundle_id": "m4_audio_bundle_test",
        "request_id": "m4_blender_custom_two_source_foa_v1",
        "overall_status": "blocked",
        "failure_reasons": [
            "No explicit licensed HRTF is installed in this unit test."
        ],
        "source_identity_manifest": _record(identity_copy, root=tmp_path),
        "listener_id": "listener0",
        "canonical_source_order": ["source0", "source1"],
        "sample_rate_hz": 16000,
        "spatial_audio": copy.deepcopy(FOA_CONTRACT),
        "pairs": pairs,
        "binaural_decoder": {
            "status": "not_run",
            "qualification_claim": False,
            "reason": "No explicit licensed HRTF is installed in this unit test.",
        },
        "bundle_content_sha256": "0" * 64,
    }
    _rehash(bundle, "bundle_content_sha256")
    bundle_path = tmp_path / "audio_bundle.json"
    return bundle_path, bundle


def test_audio_bundle_schema_and_python_validate_pair_closure(tmp_path: Path) -> None:
    bundle_path, bundle = _audio_bundle(tmp_path)

    assert json_schema_errors(bundle, AUDIO_BUNDLE_SCHEMA) == []
    assert validate_audio_bundle(bundle, bundle_path=bundle_path) == []

    bundle["pairs"][1]["dry"]["dry_audio_id"] = "dry0"
    bundle["pairs"][1]["stem"]["sample_count"] = 12
    _rehash(bundle, "bundle_content_sha256")
    errors = validate_audio_bundle(bundle, bundle_path=bundle_path)
    assert any("dry_audio_id" in error for error in errors)
    assert any("full convolution length" in error for error in errors)


def test_audio_bundle_artifact_tamper_is_rejected(tmp_path: Path) -> None:
    bundle_path, bundle = _audio_bundle(tmp_path)
    artifact = tmp_path / bundle["pairs"][0]["rir"]["file"]["path"]
    artifact.write_bytes(b"tampered-rir")

    errors = validate_audio_bundle(bundle, bundle_path=bundle_path)

    assert any("sha256 does not match" in error for error in errors)


def _evidence(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    request_snapshot = tmp_path / "request.json"
    pair_snapshot = tmp_path / "source0.npy"
    request_snapshot.write_bytes(REQUEST_PATH.read_bytes())
    pair_snapshot.write_bytes(b"array-placeholder")
    role = "source0_ir"
    source_ids = ["source0", "source1"]
    identity_sources = {
        source_id: {
            "actor_id": f"actor{index}",
            "event_id": f"event{index}",
            "anchor_id": f"actor{index}.muzzle",
            "semantic_anchor_id": "muzzle",
            "m1_source_id": source_id,
            "position_m": [-1.2, 0.3, -1.0]
            if index == 0
            else [2.0, 0.3, 0.0],
            "position_authority": "m1_capture_request_source_pose",
            "dry_audio_id": f"dry{index}",
            "m2_anchor_evidence": {
                "status": "not_run",
                "qualification_claim": False,
                "reason": "Unit fixture has no tracked qualified M2 anchor evidence.",
            },
        }
        for index, source_id in enumerate(source_ids)
    }

    def binaural_report() -> dict[str, Any]:
        return {
            "status": "pass",
            "rendering_method": "rlr_native_binaural_v1",
            "channel_order": ["left", "right"],
            "sample_count": 16,
            "minimum_ild_db": 6.0,
            "measurements": {
                "+X": {
                    "left_energy": 1.0,
                    "right_energy": 4.0,
                    "left_minus_right_ild_db": -6.020599913279624,
                },
                "-X": {
                    "left_energy": 4.0,
                    "right_energy": 1.0,
                    "left_minus_right_ild_db": 6.020599913279624,
                },
            },
        }

    directions = (
        ("+X", "right", 3, 1.7320508075688772),
        ("-X", "left", 3, -1.7320508075688772),
        ("+Y", "up", 1, 1.7320508075688772),
        ("-Y", "down", 1, -1.7320508075688772),
        ("+Z", "back", 2, 1.7320508075688772),
        ("-Z", "front", 2, -1.7320508075688772),
    )
    foa_report = {
        "status": "pass",
        "spatial_format": copy.deepcopy(FOA_CONTRACT),
        "expected_directional_to_w_magnitude": 1.7320508075688772,
        "direct_arrival_sample": 12,
        "measurements": [
            {
                "direction": direction,
                "semantic_direction": semantic,
                "direct_arrival_sample": 12,
                "w_amplitude": 0.5,
                "directional_channel_index": channel,
                "directional_to_w_ratio": ratio,
                "maximum_off_axis_to_w_ratio": 0.0,
            }
            for direction, semantic, channel, ratio in directions
        ],
    }

    upload_report = {
        "object_count": 1,
        "vertex_count": 8,
        "triangle_count": 12,
        "material_category_count": 1,
        "object_ids": ["room"],
        "triangle_count_by_material": {"wall": 12},
        "material_upload_call_count": {"wall": 1},
        "resolved_material_name_by_category": {"wall": "brick"},
        "resolved_material_index_by_category": {"wall": 0},
        "material_upload_receipts": [
            {
                "object_id": "room",
                "material_category": "wall",
                "triangle_count": 12,
                "index_count": 36,
                "canonical_payload_byte_count": 144,
                "canonical_payload_sha1": "1" * 40,
            }
        ],
        "expected_material_block_count": 1,
        "material_database_sha1": "2" * 40,
        "expected_world_geometry_sha1": "3" * 40,
        "expected_canonical_byte_count": 144,
        "expected_material_coefficient_sha1": "4" * 40,
        "expected_material_coefficient_byte_count": 48,
    }

    configuration = {
        "frequency_bands": 4,
        "direct_sh_order": 1,
        "indirect_sh_order": 1,
        "direct_ray_count": 500,
        "indirect_ray_count": 5000,
        "indirect_ray_depth": 100,
        "source_ray_count": 500,
        "source_ray_depth": 20,
        "max_diffraction_order": 10,
        "thread_count": 1,
        "sample_rate_hz": 16000.0,
        "max_ir_seconds": 4.0,
        "unit_scale": 1.0,
        "global_volume": 1.0,
        "direct": True,
        "indirect": True,
        "diffraction": True,
        "transmission": False,
        "mesh_simplification": False,
        "temporal_coherence": False,
    }

    def endpoint_receipts(layout: str) -> dict[str, Any]:
        binaural = layout == "binaural"
        return {
            "authority": "native_registration_readback",
            "sources": [
                {
                    "source_id": source_id,
                    "canonical_native_index": index,
                    "position_m": identity_sources[source_id]["position_m"],
                    "radius_m": 0.0,
                    "native_realized": True,
                }
                for index, source_id in enumerate(source_ids)
            ],
            "listener": {
                "listener_id": "listener0",
                "canonical_native_index": 0,
                "position_m": [-2.5, 1.55, 0.0],
                "orientation_wxyz": [0.7071067811865476, 0.0, -0.7071067811865475, 0.0],
                "radius_m": 0.0,
                "layout_type": layout,
                "channel_count": 2 if binaural else 4,
                "hrtf_mode": "external_file" if binaural else "rlr_builtin_default",
                "hrtf_file_path": "/staged/hrtf.sofa" if binaural else "",
                "native_realized": True,
            },
        }

    def performance_condition(source_count: int) -> dict[str, Any]:
        runs = [
            {
                "repeat_index": index,
                "wall_seconds": 0.1 * source_count,
                "process_cpu_seconds": 0.08 * source_count,
                "peak_rss_before_bytes": 1000,
                "peak_rss_after_bytes": 2000,
                "ir_payload_bytes": 256 * source_count,
                "pair_count": source_count,
            }
            for index in range(2)
        ]
        return {
            "source_count": source_count,
            "pair_count": source_count,
            "repeat_count": 2,
            "median_wall_seconds": 0.1 * source_count,
            "p95_wall_seconds": 0.1 * source_count,
            "median_process_cpu_seconds": 0.08 * source_count,
            "maximum_peak_rss_bytes": 2000,
            "median_ir_payload_bytes": 256 * source_count,
            "runs": runs,
        }

    rate_binding = {
        "policy": "strict_match",
        "render_sample_rate_hz": 16000,
        "avengine_resampling_performed": False,
        "hrtf_input_sample_rate_hz": 16000,
        "native_rate_adaptation": "not_required",
    }
    binaural_contract = {
        "schema": "avengine_rlr_native_binaural_preflight_v1",
        "rendering_method": "rlr_native_binaural_v1",
        "renderer": "RLR native binaural listener",
        "channel_layout": {
            "type": "binaural",
            "channel_count": 2,
            "channel_order": ["left", "right"],
        },
        "hrtf_policy": "explicit_hash_and_license_required",
        "avengine_resampling_policy": "forbidden",
        "native_rate_adaptation_policy": (
            "allowed only when explicitly bound to an RLR binary SHA-256"
        ),
        "normalization_policy": "forbidden",
        "limiter_policy": "forbidden",
        "render_sample_rate_hz": 16000,
        "verification_scope": "hrtf_dependency_preflight_only",
        "native_cardinal_validation": "pass",
        "sample_rate_binding": copy.deepcopy(rate_binding),
        "hrtf": {
            "asset_id": "fixture_hrtf",
            "byte_size": 16,
            "sha256": "5" * 64,
            "expected_sha256": "5" * 64,
            "sample_rate_hz": 16000,
            "artifact_role": "request_snapshot",
        },
        "rights": {
            "license_id": "fixture-license",
            "citation": "Fixture HRTF",
            "license_text_byte_size": 16,
            "license_text_sha256": "6" * 64,
            "expected_license_sha256": "6" * 64,
            "license_artifact_role": "request_snapshot",
        },
        "status": "pass",
        "native_cardinal_report": binaural_report(),
    }
    pairs = {}
    for index, source_id in enumerate(source_ids):
        distance = 1.0 + index
        expected = distance / 343.0 * 16000.0
        detected = round(expected)
        pairs[source_id] = {
            "listener_id": "listener0",
            "source_id": source_id,
            "actor_id": f"actor{index}",
            "event_id": f"event{index}",
            "anchor_id": f"actor{index}.muzzle",
            "semantic_anchor_id": "muzzle",
            "dry_audio_id": f"dry{index}",
            "dry_wav_role": role,
            "dry_sidecar_role": role,
            "foa_ir_order_a_role": role,
            "foa_ir_order_b_role": role,
            "binaural_ir_role": role,
            "direct_arrival": {
                "distance_m": distance,
                "speed_of_sound_m_s": 343.0,
                "expected_sample": expected,
                "detected_sample": detected,
                "absolute_error_samples": abs(detected - expected),
                "maximum_absolute_error_samples": 2.0,
            },
            "foa_stem_wav_role": role,
            "foa_stem_sidecar_role": role,
            "binaural_stem_wav_role": role,
            "binaural_stem_sidecar_role": role,
        }
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "request_id": "m4_blender_custom_two_source_foa_v1",
        "qualification_claim": True,
        "overall_status": "pass",
        "failure_reasons": [],
        "artifacts": {
            "request_snapshot": _record(request_snapshot, root=tmp_path),
            "source0_ir": _record(pair_snapshot, root=tmp_path),
        },
        "inputs": {
            "request_role": "request_snapshot",
            "m1_capture_request_role": "request_snapshot",
            "m3_acoustic_canary_request_role": "request_snapshot",
            "source_identity_manifest_role": "request_snapshot",
            "runtime_lock_role": "request_snapshot",
            "hrtf_role": "request_snapshot",
            "hrtf_license_role": "request_snapshot",
            "acoustic_scene_package_manifest_role": "request_snapshot",
            "acoustic_scene_package_file_roles": {
                "manifest.json": "request_snapshot"
            },
            "request_content_sha256": "7" * 64,
            "package_id": "fixture_package",
            "package_content_sha256": "8" * 64,
        },
        "identity": {
            "listener_id": "listener0",
            "listener_count": 1,
            "canonical_source_ids": source_ids,
            "source_count": 2,
            "source_identities": identity_sources,
        },
        "execution": {
            "one_context_per_output_layout": True,
            "foa_context_listener_count": 1,
            "binaural_context_listener_count": 1,
            "requested_registration_orders": [
                {"order_id": "canonical", "source_ids": source_ids},
                {"order_id": "reversed", "source_ids": list(reversed(source_ids))},
            ],
            "canonical_native_source_order": source_ids,
            "static_authority_policy": "fresh_context_temporal_false",
            "independent_episode_policy": "reset_reload_before_first_frame",
        },
        "audio_contracts": {
            "foa": copy.deepcopy(FOA_CONTRACT),
            "binaural": binaural_contract,
            "stem_equation": (
                "wet[source,channel]=dry[source]*rir[listener,source,channel]"
            ),
            "mixture_equation": "canonical_float64_sum(per_source_stems)",
            "implicit_normalization": False,
            "avengine_resampling_performed": False,
            "native_rate_adaptation": copy.deepcopy(rate_binding),
            "limiter": False,
            "m4_tail_policy": "full_linear_convolution",
            "m5_video_mux": "not_owned_by_m4",
        },
        "pairs": pairs,
        "mixtures": {
            "foa_wav_role": role,
            "foa_sidecar_role": role,
            "binaural_wav_role": role,
            "binaural_sidecar_role": role,
            "source_ids": source_ids,
            "summation_order": source_ids,
        },
        "probes": {
            "foa": {
                "cardinal_roles": {
                    direction: role for direction, *_ in directions
                },
                "cardinal_report": foa_report,
                "world_identity_role": role,
                "world_rotated_role": role,
                "world_alignment_report": {
                    "status": "pass",
                    "format_id": "rlr_foa_acn_n3d_world_v1",
                    "sample_count": 16,
                    "maximum_absolute_difference": 0.0,
                    "rtol": 0.0,
                    "atol": 0.0,
                },
            },
            "binaural": {
                "cardinal_roles": {"+X": role, "-X": role},
                "minimum_ild_db": 6.0,
                "cardinal_report": binaural_report(),
            },
        },
        "lifecycle": {
            "moved_source_id": "source0",
            "moved_distance_m": 0.25,
            "original_position_m": [-1.2, 0.3, -1.0],
            "updated_position_m": [-1.3, 0.3, -1.1],
            "source_registration_receipts_after_update": [
                {
                    "source_id": source_id,
                    "canonical_native_index": index,
                    "position_m": identity_sources[source_id]["position_m"],
                    "radius_m": 0.0,
                    "native_realized": True,
                }
                for index, source_id in enumerate(source_ids)
            ],
            "reset_matches_fresh_first": True,
            "source_update_preserves_identity": True,
            "temporal_sequence_executed": True,
            "reset_boundary_policy": "reset_reload_before_independent_episode",
            "counts_after_reset": {
                "object_count": 0,
                "source_count": 0,
                "listener_count": 0,
            },
            "upload_report": upload_report,
            "fresh_first_roles": {source_id: role for source_id in source_ids},
            "updated_roles": {source_id: role for source_id in source_ids},
            "reset_first_roles": {source_id: role for source_id in source_ids},
        },
        "performance": {
            "one_source": performance_condition(1),
            "multi_source": performance_condition(2),
            "comparison": {
                "multi_to_one_median_wall_ratio": 2.0,
                "multi_pair_throughput_pairs_per_second": 10.0,
                "hard_speed_gate": None,
                "interpretation": "measurement_only_platform_dependent",
            },
        },
        "runtime": {
            "binding_api": "habitat_sim.RLRAcousticContext_v1",
            "native_binaries": {
                "habitat_sim_bindings": {
                    "path": "/runtime/habitat_sim_bindings.so",
                    "byte_size": 100,
                    "sha256": "9" * 64,
                },
                "rlr_audio_propagation": {
                    "path": "/runtime/libRLRAudioPropagation.so",
                    "byte_size": 100,
                    "sha256": "a" * 64,
                },
            },
            "foa_configuration_readback": configuration,
            "binaural_configuration_readback": copy.deepcopy(configuration),
            "foa_upload_report": copy.deepcopy(upload_report),
            "foa_endpoint_receipts": endpoint_receipts("ambisonics"),
            "binaural_endpoint_receipts": endpoint_receipts("binaural"),
        },
        "checks": [
            {
                "check_id": "contract",
                "required": True,
                "status": "pass",
                "measured": {"valid": True},
                "threshold": {"valid": True},
            }
        ],
        "evidence_content_sha256": "0" * 64,
    }
    _rehash(evidence, "evidence_content_sha256")
    return tmp_path / "evidence.json", evidence


def test_evidence_schema_role_and_file_closure(tmp_path: Path) -> None:
    evidence_path, evidence = _evidence(tmp_path)

    assert json_schema_errors(evidence, EVIDENCE_SCHEMA) == []
    assert validate_multi_source_canary_evidence(
        evidence, evidence_path=evidence_path
    ) == []

    evidence["inputs"]["request_role"] = "missing_role"
    _rehash(evidence, "evidence_content_sha256")
    errors = validate_multi_source_canary_evidence(
        evidence, evidence_path=evidence_path
    )
    assert any("does not resolve" in error for error in errors)


def test_evidence_artifact_tamper_and_status_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    evidence_path, evidence = _evidence(tmp_path)
    (tmp_path / evidence["artifacts"]["source0_ir"]["path"]).write_bytes(
        b"tampered"
    )
    evidence["overall_status"] = "blocked"
    evidence["qualification_claim"] = False
    evidence["failure_reasons"] = []
    _rehash(evidence, "evidence_content_sha256")

    errors = validate_multi_source_canary_evidence(
        evidence, evidence_path=evidence_path
    )

    assert any("sha256 does not match" in error for error in errors)
    assert any("failure reason" in error for error in errors)


def test_evidence_nested_sections_fail_closed_after_rehash(tmp_path: Path) -> None:
    evidence_path, evidence = _evidence(tmp_path)
    evidence["runtime"]["silent_fallback"] = True
    del evidence["pairs"]["source0"]["foa_ir_order_a_role"]
    _rehash(evidence, "evidence_content_sha256")

    schema_errors = json_schema_errors(evidence, EVIDENCE_SCHEMA)
    assert any("silent_fallback" in error for error in schema_errors)
    assert any("foa_ir_order_a_role" in error for error in schema_errors)

    evidence_path, evidence = _evidence(tmp_path)
    evidence["pairs"]["source0"]["actor_id"] = "actor1"
    evidence["mixtures"]["summation_order"].reverse()
    _rehash(evidence, "evidence_content_sha256")
    semantic_errors = validate_multi_source_canary_evidence(
        evidence, evidence_path=evidence_path
    )
    assert any("pairs.source0.actor_id differs" in error for error in semantic_errors)
    assert any("summation_order must be canonical" in error for error in semantic_errors)


def test_endpoint_receipts_require_native_readback_and_realization(
    tmp_path: Path,
) -> None:
    evidence_path, evidence = _evidence(tmp_path)
    evidence["runtime"]["foa_endpoint_receipts"]["authority"] = "requested_state"
    evidence["runtime"]["foa_endpoint_receipts"]["sources"][0][
        "native_realized"
    ] = False
    evidence["runtime"]["foa_endpoint_receipts"]["listener"][
        "native_realized"
    ] = False
    _rehash(evidence, "evidence_content_sha256")

    errors = validate_multi_source_canary_evidence(
        evidence, evidence_path=evidence_path
    )
    assert any("authority is not native readback" in error for error in errors)
    assert any("sources are not native-realized" in error for error in errors)
    assert any("listener is not native-realized" in error for error in errors)


def test_current_installed_v2_evidence_schema_has_no_historical_runtime_lock(
    tmp_path: Path,
) -> None:
    evidence_path, evidence = _evidence(tmp_path)
    identity = {
        "identity_schema": "avengine_current_installed_rlr_runtime_v1",
        "mode": "current-installed",
        "habitat_runtime_prefix": "/current/habitat",
        "habitat_sim_module": "/current/habitat/python/habitat_sim/__init__.py",
        "habitat_sim_binding": "/current/habitat/python/habitat_sim/_ext.so",
        "magnum_python_site": "/current/magnum/python",
        "rlr_sdk_root": "/current/sdk",
        "rlr_sdk_header": "/current/sdk/include/RLRAcousticContext.h",
        "rlr_sdk_library": "/current/sdk/lib/libRLR.so",
        "rlr_adapter_enabled": True,
        "binding_api": "habitat_sim.RLRAcousticContext_v1",
    }
    evidence["schema"] = CURRENT_INSTALLED_EVIDENCE_SCHEMA
    evidence["qualification_claim"] = False
    evidence["inputs"].pop("runtime_lock_role")
    evidence["execution"]["runtime_mode"] = "current-installed"
    for condition in ("one_source", "multi_source"):
        for run in evidence["performance"][condition]["runs"]:
            run["runtime_identity"] = copy.deepcopy(identity)
    records = [copy.deepcopy(identity) for _ in range(11)]
    evidence["runtime"].pop("native_binaries")
    evidence["runtime"].update(
        {
            "runtime_mode": "current-installed",
            "current_installed_identity": copy.deepcopy(identity),
            "current_installed_identity_records": records,
        }
    )
    evidence["checks"].append(
        {
            "check_id": "runtime_current_installed_identity",
            "required": True,
            "status": "pass",
            "measured": {
                "record_count": len(records),
                "unique_identity_count": 1,
                "identities": copy.deepcopy(records),
            },
            "threshold": {
                "same_runtime_identity_every_native_call": True,
                "runtime_mode": "current-installed",
            },
        }
    )
    _rehash(evidence, "evidence_content_sha256")

    assert json_schema_errors(evidence, CURRENT_INSTALLED_EVIDENCE_SCHEMA) == []
    assert validate_current_installed_multi_source_canary_evidence(
        evidence, evidence_path=evidence_path
    ) == []
