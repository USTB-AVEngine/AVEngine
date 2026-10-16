from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256
from avengine.spatial_audio import evidence as evidence_module
from avengine.spatial_audio.audio import (
    generate_sine_wave,
    read_float32_wav,
    write_float32_wav,
)
from avengine.spatial_audio.binaural import (
    build_rlr_native_binaural_metadata,
    rlr_native_binaural_contract,
)
from avengine.spatial_audio.evidence import (
    _binaural_lock_errors,
    _direct_arrival_errors,
    _dry_recipe_errors,
    _input_authority,
    _portable_binaural_metadata,
    _wav_metadata_errors,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _locked_binaural_fixture(tmp_path):
    hrtf_payload = b"unit-test SOFA authority bytes"
    license_payload = b"unit-test HRTF license authority bytes\n"
    hrtf_path = tmp_path / "hrtf.sofa"
    license_path = tmp_path / "hrtf_license.txt"
    lock_path = tmp_path / "runtime_lock.json"
    hrtf_path.write_bytes(hrtf_payload)
    license_path.write_bytes(license_payload)
    rlr_sha256 = "ab" * 32
    native_binaries = {
        "habitat_sim_bindings": {"byte_size": 123, "sha256": "cd" * 32},
        "rlr_audio_propagation": {"byte_size": 456, "sha256": rlr_sha256},
    }
    lock = {
        "schema": "avengine_m4_runtime_lock_v1",
        "native_binaries": native_binaries,
        "hrtf": {
            "asset_id": "unit_hrtf_v1",
            "sha256": _sha256(hrtf_payload),
            "sample_rate_hz": 44_100,
            "license_id": "unit-test-license",
            "citation": "Unit Test HRTF Authors (2026)",
            "license_text_sha256": _sha256(license_payload),
            "sample_rate_policy": "rlr_native_internal_bound_to_binary",
        },
        "output_contracts": {
            "foa": "rlr_foa_acn_n3d_world_v1",
            "binaural": "rlr_binaural_lr_v1",
            "render_sample_rate_hz": 16_000,
            "avengine_resampling_performed": False,
            "normalization": "none",
            "limiter": "none",
        },
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    preflight = build_rlr_native_binaural_metadata(
        16_000,
        hrtf_path=hrtf_path,
        expected_hrtf_sha256=lock["hrtf"]["sha256"],
        hrtf_sample_rate_hz=lock["hrtf"]["sample_rate_hz"],
        license_id=lock["hrtf"]["license_id"],
        citation=lock["hrtf"]["citation"],
        license_text_path=license_path,
        expected_license_sha256=lock["hrtf"]["license_text_sha256"],
        asset_id=lock["hrtf"]["asset_id"],
        sample_rate_policy=lock["hrtf"]["sample_rate_policy"],
        rlr_binary_sha256=rlr_sha256,
    )
    assert preflight["status"] == "pass"
    authority = _portable_binaural_metadata(
        preflight,
        hrtf_role="input_hrtf",
        license_role="input_hrtf_license",
    )
    authority["native_cardinal_validation"] = "pass"
    authority["native_cardinal_report"] = {"status": "pass", "probe": "unit"}
    evidence = {
        "inputs": {
            "runtime_lock_role": "input_runtime_lock",
            "hrtf_role": "input_hrtf",
            "hrtf_license_role": "input_hrtf_license",
        },
        "runtime": {"native_binaries": native_binaries},
        "audio_contracts": {
            "binaural": authority,
            "implicit_normalization": False,
            "avengine_resampling_performed": False,
            "native_rate_adaptation": copy.deepcopy(authority["sample_rate_binding"]),
            "limiter": False,
        },
    }
    paths = {
        "input_runtime_lock": lock_path,
        "input_hrtf": hrtf_path,
        "input_hrtf_license": license_path,
    }
    return paths, evidence, hrtf_payload, license_payload


def test_hrtf_license_and_binary_bound_rate_are_rebuilt_from_bytes(tmp_path) -> None:
    paths, evidence, _, _ = _locked_binaural_fixture(tmp_path)

    assert _binaural_lock_errors(paths, evidence, sample_rate_hz=16_000) == []

    evidence["audio_contracts"]["binaural"]["sample_rate_binding"][
        "rlr_binary_sha256"
    ] = "ef" * 32
    errors = _binaural_lock_errors(paths, evidence, sample_rate_hz=16_000)
    assert any(
        "differs from HRTF/license/runtime-lock bytes" in error for error in errors
    )


def test_rehashed_hrtf_or_license_bytes_cannot_bypass_runtime_lock(tmp_path) -> None:
    paths, evidence, hrtf_payload, license_payload = _locked_binaural_fixture(tmp_path)

    paths["input_hrtf"].write_bytes(b"replacement SOFA bytes")
    errors = _binaural_lock_errors(paths, evidence, sample_rate_hz=16_000)
    assert any("HRTF preflight did not pass" in error for error in errors)

    paths["input_hrtf"].write_bytes(hrtf_payload)
    paths["input_hrtf_license"].write_bytes(b"replacement license bytes")
    errors = _binaural_lock_errors(paths, evidence, sample_rate_hz=16_000)
    assert any("HRTF preflight did not pass" in error for error in errors)

    paths["input_hrtf_license"].write_bytes(license_payload)
    evidence["audio_contracts"]["implicit_resampling"] = False
    errors = _binaural_lock_errors(paths, evidence, sample_rate_hz=16_000)
    assert "ambiguous implicit_resampling field is forbidden" in errors


def test_authenticated_sidecar_must_match_full_binaural_semantics(tmp_path) -> None:
    _, evidence, _, _ = _locked_binaural_fixture(tmp_path)
    authority = evidence["audio_contracts"]["binaural"]
    expected = {
        "spatial_format": rlr_native_binaural_contract(),
        "hrtf": authority,
        "audio_role": "per_source_wet_stem",
        "source_id": "source0",
        "listener_id": "listener0",
        "lineage": "dry_linear_convolution_pair_binaural_ir",
        "linear_gain": 1.0,
        "normalization": "none",
        "tail_policy": "full_linear_convolution",
    }
    correct = write_float32_wav(
        tmp_path / "correct.wav",
        np.ones((2, 8), dtype=np.float64),
        16_000,
        metadata=expected,
    )
    decoded = read_float32_wav(correct.audio_path)
    assert _wav_metadata_errors(decoded, expected, owner="binaural stem") == []

    stale = copy.deepcopy(expected)
    stale["hrtf"]["native_cardinal_validation"] = "not_run"
    stale["hrtf"].pop("native_cardinal_report")
    stale_file = write_float32_wav(
        tmp_path / "stale.wav",
        np.ones((2, 8), dtype=np.float64),
        16_000,
        metadata=stale,
    )
    stale_decoded = read_float32_wav(stale_file.audio_path)
    errors = _wav_metadata_errors(stale_decoded, expected, owner="binaural stem")
    assert errors == [
        "binaural stem: sidecar semantic metadata differs from evidence contract"
    ]


def _retained_authority_fixture(tmp_path, monkeypatch):
    repository_root = Path(__file__).resolve().parents[1]
    source_root = repository_root / "examples/spatial_audio/blender_custom"
    source_files = {
        "input_request": source_root / "multi_source_canary_request.json",
        "input_m1_capture_request": (
            repository_root / "examples/rooms/requests/blender_custom.json"
        ),
        "input_m3_acoustic_canary_request": (
            repository_root / "examples/acoustics/blender_custom/canary_request.json"
        ),
        "input_source_identity_manifest": source_root / "source_identity_manifest.json",
    }
    paths: dict[str, Path] = {}
    for role, source in source_files.items():
        destination = tmp_path / "inputs" / f"{role}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        paths[role] = destination
    package_manifest = tmp_path / "inputs/acoustic_scene_package/manifest.json"
    package_manifest.parent.mkdir(parents=True, exist_ok=True)
    package_manifest.write_text("{}\n", encoding="utf-8")
    paths["input_package_000"] = package_manifest

    request = json.loads(paths["input_request"].read_text(encoding="utf-8"))
    identity = json.loads(
        paths["input_source_identity_manifest"].read_text(encoding="utf-8")
    )
    identities = {item["source_id"]: item for item in identity["sources"]}
    canonical_ids = request["canonical_source_order"]
    retained_fields = (
        "actor_id",
        "event_id",
        "anchor_id",
        "semantic_anchor_id",
        "m1_source_id",
        "position_m",
        "position_authority",
        "dry_audio_id",
        "m2_anchor_evidence",
    )
    package_hash = "12" * 32
    scene = SimpleNamespace(
        package_id="unit_package",
        package_content_sha256=package_hash,
        manifest={"source_room": {"room_id": "blender_custom_two_zone_v1"}},
    )
    monkeypatch.setattr(
        evidence_module, "load_compiled_acoustic_scene", lambda path: scene
    )
    monkeypatch.setattr(
        evidence_module, "_verify_upload_report", lambda scene, report: None
    )
    evidence = {
        "request_id": request["request_id"],
        "inputs": {
            "request_role": "input_request",
            "m1_capture_request_role": "input_m1_capture_request",
            "m3_acoustic_canary_request_role": "input_m3_acoustic_canary_request",
            "source_identity_manifest_role": "input_source_identity_manifest",
            "acoustic_scene_package_manifest_role": "input_package_000",
            "acoustic_scene_package_file_roles": {"manifest.json": "input_package_000"},
            "request_content_sha256": request["request_content_sha256"],
            "package_id": scene.package_id,
            "package_content_sha256": package_hash,
        },
        "identity": {
            "listener_id": request["listeners"][0]["listener_id"],
            "listener_count": 1,
            "canonical_source_ids": canonical_ids,
            "source_count": len(canonical_ids),
            "source_identities": {
                source_id: {
                    field: copy.deepcopy(identities[source_id][field])
                    for field in retained_fields
                }
                for source_id in canonical_ids
            },
        },
        "pairs": {
            source_id: {
                "listener_id": request["listeners"][0]["listener_id"],
                "source_id": source_id,
                "actor_id": identities[source_id]["actor_id"],
                "event_id": identities[source_id]["event_id"],
                "anchor_id": identities[source_id]["anchor_id"],
                "semantic_anchor_id": identities[source_id]["semantic_anchor_id"],
                "dry_audio_id": identities[source_id]["dry_audio_id"],
            }
            for source_id in canonical_ids
        },
        "execution": {
            "requested_registration_orders": request["registration_orders"],
            "canonical_native_source_order": canonical_ids,
        },
        "runtime": {"foa_upload_report": {}},
    }
    return paths, evidence


def test_retained_inputs_bind_identity_position_and_package(
    tmp_path, monkeypatch
) -> None:
    paths, evidence = _retained_authority_fixture(tmp_path, monkeypatch)

    authority, errors = _input_authority(paths, evidence)
    assert errors == []
    assert authority["canonical_source_ids"] == ("source0", "source1")

    evidence["identity"]["source_identities"]["source0"]["actor_id"] = "forged"
    _, errors = _input_authority(paths, evidence)
    assert any("evidence identity differs" in error for error in errors)

    evidence["identity"]["source_identities"]["source0"]["actor_id"] = "actor0"
    request = json.loads(paths["input_request"].read_text(encoding="utf-8"))
    request["sources"][0]["position_m"][0] += 0.5
    request["request_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in request.items()
            if key != "request_content_sha256"
        }
    )
    paths["input_request"].write_text(json.dumps(request), encoding="utf-8")
    evidence["inputs"]["request_content_sha256"] = request["request_content_sha256"]
    _, errors = _input_authority(paths, evidence)
    assert any("request position differs from M1" in error for error in errors)

    evidence["inputs"]["package_id"] = "forged_package"
    _, errors = _input_authority(paths, evidence)
    assert any("package ID differs" in error for error in errors)


def test_dry_recipe_is_regenerated_from_retained_identity(tmp_path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    identity = json.loads(
        (
            repository_root / "examples/spatial_audio/blender_custom/source_identity_manifest.json"
        ).read_text(encoding="utf-8")
    )["sources"][0]
    signal = identity["deterministic_signal"]
    dry = generate_sine_wave(
        16_000,
        signal["duration_samples"],
        signal["frequency_hz"],
        amplitude=signal["amplitude"],
        phase_radians=signal["phase"],
    )
    metadata = {
        "audio_role": "deterministic_canary_dry",
        "source_id": identity["source_id"],
        "actor_id": identity["actor_id"],
        "event_id": identity["event_id"],
        "anchor_id": identity["anchor_id"],
        "dry_audio_id": identity["dry_audio_id"],
        "signal": signal,
        "processing": "none",
    }
    artifact = write_float32_wav(
        tmp_path / "dry.wav", dry[np.newaxis, :], 16_000, metadata=metadata
    )
    decoded = read_float32_wav(artifact.audio_path)
    assert (
        _dry_recipe_errors(decoded, source_id=identity["source_id"], identity=identity)
        == []
    )

    forged_identity = copy.deepcopy(identity)
    forged_identity["deterministic_signal"]["frequency_hz"] = 880
    errors = _dry_recipe_errors(
        decoded, source_id=identity["source_id"], identity=forged_identity
    )
    assert any("dry WAVE differs from retained recipe" in error for error in errors)


def test_direct_arrival_expected_sample_is_rebuilt_from_geometry(monkeypatch) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    request = json.loads(
        (
            repository_root
            / "examples/spatial_audio/blender_custom/multi_source_canary_request.json"
        ).read_text(encoding="utf-8")
    )
    source = request["sources"][0]
    listener = request["listeners"][0]
    simulation = request["simulation"]
    distance = math.dist(source["position_m"], listener["position_m"])
    expected = (
        distance / simulation["speed_of_sound_m_s"] * simulation["sample_rate_hz"]
    )
    detected = round(expected)
    monkeypatch.setattr(
        evidence_module,
        "analyze_ir",
        lambda value, sample_rate: SimpleNamespace(direct_arrival_sample=detected),
    )
    authority = {
        "request": request,
        "listener": listener,
        "sources": {source["source_id"]: source},
    }
    declaration = {
        "distance_m": distance,
        "speed_of_sound_m_s": simulation["speed_of_sound_m_s"],
        "expected_sample": expected,
        "detected_sample": detected,
        "absolute_error_samples": abs(detected - expected),
        "maximum_absolute_error_samples": request["thresholds"][
            "maximum_direct_arrival_error_samples"
        ],
    }
    pairs = {source["source_id"]: {"direct_arrival": declaration}}
    order_a = {source["source_id"]: np.ones((4, 16), dtype=np.float32)}

    assert (
        _direct_arrival_errors(
            order_a,
            pairs,
            [source["source_id"]],
            sample_rate_hz=simulation["sample_rate_hz"],
            authority=authority,
        )
        == []
    )

    pairs[source["source_id"]]["direct_arrival"]["expected_sample"] += 1.0
    errors = _direct_arrival_errors(
        order_a,
        pairs,
        [source["source_id"]],
        sample_rate_hz=simulation["sample_rate_hz"],
        authority=authority,
    )
    assert any("declared expected_sample differs" in error for error in errors)
