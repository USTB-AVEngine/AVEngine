from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

import avengine.m3.canary as canary_module
import avengine.cli as cli_module
from avengine.cli import main as cli_main
from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.m3.canary import (
    _array_sha256,
    _expected_configuration_readback,
    load_and_verify_canary_evidence,
    run_material_activation_canary,
    verify_canary_evidence,
)
from avengine.m3.compiler import compile_canary_request
from avengine.m3.runtime import (
    RUNTIME_IMPORT_WORKAROUND,
    CompiledAcousticScene,
    RLRSimulationConfig,
    RuntimeAnchor,
    RuntimeIRResult,
    _cpu_first_hit_distance,
    _expected_native_scene_readback_report,
    _expected_scene_readback,
    _expected_upload_report,
    _parse_scene_obj,
)

REPOSITORY = Path(__file__).resolve().parents[2]
REQUEST = REPOSITORY / "examples/m3/blender_custom/canary_request.json"
_FAKE_BINDING_BYTES = b"binding-fixture-v1"
_FAKE_RLR_BYTES = b"rlr-fixture-v1"
_FAKE_BINDING_SHA256 = hashlib.sha256(_FAKE_BINDING_BYTES).hexdigest()
_FAKE_RLR_SHA256 = hashlib.sha256(_FAKE_RLR_BYTES).hexdigest()

_RUN_TOOL_SPEC = importlib.util.spec_from_file_location(
    "avengine_test_run_material_canary",
    REPOSITORY / "tools/m3/run_material_canary.py",
)
assert _RUN_TOOL_SPEC is not None and _RUN_TOOL_SPEC.loader is not None
run_tool = importlib.util.module_from_spec(_RUN_TOOL_SPEC)
_RUN_TOOL_SPEC.loader.exec_module(run_tool)


@pytest.fixture(autouse=True)
def _matching_fake_runtime_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    runtime_lock = tmp_path / "fake_runtime.lock.yaml"
    runtime_lock.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "status: test_runtime_lock",
                "",
                "runtime_test_environment:",
                f"  required_m3_native_binding_sha256: {_FAKE_BINDING_SHA256}",
                f"  required_m3_rlr_library_sha256: {_FAKE_RLR_SHA256}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(canary_module, "_runtime_lock_path", lambda: runtime_lock)
    return runtime_lock


def _write_expected_rlr_obj(path: Path, scene: CompiledAcousticScene) -> None:
    lines = ["# Objects"]
    vertex_offset = 0
    for item in scene.objects:
        material_ids = np.asarray(item["triangle_material_ids"], dtype=np.int64)
        used_material_ids = [
            material_id
            for material_id in range(len(scene.material_categories))
            if np.any(material_ids == material_id)
        ]
        for local_index, material_id in enumerate(used_material_ids):
            category = scene.material_categories[material_id]
            material = next(
                value
                for value in scene.rlr_material_database["materials"]
                if category.casefold()
                in {str(label).casefold() for label in value["labels"]}
            )
            lines.append(f"# Material Index : {local_index}")
            for coefficient_name in (
                "absorption",
                "scattering",
                "transmission",
            ):
                for coefficient_index, value in enumerate(
                    material[coefficient_name][1::2]
                ):
                    lines.append(
                        f"# {coefficient_name.capitalize()} - "
                        f"Index:{coefficient_index}, Value: {float(value):.6f}"
                    )
        lines.extend(
            [
                f"# Vertex Count: {len(item['vertices'])}",
                f"# Triangle Count: {len(item['triangles'])}",
                f"# Material Count: {len(item['triangles'])}",
            ]
        )
        for vertex in item["vertices"]:
            lines.append("v " + " ".join(f"{float(value):.6f}" for value in vertex))
        for triangle in item["triangles"]:
            indices = [int(value) + vertex_offset + 1 for value in triangle]
            lines.append("f " + " ".join(str(value) for value in indices))
        vertex_offset += len(item["vertices"])
    lines.extend(["# Listeners", "# Sources"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _DeterministicRunner:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        binding = root / "fake_habitat_sim_bindings.so"
        rlr = root / "fake_libRLRAudioPropagation.so"
        binding.write_bytes(_FAKE_BINDING_BYTES)
        rlr.write_bytes(_FAKE_RLR_BYTES)
        self.native_binaries = {
            "habitat_sim_bindings": {
                "path": str(binding.resolve()),
                "byte_size": binding.stat().st_size,
                "sha256": sha256_file(binding),
            },
            "rlr_audio_propagation": {
                "path": str(rlr.resolve()),
                "byte_size": rlr.stat().st_size,
                "sha256": sha256_file(rlr),
            },
        }

    def __call__(
        self,
        scene: CompiledAcousticScene,
        simulation: RLRSimulationConfig,
        *,
        source: RuntimeAnchor,
        listener: RuntimeAnchor,
        scene_readback_obj: Path,
        ray_checks: tuple[Mapping[str, Any], ...],
        ray_distance_tolerance_m: float,
    ) -> RuntimeIRResult:
        _write_expected_rlr_obj(scene_readback_obj, scene)
        parsed = _parse_scene_obj(scene_readback_obj)
        assert parsed == _expected_scene_readback(scene)
        upload = _expected_upload_report(scene)
        native_report = _expected_native_scene_readback_report(parsed, upload)
        runtime = {
            "import_workaround": dict(RUNTIME_IMPORT_WORKAROUND),
            "quaternion_module": {"path": "/fixture/quaternion", "version": "test"},
            "habitat_sim_module": {"path": "/fixture/habitat_sim", "version": "test"},
            "binding_api": "habitat_sim.RLRAcousticContext_v1",
            "native_binaries": copy.deepcopy(self.native_binaries),
            "configuration_readback": _expected_configuration_readback(simulation),
            "scene_mesh_readback": {
                "path": str(scene_readback_obj.resolve()),
                "byte_size": scene_readback_obj.stat().st_size,
                "sha256": sha256_file(scene_readback_obj),
                "native_report": native_report,
                **parsed,
            },
        }

        reports: list[dict[str, Any]] = []
        for declaration in ray_checks:
            origin = np.asarray(declaration["origin_m"], dtype=np.float64)
            direction = np.asarray(declaration["direction"], dtype=np.float64)
            direction /= np.linalg.norm(direction)
            maximum_distance = float(declaration["distance_m"])
            cpu_distance = _cpu_first_hit_distance(
                scene.objects,
                origin=origin,
                direction=direction,
                minimum_distance_m=0.0,
                maximum_distance_m=maximum_distance,
            )
            hit = cpu_distance is not None
            expected_hit = declaration["expectation"] == "hit_within_m"
            passed = hit == expected_hit
            reports.append(
                {
                    "check_id": declaration["check_id"],
                    "expectation": declaration["expectation"],
                    "maximum_distance_m": maximum_distance,
                    "cpu_first_hit_distance_m": cpu_distance,
                    "rlr_any_hit": {
                        "hit": hit,
                        "has_hit_details": False,
                        "distance_m": 0.0,
                        "normal": [0.0, 0.0, 0.0],
                    },
                    "rlr_first_hit": {
                        "hit": hit,
                        "has_hit_details": hit,
                        "distance_m": float(cpu_distance) if hit else 0.0,
                        # RLR first-hit normals are explicitly un-normalized.
                        "normal": [-7.2, 0.0, 0.0] if hit else [0.0, 0.0, 0.0],
                    },
                    "cpu_rlr_hit_consistent": True,
                    "cpu_rlr_distance_consistent": True,
                    "distance_tolerance_m": ray_distance_tolerance_m,
                    "passed": passed,
                }
            )

        sample_count = int(simulation.max_ir_seconds * simulation.sample_rate_hz)
        distance = math.dist(source.position_m, listener.position_m)
        arrival = int(
            round(distance / simulation.speed_of_sound_m_s * simulation.sample_rate_hz)
        )
        samples = np.zeros((1, sample_count), dtype="<f4")
        samples[0, arrival] = 1.0
        tau_seconds = 0.20 if "low_absorption" in scene.package_id else 0.04
        time = np.arange(sample_count - arrival - 1, dtype=np.float64)
        time /= simulation.sample_rate_hz
        samples[0, arrival + 1 :] = 0.1 * np.exp(-time / tau_seconds)
        return RuntimeIRResult(
            listener_id=listener.anchor_id,
            source_id=source.anchor_id,
            sample_rate_hz=simulation.sample_rate_hz,
            samples=samples,
            package_manifest_sha256=scene.manifest_sha256,
            package_content_sha256=scene.package_content_sha256,
            runtime=runtime,
            upload_report=upload,
            indirect_ray_efficiency=0.75,
            ray_checks=tuple(reports),
        )


def _rehash_evidence(path: Path, evidence: dict[str, Any]) -> None:
    evidence["evidence_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in evidence.items()
            if key != "evidence_content_sha256"
        }
    )
    write_json(path, evidence)


def _clone_evidence(source: Path, root: Path, name: str) -> Path:
    destination = root / name
    shutil.copytree(source.parent, destination)
    return destination / source.name


def test_canary_pass_evidence_and_rehashed_tamper_matrix(tmp_path: Path) -> None:
    compile_evidence = compile_canary_request(REQUEST, tmp_path / "compile")
    evidence_path = run_material_activation_canary(
        REQUEST,
        compile_evidence,
        tmp_path / "runtime",
        runner=_DeterministicRunner(tmp_path / "native"),
    )
    evidence = load_json(evidence_path)
    assert evidence["overall_status"] == "pass"
    assert verify_canary_evidence(evidence_path) == []
    assert cli_main(["m3", "verify-canary", str(evidence_path)]) == 0
    first_hit = evidence["conditions"]["low_absorption"]["runs"][0][
        "ray_checks"
    ][1]["rlr_first_hit"]
    assert first_hit["hit"] is True
    assert np.linalg.norm(first_hit["normal"]) == pytest.approx(7.2)

    upload_path = _clone_evidence(evidence_path, tmp_path, "tamper_upload")
    upload = load_json(upload_path)
    upload["conditions"]["low_absorption"]["runs"][0]["upload_report"][
        "object_count"
    ] += 1
    _rehash_evidence(upload_path, upload)
    assert any(
        "upload_report" in error for error in verify_canary_evidence(upload_path)
    )

    ray_path = _clone_evidence(evidence_path, tmp_path, "tamper_ray")
    ray = load_json(ray_path)
    ray["conditions"]["low_absorption"]["runs"][0]["ray_checks"][0][
        "cpu_first_hit_distance_m"
    ] = 0.123
    _rehash_evidence(ray_path, ray)
    assert any("ray_checks" in error for error in verify_canary_evidence(ray_path))

    zero_normal_path = _clone_evidence(evidence_path, tmp_path, "zero_hit_normal")
    zero_normal = load_json(zero_normal_path)
    zero_normal["conditions"]["low_absorption"]["runs"][0]["ray_checks"][1][
        "rlr_first_hit"
    ]["normal"] = [0.0, 0.0, 0.0]
    _rehash_evidence(zero_normal_path, zero_normal)
    assert any(
        "normal must be finite and non-zero" in error
        for error in verify_canary_evidence(zero_normal_path)
    )

    any_sentinel_path = _clone_evidence(evidence_path, tmp_path, "any_hit_sentinel")
    any_sentinel = load_json(any_sentinel_path)
    any_sentinel["conditions"]["low_absorption"]["runs"][0]["ray_checks"][1][
        "rlr_any_hit"
    ]["normal"] = [0.0, 2.0, 0.0]
    _rehash_evidence(any_sentinel_path, any_sentinel)
    assert any(
        "rlr_any_hit must use zero distance/normal sentinels" in error
        for error in verify_canary_evidence(any_sentinel_path)
    )

    miss_sentinel_path = _clone_evidence(
        evidence_path, tmp_path, "first_hit_miss_sentinel"
    )
    miss_sentinel = load_json(miss_sentinel_path)
    miss_sentinel["conditions"]["low_absorption"]["runs"][0]["ray_checks"][0][
        "rlr_first_hit"
    ]["distance_m"] = 0.25
    _rehash_evidence(miss_sentinel_path, miss_sentinel)
    assert any(
        "first_hit miss must use zero distance/normal sentinels" in error
        for error in verify_canary_evidence(miss_sentinel_path)
    )

    identity_path = _clone_evidence(evidence_path, tmp_path, "tamper_identity")
    identity = load_json(identity_path)
    identity["conditions"]["high_absorption"]["runs"][0][
        "runtime_result_identity"
    ]["package_content_sha256"] = "0" * 64
    _rehash_evidence(identity_path, identity)
    assert any(
        "runtime_result_identity" in error
        for error in verify_canary_evidence(identity_path)
    )

    obj_path = _clone_evidence(evidence_path, tmp_path, "tamper_coefficient")
    obj_evidence = load_json(obj_path)
    run = obj_evidence["conditions"]["low_absorption"]["runs"][0]
    record = run["runtime"]["scene_mesh_readback"]["artifact"]
    artifact = obj_path.parent / record["path"]
    artifact.write_text(
        artifact.read_text(encoding="utf-8").replace(
            "Value: 0.020000", "Value: 0.020001", 1
        ),
        encoding="utf-8",
    )
    updated_record = file_record(artifact, relative_to=obj_path.parent)
    readback = run["runtime"]["scene_mesh_readback"]
    readback["artifact"] = updated_record
    readback["byte_size"] = updated_record["byte_size"]
    readback["sha256"] = updated_record["sha256"]
    readback.update(_parse_scene_obj(artifact))
    _rehash_evidence(obj_path, obj_evidence)
    assert any(
        "fingerprint differs from package" in error
        for error in verify_canary_evidence(obj_path)
    )

    winding_path = _clone_evidence(evidence_path, tmp_path, "tamper_winding")
    winding = load_json(winding_path)
    run = winding["conditions"]["low_absorption"]["runs"][0]
    record = run["runtime"]["scene_mesh_readback"]["artifact"]
    artifact = winding_path.parent / record["path"]
    lines = artifact.read_text(encoding="utf-8").splitlines()
    face_index = next(index for index, line in enumerate(lines) if line.startswith("f "))
    fields = lines[face_index].split()
    lines[face_index] = " ".join([fields[0], fields[1], fields[3], fields[2]])
    artifact.write_text("\n".join(lines) + "\n", encoding="utf-8")
    updated_record = file_record(artifact, relative_to=winding_path.parent)
    readback = run["runtime"]["scene_mesh_readback"]
    readback["artifact"] = updated_record
    readback["byte_size"] = updated_record["byte_size"]
    readback["sha256"] = updated_record["sha256"]
    readback.update(_parse_scene_obj(artifact))
    _rehash_evidence(winding_path, winding)
    assert any(
        "fingerprint differs from package" in error
        for error in verify_canary_evidence(winding_path)
    )

    schema_path = _clone_evidence(evidence_path, tmp_path, "tamper_pass_schema")
    schema_evidence = load_json(schema_path)
    schema_evidence["conditions"]["low_absorption"]["runs"][0][
        "direct_arrival"
    ]["passed"] = False
    _rehash_evidence(schema_path, schema_evidence)
    assert any(
        "True was expected" in error
        for error in verify_canary_evidence(schema_path)
    )

    check_path = _clone_evidence(evidence_path, tmp_path, "tamper_check_set")
    check_evidence = load_json(check_path)
    check_evidence["checks"].append(
        {
            "check_id": "unrecognized_admission_shortcut",
            "required": True,
            "status": "pass",
            "measured": True,
            "threshold": True,
        }
    )
    _rehash_evidence(check_path, check_evidence)
    assert any(
        "check_id set differs" in error
        for error in verify_canary_evidence(check_path)
    )

    request_record_path = _clone_evidence(
        evidence_path, tmp_path, "tamper_request_record"
    )
    request_record = load_json(request_record_path)
    request_record["request"]["source"]["byte_size"] += 1
    _rehash_evidence(request_record_path, request_record)
    assert any(
        "request source record differs" in error
        for error in verify_canary_evidence(request_record_path)
    )


def test_ir_metric_tamper_is_recomputed_from_raw_artifact(tmp_path: Path) -> None:
    compile_evidence = compile_canary_request(REQUEST, tmp_path / "compile")
    evidence_path = run_material_activation_canary(
        REQUEST,
        compile_evidence,
        tmp_path / "runtime",
        runner=_DeterministicRunner(tmp_path / "native"),
    )
    evidence = load_json(evidence_path)
    run = evidence["conditions"]["low_absorption"]["runs"][0]
    record = run["ir_artifact"]
    artifact = evidence_path.parent / record["path"]
    samples = np.load(artifact, allow_pickle=False)
    samples[0, -1] += np.float32(0.01)
    np.save(artifact, samples, allow_pickle=False)
    run["ir_artifact"] = file_record(artifact, relative_to=evidence_path.parent)
    run["ir_array"]["raw_array_sha256"] = _array_sha256(samples)
    _rehash_evidence(evidence_path, evidence)
    errors = verify_canary_evidence(evidence_path)
    assert any("metrics differ from raw IR recomputation" in error for error in errors)


def test_ir_record_and_numpy_parse_share_one_snapshot_under_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_evidence = compile_canary_request(REQUEST, tmp_path / "compile")
    evidence_path = run_material_activation_canary(
        REQUEST,
        compile_evidence,
        tmp_path / "runtime",
        runner=_DeterministicRunner(tmp_path / "native"),
    )
    evidence = load_json(evidence_path)
    record = evidence["conditions"]["low_absorption"]["runs"][0]["ir_artifact"]
    target = (evidence_path.parent / record["path"]).resolve()
    original_read = canary_module._read_file_once
    swapped = False

    def read_then_swap(path: Path) -> bytes:
        nonlocal swapped
        payload = original_read(path)
        if Path(path).resolve() == target and not swapped:
            target.write_bytes(b"post-snapshot-invalid-npy")
            swapped = True
        return payload

    monkeypatch.setattr(canary_module, "_read_file_once", read_then_swap)
    assert verify_canary_evidence(evidence_path) == []
    assert swapped is True


def test_cli_verify_uses_verified_evidence_snapshot_for_declared_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_evidence = compile_canary_request(REQUEST, tmp_path / "compile")
    evidence_path = run_material_activation_canary(
        REQUEST,
        compile_evidence,
        tmp_path / "runtime",
        runner=_DeterministicRunner(tmp_path / "native"),
    )
    original_load = load_and_verify_canary_evidence

    def load_then_replace(path: str | Path):
        result = original_load(path)
        Path(path).write_text('{"overall_status":"fail"}\n', encoding="utf-8")
        return result

    monkeypatch.setattr(
        cli_module, "load_and_verify_canary_evidence", load_then_replace
    )
    assert cli_main(["m3", "verify-canary", str(evidence_path)]) == 0


def _binary_identity_check(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return next(
        check
        for check in evidence["checks"]
        if check["check_id"] == "runtime_native_binary_identity"
    )


def test_native_binary_pins_reject_rehashed_evidence_and_lock(
    tmp_path: Path,
    _matching_fake_runtime_lock: Path,
) -> None:
    compile_evidence = compile_canary_request(REQUEST, tmp_path / "compile")
    evidence_path = run_material_activation_canary(
        REQUEST,
        compile_evidence,
        tmp_path / "runtime_binary_tamper",
        runner=_DeterministicRunner(tmp_path / "native_binary_tamper"),
    )
    evidence = load_json(evidence_path)
    runs = [
        run
        for condition in ("low_absorption", "high_absorption")
        for run in evidence["conditions"][condition]["runs"]
    ]
    binding_record = runs[0]["runtime"]["native_binaries"][
        "habitat_sim_bindings"
    ]
    binding_path = Path(binding_record["path"])
    binding_path.write_bytes(b"different-but-consistent-binding")
    changed_binding_record = {
        "path": str(binding_path.resolve()),
        "byte_size": binding_path.stat().st_size,
        "sha256": sha256_file(binding_path),
    }
    for run in runs:
        run["runtime"]["native_binaries"]["habitat_sim_bindings"] = copy.deepcopy(
            changed_binding_record
        )
    _binary_identity_check(evidence)["measured"]["native_binaries"][
        "habitat_sim_bindings"
    ] = copy.deepcopy(changed_binding_record)
    _rehash_evidence(evidence_path, evidence)
    assert any(
        "runtime_native_binary_identity check differs" in error
        for error in verify_canary_evidence(evidence_path)
    )

    lock_evidence_path = run_material_activation_canary(
        REQUEST,
        compile_evidence,
        tmp_path / "runtime_lock_tamper",
        runner=_DeterministicRunner(tmp_path / "native_lock_tamper"),
    )
    lock_evidence = load_json(lock_evidence_path)
    lock_text = _matching_fake_runtime_lock.read_text(encoding="utf-8")
    _matching_fake_runtime_lock.write_text(
        lock_text.replace(_FAKE_BINDING_SHA256, "0" * 64),
        encoding="utf-8",
    )
    changed_lock_record = {
        "path": str(_matching_fake_runtime_lock.resolve()),
        "byte_size": _matching_fake_runtime_lock.stat().st_size,
        "sha256": sha256_file(_matching_fake_runtime_lock),
    }
    lock_evidence["inputs"]["runtime_lock"] = changed_lock_record
    _binary_identity_check(lock_evidence)["threshold"][
        "runtime_lock_sha256"
    ] = changed_lock_record["sha256"]
    _rehash_evidence(lock_evidence_path, lock_evidence)
    assert any(
        "runtime_native_binary_identity check differs" in error
        for error in verify_canary_evidence(lock_evidence_path)
    )


def test_invalid_runtime_lock_pin_fails_binary_identity_check(
    tmp_path: Path,
    _matching_fake_runtime_lock: Path,
) -> None:
    lock_text = _matching_fake_runtime_lock.read_text(encoding="utf-8")
    _matching_fake_runtime_lock.write_text(
        lock_text.replace(_FAKE_RLR_SHA256, "not-a-sha256"),
        encoding="utf-8",
    )
    compile_evidence = compile_canary_request(REQUEST, tmp_path / "compile")
    evidence_path = run_material_activation_canary(
        REQUEST,
        compile_evidence,
        tmp_path / "runtime",
        runner=_DeterministicRunner(tmp_path / "native"),
    )
    evidence = load_json(evidence_path)
    binary_check = _binary_identity_check(evidence)

    assert evidence["overall_status"] == "fail"
    assert binary_check["status"] == "fail"
    assert binary_check["measured"]["native_binaries"] is None
    assert verify_canary_evidence(evidence_path) == []


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [("pass", 0), ("fail", 1), ("blocked", 1)],
)
def test_run_tool_exit_code_follows_evidence_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expected_exit: int,
) -> None:
    evidence_path = tmp_path / f"{status}.json"
    write_json(evidence_path, {"overall_status": status})
    monkeypatch.setattr(
        run_tool,
        "run_material_activation_canary",
        lambda *_args, **_kwargs: evidence_path,
    )
    monkeypatch.setattr(
        run_tool,
        "load_and_verify_canary_evidence",
        lambda _path: SimpleNamespace(
            evidence={"overall_status": status},
            errors=(),
        ),
    )
    assert (
        run_tool.main(
            [
                "--request",
                str(tmp_path / "request.json"),
                "--compile-evidence",
                str(tmp_path / "compile.json"),
                "--output",
                str(tmp_path / "output"),
            ]
        )
        == expected_exit
    )
