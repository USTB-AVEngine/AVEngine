from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable

import numpy as np
import pytest

from avengine.contracts.json_io import canonical_json_sha256
from avengine.acoustics.canary import _expected_configuration_readback
from avengine.acoustics.compiler import compile_canary_request
from avengine.acoustics.runtime import _expected_upload_report
from avengine.spatial_audio import canary
from avengine.spatial_audio.audio import read_float32_wav, write_float32_wav
from avengine.spatial_audio.evidence import artifact_record, verify_m4_canary_evidence
from avengine.spatial_audio.runtime import (
    BINAURAL_LAYOUT_ID,
    FOA_LAYOUT_ID,
    LIFECYCLE_MOVED_DISTANCE_M,
    MultiSourceRenderResult,
    NamedPairIR,
)

REPOSITORY = Path(__file__).resolve().parents[2]
M3_REQUEST = REPOSITORY / "examples/acoustics/blender_custom/canary_request.json"
M4_REQUEST = REPOSITORY / "examples/spatial_audio/blender_custom/multi_source_canary_request.json"


def _identity(
    root: Path,
    *,
    canonical_paths: bool = True,
) -> dict[str, object]:
    habitat, sdk, magnum = root / "habitat", root / "sdk", root / "magnum"
    module = habitat / "python/habitat_sim/__init__.py"
    binding = habitat / "python/habitat_sim/_ext.so"
    magnum_site = magnum / "python"
    header = sdk / "include/RLRAcousticContext.h"
    library = sdk / "lib/libRLR.so"
    for path in (module, binding, header, library):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    magnum_site.mkdir(parents=True, exist_ok=True)

    def identity_path(path: Path) -> str:
        return str(path.resolve() if canonical_paths else path.absolute())

    return {
        "identity_schema": "avengine_current_installed_rlr_runtime_v1",
        "mode": "current-installed",
        "habitat_runtime_prefix": identity_path(habitat),
        "habitat_sim_module": identity_path(module),
        "habitat_sim_binding": identity_path(binding),
        "magnum_python_site": identity_path(magnum_site),
        "rlr_sdk_root": identity_path(sdk),
        "rlr_sdk_header": identity_path(header),
        "rlr_sdk_library": identity_path(library),
        "rlr_adapter_enabled": True,
        "binding_api": "habitat_sim.RLRAcousticContext_v1",
    }


def _normal_foa(source: Any, listener: Any, rate: int) -> np.ndarray:
    result = np.zeros((4, 2048), dtype="<f4")
    arrival = round(math.dist(source.position_m, listener.position_m) / 343.0 * rate)
    result[0, arrival] = 1.0
    result[0, arrival + 1 :] = np.float32(0.08) * np.exp(
        -np.arange(2048 - arrival - 1, dtype=np.float32) / 450.0
    )
    result[3, arrival] = np.float32(0.2)
    return result


def _normal_binaural(source: Any, listener: Any, rate: int) -> np.ndarray:
    result = np.zeros((2, 2048), dtype="<f4")
    arrival = round(math.dist(source.position_m, listener.position_m) / 343.0 * rate)
    result[:, arrival] = (0.5, 0.75)
    result[:, arrival + 1 :] = np.float32(0.02) * np.exp(
        -np.arange(2048 - arrival - 1, dtype=np.float32) / 220.0
    )
    return result


def _foa_probe(source: Any) -> np.ndarray:
    result = np.zeros((4, 256), dtype="<f4")
    result[0, 20] = 1.0
    channel, sign = {
        "probe_px": (3, 1.0), "probe_nx": (3, -1.0),
        "probe_py": (1, 1.0), "probe_ny": (1, -1.0),
        "probe_pz": (2, 1.0), "probe_nz": (2, -1.0),
    }[source.anchor_id]
    result[channel, 20] = np.float32(sign * math.sqrt(3.0))
    return result


def _binaural_probe(source: Any) -> np.ndarray:
    result = np.zeros((2, 256), dtype="<f4")
    result[:, 20] = (0.1, 1.0) if source.anchor_id == "probe_px" else (1.0, 0.1)
    return result


def _receipts(sources: Any, listener: Any, layout: str, channels: int, hrtf: str) -> dict[str, object]:
    ids = tuple(sorted((source.anchor_id for source in sources), key=str.encode))
    by_id = {source.anchor_id: source for source in sources}
    return {
        "authority": "native_registration_readback",
        "sources": [
            {"source_id": source_id, "canonical_native_index": index,
             "position_m": list(by_id[source_id].position_m),
             "radius_m": by_id[source_id].radius_m, "native_realized": True}
            for index, source_id in enumerate(ids)
        ],
        "listener": {
            "listener_id": listener.anchor_id, "canonical_native_index": 0,
            "position_m": list(listener.position_m),
            "orientation_wxyz": list(listener.orientation_wxyz),
            "radius_m": listener.radius_m, "layout_type": layout,
            "channel_count": channels,
            "hrtf_mode": "external_file" if hrtf else "rlr_builtin_default",
            "hrtf_file_path": hrtf, "native_realized": True,
        },
    }


def _full_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    compiled_path = compile_canary_request(M3_REQUEST, tmp_path / "compile")
    compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
    package = compiled_path.parent / compiled["packages"]["low_absorption"]["path"]
    hrtf = tmp_path / "fixture.sofa"
    hrtf.write_bytes(b"fresh current-installed HRTF fixture")
    license_path = tmp_path / "fixture-license.txt"
    license_path.write_text("fixture current-installed HRTF terms", encoding="utf-8")
    identity = _identity(tmp_path / "current-runtime")

    def render(scene: Any, simulation: Any, *, sources: Any, listener: Any,
               layout_type: str | None = None, channel_count: int | None = None,
               hrtf_file_path: str = "", **_kwargs: Any) -> MultiSourceRenderResult:
        layout = layout_type or simulation.channel_layout.layout_type
        channels = channel_count or simulation.channel_layout.channel_count
        ids = tuple(sorted((source.anchor_id for source in sources), key=str.encode))
        by_id = {source.anchor_id: source for source in sources}
        pairs: list[NamedPairIR] = []
        for source_id in ids:
            source = by_id[source_id]
            if source_id.startswith("probe_"):
                samples = _binaural_probe(source) if layout == "binaural" else _foa_probe(source)
            else:
                samples = _normal_binaural(source, listener, int(simulation.sample_rate_hz)) if layout == "binaural" else _normal_foa(source, listener, int(simulation.sample_rate_hz))
            if layout == "binaural":
                layout_id, labels, normalization, frame = BINAURAL_LAYOUT_ID, ("left", "right"), "not_applicable", "listener_local"
            else:
                layout_id, labels, normalization, frame = FOA_LAYOUT_ID, ("W", "Y", "Z", "X"), "N3D", "avengine_world"
            pairs.append(NamedPairIR(listener.anchor_id, source_id, float(simulation.sample_rate_hz), samples, layout, layout_id, labels, normalization, frame))
        return MultiSourceRenderResult(
            tuple(pairs), tuple(source.anchor_id for source in sources), ids,
            listener.anchor_id, float(simulation.sample_rate_hz), layout, layout_id,
            {"binding_api": "habitat_sim.RLRAcousticContext_v1",
             "configuration_readback": _expected_configuration_readback(simulation),
             "runtime_mode": "current-installed",
             "runtime_identity": copy.deepcopy(identity)},
            copy.deepcopy(_expected_upload_report(scene)),
            _receipts(sources, listener, layout, channels, hrtf_file_path), 0.5,
            {"wall_seconds": 0.01, "process_cpu_seconds": 0.005,
             "peak_rss_before_bytes": 100, "peak_rss_after_bytes": 200,
             "ir_payload_bytes": sum(pair.samples.nbytes for pair in pairs),
             "pair_count": len(pairs)},
        )

    def lifecycle(scene: Any, simulation: Any, *, sources: Any, listener: Any,
                  **_kwargs: Any) -> dict[str, object]:
        ids = tuple(sorted((source.anchor_id for source in sources), key=str.encode))
        by_id = {source.anchor_id: source for source in sources}
        fresh = {source.anchor_id: _normal_foa(source, listener, int(simulation.sample_rate_hz)) for source in sources}
        updated = {source_id: value.copy() for source_id, value in fresh.items()}
        updated[ids[0]][0, -1] += np.float32(0.25)
        original = np.asarray(by_id[ids[0]].position_m, dtype=np.float64)
        listener_position = np.asarray(listener.position_m, dtype=np.float64)
        direction = original - listener_position
        changed = (
            original
            + direction / np.linalg.norm(direction) * float(LIFECYCLE_MOVED_DISTANCE_M)
        )
        updated_positions = {
            source_id: np.asarray(
                by_id[source_id].position_m, dtype=np.float32
            ).astype(np.float64).tolist()
            for source_id in ids
        }
        updated_positions[ids[0]] = np.asarray(
            changed, dtype=np.float32
        ).astype(np.float64).tolist()
        return {
            "fresh_first": fresh, "updated": updated,
            "reset_first": {source_id: value.copy() for source_id, value in fresh.items()},
            "moved_source_id": ids[0],
            "moved_distance_m": float(LIFECYCLE_MOVED_DISTANCE_M),
            "original_position_m": original.tolist(),
            "updated_position_m": changed.tolist(),
            "source_registration_receipts_after_update": [
                {"source_id": source_id, "canonical_native_index": index,
                 "position_m": updated_positions[source_id],
                 "radius_m": float(np.float32(by_id[source_id].radius_m)),
                 "native_realized": True}
                for index, source_id in enumerate(ids)
            ],
            "reset_matches_fresh_first": True, "source_update_preserves_identity": True,
            "temporal_sequence_executed": True,
            "reset_boundary_policy": "reset_reload_before_independent_episode",
            "counts_after_reset": {"object_count": 0, "source_count": 0, "listener_count": 0},
            "runtime": {"runtime_identity": copy.deepcopy(identity)},
            "upload_report": copy.deepcopy(_expected_upload_report(scene)),
        }

    def benchmark(_scene: Any, _simulation: Any, *, sources: Any, listener: Any,
                  repeat_count: int, **_kwargs: Any) -> dict[str, object]:
        del listener
        def condition(count: int) -> dict[str, object]:
            runs = [
                {"repeat_index": index, "wall_seconds": 0.01 * count,
                 "process_cpu_seconds": 0.005 * count, "peak_rss_before_bytes": 100,
                 "peak_rss_after_bytes": 200, "ir_payload_bytes": 128 * count,
                 "pair_count": count, "runtime_identity": copy.deepcopy(identity)}
                for index in range(repeat_count)
            ]
            return {
                "source_count": count, "pair_count": count, "repeat_count": repeat_count,
                "median_wall_seconds": 0.01 * count, "p95_wall_seconds": 0.01 * count,
                "median_process_cpu_seconds": 0.005 * count,
                "maximum_peak_rss_bytes": 200, "median_ir_payload_bytes": 128 * count,
                "runs": runs,
            }
        return {
            "one_source": condition(1), "multi_source": condition(len(sources)),
            "comparison": {"multi_to_one_median_wall_ratio": float(len(sources)),
             "multi_pair_throughput_pairs_per_second": 100.0, "hard_speed_gate": None,
             "interpretation": "measurement_only_platform_dependent"},
        }

    monkeypatch.setattr(canary, "render_named_sources", render)
    monkeypatch.setattr(canary, "exercise_endpoint_lifecycle", lifecycle)
    monkeypatch.setattr(canary, "benchmark_source_scaling", benchmark)
    return canary.run_m4_canary(
        M4_REQUEST, package, None, tmp_path / "current-v2",
        hrtf_path=hrtf, hrtf_license_path=license_path,
        runtime_mode="current-installed",
        runtime_prefix=identity["habitat_runtime_prefix"],
        rlr_sdk_root=identity["rlr_sdk_root"],
        magnum_python_site=identity["magnum_python_site"],
        current_hrtf_sample_rate_hz=16000,
        current_hrtf_license_id="fixture-license",
        current_hrtf_citation="fixture current-installed HRTF",
    )


def _artifact(path: Path, evidence: dict[str, object], role: str) -> Path:
    records = evidence["artifacts"]
    assert isinstance(records, dict) and isinstance(records[role], dict)
    return path.parent / str(records[role]["path"])


def _refresh(path: Path, evidence: dict[str, object], role: str) -> None:
    records = evidence["artifacts"]
    assert isinstance(records, dict)
    records[role] = artifact_record(_artifact(path, evidence, role), root=path.parent)


def _rehash(path: Path, evidence: dict[str, object]) -> None:
    evidence["evidence_content_sha256"] = canonical_json_sha256(
        {key: value for key, value in evidence.items() if key != "evidence_content_sha256"}
    )
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_wav(wav: Path, sidecar: Path) -> None:
    loaded = read_float32_wav(wav, sidecar_path=sidecar)
    samples = loaded.samples.copy()
    samples[0, 0] += np.float32(0.25)
    metadata = copy.deepcopy(loaded.sidecar["metadata"])
    wav.unlink()
    sidecar.unlink()
    write_float32_wav(wav, samples, loaded.sample_rate_hz, channel_axis=0,
                      metadata=metadata, sidecar_path=sidecar)


def _forgery(source: Path, root: Path, name: str, check_id: str,
             mutate: Callable[[Path, dict[str, object]], None]) -> None:
    destination = root / name
    shutil.copytree(source.parent, destination)
    path = destination / source.name
    evidence = json.loads(path.read_text(encoding="utf-8"))
    mutate(path, evidence)
    _rehash(path, evidence)
    status, checks = verify_m4_canary_evidence(path)
    assert status == "fail"
    assert any(check["check_id"] == check_id and check["status"] == "fail" for check in checks)


def test_current_installed_m4_recomputes_complete_fresh_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rehashed current v2 receipts cannot relabel altered artifacts as pass."""

    source = _full_bundle(tmp_path, monkeypatch)
    status, checks = verify_m4_canary_evidence(source)
    assert status == "pass"
    assert all(check["status"] == "pass" for check in checks)

    def ir(path: Path, evidence: dict[str, object]) -> None:
        pair = evidence["pairs"]["source0"]
        assert isinstance(pair, dict)
        role = str(pair["foa_ir_order_a_role"])
        target = _artifact(path, evidence, role)
        value = np.load(target, allow_pickle=False)
        value[0, 0] = np.float32(0.25)
        np.save(target, value, allow_pickle=False)
        _refresh(path, evidence, role)

    _forgery(source, tmp_path, "forged-ir", "source_order_invariance", ir)

    def dry(path: Path, evidence: dict[str, object]) -> None:
        pair = evidence["pairs"]["source0"]
        assert isinstance(pair, dict)
        wav, sidecar = str(pair["dry_wav_role"]), str(pair["dry_sidecar_role"])
        _rewrite_wav(_artifact(path, evidence, wav), _artifact(path, evidence, sidecar))
        _refresh(path, evidence, wav)
        _refresh(path, evidence, sidecar)

    _forgery(source, tmp_path, "forged-dry", "pair_artifacts_readable", dry)

    def stem(path: Path, evidence: dict[str, object]) -> None:
        pair = evidence["pairs"]["source0"]
        assert isinstance(pair, dict)
        wav, sidecar = str(pair["foa_stem_wav_role"]), str(pair["foa_stem_sidecar_role"])
        _rewrite_wav(_artifact(path, evidence, wav), _artifact(path, evidence, sidecar))
        _refresh(path, evidence, wav)
        _refresh(path, evidence, sidecar)

    _forgery(source, tmp_path, "forged-stem", "stem_and_mixture_reconstruction", stem)

    def mix(path: Path, evidence: dict[str, object]) -> None:
        mixtures = evidence["mixtures"]
        assert isinstance(mixtures, dict)
        wav, sidecar = str(mixtures["foa_wav_role"]), str(mixtures["foa_sidecar_role"])
        _rewrite_wav(_artifact(path, evidence, wav), _artifact(path, evidence, sidecar))
        _refresh(path, evidence, wav)
        _refresh(path, evidence, sidecar)

    _forgery(source, tmp_path, "forged-mix", "stem_and_mixture_reconstruction", mix)

    def foa(path: Path, evidence: dict[str, object]) -> None:
        probes = evidence["probes"]["foa"]
        assert isinstance(probes, dict) and isinstance(probes["cardinal_roles"], dict)
        role = str(probes["cardinal_roles"]["+X"])
        target = _artifact(path, evidence, role)
        value = np.load(target, allow_pickle=False)
        value[3, 20] = np.float32(0.0)
        np.save(target, value, allow_pickle=False)
        _refresh(path, evidence, role)

    _forgery(source, tmp_path, "forged-foa", "spatial_direction_probes", foa)

    def hrtf(path: Path, evidence: dict[str, object]) -> None:
        inputs = evidence["inputs"]
        assert isinstance(inputs, dict)
        role = str(inputs["hrtf_role"])
        target = _artifact(path, evidence, role)
        target.write_bytes(target.read_bytes() + b" forged")
        _refresh(path, evidence, role)

    _forgery(source, tmp_path, "forged-hrtf", "hrtf_license_and_rate_binding", hrtf)

    def readback(_path: Path, evidence: dict[str, object]) -> None:
        runtime = evidence["runtime"]
        assert isinstance(runtime, dict) and isinstance(runtime["foa_configuration_readback"], dict)
        runtime["foa_configuration_readback"]["direct_ray_count"] += 1

    _forgery(source, tmp_path, "forged-readback", "runtime_configuration_readback", readback)

    def lifecycle_distance(_path: Path, evidence: dict[str, object]) -> None:
        lifecycle = evidence["lifecycle"]
        assert isinstance(lifecycle, dict)
        lifecycle["moved_distance_m"] = 987.0

    _forgery(
        source,
        tmp_path,
        "forged-lifecycle-distance",
        "lifecycle_policy_receipt",
        lifecycle_distance,
    )

    def lifecycle_position(_path: Path, evidence: dict[str, object]) -> None:
        lifecycle = evidence["lifecycle"]
        assert isinstance(lifecycle, dict)
        position = lifecycle["updated_position_m"]
        assert isinstance(position, list)
        position[0] = float(position[0]) + 1.0

    _forgery(
        source,
        tmp_path,
        "forged-lifecycle-position",
        "lifecycle_policy_receipt",
        lifecycle_position,
    )

    def lifecycle_receipt_radius(_path: Path, evidence: dict[str, object]) -> None:
        lifecycle = evidence["lifecycle"]
        assert isinstance(lifecycle, dict)
        receipts = lifecycle["source_registration_receipts_after_update"]
        assert isinstance(receipts, list) and isinstance(receipts[0], dict)
        receipts[0]["radius_m"] = float(receipts[0]["radius_m"]) + 1.0

    _forgery(
        source,
        tmp_path,
        "forged-lifecycle-receipt-radius",
        "lifecycle_policy_receipt",
        lifecycle_receipt_radius,
    )

    def lifecycle_upload(_path: Path, evidence: dict[str, object]) -> None:
        lifecycle = evidence["lifecycle"]
        assert isinstance(lifecycle, dict) and isinstance(lifecycle["upload_report"], dict)
        lifecycle["upload_report"]["object_count"] += 1

    _forgery(
        source,
        tmp_path,
        "forged-lifecycle-upload",
        "lifecycle_policy_receipt",
        lifecycle_upload,
    )

    def one_source_summary(_path: Path, evidence: dict[str, object]) -> None:
        performance = evidence["performance"]
        assert isinstance(performance, dict) and isinstance(performance["one_source"], dict)
        performance["one_source"]["median_wall_seconds"] = 777.0

    _forgery(
        source,
        tmp_path,
        "forged-one-source-summary",
        "performance_report",
        one_source_summary,
    )

    def one_source_run(_path: Path, evidence: dict[str, object]) -> None:
        performance = evidence["performance"]
        assert isinstance(performance, dict) and isinstance(performance["one_source"], dict)
        runs = performance["one_source"]["runs"]
        assert isinstance(runs, list) and isinstance(runs[0], dict)
        runs[0]["wall_seconds"] = 666.0

    _forgery(source, tmp_path, "forged-one-source-run", "performance_report", one_source_run)

    def comparison_throughput(_path: Path, evidence: dict[str, object]) -> None:
        performance = evidence["performance"]
        assert isinstance(performance, dict) and isinstance(performance["comparison"], dict)
        performance["comparison"]["multi_pair_throughput_pairs_per_second"] = 777.0

    _forgery(
        source,
        tmp_path,
        "forged-comparison-throughput",
        "performance_report",
        comparison_throughput,
    )

    def multi_source_pair_count(_path: Path, evidence: dict[str, object]) -> None:
        performance = evidence["performance"]
        assert isinstance(performance, dict) and isinstance(performance["multi_source"], dict)
        performance["multi_source"]["pair_count"] = int(
            performance["multi_source"]["pair_count"]
        ) + 1

    _forgery(
        source,
        tmp_path,
        "forged-multi-source-pair-count",
        "performance_report",
        multi_source_pair_count,
    )

    historical_root = tmp_path / "historical-runtime-checkout"
    historical_root.mkdir()
    (historical_root / ".git").mkdir()
    historical = _identity(historical_root)
    historical_alias = tmp_path / "historical-runtime-checkout-alias"
    historical_alias.symlink_to(historical_root, target_is_directory=True)
    historical_symlink = _identity(historical_alias, canonical_paths=False)

    def replace_all_runtime_identities(
        _path: Path,
        evidence: dict[str, object],
        replacement: dict[str, object],
    ) -> None:
        runtime = evidence["runtime"]
        assert isinstance(runtime, dict)
        records = runtime["current_installed_identity_records"]
        assert isinstance(records, list)
        refreshed_records = [copy.deepcopy(replacement) for _ in records]
        runtime["current_installed_identity_records"] = refreshed_records
        runtime["current_installed_identity"] = copy.deepcopy(replacement)
        performance = evidence["performance"]
        assert isinstance(performance, dict)
        for condition_name in ("one_source", "multi_source"):
            condition = performance[condition_name]
            assert isinstance(condition, dict)
            runs = condition["runs"]
            assert isinstance(runs, list)
            for run in runs:
                assert isinstance(run, dict)
                run["runtime_identity"] = copy.deepcopy(replacement)
        checks = evidence["checks"]
        assert isinstance(checks, list)
        matching = [
            check
            for check in checks
            if isinstance(check, dict)
            and check["check_id"] == "runtime_current_installed_identity"
        ]
        assert len(matching) == 1
        measured = matching[0]["measured"]
        assert isinstance(measured, dict)
        measured["record_count"] = len(refreshed_records)
        measured["unique_identity_count"] = 1
        measured["identities"] = copy.deepcopy(refreshed_records)

    for name, replacement in (
        ("forged-checkout-runtime-identity", historical),
        ("forged-checkout-symlink-runtime-identity", historical_symlink),
    ):
        _forgery(
            source,
            tmp_path,
            name,
            "current_runtime_identity",
            lambda path, evidence, replacement=replacement: replace_all_runtime_identities(
                path, evidence, replacement
            ),
        )
