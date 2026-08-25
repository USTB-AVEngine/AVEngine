"""Executable M4 named multi-source, FOA and binaural canary."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import load_json, write_json
from avengine.current_installed_runtime import (
    is_current_installed_runtime_identity as _is_current_installed_runtime_identity,
)
from avengine.acoustics.metrics import analyze_ir
from avengine.spatial_audio.audio import (
    generate_sine_wave,
    read_float32_wav,
    render_stems_and_mix,
    write_float32_wav,
)
from avengine.spatial_audio.binaural import (
    build_rlr_native_binaural_metadata,
    rlr_native_binaural_contract,
    validate_binaural_cardinals,
)
from avengine.spatial_audio.contracts import (
    CURRENT_INSTALLED_EVIDENCE_SCHEMA,
    EVIDENCE_SCHEMA,
    M4ContractError,
    load_and_validate_multi_source_canary_request,
)
from avengine.spatial_audio.evidence import (
    M4EvidenceError,
    artifact_record,
    finalize_evidence,
    make_check,
    verify_m4_canary_evidence,
)
from avengine.acoustics.runtime import (
    RUNTIME_MODE_CURRENT_INSTALLED,
    RUNTIME_MODE_HISTORICAL,
    RuntimeAnchor,
    RuntimeUnavailableError,
    load_compiled_acoustic_scene,
    load_habitat_runtime,
    require_runtime_mode,
)
from avengine.spatial_audio.runtime import (
    M4SimulationConfig,
    benchmark_source_scaling,
    direct_only_simulation,
    exercise_endpoint_lifecycle,
    render_named_sources,
)
from avengine.spatial_audio.spatial import (
    rlr_foa_contract,
    rlr_foa_wav_metadata,
    validate_cardinal_foa,
    validate_world_aligned_foa,
)


class M4CanaryError(ValueError):
    """M4 inputs or generated evidence fail the executable gate."""


_DIRECTION_TO_ID = {
    "+X": "px",
    "-X": "nx",
    "+Y": "py",
    "-Y": "ny",
    "+Z": "pz",
    "-Z": "nz",
}
_DIRECTION_VECTOR = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}




def _current_installed_identity_summary(
    records: Sequence[Any],
) -> tuple[bool, dict[str, Any] | None, int]:
    unique: list[Any] = []
    for record in records:
        if record not in unique:
            unique.append(record)
    passed = bool(records) and all(
        _is_current_installed_runtime_identity(record)
        and record == records[0]
        for record in records
    )
    identity = copy.deepcopy(records[0]) if passed else None
    return passed, identity, len(unique)


def _current_runtime_call_kwargs(
    *,
    runtime_mode: str,
    runtime_prefix: str | Path | None,
    rlr_sdk_root: str | Path | None,
    magnum_python_site: str | Path | None,
) -> dict[str, str | Path]:
    if runtime_mode == RUNTIME_MODE_HISTORICAL:
        return {}
    missing = [
        name
        for name, value in (
            ("runtime_prefix", runtime_prefix),
            ("rlr_sdk_root", rlr_sdk_root),
            ("magnum_python_site", magnum_python_site),
        )
        if value is None or not str(value).strip()
    ]
    if missing:
        raise M4CanaryError(
            "current-installed M4 requires explicit " + ", ".join(missing)
        )
    return {
        "runtime_mode": runtime_mode,
        "runtime_prefix": runtime_prefix,
        "rlr_sdk_root": rlr_sdk_root,
        "magnum_python_site": magnum_python_site,
    }

def _copy_artifact(
    source: Path,
    destination: Path,
    *,
    role: str,
    root: Path,
    artifacts: dict[str, dict[str, Any]],
) -> Path:
    if role in artifacts:
        raise M4CanaryError(f"duplicate artifact role {role!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise M4CanaryError(f"refusing to replace staged artifact {destination}")
    shutil.copy2(source, destination)
    artifacts[role] = artifact_record(destination, root=root)
    return destination


def _record_tree(
    root_directory: Path,
    *,
    evidence_root: Path,
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, str]:
    roles: dict[str, str] = {}
    for index, path in enumerate(
        sorted(item for item in root_directory.rglob("*") if item.is_file())
    ):
        role = f"input_package_{index:03d}"
        artifacts[role] = artifact_record(path, root=evidence_root)
        roles[path.relative_to(root_directory).as_posix()] = role
    return roles


def _save_array(
    path: Path,
    value: np.ndarray,
    *,
    role: str,
    root: Path,
    artifacts: dict[str, dict[str, Any]],
) -> str:
    if role in artifacts:
        raise M4CanaryError(f"duplicate artifact role {role!r}")
    array = np.ascontiguousarray(value, dtype="<f4")
    if array.ndim != 2 or array.shape[1] < 2 or not np.all(np.isfinite(array)):
        raise M4CanaryError(f"array artifact {role} is malformed")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise M4CanaryError(f"refusing to replace staged array {path}")
    np.save(path, array, allow_pickle=False)
    artifacts[role] = artifact_record(path, root=root)
    return role


def _write_wav_roles(
    path: Path,
    samples: np.ndarray,
    sample_rate_hz: int,
    *,
    role_prefix: str,
    metadata: Mapping[str, Any],
    root: Path,
    artifacts: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    wav_role = f"{role_prefix}_wav"
    sidecar_role = f"{role_prefix}_sidecar"
    if wav_role in artifacts or sidecar_role in artifacts:
        raise M4CanaryError(f"duplicate WAVE roles for {role_prefix}")
    result = write_float32_wav(
        path,
        samples,
        sample_rate_hz,
        channel_axis=0,
        metadata=metadata,
    )
    artifacts[wav_role] = artifact_record(result.audio_path, root=root)
    artifacts[sidecar_role] = artifact_record(result.sidecar_path, root=root)
    # Authority arithmetic uses the exact float32 dry bytes that are retained,
    # so the verifier can reconstruct every stem without hidden precision.
    read_float32_wav(
        result.audio_path, sidecar_path=result.sidecar_path, verify_sidecar=True
    )
    return wav_role, sidecar_role


def _portable_binaural_metadata(
    preflight: Mapping[str, Any],
    *,
    hrtf_role: str,
    license_role: str,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(preflight))
    hrtf = result.get("hrtf")
    if isinstance(hrtf, dict):
        hrtf.pop("path", None)
        hrtf["artifact_role"] = hrtf_role
    rights = result.get("rights")
    if isinstance(rights, dict):
        rights.pop("license_text_path", None)
        rights["license_artifact_role"] = license_role
    return result


def _runtime_lock(path: Path) -> dict[str, Any]:
    try:
        value = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise M4CanaryError(f"invalid M4 runtime lock: {exc}") from exc
    if value.get("schema") != "avengine_m4_runtime_lock_v1":
        raise M4CanaryError("runtime lock schema is not avengine_m4_runtime_lock_v1")
    binaries = value.get("native_binaries")
    hrtf = value.get("hrtf")
    if not isinstance(binaries, Mapping) or not isinstance(hrtf, Mapping):
        raise M4CanaryError("runtime lock lacks native_binaries or hrtf pins")
    return value


def _runtime_binary_mismatches(
    lock: Mapping[str, Any], observed: Mapping[str, Any]
) -> list[str]:
    expected = lock.get("native_binaries")
    if not isinstance(expected, Mapping):
        return ["native_binaries"]
    errors: list[str] = []
    for name in ("habitat_sim_bindings", "rlr_audio_propagation"):
        expected_record = expected.get(name)
        observed_record = observed.get(name)
        if not isinstance(expected_record, Mapping) or not isinstance(
            observed_record, Mapping
        ):
            errors.append(name)
            continue
        for field in ("byte_size", "sha256"):
            if observed_record.get(field) != expected_record.get(field):
                errors.append(f"{name}.{field}")
    return errors


def _preflight_runtime_binary_lock(lock: Mapping[str, Any]) -> None:
    """Stop before an RLR job when a historical lock names other binaries."""

    _, runtime = load_habitat_runtime()
    observed = runtime.get("native_binaries")
    if not isinstance(observed, Mapping):
        raise RuntimeUnavailableError(
            "Installed Habitat runtime did not report native binary identity before "
            "the M4 RLR preflight."
        )
    mismatches = _runtime_binary_mismatches(lock, observed)
    if mismatches:
        raise RuntimeUnavailableError(
            "Installed Habitat/RLR binaries do not match the selected historical M4 "
            "runtime lock before executing an RLR job: " + ", ".join(mismatches)
        )


def _anchors(
    request: Mapping[str, Any],
) -> tuple[tuple[RuntimeAnchor, ...], RuntimeAnchor]:
    sources = tuple(
        RuntimeAnchor(
            anchor_id=item["source_id"],
            position_m=tuple(float(value) for value in item["position_m"]),
            radius_m=float(item["radius_m"]),
        )
        for item in request["sources"]
    )
    item = request["listeners"][0]
    listener = RuntimeAnchor(
        anchor_id=item["listener_id"],
        position_m=tuple(float(value) for value in item["position_m"]),
        radius_m=float(item["radius_m"]),
        orientation_wxyz=tuple(float(value) for value in item["orientation_wxyz"]),
    )
    return sources, listener


def _ordered_sources(
    source_ids: Sequence[str], sources: Sequence[RuntimeAnchor]
) -> tuple[RuntimeAnchor, ...]:
    by_id = {source.anchor_id: source for source in sources}
    if set(source_ids) != set(by_id) or len(source_ids) != len(by_id):
        raise M4CanaryError("registration order is not a permutation of source IDs")
    return tuple(by_id[source_id] for source_id in source_ids)


def _cardinal_sources(
    listener: RuntimeAnchor,
    distance_m: float,
    *,
    directions: Sequence[str],
) -> tuple[RuntimeAnchor, ...]:
    origin = np.asarray(listener.position_m, dtype=np.float64)
    return tuple(
        RuntimeAnchor(
            anchor_id=f"probe_{_DIRECTION_TO_ID[direction]}",
            position_m=tuple(
                float(value)
                for value in origin
                + np.asarray(_DIRECTION_VECTOR[direction], dtype=np.float64)
                * distance_m
            ),
            radius_m=0.0,
        )
        for direction in directions
    )


def _direction_arrays(result: Any, directions: Sequence[str]) -> dict[str, np.ndarray]:
    by_source = {pair.source_id: pair.samples for pair in result.pairs}
    return {
        direction: by_source[f"probe_{_DIRECTION_TO_ID[direction]}"]
        for direction in directions
    }


def run_m4_canary(
    request_path: str | Path,
    package_manifest_path: str | Path,
    runtime_lock_path: str | Path | None,
    output_directory: str | Path,
    *,
    hrtf_path: str | Path,
    hrtf_license_path: str | Path,
    runtime_mode: str = RUNTIME_MODE_HISTORICAL,
    runtime_prefix: str | Path | None = None,
    rlr_sdk_root: str | Path | None = None,
    magnum_python_site: str | Path | None = None,
    current_hrtf_sample_rate_hz: int | None = None,
    current_hrtf_license_id: str | None = None,
    current_hrtf_citation: str | None = None,
) -> Path:
    """Run M4 and publish only a self-verifying, atomic evidence bundle."""

    runtime_mode = require_runtime_mode(runtime_mode)
    runtime_call_kwargs = _current_runtime_call_kwargs(
        runtime_mode=runtime_mode,
        runtime_prefix=runtime_prefix,
        rlr_sdk_root=rlr_sdk_root,
        magnum_python_site=magnum_python_site,
    )
    if runtime_mode == RUNTIME_MODE_CURRENT_INSTALLED:
        missing_current_hrtf_inputs = [
            name
            for name, value in (
                ("current_hrtf_sample_rate_hz", current_hrtf_sample_rate_hz),
                ("current_hrtf_license_id", current_hrtf_license_id),
                ("current_hrtf_citation", current_hrtf_citation),
            )
            if value is None or not str(value).strip()
        ]
        if missing_current_hrtf_inputs:
            raise M4CanaryError(
                "current-installed M4 requires explicit "
                + ", ".join(missing_current_hrtf_inputs)
            )
    request_file = Path(request_path).resolve()
    package_manifest = Path(package_manifest_path).resolve()
    source_hrtf = Path(hrtf_path).resolve()
    source_license = Path(hrtf_license_path).resolve()
    lock_file: Path | None = None
    lock: dict[str, Any] | None = None
    if runtime_mode == RUNTIME_MODE_HISTORICAL:
        if runtime_lock_path is None:
            raise M4CanaryError("historical M4 requires a runtime lock path")
        lock_file = Path(runtime_lock_path).resolve()
    try:
        validated = load_and_validate_multi_source_canary_request(request_file)
    except M4ContractError as exc:
        raise M4CanaryError(str(exc)) from exc
    request = validated.request
    if runtime_mode == RUNTIME_MODE_HISTORICAL:
        assert lock_file is not None
        lock = _runtime_lock(lock_file)
    # Validate the source package completely before it is copied, then consume
    # only the private copy below.
    source_scene = load_compiled_acoustic_scene(package_manifest)
    package_root = source_scene.manifest_path.parent.resolve()

    destination = Path(output_directory).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise M4CanaryError(f"refusing to replace existing output: {destination}")
    if runtime_mode == RUNTIME_MODE_HISTORICAL:
        assert lock is not None
        _preflight_runtime_binary_lock(lock)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    ).resolve()
    artifacts: dict[str, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    try:
        input_root = staging / "inputs"
        _copy_artifact(
            request_file,
            input_root / "request.json",
            role="input_request",
            root=staging,
            artifacts=artifacts,
        )
        referenced_paths = {
            "m1_capture_request": validated.m1_capture_request_path,
            "m3_acoustic_canary_request": validated.m3_acoustic_canary_request_path,
            "source_identity_manifest": validated.identity_manifest_path,
        }
        input_roles: dict[str, str] = {"request_role": "input_request"}
        for name, path in referenced_paths.items():
            role = f"input_{name}"
            _copy_artifact(
                path,
                input_root / f"{name}.json",
                role=role,
                root=staging,
                artifacts=artifacts,
            )
            input_roles[f"{name}_role"] = role
        if runtime_mode == RUNTIME_MODE_HISTORICAL:
            assert lock_file is not None
            _copy_artifact(
                lock_file,
                input_root / "runtime_lock.json",
                role="input_runtime_lock",
                root=staging,
                artifacts=artifacts,
            )
            input_roles["runtime_lock_role"] = "input_runtime_lock"
        staged_hrtf = _copy_artifact(
            source_hrtf,
            input_root / "hrtf.sofa",
            role="input_hrtf",
            root=staging,
            artifacts=artifacts,
        )
        staged_license = _copy_artifact(
            source_license,
            input_root / "hrtf_license.txt",
            role="input_hrtf_license",
            root=staging,
            artifacts=artifacts,
        )
        input_roles.update(
            {"hrtf_role": "input_hrtf", "hrtf_license_role": "input_hrtf_license"}
        )
        staged_package_root = input_root / "acoustic_scene_package"
        shutil.copytree(package_root, staged_package_root, symlinks=False)
        package_roles = _record_tree(
            staged_package_root, evidence_root=staging, artifacts=artifacts
        )
        manifest_relative = source_scene.manifest_path.relative_to(package_root)
        input_roles["acoustic_scene_package_manifest_role"] = package_roles[
            manifest_relative.as_posix()
        ]
        input_roles["acoustic_scene_package_file_roles"] = package_roles
        scene = load_compiled_acoustic_scene(staged_package_root / manifest_relative)

        simulation = M4SimulationConfig.from_mapping(request["simulation"])
        sources, listener = _anchors(request)
        identity_manifest = load_json(referenced_paths["source_identity_manifest"])
        identities = {item["source_id"]: item for item in identity_manifest["sources"]}
        canonical_ids = tuple(request["canonical_source_order"])
        if set(identities) != set(canonical_ids):
            raise M4CanaryError("identity manifest/source request key set differs")

        orders = request["registration_orders"]
        if len(orders) < 2:
            raise M4CanaryError("M4 requires at least two registration orders")
        order_a_sources = _ordered_sources(orders[0]["source_ids"], sources)
        order_b_sources = _ordered_sources(orders[1]["source_ids"], sources)
        foa_a = render_named_sources(
            scene,
            simulation,
            sources=order_a_sources,
            listener=listener,
            layout_type="ambisonics",
            channel_count=4,
            **runtime_call_kwargs,
        )
        foa_b = render_named_sources(
            scene,
            simulation,
            sources=order_b_sources,
            listener=listener,
            layout_type="ambisonics",
            channel_count=4,
            **runtime_call_kwargs,
        )
        foa_a_by_id = {
            pair.source_id: np.ascontiguousarray(pair.samples, dtype="<f4")
            for pair in foa_a.pairs
        }
        foa_b_by_id = {
            pair.source_id: np.ascontiguousarray(pair.samples, dtype="<f4")
            for pair in foa_b.pairs
        }
        order_exact = all(
            np.array_equal(foa_a_by_id[source_id], foa_b_by_id[source_id])
            for source_id in canonical_ids
        )
        checks.append(
            make_check(
                "canonical_source_order_invariance",
                order_exact,
                measured={
                    "requested_orders": [
                        orders[0]["source_ids"],
                        orders[1]["source_ids"],
                    ],
                    "canonical_native_order_a": list(
                        foa_a.canonical_native_source_order
                    ),
                    "canonical_native_order_b": list(
                        foa_b.canonical_native_source_order
                    ),
                    "all_mapped_arrays_exact": order_exact,
                },
                threshold={"all_mapped_arrays_exact": True},
                failure_reason="Canonical mapped full-indirect RIRs changed with caller order",
            )
        )

        observed_binaries: Mapping[str, Any] | None = None
        if runtime_mode == RUNTIME_MODE_HISTORICAL:
            assert lock is not None
            observed_binaries = copy.deepcopy(foa_a.runtime["native_binaries"])
            binary_errors = _runtime_binary_mismatches(lock, observed_binaries)
            checks.append(
                make_check(
                    "runtime_binary_lock",
                    not binary_errors,
                    measured={"mismatches": binary_errors, "observed": observed_binaries},
                    threshold=lock["native_binaries"],
                    failure_reason="Loaded Habitat/RLR binaries differ from M4 lock",
                )
            )
            hrtf_lock = lock["hrtf"]
            preflight = build_rlr_native_binaural_metadata(
                int(simulation.sample_rate_hz),
                hrtf_path=staged_hrtf,
                expected_hrtf_sha256=hrtf_lock.get("sha256"),
                hrtf_sample_rate_hz=hrtf_lock.get("sample_rate_hz"),
                license_id=hrtf_lock.get("license_id"),
                citation=hrtf_lock.get("citation"),
                license_text_path=staged_license,
                expected_license_sha256=hrtf_lock.get("license_text_sha256"),
                asset_id=hrtf_lock.get("asset_id", "explicit_sofa_hrtf"),
                sample_rate_policy=hrtf_lock.get("sample_rate_policy", "strict_match"),
                rlr_binary_sha256=observed_binaries["rlr_audio_propagation"]["sha256"],
            )
            hrtf_threshold = {"status": "pass", "explicit_hash_and_license": True}
        else:
            # The v2 receipt has no external HRTF/binary lock. These hashes only
            # authenticate fresh copied inputs inside this output bundle; they are
            # not retained as a reusable baseline or runtime identity.
            preflight = build_rlr_native_binaural_metadata(
                int(simulation.sample_rate_hz),
                hrtf_path=staged_hrtf,
                expected_hrtf_sha256=artifacts["input_hrtf"]["sha256"],
                hrtf_sample_rate_hz=current_hrtf_sample_rate_hz,
                license_id=current_hrtf_license_id,
                citation=current_hrtf_citation,
                license_text_path=staged_license,
                expected_license_sha256=artifacts["input_hrtf_license"]["sha256"],
                asset_id="current_installed_explicit_sofa_hrtf",
                sample_rate_policy="strict_match",
            )
            hrtf_threshold = {"status": "pass", "current_explicit_inputs": True}
        checks.append(
            make_check(
                "explicit_hrtf_preflight",
                preflight.get("status") == "pass",
                measured=_portable_binaural_metadata(
                    preflight,
                    hrtf_role="input_hrtf",
                    license_role="input_hrtf_license",
                ),
                threshold=hrtf_threshold,
                failure_reason=str(preflight.get("reason", "HRTF preflight failed")),
                blocked=preflight.get("status") == "blocked",
            )
        )
        if preflight.get("status") != "pass":
            raise M4CanaryError(str(preflight.get("reason", "HRTF preflight failed")))
        binaural_metadata = _portable_binaural_metadata(
            preflight, hrtf_role="input_hrtf", license_role="input_hrtf_license"
        )
        binaural_render = render_named_sources(
            scene,
            simulation,
            sources=_ordered_sources(canonical_ids, sources),
            listener=listener,
            layout_type="binaural",
            channel_count=2,
            hrtf_file_path=str(staged_hrtf),
            **runtime_call_kwargs,
        )
        binaural_by_id = {
            pair.source_id: np.ascontiguousarray(pair.samples, dtype="<f4")
            for pair in binaural_render.pairs
        }

        pair_records: dict[str, dict[str, Any]] = {}
        dry_by_id: dict[str, np.ndarray] = {}
        for source_id in canonical_ids:
            identity = identities[source_id]
            signal = identity["deterministic_signal"]
            generated = generate_sine_wave(
                int(simulation.sample_rate_hz),
                int(signal["duration_samples"]),
                float(signal["frequency_hz"]),
                amplitude=float(signal["amplitude"]),
                phase_radians=float(signal["phase"]),
            )
            dry_wav_role, dry_sidecar_role = _write_wav_roles(
                staging / "audio" / "dry" / f"{source_id}.wav",
                generated[np.newaxis, :],
                int(simulation.sample_rate_hz),
                role_prefix=f"dry_{source_id}",
                metadata={
                    "audio_role": "deterministic_canary_dry",
                    "source_id": source_id,
                    "actor_id": identity["actor_id"],
                    "event_id": identity["event_id"],
                    "anchor_id": identity["anchor_id"],
                    "dry_audio_id": identity["dry_audio_id"],
                    "signal": signal,
                    "processing": "none",
                },
                root=staging,
                artifacts=artifacts,
            )
            dry_by_id[source_id] = (
                read_float32_wav(
                    staging / artifacts[dry_wav_role]["path"],
                    sidecar_path=staging / artifacts[dry_sidecar_role]["path"],
                )
                .samples[0]
                .astype(np.float64)
            )
            foa_a_role = _save_array(
                staging / "raw_ir" / "foa_order_a" / f"{source_id}.npy",
                foa_a_by_id[source_id],
                role=f"foa_a_{source_id}",
                root=staging,
                artifacts=artifacts,
            )
            foa_b_role = _save_array(
                staging / "raw_ir" / "foa_order_b" / f"{source_id}.npy",
                foa_b_by_id[source_id],
                role=f"foa_b_{source_id}",
                root=staging,
                artifacts=artifacts,
            )
            binaural_role = _save_array(
                staging / "raw_ir" / "binaural" / f"{source_id}.npy",
                binaural_by_id[source_id],
                role=f"binaural_ir_{source_id}",
                root=staging,
                artifacts=artifacts,
            )
            distance = math.dist(
                next(
                    source.position_m
                    for source in sources
                    if source.anchor_id == source_id
                ),
                listener.position_m,
            )
            expected_arrival = (
                distance / simulation.speed_of_sound_m_s * simulation.sample_rate_hz
            )
            detected = analyze_ir(
                foa_a_by_id[source_id], simulation.sample_rate_hz
            ).direct_arrival_sample
            maximum_error = float(
                request["thresholds"]["maximum_direct_arrival_error_samples"]
            )
            direct_arrival = {
                "distance_m": distance,
                "speed_of_sound_m_s": simulation.speed_of_sound_m_s,
                "expected_sample": expected_arrival,
                "detected_sample": detected,
                "absolute_error_samples": abs(detected - expected_arrival),
                "maximum_absolute_error_samples": maximum_error,
            }
            pair_records[source_id] = {
                "listener_id": listener.anchor_id,
                "source_id": source_id,
                "actor_id": identity["actor_id"],
                "event_id": identity["event_id"],
                "anchor_id": identity["anchor_id"],
                "semantic_anchor_id": identity["semantic_anchor_id"],
                "dry_audio_id": identity["dry_audio_id"],
                "dry_wav_role": dry_wav_role,
                "dry_sidecar_role": dry_sidecar_role,
                "foa_ir_order_a_role": foa_a_role,
                "foa_ir_order_b_role": foa_b_role,
                "binaural_ir_role": binaural_role,
                "direct_arrival": direct_arrival,
            }
            checks.append(
                make_check(
                    f"direct_arrival_{source_id}",
                    direct_arrival["absolute_error_samples"] <= maximum_error,
                    measured=direct_arrival,
                    threshold={"maximum_absolute_error_samples": maximum_error},
                    failure_reason=f"{source_id} direct arrival differs from geometry",
                )
            )

        foa_stems, foa_mix = render_stems_and_mix(dry_by_id, foa_a_by_id)
        binaural_stems, binaural_mix = render_stems_and_mix(dry_by_id, binaural_by_id)
        for source_id in canonical_ids:
            foa_wav_role, foa_sidecar_role = _write_wav_roles(
                staging / "audio" / "foa_stems" / f"{source_id}.wav",
                foa_stems[source_id],
                int(simulation.sample_rate_hz),
                role_prefix=f"foa_stem_{source_id}",
                metadata={
                    **rlr_foa_wav_metadata(),
                    "audio_role": "per_source_wet_stem",
                    "source_id": source_id,
                    "listener_id": listener.anchor_id,
                    "lineage": "dry_linear_convolution_pair_foa_ir",
                    "linear_gain": 1.0,
                    "normalization": "none",
                    "tail_policy": "full_linear_convolution",
                },
                root=staging,
                artifacts=artifacts,
            )
            pair_records[source_id].update(
                {
                    "foa_stem_wav_role": foa_wav_role,
                    "foa_stem_sidecar_role": foa_sidecar_role,
                }
            )
        foa_mix_roles = _write_wav_roles(
            staging / "audio" / "mixtures" / "canary_foa_mix.wav",
            foa_mix,
            int(simulation.sample_rate_hz),
            role_prefix="foa_mix",
            metadata={
                **rlr_foa_wav_metadata(),
                "audio_role": "m4_canary_full_tail_mixture",
                "source_ids": list(canonical_ids),
                "summation_order": list(canonical_ids),
                "normalization": "none",
                "limiter": "none",
                "timeline_crop": "not_applied_m5_owned",
            },
            root=staging,
            artifacts=artifacts,
        )

        probe_distance = float(request["thresholds"]["cardinal_probe_distance_m"])
        identity_listener = RuntimeAnchor(
            "probe_listener",
            listener.position_m,
            radius_m=0.0,
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        rotated_listener = RuntimeAnchor(
            "probe_listener",
            listener.position_m,
            radius_m=0.0,
            orientation_wxyz=listener.orientation_wxyz,
        )
        directions = tuple(_DIRECTION_TO_ID)
        cardinal_sources = _cardinal_sources(
            identity_listener, probe_distance, directions=directions
        )
        direct_foa = direct_only_simulation(
            simulation, layout_type="ambisonics", channel_count=4
        )
        foa_identity_probe = render_named_sources(
            scene,
            direct_foa,
            sources=cardinal_sources,
            listener=identity_listener,
            **runtime_call_kwargs,
        )
        foa_rotated_probe = render_named_sources(
            scene,
            direct_foa,
            sources=cardinal_sources,
            listener=rotated_listener,
            **runtime_call_kwargs,
        )
        foa_cardinals = _direction_arrays(foa_identity_probe, directions)
        foa_rotated = _direction_arrays(foa_rotated_probe, directions)
        foa_probe_report = validate_cardinal_foa(foa_cardinals)
        world_report = validate_world_aligned_foa(
            foa_cardinals["+X"], foa_rotated["+X"]
        )
        foa_cardinal_roles: dict[str, str] = {}
        for direction in directions:
            suffix = _DIRECTION_TO_ID[direction]
            foa_cardinal_roles[direction] = _save_array(
                staging / "probes" / "foa" / f"{suffix}.npy",
                foa_cardinals[direction],
                role=f"foa_probe_{suffix}",
                root=staging,
                artifacts=artifacts,
            )
        world_rotated_role = _save_array(
            staging / "probes" / "foa" / "px_rotated_listener.npy",
            foa_rotated["+X"],
            role="foa_probe_px_rotated",
            root=staging,
            artifacts=artifacts,
        )

        binaural_directions = ("+X", "-X")
        binaural_probe_sources = _cardinal_sources(
            identity_listener, probe_distance, directions=binaural_directions
        )
        direct_binaural = direct_only_simulation(
            simulation, layout_type="binaural", channel_count=2
        )
        binaural_probe_render = render_named_sources(
            scene,
            direct_binaural,
            sources=binaural_probe_sources,
            listener=identity_listener,
            hrtf_file_path=str(staged_hrtf),
            **runtime_call_kwargs,
        )
        binaural_cardinals = _direction_arrays(
            binaural_probe_render, binaural_directions
        )
        minimum_ild = float(request["thresholds"]["minimum_binaural_ild_db"])
        binaural_probe_report = validate_binaural_cardinals(
            binaural_cardinals, minimum_ild_db=minimum_ild
        )
        binaural_cardinal_roles: dict[str, str] = {}
        for direction in binaural_directions:
            suffix = _DIRECTION_TO_ID[direction]
            binaural_cardinal_roles[direction] = _save_array(
                staging / "probes" / "binaural" / f"{suffix}.npy",
                binaural_cardinals[direction],
                role=f"binaural_probe_{suffix}",
                root=staging,
                artifacts=artifacts,
            )
        checks.extend(
            [
                make_check(
                    "foa_cardinal_axis_contract",
                    True,
                    measured=foa_probe_report,
                    threshold={"format_id": "rlr_foa_acn_n3d_world_v1"},
                    failure_reason="FOA cardinal axis contract failed",
                ),
                make_check(
                    "foa_world_alignment",
                    True,
                    measured=world_report,
                    threshold={"maximum_absolute_difference": 0.0},
                    failure_reason="Raw FOA rotated with listener pose",
                ),
                make_check(
                    "binaural_left_right_direction",
                    True,
                    measured=binaural_probe_report,
                    threshold={"minimum_ild_db": minimum_ild},
                    failure_reason="Explicit HRTF binaural direction failed",
                ),
            ]
        )

        lifecycle_result = exercise_endpoint_lifecycle(
            scene,
            simulation,
            sources=_ordered_sources(canonical_ids, sources),
            listener=listener,
            **runtime_call_kwargs,
        )
        lifecycle_roles: dict[str, dict[str, str]] = {
            "fresh_first_roles": {},
            "updated_roles": {},
            "reset_first_roles": {},
        }
        for phase, evidence_key in (
            ("fresh_first", "fresh_first_roles"),
            ("updated", "updated_roles"),
            ("reset_first", "reset_first_roles"),
        ):
            for source_id in canonical_ids:
                role = _save_array(
                    staging / "lifecycle" / phase / f"{source_id}.npy",
                    lifecycle_result[phase][source_id],
                    role=f"lifecycle_{phase}_{source_id}",
                    root=staging,
                    artifacts=artifacts,
                )
                lifecycle_roles[evidence_key][source_id] = role
        lifecycle_evidence = {
            **{
                key: copy.deepcopy(value)
                for key, value in lifecycle_result.items()
                if key not in {"fresh_first", "updated", "reset_first", "runtime"}
            },
            **lifecycle_roles,
        }
        checks.append(
            make_check(
                "endpoint_lifecycle",
                lifecycle_evidence["reset_matches_fresh_first"] is True
                and lifecycle_evidence["source_update_preserves_identity"] is True
                and lifecycle_evidence["temporal_sequence_executed"] is True,
                measured=lifecycle_evidence,
                threshold={
                    "reset_matches_fresh_first": True,
                    "source_update_preserves_identity": True,
                    "temporal_sequence_executed": True,
                },
                failure_reason="Source update/temporal/reset lifecycle failed",
            )
        )

        performance = benchmark_source_scaling(
            scene,
            simulation,
            sources=_ordered_sources(canonical_ids, sources),
            listener=listener,
            repeat_count=int(request["thresholds"]["performance_repeat_count"]),
            **runtime_call_kwargs,
        )
        checks.append(
            make_check(
                "multi_source_performance_report",
                math.isfinite(
                    float(
                        performance["comparison"][
                            "multi_pair_throughput_pairs_per_second"
                        ]
                    )
                )
                and float(
                    performance["comparison"]["multi_pair_throughput_pairs_per_second"]
                )
                > 0.0,
                measured=performance,
                threshold={"measurement_complete": True, "hard_speed_gate": None},
                failure_reason="One-source/multi-source performance report is invalid",
            )
        )

        current_identity_records: list[Any] = []
        current_identity_passed = True
        current_identity: dict[str, Any] | None = None
        current_identity_count = 0
        if runtime_mode == RUNTIME_MODE_CURRENT_INSTALLED:
            current_identity_records = [
                foa_a.runtime.get("runtime_identity"),
                foa_b.runtime.get("runtime_identity"),
                binaural_render.runtime.get("runtime_identity"),
                foa_identity_probe.runtime.get("runtime_identity"),
                foa_rotated_probe.runtime.get("runtime_identity"),
                binaural_probe_render.runtime.get("runtime_identity"),
                lifecycle_result.get("runtime", {}).get("runtime_identity"),
            ]
            for condition in ("one_source", "multi_source"):
                for run in performance.get(condition, {}).get("runs", []):
                    current_identity_records.append(
                        run.get("runtime_identity")
                        if isinstance(run, Mapping)
                        else None
                    )
            (
                current_identity_passed,
                current_identity,
                current_identity_count,
            ) = _current_installed_identity_summary(current_identity_records)
            checks.append(
                make_check(
                    "runtime_current_installed_identity",
                    current_identity_passed,
                    measured={
                        "record_count": len(current_identity_records),
                        "unique_identity_count": current_identity_count,
                        "identities": copy.deepcopy(current_identity_records),
                    },
                    threshold={
                        "same_runtime_identity_every_native_call": True,
                        "runtime_mode": RUNTIME_MODE_CURRENT_INSTALLED,
                    },
                    failure_reason=(
                        "Current-installed Habitat/RLR SDK/binding identity changed "
                        "between fresh M4 native calls"
                    ),
                )
            )

        binaural_metadata["native_cardinal_validation"] = "pass"
        binaural_metadata["native_cardinal_report"] = binaural_probe_report
        # Binaural files are published only after the native cardinal canary has
        # passed.  This makes the authenticated sidecars and the enclosing
        # evidence carry the same final HRTF/direction qualification state.
        for source_id in canonical_ids:
            bin_wav_role, bin_sidecar_role = _write_wav_roles(
                staging / "audio" / "binaural_stems" / f"{source_id}.wav",
                binaural_stems[source_id],
                int(simulation.sample_rate_hz),
                role_prefix=f"binaural_stem_{source_id}",
                metadata={
                    "spatial_format": rlr_native_binaural_contract(),
                    "hrtf": binaural_metadata,
                    "audio_role": "per_source_wet_stem",
                    "source_id": source_id,
                    "listener_id": listener.anchor_id,
                    "lineage": "dry_linear_convolution_pair_binaural_ir",
                    "linear_gain": 1.0,
                    "normalization": "none",
                    "tail_policy": "full_linear_convolution",
                },
                root=staging,
                artifacts=artifacts,
            )
            pair_records[source_id].update(
                {
                    "binaural_stem_wav_role": bin_wav_role,
                    "binaural_stem_sidecar_role": bin_sidecar_role,
                }
            )
        binaural_mix_roles = _write_wav_roles(
            staging / "audio" / "mixtures" / "canary_binaural_mix.wav",
            binaural_mix,
            int(simulation.sample_rate_hz),
            role_prefix="binaural_mix",
            metadata={
                "spatial_format": rlr_native_binaural_contract(),
                "hrtf": binaural_metadata,
                "audio_role": "m4_canary_full_tail_mixture",
                "source_ids": list(canonical_ids),
                "summation_order": list(canonical_ids),
                "normalization": "none",
                "limiter": "none",
                "timeline_crop": "not_applied_m5_owned",
            },
            root=staging,
            artifacts=artifacts,
        )
        execution: dict[str, Any] = {
            "one_context_per_output_layout": True,
            "foa_context_listener_count": 1,
            "binaural_context_listener_count": 1,
            "requested_registration_orders": [
                copy.deepcopy(orders[0]),
                copy.deepcopy(orders[1]),
            ],
            "canonical_native_source_order": list(foa_a.canonical_native_source_order),
            "static_authority_policy": "fresh_context_temporal_false",
            "independent_episode_policy": "reset_reload_before_first_frame",
        }
        runtime_evidence: dict[str, Any] = {
            "binding_api": foa_a.runtime.get("binding_api"),
            "foa_configuration_readback": foa_a.runtime.get(
                "configuration_readback"
            ),
            "binaural_configuration_readback": binaural_render.runtime.get(
                "configuration_readback"
            ),
            "foa_upload_report": foa_a.upload_report,
            "foa_endpoint_receipts": foa_a.endpoint_receipts,
            "binaural_endpoint_receipts": binaural_render.endpoint_receipts,
        }
        if runtime_mode == RUNTIME_MODE_HISTORICAL:
            runtime_evidence["native_binaries"] = observed_binaries
        else:
            execution["runtime_mode"] = RUNTIME_MODE_CURRENT_INSTALLED
            runtime_evidence.update(
                {
                    "runtime_mode": RUNTIME_MODE_CURRENT_INSTALLED,
                    "current_installed_identity": current_identity,
                    "current_installed_identity_records": copy.deepcopy(
                        current_identity_records
                    ),
                }
            )

        evidence: dict[str, Any] = {
            "schema": (
                CURRENT_INSTALLED_EVIDENCE_SCHEMA
                if runtime_mode == RUNTIME_MODE_CURRENT_INSTALLED
                else EVIDENCE_SCHEMA
            ),
            "request_id": request["request_id"],
            "qualification_claim": runtime_mode == RUNTIME_MODE_HISTORICAL,
            "overall_status": "fail",
            "failure_reasons": [],
            "artifacts": artifacts,
            "inputs": {
                **input_roles,
                "request_content_sha256": request["request_content_sha256"],
                "package_id": scene.package_id,
                "package_content_sha256": scene.package_content_sha256,
            },
            "identity": {
                "listener_id": listener.anchor_id,
                "listener_count": 1,
                "canonical_source_ids": list(canonical_ids),
                "source_count": len(canonical_ids),
                "source_identities": {
                    source_id: {
                        key: copy.deepcopy(identities[source_id][key])
                        for key in (
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
                    }
                    for source_id in canonical_ids
                },
            },
            "execution": execution,
            "audio_contracts": {
                "foa": rlr_foa_contract(),
                "binaural": binaural_metadata,
                "stem_equation": "wet[source,channel]=dry[source]*rir[listener,source,channel]",
                "mixture_equation": "canonical_float64_sum(per_source_stems)",
                "implicit_normalization": False,
                "avengine_resampling_performed": False,
                "native_rate_adaptation": copy.deepcopy(
                    binaural_metadata["sample_rate_binding"]
                ),
                "limiter": False,
                "m4_tail_policy": "full_linear_convolution",
                "m5_video_mux": "not_owned_by_m4",
            },
            "pairs": pair_records,
            "mixtures": {
                "foa_wav_role": foa_mix_roles[0],
                "foa_sidecar_role": foa_mix_roles[1],
                "binaural_wav_role": binaural_mix_roles[0],
                "binaural_sidecar_role": binaural_mix_roles[1],
                "source_ids": list(canonical_ids),
                "summation_order": list(canonical_ids),
            },
            "probes": {
                "foa": {
                    "cardinal_roles": foa_cardinal_roles,
                    "cardinal_report": foa_probe_report,
                    "world_identity_role": foa_cardinal_roles["+X"],
                    "world_rotated_role": world_rotated_role,
                    "world_alignment_report": world_report,
                },
                "binaural": {
                    "cardinal_roles": binaural_cardinal_roles,
                    "minimum_ild_db": minimum_ild,
                    "cardinal_report": binaural_probe_report,
                },
            },
            "lifecycle": lifecycle_evidence,
            "performance": performance,
            "runtime": runtime_evidence,
            "checks": checks,
            "evidence_content_sha256": "0" * 64,
        }
        finalize_evidence(evidence)
        if (
            runtime_mode == RUNTIME_MODE_HISTORICAL
            and evidence["overall_status"] != "pass"
        ):
            raise M4CanaryError("M4 checks did not all pass")
        evidence_path = staging / "m4_canary_evidence.json"
        write_json(evidence_path, evidence)
        verification_status, verification_checks = verify_m4_canary_evidence(
            evidence_path
        )
        verification_valid = all(
            check.get("status") == "pass" for check in verification_checks
        )
        if not verification_valid or (
            runtime_mode == RUNTIME_MODE_HISTORICAL
            and verification_status != "pass"
        ):
            failures = [
                check.get("failure_reason", check["check_id"])
                for check in verification_checks
                if check["status"] != "pass"
            ]
            raise M4EvidenceError(
                "generated M4 evidence failed independent verification: "
                + "; ".join(str(value) for value in failures)
            )
        os.rename(staging, destination)
        return destination / evidence_path.name
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = ["M4CanaryError", "run_m4_canary"]
