"""Independent artifact verification for the executable M4 canary."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
)
from avengine.contracts.transforms import (
    compose_transforms,
    normalized_quaternion_xyzw,
)
from avengine.current_installed_runtime import (
    is_current_installed_runtime_identity as _is_current_installed_runtime_identity,
)
from avengine.m1.contracts import (
    validate_capture_request as validate_m1_capture_request,
)
from avengine.m3.contracts import validate_canary_request as validate_m3_canary_request
from avengine.m3.metrics import AcousticMetricError, analyze_ir
from avengine.m3.runtime import (
    RuntimeContractError,
    _verify_upload_report,
    load_compiled_acoustic_scene,
)
from avengine.spatial_audio.audio import (
    AudioContractError,
    generate_sine_wave,
    read_float32_wav,
    render_stems_and_mix,
)
from avengine.spatial_audio.binaural import (
    BinauralContractError,
    build_rlr_native_binaural_metadata,
    rlr_native_binaural_contract,
    validate_binaural_cardinals,
)
from avengine.spatial_audio.contracts import (
    CURRENT_INSTALLED_EVIDENCE_SCHEMA,
    EVIDENCE_SCHEMA,
    IDENTITY_SCHEMA,
    REQUEST_SCHEMA,
    canonical_source_ids,
    json_schema_errors,
    validate_current_installed_multi_source_canary_evidence,
    validate_multi_source_canary_evidence,
)
from avengine.spatial_audio.runtime import LIFECYCLE_MOVED_DISTANCE_M
from avengine.spatial_audio.spatial import (
    SpatialContractError,
    rlr_foa_contract,
    rlr_foa_wav_metadata,
    validate_cardinal_foa,
    validate_world_aligned_foa,
)


class M4EvidenceError(ValueError):
    """M4 evidence is malformed, escaping, or semantically inconsistent."""


def artifact_record(path: str | Path, *, root: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    base = Path(root).resolve()
    return {
        "path": resolved.relative_to(base).as_posix(),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def array_content_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "memory_order": "C",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def finalize_evidence(evidence: dict[str, Any]) -> None:
    required = [
        check
        for check in evidence.get("checks", [])
        if check.get("required", True) is True
    ]
    statuses = {check.get("status") for check in required}
    if "fail" in statuses:
        overall = "fail"
    elif "blocked" in statuses or "not_run" in statuses:
        overall = "blocked"
    else:
        overall = "pass"
    evidence["overall_status"] = overall
    evidence["failure_reasons"] = [
        str(check.get("failure_reason", check.get("check_id")))
        for check in required
        if check.get("status") != "pass"
    ]
    evidence.pop("evidence_content_sha256", None)
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)


def make_check(
    check_id: str,
    passed: bool,
    *,
    measured: Any,
    threshold: Any,
    failure_reason: str,
    blocked: bool = False,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "required": True,
        "status": "pass" if passed else ("blocked" if blocked else "fail"),
        "measured": measured,
        "threshold": threshold,
        **({} if passed else {"failure_reason": failure_reason}),
    }


def _derived_check(
    check_id: str,
    passed: bool,
    detail: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return make_check(
        check_id,
        passed,
        measured=detail,
        threshold={"independently_recomputed": True},
        failure_reason=failure_reason,
    )


def _confined_artifacts(
    evidence_path: Path,
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Path], list[str]]:
    base = evidence_path.parent.resolve()
    records = evidence.get("artifacts")
    if not isinstance(records, Mapping) or not records:
        return {}, ["artifacts must be a non-empty role mapping"]
    paths: dict[str, Path] = {}
    errors: list[str] = []
    for role, record in records.items():
        if not isinstance(role, str) or not role or not isinstance(record, Mapping):
            errors.append(f"invalid artifact role/record {role!r}")
            continue
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"{role}: missing path")
            continue
        declared = Path(raw_path)
        if declared.is_absolute() or ".." in declared.parts:
            errors.append(f"{role}: path is absolute or escaping")
            continue
        path = (base / declared).resolve()
        try:
            path.relative_to(base)
        except ValueError:
            errors.append(f"{role}: path or symlink escapes evidence root")
            continue
        try:
            size = path.stat().st_size
            digest = sha256_file(path)
        except OSError as exc:
            errors.append(f"{role}: missing or unreadable: {exc}")
            continue
        if size != record.get("byte_size"):
            errors.append(f"{role}: byte_size mismatch")
        if digest != record.get("sha256"):
            errors.append(f"{role}: sha256 mismatch")
        if path in paths.values():
            errors.append(f"{role}: two artifact roles alias one path")
        paths[role] = path
    return paths, errors


def _json_input_role(
    paths: Mapping[str, Path],
    inputs: Mapping[str, Any],
    key: str,
    *,
    owner: str,
    errors: list[str],
) -> dict[str, Any] | None:
    role = inputs.get(key)
    if not isinstance(role, str) or role not in paths:
        errors.append(f"{owner}: artifact role is missing")
        return None
    try:
        value = load_json(paths[role])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{owner}: invalid JSON: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{owner}: root must be an object")
        return None
    return value


def _document_content_hash_errors(
    value: Mapping[str, Any],
    field: str,
    *,
    owner: str,
) -> list[str]:
    try:
        actual = canonical_json_sha256(
            {key: item for key, item in value.items() if key != field}
        )
    except (TypeError, ValueError) as exc:
        return [f"{owner}: {field} cannot be recomputed: {exc}"]
    if value.get(field) != actual:
        return [f"{owner}: {field} does not match retained document bytes"]
    return []


def _file_record_errors(
    record: Any,
    path: Path,
    *,
    owner: str,
) -> list[str]:
    if not isinstance(record, Mapping):
        return [f"{owner}: file record is missing"]
    try:
        size = path.stat().st_size
        digest = sha256_file(path)
    except OSError as exc:
        return [f"{owner}: retained file is unreadable: {exc}"]
    errors: list[str] = []
    if record.get("byte_size") != size:
        errors.append(f"{owner}: byte_size differs from retained bytes")
    if record.get("sha256") != digest:
        errors.append(f"{owner}: sha256 differs from retained bytes")
    return errors


def _maximum_vector_error(left: Any, right: Any, *, length: int) -> float:
    try:
        left_array = np.asarray(left, dtype=np.float64)
        right_array = np.asarray(right, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return math.inf
    if (
        left_array.shape != (length,)
        or right_array.shape != (length,)
        or not np.all(np.isfinite(left_array))
        or not np.all(np.isfinite(right_array))
    ):
        return math.inf
    return float(np.max(np.abs(left_array - right_array), initial=0.0))


def _input_authority(
    paths: Mapping[str, Path],
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Rebuild source/listener/package authority from retained input bytes."""

    errors: list[str] = []
    authority: dict[str, Any] = {}
    inputs = evidence.get("inputs")
    if not isinstance(inputs, Mapping):
        return authority, ["evidence inputs are missing"]
    request = _json_input_role(
        paths,
        inputs,
        "request_role",
        owner="M4 request",
        errors=errors,
    )
    m1 = _json_input_role(
        paths,
        inputs,
        "m1_capture_request_role",
        owner="M1 capture request",
        errors=errors,
    )
    m3 = _json_input_role(
        paths,
        inputs,
        "m3_acoustic_canary_request_role",
        owner="M3 acoustic request",
        errors=errors,
    )
    identity = _json_input_role(
        paths,
        inputs,
        "source_identity_manifest_role",
        owner="source identity manifest",
        errors=errors,
    )
    authority.update({"request": request, "m1": m1, "m3": m3, "identity": identity})
    if any(value is None for value in (request, m1, m3, identity)):
        return authority, errors
    assert request is not None and m1 is not None and m3 is not None
    assert identity is not None

    errors.extend(
        f"M4 request: {item}" for item in json_schema_errors(request, REQUEST_SCHEMA)
    )
    errors.extend(
        _document_content_hash_errors(
            request,
            "request_content_sha256",
            owner="M4 request",
        )
    )
    errors.extend(
        f"source identity manifest: {item}"
        for item in json_schema_errors(identity, IDENTITY_SCHEMA)
    )
    errors.extend(
        _document_content_hash_errors(
            identity,
            "manifest_content_sha256",
            owner="source identity manifest",
        )
    )
    errors.extend(
        f"M1 capture request: {item}" for item in validate_m1_capture_request(m1)
    )
    errors.extend(
        f"M3 acoustic request: {item}" for item in validate_m3_canary_request(m3)
    )

    request_inputs = request.get("inputs")
    request_inputs = request_inputs if isinstance(request_inputs, Mapping) else {}
    role_by_request_input = {
        "m1_capture_request": "m1_capture_request_role",
        "m3_acoustic_canary_request": "m3_acoustic_canary_request_role",
        "source_identity_manifest": "source_identity_manifest_role",
    }
    for input_name, role_key in role_by_request_input.items():
        role = inputs.get(role_key)
        if isinstance(role, str) and role in paths:
            errors.extend(
                _file_record_errors(
                    request_inputs.get(input_name),
                    paths[role],
                    owner=f"M4 request inputs.{input_name}",
                )
            )
    m1_role = inputs.get("m1_capture_request_role")
    if isinstance(m1_role, str) and m1_role in paths:
        errors.extend(
            _file_record_errors(
                identity.get("m1_capture_request"),
                paths[m1_role],
                owner="source identity manifest m1_capture_request",
            )
        )

    if evidence.get("request_id") != request.get("request_id"):
        errors.append("evidence request_id differs from retained M4 request")
    if inputs.get("request_content_sha256") != request.get("request_content_sha256"):
        errors.append("evidence request hash differs from retained M4 request")

    sources = request.get("sources")
    listeners = request.get("listeners")
    identity_sources_raw = identity.get("sources")
    if (
        not isinstance(sources, list)
        or not isinstance(listeners, list)
        or len(listeners) != 1
        or not isinstance(identity_sources_raw, list)
    ):
        errors.append("retained request/identity endpoint arrays are malformed")
        return authority, errors
    source_by_id = {
        item.get("source_id"): item for item in sources if isinstance(item, Mapping)
    }
    identity_by_id = {
        item.get("source_id"): item
        for item in identity_sources_raw
        if isinstance(item, Mapping)
    }
    try:
        canonical_ids = canonical_source_ids(source_by_id)
    except Exception as exc:  # schema errors above retain the precise root cause
        errors.append(f"retained request source IDs are invalid: {exc}")
        canonical_ids = ()
    authority.update(
        {
            "listener": listeners[0],
            "sources": source_by_id,
            "identities": identity_by_id,
            "canonical_source_ids": canonical_ids,
        }
    )
    if list(canonical_ids) != request.get("canonical_source_order"):
        errors.append("retained M4 request canonical source order is inconsistent")
    if list(canonical_ids) != identity.get("canonical_source_order"):
        errors.append("retained identity canonical source order is inconsistent")
    if set(source_by_id) != set(identity_by_id):
        errors.append("retained request and identity source sets differ")

    tolerance_value = request.get("thresholds", {}).get(
        "maximum_anchor_transform_error", 1.0e-9
    )
    tolerance = (
        float(tolerance_value)
        if isinstance(tolerance_value, (int, float))
        and not isinstance(tolerance_value, bool)
        else 1.0e-9
    )
    listener = listeners[0]
    rig = m1.get("primary_camera_rig")
    m1_listener = m1.get("listener")
    if (
        isinstance(listener, Mapping)
        and isinstance(rig, Mapping)
        and isinstance(m1_listener, Mapping)
    ):
        try:
            world_from_listener = compose_transforms(
                rig["world_from_rig"], m1_listener["rig_from_listener"]
            )
            expected_xyzw = normalized_quaternion_xyzw(
                world_from_listener["rotation_xyzw"]
            )
            expected_wxyz = [
                float(expected_xyzw[3]),
                float(expected_xyzw[0]),
                float(expected_xyzw[1]),
                float(expected_xyzw[2]),
            ]
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"M1 listener transform cannot be composed: {exc}")
        else:
            if listener.get("listener_id") != m1_listener.get("listener_id"):
                errors.append("retained listener ID differs from M1")
            if listener.get("camera_rig_id") != rig.get("rig_id"):
                errors.append("retained listener camera rig differs from M1")
            if listener.get("view_id") != rig.get("view_id"):
                errors.append("retained listener view differs from M1")
            if (
                _maximum_vector_error(
                    listener.get("position_m"),
                    world_from_listener["translation_m"],
                    length=3,
                )
                > tolerance
            ):
                errors.append("retained listener position differs from M1")
            if (
                _maximum_vector_error(
                    listener.get("orientation_wxyz"), expected_wxyz, length=4
                )
                > tolerance
            ):
                errors.append("retained listener orientation differs from M1")
    else:
        errors.append("retained M1 listener authority is malformed")

    m1_sources = {
        item.get("source_id"): item
        for item in m1.get("sources", [])
        if isinstance(item, Mapping)
    }
    for source_id in canonical_ids:
        request_source = source_by_id.get(source_id)
        identity_source = identity_by_id.get(source_id)
        m1_source = m1_sources.get(source_id)
        if not all(
            isinstance(item, Mapping)
            for item in (request_source, identity_source, m1_source)
        ):
            errors.append(f"{source_id}: retained endpoint authority is incomplete")
            continue
        assert isinstance(request_source, Mapping)
        assert isinstance(identity_source, Mapping)
        assert isinstance(m1_source, Mapping)
        m1_position = m1_source.get("world_from_source", {}).get("translation_m")
        if (
            _maximum_vector_error(
                request_source.get("position_m"), m1_position, length=3
            )
            > tolerance
        ):
            errors.append(f"{source_id}: request position differs from M1")
        if (
            _maximum_vector_error(
                identity_source.get("position_m"), m1_position, length=3
            )
            > tolerance
        ):
            errors.append(f"{source_id}: identity position differs from M1")

    m3_listener = m3.get("listener")
    m3_source = m3.get("source")
    simulation = request.get("simulation")
    m3_simulation = m3.get("simulation")
    if isinstance(m3_listener, Mapping) and isinstance(listener, Mapping):
        if m3_listener.get("id") != listener.get("listener_id"):
            errors.append("M3 listener ID differs from retained M4 listener")
        if (
            _maximum_vector_error(
                m3_listener.get("position_m"), listener.get("position_m"), length=3
            )
            > tolerance
        ):
            errors.append("M3 listener position differs from retained M4 listener")
    if isinstance(m3_source, Mapping):
        m3_source_id = m3_source.get("id")
        bound_source = source_by_id.get(m3_source_id)
        if not isinstance(bound_source, Mapping):
            errors.append("M3 source ID is absent from retained M4 sources")
        elif (
            _maximum_vector_error(
                m3_source.get("position_m"), bound_source.get("position_m"), length=3
            )
            > tolerance
        ):
            errors.append("M3 source position differs from retained M4 source")
    if isinstance(simulation, Mapping) and isinstance(m3_simulation, Mapping):
        for field in ("sample_rate_hz", "max_ir_seconds", "speed_of_sound_m_s"):
            if simulation.get(field) != m3_simulation.get(field):
                errors.append(f"M3/M4 simulation field {field} differs")

    evidence_identity = evidence.get("identity")
    if not isinstance(evidence_identity, Mapping):
        errors.append("evidence identity section is missing")
    else:
        if evidence_identity.get("listener_id") != listener.get("listener_id"):
            errors.append("evidence listener ID differs from retained request")
        if evidence_identity.get("canonical_source_ids") != list(canonical_ids):
            errors.append("evidence source order differs from retained request")
        if evidence_identity.get("source_count") != len(canonical_ids):
            errors.append("evidence source count differs from retained request")
        declared_identities = evidence_identity.get("source_identities")
        if not isinstance(declared_identities, Mapping):
            errors.append("evidence source identities are missing")
        else:
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
            for source_id in canonical_ids:
                retained = identity_by_id.get(source_id)
                expected = (
                    {
                        field: copy.deepcopy(retained.get(field))
                        for field in retained_fields
                    }
                    if isinstance(retained, Mapping)
                    else None
                )
                if declared_identities.get(source_id) != expected:
                    errors.append(
                        f"{source_id}: evidence identity differs from retained manifest"
                    )

    pairs = evidence.get("pairs")
    if isinstance(pairs, Mapping):
        for source_id in canonical_ids:
            pair = pairs.get(source_id)
            retained = identity_by_id.get(source_id)
            if not isinstance(pair, Mapping) or not isinstance(retained, Mapping):
                errors.append(f"{source_id}: pair identity binding is missing")
                continue
            expected_pair_identity = {
                "listener_id": listener.get("listener_id"),
                "source_id": source_id,
                "actor_id": retained.get("actor_id"),
                "event_id": retained.get("event_id"),
                "anchor_id": retained.get("anchor_id"),
                "semantic_anchor_id": retained.get("semantic_anchor_id"),
                "dry_audio_id": retained.get("dry_audio_id"),
            }
            for field, expected in expected_pair_identity.items():
                if pair.get(field) != expected:
                    errors.append(
                        f"{source_id}: pair {field} differs from retained authority"
                    )

    execution = evidence.get("execution")
    if isinstance(execution, Mapping):
        if execution.get("requested_registration_orders") != request.get(
            "registration_orders"
        ):
            errors.append("evidence registration orders differ from retained request")
        if execution.get("canonical_native_source_order") != list(canonical_ids):
            errors.append("native source order differs from retained request")

    manifest_role = inputs.get("acoustic_scene_package_manifest_role")
    package_roles = inputs.get("acoustic_scene_package_file_roles")
    if not isinstance(manifest_role, str) or manifest_role not in paths:
        errors.append("compiled package manifest role is missing")
        return authority, errors
    if not isinstance(package_roles, Mapping):
        errors.append("compiled package file-role map is missing")
        return authority, errors
    manifest_path = paths[manifest_role]
    package_root = manifest_path.parent.resolve()
    mapped_relatives: set[str] = set()
    for relative_text, role in package_roles.items():
        if (
            not isinstance(relative_text, str)
            or not isinstance(role, str)
            or role not in paths
        ):
            errors.append("compiled package file-role map is malformed")
            continue
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"compiled package relative path escapes: {relative_text}")
            continue
        expected_path = (package_root / relative).resolve()
        if paths[role] != expected_path:
            errors.append(f"compiled package role/path differs: {relative_text}")
        mapped_relatives.add(relative.as_posix())
    actual_relatives = {
        item.relative_to(package_root).as_posix()
        for item in package_root.rglob("*")
        if item.is_file()
    }
    if mapped_relatives != actual_relatives:
        errors.append(
            "compiled package file-role map does not cover exactly the package"
        )
    manifest_relative = manifest_path.relative_to(package_root).as_posix()
    if package_roles.get(manifest_relative) != manifest_role:
        errors.append(
            "compiled package manifest role is not bound in the file-role map"
        )
    try:
        scene = load_compiled_acoustic_scene(manifest_path)
    except (OSError, ValueError, RuntimeContractError) as exc:
        errors.append(f"compiled package failed independent loading: {exc}")
    else:
        authority["scene"] = scene
        if inputs.get("package_id") != scene.package_id:
            errors.append("evidence package ID differs from retained package")
        if inputs.get("package_content_sha256") != scene.package_content_sha256:
            errors.append("evidence package hash differs from retained package")
        source_room = scene.manifest.get("source_room")
        if isinstance(source_room, Mapping) and source_room.get("room_id") != m1.get(
            "room_id"
        ):
            errors.append("compiled package room differs from retained M1 request")
        upload_report = evidence.get("runtime", {}).get("foa_upload_report")
        try:
            _verify_upload_report(scene, upload_report)
        except (TypeError, ValueError, RuntimeContractError) as exc:
            errors.append(
                f"runtime upload receipt differs from retained package: {exc}"
            )
    return authority, errors


def _runtime_configuration_readback_errors(
    authority: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[str]:
    """Rebuild native configuration readback from the retained M4 request."""

    request = authority.get("request")
    if not isinstance(request, Mapping):
        return ["retained M4 request is unavailable for configuration replay"]
    simulation = request.get("simulation")
    if not isinstance(simulation, Mapping):
        return ["retained M4 simulation is unavailable for configuration replay"]
    fields = (
        "frequency_bands",
        "direct_sh_order",
        "indirect_sh_order",
        "direct_ray_count",
        "indirect_ray_count",
        "indirect_ray_depth",
        "source_ray_count",
        "source_ray_depth",
        "max_diffraction_order",
        "thread_count",
        "sample_rate_hz",
        "max_ir_seconds",
        "unit_scale",
        "global_volume",
        "direct",
        "indirect",
        "diffraction",
        "transmission",
        "mesh_simplification",
        "temporal_coherence",
    )
    if any(field not in simulation for field in fields):
        return ["retained M4 simulation is incomplete for configuration replay"]
    expected = {field: simulation[field] for field in fields}
    runtime = evidence.get("runtime")
    if not isinstance(runtime, Mapping):
        return ["runtime receipt is missing for configuration replay"]
    errors: list[str] = []
    for layout in ("foa", "binaural"):
        if runtime.get(f"{layout}_configuration_readback") != expected:
            errors.append(
                f"{layout} native configuration readback differs from retained M4 request"
            )
    return errors


def _load_array(paths: Mapping[str, Path], role: Any, *, channels: int) -> np.ndarray:
    if not isinstance(role, str) or role not in paths:
        raise M4EvidenceError(f"unknown array artifact role {role!r}")
    try:
        value = np.load(paths[role], allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise M4EvidenceError(f"cannot load array role {role}: {exc}") from exc
    if value.dtype != np.dtype("<f4") or value.ndim != 2 or value.shape[0] != channels:
        raise M4EvidenceError(
            f"array role {role} must be little-endian float32 [{channels},N]"
        )
    if value.shape[1] < 2 or not np.all(np.isfinite(value)) or not np.any(value != 0.0):
        raise M4EvidenceError(
            f"array role {role} must be finite, nonempty and non-silent"
        )
    return np.ascontiguousarray(value)


def _load_wav(paths: Mapping[str, Path], audio_role: Any, sidecar_role: Any):
    if not isinstance(audio_role, str) or audio_role not in paths:
        raise M4EvidenceError(f"unknown WAVE role {audio_role!r}")
    if not isinstance(sidecar_role, str) or sidecar_role not in paths:
        raise M4EvidenceError(f"unknown WAVE sidecar role {sidecar_role!r}")
    return read_float32_wav(
        paths[audio_role], sidecar_path=paths[sidecar_role], verify_sidecar=True
    )


def _wav_metadata_errors(
    value: Any, expected: Mapping[str, Any], *, owner: str
) -> list[str]:
    """Compare authenticated WAVE metadata with its complete semantic contract."""

    sidecar = getattr(value, "sidecar", None)
    if not isinstance(sidecar, Mapping):
        return [f"{owner}: authenticated sidecar is missing"]
    metadata = sidecar.get("metadata")
    if not isinstance(metadata, Mapping):
        return [f"{owner}: sidecar metadata is missing"]
    if dict(metadata) != dict(expected):
        return [f"{owner}: sidecar semantic metadata differs from evidence contract"]
    return []


def _dry_recipe_errors(
    value: Any,
    *,
    source_id: str,
    identity: Mapping[str, Any],
) -> list[str]:
    """Regenerate a retained deterministic dry recipe and compare exact float32."""

    signal = identity.get("deterministic_signal")
    if not isinstance(signal, Mapping):
        return [f"{source_id}: deterministic dry recipe is missing"]
    try:
        expected = generate_sine_wave(
            value.sample_rate_hz,
            int(signal["duration_samples"]),
            float(signal["frequency_hz"]),
            amplitude=float(signal["amplitude"]),
            phase_radians=float(signal["phase"]),
        ).astype(np.float32)
    except (KeyError, TypeError, ValueError, AudioContractError) as exc:
        return [f"{source_id}: deterministic dry recipe is invalid: {exc}"]
    errors: list[str] = []
    if value.channel_count != 1 or not np.array_equal(value.samples[0], expected):
        errors.append(f"{source_id}: dry WAVE differs from retained recipe")
    errors.extend(
        _wav_metadata_errors(
            value,
            {
                "audio_role": "deterministic_canary_dry",
                "source_id": source_id,
                "actor_id": identity.get("actor_id"),
                "event_id": identity.get("event_id"),
                "anchor_id": identity.get("anchor_id"),
                "dry_audio_id": identity.get("dry_audio_id"),
                "signal": dict(signal),
                "processing": "none",
            },
            owner=f"{source_id} dry WAVE",
        )
    )
    return errors


def _portable_binaural_metadata(
    preflight: Mapping[str, Any],
    *,
    hrtf_role: str,
    license_role: str,
) -> dict[str, Any]:
    """Replace machine-local preflight paths with confined artifact roles."""

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


def _binaural_lock_errors(
    paths: Mapping[str, Path],
    evidence: Mapping[str, Any],
    *,
    sample_rate_hz: int | None,
) -> list[str]:
    """Rebuild HRTF preflight exclusively from confined bytes and the runtime lock."""

    errors: list[str] = []
    inputs = evidence.get("inputs")
    if not isinstance(inputs, Mapping):
        return ["evidence inputs are missing"]
    lock_role = inputs.get("runtime_lock_role")
    hrtf_role = inputs.get("hrtf_role")
    license_role = inputs.get("hrtf_license_role")
    for owner, role in (
        ("runtime lock", lock_role),
        ("HRTF", hrtf_role),
        ("HRTF license", license_role),
    ):
        if not isinstance(role, str) or role not in paths:
            errors.append(f"{owner} artifact role is missing")
    if errors:
        return errors
    assert isinstance(lock_role, str)
    assert isinstance(hrtf_role, str)
    assert isinstance(license_role, str)
    try:
        lock = load_json(paths[lock_role])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"runtime lock is invalid JSON: {exc}"]
    if lock.get("schema") != "avengine_m4_runtime_lock_v1":
        errors.append("runtime lock schema is not avengine_m4_runtime_lock_v1")
    hrtf_lock = lock.get("hrtf")
    output_lock = lock.get("output_contracts")
    observed_binaries = evidence.get("runtime", {}).get("native_binaries")
    if not isinstance(hrtf_lock, Mapping):
        errors.append("runtime lock HRTF pin is missing")
    if not isinstance(output_lock, Mapping):
        errors.append("runtime lock output contracts are missing")
    if not isinstance(observed_binaries, Mapping):
        errors.append("observed native binary receipt is missing")
    if errors:
        return errors
    assert isinstance(hrtf_lock, Mapping)
    assert isinstance(output_lock, Mapping)
    assert isinstance(observed_binaries, Mapping)
    observed_rlr = observed_binaries.get("rlr_audio_propagation")
    if not isinstance(observed_rlr, Mapping):
        return ["observed RLR binary receipt is missing"]
    rlr_sha256 = observed_rlr.get("sha256")
    if sample_rate_hz is None:
        return ["render sample rate cannot be derived from retained WAVE bytes"]
    if output_lock.get("render_sample_rate_hz") != sample_rate_hz:
        errors.append("retained WAVE sample rate differs from the runtime lock")
    if output_lock.get("foa") != rlr_foa_contract()["format_id"]:
        errors.append("runtime lock FOA format differs from the frozen contract")
    if output_lock.get("binaural") != "rlr_binaural_lr_v1":
        errors.append("runtime lock binaural layout differs from the frozen contract")
    if output_lock.get("avengine_resampling_performed") is not False:
        errors.append("runtime lock must forbid AVEngine resampling")
    if output_lock.get("normalization") != "none":
        errors.append("runtime lock must forbid output normalization")
    if output_lock.get("limiter") != "none":
        errors.append("runtime lock must forbid output limiting")

    try:
        preflight = build_rlr_native_binaural_metadata(
            sample_rate_hz,
            hrtf_path=paths[hrtf_role],
            expected_hrtf_sha256=hrtf_lock.get("sha256"),
            hrtf_sample_rate_hz=hrtf_lock.get("sample_rate_hz"),
            license_id=hrtf_lock.get("license_id"),
            citation=hrtf_lock.get("citation"),
            license_text_path=paths[license_role],
            expected_license_sha256=hrtf_lock.get("license_text_sha256"),
            asset_id=hrtf_lock.get("asset_id", "explicit_sofa_hrtf"),
            sample_rate_policy=hrtf_lock.get("sample_rate_policy", "strict_match"),
            rlr_binary_sha256=rlr_sha256,
        )
    except BinauralContractError as exc:
        return [f"HRTF preflight could not be reconstructed: {exc}"]
    if preflight.get("status") != "pass":
        errors.append(
            "reconstructed HRTF preflight did not pass: "
            + str(preflight.get("reason", preflight.get("reason_code", "unknown")))
        )
        return errors

    expected = _portable_binaural_metadata(
        preflight,
        hrtf_role=hrtf_role,
        license_role=license_role,
    )
    expected["native_cardinal_validation"] = "pass"
    audio_contracts = evidence.get("audio_contracts")
    if not isinstance(audio_contracts, Mapping):
        return ["audio_contracts are missing"]
    actual = audio_contracts.get("binaural")
    if not isinstance(actual, Mapping):
        return ["audio_contracts.binaural is missing"]
    actual_without_report = copy.deepcopy(dict(actual))
    report = actual_without_report.pop("native_cardinal_report", None)
    if not isinstance(report, Mapping):
        errors.append("binaural native cardinal report is missing")
    if actual_without_report != expected:
        errors.append(
            "audio_contracts.binaural differs from HRTF/license/runtime-lock bytes"
        )
    if audio_contracts.get("avengine_resampling_performed") is not False:
        errors.append("AVEngine resampling must be explicitly recorded as false")
    if audio_contracts.get("implicit_normalization") is not False:
        errors.append("implicit normalization must be explicitly recorded as false")
    if audio_contracts.get("limiter") is not False:
        errors.append("limiter must be explicitly recorded as false")
    if "implicit_resampling" in audio_contracts:
        errors.append("ambiguous implicit_resampling field is forbidden")
    if audio_contracts.get("native_rate_adaptation") != expected.get(
        "sample_rate_binding"
    ):
        errors.append("native rate-adaptation receipt differs from HRTF preflight")
    return errors


def _current_binaural_preflight_errors(
    paths: Mapping[str, Path],
    evidence: Mapping[str, Any],
    *,
    sample_rate_hz: int | None,
) -> list[str]:
    """Rebuild v2 HRTF preflight from this receipt's verified input bytes.

    The current-installed route has no historical HRTF or native-binary pin.
    It authenticates only the HRTF/license copies retained in this fresh bundle,
    then requires strict render-rate matching without native adaptation.
    """

    errors: list[str] = []
    inputs = evidence.get("inputs")
    audio_contracts = evidence.get("audio_contracts")
    if not isinstance(inputs, Mapping) or not isinstance(audio_contracts, Mapping):
        return ["current-installed HRTF inputs or audio contracts are missing"]
    if "runtime_lock_role" in inputs:
        errors.append("current-installed evidence must not retain a historical runtime lock")
    hrtf_role = inputs.get("hrtf_role")
    license_role = inputs.get("hrtf_license_role")
    for owner, role in (("HRTF", hrtf_role), ("HRTF license", license_role)):
        if not isinstance(role, str) or role not in paths:
            errors.append(f"{owner} artifact role is missing")
    if sample_rate_hz is None:
        errors.append("render sample rate cannot be derived from retained WAVE bytes")
    binaural = audio_contracts.get("binaural")
    if not isinstance(binaural, Mapping):
        errors.append("audio_contracts.binaural is missing")
    if errors:
        return errors
    assert isinstance(hrtf_role, str) and isinstance(license_role, str)
    assert sample_rate_hz is not None and isinstance(binaural, Mapping)

    hrtf = binaural.get("hrtf")
    rights = binaural.get("rights")
    binding = binaural.get("sample_rate_binding")
    if not isinstance(hrtf, Mapping) or not isinstance(rights, Mapping):
        return ["current-installed HRTF or license metadata is missing"]
    if not isinstance(binding, Mapping):
        return ["current-installed HRTF rate binding is missing"]
    if binding.get("policy") != "strict_match":
        errors.append("current-installed HRTF rate policy must be strict_match")
    if binding.get("native_rate_adaptation") != "not_required":
        errors.append("current-installed HRTF receipt must not claim native adaptation")
    if "rlr_binary_sha256" in binding:
        errors.append("current-installed HRTF receipt must not retain a binary hash")
    try:
        current_hrtf_sha256 = sha256_file(paths[hrtf_role])
        current_license_sha256 = sha256_file(paths[license_role])
    except OSError as exc:
        return [f"current-installed HRTF/license bytes are unreadable: {exc}"]
    try:
        preflight = build_rlr_native_binaural_metadata(
            sample_rate_hz,
            hrtf_path=paths[hrtf_role],
            expected_hrtf_sha256=current_hrtf_sha256,
            hrtf_sample_rate_hz=hrtf.get("sample_rate_hz"),
            license_id=rights.get("license_id"),
            citation=rights.get("citation"),
            license_text_path=paths[license_role],
            expected_license_sha256=current_license_sha256,
            asset_id=hrtf.get("asset_id", "current_installed_explicit_sofa_hrtf"),
            sample_rate_policy="strict_match",
        )
    except BinauralContractError as exc:
        return [f"current-installed HRTF preflight could not be reconstructed: {exc}"]
    if preflight.get("status") != "pass":
        return [
            "reconstructed current-installed HRTF preflight did not pass: "
            + str(preflight.get("reason", preflight.get("reason_code", "unknown")))
        ]
    expected = _portable_binaural_metadata(
        preflight,
        hrtf_role=hrtf_role,
        license_role=license_role,
    )
    expected["native_cardinal_validation"] = "pass"
    actual_without_report = copy.deepcopy(dict(binaural))
    report = actual_without_report.pop("native_cardinal_report", None)
    if not isinstance(report, Mapping):
        errors.append("binaural native cardinal report is missing")
    if actual_without_report != expected:
        errors.append(
            "audio_contracts.binaural differs from current HRTF/license artifact bytes"
        )
    if audio_contracts.get("avengine_resampling_performed") is not False:
        errors.append("AVEngine resampling must be explicitly recorded as false")
    if audio_contracts.get("implicit_normalization") is not False:
        errors.append("implicit normalization must be explicitly recorded as false")
    if audio_contracts.get("limiter") is not False:
        errors.append("limiter must be explicitly recorded as false")
    if "implicit_resampling" in audio_contracts:
        errors.append("ambiguous implicit_resampling field is forbidden")
    if audio_contracts.get("native_rate_adaptation") != expected.get(
        "sample_rate_binding"
    ):
        errors.append("native rate-adaptation receipt differs from reconstructed preflight")
    return errors


def _runtime_lock_errors(
    paths: Mapping[str, Path], evidence: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    role = evidence.get("inputs", {}).get("runtime_lock_role")
    if not isinstance(role, str) or role not in paths:
        return ["runtime lock artifact role is missing"]
    try:
        lock = load_json(paths[role])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"runtime lock is invalid JSON: {exc}"]
    if lock.get("schema") != "avengine_m4_runtime_lock_v1":
        errors.append("runtime lock schema is not avengine_m4_runtime_lock_v1")
    expected = lock.get("native_binaries")
    observed = evidence.get("runtime", {}).get("native_binaries")
    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
        return ["runtime binary pins or observed receipt are missing"]
    for name in ("habitat_sim_bindings", "rlr_audio_propagation"):
        expected_record = expected.get(name)
        observed_record = observed.get(name)
        if not isinstance(expected_record, Mapping) or not isinstance(
            observed_record, Mapping
        ):
            errors.append(f"runtime binary record {name} is missing")
            continue
        for key in ("byte_size", "sha256"):
            if observed_record.get(key) != expected_record.get(key):
                errors.append(f"runtime binary {name}.{key} differs from lock")
    return errors


def _direct_arrival_errors(
    order_a: Mapping[str, np.ndarray],
    pairs: Mapping[str, Any],
    source_ids: list[str],
    *,
    sample_rate_hz: int,
    authority: Mapping[str, Any],
) -> list[str]:
    """Measure arrivals while deriving every expectation from retained request bytes."""

    errors: list[str] = []
    request = authority.get("request")
    retained_sources = authority.get("sources")
    retained_listener = authority.get("listener")
    simulation = request.get("simulation") if isinstance(request, Mapping) else None
    thresholds = request.get("thresholds") if isinstance(request, Mapping) else None
    if not all(
        isinstance(value, Mapping)
        for value in (retained_sources, retained_listener, simulation, thresholds)
    ):
        return ["retained geometry/simulation authority is missing"]
    assert isinstance(retained_sources, Mapping)
    assert isinstance(retained_listener, Mapping)
    assert isinstance(simulation, Mapping)
    assert isinstance(thresholds, Mapping)
    for source_id in source_ids:
        try:
            retained_source = retained_sources[source_id]
            source_position = retained_source["position_m"]
            listener_position = retained_listener["position_m"]
            speed = float(simulation["speed_of_sound_m_s"])
            retained_rate = int(simulation["sample_rate_hz"])
            if retained_rate != sample_rate_hz:
                raise M4EvidenceError(
                    "retained request sample rate differs from WAVE bytes"
                )
            distance = math.dist(source_position, listener_position)
            expected_sample = distance / speed * retained_rate
            maximum = float(thresholds["maximum_direct_arrival_error_samples"])
            metrics = analyze_ir(order_a[source_id], sample_rate_hz)
            declared = pairs[source_id]["direct_arrival"]
            detected = metrics.direct_arrival_sample
            absolute_error = abs(detected - expected_sample)
            if absolute_error > maximum:
                errors.append(f"{source_id}: geometric direct arrival mismatch")
            expected_declaration = {
                "distance_m": distance,
                "speed_of_sound_m_s": speed,
                "expected_sample": expected_sample,
                "detected_sample": detected,
                "absolute_error_samples": absolute_error,
                "maximum_absolute_error_samples": maximum,
            }
            for field, expected in expected_declaration.items():
                observed = declared.get(field)
                if isinstance(expected, float):
                    try:
                        matches = math.isclose(
                            float(observed),
                            expected,
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                    except (TypeError, ValueError):
                        matches = False
                else:
                    matches = observed == expected
                if not matches:
                    errors.append(
                        f"{source_id}: declared {field} differs from retained geometry"
                    )
        except (
            KeyError,
            TypeError,
            ValueError,
            M4EvidenceError,
            AcousticMetricError,
        ) as exc:
            errors.append(f"{source_id}: {exc}")
    return errors


def _performance_replay_errors(
    authority: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[str]:
    """Recompute M4 performance summaries and comparison from retained runs."""

    request = authority.get("request")
    source_ids = authority.get("canonical_source_ids")
    thresholds = request.get("thresholds") if isinstance(request, Mapping) else None
    if not isinstance(source_ids, tuple) or not isinstance(thresholds, Mapping):
        return ["retained M4 performance authority is incomplete"]
    try:
        expected_repeat_count = int(thresholds["performance_repeat_count"])
    except (KeyError, TypeError, ValueError):
        return ["retained M4 performance repeat count is invalid"]
    if (
        isinstance(thresholds.get("performance_repeat_count"), bool)
        or expected_repeat_count < 1
    ):
        return ["retained M4 performance repeat count is invalid"]
    performance = evidence.get("performance")
    if not isinstance(performance, Mapping):
        return ["performance receipt is missing"]
    errors: list[str] = []
    medians: dict[str, float] = {}
    for name, expected_pair_count in (
        ("one_source", 1),
        ("multi_source", len(source_ids)),
    ):
        condition = performance.get(name)
        if not isinstance(condition, Mapping):
            errors.append(f"performance.{name} is missing")
            continue
        runs = condition.get("runs")
        if not isinstance(runs, list):
            errors.append(f"performance.{name}.runs is missing")
            continue
        if condition.get("source_count") != expected_pair_count:
            errors.append(f"performance.{name}.source_count differs from retained inputs")
        if condition.get("pair_count") != expected_pair_count:
            errors.append(f"performance.{name}.pair_count differs from retained inputs")
        if condition.get("repeat_count") != expected_repeat_count:
            errors.append(f"performance.{name}.repeat_count differs from retained request")
        if len(runs) != expected_repeat_count:
            errors.append(f"performance.{name}.runs count differs from retained request")
        wall: list[float] = []
        cpu: list[float] = []
        peak_after: list[int] = []
        payload: list[int] = []
        for repeat_index, run in enumerate(runs):
            if not isinstance(run, Mapping):
                errors.append(f"performance.{name}.runs[{repeat_index}] is malformed")
                continue
            if run.get("repeat_index") != repeat_index:
                errors.append(
                    f"performance.{name}.runs[{repeat_index}].repeat_index differs"
                )
            if run.get("pair_count") != expected_pair_count:
                errors.append(
                    f"performance.{name}.runs[{repeat_index}].pair_count differs"
                )
            try:
                current_wall = float(run["wall_seconds"])
                current_cpu = float(run["process_cpu_seconds"])
                current_before = run["peak_rss_before_bytes"]
                current_after = run["peak_rss_after_bytes"]
                current_payload = run["ir_payload_bytes"]
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"performance.{name}.runs[{repeat_index}]: {exc}")
                continue
            if (
                not math.isfinite(current_wall)
                or current_wall <= 0.0
                or not math.isfinite(current_cpu)
                or current_cpu < 0.0
                or isinstance(current_before, bool)
                or not isinstance(current_before, int)
                or current_before < 0
                or isinstance(current_after, bool)
                or not isinstance(current_after, int)
                or current_after < 0
                or isinstance(current_payload, bool)
                or not isinstance(current_payload, int)
                or current_payload < 1
            ):
                errors.append(
                    f"performance.{name}.runs[{repeat_index}] has invalid timing data"
                )
                continue
            wall.append(current_wall)
            cpu.append(current_cpu)
            peak_after.append(current_after)
            payload.append(current_payload)
        if len(wall) != expected_repeat_count:
            continue
        sorted_wall = sorted(wall)
        expected_summary = {
            "median_wall_seconds": statistics.median(sorted_wall),
            "p95_wall_seconds": sorted_wall[
                min(len(sorted_wall) - 1, math.ceil(0.95 * len(sorted_wall)) - 1)
            ],
            "median_process_cpu_seconds": statistics.median(sorted(cpu)),
            "maximum_peak_rss_bytes": max(peak_after),
            "median_ir_payload_bytes": int(statistics.median(payload)),
        }
        for field, expected in expected_summary.items():
            if condition.get(field) != expected:
                errors.append(
                    f"performance.{name}.{field} differs from retained runs"
                )
        medians[name] = expected_summary["median_wall_seconds"]
    if set(medians) == {"one_source", "multi_source"}:
        one = medians["one_source"]
        multi = medians["multi_source"]
        expected_comparison = {
            "multi_to_one_median_wall_ratio": multi / one,
            "multi_pair_throughput_pairs_per_second": len(source_ids) / multi,
            "hard_speed_gate": None,
            "interpretation": "measurement_only_platform_dependent",
        }
        if performance.get("comparison") != expected_comparison:
            errors.append("performance.comparison differs from retained runs")
    return errors


def _lifecycle_replay_errors(
    authority: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
) -> list[str]:
    """Rebuild lifecycle movement, native receipt, and upload claims from inputs."""

    request = authority.get("request")
    sources = authority.get("sources")
    listener = authority.get("listener")
    source_ids = authority.get("canonical_source_ids")
    scene = authority.get("scene")
    if not all(
        isinstance(value, Mapping) for value in (request, sources, listener)
    ) or not isinstance(source_ids, tuple) or not source_ids:
        return ["retained M4 lifecycle authority is incomplete"]
    if scene is None:
        return ["retained M4 package is unavailable for lifecycle upload replay"]
    assert isinstance(request, Mapping)
    assert isinstance(sources, Mapping)
    assert isinstance(listener, Mapping)
    thresholds = request.get("thresholds")
    if not isinstance(thresholds, Mapping):
        return ["retained M4 lifecycle geometry threshold is missing"]
    try:
        geometry_tolerance = float(thresholds["maximum_anchor_transform_error"])
    except (KeyError, TypeError, ValueError):
        return ["retained M4 lifecycle geometry threshold is invalid"]
    if (
        isinstance(thresholds.get("maximum_anchor_transform_error"), bool)
        or not math.isfinite(geometry_tolerance)
        or geometry_tolerance < 0.0
    ):
        return ["retained M4 lifecycle geometry threshold is invalid"]
    moved_source_id = source_ids[0]
    source = sources.get(moved_source_id)
    if not isinstance(source, Mapping):
        return ["retained M4 moved source is missing"]
    try:
        original = np.asarray(source["position_m"], dtype=np.float64)
        listener_position = np.asarray(listener["position_m"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"retained M4 lifecycle geometry is invalid: {exc}"]
    if (
        original.shape != (3,)
        or listener_position.shape != (3,)
        or not np.all(np.isfinite(original))
        or not np.all(np.isfinite(listener_position))
    ):
        return ["retained M4 lifecycle geometry is invalid"]
    direction = original - listener_position
    source_distance = float(np.linalg.norm(direction))
    if not math.isfinite(source_distance) or source_distance <= 0.0:
        return ["retained M4 lifecycle source is co-located with listener"]
    expected_updated = (
        original
        + direction / source_distance * float(LIFECYCLE_MOVED_DISTANCE_M)
    )
    errors: list[str] = []
    if lifecycle.get("moved_source_id") != moved_source_id:
        errors.append("lifecycle moved source differs from canonical first source")
    try:
        moved_distance = float(lifecycle.get("moved_distance_m"))
    except (TypeError, ValueError):
        moved_distance = math.nan
    if not math.isclose(
        moved_distance,
        float(LIFECYCLE_MOVED_DISTANCE_M),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        errors.append("lifecycle moved distance differs from the executed policy")
    if _maximum_vector_error(
        lifecycle.get("original_position_m"), original, length=3
    ) > geometry_tolerance:
        errors.append("lifecycle original position differs from retained source")
    if _maximum_vector_error(
        lifecycle.get("updated_position_m"), expected_updated, length=3
    ) > geometry_tolerance:
        errors.append("lifecycle updated position differs from retained geometry")
    receipts = lifecycle.get("source_registration_receipts_after_update")
    if not isinstance(receipts, list) or len(receipts) != len(source_ids):
        errors.append("lifecycle updated source receipt count differs from retained inputs")
    else:
        for index, source_id in enumerate(source_ids):
            observed = receipts[index]
            expected_source = sources.get(source_id)
            if not isinstance(observed, Mapping) or not isinstance(
                expected_source, Mapping
            ):
                errors.append(f"lifecycle updated receipt {source_id} is malformed")
                continue
            expected_position = (
                expected_updated
                if source_id == moved_source_id
                else expected_source.get("position_m")
            )
            if (
                observed.get("source_id") != source_id
                or observed.get("canonical_native_index") != index
                or observed.get("native_realized") is not True
            ):
                errors.append(
                    f"lifecycle updated receipt {source_id} identity/index differs"
                )
            try:
                expected_native_position = np.asarray(
                    expected_position, dtype=np.float32
                ).astype(np.float64)
                expected_native_radius = float(
                    np.float32(float(expected_source["radius_m"]))
                )
                observed_radius = float(observed.get("radius_m"))
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append(
                    f"lifecycle updated receipt {source_id} is not a native readback"
                )
                continue
            if _maximum_vector_error(
                observed.get("position_m"), expected_native_position, length=3
            ) > 0.0:
                errors.append(
                    f"lifecycle updated receipt {source_id} position differs"
                )
            if (
                not math.isfinite(observed_radius)
                or observed_radius != expected_native_radius
            ):
                errors.append(
                    f"lifecycle updated receipt {source_id} radius differs"
                )
    try:
        _verify_upload_report(scene, lifecycle.get("upload_report"))
    except (TypeError, ValueError, RuntimeContractError) as exc:
        errors.append(f"lifecycle upload receipt differs from retained package: {exc}")
    return errors


def _verify_historical_m4_canary_evidence(
    evidence_path: str | Path,
    *,
    current_installed: bool = False,
    evidence: Mapping[str, Any] | None = None,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Recompute shared M4 claims from confined v1 or v2 receipt bytes."""

    path = Path(evidence_path).resolve()
    checks: list[dict[str, Any]] = []
    if evidence is None:
        try:
            evidence = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return "fail", (
                _derived_check(
                    "evidence_json",
                    False,
                    str(exc),
                    "Evidence is missing or invalid JSON",
                ),
            )

    contract_errors = (
        validate_current_installed_multi_source_canary_evidence(
            evidence, evidence_path=path
        )
        if current_installed
        else validate_multi_source_canary_evidence(evidence, evidence_path=path)
    )
    expected_schema = (
        CURRENT_INSTALLED_EVIDENCE_SCHEMA if current_installed else EVIDENCE_SCHEMA
    )
    checks.append(
        _derived_check(
            "evidence_contract",
            not contract_errors and evidence.get("schema") == expected_schema,
            contract_errors,
            "Evidence violates the M4 schema or semantic contract",
        )
    )
    declared_hash = evidence.get("evidence_content_sha256")
    try:
        actual_hash = canonical_json_sha256(
            {
                key: value
                for key, value in evidence.items()
                if key != "evidence_content_sha256"
            }
        )
    except (TypeError, ValueError):
        actual_hash = None
    checks.append(
        _derived_check(
            "evidence_content_hash",
            actual_hash == declared_hash,
            {"actual": actual_hash, "declared": declared_hash},
            "Evidence canonical content hash changed",
        )
    )

    paths, artifact_errors = _confined_artifacts(path, evidence)
    checks.append(
        _derived_check(
            "artifact_records",
            not artifact_errors,
            artifact_errors,
            "An evidence artifact is missing, changed, aliased, or escaping",
        )
    )
    if artifact_errors:
        return "fail", tuple(checks)

    authority, authority_errors = _input_authority(paths, evidence)
    checks.append(
        _derived_check(
            "retained_input_authority",
            not authority_errors,
            authority_errors,
            "Retained M4/M1/M3/identity/package inputs do not bind the evidence",
        )
    )

    configuration_readback_errors = _runtime_configuration_readback_errors(
        authority, evidence
    )
    checks.append(
        _derived_check(
            "runtime_configuration_readback",
            not configuration_readback_errors,
            configuration_readback_errors,
            "Native configuration readback differs from retained M4 request",
        )
    )

    pairs = evidence.get("pairs")
    source_ids = evidence.get("identity", {}).get("canonical_source_ids")
    pair_errors: list[str] = []
    order_a: dict[str, np.ndarray] = {}
    order_b: dict[str, np.ndarray] = {}
    binaural_rirs: dict[str, np.ndarray] = {}
    dry: dict[str, np.ndarray] = {}
    stored_foa_stems: dict[str, np.ndarray] = {}
    stored_binaural_stems: dict[str, np.ndarray] = {}
    sample_rate: int | None = None
    audio_contracts = evidence.get("audio_contracts")
    binaural_authority = (
        audio_contracts.get("binaural")
        if isinstance(audio_contracts, Mapping)
        else None
    )
    listener_id = evidence.get("identity", {}).get("listener_id")
    retained_request = authority.get("request")
    retained_identities = authority.get("identities")
    retained_simulation = (
        retained_request.get("simulation")
        if isinstance(retained_request, Mapping)
        else None
    )
    if (
        not isinstance(audio_contracts, Mapping)
        or audio_contracts.get("foa") != rlr_foa_contract()
    ):
        pair_errors.append(
            "audio_contracts.foa differs from the frozen raw-FOA contract"
        )
    if (
        not isinstance(pairs, Mapping)
        or not isinstance(source_ids, list)
        or set(pairs) != set(source_ids)
        or len(source_ids) < 2
    ):
        pair_errors.append("pair keys must equal at least two canonical source IDs")
    else:
        for source_id in source_ids:
            record = pairs[source_id]
            try:
                order_a[source_id] = _load_array(
                    paths, record["foa_ir_order_a_role"], channels=4
                )
                order_b[source_id] = _load_array(
                    paths, record["foa_ir_order_b_role"], channels=4
                )
                binaural_rirs[source_id] = _load_array(
                    paths, record["binaural_ir_role"], channels=2
                )
                dry_wav = _load_wav(
                    paths, record["dry_wav_role"], record["dry_sidecar_role"]
                )
                foa_stem = _load_wav(
                    paths,
                    record["foa_stem_wav_role"],
                    record["foa_stem_sidecar_role"],
                )
                binaural_stem = _load_wav(
                    paths,
                    record["binaural_stem_wav_role"],
                    record["binaural_stem_sidecar_role"],
                )
                if dry_wav.channel_count != 1:
                    raise M4EvidenceError("dry WAVE must be mono")
                if foa_stem.channel_count != 4:
                    raise M4EvidenceError("FOA stem WAVE must have four channels")
                if binaural_stem.channel_count != 2:
                    raise M4EvidenceError("binaural stem WAVE must have two channels")
                rates = {
                    dry_wav.sample_rate_hz,
                    foa_stem.sample_rate_hz,
                    binaural_stem.sample_rate_hz,
                }
                if len(rates) != 1:
                    raise M4EvidenceError("source WAVE sample rates differ")
                current_rate = rates.pop()
                if sample_rate is None:
                    sample_rate = current_rate
                elif current_rate != sample_rate:
                    raise M4EvidenceError("source sample rates differ")
                if isinstance(
                    retained_simulation, Mapping
                ) and current_rate != retained_simulation.get("sample_rate_hz"):
                    raise M4EvidenceError(
                        "source WAVE sample rate differs from retained request"
                    )
                dry[source_id] = dry_wav.samples[0].astype(np.float64)
                stored_foa_stems[source_id] = foa_stem.samples
                stored_binaural_stems[source_id] = binaural_stem.samples
                retained_identity = (
                    retained_identities.get(source_id)
                    if isinstance(retained_identities, Mapping)
                    else None
                )
                if not isinstance(retained_identity, Mapping):
                    pair_errors.append(f"{source_id}: retained dry identity is missing")
                else:
                    pair_errors.extend(
                        _dry_recipe_errors(
                            dry_wav,
                            source_id=source_id,
                            identity=retained_identity,
                        )
                    )
                pair_errors.extend(
                    _wav_metadata_errors(
                        foa_stem,
                        {
                            **rlr_foa_wav_metadata(),
                            "audio_role": "per_source_wet_stem",
                            "source_id": source_id,
                            "listener_id": listener_id,
                            "lineage": "dry_linear_convolution_pair_foa_ir",
                            "linear_gain": 1.0,
                            "normalization": "none",
                            "tail_policy": "full_linear_convolution",
                        },
                        owner=f"{source_id} FOA stem",
                    )
                )
                if not isinstance(binaural_authority, Mapping):
                    pair_errors.append("audio_contracts.binaural is missing")
                else:
                    pair_errors.extend(
                        _wav_metadata_errors(
                            binaural_stem,
                            {
                                "spatial_format": rlr_native_binaural_contract(),
                                "hrtf": dict(binaural_authority),
                                "audio_role": "per_source_wet_stem",
                                "source_id": source_id,
                                "listener_id": listener_id,
                                "lineage": "dry_linear_convolution_pair_binaural_ir",
                                "linear_gain": 1.0,
                                "normalization": "none",
                                "tail_policy": "full_linear_convolution",
                            },
                            owner=f"{source_id} binaural stem",
                        )
                    )
            except (KeyError, M4EvidenceError, AudioContractError) as exc:
                pair_errors.append(f"{source_id}: {exc}")
    checks.append(
        _derived_check(
            "pair_artifacts_readable",
            not pair_errors,
            pair_errors,
            "Per-source IR/dry/stem artifacts are not independently readable",
        )
    )

    order_errors: list[str] = []
    if not pair_errors:
        for source_id in source_ids:
            if not np.array_equal(order_a[source_id], order_b[source_id]):
                order_errors.append(
                    f"{source_id}: full RIR differs after request reorder"
                )
    checks.append(
        _derived_check(
            "source_order_invariance",
            not pair_errors and not order_errors,
            order_errors,
            "Mapped full-indirect RIR changed with caller source order",
        )
    )

    reconstruction_errors: list[str] = []
    if not pair_errors:
        try:
            expected_foa_stems, expected_foa_mix = render_stems_and_mix(dry, order_a)
            expected_bin_stems, expected_bin_mix = render_stems_and_mix(
                dry, binaural_rirs
            )
            for source_id in source_ids:
                if not np.array_equal(
                    stored_foa_stems[source_id],
                    np.asarray(expected_foa_stems[source_id], dtype=np.float32),
                ):
                    reconstruction_errors.append(f"{source_id}: FOA stem mismatch")
                if not np.array_equal(
                    stored_binaural_stems[source_id],
                    np.asarray(expected_bin_stems[source_id], dtype=np.float32),
                ):
                    reconstruction_errors.append(f"{source_id}: binaural stem mismatch")
            mixture = evidence.get("mixtures", {})
            stored_foa_mix = _load_wav(
                paths, mixture["foa_wav_role"], mixture["foa_sidecar_role"]
            )
            stored_bin_mix = _load_wav(
                paths,
                mixture["binaural_wav_role"],
                mixture["binaural_sidecar_role"],
            )
            if stored_foa_mix.channel_count != 4:
                reconstruction_errors.append("FOA mixture WAVE must have four channels")
            if stored_bin_mix.channel_count != 2:
                reconstruction_errors.append(
                    "binaural mixture WAVE must have two channels"
                )
            if (
                stored_foa_mix.sample_rate_hz != sample_rate
                or stored_bin_mix.sample_rate_hz != sample_rate
            ):
                reconstruction_errors.append(
                    "mixture WAVE sample rate differs from stems"
                )
            reconstruction_errors.extend(
                _wav_metadata_errors(
                    stored_foa_mix,
                    {
                        **rlr_foa_wav_metadata(),
                        "audio_role": "m4_canary_full_tail_mixture",
                        "source_ids": list(source_ids),
                        "summation_order": list(source_ids),
                        "normalization": "none",
                        "limiter": "none",
                        "timeline_crop": "not_applied_m5_owned",
                    },
                    owner="FOA mixture",
                )
            )
            if not isinstance(binaural_authority, Mapping):
                reconstruction_errors.append("audio_contracts.binaural is missing")
            else:
                reconstruction_errors.extend(
                    _wav_metadata_errors(
                        stored_bin_mix,
                        {
                            "spatial_format": rlr_native_binaural_contract(),
                            "hrtf": dict(binaural_authority),
                            "audio_role": "m4_canary_full_tail_mixture",
                            "source_ids": list(source_ids),
                            "summation_order": list(source_ids),
                            "normalization": "none",
                            "limiter": "none",
                            "timeline_crop": "not_applied_m5_owned",
                        },
                        owner="binaural mixture",
                    )
                )
            if not np.array_equal(
                stored_foa_mix.samples, np.asarray(expected_foa_mix, dtype=np.float32)
            ):
                reconstruction_errors.append(
                    "FOA mixture does not equal canonical stem sum"
                )
            if not np.array_equal(
                stored_bin_mix.samples, np.asarray(expected_bin_mix, dtype=np.float32)
            ):
                reconstruction_errors.append(
                    "binaural mixture does not equal canonical stem sum"
                )
        except (KeyError, M4EvidenceError, AudioContractError) as exc:
            reconstruction_errors.append(str(exc))
    checks.append(
        _derived_check(
            "stem_and_mixture_reconstruction",
            not pair_errors and not reconstruction_errors,
            reconstruction_errors,
            "Stored stems/mixtures do not reconstruct from dry signals and mapped IRs",
        )
    )

    arrival_errors: list[str] = []
    if not pair_errors and sample_rate is not None:
        arrival_errors.extend(
            _direct_arrival_errors(
                order_a,
                pairs,
                source_ids,
                sample_rate_hz=sample_rate,
                authority=authority,
            )
        )
    checks.append(
        _derived_check(
            "direct_arrival_geometry",
            not pair_errors and not arrival_errors,
            arrival_errors,
            "Direct arrivals disagree with source/listener geometry",
        )
    )

    probe_errors: list[str] = []
    try:
        foa_probe = evidence["probes"]["foa"]
        cardinal = {
            direction: _load_array(paths, role, channels=4)
            for direction, role in foa_probe["cardinal_roles"].items()
        }
        foa_report = validate_cardinal_foa(cardinal)
        if foa_probe.get("cardinal_report") != foa_report:
            probe_errors.append("declared FOA cardinal report was not reproduced")
        identity = _load_array(paths, foa_probe["world_identity_role"], channels=4)
        rotated = _load_array(paths, foa_probe["world_rotated_role"], channels=4)
        world_report = validate_world_aligned_foa(identity, rotated)
        if foa_probe.get("world_alignment_report") != world_report:
            probe_errors.append(
                "declared FOA world-alignment report was not reproduced"
            )
        binaural_probe = evidence["probes"]["binaural"]
        binaural_cardinals = {
            direction: _load_array(paths, role, channels=2)
            for direction, role in binaural_probe["cardinal_roles"].items()
        }
        binaural_report = validate_binaural_cardinals(
            binaural_cardinals,
            minimum_ild_db=float(binaural_probe["minimum_ild_db"]),
        )
        if binaural_probe.get("cardinal_report") != binaural_report:
            probe_errors.append("declared binaural cardinal report was not reproduced")
        if not isinstance(binaural_authority, Mapping):
            probe_errors.append("audio_contracts.binaural is missing")
        elif (
            binaural_authority.get("native_cardinal_validation") != "pass"
            or binaural_authority.get("native_cardinal_report") != binaural_report
        ):
            probe_errors.append(
                "binaural authority metadata does not contain the reproduced pass report"
            )
    except (
        KeyError,
        TypeError,
        ValueError,
        M4EvidenceError,
        SpatialContractError,
        BinauralContractError,
    ) as exc:
        probe_errors.append(str(exc))
    checks.append(
        _derived_check(
            "spatial_direction_probes",
            not probe_errors,
            probe_errors,
            "FOA axis/world-frame or binaural direction probe failed",
        )
    )

    if current_installed:
        runtime_errors = _current_runtime_identity_errors(evidence)
        checks.append(
            _derived_check(
                "current_runtime_identity",
                not runtime_errors,
                runtime_errors,
                "Current-installed runtime identity is invalid, mixed, or lock-backed",
            )
        )
        binaural_errors = _current_binaural_preflight_errors(
            paths,
            evidence,
            sample_rate_hz=sample_rate,
        )
        checks.append(
            _derived_check(
                "hrtf_license_and_rate_binding",
                not binaural_errors,
                binaural_errors,
                "HRTF/license preflight does not reconstruct from current receipt bytes",
            )
        )
    else:
        lock_errors = _runtime_lock_errors(paths, evidence)
        checks.append(
            _derived_check(
                "runtime_binary_lock",
                not lock_errors,
                lock_errors,
                "Observed native binaries differ from the immutable M4 runtime lock",
            )
        )
        binaural_lock_errors = _binaural_lock_errors(
            paths,
            evidence,
            sample_rate_hz=sample_rate,
        )
        checks.append(
            _derived_check(
                "hrtf_license_and_rate_binding",
                not binaural_lock_errors,
                binaural_lock_errors,
                "HRTF, license, or native sample-rate binding differs from the lock",
            )
        )


    performance_errors = _performance_replay_errors(authority, evidence)
    checks.append(
        _derived_check(
            "performance_report",
            not performance_errors,
            performance_errors,
            "One-source/multi-source performance evidence does not replay from runs",
        )
    )

    lifecycle = evidence.get("lifecycle", {})
    lifecycle_errors: list[str] = []
    if isinstance(lifecycle, Mapping):
        lifecycle_errors.extend(_lifecycle_replay_errors(authority, lifecycle))
    lifecycle_passed = False
    if isinstance(lifecycle, Mapping) and isinstance(source_ids, list):
        try:
            fresh_roles = lifecycle["fresh_first_roles"]
            updated_roles = lifecycle["updated_roles"]
            reset_roles = lifecycle["reset_first_roles"]
            moved_source_id = lifecycle["moved_source_id"]
            if not all(
                isinstance(value, Mapping) and set(value) == set(source_ids)
                for value in (fresh_roles, updated_roles, reset_roles)
            ):
                raise M4EvidenceError("lifecycle phase roles differ from source IDs")
            if moved_source_id not in source_ids:
                raise M4EvidenceError("lifecycle moved_source_id is unknown")
            fresh = {
                source_id: _load_array(paths, fresh_roles[source_id], channels=4)
                for source_id in source_ids
            }
            updated = {
                source_id: _load_array(paths, updated_roles[source_id], channels=4)
                for source_id in source_ids
            }
            reset_first = {
                source_id: _load_array(paths, reset_roles[source_id], channels=4)
                for source_id in source_ids
            }
            reset_exact = all(
                np.array_equal(fresh[source_id], reset_first[source_id])
                for source_id in source_ids
            )
            moved_changed = not np.array_equal(
                fresh[moved_source_id], updated[moved_source_id]
            )
            declared_consistent = (
                lifecycle.get("reset_matches_fresh_first") is reset_exact
                and lifecycle.get("source_update_preserves_identity") is moved_changed
                and lifecycle.get("temporal_sequence_executed") is True
                and lifecycle.get("reset_boundary_policy")
                == "reset_reload_before_independent_episode"
                and lifecycle.get("counts_after_reset")
                == {"object_count": 0, "source_count": 0, "listener_count": 0}
            )
            lifecycle_passed = (
                reset_exact
                and moved_changed
                and declared_consistent
                and not lifecycle_errors
            )
            if not lifecycle_passed:
                lifecycle_errors.append(
                    f"reset_exact={reset_exact}, moved_changed={moved_changed}, "
                    f"declared_consistent={declared_consistent}"
                )
        except (KeyError, M4EvidenceError) as exc:
            lifecycle_errors.append(str(exc))
    else:
        lifecycle_errors.append("lifecycle or canonical source IDs are malformed")
    checks.append(
        _derived_check(
            "lifecycle_policy_receipt",
            lifecycle_passed,
            lifecycle_errors,
            "Reset/update/temporal policy receipt is incomplete",
        )
    )

    independently_passed = all(check["status"] == "pass" for check in checks)
    declared_checks = evidence.get("checks")
    declared_pass = bool(
        evidence.get("overall_status") == "pass"
        and isinstance(declared_checks, list)
        and declared_checks
        and all(
            check.get("status") == "pass"
            for check in declared_checks
            if isinstance(check, Mapping) and check.get("required", True)
        )
    )
    checks.append(
        _derived_check(
            "declared_status_consistency",
            independently_passed and declared_pass,
            {
                "independently_passed": independently_passed,
                "declared_overall_status": evidence.get("overall_status"),
            },
            "Declared M4 pass does not match independent verification",
        )
    )
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return status, tuple(checks)




def _current_runtime_identity_errors(evidence: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    inputs = evidence.get("inputs")
    if not isinstance(inputs, Mapping):
        errors.append("current-installed evidence inputs are missing")
    elif "runtime_lock_role" in inputs:
        errors.append("current-installed evidence must not retain a historical runtime lock")

    execution = evidence.get("execution")
    if not isinstance(execution, Mapping):
        errors.append("current-installed execution is missing")
    elif execution.get("runtime_mode") != "current-installed":
        errors.append("current-installed execution mode is invalid")

    runtime = evidence.get("runtime")
    if not isinstance(runtime, Mapping):
        return [*errors, "current-installed runtime receipt is missing"]
    if runtime.get("runtime_mode") != "current-installed":
        errors.append("current-installed runtime receipt mode is invalid")
    if "native_binaries" in runtime:
        errors.append(
            "current-installed runtime receipt must not retain native binary hashes"
        )
    records = runtime.get("current_installed_identity_records")
    if not isinstance(records, list) or not records:
        return [*errors, "current-installed identity records are missing"]
    unique: list[Any] = []
    for record in records:
        if record not in unique:
            unique.append(record)
    passed = all(
        _is_current_installed_runtime_identity(record) and record == records[0]
        for record in records
    )
    expected_identity = records[0] if passed else None
    if runtime.get("current_installed_identity") != expected_identity:
        errors.append(
            "current-installed summary identity differs from the fresh-call records"
        )

    performance = evidence.get("performance")
    performance_records: list[Any] = []
    if isinstance(performance, Mapping):
        for condition in ("one_source", "multi_source"):
            summary = performance.get(condition)
            runs = summary.get("runs") if isinstance(summary, Mapping) else None
            if not isinstance(runs, list):
                errors.append(
                    f"current-installed performance.{condition}.runs is missing"
                )
                continue
            for run in runs:
                if not isinstance(run, Mapping):
                    errors.append(
                        f"current-installed performance.{condition} has a non-object run"
                    )
                    continue
                performance_records.append(run.get("runtime_identity"))
    else:
        errors.append("current-installed performance receipt is missing")
    if len(records) != 7 + len(performance_records):
        errors.append(
            "current-installed runtime identity record count differs from native calls"
        )
    elif records[7:] != performance_records:
        errors.append(
            "current-installed runtime identity records differ from performance repeats"
        )

    declared_checks = evidence.get("checks")
    matching = [
        check
        for check in declared_checks
        if isinstance(check, Mapping)
        and check.get("check_id") == "runtime_current_installed_identity"
    ] if isinstance(declared_checks, list) else []
    if len(matching) != 1:
        errors.append(
            "current-installed evidence requires exactly one runtime identity check"
        )
    else:
        declared = matching[0]
        expected_measured = {
            "record_count": len(records),
            "unique_identity_count": len(unique),
            "identities": records,
        }
        if declared.get("status") != ("pass" if passed else "fail"):
            errors.append(
                "current-installed runtime identity check status differs from repeats"
            )
        if declared.get("measured") != expected_measured:
            errors.append(
                "current-installed runtime identity check measurements differ"
            )
        if declared.get("threshold") != {
            "same_runtime_identity_every_native_call": True,
            "runtime_mode": "current-installed",
        }:
            errors.append(
                "current-installed runtime identity check threshold differs"
            )
    return errors


def _verify_current_installed_m4_canary_evidence(
    path: Path, evidence: Mapping[str, Any]
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Reuse every v1 artifact/semantic reconstruction under v2 runtime policy."""

    return _verify_historical_m4_canary_evidence(
        path,
        current_installed=True,
        evidence=evidence,
    )


def verify_m4_canary_evidence(
    evidence_path: str | Path,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Dispatch v2 fresh receipts without invoking the v1 lock reader."""

    path = Path(evidence_path).resolve()
    try:
        evidence = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return "fail", (
            _derived_check(
                "evidence_json",
                False,
                str(exc),
                "Evidence is missing or invalid JSON",
            ),
        )
    if evidence.get("schema") == CURRENT_INSTALLED_EVIDENCE_SCHEMA:
        return _verify_current_installed_m4_canary_evidence(path, evidence)
    return _verify_historical_m4_canary_evidence(path)

__all__ = [
    "M4EvidenceError",
    "array_content_sha256",
    "artifact_record",
    "finalize_evidence",
    "make_check",
    "verify_m4_canary_evidence",
]
