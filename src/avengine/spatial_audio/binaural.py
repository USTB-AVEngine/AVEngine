"""Fail-closed metadata and direction checks for RLR-native binaural output.

M4 does not rely on an implicit RLR HRTF or on a workstation-only Python
decoder.  A native-binaural canary must name and authenticate an explicit
SOFA file, authenticate its license evidence, preserve the renderer's
``[left, right]`` channel order, and prove the two horizontal cardinal signs.

This module does not decode FOA, resample HRTFs, normalize audio, or claim that
an asset preflight is itself a successful native RLR simulation.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


RLR_NATIVE_BINAURAL_SCHEMA = "avengine_rlr_native_binaural_preflight_v1"
RLR_NATIVE_BINAURAL_METHOD = "rlr_native_binaural_v1"
BINAURAL_CHANNEL_ORDER = ("left", "right")
STRICT_SAMPLE_RATE_POLICY = "strict_match"
RLR_BOUND_SAMPLE_RATE_POLICY = "rlr_native_internal_bound_to_binary"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STABLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class BinauralContractError(ValueError):
    """Binaural samples or configuration violate the explicit M4 boundary."""


def rlr_native_binaural_contract() -> dict[str, Any]:
    """Return the renderer/layout portion shared by preflight and canary evidence."""

    return {
        "rendering_method": RLR_NATIVE_BINAURAL_METHOD,
        "renderer": "RLR native binaural listener",
        "channel_layout": {
            "type": "binaural",
            "channel_count": 2,
            "channel_order": list(BINAURAL_CHANNEL_ORDER),
        },
        "hrtf_policy": "explicit_hash_and_license_required",
        "avengine_resampling_policy": "forbidden",
        "native_rate_adaptation_policy": (
            "allowed only when explicitly bound to an RLR binary SHA-256"
        ),
        "normalization_policy": "forbidden",
        "limiter_policy": "forbidden",
    }


def _positive_integer(value: Any, *, owner: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise BinauralContractError(f"{owner} must be a positive integer")
    return int(value)


def _nonempty_text(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BinauralContractError(f"{owner} must be non-empty text")
    return value.strip()


def _preflight_result(
    base: dict[str, Any],
    *,
    status: str,
    reason_code: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    result = dict(base)
    result["status"] = status
    if reason_code is not None:
        result["reason_code"] = reason_code
    if reason is not None:
        result["reason"] = reason
    return result


def build_rlr_native_binaural_metadata(
    render_sample_rate_hz: int,
    *,
    hrtf_path: str | Path | None,
    expected_hrtf_sha256: str | None,
    hrtf_sample_rate_hz: int | None,
    license_id: str | None,
    citation: str | None,
    license_text_path: str | Path | None = None,
    expected_license_sha256: str | None = None,
    asset_id: str = "explicit_sofa_hrtf",
    sample_rate_policy: str = STRICT_SAMPLE_RATE_POLICY,
    rlr_binary_sha256: str | None = None,
) -> dict[str, Any]:
    """Authenticate all portable inputs for one RLR-native binaural canary.

    Missing external files produce ``blocked``.  An invalid declaration or a
    byte-hash mismatch produces ``fail``.  A sample-rate mismatch is
    ``blocked`` by default.  The sole explicit exception is RLR-native rate
    adaptation bound to the exact native binary SHA-256; AVEngine itself still
    performs no resampling.
    ``pass`` means only that the HRTF dependency preflight succeeded;
    ``native_cardinal_validation`` remains ``not_run`` until real RLR output
    passes :func:`validate_binaural_cardinals`.
    """

    rate = _positive_integer(render_sample_rate_hz, owner="render_sample_rate_hz")
    base: dict[str, Any] = {
        "schema": RLR_NATIVE_BINAURAL_SCHEMA,
        **rlr_native_binaural_contract(),
        "render_sample_rate_hz": rate,
        "verification_scope": "hrtf_dependency_preflight_only",
        "native_cardinal_validation": "not_run",
    }
    if sample_rate_policy not in {
        STRICT_SAMPLE_RATE_POLICY,
        RLR_BOUND_SAMPLE_RATE_POLICY,
    }:
        return _preflight_result(
            base,
            status="fail",
            reason_code="sample_rate_policy_invalid",
            reason=(
                "sample_rate_policy must be strict_match or "
                "rlr_native_internal_bound_to_binary"
            ),
        )
    base["sample_rate_binding"] = {
        "policy": sample_rate_policy,
        "render_sample_rate_hz": rate,
        "avengine_resampling_performed": False,
    }
    if not isinstance(asset_id, str) or not _STABLE_ID.fullmatch(asset_id):
        return _preflight_result(
            base,
            status="fail",
            reason_code="hrtf_asset_id_invalid",
            reason=(
                "asset_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
            ),
        )
    if hrtf_path is None:
        return _preflight_result(
            base,
            status="blocked",
            reason_code="explicit_hrtf_missing",
            reason="an implicit/default RLR HRTF cannot support reproducible output",
        )

    try:
        path = Path(hrtf_path).absolute()
    except (TypeError, ValueError, OSError) as exc:
        return _preflight_result(
            base,
            status="fail",
            reason_code="hrtf_path_invalid",
            reason=f"explicit HRTF path is invalid: {exc}",
        )
    base["hrtf"] = {"asset_id": asset_id, "path": str(path)}
    if path.suffix.casefold() != ".sofa":
        return _preflight_result(
            base,
            status="fail",
            reason_code="hrtf_format_invalid",
            reason="RLR native binaural requires an explicit .sofa HRTF",
        )
    if not path.is_file():
        return _preflight_result(
            base,
            status="blocked",
            reason_code="hrtf_file_unavailable",
            reason=f"explicit HRTF is missing: {path}",
        )
    if not isinstance(expected_hrtf_sha256, str) or not _SHA256.fullmatch(
        expected_hrtf_sha256
    ):
        return _preflight_result(
            base,
            status="fail",
            reason_code="hrtf_hash_declaration_invalid",
            reason="expected_hrtf_sha256 must be an explicit lowercase SHA-256",
        )
    try:
        hrtf_payload = path.read_bytes()
    except OSError as exc:
        return _preflight_result(
            base,
            status="blocked",
            reason_code="hrtf_file_unavailable",
            reason=f"explicit HRTF became unreadable: {exc}",
        )
    actual_hrtf_sha256 = hashlib.sha256(hrtf_payload).hexdigest()
    base["hrtf"] = {
        "asset_id": asset_id,
        "path": str(path),
        "byte_size": len(hrtf_payload),
        "sha256": actual_hrtf_sha256,
        "expected_sha256": expected_hrtf_sha256,
    }
    if actual_hrtf_sha256 != expected_hrtf_sha256:
        return _preflight_result(
            base,
            status="fail",
            reason_code="hrtf_hash_mismatch",
            reason="explicit HRTF bytes differ from the declared asset",
        )

    if hrtf_sample_rate_hz is None:
        return _preflight_result(
            base,
            status="fail",
            reason_code="hrtf_sample_rate_undeclared",
            reason="the SOFA sample rate must be declared and verified upstream",
        )
    try:
        hrtf_rate = _positive_integer(
            hrtf_sample_rate_hz,
            owner="hrtf_sample_rate_hz",
        )
    except BinauralContractError as exc:
        return _preflight_result(
            base,
            status="fail",
            reason_code="hrtf_sample_rate_invalid",
            reason=str(exc),
        )
    base["hrtf"]["sample_rate_hz"] = hrtf_rate
    base["sample_rate_binding"]["hrtf_input_sample_rate_hz"] = hrtf_rate
    if hrtf_rate != rate:
        if sample_rate_policy != RLR_BOUND_SAMPLE_RATE_POLICY:
            return _preflight_result(
                base,
                status="blocked",
                reason_code="implicit_resampling_forbidden",
                reason=(
                    f"HRTF sample rate {hrtf_rate} does not equal render sample rate "
                    f"{rate}; select the explicit binary-bound RLR native policy or "
                    "provide a separately pinned matching-rate asset"
                ),
            )
        if not isinstance(rlr_binary_sha256, str) or not _SHA256.fullmatch(
            rlr_binary_sha256
        ):
            return _preflight_result(
                base,
                status="fail",
                reason_code="rlr_binary_hash_required",
                reason=(
                    "sample-rate adaptation requires the exact lowercase "
                    "SHA-256 of the RLR native binary"
                ),
            )
        base["sample_rate_binding"].update(
            {
                "native_rate_adaptation": "performed_inside_pinned_rlr_binary",
                "rlr_binary_sha256": rlr_binary_sha256,
            }
        )
    else:
        base["sample_rate_binding"]["native_rate_adaptation"] = "not_required"

    try:
        declared_license_id = _nonempty_text(license_id, owner="license_id")
        declared_citation = _nonempty_text(citation, owner="citation")
    except BinauralContractError as exc:
        return _preflight_result(
            base,
            status="fail",
            reason_code="hrtf_rights_declaration_invalid",
            reason=str(exc),
        )
    if license_text_path is None:
        return _preflight_result(
            base,
            status="fail",
            reason_code="license_evidence_undeclared",
            reason="license_text_path and its expected SHA-256 are required",
        )
    try:
        license_path = Path(license_text_path).absolute()
    except (TypeError, ValueError, OSError) as exc:
        return _preflight_result(
            base,
            status="fail",
            reason_code="license_evidence_path_invalid",
            reason=f"HRTF license evidence path is invalid: {exc}",
        )
    if not license_path.is_file():
        return _preflight_result(
            base,
            status="blocked",
            reason_code="license_evidence_unavailable",
            reason=f"HRTF license evidence is missing: {license_path}",
        )
    if not isinstance(expected_license_sha256, str) or not _SHA256.fullmatch(
        expected_license_sha256
    ):
        return _preflight_result(
            base,
            status="fail",
            reason_code="license_hash_declaration_invalid",
            reason="expected_license_sha256 must be an explicit lowercase SHA-256",
        )
    try:
        license_payload = license_path.read_bytes()
    except OSError as exc:
        return _preflight_result(
            base,
            status="blocked",
            reason_code="license_evidence_unavailable",
            reason=f"HRTF license evidence became unreadable: {exc}",
        )
    actual_license_sha256 = hashlib.sha256(license_payload).hexdigest()
    base["rights"] = {
        "license_id": declared_license_id,
        "citation": declared_citation,
        "license_text_path": str(license_path),
        "license_text_byte_size": len(license_payload),
        "license_text_sha256": actual_license_sha256,
        "expected_license_sha256": expected_license_sha256,
    }
    if actual_license_sha256 != expected_license_sha256:
        return _preflight_result(
            base,
            status="fail",
            reason_code="license_hash_mismatch",
            reason="HRTF license evidence bytes differ from the declaration",
        )
    return _preflight_result(base, status="pass")


def _binaural_channel_major(
    value: Any,
    *,
    owner: str,
    channel_axis: int,
) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise BinauralContractError(f"{owner} must contain real numeric samples")
    if source.ndim != 2:
        raise BinauralContractError(f"{owner} must have two dimensions")
    if channel_axis in (0, -2):
        oriented = source
    elif channel_axis in (1, -1):
        oriented = source.T
    else:
        raise BinauralContractError("channel_axis must explicitly identify axis 0 or 1")
    if oriented.shape[0] != 2 or oriented.shape[1] < 1:
        raise BinauralContractError(
            f"{owner} must have shape [2, samples] in [left, right] order"
        )
    try:
        result = np.ascontiguousarray(oriented, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BinauralContractError(f"{owner} cannot be represented as float64") from exc
    if not np.all(np.isfinite(result)):
        raise BinauralContractError(f"{owner} must contain only finite samples")
    return result


def validate_binaural_samples(
    samples: Any,
    *,
    channel_axis: int = 0,
) -> np.ndarray:
    """Return a validated, owned ``float64`` ``[left, right]`` array."""

    return _binaural_channel_major(
        samples,
        owner="binaural samples",
        channel_axis=channel_axis,
    ).copy()


def validate_binaural_cardinals(
    impulse_responses: Mapping[str, Any],
    *,
    channel_axis: int = 0,
    minimum_ild_db: float = 3.0,
) -> dict[str, Any]:
    """Validate left/right channel order using direct-only +/-X canaries.

    Additional cardinal directions may be supplied and are measured, but only
    the horizontal signs are used as a robust pass/fail condition: +X/right
    must favor the right channel and -X/left must favor the left channel.
    """

    if not isinstance(impulse_responses, Mapping):
        raise BinauralContractError("impulse_responses must be a direction mapping")
    if "+X" not in impulse_responses or "-X" not in impulse_responses:
        raise BinauralContractError("binaural canary requires +X and -X responses")
    if any(
        not isinstance(direction, str) or direction not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}
        for direction in impulse_responses
    ):
        raise BinauralContractError("binaural canary contains an unknown direction")
    if (
        isinstance(minimum_ild_db, bool)
        or not isinstance(minimum_ild_db, (int, float))
        or not math.isfinite(float(minimum_ild_db))
        or float(minimum_ild_db) <= 0.0
    ):
        raise BinauralContractError("minimum_ild_db must be finite and positive")

    measurements: dict[str, Any] = {}
    sample_count: int | None = None
    for direction in sorted(impulse_responses):
        channels = _binaural_channel_major(
            impulse_responses[direction],
            owner=f"impulse_responses[{direction!r}]",
            channel_axis=channel_axis,
        )
        if sample_count is None:
            sample_count = int(channels.shape[1])
        elif channels.shape[1] != sample_count:
            raise BinauralContractError("all binaural cardinal responses must have equal length")
        energies = np.sum(np.square(channels), axis=1, dtype=np.float64)
        if not np.all(np.isfinite(energies)) or np.any(energies <= 0.0):
            raise BinauralContractError(
                f"{direction} must contain finite positive energy in both ears"
            )
        ild = 10.0 * math.log10(float(energies[0] / energies[1]))
        measurements[direction] = {
            "left_energy": float(energies[0]),
            "right_energy": float(energies[1]),
            "left_minus_right_ild_db": ild,
        }

    right_ild = float(measurements["+X"]["left_minus_right_ild_db"])
    left_ild = float(measurements["-X"]["left_minus_right_ild_db"])
    threshold = float(minimum_ild_db)
    if right_ild > -threshold:
        raise BinauralContractError(
            f"+X/right source does not favor right ear by {threshold:g} dB"
        )
    if left_ild < threshold:
        raise BinauralContractError(
            f"-X/left source does not favor left ear by {threshold:g} dB"
        )
    return {
        "status": "pass",
        "rendering_method": RLR_NATIVE_BINAURAL_METHOD,
        "channel_order": list(BINAURAL_CHANNEL_ORDER),
        "sample_count": sample_count,
        "minimum_ild_db": threshold,
        "measurements": measurements,
    }
