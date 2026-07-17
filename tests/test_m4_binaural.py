from __future__ import annotations

import hashlib

import numpy as np
import pytest

from avengine.m4.binaural import (
    BINAURAL_CHANNEL_ORDER,
    BinauralContractError,
    build_rlr_native_binaural_metadata,
    rlr_native_binaural_contract,
    validate_binaural_cardinals,
    validate_binaural_samples,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _preflight(tmp_path, *, render_rate: int = 16_000, hrtf_rate: int = 16_000):
    hrtf_payload = b"unit-test SOFA bytes"
    license_payload = b"unit-test redistributable HRTF license evidence\n"
    hrtf = tmp_path / "unit_hrtf.sofa"
    license_text = tmp_path / "HRTF_LICENSE.txt"
    hrtf.write_bytes(hrtf_payload)
    license_text.write_bytes(license_payload)
    result = build_rlr_native_binaural_metadata(
        render_rate,
        hrtf_path=hrtf,
        expected_hrtf_sha256=_sha256(hrtf_payload),
        hrtf_sample_rate_hz=hrtf_rate,
        license_id="unit-test-license",
        citation="Unit Test HRTF Authors (2026)",
        license_text_path=license_text,
        expected_license_sha256=_sha256(license_payload),
        asset_id="unit_hrtf_v1",
    )
    return result, hrtf, license_text


def _horizontal_cardinals() -> dict[str, np.ndarray]:
    right = np.zeros((2, 16), dtype=np.float64)
    right[:, 4] = (1.0, 4.0)
    left = np.zeros((2, 16), dtype=np.float64)
    left[:, 4] = (4.0, 1.0)
    return {"+X": right, "-X": left}


def test_native_binaural_contract_is_explicit_and_two_channel() -> None:
    contract = rlr_native_binaural_contract()

    assert contract["rendering_method"] == "rlr_native_binaural_v1"
    assert contract["channel_layout"] == {
        "type": "binaural",
        "channel_count": 2,
        "channel_order": list(BINAURAL_CHANNEL_ORDER),
    }
    assert contract["hrtf_policy"] == "explicit_hash_and_license_required"
    assert contract["avengine_resampling_policy"] == "forbidden"


def test_missing_or_implicit_hrtf_is_blocked_not_promoted() -> None:
    result = build_rlr_native_binaural_metadata(
        16_000,
        hrtf_path=None,
        expected_hrtf_sha256=None,
        hrtf_sample_rate_hz=None,
        license_id=None,
        citation=None,
    )

    assert result["status"] == "blocked"
    assert result["reason_code"] == "explicit_hrtf_missing"
    assert result["native_cardinal_validation"] == "not_run"


def test_hrtf_preflight_hashes_asset_and_license_without_claiming_render(tmp_path) -> None:
    result, hrtf, license_text = _preflight(tmp_path)

    assert result["status"] == "pass"
    assert result["verification_scope"] == "hrtf_dependency_preflight_only"
    assert result["native_cardinal_validation"] == "not_run"
    assert result["hrtf"]["path"] == str(hrtf.absolute())
    assert result["hrtf"]["sha256"] == result["hrtf"]["expected_sha256"]
    assert result["hrtf"]["sample_rate_hz"] == 16_000
    assert result["rights"]["license_text_path"] == str(license_text.absolute())
    assert (
        result["rights"]["license_text_sha256"]
        == result["rights"]["expected_license_sha256"]
    )


def test_hrtf_or_license_tampering_fails_closed(tmp_path) -> None:
    result, hrtf, license_text = _preflight(tmp_path)
    assert result["status"] == "pass"

    hrtf.write_bytes(b"tampered")
    tampered_hrtf = build_rlr_native_binaural_metadata(
        16_000,
        hrtf_path=hrtf,
        expected_hrtf_sha256=result["hrtf"]["expected_sha256"],
        hrtf_sample_rate_hz=16_000,
        license_id="unit-test-license",
        citation="Unit Test HRTF Authors (2026)",
        license_text_path=license_text,
        expected_license_sha256=result["rights"]["expected_license_sha256"],
    )
    assert tampered_hrtf["status"] == "fail"
    assert tampered_hrtf["reason_code"] == "hrtf_hash_mismatch"

    # Restore only the HRTF and alter the separately authenticated rights file.
    hrtf.write_bytes(b"unit-test SOFA bytes")
    license_text.write_bytes(b"different license bytes")
    tampered_license = build_rlr_native_binaural_metadata(
        16_000,
        hrtf_path=hrtf,
        expected_hrtf_sha256=result["hrtf"]["expected_sha256"],
        hrtf_sample_rate_hz=16_000,
        license_id="unit-test-license",
        citation="Unit Test HRTF Authors (2026)",
        license_text_path=license_text,
        expected_license_sha256=result["rights"]["expected_license_sha256"],
    )
    assert tampered_license["status"] == "fail"
    assert tampered_license["reason_code"] == "license_hash_mismatch"


def test_sample_rate_mismatch_is_explicitly_blocked_without_resampling(tmp_path) -> None:
    result, _, _ = _preflight(tmp_path, render_rate=16_000, hrtf_rate=44_100)

    assert result["status"] == "blocked"
    assert result["reason_code"] == "implicit_resampling_forbidden"
    assert "binary-bound" in result["reason"]


def test_sample_rate_mismatch_can_only_pass_when_bound_to_rlr_binary(tmp_path) -> None:
    preflight, hrtf, license_text = _preflight(tmp_path)
    binary_sha256 = "ab" * 32

    missing_binary = build_rlr_native_binaural_metadata(
        16_000,
        hrtf_path=hrtf,
        expected_hrtf_sha256=preflight["hrtf"]["expected_sha256"],
        hrtf_sample_rate_hz=44_100,
        license_id="unit-test-license",
        citation="Unit Test HRTF Authors (2026)",
        license_text_path=license_text,
        expected_license_sha256=preflight["rights"]["expected_license_sha256"],
        sample_rate_policy="rlr_native_internal_bound_to_binary",
    )
    assert missing_binary["status"] == "fail"
    assert missing_binary["reason_code"] == "rlr_binary_hash_required"

    result = build_rlr_native_binaural_metadata(
        16_000,
        hrtf_path=hrtf,
        expected_hrtf_sha256=preflight["hrtf"]["expected_sha256"],
        hrtf_sample_rate_hz=44_100,
        license_id="unit-test-license",
        citation="Unit Test HRTF Authors (2026)",
        license_text_path=license_text,
        expected_license_sha256=preflight["rights"]["expected_license_sha256"],
        sample_rate_policy="rlr_native_internal_bound_to_binary",
        rlr_binary_sha256=binary_sha256,
    )

    assert result["status"] == "pass"
    assert result["sample_rate_binding"] == {
        "policy": "rlr_native_internal_bound_to_binary",
        "render_sample_rate_hz": 16_000,
        "avengine_resampling_performed": False,
        "hrtf_input_sample_rate_hz": 44_100,
        "native_rate_adaptation": "performed_inside_pinned_rlr_binary",
        "rlr_binary_sha256": binary_sha256,
    }


def test_horizontal_cardinals_freeze_left_right_channel_order() -> None:
    evidence = validate_binaural_cardinals(
        _horizontal_cardinals(), minimum_ild_db=6.0
    )

    assert evidence["status"] == "pass"
    assert evidence["channel_order"] == ["left", "right"]
    assert evidence["measurements"]["+X"]["left_minus_right_ild_db"] < -6.0
    assert evidence["measurements"]["-X"]["left_minus_right_ild_db"] > 6.0


def test_swapped_or_malformed_binaural_canary_fails_closed() -> None:
    swapped = {
        direction: value[::-1].copy()
        for direction, value in _horizontal_cardinals().items()
    }
    with pytest.raises(BinauralContractError, match="does not favor right ear"):
        validate_binaural_cardinals(swapped)

    with pytest.raises(BinauralContractError, match=r"requires \+X and -X"):
        validate_binaural_cardinals({"+X": np.ones((2, 10))})
    with pytest.raises(BinauralContractError, match=r"shape \[2, samples\]"):
        validate_binaural_cardinals(
            {"+X": np.ones((4, 10)), "-X": np.ones((4, 10))}
        )
    with pytest.raises(BinauralContractError, match="finite"):
        validate_binaural_samples(np.full((2, 10), np.nan))
