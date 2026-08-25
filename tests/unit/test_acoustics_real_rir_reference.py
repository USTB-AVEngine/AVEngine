from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import pytest

from avengine.cli import main
from avengine.contracts.json_io import canonical_json_sha256, load_json
from avengine.acoustics.real_rir_reference import (
    RealRIRReferenceError,
    verify_soundspaces2_real_rir_reference,
)


_MEASURED_RT60 = (
    1.476178,
    0.503969,
    1.493745,
    1.536138,
    1.301222,
    1.418662,
    0.682641,
)
_NEW_RT60 = (
    1.656730,
    0.573920,
    1.610374,
    1.618758,
    1.595985,
    1.676148,
    0.730463,
)
_OLD_RT60 = (
    1.655618,
    0.563774,
    1.660633,
    1.599005,
    1.595985,
    1.674542,
    0.730463,
)
_MEASURED_DRR = (
    -5.320621,
    -10.462627,
    -3.636609,
    -3.946469,
    -9.255854,
    -6.905437,
    -13.451425,
)
_NEW_DRR = (
    -6.104416,
    -8.496977,
    -3.766205,
    -4.993347,
    -8.737411,
    -4.594095,
    -13.569621,
)
_OLD_DRR = (
    4.493906,
    2.130483,
    6.521020,
    5.517731,
    2.027345,
    5.832680,
    -2.827158,
)


def _wav_bytes(*, format_code: int, sample_rate_hz: int, bits: int) -> bytes:
    channels = 1
    block_align = channels * bits // 8
    sample_payload = b"\0" * (block_align * 8)
    fmt = struct.pack(
        "<HHIIHH",
        format_code,
        channels,
        sample_rate_hz,
        sample_rate_hz * block_align,
        block_align,
        bits,
    )
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(sample_payload)) + sample_payload
    return b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks


def _metric_text(*, rt60: float, drr: float) -> str:
    return (
        "Bands\t63.000000\t1000.000000\t16000.000000\n"
        f"RT60\t1.000000\t{rt60:.6f}\t1.000000\n"
        "EDT\t0.000000\t0.000000\t0.000000\n"
        f"DRR\t-40.000000\t{drr:.6f}\t-4.000000\n"
        "C50\t1.000000\t2.000000\t3.000000\n"
        "C80\t1.000000\t2.000000\t3.000000\n"
        "D50\t1.000000\t2.000000\t3.000000\n"
        "TS\t1.000000\t2.000000\t3.000000\n"
    )


def _write_reference_fixture(root: Path) -> None:
    series = (
        (
            "Measured",
            1,
            48_000,
            16,
            _MEASURED_RT60,
            _MEASURED_DRR,
        ),
        ("Simulated New", 3, 44_100, 32, _NEW_RT60, _NEW_DRR),
        ("Simulated Old", 3, 44_100, 32, _OLD_RT60, _OLD_DRR),
    )
    for directory, format_code, sample_rate, bits, rt60, drr in series:
        metrics_directory = root / directory / "Metrics"
        metrics_directory.mkdir(parents=True)
        for index in range(7):
            anchor_id = f"ir{index + 1}"
            (root / directory / f"{anchor_id}.wav").write_bytes(
                _wav_bytes(
                    format_code=format_code,
                    sample_rate_hz=sample_rate,
                    bits=bits,
                )
            )
            (metrics_directory / f"{anchor_id} metrics.txt").write_text(
                _metric_text(rt60=rt60[index], drr=drr[index]),
                encoding="utf-8",
            )


def _pin_reference_fixture(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert len(hashes) == 42
    monkeypatch.setattr(
        "avengine.acoustics.real_rir_reference._PINNED_REFERENCE_SHA256_BY_PATH",
        hashes,
    )
    return hashes


def test_reference_fixture_reproduces_published_middle_band_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_reference_fixture(tmp_path)
    _pin_reference_fixture(tmp_path, monkeypatch)

    report = verify_soundspaces2_real_rir_reference(tmp_path)

    summary = report["computed_summary"]
    assert summary["simulated_new_mean_absolute_drr_error_db"] == pytest.approx(
        0.981985714285714
    )
    assert summary["simulated_old_mean_absolute_drr_error_db"] == pytest.approx(
        10.953578428571427
    )
    assert summary["simulated_new_mean_relative_rt60_error_percent"] == pytest.approx(
        12.44363646263409
    )
    assert report["reference_verified"] is True
    assert report["pinned_snapshot_identity_verified"] is True
    assert report["published_summary_reproduced"] is True
    assert report["engine_reexecution"] is False
    assert report["qualification_claim"] is False
    assert report["coordinate_binding"] == "blocked"
    assert report["metric_band_hz"] == 1000.0
    assert report["anchor_count"] == 7
    assert report["bound_file_count"] == 42
    assert len(report["inputs"]) == 21
    first = report["inputs"][0]
    assert first["series"] == "measured"
    assert first["wav"]["header"]["encoding"] == "pcm_s16le"
    assert first["wav"]["header"]["sample_rate_hz"] == 48_000
    assert first["wav"]["header"]["channel_count"] == 1
    assert (
        first["wav"]["sha256"]
        == hashlib.sha256((tmp_path / "Measured" / "ir1.wav").read_bytes()).hexdigest()
    )
    declared_hash = report.pop("report_content_sha256")
    assert declared_hash == canonical_json_sha256(report)


def test_missing_reference_file_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_reference_fixture(tmp_path)
    _pin_reference_fixture(tmp_path, monkeypatch)
    (tmp_path / "Simulated Old" / "ir7.wav").unlink()

    with pytest.raises(RealRIRReferenceError, match="missing"):
        verify_soundspaces2_real_rir_reference(tmp_path)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (
            "Bands\t63.000000\t500.000000\t16000.000000",
            "exactly the 63, 1000, and 16000 Hz bands",
        ),
        (
            "Bands\t63.000000\t1000.000000\t1000.000000",
            "duplicate metric bands",
        ),
        (
            "DRR\t-40.000000\t-5.320621\t-4.000000\n"
            "DRR\t-40.000000\t-5.320621\t-4.000000",
            "duplicate 'DRR' metric rows",
        ),
        (
            "RT60\t1.000000\t1.476178\t1.000000",
            "contains a non-finite value",
        ),
    ],
)
def test_malformed_metrics_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
    message: str,
) -> None:
    _write_reference_fixture(tmp_path)
    _pin_reference_fixture(tmp_path, monkeypatch)
    path = tmp_path / "Measured" / "Metrics" / "ir1 metrics.txt"
    text = path.read_text(encoding="utf-8")
    if replacement.startswith("Bands"):
        text = text.replace(text.splitlines()[0], replacement)
    elif replacement.startswith("DRR"):
        original = next(line for line in text.splitlines() if line.startswith("DRR"))
        text = text.replace(original, replacement)
    else:
        text = text.replace(replacement, replacement.replace("1.476178", "nan"))
    path.write_text(text, encoding="utf-8")

    with pytest.raises(RealRIRReferenceError, match=message):
        verify_soundspaces2_real_rir_reference(tmp_path)


@pytest.mark.parametrize(
    ("relative_path", "patch", "message"),
    [
        (
            Path("Measured/ir1.wav"),
            lambda payload: payload[:22] + struct.pack("<H", 2) + payload[24:],
            "must be mono",
        ),
        (
            Path("Simulated New/ir1.wav"),
            lambda payload: payload[:20] + struct.pack("<H", 1) + payload[22:],
            "unexpected WAV encoding",
        ),
    ],
)
def test_wrong_wav_header_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: Path,
    patch: object,
    message: str,
) -> None:
    _write_reference_fixture(tmp_path)
    _pin_reference_fixture(tmp_path, monkeypatch)
    path = tmp_path / relative_path
    payload = path.read_bytes()
    path.write_bytes(patch(payload))  # type: ignore[operator]

    with pytest.raises(RealRIRReferenceError, match=message):
        verify_soundspaces2_real_rir_reference(tmp_path)


def test_valid_but_modified_wav_fails_pinned_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_reference_fixture(tmp_path)
    _pin_reference_fixture(tmp_path, monkeypatch)
    path = tmp_path / "Measured" / "ir1.wav"
    payload = bytearray(path.read_bytes())
    payload[-1] = 1
    path.write_bytes(payload)

    with pytest.raises(RealRIRReferenceError, match="pinned official snapshot"):
        verify_soundspaces2_real_rir_reference(tmp_path)


def test_cli_writes_verified_reference_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference = tmp_path / "reference"
    _write_reference_fixture(reference)
    _pin_reference_fixture(reference, monkeypatch)
    report_path = tmp_path / "reference_report.json"

    arguments = [
        "m3",
        "verify-soundspaces-reference",
        "--reference-root",
        str(reference),
        "--output",
        str(report_path),
    ]
    assert main(arguments) == 0
    report = load_json(report_path)
    assert report["reference_verified"] is True
    assert report["pinned_snapshot_identity_verified"] is True
    assert report["published_summary_reproduced"] is True
    assert report["qualification_claim"] is False
    assert report["coordinate_binding"] == "blocked"
    assert main(arguments) == 2
    assert list(tmp_path.glob(f".{report_path.name}.staging-*")) == []
    assert '"status": "pass"' in capsys.readouterr().out
