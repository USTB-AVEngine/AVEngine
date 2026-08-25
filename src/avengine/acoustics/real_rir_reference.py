"""Verify the public SoundSpaces 2 real-RIR reference package.

This module verifies the *published, precomputed* comparison bundle.  It does
not run AVEngine, RLR, or SoundSpaces, and therefore cannot qualify a current
room simulation.  The public bundle also omits the seven source/listener
coordinate pairs and the exact scan identifier, so coordinate binding remains
explicitly blocked.

Only WAV container metadata is parsed.  Sample payloads are hashed but are
never decoded or resampled.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
from typing import Any

from avengine.contracts.json_io import canonical_json_sha256


REFERENCE_SCHEMA = "avengine_m3_soundspaces2_real_rir_reference_v1"
MIDDLE_BAND_HZ = 1000.0

_ANCHOR_IDS = tuple(f"ir{index}" for index in range(1, 8))
_BANDS_HZ = (63.0, MIDDLE_BAND_HZ, 16000.0)
_METRIC_NAMES = ("RT60", "EDT", "DRR", "C50", "C80", "D50", "TS")
_SERIES = (
    ("measured", "Measured", 1, 48_000, 16, "pcm_s16le"),
    (
        "simulated_new",
        "Simulated New",
        3,
        44_100,
        32,
        "ieee_float32le",
    ),
    (
        "simulated_old",
        "Simulated Old",
        3,
        44_100,
        32,
        "ieee_float32le",
    ),
)

# Values printed by the public workbook/paper, with tolerance for the six
# decimal places retained by the bundled text files.
_PUBLISHED_TARGETS = {
    "simulated_new_mean_absolute_drr_error_db": 0.981986,
    "simulated_old_mean_absolute_drr_error_db": 10.9536,
    "simulated_new_mean_relative_rt60_error_percent": 12.4436,
}
_PUBLISHED_TARGET_ABSOLUTE_TOLERANCE = 5e-5
_PINNED_SOURCE_ARCHIVE = {
    "uri": "https://dl.fbaipublicfiles.com/SoundSpaces/real_measurements.zip",
    "supplemental_copy": (
        "https://proceedings.neurips.cc/paper_files/paper/2022/file/"
        "3a48b0eaba26ba862220a307a9edb0bb-"
        "Supplemental-Datasets_and_Benchmarks.zip#supp/real_measurements.zip"
    ),
    "byte_size": 5_672_029,
    "sha256": "bdc6a9673b44a89c90caf227fe816d40b979f520771aff955921b5e141ca05f3",
}
# This manifest was derived from the 42 WAV/metric members of the pinned
# public archive above. The verifier checks the extracted files against these
# hashes; merely reproducing the printed metric values is not sufficient.
_PINNED_REFERENCE_SHA256_BY_PATH = {
    "Measured/ir1.wav": (
        "0fb4bdcbddd6b591ebfc3488276d28ff52d57e98ab63fd852c837eb2a39cf30b"
    ),
    "Measured/Metrics/ir1 metrics.txt": (
        "837002f423458668ea13b21bd71a4719708296bd9744eeb4a1e0e50600906ab1"
    ),
    "Measured/ir2.wav": (
        "050aaf878c0e81ece2b1c83764956a240a2886b9d0cbfdc841ac8cc5de03b25a"
    ),
    "Measured/Metrics/ir2 metrics.txt": (
        "5bc5d71266509bd5e41895d86f1fc39b17b7a41667b1b0d4e34a1b8778e00802"
    ),
    "Measured/ir3.wav": (
        "ca8ce1dd5950b42f4c1116f86958b6073a31be729adbdcd21a7bcbcbf9ffc6d4"
    ),
    "Measured/Metrics/ir3 metrics.txt": (
        "9d3461452e42612649eeeef281eeecbfe4b343a4ed8b108b09e4991142c689f6"
    ),
    "Measured/ir4.wav": (
        "4239c29b85b1dbbe058b938230eaf81629ce283cd4b62a1f0f04f7526bcf0c11"
    ),
    "Measured/Metrics/ir4 metrics.txt": (
        "e1f9332728ce7590c7bda6cca7d7e2e11c1d704975ed8908f9c1428d6cceaede"
    ),
    "Measured/ir5.wav": (
        "da93c1c51612a6b08115d507814b171ec1e8cf20b74f5726ffb66e38eb611ae0"
    ),
    "Measured/Metrics/ir5 metrics.txt": (
        "12ec067d3030e8bbbd192a6454d7da5f8c8e6899a75e34fb31874fa6f47407f7"
    ),
    "Measured/ir6.wav": (
        "6cc21cd51081001f6c9ee702134485b0fbaec6c65c0c2d4cff375c9527014412"
    ),
    "Measured/Metrics/ir6 metrics.txt": (
        "c7bdac49cdd677c4c14ab49e44d800ca941458df0927578e1d2a30b0af17acf6"
    ),
    "Measured/ir7.wav": (
        "df4a1e026084375afec59c3354f4413e6abfe444561239d523fc2d323250e791"
    ),
    "Measured/Metrics/ir7 metrics.txt": (
        "408a1bc9585b82bcf83f178601dfba1311448dd23d4e74ee2b9b3573ae2e9a03"
    ),
    "Simulated New/ir1.wav": (
        "c66616e783ae7b7160e16c6a0d9fbe91c55fffa7551f093d7e2bb3af910d4719"
    ),
    "Simulated New/Metrics/ir1 metrics.txt": (
        "06134d922e028c12876e02d196c32120aad1a6a9f47b733fea950e262f4128cd"
    ),
    "Simulated New/ir2.wav": (
        "32fa3bcd1dd8ff60b4e260baf3ede95353e4775ef637e01497cf37b7d18f0f62"
    ),
    "Simulated New/Metrics/ir2 metrics.txt": (
        "fb61feae0d7470d4a0c7e67abe7db28e606a3aa249610de0681d28b9710de7bb"
    ),
    "Simulated New/ir3.wav": (
        "b7e7bc9ef3481dddf75729ffc5c698e7120d5ac3f175171fe6b807bf135742d1"
    ),
    "Simulated New/Metrics/ir3 metrics.txt": (
        "0fde6131cf38445608c353644f493a9e959539875c0cb400b1851da08a59dfd3"
    ),
    "Simulated New/ir4.wav": (
        "fd7a85946384902bdaff6aec0a912215a0779b5aaeefc79aae73cbd002e7128c"
    ),
    "Simulated New/Metrics/ir4 metrics.txt": (
        "e0e5c436234201c885353b237f29acc813ac49239e9513c39a7dc886180fc0c3"
    ),
    "Simulated New/ir5.wav": (
        "9f5037ae77c89c94c66631591189d5f90593a8e768701acf78650c3b216f5701"
    ),
    "Simulated New/Metrics/ir5 metrics.txt": (
        "1bcffd6336f90d41ab31d06472e5c84cdfe84195be19d6902f90587ec42eaf61"
    ),
    "Simulated New/ir6.wav": (
        "688ee28325887123796cb8f20f9a3062bcb15c2d6af0453f9133742f30b5d6e1"
    ),
    "Simulated New/Metrics/ir6 metrics.txt": (
        "9d5780737805eb606cadc9242f6334a3a9ef73916a103c0ea152ba468b391e0e"
    ),
    "Simulated New/ir7.wav": (
        "08e3a8a8db7a83d7e1c9dd29f475b064e65fc3744f2c83419b3fbee63acc450d"
    ),
    "Simulated New/Metrics/ir7 metrics.txt": (
        "1615fd88528576db9d979a29914601d372b1918c8d6eadbb2050cd8fb7e9e091"
    ),
    "Simulated Old/ir1.wav": (
        "f015e3e451173a33cee0af53ea271ae90a792de891075b18cc8116cacebc28fe"
    ),
    "Simulated Old/Metrics/ir1 metrics.txt": (
        "7fdd0314f1b2c2f43e675ff05b4e628b465464b52ebb5d399ecc36b4e54254c2"
    ),
    "Simulated Old/ir2.wav": (
        "7c8ad4d6a273ade55f583d8c22788c59ccc254aeb6ddf72e5b189470f17439a7"
    ),
    "Simulated Old/Metrics/ir2 metrics.txt": (
        "6deaa33ba25e78c033af7971e838828fec9c8d889964f6cb2b863d3fa3f83309"
    ),
    "Simulated Old/ir3.wav": (
        "70620ce5fec015ee51962742efe45aa930c3de06bcf9d0cd14d5318753286e27"
    ),
    "Simulated Old/Metrics/ir3 metrics.txt": (
        "d38883898e16650197deb965a1895ead796b21848b0cf84f252b2b80e2d0eecc"
    ),
    "Simulated Old/ir4.wav": (
        "abb1defbf321d5864d2157811d16b8894373b7454f3c1528e80966eb7b76a4ba"
    ),
    "Simulated Old/Metrics/ir4 metrics.txt": (
        "0c9ad2902da13509fdf62145c0ff7d650ff89ce9291194aa4b50cbfef07f4b4b"
    ),
    "Simulated Old/ir5.wav": (
        "e6963b92ed6c72cffed954e9cdb18c72b0bcea7ebe94f582b088f14a3edac4e6"
    ),
    "Simulated Old/Metrics/ir5 metrics.txt": (
        "8686cb44c4e2b7eff404c0ebb0e577111439c424d1e7e31b3d7d9e68634ac6b9"
    ),
    "Simulated Old/ir6.wav": (
        "bb5e757dbc9add9a3d572083c8ded20cd8b8c146fd9017bce79bf4465f8a7fd4"
    ),
    "Simulated Old/Metrics/ir6 metrics.txt": (
        "5863b19293cc47369075ac1d3f200f060ceac7abdab0a0c8f9b0ce77059a2c52"
    ),
    "Simulated Old/ir7.wav": (
        "958b886edcd18d2ff3f9a09d4a780a7fbd5c1360b3a6661613003a19ab6e7457"
    ),
    "Simulated Old/Metrics/ir7 metrics.txt": (
        "fd0464b4395de13e4759617c50a6f304288bf477a47c4d6fb04bc8933c375330"
    ),
}


class RealRIRReferenceError(ValueError):
    """The reference package cannot support a verified comparison."""


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    relative_path: str
    payload: bytes
    byte_size: int
    sha256: str

    def file_record(self) -> dict[str, Any]:
        return {
            "path": self.relative_path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class _WavHeader:
    encoding: str
    sample_rate_hz: int
    channel_count: int
    bits_per_sample: int
    sample_count: int
    data_byte_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "sample_rate_hz": self.sample_rate_hz,
            "channel_count": self.channel_count,
            "bits_per_sample": self.bits_per_sample,
            "sample_count": self.sample_count,
            "data_byte_size": self.data_byte_size,
            "duration_seconds": self.sample_count / self.sample_rate_hz,
        }


def _snapshot(
    root: Path,
    relative_path: Path,
    *,
    seen_paths: set[Path],
) -> _FileSnapshot:
    declared_path = root / relative_path
    try:
        resolved = declared_path.resolve(strict=True)
        resolved.relative_to(root)
    except FileNotFoundError as exc:
        raise RealRIRReferenceError(
            f"required reference file is missing: {relative_path.as_posix()}"
        ) from exc
    except ValueError as exc:
        raise RealRIRReferenceError(
            f"reference file escapes the package root: {relative_path.as_posix()}"
        ) from exc
    if resolved in seen_paths:
        raise RealRIRReferenceError(
            f"duplicate reference file binding: {relative_path.as_posix()}"
        )
    seen_paths.add(resolved)
    if not resolved.is_file():
        raise RealRIRReferenceError(
            f"required reference path is not a file: {relative_path.as_posix()}"
        )
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise RealRIRReferenceError(
            f"required reference file is unreadable: {relative_path.as_posix()}"
        ) from exc
    return _FileSnapshot(
        path=resolved,
        relative_path=relative_path.as_posix(),
        payload=payload,
        byte_size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _verify_pinned_file_identity(snapshot: _FileSnapshot) -> None:
    expected = _PINNED_REFERENCE_SHA256_BY_PATH.get(snapshot.relative_path)
    if expected is None:
        raise RealRIRReferenceError(
            "reference path is absent from the pinned official snapshot: "
            f"{snapshot.relative_path}"
        )
    if snapshot.sha256 != expected:
        raise RealRIRReferenceError(
            "reference file does not match the pinned official snapshot: "
            f"{snapshot.relative_path}"
        )


def _parse_wav_header(
    snapshot: _FileSnapshot,
    *,
    expected_format_code: int,
    expected_sample_rate_hz: int,
    expected_bits_per_sample: int,
    expected_encoding: str,
) -> _WavHeader:
    payload = snapshot.payload
    owner = snapshot.relative_path
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise RealRIRReferenceError(f"{owner} is not a little-endian RIFF/WAVE file")
    declared_size = struct.unpack_from("<I", payload, 4)[0] + 8
    if declared_size != len(payload):
        raise RealRIRReferenceError(
            f"{owner} RIFF byte size does not match the bound file"
        )

    offset = 12
    chunks: dict[bytes, bytes | int] = {}
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise RealRIRReferenceError(f"{owner} has a truncated WAV chunk header")
        chunk_id, chunk_size = struct.unpack_from("<4sI", payload, offset)
        data_start = offset + 8
        data_end = data_start + chunk_size
        padded_end = data_end + (chunk_size & 1)
        if data_end > len(payload) or padded_end > len(payload):
            raise RealRIRReferenceError(f"{owner} has a truncated WAV chunk")
        if chunk_id in chunks:
            display_id = chunk_id.decode("ascii", errors="replace")
            raise RealRIRReferenceError(
                f"{owner} has duplicate {display_id!r} WAV chunks"
            )
        chunks[chunk_id] = (
            chunk_size if chunk_id == b"data" else payload[data_start:data_end]
        )
        offset = padded_end
    if offset != len(payload):
        raise RealRIRReferenceError(f"{owner} WAV chunk alignment is invalid")

    fmt = chunks.get(b"fmt ")
    data_size = chunks.get(b"data")
    if not isinstance(fmt, bytes) or not isinstance(data_size, int):
        raise RealRIRReferenceError(f"{owner} must contain one fmt and one data chunk")
    if len(fmt) != 16:
        raise RealRIRReferenceError(f"{owner} must use a 16-byte PCM-style fmt chunk")
    (
        format_code,
        channel_count,
        sample_rate_hz,
        byte_rate,
        block_align,
        bits_per_sample,
    ) = struct.unpack("<HHIIHH", fmt)
    if channel_count != 1:
        raise RealRIRReferenceError(f"{owner} must be mono")
    if (
        format_code != expected_format_code
        or sample_rate_hz != expected_sample_rate_hz
        or bits_per_sample != expected_bits_per_sample
    ):
        raise RealRIRReferenceError(
            f"{owner} has an unexpected WAV encoding or sample rate"
        )
    expected_block_align = channel_count * bits_per_sample // 8
    if (
        bits_per_sample % 8 != 0
        or block_align != expected_block_align
        or byte_rate != sample_rate_hz * block_align
    ):
        raise RealRIRReferenceError(f"{owner} WAV rate/alignment fields are invalid")
    if data_size <= 0 or data_size % block_align != 0:
        raise RealRIRReferenceError(f"{owner} WAV data size is invalid")
    return _WavHeader(
        encoding=expected_encoding,
        sample_rate_hz=sample_rate_hz,
        channel_count=channel_count,
        bits_per_sample=bits_per_sample,
        sample_count=data_size // block_align,
        data_byte_size=data_size,
    )


def _parse_number(raw: str, *, owner: str) -> float:
    try:
        result = float(raw)
    except ValueError as exc:
        raise RealRIRReferenceError(f"{owner} contains a non-numeric value") from exc
    if not math.isfinite(result):
        raise RealRIRReferenceError(f"{owner} contains a non-finite value")
    return result


def _parse_metrics(snapshot: _FileSnapshot) -> dict[str, tuple[float, ...]]:
    owner = snapshot.relative_path
    try:
        text = snapshot.payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RealRIRReferenceError(f"{owner} is not UTF-8 text") from exc
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise RealRIRReferenceError(f"{owner} has an empty or malformed metrics row")

    header = lines[0].split("\t")
    if len(header) != 4 or header[0] != "Bands":
        raise RealRIRReferenceError(f"{owner} has an invalid Bands header")
    bands = tuple(_parse_number(value, owner=f"{owner} Bands") for value in header[1:])
    if len(set(bands)) != len(bands):
        raise RealRIRReferenceError(f"{owner} contains duplicate metric bands")
    if bands != _BANDS_HZ:
        raise RealRIRReferenceError(
            f"{owner} must contain exactly the 63, 1000, and 16000 Hz bands"
        )

    metrics: dict[str, tuple[float, ...]] = {}
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 4 or not fields[0]:
            raise RealRIRReferenceError(f"{owner} has a malformed metrics row")
        metric_name = fields[0]
        if metric_name in metrics:
            raise RealRIRReferenceError(
                f"{owner} contains duplicate {metric_name!r} metric rows"
            )
        if metric_name not in _METRIC_NAMES:
            raise RealRIRReferenceError(
                f"{owner} contains unexpected metric {metric_name!r}"
            )
        metrics[metric_name] = tuple(
            _parse_number(value, owner=f"{owner} {metric_name}") for value in fields[1:]
        )
    missing = [name for name in _METRIC_NAMES if name not in metrics]
    if missing:
        raise RealRIRReferenceError(
            f"{owner} is missing metric rows: {', '.join(missing)}"
        )
    if any(value <= 0.0 for value in metrics["RT60"]):
        raise RealRIRReferenceError(f"{owner} RT60 values must be positive")
    return metrics


def _mean_absolute_error(reference: list[float], candidate: list[float]) -> float:
    return sum(
        abs(actual - predicted) for actual, predicted in zip(reference, candidate)
    ) / len(reference)


def _mean_relative_error_percent(
    reference: list[float], candidate: list[float]
) -> float:
    if any(value == 0.0 for value in reference):
        raise RealRIRReferenceError("measured RT60 cannot be zero")
    return (
        sum(
            abs(actual - predicted) / abs(actual)
            for actual, predicted in zip(reference, candidate)
        )
        / len(reference)
        * 100.0
    )


def _verify_published_target(name: str, value: float) -> None:
    target = _PUBLISHED_TARGETS[name]
    if not math.isclose(
        value,
        target,
        rel_tol=0.0,
        abs_tol=_PUBLISHED_TARGET_ABSOLUTE_TOLERANCE,
    ):
        raise RealRIRReferenceError(
            f"computed {name}={value:.9f} does not reproduce "
            f"the published target {target:.6f}"
        )


def verify_soundspaces2_real_rir_reference(
    reference_root: str | Path,
) -> dict[str, Any]:
    """Verify and bind the extracted public SoundSpaces 2 RIR comparison.

    ``reference_root`` must directly contain ``Measured``, ``Simulated New``,
    and ``Simulated Old``.  A successful report verifies the integrity and
    published summary of those supplied files only.  It deliberately keeps
    ``qualification_claim`` false because no simulation is rerun.
    """

    try:
        root = Path(reference_root).resolve(strict=True)
    except FileNotFoundError as exc:
        raise RealRIRReferenceError(
            f"reference root does not exist: {reference_root}"
        ) from exc
    if not root.is_dir():
        raise RealRIRReferenceError("reference root must be a directory")

    required_paths = {
        relative_path
        for _series_id, directory_name, *_format in _SERIES
        for anchor_id in _ANCHOR_IDS
        for relative_path in (
            f"{directory_name}/{anchor_id}.wav",
            f"{directory_name}/Metrics/{anchor_id} metrics.txt",
        )
    }
    if set(_PINNED_REFERENCE_SHA256_BY_PATH) != required_paths:
        raise RealRIRReferenceError(
            "compiled-in SoundSpaces 2 reference manifest is incomplete"
        )

    seen_paths: set[Path] = set()
    inputs: list[dict[str, Any]] = []
    values_by_anchor: dict[str, dict[str, dict[str, float]]] = {
        anchor_id: {} for anchor_id in _ANCHOR_IDS
    }

    for (
        series_id,
        directory_name,
        format_code,
        sample_rate_hz,
        bits_per_sample,
        encoding,
    ) in _SERIES:
        for anchor_id in _ANCHOR_IDS:
            wav_snapshot = _snapshot(
                root,
                Path(directory_name) / f"{anchor_id}.wav",
                seen_paths=seen_paths,
            )
            metrics_snapshot = _snapshot(
                root,
                Path(directory_name) / "Metrics" / f"{anchor_id} metrics.txt",
                seen_paths=seen_paths,
            )
            wav_header = _parse_wav_header(
                wav_snapshot,
                expected_format_code=format_code,
                expected_sample_rate_hz=sample_rate_hz,
                expected_bits_per_sample=bits_per_sample,
                expected_encoding=encoding,
            )
            metrics = _parse_metrics(metrics_snapshot)
            _verify_pinned_file_identity(wav_snapshot)
            _verify_pinned_file_identity(metrics_snapshot)
            middle_index = _BANDS_HZ.index(MIDDLE_BAND_HZ)
            values_by_anchor[anchor_id][series_id] = {
                "rt60_seconds": metrics["RT60"][middle_index],
                "drr_db": metrics["DRR"][middle_index],
            }
            inputs.append(
                {
                    "anchor_id": anchor_id,
                    "series": series_id,
                    "wav": {
                        **wav_snapshot.file_record(),
                        "header": wav_header.to_dict(),
                    },
                    "metrics": metrics_snapshot.file_record(),
                }
            )

    measured_drr = [
        values_by_anchor[anchor]["measured"]["drr_db"] for anchor in _ANCHOR_IDS
    ]
    new_drr = [
        values_by_anchor[anchor]["simulated_new"]["drr_db"] for anchor in _ANCHOR_IDS
    ]
    old_drr = [
        values_by_anchor[anchor]["simulated_old"]["drr_db"] for anchor in _ANCHOR_IDS
    ]
    measured_rt60 = [
        values_by_anchor[anchor]["measured"]["rt60_seconds"] for anchor in _ANCHOR_IDS
    ]
    new_rt60 = [
        values_by_anchor[anchor]["simulated_new"]["rt60_seconds"]
        for anchor in _ANCHOR_IDS
    ]
    computed = {
        "simulated_new_mean_absolute_drr_error_db": _mean_absolute_error(
            measured_drr, new_drr
        ),
        "simulated_old_mean_absolute_drr_error_db": _mean_absolute_error(
            measured_drr, old_drr
        ),
        "simulated_new_mean_relative_rt60_error_percent": (
            _mean_relative_error_percent(measured_rt60, new_rt60)
        ),
    }
    for name, value in computed.items():
        if not math.isfinite(value):
            raise RealRIRReferenceError(f"computed {name} is non-finite")
        _verify_published_target(name, value)

    report: dict[str, Any] = {
        "schema": REFERENCE_SCHEMA,
        "reference_verified": True,
        "pinned_snapshot_identity_verified": True,
        "published_summary_reproduced": True,
        "verification_scope": (
            "pinned_official_precomputed_files_and_published_summary_only"
        ),
        "engine_reexecution": False,
        "qualification_claim": False,
        "coordinate_binding": "blocked",
        "coordinate_binding_reason": (
            "the public bundle does not publish the seven source/listener "
            "coordinate pairs or exact scan identifier"
        ),
        "anchor_count": len(_ANCHOR_IDS),
        "bound_file_count": len(inputs) * 2,
        "metric_band_hz": MIDDLE_BAND_HZ,
        "pinned_source_snapshot": {
            **_PINNED_SOURCE_ARCHIVE,
            "archive_readback_performed": False,
            "identity_basis": (
                "compiled-in SHA-256 manifest derived from the pinned public archive"
            ),
            "bound_reference_file_count": len(_PINNED_REFERENCE_SHA256_BY_PATH),
            "bound_reference_file_manifest_sha256": canonical_json_sha256(
                _PINNED_REFERENCE_SHA256_BY_PATH
            ),
        },
        "calculation": {
            "drr": "mean(abs(simulated_drr_db - measured_drr_db))",
            "rt60": (
                "100 * mean(abs(simulated_rt60 - measured_rt60) / abs(measured_rt60))"
            ),
            "published_target_absolute_tolerance": (
                _PUBLISHED_TARGET_ABSOLUTE_TOLERANCE
            ),
        },
        "published_targets": dict(_PUBLISHED_TARGETS),
        "computed_summary": computed,
        "anchors": [
            {
                "anchor_id": anchor_id,
                "metrics_1000_hz": values_by_anchor[anchor_id],
            }
            for anchor_id in _ANCHOR_IDS
        ],
        "inputs": inputs,
    }
    report["report_content_sha256"] = canonical_json_sha256(report)
    return report
