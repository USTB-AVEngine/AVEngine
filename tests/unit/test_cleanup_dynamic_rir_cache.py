from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avengine.acoustics.dynamic_cache import (
    DYNAMIC_RIR_CACHE_SCHEMA,
    _array_digest,
    _request_metadata,
)
from avengine.contracts.json_io import canonical_json_sha256
from tools.acoustics.cleanup_dynamic_rir_cache import (
    DynamicRIRCacheCleanupError,
    cleanup,
)


def _cache(tmp_path: Path) -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    samples = np.ones((1, 2, 2, 2), dtype="<f4")
    lengths = np.full((1, 2), 2, dtype="<u4")
    metadata = {
        "schema": DYNAMIC_RIR_CACHE_SCHEMA,
        "source_ids": ["a", "b"],
        "keyframe_samples": [0],
        "sample_rate_hz": 16_000,
        "layout_type": "binaural",
        "array_shape": list(samples.shape),
        "array_dtype": samples.dtype.str,
        "lengths_shape": list(lengths.shape),
        "lengths_dtype": lengths.dtype.str,
    }
    metadata["content_sha256"] = _array_digest(samples, lengths)
    metadata["cache_identity_sha256"] = canonical_json_sha256(
        _request_metadata(metadata)
    )
    cache.mkdir()
    np.savez_compressed(cache / "sequence.npz", samples=samples, lengths=lengths)
    (cache / "manifest.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    report = tmp_path / "research_report.json"
    report.write_text(
        json.dumps({"dynamic_rir": {"cache": {"path": str(cache)}}}),
        encoding="utf-8",
    )
    return cache, report


def test_cleanup_dry_run_reports_payload_and_preserves_cache(tmp_path: Path) -> None:
    cache, report = _cache(tmp_path)
    result = cleanup(
        cache_root=cache,
        owned_root=tmp_path,
        reports=[report],
        execute=False,
    )
    assert result["status"] == "planned"
    assert result["storage"]["numeric_payload_bytes_before"] > 0
    assert (cache / "sequence.npz").is_file()


def test_cleanup_execute_removes_only_numeric_payload(tmp_path: Path) -> None:
    cache, report = _cache(tmp_path)
    result = cleanup(
        cache_root=cache,
        owned_root=tmp_path,
        reports=[report],
        execute=True,
    )
    assert result["status"] == "pass"
    assert result["numeric_payload_lifecycle"] == "numeric_payload_cleaned"
    assert result["historical_cache_path"] == str(cache.resolve())
    assert result["payload_bytes_cleared"] == result["storage"]["numeric_payload_bytes_cleared"]
    assert result["storage"]["numeric_payload_bytes_cleared"] > 0
    assert not (cache / "sequence.npz").exists()
    assert (cache / "manifest.json").is_file()
    record = json.loads((cache / "cleanup_record.json").read_text())
    assert record["status"] == "pass"


def test_cleanup_execute_keeps_historical_reference_metadata(tmp_path: Path) -> None:
    cache, report = _cache(tmp_path)
    historical = tmp_path / "old_report.json"
    historical.write_text(
        json.dumps({"status": "research", "cache": str(cache)}), encoding="utf-8"
    )
    result = cleanup(
        cache_root=cache,
        owned_root=tmp_path,
        reports=[report],
        execute=True,
    )
    assert result["unlisted_reference_count"] == 1
    assert str(historical.resolve()) in result["historical_durable_references"]
    record = json.loads((cache / "cleanup_record.json").read_text())
    assert record["numeric_payload_lifecycle"] == "numeric_payload_cleaned"
    assert str(historical.resolve()) in record["historical_durable_references"]
    assert historical.is_file()


def test_cleanup_refuses_failed_cache(tmp_path: Path) -> None:
    cache, report = _cache(tmp_path)
    (cache / "FAILED.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DynamicRIRCacheCleanupError, match="FAILED.json"):
        cleanup(cache_root=cache, owned_root=tmp_path, reports=[report], execute=True)


def test_cleanup_records_unlisted_historical_reference(tmp_path: Path) -> None:
    cache, report = _cache(tmp_path)
    historical = tmp_path / "other_episode.json"
    historical.write_text(
        json.dumps({"status": "pass", "cache": str(cache)}), encoding="utf-8"
    )
    result = cleanup(
        cache_root=cache,
        owned_root=tmp_path,
        reports=[report],
        execute=False,
    )
    assert result["status"] == "planned"
    assert result["unlisted_reference_count"] == 1
    assert str(historical.resolve()) in result["historical_durable_references"]


def test_cleanup_refuses_active_unlisted_dependency(tmp_path: Path) -> None:
    cache, report = _cache(tmp_path)
    (tmp_path / "active_episode.json").write_text(
        json.dumps({"status": "running", "cache": str(cache)}), encoding="utf-8"
    )
    with pytest.raises(DynamicRIRCacheCleanupError, match="active or incomplete"):
        cleanup(cache_root=cache, owned_root=tmp_path, reports=[report], execute=False)


def test_cleanup_dry_run_understands_existing_pair_cache_index(tmp_path: Path) -> None:
    cache = tmp_path / "pair_cache"
    pair = cache / "pair_00"
    (pair / "shards").mkdir(parents=True)
    (pair / "shards" / "shard_000000.npz").write_bytes(b"numeric")
    (pair / "request.json").write_text(
        json.dumps({"schema": "avengine_rlr_rir_cache_request_v1"}), encoding="utf-8"
    )
    (pair / "index.json").write_text(
        json.dumps({
            "schema": "avengine_rlr_rir_cache_index_v1",
            "status": "pass",
            "entries": [{"shard": "shards/shard_000000.npz"}],
        }), encoding="utf-8"
    )
    (pair / "receipt.json").write_text(
        json.dumps({
            "schema": "avengine_rlr_rir_cache_receipt_v1",
            "status": "pass",
            "full_plan_complete": True,
        }), encoding="utf-8"
    )
    (cache / "pair_sequence_index.json").write_text(
        json.dumps({
            "kind": "dynamic_rir_existing_cache_pair_sequence",
            "status": "pass",
            "pairs": [{"source_ids": ["a", "b"], "cache_path": "pair_00"}],
        }), encoding="utf-8"
    )
    report = tmp_path / "pair_report.json"
    report.write_text(json.dumps({"cache": str(cache)}), encoding="utf-8")
    result = cleanup(
        cache_root=cache,
        owned_root=tmp_path,
        reports=[report],
        execute=False,
    )
    assert result["cache_format"] == "avengine_rlr_rir_cache_v1_pair_shards"
    assert result["storage"]["numeric_payload_bytes_before"] == len(b"numeric")
