from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import wave

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "replay_qa_batch_audio_test_module", ROOT / "tools/dataset/replay_qa_batch_audio.py"
)
_replay = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _replay
assert _spec.loader is not None
_spec.loader.exec_module(_replay)


def context():
    return {
        "input_manifest_path": "/inputs.json",
        "input_manifest": {
            "normalization": {
                "mode": "peak_dbfs",
                "target_dbfs": -3.0,
                "applied_once_at": "event_pcm_split",
            }
        },
        "mapping_path": "/mapping.json",
        "target_pool_path": "/new-pool.json",
        "mapping": {
            "old_event": {
                "old_sound_asset_id": "old_event",
                "new_sound_asset_id": "new_event",
                "old_sample_count": 160,
                "new_sample_count": 160,
                "old_sample_rate_hz": 16000,
                "new_sample_rate_hz": 16000,
            },
            "old_speech": {
                "old_sound_asset_id": "old_speech",
                "new_sound_asset_id": "new_speech",
                "old_sample_count": 320,
                "new_sample_count": 320,
                "old_sample_rate_hz": 16000,
                "new_sample_rate_hz": 16000,
            },
        },
        "target_sounds": {
            "new_event": {
                "sound_asset_id": "new_event",
                "path": "/new/event.wav",
                "sample_count": 160,
                "sample_rate_hz": 16000,
                "active_duration_s": 0.01,
                "source_normalization": {
                    "policy": {"mode": "peak_dbfs", "target_dbfs": -3.0},
                    "applied_gain_db": 6.0,
                    "measured_peak_dbfs": -3.0,
                    "source": {"event_manifest_path": "/new/events.json"},
                },
                "normalization_applied": False,
            },
            "new_speech": {
                "sound_asset_id": "new_speech",
                "path": "/new/speech.wav",
                "sample_count": 320,
                "sample_rate_hz": 16000,
                "prepared_audio_id": "new_speech",
                "source_normalization": {"policy": None, "applied_gain_db": 0.0},
            },
        },
    }


def test_rebind_record_replaces_audio_identity_without_timing_or_runtime_gain_change():
    before = {
        "event_id": "event_001",
        "actor_id": "source1",
        "sound_asset_id": "old_event",
        "path": "/old/event.wav",
        "sample_count": 160,
        "sample_rate_hz": 16000,
        "start_sample": 400,
        "end_sample_exclusive": 560,
        "start_tick": 1200,
        "end_tick_exclusive": 1680,
        "linear_gain": 0.15,
    }
    original = deepcopy(before)
    rebound = _replay.rebind_audio_payload(before, context())
    assert before == original
    assert rebound["sound_asset_id"] == "new_event"
    assert rebound["path"] == "/new/event.wav"
    assert rebound["source_normalization"]["policy"] == {
        "mode": "peak_dbfs", "target_dbfs": -3.0
    }
    assert rebound["source_normalization"]["applied_gain_db"] == 6.0
    assert rebound["source_normalization"]["normalization_applied"] is True
    assert rebound["source_normalization"]["applied_once_at"] == "event_pcm_split"
    assert rebound["start_sample"] == 400
    assert rebound["end_sample_exclusive"] == 560
    assert rebound["start_tick"] == 1200
    assert rebound["end_tick_exclusive"] == 1680
    assert rebound["linear_gain"] == 0.15


def test_rebind_request_maps_actor_allowlists_and_keeps_new_qa_fields():
    request = {
        "episode_id": "episode",
        "frame_count": 150,
        "frame_rate_hz": 15,
        "qa_ids": ["QA-new"],
        "qa_sampling": {"items_per_type": 7, "candidate_policy": "new-policy"},
        "sound_pool": "/old-pool.json",
        "sound_selection": {
            "preallocated_sound_asset_ids_by_actor": {
                "source1": ["old_event"], "source2": ["old_speech"]
            }
        },
    }
    rebound = _replay.rebind_request(request, context())
    assert request["sound_pool"] == "/old-pool.json"
    assert rebound["sound_pool"] == "/new-pool.json"
    assert rebound["qa_ids"] == ["QA-new"]
    assert rebound["qa_sampling"] == {"items_per_type": 7, "candidate_policy": "new-policy"}
    assert rebound["sound_selection"]["preallocated_sound_asset_ids_by_actor"] == {
        "source1": ["new_event"], "source2": ["new_speech"]
    }


def test_rebinds_selected_allowlist_and_prepared_audio_id():
    payload = {
        "sound_selection": {
            "selected_sound_asset_ids_by_actor": {
                "source1": ["old_event"],
            },
        },
        "sound_asset_id": "old_speech",
        "prepared_audio_id": "old_speech",
    }
    rebound = _replay.rebind_audio_payload(payload, context())
    assert rebound["sound_selection"]["selected_sound_asset_ids_by_actor"] == {
        "source1": ["new_event"]
    }
    assert rebound["sound_asset_id"] == "new_speech"
    assert rebound["prepared_audio_id"] == "new_speech"


def test_replay_compatibility_allows_qa_changes_but_rejects_capture_changes():
    source = {
        "schema": "request",
        "room_id": "room",
        "source_asset_ids": ["asset_a", "asset_b"],
        "frame_count": 150,
        "frame_rate_hz": 15,
        "sample_rate_hz": 16000,
        "camera": {"fov_deg": 85, "resolution_hw": [720, 1280]},
        "entities": {"total_count": 2},
        "profile": {"event_relation": "sequential"},
        "sampling_policy": "conditioned_static_v2",
        "rir_stride": 5,
        "diffraction": False,
        "max_diffraction_order": 0,
        "source_registry": "/registry.json",
        "room_catalog": str((ROOT / "examples/rooms/packages/catalog.json").resolve()),
        "qa_ids": ["QA-old"],
        "qa_sampling": {"items_per_type": 1},
    }
    requested = deepcopy(source)
    requested["qa_ids"] = ["QA-new"]
    requested["qa_sampling"] = {"items_per_type": 4}
    requested["rir_stride"] = 3
    requested["diffraction"] = True
    requested["max_diffraction_order"] = 2
    requested["source_registry"] = str(ROOT / "examples/runtime/source_asset_runtime_profiles.json")
    _replay.validate_replay_request_compatibility(
        source, requested,
        {"clock": {"frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000}},
        ROOT / "examples/rooms/packages/catalog.json",
    )
    for field, value in (
        ("frame_count", 180),
        ("room_id", "other-room"),
        ("source_asset_ids", ["asset_a", "asset_c"]),
        ("camera", {"fov_deg": 90, "resolution_hw": [720, 1280]}),
    ):
        changed = deepcopy(source)
        changed[field] = value
        with pytest.raises(ValueError, match=field):
            _replay.validate_replay_request_compatibility(
                source, changed,
                {"clock": {"frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000}},
                ROOT / "examples/rooms/packages/catalog.json",
            )


def test_replay_compatibility_accepts_explicit_valid_registry_path(tmp_path):
    source = {
        "schema": "request",
        "room_id": "room",
        "source_asset_ids": ["asset_a", "asset_b"],
        "frame_count": 150,
        "frame_rate_hz": 15,
        "sample_rate_hz": 16000,
        "camera": {"fov_deg": 85, "resolution_hw": [720, 1280]},
        "entities": {"total_count": 2},
        "profile": {"event_relation": "sequential"},
        "sampling_policy": "conditioned_static_v2",
        "room_catalog": str((ROOT / "examples/rooms/packages/catalog.json").resolve()),
        "source_registry": "/old/registry.json",
    }
    requested = deepcopy(source)
    registry = tmp_path / "explicit_registry.json"
    registry.write_text("{}" + chr(10))
    requested["source_registry"] = str(registry)
    requested["rir_stride"] = 2
    requested["diffraction"] = True
    requested["max_diffraction_order"] = 1
    _replay.validate_replay_request_compatibility(
        source, requested,
        {"clock": {"frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000}},
        ROOT / "examples/rooms/packages/catalog.json",
    )
    assert _replay._effective_source_registry(requested) == str(registry.resolve())


def test_rebind_replay_plan_only_changes_current_audio_fields():
    source_plan = {
        "visual_plan": {"sound_asset_id": "old_event", "prepared_audio_id": "old_event"},
        "historical_metadata": {"sound_asset_id": "old_event"},
        "audio_events": [{"sound_asset_id": "old_event", "path": "/old/event.wav"}],
        "voice_bindings": [{"sound_asset_id": "old_event", "path": "/old/event.wav"}],
    }
    request = {"sound_pool": "/old-pool.json"}
    replay_plan = _replay.rebind_replay_plan(source_plan, request, context())
    assert replay_plan["visual_plan"] == source_plan["visual_plan"]
    assert replay_plan["historical_metadata"] == source_plan["historical_metadata"]
    assert replay_plan["audio_events"][0]["sound_asset_id"] == "new_event"
    assert replay_plan["voice_bindings"][0]["sound_asset_id"] == "new_event"
    assert replay_plan["request"] == request


def test_rebind_manifest_entry_preserves_history_and_original_fields():
    entry = {
        "request": {
            "sound_selection": {
                "selected_sound_asset_ids_by_actor": {"source1": ["old_event"]}
            }
        },
        "source_assignments": [
            {
                "sound_asset_ids": ["old_event"],
                "original_sound_asset_id": "old_event",
            }
        ],
        "historical_request": {"sound_asset_id": "old_event"},
        "visual_plan": {"sound_asset_id": "old_event"},
    }
    rebound = _replay.rebind_manifest_entry(entry, context())
    assert rebound["request"]["sound_selection"]["selected_sound_asset_ids_by_actor"] == {
        "source1": ["new_event"]
    }
    assert rebound["source_assignments"][0]["sound_asset_ids"] == ["new_event"]
    assert rebound["source_assignments"][0]["original_sound_asset_id"] == "old_event"
    assert rebound["historical_request"] == entry["historical_request"]
    assert rebound["visual_plan"] == entry["visual_plan"]


def _write_wav(path, frame_count, channels=1, sample_rate=16000):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * frame_count * channels)


def test_rebind_rejects_actual_pcm_header_mismatch(tmp_path):
    old_path = tmp_path / "old.wav"
    new_path = tmp_path / "new.wav"
    _write_wav(old_path, frame_count=4)
    _write_wav(new_path, frame_count=5)
    ctx = context()
    ctx["mapping"]["old_event"].update(
        old_pool_path=str(old_path),
        new_pool_path=str(new_path),
    )
    ctx["target_sounds"]["new_event"]["path"] = str(new_path)
    with pytest.raises(ValueError, match="header sample_count"):
        _replay.rebind_audio_payload(
            {
                "sound_asset_id": "old_event",
                "path": str(old_path),
                "sample_count": 4,
                "sample_rate_hz": 16000,
            },
            ctx,
        )


def test_reuse_rejects_changed_audio_but_allows_unchanged_speech():
    report = {
        "events": [{
            "event_id": "event_001",
            "voice_binding_actor_id": "source1",
            "sound_asset_id": "old_event",
            "linear_gain": 1.0,
        }],
        "inputs": {"dry_assets": {
            "old_event": {"path": "/old/event.wav"},
        }},
        "gain_application": {"post_assembly_convolution_gain": 0.5},
    }
    old_event = [{
        "event_id": "event_001",
        "actor_id": "source1",
        "sound_asset_id": "old_event",
        "path": "/old/event.wav",
        "linear_gain": 1.0,
    }]
    _replay.validate_reusable_audio_inputs(old_event, report, 0.5)
    with pytest.raises(ValueError, match="sound ID differs"):
        _replay.validate_reusable_audio_inputs(
            [{**old_event[0], "sound_asset_id": "new_event", "path": "/new/event.wav"}],
            report,
            0.5,
        )
    speech_report = deepcopy(report)
    speech_report["events"][0]["sound_asset_id"] = "new_speech"
    speech_report["inputs"]["dry_assets"] = {
        "new_speech": {"path": "/new/speech.wav"},
    }
    _replay.validate_reusable_audio_inputs(
        [{**old_event[0], "sound_asset_id": "new_speech", "path": "/new/speech.wav"}],
        speech_report,
        0.5,
    )


def test_rebind_rejects_unknown_and_length_changed_audio():
    with pytest.raises(ValueError, match="absent from replay map"):
        _replay.rebind_audio_payload({"sound_asset_id": "unknown"}, context())
    bad = context()
    bad["mapping"]["old_event"]["new_sample_count"] = 192
    with pytest.raises(ValueError, match="changes sample_count"):
        _replay.rebind_audio_payload({"sound_asset_id": "old_event", "sample_count": 160}, bad)
