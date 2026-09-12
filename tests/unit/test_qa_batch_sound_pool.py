from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from avengine.qa import batch_sound_pool as mod


def fixture(tmp_path, monkeypatch, *, duration=1, header_delta=0, truncated=False):
    rate = 16000
    x = np.sin(2 * np.pi * 300 * np.arange(int(duration * rate)) / rate) * 0.2
    audio = tmp_path / "event.wav"
    sf.write(audio, x, rate, subtype="PCM_16")
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}")
    event_registry = tmp_path / "registry.json"
    event_registry.write_text(json.dumps({"sound_assets": [{
        "sound_asset_id": "bark", "semantic_sound_class": "dog_bark",
        "normalization_policy": {"mode": "future_rms_label", "target_dbfs": -18.5},
        "custom_gain_metadata": {"gain_db": 2.25, "origin": "registry"},
        "dry_audio": {"uri": audio.as_uri(), "sample_count": len(x) + header_delta,
                      "sample_rate_hz": rate, "channel_count": 1}}]}))
    manifest = tmp_path / "events.json"
    manifest.write_text(json.dumps({"library_root": str(tmp_path / "original"),
        "clips": [{"status": "event", "sound_asset_id": "bark", "source": "same_recording.wav",
                    "applied_gain_db": -1.25,
                    "custom_gain_metadata": {"gain_db": 3.5, "origin": "manifest"},
                    "truncated": truncated}]}))
    monkeypatch.setattr(mod, "load_conditioned_sound_pool", lambda *args, **kwargs: [])
    spec = {"prepared_speech_manifest": str(prepared), "sound_event_registry": str(event_registry),
            "sound_event_manifest": str(manifest), "object_sound_classes": {"speaker": ["speech_playback"]},
            "species_sound_classes": {"dog": ["dog_bark"]}}
    registry = {"assets": [{"asset_id": "dog", "entity_class": "articulated_animal",
                            "identity": {"species_id": "dog"}},
                           {"asset_id": "speaker", "entity_class": "rigid_object",
                            "identity": {"object_type": "speaker", "category": "audio_playback"}}]}
    return spec, registry, audio


def test_joined_pool_preserves_pcm_and_uses_explicit_species_mapping(tmp_path, monkeypatch):
    spec, registry, audio = fixture(tmp_path, monkeypatch)
    before = audio.read_bytes()
    result = mod.build_batch_sound_pool(spec, registry)
    assert audio.read_bytes() == before
    item = result["sounds"][0]
    assert item["compatible_asset_ids"] == ["dog"]
    assert item["source_activity_intervals_samples"]
    assert item["sound_identity_id"].endswith("same_recording.wav")
    assert item["activity_calibration"] == "placeholder"
    assert item["activity_is_qa_event_count"] is False


def test_source_normalization_metadata_preserves_policy_gain_peak_and_source(tmp_path, monkeypatch):
    spec, registry, audio = fixture(tmp_path, monkeypatch)
    result = mod.build_batch_sound_pool(spec, registry)
    item = result["sounds"][0]
    metadata = item["source_normalization"]
    expected_peak = float(np.max(np.abs(sf.read(audio, dtype="float64")[0])))

    assert metadata["policy"] == {"mode": "future_rms_label", "target_dbfs": -18.5}
    assert metadata["target_dbfs"] == -18.5
    assert metadata["applied_gain_db"] == -1.25
    assert metadata["measured_peak_source"] == "registered_event_pcm"
    assert metadata["measured_peak_dbfs"] == pytest.approx(20 * np.log10(expected_peak))
    assert metadata["source"]["event_registry_path"] == str((tmp_path / "registry.json").resolve())
    assert metadata["source"]["event_manifest_path"] == str((tmp_path / "events.json").resolve())
    assert metadata["metadata"]["raw"]["custom_gain_metadata"] == {
        "gain_db": 2.25, "origin": "registry"
    }
    assert metadata["metadata"]["related_0"]["custom_gain_metadata"] == {
        "gain_db": 3.5, "origin": "manifest"
    }
    assert item["linear_gain"] == 1.0
    assert item["normalization_applied"] is False


def test_conditioned_loader_preserves_unknown_policy_and_gain_facts(tmp_path):
    audio = tmp_path / "speech" / "clip.wav"
    audio.parent.mkdir()
    sf.write(audio, np.full(16000, 0.125), 16000, subtype="PCM_16")
    manifest = tmp_path / "prepared.json"
    manifest.write_text(json.dumps({
        "prepared_set_id": "prepared_custom",
        "target_peak_dbfs": -3.0,
        "clips": [{
            "status": "prepared",
            "prepared": "speech/clip.wav",
            "prepared_audio_id": "prepared_custom_clip",
            "source_crop_start_sample": 0,
            "facts": {
                "source_rate_hz": 16000,
                "prepared_peak_dbfs": -7.25,
                "applied_gain_db": 2.5,
                "normalization_applied": False,
                "normalization_policy": {
                    "mode": "future_unseen_policy",
                    "target_dbfs": -14.5,
                },
                "custom_gain_metadata": {"gain_db": 1.75},
            },
            "source_activity": {
                "active_duration_s": 1.0,
                "intervals": [{"start_sample": 0, "end_sample_exclusive": 16000}],
            },
            "gender": "M",
            "transcript": "custom source",
            "human_review": {"status": "pending"},
        }],
    }))

    item = mod.load_conditioned_sound_pool(
        json.loads(manifest.read_text()), source_path=manifest
    )[0]
    metadata = item["source_normalization"]
    assert metadata["policy"] == {
        "mode": "future_unseen_policy", "target_dbfs": -14.5
    }
    assert metadata["target_dbfs"] == -14.5
    assert metadata["applied_gain_db"] == 2.5
    assert metadata["normalization_applied"] is False
    assert metadata["measured_peak_dbfs"] == -7.25
    assert metadata["measured_peak_source"] == "facts.prepared_peak_dbfs"
    assert metadata["metadata"]["facts"]["custom_gain_metadata"] == {"gain_db": 1.75}
    assert metadata["metadata"]["container"]["target_peak_dbfs"] == -3.0
    assert metadata["source"]["manifest_path"] == str(manifest.resolve())
    assert item.get("linear_gain") is None
    assert "facts" not in item


def test_registered_header_disagreement_stops_before_preallocation(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch, header_delta=1)
    with pytest.raises(ValueError, match="header differs"):
        mod.build_batch_sound_pool(spec, registry)


def test_truncation_and_too_short_animal_keep_explicit_rejections(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch, truncated=True)
    result = mod.build_batch_sound_pool(spec, registry)
    assert result["sounds"] == []
    assert result["rejected"][0]["reason"] == "registered_event_is_truncated"
    second = tmp_path / "short"
    second.mkdir()
    spec, registry, _ = fixture(second, monkeypatch, duration=0.2)
    result = mod.build_batch_sound_pool(spec, registry)
    assert result["sounds"] == []
    assert result["rejected"][0]["reason"] == "animal_activity_below_declared_placeholder_minimum"


def test_loader_reentry_keeps_source_processing_separate_from_runtime_flags(tmp_path):
    audio = tmp_path / "event.wav"
    sf.write(audio, np.full(8000, 0.125), 16000, subtype="PCM_16")
    origin_manifest = str(tmp_path / "original-events.json")
    pool_path = tmp_path / "batch-pool.json"
    source_metadata = {
        "policy": {"mode": "future_reference", "target_dbfs": -4.0},
        "applied_gain_db": 12.0,
        "normalization_applied": True,
        "measured_peak_dbfs": -4.0,
        "source": {"manifest_path": origin_manifest},
        "metadata": {"raw": {"custom_gain_metadata": {"unit": "source-db"}}},
    }
    item = {
        "sound_asset_id": "new_event",
        "path": str(audio),
        "sound_class": "test_cue",
        "source_activity_intervals_samples": [[0, 8000]],
        "linear_gain": 0.35,
        "normalization_applied": False,
        "source_normalization": source_metadata,
    }
    result = mod.load_conditioned_sound_pool(
        {"sounds": [item]}, source_path=pool_path
    )[0]
    restored = result["source_normalization"]
    assert restored["normalization_applied"] is True
    assert restored["applied_gain_db"] == 12.0
    assert restored["source"]["manifest_path"] == origin_manifest
    assert restored["source"]["input_manifest_path"] == str(pool_path)
    assert restored["metadata"] == source_metadata["metadata"]
    assert result["linear_gain"] == 0.35
    assert item["source_normalization"] == source_metadata
    assert "input_manifest_path" not in source_metadata["source"]


def test_allowlists_come_from_the_shared_capability_definition(tmp_path, monkeypatch):
    """The pool builder and the selectors must agree on one semantic mapping."""
    from avengine.dataset.source_capabilities import (
        assets_accepting_sound_class, normalize_sound_class_config,
    )

    spec, registry, _ = fixture(tmp_path, monkeypatch)
    result = mod.build_batch_sound_pool(spec, registry)
    config = normalize_sound_class_config(spec)
    item = result["sounds"][0]
    assert item["sound_class"] == "dog_bark"
    assert item["compatible_asset_ids"] == assets_accepting_sound_class(
        registry, "dog_bark", config)
    # the speaker is mapped to speech_playback only, so it is not a bark candidate
    assert "speaker" not in item["compatible_asset_ids"]
    assert assets_accepting_sound_class(registry, "speech_playback", config) == ["speaker"]


def test_allowlist_ordering_matches_the_already_produced_pools(tmp_path, monkeypatch):
    """Event allowlists stay sorted and speech allowlists stay in registry order.

    Both orderings are what the produced pools already contain, so sharing one
    definition of the mapping must not rewrite them.
    """
    from avengine.dataset.source_capabilities import (
        assets_accepting_sound_class, normalize_sound_class_config,
    )

    spec, registry, _ = fixture(tmp_path, monkeypatch)
    registry["assets"].append({
        "asset_id": "aaa_last_in_registry_first_alphabetically",
        "entity_class": "articulated_animal", "identity": {"species_id": "dog"},
    })
    registry["assets"].append({
        "asset_id": "aaa_speaker_last_in_registry", "entity_class": "rigid_object",
        "identity": {"object_type": "speaker", "category": "audio_playback"},
    })
    config = normalize_sound_class_config(spec)
    # the shared index itself is registry-ordered
    assert assets_accepting_sound_class(registry, "dog_bark", config) == [
        "dog", "aaa_last_in_registry_first_alphabetically"]

    result = mod.build_batch_sound_pool(spec, registry)
    # the registered-event branch sorts, exactly as the produced pools do
    assert result["sounds"][0]["compatible_asset_ids"] == sorted(
        ["dog", "aaa_last_in_registry_first_alphabetically"])


def test_a_spec_missing_a_semantic_mapping_still_fails_closed(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch)
    del spec["object_sound_classes"]
    with pytest.raises(KeyError, match="object_sound_classes"):
        mod.build_batch_sound_pool(spec, registry)


def a_selection(**overrides):
    """A row shaped like ``sound_segments.pool_row`` actually emits."""
    base = {
        "relative_path": "same_recording.wav",
        "source_crop_start_sample": 0,
        "source_crop_end_sample_exclusive": 16000,
        # a real P25 record carries the exact source length; the fixture recording is
        # not on disk, so the row has to supply it rather than have it guessed
        "source_sample_count": 16000,
        "sample_count": 16000,
        "sample_rate_hz": 16000,
        "selection_authorized": True,
        "crop_authorization": "owner_authorized_activity_segment_selection_20260910",
        "truncated": False,
        "activity_coverage": 0.94,
        "max_internal_silence_s": 0.05,
        "source_activity_intervals_samples": [[0, 15000]],
    }
    base.update(overrides)
    return base


def test_a_declared_clip_budget_filters_but_an_absent_one_filters_nothing(tmp_path, monkeypatch):
    """max_clip_s is a declared filter; omitting it must not exclude long material."""
    spec, registry, _ = fixture(tmp_path, monkeypatch, duration=6)
    capped = dict(spec, max_clip_s=5)
    result = mod.build_batch_sound_pool(capped, registry)
    assert result["sounds"] == []
    assert result["rejected"][0]["reason"] == "registered_event_exceeds_explicit_clip_budget"
    assert result["clip_budget"] == {
        "max_clip_s": 5.0, "mode": "declared_filter",
        "note": result["clip_budget"]["note"],
    }

    uncapped = dict(spec)
    uncapped.pop("max_clip_s", None)
    opened = mod.build_batch_sound_pool(uncapped, registry)
    assert len(opened["sounds"]) == 1
    assert opened["clip_budget"]["max_clip_s"] is None
    assert opened["clip_budget"]["mode"] == "no_declared_length_filter"
    assert "registered_event_exceeds_explicit_clip_budget" not in \
        opened["counts"]["rejections_by_reason"]


def test_a_declared_budget_must_still_be_positive_and_finite(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch)
    for bad in (0, -1, float("inf")):
        with pytest.raises(ValueError, match="positive and finite"):
            mod.build_batch_sound_pool(dict(spec, max_clip_s=bad), registry)


def test_an_unexplained_truncation_is_still_refused(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch, truncated=True)
    result = mod.build_batch_sound_pool(spec, registry)
    assert result["sounds"] == []
    assert result["rejected"][0]["reason"] == "registered_event_is_truncated"
    assert result["segment_selection"]["authorized_count"] == 0


def test_a_verified_segment_selection_explains_the_truncation(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch, truncated=True)
    spec = dict(spec, segment_selections={"same_recording.wav": a_selection()})
    result = mod.build_batch_sound_pool(spec, registry)
    assert len(result["sounds"]) == 1
    sound = result["sounds"][0]
    assert sound["segment_selection"]["processing"] == "authorized_segment_selection"
    assert sound["segment_selection"]["crop_authorization"].startswith(
        "owner_authorized_activity_segment_selection")
    assert sound["segment_selection"]["activity_coverage"] == 0.94
    summary = result["segment_selection"]
    assert summary["declared_count"] == 1
    assert summary["authorized_count"] == 1
    assert summary["owner"] == "P25"
    # no P25 index was supplied, so no read-back status was available to consume
    assert summary["with_p25_readback_status"] == 0
    assert summary["readback_statuses"] == {}


def test_a_selection_for_another_recording_does_not_authorize_this_one(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch, truncated=True)
    spec = dict(spec, segment_selections={
        "some_other_recording.wav": a_selection(relative="some_other_recording.wav")})
    result = mod.build_batch_sound_pool(spec, registry)
    assert result["rejected"][0]["reason"] == "registered_event_is_truncated"


def test_an_unverifiable_selection_does_not_authorize_a_truncation(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch, truncated=True)
    for broken in (
        a_selection(selection_authorized=False),
        a_selection(activity_coverage=None),
        {"relative_path": "same_recording.wav"},
        a_selection(crop_authorization=None),
        a_selection(truncated=True),
    ):
        spec_with = dict(spec, segment_selections={"same_recording.wav": broken})
        result = mod.build_batch_sound_pool(spec_with, registry)
        assert result["rejected"][0]["reason"] == "registered_event_is_truncated"
        assert result["sounds"] == []


def test_a_selection_declared_inline_on_the_manifest_entry_is_honoured(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch, truncated=True)
    manifest_path = Path(spec["sound_event_manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest["clips"][0]["segment_selection"] = a_selection()
    manifest_path.write_text(json.dumps(manifest))
    result = mod.build_batch_sound_pool(spec, registry)
    assert len(result["sounds"]) == 1
    assert result["sounds"][0]["segment_selection"]["processing"] == \
        "authorized_segment_selection"


def test_a_selection_whose_bounds_disagree_with_the_event_is_refused(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch, truncated=True)
    manifest_path = Path(spec["sound_event_manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest["clips"][0].update(start_sample=0, end_sample_exclusive=16000)
    manifest_path.write_text(json.dumps(manifest))
    agreeing = dict(spec, segment_selections={"same_recording.wav": a_selection()})
    assert len(mod.build_batch_sound_pool(agreeing, registry)["sounds"]) == 1

    disagreeing = dict(spec, segment_selections={
        "same_recording.wav": a_selection(source_crop_end_sample_exclusive=9000,
                                          source_sample_count=16000)})
    result = mod.build_batch_sound_pool(disagreeing, registry)
    assert result["rejected"][0]["reason"] == "registered_event_is_truncated"


def test_a_speech_clip_stays_consumed_by_the_prepared_set(tmp_path, monkeypatch):
    """The rejection chain keeps its original precedence, truncated or long."""
    spec, registry, _ = fixture(tmp_path, monkeypatch, duration=6, truncated=True)
    registry_path = Path(spec["sound_event_registry"])
    payload = json.loads(registry_path.read_text())
    payload["sound_assets"][0]["semantic_sound_class"] = "speech_playback"
    registry_path.write_text(json.dumps(payload))
    result = mod.build_batch_sound_pool(dict(spec, max_clip_s=5), registry)
    assert result["rejected"][0]["reason"] == "speech_consumes_existing_P7_prepared_set"


def test_segment_selections_may_be_given_as_a_list_or_a_file(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch, truncated=True)
    as_list = dict(spec, segment_selections=[a_selection()])
    assert len(mod.build_batch_sound_pool(as_list, registry)["sounds"]) == 1

    path = tmp_path / "selections.json"
    path.write_text(json.dumps({"selections": {"same_recording.wav": a_selection()}}))
    as_file = dict(spec, segment_selections=str(path))
    assert len(mod.build_batch_sound_pool(as_file, registry)["sounds"]) == 1

    with pytest.raises(ValueError, match="segment_selections must be"):
        mod.build_batch_sound_pool(dict(spec, segment_selections=7), registry)


def a_p25_index(status="qualified", **row_overrides):
    """A prepare_segments payload: the rows plus P25's own read-back statuses."""
    row = a_selection(segment_id="sound_segment_test_v1", **row_overrides)
    return {"schema": "avengine_sound_segment_index_v1",
            "segments": [row],
            "verifications": [{"segment_id": "sound_segment_test_v1",
                               "status": status}]}


def test_a_p25_index_supplies_both_the_row_and_its_readback_status(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch, truncated=True)
    result = mod.build_batch_sound_pool(
        dict(spec, segment_selections=a_p25_index("qualified")), registry)
    assert len(result["sounds"]) == 1
    assert result["sounds"][0]["segment_selection"]["segment_readback_status"] == "qualified"
    assert result["segment_selection"]["with_p25_readback_status"] == 1
    assert result["segment_selection"]["readback_statuses"] == {"qualified": 1}


def test_a_segment_whose_p25_readback_failed_does_not_authorize_anything(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch, truncated=True)
    result = mod.build_batch_sound_pool(
        dict(spec, segment_selections=a_p25_index("fail")), registry)
    assert result["sounds"] == []
    assert result["rejected"][0]["reason"] == "registered_event_is_truncated"
    assert result["segment_selection"]["readback_statuses"] == {"fail": 1}


def test_a_segment_cut_with_other_processing_than_requested_is_refused_at_the_pool(
        tmp_path, monkeypatch):
    """The requested crop parameters must match the segment actually returned."""
    spec, registry, _ = fixture(tmp_path, monkeypatch, truncated=True)
    payload = a_p25_index("qualified", edge_fade_s=0.0)
    agreeing = mod.build_batch_sound_pool(
        dict(spec, segment_selections=payload,
             segment_processing={"edge_fade_s": 0.0}), registry)
    assert len(agreeing["sounds"]) == 1
    assert agreeing["segment_selection"]["requested_processing"] == {"edge_fade_s": 0.0}

    disagreeing = mod.build_batch_sound_pool(
        dict(spec, segment_selections=payload,
             segment_processing={"edge_fade_s": 0.005}), registry)
    assert disagreeing["sounds"] == []
    assert disagreeing["rejected"][0]["reason"] == "registered_event_is_truncated"


def test_segment_processing_must_be_an_object(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch, truncated=True)
    with pytest.raises(ValueError, match="segment_processing must be an object"):
        mod.build_batch_sound_pool(dict(spec, segment_processing=7), registry)


def test_an_authorized_segment_records_which_bounds_were_checked(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch, truncated=True)
    result = mod.build_batch_sound_pool(
        dict(spec, segment_selections={"same_recording.wav": a_selection()}), registry)
    selection = result["sounds"][0]["segment_selection"]
    bounds = selection["coordinate_bounds"]
    assert bounds["source_sample_count"] == 16000
    assert bounds["source_sample_count_from"] == "selection.source_sample_count"
    assert bounds["row_shape"] == "pool_row"
    assert "not a listening test" in selection["claim_boundary"]
