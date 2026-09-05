"""Dataset-layer remaining work: clocks, event-pool files, segment2 audio, extra pixels."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools" / "qa"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(REPO / "src"))

import qa_v3_clocks as CLOCKS  # noqa: E402
import qa_v3_event_pool as EVENT_POOL  # noqa: E402
from design_qa_v3_extended_profile import (  # noqa: E402
    CARD11_BINDING_FRAME,
    CARD11_EVENT_START_SAMPLE,
    _program_events,
    _resource_inventory,
    _runtime_descriptions,
)
from avengine.qa.runtime_artifacts import (  # noqa: E402
    PIXEL_PRODUCER_KINDS,
    registered_pixel_producer,
)


def _repo_json(path: str):
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def test_event_starts_scale_with_clip_seconds_not_a_75_frame_constant():
    starts_75 = CLOCKS.event_start_samples(
        {"CLIP_SECONDS": 5.0, "SAMPLE_RATE_HZ": 16000, "FRAME_COUNT": 75, "VIDEO_FPS": 15}
    )
    starts_150 = CLOCKS.event_start_samples(
        {"CLIP_SECONDS": 10.0, "SAMPLE_RATE_HZ": 16000, "FRAME_COUNT": 150, "VIDEO_FPS": 15}
    )
    assert starts_75 == (8000, 24000, 40000, 56000)
    assert starts_150 == (16000, 48000, 80000, 112000)
    assert starts_150[0] == 2 * starts_75[0]


def test_card11_default_clock_matches_the_historical_binding():
    assert CLOCKS.card11_binding_frame() == CARD11_BINDING_FRAME
    assert CLOCKS.card11_event_start_sample() == CARD11_EVENT_START_SAMPLE
    start = CLOCKS.card11_event_start_sample(
        {"CLIP_SECONDS": 10.0, "SAMPLE_RATE_HZ": 16000, "FRAME_COUNT": 150, "VIDEO_FPS": 15},
        binding_frame=60,
    )
    samples_per_frame = 16000 / 15
    first = int(start // samples_per_frame)
    last_excl = int(-(-(start + 4800) // samples_per_frame))
    assert first <= 60 < last_excl


def test_binding_frames_scale_from_an_explicit_clock_reference():
    profile = {
        "binding_frames": [12, 74],
        "clock_reference": {"frame_count": 75},
    }
    frames = CLOCKS.scaled_binding_frames(
        profile,
        {"FRAME_COUNT": 150, "VIDEO_FPS": 15, "CLIP_SECONDS": 10.0},
    )
    assert frames[0] == 24
    assert frames[-1] == 149
    assert CLOCKS.last_frame_index({"FRAME_COUNT": 150}) == 149


def test_canary_registry_still_reports_card12_and_speech_shortfalls():
    assets = _repo_json("examples/runtime/source_asset_runtime_profiles.json")["assets"]
    sounds = _repo_json("examples/registry/registries/sound_assets_v1.json")["sound_assets"]
    assert _resource_inventory("card12", assets, sounds)["missing"] == [
        "four_registered_semantic_sound_types"
    ]
    assert _resource_inventory("card13", assets, sounds)["missing"] == [
        "transcribed_speech_assets"
    ]
    assert len(sounds) == 3


def test_event_pool_requires_each_selected_audio_file(tmp_path: Path):
    wav = tmp_path / "bark.wav"
    wav.write_bytes(b"RIFF")
    missing = tmp_path / "gone.wav"
    pool = {
        "clips": [
            {
                "sound_asset_id": "bark_ok",
                "event_class": "dog_bark",
                "prepared": "bark.wav",
            },
            {
                "sound_asset_id": "chime_missing",
                "event_class": "chime",
                "prepared": "gone.wav",
            },
        ]
    }
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(pool), encoding="utf-8")
    with pytest.raises(EVENT_POOL.EventPoolError, match="missing"):
        EVENT_POOL.configured_sound_types(
            {
                "SOUND_EVENT_POOL": str(pool_path),
                "SOUND_EVENT_LIBRARY_ROOT": str(tmp_path),
            },
            required=2,
        )
    present_only = {
        "clips": [pool["clips"][0], {
            "sound_asset_id": "meow_ok",
            "event_class": "cat_meow",
            "prepared": "bark.wav",
        }, {
            "sound_asset_id": "chime_ok",
            "event_class": "chime",
            "prepared": "bark.wav",
        }, {
            "sound_asset_id": "bell_ok",
            "event_class": "alarm_bell",
            "prepared": "bark.wav",
        }]
    }
    present_path = tmp_path / "present.json"
    present_path.write_text(json.dumps(present_only), encoding="utf-8")
    types = EVENT_POOL.configured_sound_types(
        {
            "SOUND_EVENT_POOL": str(present_path),
            "SOUND_EVENT_LIBRARY_ROOT": str(tmp_path),
            "CARD12_SOUND_EVENT_CLASSES": ["dog_bark", "cat_meow", "chime", "alarm_bell"],
        },
        required=4,
    )
    assert [item["taxonomy_path"][-1] for item in types] == [
        "dog_bark", "cat_meow", "chime", "alarm_bell"
    ]
    assert all(Path(item["audio_path"]).is_file() for item in types)


def test_card12_program_uses_four_configured_sound_ids():
    sounds = [
        {"sound_asset_id": f"sound_{label}", "taxonomy_path": [label]}
        for label in ("dog_bark", "cat_meow", "chime", "alarm_bell")
    ]
    main, gatea, truth = _program_events("card12", 0, sounds)
    assert [event[2] for event in main] == [item["sound_asset_id"] for item in sounds]
    assert main[0][2] != gatea[0][2]
    assert sorted(event[2] for event in main) == sorted(event[2] for event in gatea)


def test_card17_declares_segment2_audio_instead_of_reusing_main(tmp_path: Path):
    (tmp_path / "timeline.json").write_text("{}", encoding="utf-8")
    (tmp_path / "timeline_segment2.json").write_text("{}", encoding="utf-8")
    (tmp_path / "actor_selection.json").write_text("{}", encoding="utf-8")
    (tmp_path / "actor_selection_gateB.json").write_text("{}", encoding="utf-8")
    (tmp_path / "timeline_gateB.json").write_text("{}", encoding="utf-8")
    profile = {
        "id": "card17",
        "segment_count": 2,
        "segment_audio_variants": {"segment1": "main", "segment2": "segment2"},
        "runtime_consumer_status": "declared_cross_segment_identity_location",
    }
    result = _runtime_descriptions(profile, tmp_path)
    assert [row["id"] for row in result["segments"]] == ["segment1", "segment2"]
    assert result["release_media"][0]["audio_variant"] == "main"
    assert result["release_media"][1]["audio_variant"] == "segment2"
    assert result["release_media"][1]["status"] == "pending"
    assert result["release_media"][1]["status"] != "pending_audio_consumer"
    assert result["runtime_consumer_status"] == "declared_cross_segment_identity_location"


def test_pixel_producer_is_declared_for_pixel_cards_and_is_registered():
    profiles = {item["id"]: item for item in _repo_json("examples/qa/qa_v3_current_profiles_v1.json")}
    for profile_id in ("card11", "card15a", "card16"):
        assert profiles[profile_id]["pixel_producer_kind"] == "qa_v3_timeline_native_pixel"
        assert profiles[profile_id]["pixel_consumer_kind"] == "qa_v3_extended_pixel"
    assert profiles["card17"]["segment_audio_variants"]["segment2"] == "segment2"
    assert "qa_v3_timeline_native_pixel" in PIXEL_PRODUCER_KINDS
    tool = registered_pixel_producer("qa_v3_timeline_native_pixel")
    assert tool is not None
    assert tool.name == "capture_qa_v3_timeline_pixel.py"
    assert registered_pixel_producer("not_a_kind") is None


def test_pipeline_unions_declared_segment2_audio_and_does_not_infer_it_from_main(
    tmp_path: Path, monkeypatch
):
    sys.path.insert(0, str(TOOLS))
    import run_qa_v3_pipeline as pipeline

    point = tmp_path / "card17_001"
    point.mkdir()
    for name in (
        "timeline.json",
        "timeline_segment2.json",
        "actor_selection.json",
        "actor_selection_gateB.json",
        "timeline_gateB.json",
        "fact_record.json",
    ):
        if name == "fact_record.json":
            continue
        (point / name).write_text("{}", encoding="utf-8")
    fact = {
        "visual_variants": [
            {"id": "main", "actor_selection": "actor_selection.json", "timeline": "timeline.json"},
            {"id": "gateB", "actor_selection": "actor_selection_gateB.json", "timeline": "timeline_gateB.json"},
        ],
        "segments": [
            {"id": "segment1", "variant": "main", "actor_selection": "actor_selection.json", "timeline": "timeline.json"},
            {
                "id": "segment2",
                "variant": "main",
                "actor_selection": "actor_selection.json",
                "timeline": "timeline_segment2.json",
            },
        ],
        "release_media": [
            {"id": "segment1", "variant": "main", "segment": "segment1", "audio_variant": "main", "release": True},
            {"id": "segment2", "variant": "main", "segment": "segment2", "audio_variant": "segment2", "release": True},
        ],
    }
    (point / "fact_record.json").write_text(json.dumps(fact), encoding="utf-8")
    request = {"audio_variants": ["main", "gateA"]}
    variants = pipeline._audio_variants_for_pair(request, tmp_path, ["card17_001"])
    assert variants == ["main", "gateA", "segment2"]
    mapping = pipeline._audio_capture_by_variant(
        tmp_path, tmp_path / "pair", ["card17_001"], ["main", "gateA", "segment2"]
    )
    assert mapping["card17_001"]["main"].endswith("capture/card17_001")
    assert "segment/segment2" in mapping["card17_001"]["segment2"]
    assert mapping["card17_001"]["segment2"] != mapping["card17_001"]["main"]


def test_produced_native_pixel_truth_is_consumed_without_runtime_override(
    monkeypatch, tmp_path: Path
):
    sys.path.insert(0, str(TOOLS))
    import run_qa_v3_pipeline as pipeline

    point = tmp_path / "card11_001"
    point.mkdir()
    (point / "actor_selection.json").write_text("{}", encoding="utf-8")
    (point / "timeline.json").write_text(
        json.dumps({"render": {"frame_count": 75}, "frames": [{} for _ in range(75)]}),
        encoding="utf-8",
    )
    fact = {
        "pixel_evidence": [{"id": "main", "kind": "qa_v3_extended_pixel"}],
        "pixel_producers": [{
            "id": "main",
            "kind": "qa_v3_timeline_native_pixel",
            "actor_selection": "actor_selection.json",
            "timeline": "timeline.json",
            "binding_frames": [30],
        }],
    }
    (point / "fact_record.json").write_text(json.dumps(fact), encoding="utf-8")
    produced = tmp_path / "pair" / "declared_pixels" / "card11_001" / "main"
    produced.mkdir(parents=True)
    truth = produced / "pixel_visibility_truth.json"
    truth.write_text(json.dumps({"status": "computed"}), encoding="utf-8")
    (produced / "native_depth_and_object_ids.npz").write_bytes(b"npz")
    (produced / "rgb_frames").mkdir()
    params = tmp_path / "params.json"
    params.write_text("{}", encoding="utf-8")

    seen = {}

    def fake_run_logged(label, command, log_path, timeout):
        seen["command"] = list(command)
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
        return {"status": "complete"}

    monkeypatch.setattr(pipeline, "_run_logged", fake_run_logged)
    result = pipeline._run_declared_pixels(
        {},
        "apartment_0000",
        "card11",
        tmp_path,
        tmp_path / "pair",
        ["card11_001"],
        params_path=params,
        resume_only=False,
    )
    assert result["status"] == "complete"
    assert result["records"][0]["pixel_truth"] == str(truth.resolve())
    assert "--pixel-truth" in seen["command"]
    assert str(truth.resolve()) in seen["command"]


def test_real_1374_registry_audio_files_exist():
    registry = Path(
        "/data/avengine_external/assets/sound_event_library_v1_20260903_v2/"
        "sound_asset_registry_v1.json"
    )
    if not registry.is_file():
        pytest.skip("external 1374 sound-event registry is not mounted")
    types = EVENT_POOL.configured_sound_types(
        {
            "SOUND_EVENT_REGISTRY": str(registry),
            "CARD12_SOUND_EVENT_CLASSES": ["dog_bark", "cat_meow", "chime", "alarm_bell"],
        },
        required=4,
    )
    assert len(types) == 4
    assert {item["taxonomy_path"][-1] for item in types} == {
        "dog_bark", "cat_meow", "chime", "alarm_bell"
    }
    assert all(Path(item["audio_path"]).is_file() for item in types)
