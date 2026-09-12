import json
from pathlib import Path

import pytest

from avengine.rooms.qa_delivery import _rendered_sound_registry, _reviewed_occluder_registry


def test_sound_semantics_come_from_matching_rendered_pool_entry(tmp_path):
    audio = tmp_path / "event.wav"
    audio.touch()
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"sounds": [
        {"sound_asset_id": "event", "path": "event.wav", "sound_class": "music_playback"},
        {"sound_asset_id": "unused", "path": "unused.wav", "sound_class": "dog_bark"},
    ]}))
    result = _rendered_sound_registry(
        {"sound_pool": str(pool)},
        {"events": [{"sound_asset_id": "event"}]},
        {"inputs": {"dry_assets": {"event": {"path": str(audio), "sha256": "existing-readback"}}}},
        repository=tmp_path,
    )
    assert result["sound_count"] == 1
    assert result["sounds"][0]["semantic_sound_class"] == "music_playback"
    assert result["sounds"][0]["rendered_pcm_sha256"] == "existing-readback"


def test_sound_registry_rejects_a_different_pcm_for_the_same_id(tmp_path):
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"sounds": [
        {"sound_asset_id": "event", "path": "requested.wav", "sound_class": "music_playback"}
    ]}))
    with pytest.raises(ValueError, match="PCM differs"):
        _rendered_sound_registry(
            {"sound_pool": str(pool)},
            {"events": [{"sound_asset_id": "event"}]},
            {"inputs": {"dry_assets": {"event": {"path": str(tmp_path / "rendered.wav")}}}},
            repository=tmp_path,
        )


def test_legacy_audio_without_input_registry_remains_supported(tmp_path):
    assert _rendered_sound_registry({}, {}, {}, repository=tmp_path)["sounds"] == []


def test_occluder_labels_do_not_append_internal_appearance_values():
    result = _reviewed_occluder_registry(
        {"actors": {
            "animal": {"status": "reviewed", "entity_kind": "animal",
                       "attribute_field": "coat_profile.value", "value": "standard_red"},
            "human": {"status": "reviewed", "entity_kind": "human",
                      "attribute_field": "top_color", "value": "green"},
        }},
        {"animal": {"display_label": "Shiba Inu v2"}, "human": {"display_label": "Human (green top)"}},
    )
    assert result["animal"]["display_label"] == "Shiba Inu"
    assert "standard_red" not in result["animal"]["display_label"]
    assert result["animal"]["appearance_value"] == "standard_red"
    assert result["human"]["display_label_zh"] == "绿色上衣的人"


def test_delivery_uses_an_explicit_asset_registry(tmp_path):
    from avengine.rooms.qa_delivery import _asset_registry
    registry = tmp_path / "alternate-assets.json"
    registry.write_text(json.dumps({"assets": [
        {"asset_id": "new_actor", "display_label": "new observable asset"}
    ]}))
    assert _asset_registry(tmp_path, registry)["new_actor"]["display_label"] == "new observable asset"


def test_habitat_audio_command_honors_configured_rir_stride(tmp_path):
    from avengine.rooms.qa_delivery import _build_habitat_audio_command
    plan = {"resources": {"acoustic_package": str(tmp_path / "package.json")},
            "voice_bindings": [{"sound_asset_id": "event", "path": str(tmp_path / "event.wav")}]}
    for request, expected in [({"rir_stride": 5}, "5"), ({}, "3")]:
        command = _build_habitat_audio_command(request, plan, tmp_path, tmp_path / "capture",
                                              tmp_path / "audio", tmp_path / "program.json",
                                              repository=tmp_path)
        assert command[command.index("--rir-stride-frames") + 1] == expected

def test_source_context_policy_reaches_both_native_room_commands(tmp_path):
    from avengine.rooms.qa_delivery import build_audio_command, _build_habitat_audio_command
    plan = {"resources": {"acoustic_package": str(tmp_path / "package.json")},
            "voice_bindings": [{"sound_asset_id": "event", "path": str(tmp_path / "event.wav")}]}
    request = {"source_context_policy": "independent_states",
               "runtime": {"runtime_prefix": "runtime", "rlr_sdk_root": "rlr",
                           "magnum_python_site": "magnum"}}
    commands = [
        build_audio_command(request, plan, tmp_path, tmp_path/"audio", repository=tmp_path),
        _build_habitat_audio_command(request, plan, tmp_path, tmp_path/"capture",
                                    tmp_path/"audio", tmp_path/"program.json", repository=tmp_path),
    ]
    for command in commands:
        assert command[command.index("--source-context-policy")+1] == "independent_states"
        assert "--rir-cache" not in command
    with pytest.raises(ValueError, match="joint RIR cache"):
        build_audio_command({**request, "rir_cache": str(tmp_path/"retained-cache")},
                            plan, tmp_path, tmp_path/"audio", repository=tmp_path)
    joint = build_audio_command({"runtime": request["runtime"]}, plan, tmp_path,
                                tmp_path/"audio", repository=tmp_path)
    assert "--rir-cache" in joint
    assert "--source-context-policy" not in joint



def test_habitat_repeated_sound_binding_keeps_one_pcm_and_rejects_conflicts(tmp_path):
    from avengine.rooms.qa_delivery import _build_habitat_audio_command
    first = {"sound_asset_id": "shared_clip", "path": str(tmp_path / "shared.wav")}
    plan = {"resources": {"acoustic_package": str(tmp_path / "package.json")},
            "voice_bindings": [first],
            "audio_events": [{**first, "event_id": "first"},
                             {**first, "event_id": "second"}]}
    args = ({"rir_stride": 5}, plan, tmp_path, tmp_path / "capture",
            tmp_path / "audio", tmp_path / "program.json")
    command = _build_habitat_audio_command(*args, repository=tmp_path)
    assert command.count("--asset-binding") == 1
    plan["audio_events"][1]["path"] = str(tmp_path / "different.wav")
    with pytest.raises(ValueError, match="conflicting dry PCM paths"):
        _build_habitat_audio_command(*args, repository=tmp_path)


# --- shared visual scope, shared video master and ancillary audio ---------


def _member_root(tmp_path, *, capture: Path, linkage: bool, symlink: bool) -> Path:
    root = tmp_path / "member"
    root.mkdir(parents=True, exist_ok=True)
    if symlink:
        (root / "capture").symlink_to(capture, target_is_directory=True)
    else:
        (root / "capture").mkdir(exist_ok=True)
    if linkage:
        (root / "native_linkage.json").write_text(
            json.dumps({"member_id": "v0_a1", "native_capture_reused": True}), encoding="utf-8"
        )
    return root


def test_shared_visual_scope_detects_a_reused_capture(tmp_path):
    from avengine.rooms.qa_delivery import _resolve_shared_visual_root

    capture = tmp_path / "visual" / "v0" / "capture"
    capture.mkdir(parents=True)
    root = _member_root(tmp_path, capture=capture, linkage=True, symlink=True)
    derived = tmp_path / "group" / "v0_a1" / "delivery"
    shared, scope = _resolve_shared_visual_root(
        root, derived, capture, shared_visual_root=None, visual_evidence_reuse=None
    )
    assert scope["enabled"] is True
    assert scope["member_id"] == "v0_a1"
    assert len(scope["reasons"]) == 2
    # Members of one group land beside each other, so they share one root.
    assert shared == (tmp_path / "group" / "shared_visual_evidence").resolve()


def test_shared_visual_scope_stays_off_for_a_private_capture(tmp_path):
    from avengine.rooms.qa_delivery import _resolve_shared_visual_root

    capture = tmp_path / "episode" / "capture"
    capture.mkdir(parents=True)
    root = _member_root(tmp_path, capture=capture, linkage=False, symlink=False)
    shared, scope = _resolve_shared_visual_root(
        root, tmp_path / "d" / "e" / "derived", capture,
        shared_visual_root=None, visual_evidence_reuse=None,
    )
    assert shared is None
    assert scope["enabled"] is False
    assert scope["source"] == "capture_is_not_declared_shared_between_audio_members"


def test_explicit_root_and_explicit_disable_both_win(tmp_path):
    from avengine.rooms.qa_delivery import _resolve_shared_visual_root

    capture = tmp_path / "visual" / "v0" / "capture"
    capture.mkdir(parents=True)
    root = _member_root(tmp_path, capture=capture, linkage=True, symlink=True)
    derived = tmp_path / "group" / "v0_a1" / "delivery"
    declared = tmp_path / "declared_shared"
    shared, scope = _resolve_shared_visual_root(
        root, derived, capture, shared_visual_root=declared, visual_evidence_reuse=None
    )
    assert shared == declared.resolve()
    assert scope["source"] == "caller_declared_shared_visual_root"
    off, off_scope = _resolve_shared_visual_root(
        root, derived, capture, shared_visual_root=declared, visual_evidence_reuse=False
    )
    assert off is None
    assert off_scope["enabled"] is False


def _encode_master(path: Path, *, frames: int, rate: float) -> None:
    import numpy as np
    from avengine.rooms.qa_delivery import _encode_rgb_frames_to_video

    values = np.zeros((frames, 16, 16, 3), dtype=np.uint8)
    values[:, :, :8] = 200
    _encode_rgb_frames_to_video(
        values, output_path=path, frame_rate_hz=rate, expected_frame_count=frames
    )


def test_shared_visual_master_is_reused_only_when_the_clock_matches(tmp_path):
    from avengine.rooms.qa_delivery import _prepare_visual_video

    capture = tmp_path / "capture"
    capture.mkdir()
    master = tmp_path / "shared" / "visual_rgb.mp4"
    master.parent.mkdir()
    _encode_master(master, frames=4, rate=2.0)

    matched, status = _prepare_visual_video(
        capture, clock={"frame_count": 4, "frame_rate_hz": 2.0},
        output_path=tmp_path / "private.mp4", shared_master_path=master,
    )
    assert matched == master.resolve()
    assert status["source"] == "shared_visual_master"
    assert status["reused"] is True

    # A master from a different capture must not be accepted for this clock.
    missing, other = _prepare_visual_video(
        capture, clock={"frame_count": 9, "frame_rate_hz": 2.0},
        output_path=tmp_path / "private.mp4", shared_master_path=master,
    )
    assert missing is None
    assert other["status"] == "not_run"


def test_shared_master_is_encoded_once_and_published_atomically(tmp_path):
    import numpy as np
    from avengine.rooms.qa_delivery import _prepare_visual_video

    capture = tmp_path / "capture"
    capture.mkdir()
    np.save(capture / "rgb.npy", np.zeros((4, 16, 16, 3), dtype=np.uint8))
    master = tmp_path / "shared" / "visual_rgb.mp4"
    clock = {"frame_count": 4, "frame_rate_hz": 2.0}

    first, first_status = _prepare_visual_video(
        capture, clock=clock, output_path=tmp_path / "a.mp4", shared_master_path=master,
    )
    assert first == master.resolve()
    assert first_status["published_shared_master"] is True
    assert not list(master.parent.glob(".*tmp.mp4")), "no partial encode is left behind"

    second, second_status = _prepare_visual_video(
        capture, clock=clock, output_path=tmp_path / "b.mp4", shared_master_path=master,
    )
    assert second == master.resolve()
    assert second_status["reused"] is True
    assert not (tmp_path / "b.mp4").exists(), "the second member re-encodes nothing"


def test_ancillary_audio_sits_beside_the_binaural_canonical(tmp_path):
    from avengine.rooms.qa_delivery import _ancillary_audio_outputs

    foa = tmp_path / "foa.wav"
    foa.write_bytes(b"RIFF")
    listed = tmp_path / "listed.wav"
    listed.write_bytes(b"RIFF")
    outputs = _ancillary_audio_outputs({
        "mixture_path": str(tmp_path / "mixture.wav"),
        "ambisonic_path": str(foa),
        "ancillary_outputs": [{"role": "foa_b_format", "path": str(listed)}],
    })
    roles = {row["role"] for row in outputs}
    assert roles == {"ancillary_audio_ambisonic_path", "ancillary_audio_foa_b_format"}
    assert all(row["required"] is False and row["canonical"] is False for row in outputs)
    # A declared path that does not exist is not invented into evidence.
    assert _ancillary_audio_outputs({"foa_path": str(tmp_path / "absent.wav")}) == []
    assert _ancillary_audio_outputs({}) == []
