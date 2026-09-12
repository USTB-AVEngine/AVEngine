"""Manifest builder writes this worktree catalog and full path_bindings."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem + "_h3_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def builder():
    return _load("tools/dataset/build_qa_batch_manifest.py")


@pytest.fixture
def controller():
    return _load("tools/studio/run_qa_episode.py")


def test_declared_catalog_path_is_preserved_without_directory_heuristics(builder, tmp_path, monkeypatch):
    production = builder.production_room_catalog_path()
    monkeypatch.chdir(tmp_path)
    wt_catalog = tmp_path / "wt-production" / "catalog.json"
    wt_catalog.parent.mkdir()
    wt_catalog.write_text(json.dumps({"rooms": []}) + "\n", encoding="utf-8")
    assert builder.resolve_request_room_catalog(str(wt_catalog)) == wt_catalog.resolve()
    assert builder.resolve_request_room_catalog("examples/rooms/packages/catalog.json") == production
    assert builder.resolve_request_room_catalog(None) == production
    assert builder.resolve_request_room_catalog("") == production
    other = tmp_path / "custom_catalog.json"
    other.write_text("{}\n", encoding="utf-8")
    assert builder.resolve_request_room_catalog(str(other)) == other.resolve()
    explicit = tmp_path / "explicit.json"
    explicit.write_text("{}\n", encoding="utf-8")
    assert builder.resolve_request_room_catalog(str(wt_catalog), explicit=explicit) == explicit.resolve()


def test_catalog_resolution_does_not_follow_cwd(builder, tmp_path, monkeypatch):
    decoy = tmp_path / "examples" / "rooms" / "packages"
    decoy.mkdir(parents=True)
    (decoy / "catalog.json").write_text(json.dumps({"rooms": []}) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    resolved = builder.resolve_request_room_catalog("examples/rooms/packages/catalog.json")
    assert resolved == builder.production_room_catalog_path()
    assert resolved != (decoy / "catalog.json").resolve()


def test_stamp_request_writes_absolute_catalog_and_full_bindings(builder):
    catalog = json.loads(builder.production_room_catalog_path().read_text(encoding="utf-8"))
    bindings = builder.catalog_path_bindings(catalog)
    assert "AVENGINE_MULTI_HOME_AUTHORING_ROOT" in bindings
    request = {
        "episode_id": "demo",
        "room_catalog": "/data/jzy/tmp/wt-multi-home-activity-integration/examples/rooms/packages/catalog.json",
        "runtime": {"graphics_adapter": 0, "path_bindings": {"AVENGINE_CUSTOM": "/custom"}},
    }
    builder.stamp_request_catalog(
        request, catalog_path=builder.production_room_catalog_path(), path_bindings=bindings)
    assert request["room_catalog"] == str(builder.production_room_catalog_path())
    assert request["runtime"]["path_bindings"]["AVENGINE_CUSTOM"] == "/custom"
    assert request["runtime"]["path_bindings"]["AVENGINE_MULTI_HOME_ROOT"] == bindings["AVENGINE_MULTI_HOME_ROOT"]
    assert request["runtime"]["path_bindings"]["AVENGINE_MULTI_HOME_AUTHORING_ROOT"] == (
        bindings["AVENGINE_MULTI_HOME_AUTHORING_ROOT"])
    assert request["runtime"]["graphics_adapter"] == 0


def test_request_path_bindings_override_catalog(controller):
    catalog = {"path_bindings": {"AVENGINE_A": "/catalog", "AVENGINE_B": "/catalog-b"}}
    request = {"runtime": {"graphics_adapter": 1, "path_bindings": {"AVENGINE_A": "/request"}}}
    runtime = controller.request_package_runtime(request, catalog)
    assert runtime["path_bindings"]["AVENGINE_A"] == "/request"
    assert runtime["path_bindings"]["AVENGINE_B"] == "/catalog-b"
    assert runtime["graphics_adapter"] == 1


def test_request_without_bindings_uses_catalog(controller):
    catalog = {"path_bindings": {"AVENGINE_A": "/catalog"}}
    runtime = controller.request_package_runtime({"runtime": {"rpc_port": 1}}, catalog)
    assert runtime["path_bindings"] == {"AVENGINE_A": "/catalog"}
    assert runtime["rpc_port"] == 1


def _human(asset_id, color):
    return {"asset_id": asset_id, "revision": "v1", "entity_class": "articulated_human",
            "identity": {"species_id": "human"},
            "realized_attributes": {"sex_or_gender_label": "male", "top_color": color},
            "display_label": asset_id, "default_emitter_anchor_id": "mouth",
            "emitter_anchors": [{"anchor_id": "mouth", "offset_m": [0, 1.6, 0],
                                 "offset_space": "final_scaled_asset_root"}],
            "runtime_backends": {"spear_unreal": {"binding": "existing"},
                                 "habitat": {"resting_pose": {"attachment_surface": "floor",
                                                              "base_plane_offset_m": 0}}}}


def _write_mono_wav(path, *, seconds=2.0, rate=16000):
    """The conditioned pool loader reads real mono PCM, not a declared count."""
    import math
    import wave
    count = int(round(seconds * rate))
    frames = bytearray()
    for index in range(count):
        value = int(12000 * math.sin(2 * math.pi * 220 * index / rate))
        frames += int(value).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(bytes(frames))
    return count


def _speech(sound_id, identity, directory):
    path = Path(directory) / f"{sound_id}.wav"
    count = _write_mono_wav(path)
    return {"sound_asset_id": sound_id, "sound_identity_id": identity,
            "source_pcm_path": f"/source/{identity}.wav", "sound_class": "speech",
            "gender": "male", "sample_rate_hz": 16000, "sample_count": count,
            "active_duration_s": 2, "audible_start_sample": 0,
            "audible_end_sample_exclusive": count,
            "source_activity_intervals_samples": [[0, count]],
            "activity_coordinate": "prepared_clip_samples",
            "transcript": sound_id, "path": str(path)}


def _member(request_id):
    return {"request_id": request_id,
            "instances": [{"instance_id": "source1", "asset_id": "red",
                           "source_class": "articulated_human"},
                          {"instance_id": "source2", "asset_id": "blue",
                           "source_class": "articulated_human"}]}


def test_prepare_cli_accepts_a_production_spec_config(builder, tmp_path, monkeypatch):
    """One `prepare` run turns a V1 config into requests, targets and stage items."""
    catalog = {"path_bindings": {"AVENGINE_TEST_ROOT": "/data/test"},
               "rooms": [{"room_id": "room_a", "family": "apartment", "renderer": "ue_spear"},
                         {"room_id": "room_b", "family": "kujiale", "renderer": "ue_spear"}]}
    registry = {"assets": [_human("red", "red"), _human("blue", "blue"), _human("green", "green")]}
    pcm = tmp_path / "pcm"
    pcm.mkdir()
    sounds = {"sounds": [_speech("one", "speaker1", pcm), _speech("two", "speaker2", pcm),
                         _speech("three", "speaker3", pcm), _speech("four", "speaker4", pcm)]}
    production = {
        "schema": "avengine_v1_production_spec_v1",
        "batch_id": "p01_cli",
        "seed": 11,
        "defaults": {
            "clock": {"frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000},
            "rig": {"resolution_hw": [720, 1280], "fov_deg": 85},
            "reserve_tail_s": 3.0,
            "post_assembly_convolution_gain": 0.5,
            "audio_layouts": [{"type": "binaural", "role": "primary", "indirect_sh_order": 1},
                              {"type": "ambisonics", "ambisonic_order": 1,
                               "role": "attached_view", "indirect_sh_order": 1}],
            "resources": {"graphics_adapter": 0, "capture": {"min_free_vram_mb": 12000}},
            "profile": {"separation_bin_deg": [30, 60], "anchor_count": 1},
            "request_extras": {"binding_motion": {
                "minimum_motion_s": 2.0, "end_hold_s": 0.5, "angle_tolerance_deg": 10,
                "minimum_entity_separation_m": 0.95, "source_start_s": 0.1,
                "walk_speed_range_mps": [0.5, 0.8]}},
            "qa_ids": ["QA-05", "QA-20"],
            "quota_by_qa": {"QA-05": 8, "QA-20": 8},
        },
        "episodes": [{**_member("p01_cli_episode_01"), "room_id": "room_a",
                      "condition_group": "identity_binding"}],
        "core_groups": [{"group_id": "p01_cli_state", "task_family": "cross_time_state",
                         "room_id": "room_b",
                         "members": [_member(f"p01_cli_state_m{index + 1}") for index in range(4)]}],
        "coverage_quota": {"min_main_questions_per_qa_id": 8, "min_worlds_per_qa_id": 2},
    }
    config = {
        "batch_id": "p01_cli",
        "seed": 11,
        "base_request": {
            "camera": {"motion": "static"},
            "room_catalog": str(tmp_path / "catalog.json"),
            "source_registry": str(tmp_path / "registry.json"),
            "runtime": {"graphics_adapter": 0, "rpc_port": 39782},
            "sound_pool": str(tmp_path / "sounds.json"),
            "sound_selection": {"max_clip_s": 5.0},
        },
        "production": production,
    }
    (tmp_path / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    (tmp_path / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    (tmp_path / "sounds.json").write_text(json.dumps(sounds), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(builder, "load_source_asset_runtime_registry",
                        lambda path: json.loads(Path(path).read_text(encoding="utf-8")))
    output = tmp_path / "prepared"
    builder.main(["prepare", "--config", str(config_path), "--output", str(output)])

    manifest = json.loads((output / "batch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["requested_episode_count"] == 5
    assert manifest["production"]["episode_count"] == 1
    assert manifest["production"]["core_group_count"] == 1
    assert manifest["production"]["core_member_count"] == 4
    assert manifest["stage_protocol"]["stages"][:3] == ["plan", "capture", "audio"]

    rows = {row["episode_id"]: row for row in manifest["episodes"]}
    episode = rows["p01_cli_episode_01"]
    assert episode["requested_qa_ids"] == ["QA-05", "QA-20"]
    assert episode["qa_ids_source"] == "config"
    assert episode["requested_quota_by_qa"] == {"QA-05": 8, "QA-20": 8}
    assert episode["requested_quota_source"] == "config"
    assert episode["entity_instance_count"] == 2
    assert episode["distinct_asset_count"] == 2
    assert [target["event"]["kind"] for target in episode["qa_targets"]] == [
        "target_audible_window", "target_audible_window"]
    assert [item["stage"] for item in episode["stage_work_items"]] == ["plan"]
    assert episode["production_request"]["stage_plan"] == ["plan", "capture", "audio", "delivery"]

    member = rows["p01_cli_state_m1"]
    assert member["task_family"] == "cross_time_state"
    assert member["condition_group"] == "post_sound_state"
    assert member["production_request"]["stage_plan"] is None
    assert member["stage_work_items"] == []
    assert member["stage_scope"]["delivering_unit_id"] == "v0_a0"
    group = manifest["production"]["core_groups"][0]
    assert [row["unit_id"] for row in group["stage_units"]] == [
        "v0", "v0_capture", "v0_a0", "v0_a1", "v1", "v1_capture", "v1_a0", "v1_a1", "group"]
    units = {row["unit_id"]: row for row in group["stage_units"]}
    assert units["v0_capture"]["member_request_ids"] == [
        "p01_cli_state_m1", "p01_cli_state_m2"]
    assert units["v1_capture"]["member_request_ids"] == [
        "p01_cli_state_m3", "p01_cli_state_m4"]
    assert units["v0_a0"]["default_resource_kind"] == "cpu_native_acoustic"
    assert group["initial_work_items"][0]["payload"]["audio_layouts"] == [
        {"type": "binaural", "channel_count": 2, "role": "primary", "indirect_sh_order": 1},
        {"type": "ambisonics", "channel_count": 4, "role": "attached_view",
         "ambisonic_order": 1, "indirect_sh_order": 1}]

    # The saved request is the one a controller executes on its own.
    for row in manifest["episodes"]:
        saved = json.loads(Path(row["request_path"]).read_text(encoding="utf-8"))
        assert saved == row["request"]
        assert saved["schema"] == "avengine_native_qa_room_request_v1"
        assert saved["frame_count"] == 150 and saved["frame_rate_hz"] == 15.0
        assert saved["profile"]["reserve_tail_s"] == 3.0
        assert saved["post_assembly_convolution_gain"] == 0.5
        assert saved["qa_ids"] == ["QA-05", "QA-20"]
        assert "qa_targets" not in saved
        assert row["qa_targets"]  # Derived catalog metadata remains available.
        assert saved["entity_instances"]
        assert saved["camera"]["motion"] == "static"
        assert saved["runtime"]["path_bindings"]["AVENGINE_TEST_ROOT"] == "/data/test"
        assert Path(saved["room_catalog"]).is_absolute()
        assert row["controller_entrypoint"].endswith("tools/studio/run_qa_episode.py")
    assert manifest["producer"]["room_catalog"] == str((tmp_path / "catalog.json").resolve())
