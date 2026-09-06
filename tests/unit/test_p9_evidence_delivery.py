from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.rooms.qa_delivery import (
    _build_habitat_audio_command,
    _write_habitat_audio_program,
    finalize_qa_episode,
)
from avengine.rooms.qa_evidence import build_pixel_appearance_review, derive_actor_occluders
from avengine.spatial_audio.audio import write_float32_wav


def _write_truth(root: Path, *, frame_indices: list[int], actor_ids=("source1", "source2")) -> None:
    semantic = {actor_ids[0]: 11, actor_ids[1]: 22}
    truth = {
        "schema": "avengine_qa_pixel_visibility_truth_v1",
        "status": "computed_modal_target_only_v1",
        "renderer_backend": "test",
        "rgb_renderer_backend": "test",
        "resolution_hw": [4, 4],
        "frame_indices": frame_indices,
        "camera_pose_ids": [f"camera:{index}" for index in frame_indices],
        "semantic_id_namespace": "test",
        "per_instance": {
            actor: {
                "semantic_id": sid,
                "frames": [
                    {
                        "frame_index": index,
                        "state": "visible_clear",
                        "target_bbox_xyxy_px": [0, 0, 4, 4],
                        "target_pixels": 8,
                        "visible_pixels": 8,
                        "visible_fraction": 1.0,
                        "occlusion_fraction": 0.0,
                    }
                    for index in frame_indices
                ],
            }
            for actor, sid in semantic.items()
        },
    }
    (root / "pixel_visibility_truth.json").write_text(json.dumps(truth), encoding="utf-8")
    frames = len(frame_indices)
    modal = np.zeros((frames, 4, 4), dtype=np.uint32)
    modal[:, :, :2] = semantic[actor_ids[0]]
    modal[:, :, 2:] = semantic[actor_ids[1]]
    np.savez(
        root / "native_pixel_masks_depth_authority_v1.npz",
        depth_derived_modal_semantic=modal,
        modal=modal,
        **{f"target_only_{actor}": modal == sid for actor, sid in semantic.items()},
    )


def _neutral(root: Path, *, frame_indices: list[int]) -> dict:
    clock = {
        "frame_count": 3,
        "frame_rate_hz": 3,
        "sample_rate_hz": 16_000,
        "sample_count": 16_000,
        "time_base_hz": 6,
        "ticks_per_frame": 2,
    }
    rows = {
        actor: [
            {
                "frame_index": index,
                "pts_ticks": index * 2,
                "root": [float(offset), 0.0, -2.0],
                "emitter": [float(offset), 1.0, -2.0],
                "moving": False,
            }
            for index in range(3)
        ]
        for actor, offset in (("source1", -1.0), ("source2", 1.0))
    }
    camera = [
        {
            "frame_index": index,
            "pts_ticks": index * 2,
            "position_m": [0.0, 1.5, 0.0],
            "basis": {
                "forward": [0.0, 0.0, -1.0],
                "right": [1.0, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0],
            },
        }
        for index in range(3)
    ]
    return {
        "schema": "avengine_neutral_readback_v1",
        "clock": clock,
        "coordinate_frame": {"linear_unit": "meter", "up_axis": "+Y", "handedness": "right"},
        "camera": camera,
        "entities": rows,
        "producer": {"module": "test_p9", "source_readbacks": [str(root / "native_readbacks.json")]},
    }


def test_occluders_support_sparse_contract_frames_without_filling_gaps(tmp_path: Path) -> None:
    frame_indices = [0, 2]
    truth = {
        "schema": "avengine_qa_pixel_visibility_truth_v1",
        "status": "computed_modal_target_only_v1",
        "frame_indices": frame_indices,
        "per_instance": {
            "source1": {
                "semantic_id": 11,
                "frames": [{"frame_index": i, "state": "visible_occluded"} for i in frame_indices],
            },
            "source2": {
                "semantic_id": 22,
                "frames": [{"frame_index": i, "state": "visible_clear"} for i in frame_indices],
            },
        },
    }
    modal = np.zeros((2, 2, 2), dtype=np.uint32)
    modal[:, 0, 0] = 22
    target = np.zeros_like(modal)
    target[:, 0, 0] = 1
    np.savez(
        tmp_path / "masks.npz",
        modal=modal,
        target_only_source1=target,
        target_only_source2=(modal == 22),
    )
    result = derive_actor_occluders(tmp_path / "masks.npz", truth, minimum_covered_pixels=1)
    assert [row["frame_index"] for row in result["frame_records"]] == [0, 2]
    assert all(row["occluder_instance_ids"] == ["source2"] for row in result["frame_records"])


def test_appearance_review_uses_actual_masked_rgb_for_animal_and_device(tmp_path: Path) -> None:
    _write_truth(tmp_path, frame_indices=[0, 1])
    rgb = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    rgb[:, :, :2] = [25, 25, 25]
    rgb[:, 0, 0] = [230, 230, 230]
    rgb[:, 0, 1] = [220, 220, 220]
    rgb[:, 1, 0] = [120, 70, 40]
    rgb[:, 1, 1] = [110, 65, 35]
    rgb[:, :, 2:] = [28, 18, 10]
    np.save(tmp_path / "rgb.npy", rgb)
    plan = {
        "clock": {"frame_count": 2, "frame_rate_hz": 2, "sample_rate_hz": 4, "sample_count": 4},
        "visual_plan": {
            "actors": [
                {
                    "actor_id": "source1",
                    "asset_id": "beagle",
                    "entity_class": "articulated_animal",
                    "realized_attributes": {"coat_profile": {"value": "standard_tricolor"}},
                },
                {
                    "actor_id": "source2",
                    "asset_id": "speaker",
                    "entity_class": "rigid_object",
                    "realized_attributes": {"finish": "black_ash"},
                },
            ]
        },
    }
    result = build_pixel_appearance_review(tmp_path, plan, frame_stride=1)
    assert result["actors"]["source1"]["status"] == "reviewed"
    assert result["actors"]["source1"]["appearance_source"] == "realized_attributes"
    assert result["actors"]["source2"]["status"] == "reviewed"
    assert result["actors"]["source2"]["attribute_field"] == "finish"



def _write_finalize_fixture(root: Path, *, source_name: str) -> Path:
    capture = root / "capture"
    plan_dir = root / "plan"
    capture.mkdir(parents=True)
    plan_dir.mkdir()
    frame_indices = [0, 1, 2]
    _write_truth(capture, frame_indices=frame_indices)
    neutral = _neutral(capture, frame_indices=frame_indices)
    (capture / "neutral_readback.json").write_text(json.dumps(neutral), encoding="utf-8")
    (capture / source_name).write_text(json.dumps({"source": source_name}), encoding="utf-8")
    rgb = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    rgb[:, :, :2] = [25, 25, 25]
    rgb[:, 0, 0] = [230, 230, 230]
    rgb[:, 0, 1] = [220, 220, 220]
    rgb[:, 1, 0] = [120, 70, 40]
    rgb[:, 1, 1] = [110, 65, 35]
    rgb[:, :, 2:] = [28, 18, 10]
    np.save(capture / "rgb.npy", rgb)
    clock = neutral["clock"]
    actors = [
        {"actor_id": "source1", "asset_id": "beagle", "entity_class": "articulated_animal", "realized_attributes": {"coat_profile": {"value": "standard_tricolor"}}, "source_endpoint_id": "source1_muzzle"},
        {"actor_id": "source2", "asset_id": "speaker", "entity_class": "rigid_object", "realized_attributes": {"finish": "black_ash"}, "source_endpoint_id": "source2_muzzle"},
    ]
    plan = {
        "kind": "test_p9_plan",
        "episode_id": "same-contract-episode",
        "seed": 5,
        "clock": clock,
        "scene": {"room_id": "test-room", "scene_id": "test-scene"},
        "actors": actors,
        "visual_plan": {"actors": actors, "frames": []},
        "source_endpoint_bindings": [{"source_endpoint_id": a["source_endpoint_id"], "actor_id": a["actor_id"]} for a in actors],
        "resources": {},
    }
    (plan_dir / "episode_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (root / "request.json").write_text(json.dumps({"qa_ids": ["QA-01", "QA-02"]}), encoding="utf-8")
    audio_dir = root / "audio"
    audio_dir.mkdir()
    mixture = audio_dir / "mixture.wav"
    stems = {}
    write_float32_wav(mixture, np.ones((2, 16_000), dtype=np.float64) * 0.1, 16_000)
    for actor in ("source1", "source2"):
        path = audio_dir / f"{actor}.wav"
        write_float32_wav(path, np.ones((2, 16_000), dtype=np.float64) * 0.05, 16_000)
        stems[actor] = str(path)
    program = {
        "schema": "avengine_m6_audio_program_v1",
        "program_id": "test",
        "revision": "v1",
        "mode": "sequential_sources",
        "timeline": {"time_base_hz": 6, "ticks_per_frame": 2, "video_fps": 3, "frame_count": 3, "sample_rate_hz": 6, "sample_count": 6},
        "candidate_source_endpoint_ids": ["source1_muzzle", "source2_muzzle"],
        "events": [
            {"event_id": "e1", "source_endpoint_id": "source1_muzzle", "sound_asset_id": "s1", "start_sample": 0, "end_sample_exclusive": 2, "start_tick": 0, "end_tick_exclusive": 2, "source_start_sample": 0, "source_end_sample_exclusive": 2},
            {"event_id": "e2", "source_endpoint_id": "source2_muzzle", "sound_asset_id": "s2", "start_sample": 2, "end_sample_exclusive": 4, "start_tick": 2, "end_tick_exclusive": 4, "source_start_sample": 0, "source_end_sample_exclusive": 2},
        ],
        "source_specific_stems": True,
        "admission_state": "research",
    }
    program_path = root / "audio_program.json"
    program_path.write_text(json.dumps(program), encoding="utf-8")
    report = {
        "schema": "test_audio_report",
        "clock": clock,
        "mixture_path": str(mixture),
        "audio_program": {"path": str(program_path)},
        "audio": {"sample_rate_hz": 16_000, "sample_count": 16_000, "layouts": ["binaural"]},
        "events": [
            {"event_id": "e1", "actor_id": "source1", "source_endpoint_id": "source1_muzzle", "output_stem": stems["source1"], "wet_tail_interval": [0, 3]},
            {"event_id": "e2", "actor_id": "source2", "source_endpoint_id": "source2_muzzle", "output_stem": stems["source2"], "wet_tail_interval": [2, 5]},
        ],
    }
    audio_report = root / "audio_report.json"
    audio_report.write_text(json.dumps(report), encoding="utf-8")
    return audio_report


def test_finalize_uses_same_contract_bundle_for_ue_and_habitat_names(tmp_path: Path) -> None:
    ue = tmp_path / "ue"
    habitat = tmp_path / "habitat"
    ue_report = _write_finalize_fixture(ue, source_name="frame_readbacks.json")
    habitat_report = _write_finalize_fixture(habitat, source_name="frame_records.json")
    ue_result = finalize_qa_episode(ue, tmp_path / "ue-derived", repository=Path(__file__).resolve().parents[2], audio_report=ue_report)
    habitat_result = finalize_qa_episode(habitat, tmp_path / "hab-derived", repository=Path(__file__).resolve().parents[2], audio_report=habitat_report)
    assert ue_result["preview_status"]["mux"]["video"]["frame_count"] == 3
    assert ue_result["preview_status"]["mux"]["audio"]["channels"] == 2
    assert ue_result["preview_status"]["mux"]["audio"]["sample_rate_hz"] == 16_000
    assert Path(ue_result["preview"]).is_file()
    ue_facts = json.loads(Path(ue_result["facts"]).read_text())
    habitat_facts = json.loads(Path(habitat_result["facts"]).read_text())
    for facts in (ue_facts, habitat_facts):
        assert facts["audio"]["path"].endswith("mixture.wav")
        assert {key: value["appearance"]["value"] for key, value in facts["actors"].items()} == {
            "source1": "standard_tricolor", "source2": "black_ash"
        }
        assert facts["source_paths"]["frame_readbacks"].endswith(("frame_readbacks.json", "frame_records.json"))
    ue_facts.pop("source_paths")
    habitat_facts.pop("source_paths")

    def without_paths(value):
        if isinstance(value, dict):
            return {
                key: without_paths(item)
                for key, item in value.items()
                if not (key.endswith("_path") or key in {"path", "masks_path", "frame_path", "source_path"})
            }
        if isinstance(value, list):
            return [without_paths(item) for item in value]
        return value

    assert without_paths(ue_facts) == without_paths(habitat_facts)



def test_habitat_controller_flow_materializes_audio_program_from_common_plan(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    source_root = Path(
        "/data/datasets/avengine_workspaces/multi_home_activity_20260905/root/"
        "p5_sampler_20260906_v1/mp3d_shared_plan_v5"
    )
    plan = json.loads((source_root / "plan/episode_plan.json").read_text())
    request = json.loads((source_root / "request.json").read_text())
    neutral = json.loads((source_root / "capture/neutral_readback.json").read_text())
    program_path = _write_habitat_audio_program(
        plan,
        tmp_path / "audio_program.json",
        neutral_readback=neutral,
    )
    with pytest.raises(ValueError, match="authoritative source endpoint"):
        _write_habitat_audio_program(plan, tmp_path / "unbound.json")
    track_plan = copy.deepcopy(plan)
    track_plan["actor_tracks"] = [
        {"source_slot_id": "source1", "source_endpoint_id": "source1_emitter"},
        {"source_slot_id": "source2", "source_endpoint_id": "source2_emitter"},
    ]
    track_program = json.loads(
        _write_habitat_audio_program(track_plan, tmp_path / "track.json").read_text()
    )
    assert [event["source_endpoint_id"] for event in track_program["events"]] == [
        "source2_emitter", "source1_emitter"
    ]
    program = json.loads(program_path.read_text())
    assert [event["source_endpoint_id"] for event in program["events"]] == [
        "source2_emitter", "source1_emitter"
    ]
    assert all("path" not in event for event in program["events"])
    assert [event["source_end_sample_exclusive"] for event in program["events"]] == [
        65280, 63520
    ]
    command = _build_habitat_audio_command(
        request,
        plan,
        source_root,
        source_root / "capture",
        tmp_path / "audio",
        program_path,
        repository=repository,
    )
    assert command[:3] == [command[0], "-m", "avengine.cli"]
    assert "render-current-mp3d-dynamic-audio" in command
    assert "--neutral-readback" in command
    assert any(value.startswith("prepared_speech_band_") for value in command)


def test_human_shirt_palette_does_not_compete_with_bare_arm_skin_color():
    from avengine.rooms.qa_evidence import inspect_registered_appearance
    rgb = np.zeros((200, 100, 3), dtype=np.uint8)
    rgb[:] = [170, 105, 65]
    rgb[:, 10:50] = [30, 70, 180]
    mask = np.ones((200, 100), dtype=bool)
    blue = inspect_registered_appearance(
        rgb, mask, "blue", entity_kind="human", target_bbox=[0, 0, 100, 200])
    assert blue["status"] == "pass" and blue["observed_value"] == "blue"
    assert blue["minimum_color_pixels"] == 512
    wrong = inspect_registered_appearance(
        rgb, mask, "green", entity_kind="human", target_bbox=[0, 0, 100, 200])
    assert wrong["status"] == "not_observable" and wrong["observed_value"] == "blue"


@pytest.mark.parametrize("value,colors", [
    ("standard_black_white", [[20, 20, 20], [235, 235, 235]]),
    ("standard_red_white", [[155, 80, 40], [235, 235, 235]]),
    ("standard_red", [[155, 80, 40]]),
    ("standard_yellow", [[190, 173, 110]]),
    ("standard_blue", [[115, 125, 135]]),
])
def test_declared_animal_coat_values_use_coat_color_components(value, colors):
    from avengine.rooms.qa_evidence import inspect_registered_appearance
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    for index, color in enumerate(colors):
        rgb[:, index * 20 // len(colors):(index + 1) * 20 // len(colors)] = color
    row = inspect_registered_appearance(rgb, np.ones((20, 20), dtype=bool),
                                       value, entity_kind="animal")
    assert row["status"] == "pass" and row["observed_value"] == value
    assert row["calibration"] == "placeholder_coarse_color_only"


def test_gray_blue_coat_does_not_mean_saturated_blue_and_unknown_value_is_explicit():
    from avengine.rooms.qa_evidence import inspect_registered_appearance
    rgb = np.full((20, 20, 3), [10, 30, 230], dtype=np.uint8)
    mask = np.ones((20, 20), dtype=bool)
    row = inspect_registered_appearance(rgb, mask, "standard_blue", entity_kind="animal")
    assert row["status"] == "not_observable"
    unknown = inspect_registered_appearance(rgb, mask, "unregistered_pattern", entity_kind="animal")
    assert unknown["gap_category"] == "interface_not_implemented"
