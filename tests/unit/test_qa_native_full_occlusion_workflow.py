from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from avengine.qa.pixel_visibility import compile_depth_pixel_visibility_truth


REPOSITORY = Path(__file__).resolve().parents[2]


def _load_tool(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECIPE = _load_tool(
    "qa_native_full_occlusion_recipe_tool",
    "tools/qa/build_native_full_occlusion_reappearance_episode.py",
)
AUDIO = _load_tool(
    "qa_native_controlled_audio_tool",
    "tools/qa/build_native_controlled_audio_program.py",
)
BINDER = _load_tool(
    "qa_native_pixel_fact_binding_tool",
    "tools/qa/bind_native_pixel_fact_episode.py",
)
RECOMPILER = _load_tool(
    "qa_native_pixel_truth_recompiler_tool",
    "tools/qa/recompile_native_pixel_truth.py",
)


def _source_recipe_fixture() -> tuple[dict, dict]:
    source_episode = "retained_native_0323"
    paths = [[float(index), 0.0, 0.0] for index in range(75)]
    facts = {
        "episode_id": source_episode,
        "time": {"frame_count": 75},
        "listener": {"position_m": [-0.7, 1.471, 0.65], "yaw_deg": 55.0},
        "tracks": {
            "instances": {
                "source1": {
                    "emitter_position_m": paths,
                    "root_position_m": paths,
                },
                "source2": {
                    "emitter_position_m": paths,
                    "root_position_m": paths,
                },
            }
        },
        "instances": [
            {
                "instance_id": slot,
                "registry": {"asset_revision": "v1"},
                "emitter": {
                    "anchor_id": "mouth" if slot == "source2" else "muzzle",
                    "offset_m": [0.0, 0.0, 0.0],
                    "offset_space": "asset_local",
                },
            }
            for slot in ["source1", "source2"]
        ],
    }
    actors = [
        {
            "actor_id": f"{slot}_actor",
            "asset_id": f"{slot}_asset",
            "habitat_local_anatomical_forward_axis": "+x",
        }
        for slot in ["source1", "source2"]
    ]
    suite = {
        "scenarios": [
            {
                "scenario_id": source_episode,
                "scenario_directory": source_episode,
                "plan": {
                    "camera": {},
                    "actors": actors,
                    "frames": [
                        {"source_frame_marker": index} for index in range(75)
                    ],
                },
            }
        ]
    }
    return facts, suite


def test_native_recipe_has_real_turn_and_dynamic_camera(tmp_path: Path) -> None:
    facts, suite = _source_recipe_fixture()
    fact_path = tmp_path / "facts.json"
    suite_path = tmp_path / "suite.json"
    fact_path.write_text("{}\n", encoding="utf-8")
    suite_path.write_text("{}\n", encoding="utf-8")

    result = RECIPE.build_recipe(
        source_fact=facts,
        source_suite=suite,
        source_fact_path=fact_path,
        source_suite_path=suite_path,
    )

    assert len(result["source_frame_index_map"]) == 75
    assert result["source_frame_index_map"][28:31] == [28, 29, 28]
    frames = result["suite_execution_plan"]["scenarios"][0]["plan"]["frames"]
    assert [frames[index]["source_frame_marker"] for index in [28, 29, 30, 31]] == [
        28,
        29,
        28,
        27,
    ]
    rig = result["sensor_rig_trajectory"]
    assert len(rig["frames"]) == 75
    assert len({frame["pose_hash"] for frame in rig["frames"]}) == 75
    assert result["suite_execution_plan"]["scenarios"][0]["plan"]["camera"][
        "dynamic"
    ] is True


def _full_occlusion_facts() -> dict:
    frames = [
        {
            "frame_index": 0,
            "state": "visible_clear",
            "visible_pixels": 20,
            "target_pixels": 20,
        },
        {
            "frame_index": 1,
            "state": "fully_occluded",
            "visible_pixels": 0,
            "target_pixels": 20,
        },
        {
            "frame_index": 2,
            "state": "visible_occluded",
            "visible_pixels": 10,
            "target_pixels": 20,
        },
    ]
    return {
        "episode_id": "full_occlusion_test",
        "instances": [
            {
                "source_slot_id": "source1",
                "breed_id": "dog",
                "attributes": {"sex_or_gender_label": "female"},
            },
            {
                "source_slot_id": "source2",
                "breed_id": None,
                "attributes": {"sex_or_gender_label": "male"},
            },
        ],
        "sound_events": [
            {
                "event_id": "speech_000",
                "source_slot_id": "source2",
                "start_frame": 0,
                "end_frame": 3,
                "sound_asset_id": "controlled_speech_v1",
                "sound_class": {"species_id": "human"},
                "dry_variant": {
                    "input_path": "/controlled/speech.wav",
                    "input_sha256": "a" * 64,
                },
                "statement_id": "statement_v1",
                "transcript": "It's eleven o'clock.",
                "language": "en",
            }
        ],
        "visibility": {
            "pixel_truth": {"per_instance": {"source2": {"frames": frames}}},
            "occluder_evidence": {
                "frame_records": [
                    {
                        "frame_index": 1,
                        "target_instance_id": "source2",
                        "occluder_instance_ids": [],
                        "candidates": [
                            {"occluder_id": "sink", "pixel_count": 14},
                            {"occluder_id": "chair", "pixel_count": 6},
                        ],
                    }
                ]
            },
        },
    }


def test_full_occlusion_binder_omits_only_ambiguous_occluder_question() -> None:
    facts = _full_occlusion_facts()

    specs, _, _ = BINDER._question_inputs(
        facts,
        scenario_type="full_occlusion_to_reappearance",
        target_slot="source2",
    )

    by_id = {spec["spec_id"]: spec for spec in specs}
    assert {"QS-008", "QS-009", "QS-012"}.issubset(by_id)
    assert "QS-010" not in by_id
    assert by_id["QS-012"]["selectors"] == {
        "appearance_field": "sex_or_gender_label",
        "appearance_value": "male",
    }

    uniquely_resolved = deepcopy(facts)
    uniquely_resolved["visibility"]["occluder_evidence"]["frame_records"][0][
        "occluder_instance_ids"
    ] = ["sink"]
    unique_specs, _, _ = BINDER._question_inputs(
        uniquely_resolved,
        scenario_type="full_occlusion_to_reappearance",
        target_slot="source2",
    )
    assert "QS-010" in {spec["spec_id"] for spec in unique_specs}


def test_controlled_audio_wave_and_event_contract(tmp_path: Path) -> None:
    audio_path = tmp_path / "controlled.wav"
    with wave.open(str(audio_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 10)
    digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    AUDIO._validate_wave(
        audio_path,
        {
            "sample_rate_hz": 16000,
            "channel_count": 1,
            "sample_count": 10,
            "sha256": digest,
        },
    )
    event = AUDIO._event(
        event_id="speech_000",
        endpoint_id="mouth",
        sound_id="speech_v1",
        start_sample=24000,
        end_sample=49626,
        source_start=0,
        gain=0.22,
    )
    assert event["start_tick"] == 72000
    assert event["end_tick_exclusive"] == 148878
    assert event["source_end_sample_exclusive"] == 25626

    with pytest.raises(RuntimeError, match="sample_count drift"):
        AUDIO._validate_wave(
            audio_path,
            {
                "sample_rate_hz": 16000,
                "channel_count": 1,
                "sample_count": 11,
                "sha256": digest,
            },
        )


def _depth_context(pass_kind: str, target: str | None = None) -> dict:
    result = {
        "pass_kind": pass_kind,
        "renderer_backend": "spear_ue",
        "rgb_renderer_backend": "spear_ue",
        "camera_contract_id": "camera_v1",
        "semantic_id_namespace": "semantic_v1",
        "resolution_hw": [4, 6],
        "frame_indices": [0],
        "camera_pose_ids": ["pose_000"],
    }
    if target is not None:
        result["target_instance_id"] = target
    return result


def test_recompile_retained_depth_truth_is_state_lossless(tmp_path: Path) -> None:
    background = 100.0
    normal = np.full((1, 4, 6), background, dtype=np.float32)
    source1 = normal.copy()
    source2 = normal.copy()
    source1[0, 1:3, 1:3] = 2.0
    source2[0, 0:2, 4:6] = 3.0
    normal[0, 1:3, 1:3] = 2.0
    normal[0, 0:2, 4:6] = 3.0
    truth = compile_depth_pixel_visibility_truth(
        normal_depth_m_frames=[normal[0]],
        target_only_depth_m_frames_by_instance={
            "source1": [source1[0]],
            "source2": [source2[0]],
        },
        semantic_ids_by_instance={"source1": 1, "source2": 2},
        normal_context=_depth_context("modal_scene"),
        target_only_contexts_by_instance={
            slot: _depth_context("target_only", slot)
            for slot in ["source1", "source2"]
        },
        target_only_background_depth_m=background,
        absolute_tolerance_m=0.01,
        relative_tolerance=0.002,
    )
    old_truth = deepcopy(truth)
    for record in old_truth["per_instance"].values():
        for frame in record["frames"]:
            frame.pop("target_bbox_xyxy_px")
            frame.pop("target_centroid_xy_px")
    truth_path = tmp_path / "old_truth.json"
    truth_path.write_text(json.dumps(old_truth), encoding="utf-8")
    depth_path = tmp_path / "metric_depth.npz"
    np.savez_compressed(
        depth_path,
        normal_depth_m=normal,
        target_only_source1_depth_m=source1,
        target_only_source2_depth_m=source2,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifacts": {
                    "pixel_visibility_truth": str(truth_path),
                    "metric_depth": str(depth_path),
                },
                "artifact_records": {},
                "sha256": {},
            }
        ),
        encoding="utf-8",
    )

    new_truth_path, new_manifest_path = RECOMPILER.recompile(
        manifest_path=manifest_path,
        output=tmp_path / "recompiled",
    )
    new_truth = json.loads(new_truth_path.read_text(encoding="utf-8"))
    new_manifest = json.loads(new_manifest_path.read_text(encoding="utf-8"))

    assert new_truth["per_instance"]["source1"]["state_counts"] == old_truth[
        "per_instance"
    ]["source1"]["state_counts"]
    assert new_truth["per_instance"]["source1"]["frames"][0][
        "target_bbox_xyxy_px"
    ] == [1, 1, 3, 3]
    assert new_truth["per_instance"]["source1"]["frames"][0][
        "target_centroid_xy_px"
    ] == pytest.approx([1.5, 1.5])
    assert new_manifest["lossless_truth_recompile"]["state_counts_unchanged"] is True
