from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import avengine.rooms.native_qa_room as nq
from avengine.rooms.native_qa_room import (
    NativeApartmentResources,
    NativeQAResourceError,
    build_native_apartment_layout,
    native_apartment_room_entry,
    select_native_walking_routes,
)


def _resources(tmp_path: Path) -> NativeApartmentResources:
    room_manifest = tmp_path / "room_manifest.json"
    mesh_audit = tmp_path / "mesh_audit.json"
    surface = tmp_path / "scene.glb"
    export = tmp_path / "ue_export_manifest.json"
    navmesh = tmp_path / "apartment.navmesh"
    route_bank = tmp_path / "route_bank.json"
    for path in (room_manifest, surface, export, navmesh):
        path.write_bytes(b"placeholder")
    mesh_audit.write_text(
        json.dumps(
            {
                "bounds": {"min": [-2.0, 0.0, -3.0], "max": [4.0, 3.0, 5.0]},
                "details": {
                    "mesh_breakdown": [
                        {
                            "node": "Floor",
                            "mesh_datablock": "SM_Floor",
                            "bounds": {"min": [-2.0, 0.0, -3.0], "max": [4.0, 0.2, 5.0]},
                            "triangles": 20,
                        },
                        {
                            "node": "Sofa",
                            "mesh_datablock": "SM_Sofa",
                            "bounds": {"min": [0.0, 0.2, -1.0], "max": [1.0, 1.2, 0.0]},
                            "triangles": 30,
                        },
                    ]
                },
            }
        )
    )
    routes = []
    for route_id, y0 in (("r_left", 0.0), ("r_right", 200.0)):
        samples = [[round(300.0 * index / 74.0, 4), y0] for index in range(75)]
        routes.append(
            {
                "route_id": route_id,
                "waypoints_ue_cm": [[0.0, y0, 28.0], [300.0, y0, 28.0]],
                "samples_ue_cm": samples,
            }
        )
    route_bank.write_text(
        json.dumps({"schema": "test", "frame_count": 75, "frame_rate_hz": 15, "clip_seconds": 5.0, "routes": routes})
    )
    return NativeApartmentResources(
        source_root=tmp_path,
        room_manifest=room_manifest,
        mesh_audit=mesh_audit,
        surface_glb=surface,
        ue_export_manifest=export,
        navmesh=navmesh,
        route_bank=route_bank,
        acoustic_package=None,
        scene_id="apartment_0000",
        room_id="legacy_ue_apartment_0000_v1",
        map_path="/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000",
    )


def test_layout_comes_from_native_mesh_audit_and_preserves_real_object_semantics(tmp_path):
    layout = build_native_apartment_layout(_resources(tmp_path))

    assert layout["scene_id"] == "apartment_0000"
    assert len(layout["objects"]) == 2
    assert {item["semantic_class"] for item in layout["objects"]} == {"floor", "sofa"}
    assert layout["geometry"]["bounds_xy_m"] == [-2.0, -5.0, 4.0, 3.0]
    assert layout["native_floor_height_m"] == 0.2
    assert layout["resources"]["visual_geometry"]["resolved"].endswith("scene.glb")


def test_route_pair_reuses_native_bank_and_expands_without_replanning(tmp_path):
    routes, record = select_native_walking_routes(
        _resources(tmp_path), frame_count=240, seed=20260906
    )

    assert {key: value.shape for key, value in routes.items()} == {
        "source1": (240, 3),
        "source2": (240, 3),
    }
    assert set(record["selected_route_ids"].values()) == {"r_left", "r_right"}
    assert record["expanded_route_policy"] == "native_samples_once_then_endpoint_hold"
    assert record["minimum_actor_separation_m"] >= 0.95
    assert all(np.linalg.norm(np.diff(path, axis=0), axis=1).sum() > 2.0 for path in routes.values())
    assert all(np.allclose(path[-1], path[-2]) for path in routes.values())
    assert all(np.linalg.norm(path[-1] - path[0]) > 1.0 for path in routes.values())


def test_room_pool_entry_has_native_map_and_no_actor_coordinates(tmp_path):
    resources = _resources(tmp_path)
    entry = native_apartment_room_entry(resources)

    assert entry["room_id"] == resources.room_id
    assert entry["map_path"].endswith("apartment_0000")
    assert entry["native_room_adapter"] == "avengine_native_spear_apartment_qa_room_v1"
    assert "source1" not in entry and "source2" not in entry


def test_route_pair_rejects_request_fps_that_differs_from_native_bank(tmp_path):
    with pytest.raises(NativeQAResourceError, match="does not match native route bank clock"):
        select_native_walking_routes(_resources(tmp_path), frame_rate_hz=30)


# ------------------------------------- P05: explicit requests and instance identity

def _sounds():
    return [{"sound_asset_id": f"speech_{i}", "path": f"/prepared/{i}.wav",
             "sound_class": "speech", "gender": "M", "transcript": f"utterance {i}"}
            for i in range(2)]


def test_conditioned_route_forwards_every_explicit_statement(tmp_path, monkeypatch):
    captured = {}

    def recorder(*, room, request, source_registry, sounds):
        captured.update(request)
        return ({"visual_plan": {"frames": [], "actors": []}, "resources": {}}, {}, object())

    monkeypatch.setattr(nq, "build_qa_episode_plan", recorder, raising=False)
    monkeypatch.setattr("avengine.rooms.qa_episode.build_qa_episode_plan", recorder)
    nq.build_native_apartment_qa_plan(
        resources=_resources(tmp_path), source_registry={"assets": []}, sounds=_sounds(),
        episode_id="explicit", source_asset_ids=["human_0", "human_1"],
        sampling_policy="conditioned_static_v2", camera={"fov_deg": 62.0},
        profile={"speech_motion": "all_still", "competitor_motion": "moving"},
        entities={"instances": [{"instance_id": "a", "asset_id": "human_0"},
                                {"instance_id": "b", "asset_id": "human_1"}]},
        motion={"speed_range_mps": [1.1, 1.2]}, question_branches={"QA-06": "still"},
        frame_count=150, frame_rate_hz=15, sample_rate_hz=16000)

    assert captured["camera"]["fov_deg"] == 62.0
    assert captured["camera_fov_source"] == "request"
    # The route default never replaces what the caller stated.
    assert captured["profile"]["speech_motion"] == "all_still"
    assert captured["profile"]["competitor_motion"] == "moving"
    assert captured["profile"]["event_relation"] == "sequential"
    assert captured["motion"] == {"speed_range_mps": [1.1, 1.2]}
    assert captured["question_branches"] == {"QA-06": "still"}
    assert [row["instance_id"] for row in captured["entities"]["instances"]] == ["a", "b"]
    assert captured["frame_count"] == 150 and captured["sample_rate_hz"] == 16000


def test_conditioned_route_still_supplies_its_own_defaults(tmp_path, monkeypatch):
    captured = {}

    def recorder(*, room, request, source_registry, sounds):
        captured.update(request)
        return ({"visual_plan": {"frames": [], "actors": []}, "resources": {}}, {}, object())

    monkeypatch.setattr("avengine.rooms.qa_episode.build_qa_episode_plan", recorder)
    nq.build_native_apartment_qa_plan(
        resources=_resources(tmp_path), source_registry={"assets": []}, sounds=_sounds(),
        episode_id="default", source_asset_ids=["human_0", "human_1"],
        sampling_policy="conditioned_static_v2", audio_mode="overlap")

    assert captured["camera"]["fov_deg"] == 85.0
    assert captured["camera_fov_source"] == "route_default"
    assert captured["profile"] == {"speech_motion": "speaker_moving", "event_relation": "overlap"}
    assert "question_branches" not in captured and "entities" not in captured


def test_two_instances_of_one_asset_each_get_their_own_identity(tmp_path, monkeypatch):
    calls = []

    def recorder(registry, asset_id, actor_id, *, entity_instance_id=None, instance_ordinal=1):
        calls.append((asset_id, actor_id, instance_ordinal))
        if len(calls) < 2:
            return {"actor_id": actor_id, "asset_id": asset_id,
                    "entity_instance_id": f"{asset_id}#instance{instance_ordinal:02d}"}
        raise RuntimeError("stop after both declarations")

    monkeypatch.setattr(nq, "source_declaration", recorder)
    with pytest.raises(RuntimeError, match="stop after both declarations"):
        nq.build_native_apartment_qa_plan(
            resources=_resources(tmp_path), source_registry={"assets": []}, sounds=_sounds(),
            episode_id="same_asset", source_asset_ids=["human_0", "human_0"],
            frame_count=75, frame_rate_hz=15)
    assert calls == [("human_0", "source1", 1), ("human_0", "source2", 2)]


def test_the_native_route_still_requires_exactly_two_instances(tmp_path):
    with pytest.raises(NativeQAResourceError, match="exactly two source instances"):
        nq.build_native_apartment_qa_plan(
            resources=_resources(tmp_path), source_registry={"assets": []}, sounds=_sounds(),
            episode_id="three", source_asset_ids=["human_0", "human_1", "human_2"])
