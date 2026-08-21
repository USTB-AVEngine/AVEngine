from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from avengine.qa.full_episode_semantic_authority import (
    ADAPTER_REGISTRY_SCHEMA,
    AUTHORITY_SCHEMA,
    DYNAMIC_FINALIZATION_SCHEMA,
    LABEL_SCHEMA,
    REQUEST_SCHEMA,
    STATIC_FINALIZATION_SCHEMA,
    FullEpisodeSemanticAuthorityError,
    build_full_episode_semantic_authority,
)
from avengine.qa.full_episode_validation_batch import (
    ARTIFACT_ROLES,
    build_full_episode_validation_batch,
)
from avengine.qa.full_episode_validation_batch import (
    REQUEST_SCHEMA as VALIDATION_REQUEST_SCHEMA,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "tools" / "qa" / "build_full_episode_semantic_authority.py"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _touch(path: Path, payload: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path.resolve(strict=True)


def _selected(path: Path, selector: str = "") -> dict[str, Any]:
    return {
        "authority_ref": {"path": str(path.resolve(strict=True))},
        "authority_selector": selector,
    }


def _registry(root: Path) -> dict[str, Any]:
    path = root / "adapter_registry.json"
    _write(
        path,
        {
            "registry": {
                "schema": ADAPTER_REGISTRY_SCHEMA,
                "adapters": [
                    {
                        "finalization_schema": STATIC_FINALIZATION_SCHEMA,
                        "source_kind": "static",
                        "planning_schema": (
                            "avengine_native_strict_two_human_full75_canary_plan_v1"
                        ),
                    },
                ],
            }
        },
    )
    return _selected(path, "/registry")


def _episode_fixture(
    root: Path,
    *,
    episode_id: str,
) -> dict[str, Any]:
    scene_id = "apartment_0000"
    room_id = "legacy_ue_apartment_0000_v1"
    room_variant_id = "spear_apartment_0000_habitat_fixed_v1@v1"
    map_id = "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"
    camera_cluster_id = f"camera_{episode_id}"
    source1_asset = "rocketbox_male"
    source2_asset = "rocketbox_female"
    source1_revision = "runtime_v3"
    source2_revision = "runtime_v1"
    sound_asset_id = f"speech_{episode_id}"
    voice_id = f"voice_{episode_id}"
    content_id = f"approved_content_{episode_id}"
    event_id = f"runtime_event_{episode_id}"

    capture_root = root / "capture"
    capture_manifest = capture_root / "manifest.json"
    suite_path = root / "suite.json"
    rir_path = root / "rir_plan.json"
    samples_path = root / "binaural_v1" / "samples.json"
    delivery_path = root / "binaural_v1" / "delivery.json"
    mixture_path = _touch(
        root / "binaural_v1" / "audio" / "binaural" / f"{episode_id}__v00.wav"
    )
    _write(
        capture_manifest,
        {
            "schema": "avengine_qa_native_spear_pixel_episode_v1",
            "status": "pass",
            "scenario_id": episode_id,
            "native_map": map_id,
            "frame_contract": {
                "frame_count": 75,
                "frame_rate_hz": 15,
                "captured_frame_indices": list(range(75)),
            },
            "authoritative_capture_request": {
                "episode_id": episode_id,
                "scenario_type": "strict_two_human_static_canary",
                "target_source_slot_id": "source1",
            },
            "audio": {"authoritative_wav": str(mixture_path)},
        },
    )
    finalization_path = root / "finalization.json"
    static_artifacts = {
        "capture_manifest": str(capture_manifest.resolve(strict=True)),
        "audiovisual_mp4": str(_touch(capture_root / "native_rgb_binaural.mp4")),
        "binaural_wav": str(mixture_path),
        "pixel_visibility_truth": str(
            _touch(capture_root / "pixel_visibility_truth.json")
        ),
        "runtime_readbacks": str(_touch(capture_root / "runtime_readbacks.json")),
    }
    finalization = {
        "schema": STATIC_FINALIZATION_SCHEMA,
        "status": "pass",
        "episode_id": episode_id,
        "full75_canary_pass": True,
        "captured_frame_count": 75,
        "formal_episode_count": 0,
        "qualification_claim": False,
        "pixels": {"status": "pass", "target_side": "left"},
        "acoustics": {
            "status": "pass_exact_two_source_rir_target_only_binaural",
            "rir_job_count": 2,
            "target_active": True,
            "distractor_silent": True,
            "channel_count": 2,
            "sample_count": 80_000,
            "sample_rate_hz": 16_000,
        },
        "artifacts": {
            "capture_manifest": static_artifacts["capture_manifest"],
            "binaural_video": static_artifacts["audiovisual_mp4"],
            "binaural_wav": static_artifacts["binaural_wav"],
            "pixel_visibility_truth": static_artifacts["pixel_visibility_truth"],
            "runtime_readbacks": static_artifacts["runtime_readbacks"],
        },
    }
    _write(finalization_path, finalization)

    planning_path = root / "planning.json"
    identity_path = root / "identity_plan.json"
    planning_row = {
        "canary_index": 1,
        "row_id": camera_cluster_id,
        "episode_id": episode_id,
        "target_identity_key": "M",
        "distractor_identity_key": "F",
        "target_side": "left",
        "suite_plan": str(suite_path.resolve()),
        "output_root": str(capture_root.resolve(strict=True)),
        "audio_wav": str(mixture_path),
        "acoustic_evidence": {
            "binaural_delivery": str(delivery_path.resolve()),
            "exact_rir_plan": str(rir_path.resolve()),
        },
    }
    _write(
        planning_path,
        {
            "schema": "avengine_native_strict_two_human_full75_canary_plan_v1",
            "status": "ready_pending_gpu1_idle_gate",
            "full_batch_authorized": False,
            "canaries": [planning_row],
        },
    )
    _write(
        identity_path,
        {
            "schema": "avengine_native_strict_two_human_expansion_plan_v1",
            "timeline": {"frame_count": 75, "frame_rate_hz": 15},
            "approved_identity_catalog": {
                "M": {
                    "original_identity_id": "person_male",
                    "runtime_asset_id": source1_asset,
                    "runtime_revision": source1_revision,
                    "sound_asset_id": sound_asset_id,
                },
                "F": {
                    "original_identity_id": "person_female",
                    "runtime_asset_id": source2_asset,
                    "runtime_revision": source2_revision,
                    "sound_asset_id": "unused_silent_sound",
                },
            },
            "rows": [
                {
                    "row_id": camera_cluster_id,
                    "episode_id": episode_id,
                    "target_expected_screen_side": "left",
                    "camera_pose": {
                        "translation_m": [1.0, 1.471, 2.0],
                        "habitat_yaw_deg": 45.0,
                    },
                    "actors": [
                        {
                            "role": "target",
                            "source_slot_id": "source1",
                            "identity_key": "M",
                            "expected_screen_side": "left",
                            "voice_policy": "speaking",
                        },
                        {
                            "role": "distractor",
                            "source_slot_id": "source2",
                            "identity_key": "F",
                            "expected_screen_side": "right",
                            "voice_policy": "silent",
                        },
                    ],
                }
            ],
        },
    )

    frames = [
        {
            "frame_index": index,
            "pts_ticks": index * 3200,
            "actor_states": [],
            "camera_state": {},
        }
        for index in range(75)
    ]
    _write(
        suite_path,
        {
            "schema": "avengine_optional_spear_apartment_suite_v1",
            "scenarios": [
                {
                    "schema": "avengine_optional_spear_apartment_scenario_v1",
                    "scenario_id": episode_id,
                    "native_scene": {"map": map_id},
                    "plan": {
                        "schema": "avengine_optional_spear_visual_plan_v1",
                        "room": {
                            "room_id": room_id,
                            "room_capsule_id": "spear_apartment_0000_habitat_fixed_v1",
                            "room_capsule_revision": "v1",
                            "source_scene_provenance": {"scene_id": scene_id},
                        },
                        "render": {"frame_count": 75, "fps_num": 15, "fps_den": 1},
                        "camera": {
                            "dynamic": False,
                            "habitat_position_m": [1.0, 1.471, 2.0],
                            "habitat_yaw_deg": 45.0,
                            "sensor_rig_trajectory_id": f"camera_{episode_id}",
                        },
                        "frames": frames,
                        "actors": [
                            {
                                "actor_id": "source1_actor",
                                "asset_id": source1_asset,
                                "asset_revision": source1_revision,
                                "runtime_asset_expectation": {
                                    "source_slot_id": "source1"
                                },
                            },
                            {
                                "actor_id": "source2_actor",
                                "asset_id": source2_asset,
                                "asset_revision": source2_revision,
                                "runtime_asset_expectation": {
                                    "source_slot_id": "source2"
                                },
                            },
                        ],
                    },
                }
            ],
        },
    )

    endpoint1 = f"endpoint_{episode_id}_source1"
    endpoint2 = f"endpoint_{episode_id}_source2"
    endpoint_path = root / "endpoints.json"
    _write(
        endpoint_path,
        {
            "schema": "avengine_m6_source_endpoint_registry_v1",
            "source_endpoints": [
                {
                    "source_endpoint_id": endpoint1,
                    "binding": {
                        "entity_instance_id": "source1",
                        "entity_asset_id": source1_asset,
                        "entity_asset_revision": source1_revision,
                        "emitter_anchor_id": "mouth",
                    },
                },
                {
                    "source_endpoint_id": endpoint2,
                    "binding": {
                        "entity_instance_id": "source2",
                        "entity_asset_id": source2_asset,
                        "entity_asset_revision": source2_revision,
                        "emitter_anchor_id": "mouth",
                    },
                },
            ],
        },
    )

    program_path = root / "audio_program.json"
    event = {
        "event_id": event_id,
        "sound_asset_id": sound_asset_id,
        "source_endpoint_id": endpoint1,
        "start_sample": 7467,
        "end_sample_exclusive": 33093,
    }
    _write(
        program_path,
        {
            "schema": "avengine_m6_audio_program_v1",
            "program_id": f"program_{episode_id}",
            "revision": "v1",
            "mode": "one_active_of_n",
            "timeline": {
                "frame_count": 75,
                "video_fps": 15,
                "sample_count": 80_000,
                "sample_rate_hz": 16_000,
            },
            "events": [event],
        },
    )
    sound_path = root / "sounds.json"
    _write(
        sound_path,
        {
            "schema": "avengine_m6_sound_asset_registry_v1",
            "sound_assets": [
                {
                    "sound_asset_id": sound_asset_id,
                    "semantic_sound_class": "human_speech",
                    "instance_lineage_id": voice_id,
                }
            ],
        },
    )
    _write(
        samples_path,
        {
            "schema": "avengine_m7_asset_bound_binaural_training_samples_v1",
            "samples": [
                {
                    "episode_id": episode_id,
                    "sample_id": f"{episode_id}__v00",
                    "both_sources_active": False,
                    "asset_ids_by_source_slot": {
                        "source1": source1_asset,
                        "source2": source2_asset,
                    },
                    "audio": {"mixture": {"path": mixture_path.name}},
                    "audio_program_binding": {
                        "audio_program_ref": {
                            "program_id": f"program_{episode_id}",
                            "revision": "v1",
                        },
                        "source_endpoint_to_source_slot": {
                            endpoint1: "source1",
                            endpoint2: "source2",
                        },
                    },
                    "source_activity_summary": {
                        "active_source_slots": ["source1"],
                        "silent_source_slots": ["source2"],
                    },
                }
            ],
        },
    )
    _write(
        delivery_path,
        {
            "schema": "avengine_m7_asset_bound_binaural_batch_delivery_v1",
            "status": "pass",
            "episode_count": 1,
            "qualification_claim": False,
            "outputs": {"samples": "samples.json"},
        },
    )

    rir_ids = [f"rir_{episode_id}_source1", f"rir_{episode_id}_source2"]
    _write(
        rir_path,
        {
            "schema": "avengine_room_rir_job_plan_v2",
            "jobs": [
                {
                    "job_id": rir_ids[0],
                    "uses": [{"episode_id": episode_id, "source_slot_id": "source1"}],
                },
                {
                    "job_id": rir_ids[1],
                    "uses": [{"episode_id": episode_id, "source_slot_id": "source2"}],
                },
            ],
        },
    )

    actors = [
        {
            "source_slot_id": "source1",
            "role": "target",
            "side": "left",
            "identity_id": "person_male",
            "asset_id": source1_asset,
            "asset_revision": source1_revision,
            "voice_id": voice_id,
            "content_id": content_id,
            "sound_asset_id": sound_asset_id,
        },
        {
            "source_slot_id": "source2",
            "role": "distractor",
            "side": "right",
            "identity_id": "person_female",
            "asset_id": source2_asset,
            "asset_revision": source2_revision,
            "voice_policy": "silent",
        },
    ]
    label_path = root / "approved_label.json"
    _write(
        label_path,
        {
            "records": [
                {
                    "schema": LABEL_SCHEMA,
                    "approval": {
                        "status": "approved",
                        "approved_by": "human_reviewer",
                        "approved_at": "2026-08-13T00:00:00Z",
                    },
                    "episode_id": episode_id,
                    "mechanism": "both_static",
                    "scene": {
                        "scene_id": scene_id,
                        "room_id": room_id,
                        "room_variant_id": room_variant_id,
                        "map_id": map_id,
                    },
                    "camera": {"camera_cluster_id": f"camera_{episode_id}"},
                    "actors": actors,
                    "timeline": {
                        "frame_count": 75,
                        "frame_rate_hz": 15,
                        "duration_seconds": 5,
                    },
                    "rir_job_ids": rir_ids,
                    "question": {
                        "prompt": "Who spoke?",
                        "options": [
                            {"semantic_id": "left", "text": "Left"},
                            {"semantic_id": "right", "text": "Right"},
                        ],
                        "correct_index": 0,
                        "option_order_id": "left-right-v1",
                    },
                    "formal_episode_count": 0,
                    "qualification_claim": False,
                    "independence_claim": False,
                }
            ]
        },
    )
    entry = {
        "finalization": _selected(finalization_path),
        "planning": _selected(planning_path, "/canaries/0"),
        "suite": _selected(suite_path, "/scenarios/0"),
        "audio_sample": _selected(samples_path, "/samples/0"),
        "audio_delivery": _selected(delivery_path),
        "audio_program": _selected(program_path),
        "audio_event": _selected(program_path, "/events/0"),
        "sound_asset": _selected(sound_path, "/sound_assets/0"),
        "source_endpoints": [
            _selected(endpoint_path, "/source_endpoints/0"),
            _selected(endpoint_path, "/source_endpoints/1"),
        ],
        "rir_jobs": [
            _selected(rir_path, "/jobs/0"),
            _selected(rir_path, "/jobs/1"),
        ],
        "semantic_label": _selected(label_path, "/records/0"),
        "_validation_artifacts": static_artifacts,
    }
    entry["identity_binding"] = _selected(identity_path, "/rows/0")
    return entry


def _request(root: Path) -> dict[str, Any]:
    episodes = [
        _episode_fixture(root / f"static_{index}", episode_id=f"static_{index}")
        for index in range(4)
    ]
    for episode in episodes:
        episode.pop("_validation_artifacts")
    return {
        "schema": REQUEST_SCHEMA,
        "adapter_registry": _registry(root),
        "episodes": episodes,
    }


def _build(request: dict[str, Any], root: Path) -> dict[str, Any]:
    return build_full_episode_semantic_authority(
        request, authority_path=(root / "authority.json").resolve()
    )


def _mutate_selected(binding: dict[str, Any], mutator: Any) -> None:
    path = Path(binding["authority_ref"]["path"])
    document = json.loads(path.read_text(encoding="utf-8"))
    selector = binding["authority_selector"]
    current = document
    for token in selector[1:].split("/") if selector else []:
        current = current[int(token)] if isinstance(current, list) else current[token]
    mutator(current)
    _write(path, document)


def test_real_shaped_static_selected_refs_without_scanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("directory scanning is forbidden")

    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    first = _build(request, tmp_path)
    second = _build(request, tmp_path)

    assert first == second
    assert first["schema"] == AUTHORITY_SCHEMA
    assert first["status"] == "pass"
    assert first["episode_count"] == 4
    assert first["formal_episode_count"] == 0
    assert first["qualification_claim"] is False
    assert first["independence_claim"] is False
    assert first["readiness"]["ready_scene_count"] == 1
    assert (
        first["readiness"]["ready_groups"][0]["static_machine_full75_evidence_count"]
        == 4
    )
    assert first["records"][0]["question"]["option_semantics"] == [
        {"semantic_id": "left", "text": "Left"},
        {"semantic_id": "right", "text": "Right"},
    ]
    assert first["ready_record_selectors"][0]["authority_selector"] == "/records/0"
    assert first["ready_record_selectors"][0]["authority_ref"] == {
        "path": str((tmp_path / "authority.json").resolve())
    }
    assert first["records"][0]["actors"][0]["content_id"].startswith(
        "approved_content_"
    )
    assert not first["records"][0]["actors"][0]["content_id"].startswith(
        "runtime_event_"
    )
    assert all(
        set(selected) == {"authority_ref", "authority_selector"}
        for provenance in first["provenance"]
        for value in provenance.values()
        for selected in (value if isinstance(value, list) else [value])
    )


def test_output_record_feeds_existing_validation_builder(tmp_path: Path) -> None:
    root = tmp_path / "input"
    raw_episodes = [
        _episode_fixture(root / f"static_{index}", episode_id=f"static_{index}")
        for index in range(4)
    ]
    artifacts = raw_episodes[0].pop("_validation_artifacts")
    for episode in raw_episodes[1:]:
        episode.pop("_validation_artifacts")
    authority_path = tmp_path / "semantic_authority.json"
    result = build_full_episode_semantic_authority(
        {
            "schema": REQUEST_SCHEMA,
            "adapter_registry": _registry(root),
            "episodes": raw_episodes,
        },
        authority_path=authority_path,
    )
    _write(authority_path, result)
    finalization_ref = raw_episodes[0]["finalization"]["authority_ref"]
    artifact_refs = {
        "source_finalization": finalization_ref,
        **{
            role: {"path": artifacts[role]}
            for role in ARTIFACT_ROLES
            if role != "source_finalization"
        },
    }
    batch = build_full_episode_validation_batch(
        {
            "schema": VALIDATION_REQUEST_SCHEMA,
            "episodes": [
                {
                    "finalization_ref": finalization_ref,
                    "semantic_authority_ref": {
                        "path": str(authority_path.resolve(strict=True))
                    },
                    "semantic_selector": "/records/0",
                    "artifacts": artifact_refs,
                }
            ],
        }
    )
    assert batch["status"] == "pass"
    assert batch["episode_count"] == 1
    assert batch["formal_episode_count"] == 0


def test_rejects_unknown_schema_unapproved_label_and_semantic_drift(
    tmp_path: Path,
) -> None:
    unknown = _request(tmp_path / "unknown")
    _mutate_selected(
        unknown["episodes"][0]["finalization"],
        lambda value: value.update(schema="unknown"),
    )
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="no selected adapter"):
        _build(unknown, tmp_path / "unknown")

    dynamic = _request(tmp_path / "dynamic")
    _mutate_selected(
        dynamic["episodes"][0]["finalization"],
        lambda value: value.update(schema=DYNAMIC_FINALIZATION_SCHEMA),
    )
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="no selected adapter"):
        _build(dynamic, tmp_path / "dynamic")

    unapproved = _request(tmp_path / "unapproved")
    _mutate_selected(
        unapproved["episodes"][0]["semantic_label"],
        lambda value: value["approval"].update(status="pending"),
    )
    with pytest.raises(
        FullEpisodeSemanticAuthorityError, match="not explicitly approved"
    ):
        _build(unapproved, tmp_path / "unapproved")

    drift = _request(tmp_path / "drift")
    _mutate_selected(
        drift["episodes"][0]["semantic_label"],
        lambda value: value["scene"].update(room_id="other"),
    )
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="scene drifted"):
        _build(drift, tmp_path / "drift")


def test_three_static_and_split_scene_remain_not_ready(
    tmp_path: Path,
) -> None:
    three = _request(tmp_path / "three")
    three["episodes"].pop()
    three_result = _build(three, tmp_path / "three")
    assert three_result["status"] == "not_ready"
    assert three_result["readiness"]["status"] == "not_ready"
    assert three_result["room_readiness_records"] == []

    split = _request(tmp_path / "split")
    for episode_index in (2, 3):
        _mutate_selected(
            split["episodes"][episode_index]["semantic_label"],
            lambda value: value["scene"].update(room_variant_id="other@variant"),
        )
        _mutate_selected(
            split["episodes"][episode_index]["suite"],
            lambda value: value["plan"]["room"].update(
                room_capsule_id="other", room_capsule_revision="variant"
            ),
        )
    split_result = _build(split, tmp_path / "split")
    assert split_result["status"] == "not_ready"
    assert split_result["readiness"]["status"] == "not_ready"
    assert split_result["room_readiness_records"] == []
    assert split_result["ready_record_selectors"] == []

    mixed_root = tmp_path / "mixed"
    mixed = _request(mixed_root)
    other = _episode_fixture(mixed_root / "other", episode_id="other")
    other.pop("_validation_artifacts")
    _mutate_selected(
        other["semantic_label"],
        lambda value: value["scene"].update(room_variant_id="other@variant"),
    )
    _mutate_selected(
        other["suite"],
        lambda value: value["plan"]["room"].update(
            room_capsule_id="other", room_capsule_revision="variant"
        ),
    )
    mixed["episodes"].append(other)
    mixed_result = _build(mixed, mixed_root)
    assert mixed_result["status"] == "pass"
    assert len(mixed_result["ready_record_selectors"]) == 4
    assert all(
        row["episode_id"] != "other" for row in mixed_result["ready_record_selectors"]
    )
    assert mixed_result["records"][4]["scene"]["room_readiness"]["status"] == (
        "not_ready"
    )


def test_rejects_relative_authority_path(tmp_path: Path) -> None:
    absolute_request = _request(tmp_path / "absolute_request")
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="must be absolute"):
        build_full_episode_semantic_authority(
            absolute_request, authority_path="relative.json"
        )
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="existing directory"):
        build_full_episode_semantic_authority(
            absolute_request,
            authority_path=tmp_path / "missing_parent" / "authority.json",
        )

    relative = _request(tmp_path / "relative")
    relative["episodes"][0]["suite"]["authority_ref"]["path"] = "suite.json"
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="not absolute"):
        _build(relative, tmp_path / "relative")


def test_rejects_audio_endpoint_rir_and_timeline_drift(tmp_path: Path) -> None:
    endpoint = _request(tmp_path / "endpoint")
    _mutate_selected(
        endpoint["episodes"][0]["source_endpoints"][0],
        lambda value: value["binding"].update(entity_asset_id="other"),
    )
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="endpoint actor"):
        _build(endpoint, tmp_path / "endpoint")

    flipped = _request(tmp_path / "flipped")
    _mutate_selected(
        flipped["episodes"][0]["audio_event"],
        lambda value: value.update(source_endpoint_id="endpoint_static_0_source2"),
    )
    _mutate_selected(
        flipped["episodes"][0]["audio_sample"],
        lambda value: value["source_activity_summary"].update(
            active_source_slots=["source2"], silent_source_slots=["source1"]
        ),
    )
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="roles"):
        _build(flipped, tmp_path / "flipped")

    rir = _request(tmp_path / "rir")
    _mutate_selected(
        rir["episodes"][0]["rir_jobs"][1],
        lambda value: value["uses"][0].update(source_slot_id="source1"),
    )
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="cover both"):
        _build(rir, tmp_path / "rir")

    timeline = _request(tmp_path / "timeline")
    _mutate_selected(
        timeline["episodes"][0]["suite"],
        lambda value: value["plan"]["render"].update(frame_count=74),
    )
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="contiguous full75"):
        _build(timeline, tmp_path / "timeline")

    pixel_side = _request(tmp_path / "pixel_side")
    _mutate_selected(
        pixel_side["episodes"][0]["finalization"],
        lambda value: value["pixels"].update(target_side="right"),
    )
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="pixel target side"):
        _build(pixel_side, tmp_path / "pixel_side")

    camera_yaw = _request(tmp_path / "camera_yaw")
    _mutate_selected(
        camera_yaw["episodes"][0]["suite"],
        lambda value: value["plan"]["camera"].update(habitat_yaw_deg=46.0),
    )
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="camera pose"):
        _build(camera_yaw, tmp_path / "camera_yaw")

    camera_trajectory = _request(tmp_path / "camera_trajectory")
    _mutate_selected(
        camera_trajectory["episodes"][0]["suite"],
        lambda value: value["plan"]["camera"].update(
            sensor_rig_trajectory_id="other_trajectory"
        ),
    )
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="camera drifted"):
        _build(camera_trajectory, tmp_path / "camera_trajectory")


def test_rejects_capture_target_and_audio_drift(tmp_path: Path) -> None:
    scenario_type = _request(tmp_path / "capture_scenario_type")
    finalization_path = Path(
        scenario_type["episodes"][0]["finalization"]["authority_ref"]["path"]
    )
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    capture_path = Path(finalization["artifacts"]["capture_manifest"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["authoritative_capture_request"]["scenario_type"] = "unknown"
    _write(capture_path, capture)
    with pytest.raises(
        FullEpisodeSemanticAuthorityError, match="planned active target"
    ):
        _build(scenario_type, tmp_path / "capture_scenario_type")

    frames = _request(tmp_path / "capture_frames")
    finalization_path = Path(
        frames["episodes"][0]["finalization"]["authority_ref"]["path"]
    )
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    capture_path = Path(finalization["artifacts"]["capture_manifest"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["frame_contract"]["captured_frame_indices"][-1] = 75
    _write(capture_path, capture)
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="contiguous full75"):
        _build(frames, tmp_path / "capture_frames")

    target = _request(tmp_path / "capture_target")
    finalization_path = Path(
        target["episodes"][0]["finalization"]["authority_ref"]["path"]
    )
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    capture_path = Path(finalization["artifacts"]["capture_manifest"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["authoritative_capture_request"]["target_source_slot_id"] = "source2"
    _write(capture_path, capture)
    with pytest.raises(
        FullEpisodeSemanticAuthorityError, match="planned active target"
    ):
        _build(target, tmp_path / "capture_target")

    audio = _request(tmp_path / "capture_audio")
    finalization_path = Path(
        audio["episodes"][0]["finalization"]["authority_ref"]["path"]
    )
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    capture_path = Path(finalization["artifacts"]["capture_manifest"])
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["audio"]["authoritative_wav"] = str(
        _touch(tmp_path / "capture_audio" / "wrong.wav")
    )
    _write(capture_path, capture)
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="capture audio"):
        _build(audio, tmp_path / "capture_audio")

    delivery = _request(tmp_path / "delivery_samples")
    wrong_samples = _touch(tmp_path / "delivery_samples" / "other_samples.json")
    _mutate_selected(
        delivery["episodes"][0]["audio_delivery"],
        lambda value: value["outputs"].update(samples=wrong_samples.name),
    )
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="samples output"):
        _build(delivery, tmp_path / "delivery_samples")


def test_cli_no_clobber(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "authority.json"
    _write(request_path, _request(tmp_path / "fixture"))
    spec = importlib.util.spec_from_file_location("semantic_authority_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["--request", str(request_path), "--output", str(output_path)]) == 0
    original = output_path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cli.main(["--request", str(request_path), "--output", str(output_path)])
    assert output_path.read_bytes() == original

    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing-target.json")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cli.main(["--request", str(request_path), "--output", str(dangling)])
    assert dangling.is_symlink()

    serialization_output = tmp_path / "serialization_failure.json"
    real_dump = cli.json.dump

    def fail_after_partial_write(_value: object, handle: Any, **_kwargs: Any) -> None:
        handle.write("{")
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(cli.json, "dump", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="injected serialization failure"):
        cli.main(
            [
                "--request",
                str(request_path),
                "--output",
                str(serialization_output),
            ]
        )
    assert not serialization_output.exists()
    assert not list(tmp_path.glob(f".{serialization_output.name}.*.tmp"))
    monkeypatch.setattr(cli.json, "dump", real_dump)

    interrupted_output = tmp_path / "interrupted_publish.json"

    def interrupt_publish(_source: object, _destination: object) -> None:
        raise KeyboardInterrupt("injected publish interruption")

    monkeypatch.setattr(cli.os, "link", interrupt_publish)
    with pytest.raises(KeyboardInterrupt, match="injected publish interruption"):
        cli.main(
            [
                "--request",
                str(request_path),
                "--output",
                str(interrupted_output),
            ]
        )
    assert not interrupted_output.exists()
    assert not list(tmp_path.glob(f".{interrupted_output.name}.*.tmp"))


def test_rejects_nested_finalization_selector(tmp_path: Path) -> None:
    request = _request(tmp_path)
    final_path = Path(request["episodes"][0]["finalization"]["authority_ref"]["path"])
    finalization = json.loads(final_path.read_text(encoding="utf-8"))
    _write(final_path, {"nested": finalization})
    request["episodes"][0]["finalization"]["authority_selector"] = "/nested"
    with pytest.raises(FullEpisodeSemanticAuthorityError, match="document root"):
        _build(request, tmp_path)


@pytest.mark.parametrize("canary_index", [1, 2, 3, 4])
def test_real_static_authorities_reach_only_pending_label_gate(
    tmp_path: Path, canary_index: int
) -> None:
    repository = Path("/data/jzy/code/AVEngine-lead-a")
    canary_plan_path = (
        repository
        / "tmp/lead_a_strict_two_human_full_episode_batch_v1"
        / "cpu_plan_v2/canary_plan.json"
    )
    expansion_path = (
        repository / "examples/qa/native_strict_two_human_expansion_v1.json"
    )
    if not canary_plan_path.is_file() or not expansion_path.is_file():
        pytest.skip("real static authority smoke is available only in the A workspace")

    canary_plan = json.loads(canary_plan_path.read_text(encoding="utf-8"))
    row = canary_plan["canaries"][canary_index - 1]
    suite_path = Path(row["suite_plan"])
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    scenario = suite["scenarios"][0]
    controlled_root = (
        repository / "tmp/lead_d_strict_two_human_canary_v1/recipe_v4"
        if canary_index == 1
        else suite_path.parent
    ) / "controlled_audio_program"
    program_path = controlled_root / "audio_program.json"
    sound_path = controlled_root / "sound_asset_registry.json"
    endpoint_path = controlled_root / "source_endpoint_registry.json"
    delivery_path = Path(row["acoustic_evidence"]["binaural_delivery"])
    delivery = json.loads(delivery_path.read_text(encoding="utf-8"))
    samples_path = delivery_path.parent / delivery["outputs"]["samples"]
    rir_path = Path(row["acoustic_evidence"]["exact_rir_plan"])
    finalization_path = (
        repository
        / "tmp/lead_a_strict_two_human_full_episode_batch_v1"
        / f"full75_canary_final/canary_{canary_index:02d}/finalization.json"
    )
    assert scenario["scenario_id"] == row["episode_id"]

    pending_label_path = tmp_path / f"pending_label_{canary_index}.json"
    _write(
        pending_label_path,
        {
            "records": [
                {
                    "schema": LABEL_SCHEMA,
                    "approval": {
                        "status": "pending",
                        "approved_by": "pending",
                        "approved_at": "pending",
                    },
                    "episode_id": row["episode_id"],
                    "mechanism": "both_static",
                    "scene": {},
                    "camera": {},
                    "actors": [],
                    "timeline": {},
                    "rir_job_ids": [],
                    "question": {},
                    "formal_episode_count": 0,
                    "qualification_claim": False,
                    "independence_claim": False,
                }
            ]
        },
    )
    request = {
        "schema": REQUEST_SCHEMA,
        "adapter_registry": _registry(tmp_path),
        "episodes": [
            {
                "finalization": _selected(finalization_path),
                "planning": _selected(
                    canary_plan_path, f"/canaries/{canary_index - 1}"
                ),
                "identity_binding": _selected(
                    expansion_path, f"/rows/{canary_index - 1}"
                ),
                "suite": _selected(suite_path, "/scenarios/0"),
                "audio_sample": _selected(samples_path, "/samples/0"),
                "audio_delivery": _selected(delivery_path),
                "audio_program": _selected(program_path),
                "audio_event": _selected(program_path, "/events/0"),
                "sound_asset": _selected(sound_path, "/sound_assets/0"),
                "source_endpoints": [
                    _selected(endpoint_path, "/source_endpoints/0"),
                    _selected(endpoint_path, "/source_endpoints/1"),
                ],
                "rir_jobs": [
                    _selected(rir_path, "/jobs/0"),
                    _selected(rir_path, "/jobs/1"),
                ],
                "semantic_label": _selected(pending_label_path, "/records/0"),
            }
        ],
    }
    authority_path = tmp_path / f"must_not_exist_{canary_index}.json"
    with pytest.raises(
        FullEpisodeSemanticAuthorityError, match="not explicitly approved"
    ):
        build_full_episode_semantic_authority(request, authority_path=authority_path)
    assert not authority_path.exists()
