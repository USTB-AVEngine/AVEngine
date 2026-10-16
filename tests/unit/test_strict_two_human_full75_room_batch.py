from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPOSITORY / "tools/qa/run_strict_two_human_full75_room_batch.py"
SPOOL_PATH = REPOSITORY / "tools/qa/strict_two_human_raw_spool.py"
LIFECYCLE_PATH = REPOSITORY / "tools/qa/spear_room_batch_lifecycle.py"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _module("strict2h_room_batch_runner_test", RUNNER_PATH)
SPOOL = _module("strict2h_room_batch_spool_test", SPOOL_PATH)
LIFECYCLE = _module("strict2h_room_batch_lifecycle_test", LIFECYCLE_PATH)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _float_wav(path: Path) -> None:
    channels = 2
    sample_rate = 16_000
    sample_count = 80_000
    bits = 32
    block = channels * bits // 8
    fmt = struct.pack(
        "<HHIIHH", 3, channels, sample_rate, sample_rate * block, block, bits
    )
    data = bytes(sample_count * block)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks)


def _request_fixture(tmp_path: Path) -> Path:
    output = tmp_path / "batch_output"
    rows = []
    for ordinal in range(2):
        episode_id = f"reset_canary_{ordinal + 1:02d}"
        source = tmp_path / f"source_{ordinal}"
        frames = [
            {
                "frame_index": frame_index,
                "camera_state": {
                    "frame_index": frame_index,
                    "pose_hash": f"pose-{ordinal}-{frame_index}",
                },
                "actor_states": [
                    {"actor_id": "source1_actor"},
                    {"actor_id": "source2_actor"},
                ],
            }
            for frame_index in range(75)
        ]
        suite = _write(
            source / "suite.json",
            {
                "schema": "avengine_optional_spear_apartment_suite_v1",
                "native_map": "/Game/Test/Apartment",
                "scenarios": [
                    {
                        "scenario_id": episode_id,
                        "plan": {"frames": frames},
                    }
                ],
            },
        )
        wav = source / "audio.wav"
        _float_wav(wav)
        rir = _write(
            source / "rir.json",
            {
                "jobs": [
                    {"source_slot_id": "source1"},
                    {"source_slot_id": "source2"},
                ]
            },
        )
        cache = _write(
            source / "cache.json",
            {
                "status": "pass",
                "full_plan_complete": True,
                "selected_job_count": 2,
            },
        )
        delivery = _write(
            source / "delivery.json",
            {"status": "pass", "episode_count": 1, "qualification_claim": False},
        )
        rows.append(
            {
                "ordinal": ordinal,
                "episode_id": episode_id,
                "mechanism": "static",
                "target_source_slot": "source1",
                "target_side": "left" if ordinal else "right",
                "speech_frame_window_inclusive": [7, 31],
                "suite_plan": str(suite),
                "audio_wav": str(wav),
                "output_root": str(output / "episodes" / f"{ordinal:02d}_{episode_id}"),
                "acoustic_evidence": {
                    "exact_rir_plan": str(rir),
                    "rir_cache": str(cache),
                    "binaural_delivery": str(delivery),
                },
            }
        )
    request = {
        "schema": RUNNER.REQUEST_SCHEMA,
        "purpose": "segmentation_reset_canary",
        "batch_id": "reset_canary_v1",
        "episode_count": 2,
        "native_map": "/Game/Test/Apartment",
        "output_root": str(output),
        "physical_gpu_index": 1,
        "graphics_adapter_argument": 1,
        "forbidden_physical_gpu_indices": [0, 3],
        "stop_on_first_fail": True,
        "room_loaded_once": True,
        "fresh_actors_per_episode": True,
        "segmentation_reset_and_negative_check_per_episode": True,
        "cpu_finalize_queue_depth": 2,
        "raw_memmap_contract": RUNNER.RAW_MEMMAP_CONTRACT,
        "raw_memmap_total_bytes": RUNNER.RAW_MEMMAP_TOTAL_BYTES,
        "cpu_worker_policy": RUNNER.CPU_WORKER_POLICY,
        "runtime_environments": RUNNER.RUNTIME_ENVIRONMENTS,
        "episodes": rows,
        "formal_episode_count": 0,
        "qualification_claim": False,
        "execution_authorized": True,
        "ground_contact_release_qualified": False,
        "release_blockers": [
            "ground-contact release qualification remains separate",
            "motion-realism release qualification remains separate",
        ],
        "motion_realism_release_qualified": False,
    }
    return _write(tmp_path / "request.json", request)


def _teardown() -> dict[str, object]:
    return {
        "actors_destroyed": True,
        "segmentation_terminated": True,
        "prior_stable_names_absent": True,
        "prior_actor_handles_absent": True,
        "prior_stable_actor_names_absent": True,
        "prior_proxy_descriptors_absent": True,
        "proxy_filters_cleared": True,
        "show_only_list_cleared": True,
        "remaining_controlled_actor_handle_count": 0,
        "remaining_controlled_stable_name_count": 0,
        "remaining_controlled_proxy_descriptor_count": 0,
    }


def _final_receipt(batch, episode: object, path: Path, raw_ready: Path) -> Path:
    contract = {
        "normal_rgb_frames": 75,
        "normal_metric_depth_frames": 75,
        "source1_target_only_depth_frames": 75,
        "source2_target_only_depth_frames": 75,
        "normal_runtime_readbacks": 75,
        "target_only_runtime_readbacks": 150,
        "live_asset_readback": True,
    }
    manifest = raw_ready.parents[1] / "finalized_output" / "manifest.json"
    _write(
        manifest,
        {
            "schema": ("avengine_native_strict_two_human_raw_finalization_manifest_v1"),
            "status": "pass",
            "episode_id": episode.episode_id,
            "input_binding_sha256": episode.bindings["binding_sha256"],
            "capture_contract": contract,
            "formal_episode_count": 0,
            "qualification_claim": False,
            "ground_contact_release_qualified": False,
            "motion_realism_release_qualified": batch.request[
                "motion_realism_release_qualified"
            ],
        },
    )
    value = {
        "schema": RUNNER.FINAL_RECEIPT_SCHEMA,
        "status": "pass",
        "episode_id": episode.episode_id,
        "batch_request_sha256": batch.request_sha256,
        "input_binding_sha256": episode.bindings["binding_sha256"],
        "capture_contract": contract,
        "raw_ready": str(raw_ready.resolve()),
        "raw_ready_sha256": RUNNER._sha256(raw_ready),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": RUNNER._sha256(manifest),
        "finalized_output": str(manifest.parent.resolve()),
        "formal_episode_count": 0,
        "qualification_claim": False,
        "ground_contact_release_qualified": False,
        "motion_realism_release_qualified": batch.request[
            "motion_realism_release_qualified"
        ],
    }
    return _write(path, value)


def test_resolve_two_episode_canary_freezes_room_audio_rir_and_raw_contract(
    tmp_path: Path,
) -> None:
    batch = RUNNER.resolve_request(_request_fixture(tmp_path))
    assert batch.purpose == "segmentation_reset_canary"
    assert len(batch.episodes) == 2
    assert {item.mechanism for item in batch.episodes} == {"both_static"}
    assert all(
        item.bindings["acoustics"]["status"] == "pass_precomputed_before_gpu"
        for item in batch.episodes
    )
    plan = RUNNER.resolved_plan(batch)
    assert plan["room_process_launch_count"] == 1
    assert plan["raw_memmap_total_bytes_per_episode"] == 691_200_000
    assert plan["gates"]["production_requires_two_episode_reset_canary"] is True


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("spear_capture_python", "/data/jzy/code/AVEngine-lead-a/.venv/bin/python"),
        ("avengine_cpu_python", "/data/jzy/code/AVEngine-lead-a/.venv/bin/python"),
    ],
)
def test_repository_local_venv_is_rejected_for_batch_runtime(
    tmp_path: Path, key: str, replacement: str
) -> None:
    request_path = _request_fixture(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["runtime_environments"][key] = replacement
    _write(request_path, request)
    with pytest.raises(RuntimeError, match="official Conda runtime"):
        RUNNER.resolve_request(request_path)


def test_production_batch_cannot_bypass_two_episode_canary(tmp_path: Path) -> None:
    request_path = _request_fixture(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["purpose"] = "production_room_shard"
    request["motion_realism_release_qualified"] = True
    request["episode_count"] = 10
    request["episodes"] = request["episodes"] * 5
    for ordinal, row in enumerate(request["episodes"]):
        row = dict(row)
        row["ordinal"] = ordinal
        row["episode_id"] = f"production_{ordinal:02d}"
        row["output_root"] = str(
            Path(request["output_root"])
            / "episodes"
            / f"{ordinal:02d}_production_{ordinal:02d}"
        )
        suite = json.loads(Path(row["suite_plan"]).read_text(encoding="utf-8"))
        suite["scenarios"][0]["scenario_id"] = row["episode_id"]
        suite_path = tmp_path / f"production_suite_{ordinal}.json"
        _write(suite_path, suite)
        row["suite_plan"] = str(suite_path)
        request["episodes"][ordinal] = row
    _write(request_path, request)
    with pytest.raises(RuntimeError, match="two-Episode reset canary receipt"):
        RUNNER.resolve_request(request_path)


def test_execute_requires_explicit_request_authorization(tmp_path: Path) -> None:
    request_path = _request_fixture(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["execution_authorized"] = False
    _write(request_path, request)
    batch = RUNNER.resolve_request(request_path)
    with pytest.raises(RuntimeError, match="execution is not authorized"):
        RUNNER.execute_batch(
            batch,
            session_factory=lambda _: _Session(),
            finalize_queue_factory=lambda _: _Queue(),
            resume=False,
        )


def test_dynamic_production_row_requires_independent_motion_realism_receipt() -> None:
    with pytest.raises(RuntimeError, match="lacks motion-realism evidence"):
        RUNNER._validate_motion_realism(
            {},
            episode_id="dynamic_01",
            mechanism="target_moves",
            purpose="production_room_shard",
        )


def test_motion_realism_receipt_requires_speed_active_and_phase_gates(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "motion.json",
        {
            "schema": "avengine_strict_two_human_motion_realism_receipt_v1",
            "status": "pass",
            "episode_id": "dynamic_01",
            "mechanism": "both_move",
            "release_qualified": True,
            "no_global_time_stretch": True,
            "active_interval_gate": {"status": "pass", "frame_interval": [10, 50]},
            "speed_gate": {"status": "pass", "measured_speed_mps": 1.1},
            "clip_phase_foot_plant_sync_gate": {"status": "fail"},
        },
    )
    with pytest.raises(RuntimeError, match="motion-realism release gate failed"):
        RUNNER._validate_motion_realism(
            {"motion_realism_evidence": str(path)},
            episode_id="dynamic_01",
            mechanism="both_move",
            purpose="production_room_shard",
        )


def test_motion_realism_receipt_binds_native_rate_active_speed(
    tmp_path: Path,
) -> None:
    interval = [10, 50]
    path = _write(
        tmp_path / "motion_pass.json",
        {
            "schema": "avengine_strict_two_human_motion_realism_receipt_v1",
            "status": "pass",
            "episode_id": "dynamic_02",
            "mechanism": "target_moves",
            "release_qualified": True,
            "no_global_time_stretch": True,
            "active_interval_gate": {
                "status": "pass",
                "active_frame_interval_inclusive": interval,
                "active_frame_count": 41,
                "mapping_kind": "native_rate_active_interval",
                "active_speed_evaluated_only": True,
            },
            "speed_gate": {
                "status": "pass",
                "moving_actor_count": 1,
                "per_moving_actor": {
                    "source1_actor": {
                        "measured_active_speed_mps": 1.1,
                        "minimum_release_speed_mps": 0.8,
                        "maximum_release_speed_mps": 1.8,
                        "active_frame_interval_inclusive": interval,
                        "source_native_frame_interval_inclusive": [22, 62],
                    }
                },
            },
            "clip_phase_foot_plant_sync_gate": {
                "status": "pass",
                "per_moving_actor": {
                    "source1_actor": {
                        "phase_progression_monotonic": True,
                        "foot_plant_sync": True,
                        "phase_freeze_detected": False,
                    }
                },
            },
        },
    )
    result = RUNNER._validate_motion_realism(
        {"motion_realism_evidence": str(path)},
        episode_id="dynamic_02",
        mechanism="target_moves",
        purpose="production_room_shard",
    )
    assert result["status"] == "pass_release_qualified"
    assert result["speed_gate"]["moving_actor_count"] == 1


@pytest.mark.parametrize(
    ("mechanism", "per_slot"),
    [
        ("target_moves", {"source1": 27, "source2": 1}),
        ("distractor_moves", {"source1": 1, "source2": 16}),
        ("both_move", {"source1": 11, "source2": 12}),
    ],
)
def test_acoustics_derives_native_rate_counts_from_bound_plan(
    tmp_path: Path, mechanism: str, per_slot: dict[str, int]
) -> None:
    jobs = [
        {"source_slot_id": slot, "job_id": f"{slot}-{index}"}
        for slot, count in per_slot.items()
        for index in range(count)
    ]
    plan = _write(
        tmp_path / "rir.json",
        {
            "jobs": jobs,
            "unique_rir_job_count": len(jobs),
            "distinct_rir_state_count_by_source_slot": per_slot,
        },
    )
    cache = _write(
        tmp_path / "cache.json",
        {
            "status": "pass",
            "full_plan_complete": True,
            "selected_job_count": len(jobs),
        },
    )
    delivery = _write(
        tmp_path / "delivery.json",
        {"status": "pass", "episode_count": 1, "qualification_claim": False},
    )
    result = RUNNER._validate_acoustics(
        {
            "acoustic_evidence": {
                "exact_rir_plan": str(plan),
                "rir_cache": str(cache),
                "binaural_delivery": str(delivery),
            }
        },
        mechanism=mechanism,
    )
    assert result["canonical_mechanism"] == mechanism
    assert result["expected_unique_rir_job_count"] == len(jobs)
    assert result["expected_rir_count_by_source_slot"] == per_slot


def test_acoustics_rejects_cache_count_not_bound_to_actual_plan(
    tmp_path: Path,
) -> None:
    plan = _write(
        tmp_path / "rir.json",
        {
            "jobs": [
                {"source_slot_id": "source1"},
                {"source_slot_id": "source2"},
            ]
        },
    )
    cache = _write(
        tmp_path / "cache.json",
        {"status": "pass", "full_plan_complete": True, "selected_job_count": 76},
    )
    delivery = _write(
        tmp_path / "delivery.json",
        {"status": "pass", "episode_count": 1, "qualification_claim": False},
    )
    with pytest.raises(RuntimeError, match="exact RIR cache is incomplete"):
        RUNNER._validate_acoustics(
            {
                "acoustic_evidence": {
                    "exact_rir_plan": str(plan),
                    "rir_cache": str(cache),
                    "binaural_delivery": str(delivery),
                }
            },
            mechanism="target_moves",
        )


def test_both_move_accepts_distinct_native_intervals_per_actor(
    tmp_path: Path,
) -> None:
    intervals = {
        "source1_actor": [14, 24],
        "source2_actor": [14, 25],
    }
    active_by_actor = {
        actor_id: {
            "active_frame_interval_inclusive": interval,
            "active_frame_count": interval[1] - interval[0] + 1,
            "mapping_kind": "native_rate_active_interval",
            "active_speed_evaluated_only": True,
        }
        for actor_id, interval in intervals.items()
    }
    speed_by_actor = {
        actor_id: {
            "measured_active_speed_mps": 0.8,
            "minimum_release_speed_mps": 0.7,
            "maximum_release_speed_mps": 1.2,
            "active_frame_interval_inclusive": interval,
            "source_native_frame_interval_inclusive": [
                39 if actor_id == "source1_actor" else 62,
                49 if actor_id == "source1_actor" else 73,
            ],
        }
        for actor_id, interval in intervals.items()
    }
    phase_by_actor = {
        actor_id: {
            "phase_progression_monotonic": True,
            "foot_plant_sync": True,
            "phase_freeze_detected": False,
        }
        for actor_id in intervals
    }
    path = _write(
        tmp_path / "motion_both.json",
        {
            "schema": "avengine_strict_two_human_motion_realism_receipt_v1",
            "status": "pass",
            "episode_id": "dynamic_both",
            "mechanism": "both_move",
            "release_qualified": True,
            "no_global_time_stretch": True,
            "active_interval_gate": {
                "status": "pass",
                "per_moving_actor": active_by_actor,
            },
            "speed_gate": {
                "status": "pass",
                "moving_actor_count": 2,
                "per_moving_actor": speed_by_actor,
            },
            "clip_phase_foot_plant_sync_gate": {
                "status": "pass",
                "per_moving_actor": phase_by_actor,
            },
        },
    )
    result = RUNNER._validate_motion_realism(
        {"motion_realism_evidence": str(path)},
        episode_id="dynamic_both",
        mechanism="both_move",
        purpose="production_room_shard",
    )
    assert result["status"] == "pass_release_qualified"
    assert set(result["active_interval_gate"]["per_moving_actor"]) == set(intervals)


def test_legacy_mechanism_aliases_normalize_to_canonical_names() -> None:
    assert RUNNER._canonical_mechanism("static") == "both_static"
    assert RUNNER._canonical_mechanism("camera_pan") == "camera_pan_both_static"


def test_raw_spool_publishes_ready_last_with_exact_tiny_memmaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny = {
        "normal_depth_m.f16le": {
            "shape": [75, 2, 3],
            "dtype": "<f2",
            "size_bytes": 900,
            "semantics": "normal_scene_metric_depth_m",
        },
        "target_only_source1_depth_m.f16le": {
            "shape": [75, 2, 3],
            "dtype": "<f2",
            "size_bytes": 900,
            "semantics": "source1_show_only_metric_depth_m",
        },
        "target_only_source2_depth_m.f16le": {
            "shape": [75, 2, 3],
            "dtype": "<f2",
            "size_bytes": 900,
            "semantics": "source2_show_only_metric_depth_m",
        },
        "normal_object_ids.u32le": {
            "shape": [75, 2, 3],
            "dtype": "<u4",
            "size_bytes": 1800,
            "semantics": "normal_scene_raw_object_ids_uint32",
        },
    }
    monkeypatch.setattr(SPOOL, "RAW_MEMMAP_CONTRACT", tiny)
    monkeypatch.setattr(SPOOL, "RAW_MEMMAP_TOTAL_BYTES", 4500)
    monkeypatch.setattr(SPOOL, "HEIGHT", 2)
    monkeypatch.setattr(SPOOL, "WIDTH", 3)
    attempt = tmp_path / "attempt_001"
    attempt.mkdir()
    with SPOOL.RawSpoolWriter(attempt) as writer:
        for index in range(75):
            writer.write_frame("normal_depth", index, np.full((2, 3), index + 1))
            writer.write_frame(
                "target_only_source1_depth", index, np.full((2, 3), index + 2)
            )
            writer.write_frame(
                "target_only_source2_depth", index, np.full((2, 3), index + 3)
            )
            writer.write_frame(
                "normal_object_ids", index, np.full((2, 3), index, dtype=np.uint32)
            )
            writer.rgb_path(index).write_bytes(b"PNG" + bytes([index]))
        for name in (
            "runtime_readbacks.json",
            "runtime_asset_readbacks.json",
            "normal_object_id_descriptors.json",
            "capture_context.json",
        ):
            writer.write_metadata(name, {"status": "pass"})
        ready = writer.publish_ready(
            batch_request_sha256="a" * 64,
            episode_id="episode",
            input_binding_sha256="b" * 64,
            teardown={
                "actors_destroyed": True,
                "segmentation_terminated": True,
                "prior_stable_names_absent": True,
                "prior_actor_handles_absent": True,
                "prior_stable_actor_names_absent": True,
                "prior_proxy_descriptors_absent": True,
                "proxy_filters_cleared": True,
                "show_only_list_cleared": True,
                "remaining_controlled_actor_handle_count": 0,
                "remaining_controlled_stable_name_count": 0,
                "remaining_controlled_proxy_descriptor_count": 0,
            },
            motion_realism_release_qualified=False,
        )
    receipt = json.loads(ready.read_text(encoding="utf-8"))
    assert receipt["status"] == "pass_raw_ready"
    assert receipt["raw_memmap_total_bytes"] == 4500
    assert receipt["persistence"]["data_files_fsynced"] is True
    assert (
        sum((ready.parent / name).stat().st_size for name in SPOOL.PASS_FILE.values())
        == 4500
    )


class _FrameBoundary:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


class _Object:
    def __init__(self, handle: int) -> None:
        self.uobject = handle


class _Manager:
    def __init__(self) -> None:
        self.allowed_actors = []
        self.allowed_components = []
        self.ignored_actors = []
        self.ignored_components = []

    def SetAllowedActors(self, *, AllowedActors):
        self.allowed_actors = list(AllowedActors)

    def SetAllowedComponents(self, *, AllowedComponents):
        self.allowed_components = list(AllowedComponents)

    def SetIgnoredActors(self, *, IgnoredActors):
        self.ignored_actors = list(IgnoredActors)

    def SetIgnoredComponents(self, *, IgnoredComponents):
        self.ignored_components = list(IgnoredComponents)


class _Depth:
    def __init__(self) -> None:
        self.PrimitiveRenderMode = "PRM_UseShowOnlyList"
        self.ShowOnlyActors = [_Object(99)]

    def get_property_value(self, *, property_name, as_value):
        assert as_value is True
        return getattr(self, property_name)


class _Segmentation:
    def __init__(self) -> None:
        self.proxy_component_manager = _Manager()
        self.descriptors = []

    def initialize(self) -> None:
        self.proxy_component_manager = _Manager()

    def terminate(self) -> None:
        self.proxy_component_manager = None

    def get_mesh_proxy_geometry_descs(self, **_):
        return list(self.descriptors)

    def get_allowed_actors(self):
        return self.proxy_component_manager.allowed_actors

    def get_allowed_components(self):
        return self.proxy_component_manager.allowed_components

    def get_ignored_actors(self):
        return self.proxy_component_manager.ignored_actors

    def get_ignored_components(self):
        return self.proxy_component_manager.ignored_components


class _Unreal:
    def __init__(self) -> None:
        self.handles = {11, 12, 21, 22}
        self.stable_names = {"old_source1", "old_source2"}

    def find_actors(self, *, as_handle):
        assert as_handle is True
        return list(self.handles)

    def find_actors_as_dict(self, *, include_unreal_name):
        assert include_unreal_name is False
        return {name: object() for name in self.stable_names}


class _Game:
    def __init__(self) -> None:
        self.segmentation_service = _Segmentation()
        self.unreal_service = _Unreal()


class _Instance:
    def begin_frame(self):
        return _FrameBoundary()

    def end_frame(self):
        return _FrameBoundary()

    def step(self, *, num_frames):
        assert num_frames == 2


class _ActorRunner:
    def __init__(self, game: _Game) -> None:
        self.game = game

    def _destroy_runtime_actors(self, instance, runtimes):
        assert isinstance(instance, _Instance)
        assert set(runtimes) == {"source1_actor", "source2_actor"}
        self.game.unreal_service.handles.clear()
        self.game.unreal_service.stable_names.clear()
        self.game.segmentation_service.descriptors.clear()


def test_lifecycle_proves_actor_proxy_and_show_only_cleanup() -> None:
    game = _Game()
    instance = _Instance()
    depth = _Depth()
    runtimes = {
        "source1_actor": {"visual_actor": _Object(11), "anchor": _Object(12)},
        "source2_actor": {"visual_actor": _Object(21), "anchor": _Object(22)},
    }
    receipt = LIFECYCLE.teardown_episode(
        instance=instance,
        game=game,
        runner=_ActorRunner(game),
        runtimes=runtimes,
        depth_component=depth,
        stable_names=["old_source1", "old_source2"],
    )
    assert receipt["prior_actor_handles_absent"] is True
    assert receipt["prior_stable_actor_names_absent"] is True
    assert receipt["prior_proxy_descriptors_absent"] is True
    assert receipt["proxy_filters_cleared"] is True
    assert receipt["show_only_list_cleared"] is True
    assert receipt["remaining_controlled_actor_handle_count"] == 0
    assert game.segmentation_service.proxy_component_manager is None


class _ImmediateFuture:
    def __init__(self, path: Path) -> None:
        self.path = path

    def done(self) -> bool:
        return True

    def result(self) -> Path:
        return self.path


class _Queue:
    def __init__(self) -> None:
        self.closed = False

    def submit(self, *, batch, episode, raw_ready, attempt_root):
        path = episode.output_root / "FINAL_READY.json"
        return _ImmediateFuture(_final_receipt(batch, episode, path, raw_ready))

    def close(self) -> None:
        self.closed = True


class _ExplodingFuture:
    def done(self) -> bool:
        return True

    def result(self) -> Path:
        raise RuntimeError("synthetic CPU finalizer failure")


class _FailFirstQueue(_Queue):
    def __init__(self) -> None:
        super().__init__()
        self.submit_count = 0

    def submit(self, *, batch, episode, raw_ready, attempt_root):
        self.submit_count += 1
        if self.submit_count == 1:
            return _ExplodingFuture()
        return super().submit(
            batch=batch,
            episode=episode,
            raw_ready=raw_ready,
            attempt_root=attempt_root,
        )


class _Session:
    def __init__(self, *, fail_on: int | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[str] = []
        self.close_count = 0

    def capture_episode_raw(self, *, episode, attempt_root, batch):
        self.calls.append(episode.episode_id)
        if self.fail_on == len(self.calls):
            raise RuntimeError("synthetic native failure")
        ready = attempt_root / "raw_spool" / "RAW_READY.json"
        _write(ready, {"status": "synthetic", "episode_teardown": _teardown()})
        return ready

    def close(self) -> None:
        self.close_count += 1


def test_state_machine_uses_one_session_and_per_episode_final_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = RUNNER.resolve_request(_request_fixture(tmp_path))
    session = _Session()
    monkeypatch.setattr(
        RUNNER, "validate_raw_ready_receipt", lambda *args, **kwargs: {}
    )
    receipt = RUNNER.execute_batch(
        batch,
        session_factory=lambda _: session,
        finalize_queue_factory=lambda _: _Queue(),
        resume=False,
    )
    result = json.loads(receipt.read_text(encoding="utf-8"))
    assert session.calls == [item.episode_id for item in batch.episodes]
    assert session.close_count == 1
    assert result["room_process_launch_count"] == 1
    assert result["episode_pass_count"] == 2
    assert all(
        (item.output_root / "FINAL_READY.json").is_file() for item in batch.episodes
    )


def test_state_machine_stops_after_first_native_failure_and_preserves_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = RUNNER.resolve_request(_request_fixture(tmp_path))
    session = _Session(fail_on=2)
    monkeypatch.setattr(
        RUNNER, "validate_raw_ready_receipt", lambda *args, **kwargs: {}
    )
    with pytest.raises(RuntimeError, match="synthetic native failure"):
        RUNNER.execute_batch(
            batch,
            session_factory=lambda _: session,
            finalize_queue_factory=lambda _: _Queue(),
            resume=False,
        )
    checkpoint = json.loads(
        (batch.output_root / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert session.calls == [item.episode_id for item in batch.episodes]
    assert session.close_count == 1
    assert checkpoint["status"] == "fail_closed"
    assert "synthetic native failure" in checkpoint["error"]


def test_state_machine_observes_cpu_failure_before_starting_next_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = RUNNER.resolve_request(_request_fixture(tmp_path))
    session = _Session()
    monkeypatch.setattr(
        RUNNER, "validate_raw_ready_receipt", lambda *args, **kwargs: {}
    )
    with pytest.raises(RuntimeError, match="synthetic CPU finalizer failure"):
        RUNNER.execute_batch(
            batch,
            session_factory=lambda _: session,
            finalize_queue_factory=lambda _: _FailFirstQueue(),
            resume=False,
        )
    checkpoint = json.loads(
        (batch.output_root / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert session.calls == [batch.episodes[0].episode_id]
    assert checkpoint["failed_episode_id"] == batch.episodes[0].episode_id
    assert checkpoint["status"] == "fail_closed"


def test_resume_from_complete_raw_ready_does_not_open_spear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = RUNNER.resolve_request(_request_fixture(tmp_path))
    batch.output_root.mkdir(parents=True)
    _write(batch.output_root / "request_snapshot.json", batch.request)
    for episode in batch.episodes:
        ready = episode.output_root / "attempt_001" / "raw_spool" / "RAW_READY.json"
        _write(ready, {"status": "synthetic", "episode_teardown": _teardown()})
    monkeypatch.setattr(
        RUNNER, "validate_raw_ready_receipt", lambda *args, **kwargs: {}
    )

    def forbidden_factory(_):
        raise AssertionError("resume should not open SPEAR")

    receipt = RUNNER.execute_batch(
        batch,
        session_factory=forbidden_factory,
        finalize_queue_factory=lambda _: _Queue(),
        resume=True,
    )
    assert receipt.is_file()
    assert json.loads(receipt.read_text(encoding="utf-8"))["episode_pass_count"] == 2


def test_resume_revalidates_raw_ready_digest_before_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = RUNNER.resolve_request(_request_fixture(tmp_path))
    session = _Session()
    monkeypatch.setattr(
        RUNNER, "validate_raw_ready_receipt", lambda *args, **kwargs: {}
    )
    RUNNER.execute_batch(
        batch,
        session_factory=lambda _: session,
        finalize_queue_factory=lambda _: _Queue(),
        resume=False,
    )
    first_final = json.loads(
        (batch.episodes[0].output_root / "FINAL_READY.json").read_text(encoding="utf-8")
    )
    Path(first_final["raw_ready"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="raw-ready digest drift"):
        RUNNER.execute_batch(
            batch,
            session_factory=lambda _: (_ for _ in ()).throw(
                AssertionError("resume must fail before opening SPEAR")
            ),
            finalize_queue_factory=lambda _: _Queue(),
            resume=True,
        )


def test_resume_preserves_partial_attempt_and_allocates_next_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = RUNNER.resolve_request(_request_fixture(tmp_path))
    batch.output_root.mkdir(parents=True)
    _write(batch.output_root / "request_snapshot.json", batch.request)
    partial = batch.episodes[0].output_root / "attempt_001" / "raw_spool"
    partial.mkdir(parents=True)
    (partial / "partial.bin").write_bytes(b"preserve")
    session = _Session()
    monkeypatch.setattr(
        RUNNER, "validate_raw_ready_receipt", lambda *args, **kwargs: {}
    )
    RUNNER.execute_batch(
        batch,
        session_factory=lambda _: session,
        finalize_queue_factory=lambda _: _Queue(),
        resume=True,
    )
    assert (partial / "partial.bin").read_bytes() == b"preserve"
    assert (
        batch.episodes[0].output_root / "attempt_002/raw_spool/RAW_READY.json"
    ).is_file()
