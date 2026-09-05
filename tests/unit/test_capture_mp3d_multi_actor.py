from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import json
import numpy as np
import pytest

from avengine.rooms.contracts import ValidatedM1Inputs
import avengine.capture.mp3d_multi_actor as capture


class _Node:
    def __init__(self, events: list[str], label: str) -> None:
        self._events = events
        self._label = label
        self.matrix = np.eye(4, dtype=np.float64)

    def absolute_transformation(self) -> np.ndarray:
        self._events.append(self._label)
        return self.matrix.copy()


class _Actor:
    def __init__(self, events: list[str], index: int) -> None:
        self.index = index
        self.root_scene_node = _Node(events, f"root-{index}")
        self._emitter = _Node(events, f"emitter-{index}")
        self._joint_positions = np.zeros(4, dtype=np.float64)

    @property
    def joint_positions(self) -> np.ndarray:
        return self._joint_positions.copy()

    @joint_positions.setter
    def joint_positions(self, value: Any) -> None:
        self._joint_positions = np.asarray(value, dtype=np.float64).reshape(-1).copy()

    def get_link_ids(self) -> list[int]:
        return [41]

    def get_link_name(self, link_id: int) -> str:
        if link_id != 41:
            raise KeyError(link_id)
        return "emitter"

    def get_link_scene_node(self, link_id: int) -> _Node:
        if link_id != 41:
            raise KeyError(link_id)
        return self._emitter


class _Binding:
    def map_pose(self, rotations: Any) -> np.ndarray:
        return np.asarray(rotations, dtype=np.float64).reshape(-1)


class _ActionClip:
    sample_count = 2
    loop_duration_ticks = 6_400
    sample_ticks = (0, 3_200)
    rotations_xyzw = np.asarray(
        [
            [[0.0, 0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0, 1.0]],
        ],
        dtype=np.float64,
    )


class _ActionSet:
    def action(self, action_id: str) -> _ActionClip:
        assert action_id == "idle"
        return _ActionClip()


def _fake_bundle() -> Any:
    return SimpleNamespace(
        joint_mapping={"runtime_joint_order": ("j0",)},
        action_roles_by_id={"idle": "idle_poses"},
        action_sets_by_role={"idle_poses": _ActionSet()},
    )


class _Pathfinder:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.is_loaded = False

    def load_nav_mesh(self, path: str) -> bool:
        self._events.append("load_nav_mesh")
        self.is_loaded = Path(path).is_file()
        return self.is_loaded


class _Simulator:
    def __init__(self, events: list[str], semantic_id: int) -> None:
        self._events = events
        self.pathfinder = _Pathfinder(events)
        self.sensors = {"rgb": object(), "depth": object(), "semantic": object()}
        self.semantic_id = semantic_id
        self.render_count = 0

    def __enter__(self) -> _Simulator:
        self._events.append("sim_enter")
        return self

    def __exit__(self, *_args: object) -> None:
        self._events.append("sim_exit")

    def initialize_agent(self, index: int, state: Any) -> Any:
        assert index == 0
        self._events.append("initialize_agent")
        return SimpleNamespace(get_state=lambda: state)

    def get_world_time(self) -> float:
        return 0.0

    def render_sensors(self, sensors: list[Any]) -> dict[str, np.ndarray]:
        assert len(sensors) == 3
        self._events.append(f"render-{self.render_count}")
        self.render_count += 1
        semantic = np.asarray(
            [[self.semantic_id, self.semantic_id], [0, 0]], dtype=np.int32
        )
        return {
            "rgb": np.dstack([
                np.full((2, 2, 3), self.render_count, dtype=np.uint8),
                np.full((2, 2), 255, dtype=np.uint8),
            ]),
            "depth": np.full((2, 2), 1.5, dtype=np.float32),
            "semantic": semantic,
        }


def _require_articulation_backend(configuration, simulator):
    # A static-room M1 config otherwise selects BasePhysicsManager, which
    # cannot instantiate even kinematic articulated objects in real Habitat.
    assert configuration.sim_cfg.enable_physics is True
    return simulator


class _AgentState:
    def __init__(self) -> None:
        self.position = None
        self.rotation = None


class _Quaternion:
    @staticmethod
    def quaternion(*values: float) -> tuple[float, ...]:
        return tuple(float(value) for value in values)


def _clock(frame_count: int = 2) -> dict[str, int | float]:
    return {
        "frame_count": frame_count,
        "frame_rate_hz": 15,
        "ticks_per_frame": 3_200,
        "time_base_hz": 48_000,
        "sample_rate_hz": 16_000,
        "sample_count": round(frame_count * 16_000 / 15),
    }


def _tracks(actor_count: int, frame_count: int) -> tuple[dict[str, Any], ...]:
    tracks: list[dict[str, Any]] = []
    for actor_index in range(actor_count):
        frames: list[dict[str, Any]] = []
        for frame_index in range(frame_count):
            frames.append(
                {
                    "frame_index": frame_index,
                    "planned_world_from_skin_root": {
                        "translation_m": [float(actor_index + frame_index), 0.0, 0.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "joint_targets": [
                        {"joint_id": "j0", "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}
                    ],
                    "action_id": "idle",
                    "action_time_ticks": (frame_index % 2) * 3_200,
                    "action_sample_index": frame_index % 2,
                    "planned_route_center_m": [1_000.0 + actor_index, 2_000.0, 3_000.0],
                }
            )
        tracks.append(
            {
                "actor_id": f"actor-{actor_index}",
                "source_slot_id": f"source{actor_index + 1}",
                "source_endpoint_id": f"endpoint{actor_index + 1}",
                "semantic_id": 100 + actor_index,
                "asset": {
                    "asset_id": f"asset-{actor_index}",
                    "asset_manifest_path": f"/external/asset-{actor_index}.json",
                    "base_m2_request_path": f"/external/request-{actor_index}.json",
                    "runtime_joint_order": ["j0"],
                },
                "emitter": {"joint_id": "emitter"},
                "frames": frames,
            }
        )
    return tuple(tracks)


@pytest.mark.parametrize("actor_count", [1, 3])
def test_public_entrypoint_reads_native_state_for_configured_actor_count(
    tmp_path, monkeypatch, actor_count
):
    frame_count = 2
    events: list[str] = []
    tracks = _tracks(actor_count, frame_count)
    case = {
        "clock": _clock(frame_count),
    }
    room_request = {
        "primary_camera_rig": {
            "world_from_rig": {
                "translation_m": [0.0, 1.0, 2.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        }
    }
    room_inputs = ValidatedM1Inputs(
        room_path=tmp_path / "room.json",
        request_path=tmp_path / "m1.json",
        room={"room_id": "17DRP5sb8fy"},
        request=room_request,
    )
    (tmp_path / "room.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "m1.json").write_text("{}\n", encoding="utf-8")
    case_path = tmp_path / "planned_case.json"
    case_path.write_text("{}\n", encoding="utf-8")
    navmesh = tmp_path / "scene.navmesh"
    navmesh.write_bytes(b"fixture")

    monkeypatch.setattr(
        capture,
        "_load_case_and_m1",
        lambda **_kwargs: (case, tuple({"value": track} for track in tracks), room_inputs),
    )
    monkeypatch.setattr(
        capture,
        "_load_track_runtime",
        lambda track, *, cache: (
            SimpleNamespace(asset={"asset_id": track["asset"]["asset_id"], "anchors": [{
                "anchor_id": track["emitter"].get("anchor_id"),
                "joint_id": track["emitter"]["joint_id"],
                "joint_from_anchor": {"translation_m": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1]},
            }]}),
            _fake_bundle(),
        ),
    )
    monkeypatch.setattr(capture, "_base_template_handle", lambda *_args, **_kwargs: "base")
    actors: list[_Actor] = []

    def instantiate(*_args: Any, semantic_id: int, actor_index: int, **_kwargs: Any):
        assert semantic_id == 100 + actor_index
        actor = _Actor(events, actor_index)
        actors.append(actor)
        return actor, _Binding()

    monkeypatch.setattr(capture, "_instantiate_actor_with_semantic_template", instantiate)

    def apply_root(actor: _Actor, root: np.ndarray, **_kwargs: Any) -> None:
        events.append(f"apply-{actor.index}")
        actor.root_scene_node.matrix = root.copy()
        actor._emitter.matrix = root.copy()
        actor._emitter.matrix[:3, 3] += np.asarray([0.0, 0.25, 0.0])

    monkeypatch.setattr(capture, "_apply_root_with_habitat", apply_root)
    monkeypatch.setattr(capture, "validate_loaded_scene_asset_graph", lambda *_args, **_kwargs: ([], {}))
    monkeypatch.setattr(
        capture,
        "_make_configuration",
        lambda *_args, **_kwargs: (
            SimpleNamespace(sim_cfg=SimpleNamespace(gpu_device_id=None)),
            {"rgb": "rgb", "depth": "depth", "semantic": "semantic"},
            "listener0",
            {"navmesh": str(navmesh)},
        ),
    )
    monkeypatch.setattr(
        capture,
        "_state_snapshot",
        lambda *_args, **_kwargs: {"world_time_seconds": 0.0, "agent": {"readback": True}},
    )

    runtime = SimpleNamespace(
        mp3d_root=tmp_path,
        physics_config_path=tmp_path / "physics.json",
        habitat_sim=SimpleNamespace(AgentState=_AgentState),
        quaternion=_Quaternion(),
        magnum=SimpleNamespace(),
        quat_to_coeffs=lambda _rotation: np.asarray([0.0, 0.0, 0.0, 1.0]),
    )
    simulator = _Simulator(events, semantic_id=100)
    output = tmp_path / "native_capture"
    receipt = capture.capture_mp3d_multi_actor(
        case_manifest_path=case_path,
        room_manifest_path=room_inputs.room_path,
        m1_request_path=room_inputs.request_path,
        output_directory=output,
        gpu_device_id=2,
        runtime=runtime,
        simulator_factory=lambda configuration: _require_articulation_backend(configuration, simulator),
    )

    assert len(actors) == actor_count
    assert events.index("render-0") > events.index("root-0")
    assert events.index("render-0") < events.index("emitter-0")
    records = json.loads((output / "frame_records.json").read_text(encoding="utf-8"))
    assert len(records["frames"]) == frame_count
    assert records["source_endpoint_ids"] == [
        f"endpoint{index + 1}" for index in range(actor_count)]
    first = records["frames"][0]
    assert len(first["actor_readbacks"]) == actor_count
    # Emitter positions are read from the fake Habitat link, while planned route
    # centres are deliberately far away.  The output must preserve that boundary.
    assert first["source_positions_m"][0] == [0.0, 0.25, 0.0]
    assert first["actor_readbacks"][0]["planned_route_center_m"] == [1000.0, 2000.0, 3000.0]
    assert first["source_positions_m"][0] != first["actor_readbacks"][0]["planned_route_center_m"]
    assert np.load(output / "rgb.npy").shape == (frame_count, 2, 2, 3)
    assert first["modalities"]["rgb"]["shape"] == [2, 2, 3]
    assert np.load(output / "actor_root_readbacks.npy").shape == (frame_count, actor_count, 4, 4)
    for actor_index in range(actor_count):
        assert np.load(output / f"actor_joint_readbacks_source{actor_index + 1}.npy").shape == (frame_count, 4)
    assert receipt["inputs"]["case_manifest"] == str(case_path.resolve())
    assert receipt["object_id"]["status"] == "pending"
    last_slot = f"source{actor_count}"
    assert receipt["artifacts"]["actor_joint_readbacks_by_slot"][last_slot] == (
        f"actor_joint_readbacks_{last_slot}.npy"
    )
    assert receipt["capture"]["native_habitat_started"] is True
    assert receipt["capture"]["rgb_channel_order"] == "rgb"


def test_render_mutation_is_rejected_before_observed_output(tmp_path, monkeypatch):
    events: list[str] = []
    tracks = _tracks(2, 1)
    case = {"clock": _clock(1)}
    room_inputs = ValidatedM1Inputs(
        room_path=tmp_path / "room.json",
        request_path=tmp_path / "m1.json",
        room={"room_id": "17DRP5sb8fy"},
        request={
            "primary_camera_rig": {
                "world_from_rig": {
                    "translation_m": [0.0, 1.0, 2.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        },
    )
    for path in (room_inputs.room_path, room_inputs.request_path):
        path.write_text("{}\n", encoding="utf-8")
    case_path = tmp_path / "case.json"
    case_path.write_text("{}\n", encoding="utf-8")
    navmesh = tmp_path / "scene.navmesh"
    navmesh.write_bytes(b"fixture")
    monkeypatch.setattr(capture, "_load_case_and_m1", lambda **_kwargs: (case, tuple({"value": track} for track in tracks), room_inputs))
    monkeypatch.setattr(
        capture,
        "_load_track_runtime",
        lambda track, *, cache: (
            SimpleNamespace(asset={"asset_id": track["asset"]["asset_id"], "anchors": [{
                    "anchor_id": track["emitter"].get("anchor_id"),
                    "joint_id": track["emitter"]["joint_id"],
                    "joint_from_anchor": {"translation_m": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1]},
                }]}),
            _fake_bundle(),
        ),
    )
    monkeypatch.setattr(capture, "_base_template_handle", lambda *_args, **_kwargs: "base")
    actors: list[_Actor] = []
    monkeypatch.setattr(
        capture,
        "_instantiate_actor_with_semantic_template",
        lambda *_args, actor_index, **_kwargs: (actors.append(_Actor(events, actor_index)) or actors[-1], _Binding()),
    )

    def apply_root(actor: _Actor, root: np.ndarray, **_kwargs: Any) -> None:
        actor.root_scene_node.matrix = root.copy()
        actor._emitter.matrix = root.copy()

    monkeypatch.setattr(capture, "_apply_root_with_habitat", apply_root)
    monkeypatch.setattr(capture, "validate_loaded_scene_asset_graph", lambda *_args, **_kwargs: ([], {}))
    monkeypatch.setattr(
        capture,
        "_make_configuration",
        lambda *_args, **_kwargs: (
            SimpleNamespace(sim_cfg=SimpleNamespace(gpu_device_id=None)),
            {"rgb": "rgb", "depth": "depth", "semantic": "semantic"},
            "listener0",
            {"navmesh": str(navmesh)},
        ),
    )
    monkeypatch.setattr(capture, "_state_snapshot", lambda *_args, **_kwargs: {})

    class _MutatingSimulator(_Simulator):
        def render_sensors(self, sensors: list[Any]) -> dict[str, np.ndarray]:
            result = super().render_sensors(sensors)
            actors[0].root_scene_node.matrix[0, 3] += 0.1
            return result

    runtime = SimpleNamespace(
        mp3d_root=tmp_path,
        physics_config_path=tmp_path / "physics.json",
        habitat_sim=SimpleNamespace(AgentState=_AgentState),
        quaternion=_Quaternion(),
        magnum=SimpleNamespace(),
        quat_to_coeffs=lambda _rotation: np.asarray([0.0, 0.0, 0.0, 1.0]),
    )
    with pytest.raises(capture.MP3DMultiActorCaptureError, match="changed during render"):
        capture.capture_mp3d_multi_actor(
            case_manifest_path=case_path,
            room_manifest_path=room_inputs.room_path,
            m1_request_path=room_inputs.request_path,
            output_directory=tmp_path / "mutating_capture",
            runtime=runtime,
            simulator_factory=lambda _configuration: _MutatingSimulator(events, semantic_id=100),
        )



def test_nonzero_anchor_offset_is_composed_with_observed_joint_rotation():
    actor = _Actor([], 0)
    actor._emitter.matrix = np.array([[0, -1, 0, 1], [1, 0, 0, 2],
                                      [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float)
    local = np.eye(4)
    local[0, 3] = 0.4
    assert capture._emitter_position(actor, 41, local) == pytest.approx([1, 2.4, 3])


def _write_planned_track(
    path: Path,
    *,
    slot: str,
    endpoint: str = "endpoint-a",
    semantic_id: int = 101,
) -> None:
    track = dict(_tracks(1, 2)[0])
    track.update({
        "schema": capture.ACTOR_TRACK_SCHEMA,
        "artifact_role": "planned_habitat_actor_apply_track",
        "native_observed": False,
        "research_only": True,
        "episode_counted": False,
        "clock": _clock(2),
        "source_slot_id": slot,
        "source_endpoint_id": endpoint,
        "semantic_id": semantic_id,
    })
    path.write_text(json.dumps(track), encoding="utf-8")


def test_case_loader_accepts_one_actor_with_arbitrary_safe_slot(tmp_path: Path):
    track_path = tmp_path / "track.json"
    _write_planned_track(track_path, slot="lead_voice")
    case_path = tmp_path / "case.json"
    case = {
        "schema": capture.CASE_SCHEMA,
        "artifact_role": "planned_habitat_actor_apply_case",
        "native_observed": False,
        "research_only": True,
        "episode_counted": False,
        "clock": _clock(2),
        "actor_tracks": [{"track_path": track_path.name}],
    }

    _, tracks = capture._resolve_case_track_paths(case_path, case)

    assert len(tracks) == 1
    assert tracks[0]["value"]["source_slot_id"] == "lead_voice"


def test_case_loader_rejects_slot_that_would_escape_output_name(tmp_path: Path):
    track_path = tmp_path / "track.json"
    _write_planned_track(track_path, slot="../escape")
    case_path = tmp_path / "case.json"
    case = {
        "schema": capture.CASE_SCHEMA,
        "artifact_role": "planned_habitat_actor_apply_case",
        "native_observed": False,
        "research_only": True,
        "episode_counted": False,
        "clock": _clock(2),
        "actor_tracks": [{"track_path": track_path.name}],
    }

    with pytest.raises(
        capture.MP3DMultiActorCaptureError, match="safe identifiers"
    ):
        capture._resolve_case_track_paths(case_path, case)
