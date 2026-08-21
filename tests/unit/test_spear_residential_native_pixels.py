from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools/m6z"))
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
TOOL_PATH = REPOSITORY / "tools/m6z/run_spear_residential_episode.py"
SPEC = importlib.util.spec_from_file_location("residential_pixels", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _readback(
    x: float = 0.0, yaw: float = 0.0, *, frame_index: int = 0
) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "location_cm": [x, 0.0, 0.0],
        "rotation_deg": [0.0, 0.0, yaw],
    }


def test_derive_native_pixel_masks_keeps_per_actor_target_footprints() -> None:
    normal = [np.asarray([[1.0, 2.0, 65504.0], [1.0, 65504.0, 3.0]], dtype=np.float32)]
    target = {
        "dog0": [
            np.asarray(
                [[1.0, 65504.0, 65504.0], [65504.0, 65504.0, 3.0]],
                dtype=np.float32,
            )
        ],
        "human0": [
            np.asarray(
                [[65504.0, 2.0, 65504.0], [1.0, 65504.0, 65504.0]],
                dtype=np.float32,
            )
        ],
    }

    modal, footprints = TOOL._derive_native_pixel_masks(
        normal_depths=normal,
        target_depths_by_actor=target,
        semantic_ids_by_actor={"dog0": 1, "human0": 2},
    )

    assert modal.tolist() == [[[1, 2, 0], [2, 0, 1]]]
    assert footprints["dog0"].tolist() == [[[1, 0, 0], [0, 0, 1]]]
    assert footprints["human0"].tolist() == [[[0, 2, 0], [2, 0, 0]]]


def test_multimodal_readback_drift_requires_normal_and_target_pose_identity() -> None:
    normal = [
        {
            "camera": _readback(),
            "actors": {"dog0": _readback(1.0), "human0": _readback(2.0, 10.0)},
        }
    ]
    target = {
        "dog0": [
            {
                "camera": _readback(),
                "actors": {
                    "dog0": _readback(1.0),
                    "human0": _readback(2.0, 10.0),
                },
            }
        ],
        "human0": [
            {
                "camera": _readback(),
                "actors": {
                    "dog0": _readback(1.0),
                    "human0": _readback(2.0, 10.0),
                },
            }
        ],
    }

    assert TOOL._maximum_multimodal_readback_drift(normal, target) == {
        "maximum_location_drift_cm": 0.0,
        "maximum_rotation_drift_deg": 0.0,
    }

    target["human0"][0]["actors"]["human0"]["location_cm"][0] = 0.001
    with pytest.raises(RuntimeError, match="target pass location drift"):
        TOOL._maximum_multimodal_readback_drift(normal, target)


def _episode_authority() -> dict[str, object]:
    return {
        "visual_plan": {
            "actors": [{"actor_id": "dog0"}, {"actor_id": "human0"}],
            "frames": [{} for _ in range(75)],
        },
        "timeline": {
            "frames": [
                {"view_pose_hashes": {"view0": f"pose-{index}"}} for index in range(75)
            ]
        },
    }


def test_finalize_native_pixel_artifacts_uses_dog_and_human_ids(
    tmp_path: Path,
) -> None:
    normal = [np.asarray([[1.0, 2.0]], dtype=np.float32) for _ in range(75)]
    target = {
        "dog0": [np.asarray([[1.0, 65504.0]], dtype=np.float32) for _ in range(75)],
        "human0": [np.asarray([[65504.0, 2.0]], dtype=np.float32) for _ in range(75)],
    }
    ids = [np.asarray([[1, 2]], dtype=np.uint32) for _ in range(75)]

    def readback(frame_index: int) -> dict[str, object]:
        return {
            "camera": _readback(frame_index=frame_index),
            "actors": {
                "dog0": _readback(1.0, frame_index=frame_index),
                "human0": _readback(2.0, frame_index=frame_index),
            },
        }

    result = TOOL._finalize_native_pixel_artifacts(
        output=tmp_path,
        episode=_episode_authority(),
        normal_depths=normal,
        normal_object_ids=ids,
        target_depths_by_actor=target,
        normal_readbacks=[readback(index) for index in range(75)],
        target_readbacks={
            "dog0": [readback(index) for index in range(75)],
            "human0": [readback(index) for index in range(75)],
        },
    )

    assert result["status"] == "pass"
    assert result["semantic_ids_by_actor"] == {"dog0": 1, "human0": 2}
    truth = (tmp_path / "pixel_visibility_truth.json").read_text()
    assert "dog0" in truth and "human0" in truth
    masks = np.load(tmp_path / "native_pixel_masks_depth_authority_v1.npz")
    assert not masks["modal_visible_dog0"].all()
    assert masks["modal_visible_dog0"][:, 0, 0].all()
    assert masks["modal_visible_human0"][:, 0, 1].all()

    mismatched = [readback(index) for index in range(75)]
    mismatched[7] = readback(8)
    with pytest.raises(RuntimeError, match="target-only frame index drift"):
        TOOL._maximum_multimodal_readback_drift(
            [readback(index) for index in range(75)], {"dog0": mismatched}
        )


class _FakeFrameContext:
    def __init__(self, events: list[object], name: str) -> None:
        self._events = events
        self._name = name

    def __enter__(self) -> None:
        self._events.append((self._name, "enter"))

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self._events.append((self._name, "exit"))
        return False


class _FakeInstance:
    def __init__(self, game: object, events: list[object]) -> None:
        self._game = game
        self._events = events

    def get_game(self) -> object:
        return self._game

    def begin_frame(self) -> _FakeFrameContext:
        return _FakeFrameContext(self._events, "begin")

    def end_frame(self) -> _FakeFrameContext:
        return _FakeFrameContext(self._events, "end")

    def step(self, *, num_frames: int) -> None:
        self._events.append(("step", num_frames))

    def close(self, *, force: bool) -> None:
        self._events.append(("close", force))


class _FakeDepthComponent:
    def __init__(self, events: list[object], actors: dict[str, object]) -> None:
        self._events = events
        self._actors = actors
        self._primitive_render_mode = ""
        self._show_only_actors: list[object] = []
        self.read_count = 0

    @property
    def PrimitiveRenderMode(self) -> str:
        return self._primitive_render_mode

    @PrimitiveRenderMode.setter
    def PrimitiveRenderMode(self, value: str) -> None:
        self._primitive_render_mode = value
        self._events.append(("depth_mode", value))

    @property
    def ShowOnlyActors(self) -> list[object]:
        return self._show_only_actors

    @ShowOnlyActors.setter
    def ShowOnlyActors(self, value: list[object]) -> None:
        self._show_only_actors = list(value)
        actor_ids = tuple(
            actor_id
            for actor_id, actor in self._actors.items()
            if actor in self._show_only_actors
        )
        self._events.append(("show_only", actor_ids))

    def read_pixels(self) -> dict[str, dict[str, np.ndarray]]:
        self.read_count += 1
        actor_ids = tuple(
            actor_id
            for actor_id, actor in self._actors.items()
            if actor in self._show_only_actors
        )
        self._events.append(("depth_read", self._primitive_render_mode, actor_ids))
        if not self._show_only_actors:
            values = np.asarray([[[1.0], [2.0]]], dtype=np.float32)
        elif self._show_only_actors == [self._actors["canine_alpha"]]:
            values = np.asarray([[[1.0], [65504.0]]], dtype=np.float32)
        elif self._show_only_actors == [self._actors["person_beta"]]:
            values = np.asarray([[[65504.0], [2.0]]], dtype=np.float32)
        else:
            raise AssertionError("unexpected target-only actor selection")
        return {"arrays": {"data": values}}


class _FakeColorComponent:
    def __init__(self, *, object_ids: bool) -> None:
        self._object_ids = object_ids
        self.properties: dict[str, float] = {}
        self.read_count = 0

    def set_property_value(self, *, property_name: str, property_value: float) -> None:
        self.properties[property_name] = property_value

    def get_property_value(self, *, property_name: str) -> float:
        return self.properties[property_name]

    def read_pixels(self) -> dict[str, dict[str, np.ndarray]]:
        self.read_count += 1
        if self._object_ids:
            values = np.asarray([[[1, 0, 0, 0], [2, 0, 0, 0]]], dtype=np.uint8)
        else:
            values = np.zeros((TOOL.HEIGHT, TOOL.WIDTH, 4), dtype=np.uint8)
        return {"arrays": {"data": values}}


class _FakeManager:
    def __init__(self, events: list[object], actors: dict[str, object]) -> None:
        self._events = events
        self._actors = actors

    def SetAllowedActors(self, *, AllowedActors: list[object]) -> None:
        actor_id = next(
            actor_id
            for actor_id, actor in self._actors.items()
            if actor is AllowedActors[0]
        )
        self._events.append(("allowed", actor_id))


class _FakeSegmentation:
    def __init__(self, events: list[object], actors: dict[str, object]) -> None:
        self._events = events
        self.proxy_component_manager = _FakeManager(events, actors)

    def initialize(self) -> None:
        self._events.append(("segmentation_initialize",))


class _FakeGame:
    def __init__(self, events: list[object], actors: dict[str, object]) -> None:
        self.segmentation_service = _FakeSegmentation(events, actors)
        self._events = events

        class UnrealService:
            def find_actors_by_class(self, *, uclass: str) -> list[object]:
                assert uclass == "AUsdStageActor"
                return [object()]

        self.unreal_service = UnrealService()

    def get_unreal_object(self, *, uclass: str) -> object:
        assert uclass == "UGameplayStatics"
        events = self._events

        class GameplayStatics:
            def SetGamePaused(self, *, bPaused: bool) -> None:
                events.append(("paused", bPaused))

        return GameplayStatics()


def _native_episode() -> dict[str, object]:
    actor_ids = ["canine_alpha", "person_beta"]
    states = [{"actor_id": actor_id, "action_id": "walk"} for actor_id in actor_ids]
    return {
        "scene": {"scene_id": "kujiale_test", "map_path": "/Game/Test/Kujiale"},
        "review_lights": [],
        "visual_lighting": {"profile_id": "test"},
        "acoustic_proxy": {"scope": "test"},
        "visual_plan": {
            "backend_role": "production_visual",
            "actors": [{"actor_id": actor_id} for actor_id in actor_ids],
            "camera": {
                "horizontal_fov_deg": 90.0,
                "ue_position_cm": [0.0, 0.0, 0.0],
                "ue_yaw_deg": 0.0,
            },
            "frames": [
                {"frame_index": frame_index, "actor_states": states}
                for frame_index in range(75)
            ],
        },
        "timeline": {
            "frames": [
                {"view_pose_hashes": {"view0": f"pose-{frame_index}"}}
                for frame_index in range(75)
            ]
        },
    }


def test_run_native_multimodal_replays_two_dynamic_actor_target_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[object] = []
    actors = {"canine_alpha": object(), "person_beta": object()}
    game = _FakeGame(events, actors)
    instance = _FakeInstance(game, events)
    depth = _FakeDepthComponent(events, actors)
    rgb = _FakeColorComponent(object_ids=False)
    object_ids = _FakeColorComponent(object_ids=True)
    camera = object()
    episode_root = tmp_path / "episode"
    (episode_root / "audio").mkdir(parents=True)
    episode = _native_episode()
    frames = episode["visual_plan"]["frames"]
    for frame_index, frame in enumerate(frames):
        frame["camera_state"] = {
            "frame_index": frame_index,
            "ue_position_cm": [0.0, 0.0, 0.0],
            "ue_yaw_deg": 0.0,
            "pose_hash": f"pose-{frame_index}",
        }
    (episode_root / "episode_plan.json").write_text(
        json.dumps(episode), encoding="utf-8"
    )

    monkeypatch.setattr(TOOL, "_configure_spear", lambda *_: instance)
    monkeypatch.setattr(
        TOOL,
        "_spawn_multimodal_camera",
        lambda *_args, **_kwargs: (
            camera,
            {"rgb": rgb, "depth": depth, "object_ids": object_ids},
        ),
    )
    spawn_arguments: list[tuple[object, ...]] = []

    def spawn_runtime_actors(*args: object) -> dict[str, dict[str, object]]:
        spawn_arguments.append(args)
        events.append(("spawn_runtime_actors",))
        return {
            actor_id: {"visual_actor": actor} for actor_id, actor in actors.items()
        }

    monkeypatch.setattr(TOOL, "_spawn_runtime_actors", spawn_runtime_actors)
    monkeypatch.setattr(
        TOOL,
        "_apply_actor_state",
        lambda _runtime, state, frame_index: (
            _readback(
                float(state["actor_id"] == "person_beta"), frame_index=frame_index
            ),
            {"absolute_error_seconds": 0.0, "action_id": state["action_id"]},
        ),
    )
    camera_arguments: list[object] = []

    def apply_camera(observed_camera: object, _plan: object) -> None:
        camera_arguments.append(observed_camera)

    def actor_readback(observed_camera: object, frame_index: int) -> dict[str, object]:
        camera_arguments.append(observed_camera)
        return _readback(frame_index=frame_index)

    monkeypatch.setattr(TOOL, "_apply_camera", apply_camera)
    monkeypatch.setattr(TOOL, "_actor_readback", actor_readback)
    camera_state_calls: list[tuple[int, str]] = []

    def apply_camera_state_and_readback(
        observed_camera: object, state: dict[str, object], frame_index: int
    ) -> dict[str, object]:
        camera_arguments.append(observed_camera)
        camera_state_calls.append((frame_index, str(state["pose_hash"])))
        result = _readback(frame_index=frame_index)
        result["expected_pose_hash"] = state["pose_hash"]
        return result

    monkeypatch.setattr(
        TOOL, "_apply_camera_state_and_readback", apply_camera_state_and_readback
    )
    monkeypatch.setattr(
        TOOL,
        "_actor_bounds_readback",
        lambda _actor, frame_index: {"frame_index": frame_index},
    )
    monkeypatch.setattr(TOOL, "_spawn_review_lights", lambda *_args: [])
    monkeypatch.setattr(TOOL, "_destroy_runtime_actors", lambda *_args: None)
    monkeypatch.setattr(
        TOOL, "summarize_root_readbacks", lambda **_kwargs: {"status": "pass"}
    )
    monkeypatch.setattr(
        TOOL, "summarize_actor_bounds", lambda **_kwargs: {"status": "pass"}
    )
    monkeypatch.setattr(TOOL, "_mux_clean", lambda *_args: None)
    monkeypatch.setattr(TOOL, "_mux_topdown", lambda *_args: None)
    monkeypatch.setattr(TOOL, "_probe", lambda *_args, **_kwargs: {"status": "pass"})
    monkeypatch.setattr(TOOL, "build_png_encode_command", lambda **_kwargs: ["fake"])
    monkeypatch.setattr(TOOL.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys.modules["cv2"], "imwrite", lambda *_args: True, raising=False
    )

    evidence = TOOL.run(
        argparse.Namespace(
            episode_root=episode_root,
            uproject=tmp_path / "project.uproject",
            unreal_editor=tmp_path / "UnrealEditor",
            output=tmp_path / "output",
            rpc_port=39000,
            graphics_adapter=1,
            streaming_warmup_frames=3,
            expected_stage_actor_count=1,
            keep_frames=False,
            native_multimodal=True,
        )
    )

    assert evidence["research_only"] is True
    assert evidence["formal_dataset_count"] == 0
    assert spawn_arguments == [(game, {"plan": episode["visual_plan"]})]
    assert evidence["native_pixel"]["semantic_ids_by_actor"] == {
        "canine_alpha": 1,
        "person_beta": 2,
    }
    assert rgb.read_count == 75
    assert object_ids.read_count == 75
    assert depth.read_count == 225
    assert camera_arguments and all(item is camera for item in camera_arguments)
    spawn = events.index(("spawn_runtime_actors",))
    warmup = events.index(("step", 3))
    normal_start = events.index(("segmentation_initialize",))
    assert spawn < warmup < normal_start
    assert events[normal_start + 1 : normal_start + 3] == [
        ("depth_mode", "PRM_RenderScenePrimitives"),
        ("show_only", ()),
    ]
    normal_settle = normal_start + 3 + events[normal_start + 3 :].index(("step", 2))
    normal_first_depth = events.index(("depth_read", "PRM_RenderScenePrimitives", ()))
    assert normal_settle < normal_first_depth
    for actor_id in actors:
        allowed = events.index(("allowed", actor_id))
        assert events[allowed + 1 : allowed + 4] == [
            ("segmentation_initialize",),
            ("depth_mode", "PRM_UseShowOnlyList"),
            ("show_only", (actor_id,)),
        ]
        target_settle = allowed + 4 + events[allowed + 4 :].index(("step", 2))
        target_first_depth = events.index(
            ("depth_read", "PRM_UseShowOnlyList", (actor_id,))
        )
        assert target_settle < target_first_depth
    expected_hashes = [f"pose-{frame_index}" for frame_index in range(75)]
    assert camera_state_calls == [
        (frame_index, f"pose-{frame_index}")
        for _pass_index in range(1 + len(actors))
        for frame_index in range(75)
    ]
    readbacks = json.loads(
        (tmp_path / "output" / "native_pixel_runtime_readbacks.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        record["camera"]["expected_pose_hash"] for record in readbacks["normal"]
    ] == expected_hashes
    for actor_id in actors:
        assert [
            record["camera"]["expected_pose_hash"]
            for record in readbacks["target_only"][actor_id]
        ] == expected_hashes


def test_run_visual_only_native_multimodal_needs_no_audio_and_keeps_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[object] = []
    actors = {"canine_alpha": object(), "person_beta": object()}
    game = _FakeGame(events, actors)
    instance = _FakeInstance(game, events)
    depth = _FakeDepthComponent(events, actors)
    rgb = _FakeColorComponent(object_ids=False)
    object_ids = _FakeColorComponent(object_ids=True)
    camera = object()
    episode_root = tmp_path / "episode"
    episode_root.mkdir()
    episode = _native_episode()
    for frame_index, frame in enumerate(episode["visual_plan"]["frames"]):
        frame["camera_state"] = {
            "frame_index": frame_index,
            "ue_position_cm": [0.0, 0.0, 0.0],
            "ue_yaw_deg": 0.0,
            "pose_hash": f"pose-{frame_index}",
        }
    (episode_root / "episode_plan.json").write_text(
        json.dumps(episode), encoding="utf-8"
    )

    monkeypatch.setattr(TOOL, "_configure_spear", lambda *_: instance)
    monkeypatch.setattr(
        TOOL,
        "_spawn_multimodal_camera",
        lambda *_args, **_kwargs: (
            camera,
            {"rgb": rgb, "depth": depth, "object_ids": object_ids},
        ),
    )
    monkeypatch.setattr(
        TOOL,
        "_spawn_runtime_actors",
        lambda *_args: {
            actor_id: {"visual_actor": actor} for actor_id, actor in actors.items()
        },
    )
    monkeypatch.setattr(
        TOOL,
        "_apply_actor_state",
        lambda _runtime, state, frame_index: (
            _readback(
                float(state["actor_id"] == "person_beta"), frame_index=frame_index
            ),
            {"absolute_error_seconds": 0.0, "action_id": state["action_id"]},
        ),
    )
    monkeypatch.setattr(TOOL, "_apply_camera", lambda *_args: None)
    monkeypatch.setattr(
        TOOL,
        "_actor_readback",
        lambda _camera, frame_index: _readback(frame_index=frame_index),
    )

    def camera_state_readback(
        _camera: object, state: dict[str, object], frame_index: int
    ) -> dict[str, object]:
        result = _readback(frame_index=frame_index)
        result["expected_pose_hash"] = state["pose_hash"]
        return result

    monkeypatch.setattr(
        TOOL, "_apply_camera_state_and_readback", camera_state_readback
    )
    monkeypatch.setattr(
        TOOL,
        "_actor_bounds_readback",
        lambda _actor, frame_index: {"frame_index": frame_index},
    )
    light_record = {
        "light_id": "kitchen_ceiling_0000_anchored_fill",
        "source_prim": "/Root/Meshes/kitchen_736/ceiling_light_0000",
        "generated_review_light": True,
    }
    monkeypatch.setattr(
        TOOL, "_spawn_review_lights", lambda *_args: [light_record]
    )
    monkeypatch.setattr(TOOL, "_destroy_runtime_actors", lambda *_args: None)
    monkeypatch.setattr(
        TOOL,
        "summarize_root_readbacks",
        lambda **_kwargs: {
            "status": "pass",
            "schema": "must_not_enter_receipt",
            "checked_pose_hash_count": 75,
        },
    )
    monkeypatch.setattr(
        TOOL, "summarize_actor_bounds", lambda **_kwargs: {"status": "pass"}
    )

    def unexpected_audio(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("visual-only research must not touch audio or mux")

    monkeypatch.setattr(TOOL, "_mux_clean", unexpected_audio)
    monkeypatch.setattr(TOOL, "_mux_topdown", unexpected_audio)
    monkeypatch.setattr(TOOL, "_audio_claim_boundary", unexpected_audio)
    monkeypatch.setattr(TOOL, "_probe", lambda *_args, **_kwargs: {"status": "pass"})
    monkeypatch.setattr(TOOL, "build_png_encode_command", lambda **_kwargs: ["fake"])
    monkeypatch.setattr(TOOL.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys.modules["cv2"], "imwrite", lambda *_args: True, raising=False
    )

    receipt = TOOL.run(
        argparse.Namespace(
            episode_root=episode_root,
            uproject=tmp_path / "project.uproject",
            unreal_editor=tmp_path / "UnrealEditor",
            output=tmp_path / "output",
            rpc_port=39000,
            graphics_adapter=1,
            streaming_warmup_frames=3,
            expected_stage_actor_count=1,
            keep_frames=True,
            native_multimodal=True,
            visual_only_research=True,
        )
    )

    assert receipt["status"] == "research_only"
    assert receipt["research_only"] is True
    assert receipt["episode_counted"] is False
    assert receipt["formal_dataset_count"] == 0
    assert receipt["qualification"] is False
    assert receipt["audio"] == {"status": "not_requested"}
    assert receipt["rlr"] == {"status": "not_requested"}
    assert receipt["clock"] == {
        "frame_count": 75,
        "frame_rate_hz": 15,
        "ticks_per_frame": 3200,
    }
    assert receipt["root_readback"] == {"status": "pass"}
    assert receipt["animation_phase_readback"]
    assert receipt["visual_bounds_readback"] == {"status": "pass"}
    assert receipt["runtime_review_lights"] == [light_record]
    assert receipt["visual_lighting"] == episode["visual_lighting"]
    assert receipt["native_pixel"]["frame_count"] == 75
    truth = json.loads(
        (tmp_path / "output/pixel_visibility_truth.json").read_text(encoding="utf-8")
    )
    assert truth["camera_pose_ids"] == [
        f"current_visual_frame_{index:04d}" for index in range(75)
    ]
    assert rgb.read_count == 75
    assert object_ids.read_count == 75
    assert depth.read_count == 225
    assert (tmp_path / "output/frames").is_dir()
    assert not (episode_root / "audio").exists()
    assert not (tmp_path / "output/evidence.json").exists()
    persisted = json.loads(
        (tmp_path / "output/research_receipt.json").read_text(encoding="utf-8")
    )
    assert persisted == receipt
    encoded = json.dumps(receipt, sort_keys=True).lower()
    assert all(word not in encoded for word in ("schema", "hash", "audio_claim"))


@pytest.mark.parametrize("native_multimodal", [False, True])
def test_parse_args_exposes_opt_in_native_multimodal_flag(
    monkeypatch: pytest.MonkeyPatch, native_multimodal: bool
) -> None:
    argv = [
        "run_spear_residential_episode.py",
        "--episode-root",
        "/tmp/episode",
        "--uproject",
        "/tmp/project.uproject",
        "--unreal-editor",
        "/tmp/UnrealEditor",
        "--output",
        "/tmp/output",
    ]
    if native_multimodal:
        argv.append("--native-multimodal")
    monkeypatch.setattr(sys, "argv", argv)
    parsed = TOOL.parse_args()
    assert parsed.native_multimodal is native_multimodal
    assert not hasattr(parsed, "spear_root")


@pytest.mark.parametrize("visual_only_research", [False, True])
def test_parse_args_exposes_visual_only_research_flag(
    monkeypatch: pytest.MonkeyPatch, visual_only_research: bool
) -> None:
    argv = [
        "run_spear_residential_episode.py",
        "--episode-root",
        "/tmp/episode",
        "--uproject",
        "/tmp/project.uproject",
        "--unreal-editor",
        "/tmp/UnrealEditor",
        "--output",
        "/tmp/output",
    ]
    if visual_only_research:
        argv.append("--visual-only-research")
    monkeypatch.setattr(sys, "argv", argv)

    parsed = TOOL.parse_args()

    assert parsed.visual_only_research is visual_only_research
    assert not hasattr(parsed, "spear_root")
    assert not hasattr(parsed, "runtime_root")


def test_run_legacy_mode_keeps_native_multimodal_path_unreached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[object] = []
    actors = {"canine_alpha": object(), "person_beta": object()}
    game = _FakeGame(events, actors)
    instance = _FakeInstance(game, events)
    capture = _FakeColorComponent(object_ids=False)
    camera = object()
    episode_root = tmp_path / "episode"
    (episode_root / "audio").mkdir(parents=True)
    (episode_root / "episode_plan.json").write_text(
        json.dumps(_native_episode()), encoding="utf-8"
    )
    rendered = {"count": 0}

    monkeypatch.setattr(TOOL, "_configure_spear", lambda *_: instance)
    monkeypatch.setattr(TOOL, "_spawn_camera", lambda *_args: (camera, capture))
    monkeypatch.setattr(
        TOOL,
        "_spawn_multimodal_camera",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy mode must not spawn multimodal camera")
        ),
    )
    monkeypatch.setattr(
        TOOL,
        "_finalize_native_pixel_artifacts",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy mode must not finalize native pixels")
        ),
    )
    monkeypatch.setattr(
        TOOL,
        "_spawn_runtime_actors",
        lambda *_args: {
            actor_id: {"visual_actor": actor} for actor_id, actor in actors.items()
        },
    )
    monkeypatch.setattr(
        TOOL,
        "_apply_actor_state",
        lambda _runtime, state, frame_index: (
            _readback(
                float(state["actor_id"] == "person_beta"), frame_index=frame_index
            ),
            {"absolute_error_seconds": 0.0, "action_id": state["action_id"]},
        ),
    )
    monkeypatch.setattr(TOOL, "_apply_camera", lambda *_args: None)
    monkeypatch.setattr(
        TOOL,
        "_actor_readback",
        lambda _camera, frame_index: _readback(frame_index=frame_index),
    )
    monkeypatch.setattr(
        TOOL,
        "_actor_bounds_readback",
        lambda _actor, frame_index: {"frame_index": frame_index},
    )
    monkeypatch.setattr(TOOL, "_spawn_review_lights", lambda *_args: [])
    monkeypatch.setattr(TOOL, "_destroy_runtime_actors", lambda *_args: None)
    monkeypatch.setattr(
        TOOL, "summarize_root_readbacks", lambda **_kwargs: {"status": "pass"}
    )
    monkeypatch.setattr(
        TOOL, "summarize_actor_bounds", lambda **_kwargs: {"status": "pass"}
    )
    monkeypatch.setattr(
        TOOL,
        "_read_frame",
        lambda *_args: (
            rendered.__setitem__("count", rendered["count"] + 1),
            np.zeros((TOOL.HEIGHT, TOOL.WIDTH, 3), dtype=np.uint8),
        )[1],
    )
    monkeypatch.setattr(TOOL, "_mux_clean", lambda *_args: None)
    monkeypatch.setattr(TOOL, "_mux_topdown", lambda *_args: None)
    monkeypatch.setattr(TOOL, "_probe", lambda *_args, **_kwargs: {"status": "pass"})
    monkeypatch.setattr(TOOL, "build_png_encode_command", lambda **_kwargs: ["fake"])
    monkeypatch.setattr(TOOL.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys.modules["cv2"], "imwrite", lambda *_args: True, raising=False
    )

    evidence = TOOL.run(
        argparse.Namespace(
            episode_root=episode_root,
            uproject=tmp_path / "project.uproject",
            unreal_editor=tmp_path / "UnrealEditor",
            output=tmp_path / "output",
            rpc_port=39000,
            graphics_adapter=1,
            streaming_warmup_frames=3,
            expected_stage_actor_count=1,
            keep_frames=False,
        )
    )

    assert rendered["count"] == 75
    assert "native_pixel" not in evidence
    assert "research_only" not in evidence
    assert not any(
        event[0] == "allowed" for event in events if isinstance(event, tuple)
    )


def test_audio_claim_boundary_prefers_semantic_cached_rlr_evidence(
    tmp_path: Path,
) -> None:
    episode_root = tmp_path / "episode"
    episode_root.mkdir()
    (episode_root / "audio_evidence.json").write_text(
        json.dumps(
            {
                "audio_mode": "semantic_cached_rlr",
                "claim_boundary": "fresh native RLR cache over the selected USD room",
            }
        ),
        encoding="utf-8",
    )

    assert (
        TOOL._audio_claim_boundary(
            episode_root, {"acoustic_proxy": {"label": "legacy proxy"}}
        )
        == "fresh native RLR cache over the selected USD room"
    )


@pytest.mark.parametrize(
    "audio_evidence",
    [None, {"audio_mode": "review_proxy", "claim_boundary": "legacy audio"}],
)
def test_audio_claim_boundary_keeps_legacy_proxy_fallback(
    tmp_path: Path, audio_evidence: dict[str, object] | None
) -> None:
    episode_root = tmp_path / "episode"
    episode_root.mkdir()
    if audio_evidence is not None:
        (episode_root / "audio_evidence.json").write_text(
            json.dumps(audio_evidence), encoding="utf-8"
        )
    fallback = {"label": "legacy proxy", "rir_stride_frames": 3}

    assert (
        TOOL._audio_claim_boundary(episode_root, {"acoustic_proxy": fallback})
        == fallback
    )


@pytest.mark.parametrize("claim_boundary", [None, "", 7])
def test_audio_claim_boundary_rejects_incomplete_semantic_evidence(
    tmp_path: Path, claim_boundary: object
) -> None:
    episode_root = tmp_path / "episode"
    episode_root.mkdir()
    (episode_root / "audio_evidence.json").write_text(
        json.dumps(
            {
                "audio_mode": "semantic_cached_rlr",
                "claim_boundary": claim_boundary,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="lacks a claim_boundary"):
        TOOL._audio_claim_boundary(episode_root, {"acoustic_proxy": {}})


@pytest.mark.parametrize("audio_mode", ["semantic_cached_rlr_v2", 7])
def test_audio_claim_boundary_rejects_unknown_explicit_mode(
    tmp_path: Path, audio_mode: object
) -> None:
    episode_root = tmp_path / "episode"
    episode_root.mkdir()
    (episode_root / "audio_evidence.json").write_text(
        json.dumps(
            {
                "audio_mode": audio_mode,
                "claim_boundary": "must not be silently treated as legacy",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unsupported audio_mode"):
        TOOL._audio_claim_boundary(episode_root, {"acoustic_proxy": {}})
