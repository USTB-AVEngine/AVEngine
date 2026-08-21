from __future__ import annotations

import numpy as np
import pytest

import avengine.backends.spear_ue.research_runtime as research_runtime
from avengine.backends.spear_ue.research_runtime import (
    SpearResearchRuntimeError,
    close_scene_capture,
    read_actor_pose,
    read_rgb_bgr,
    run_frame_transaction,
    spawn_attached_visual_actor,
    warm_scene_capture_until_stable,
)


class _Frame:
    def __init__(self, events: list[str], name: str) -> None:
        self._events = events
        self._name = name

    def __enter__(self) -> "_Frame":
        self._events.append(f"{self._name}:enter")
        return self

    def __exit__(self, *_args: object) -> None:
        self._events.append(f"{self._name}:exit")


class _Instance:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def begin_frame(self) -> _Frame:
        return _Frame(self._events, "begin")

    def end_frame(self) -> _Frame:
        return _Frame(self._events, "end")


class _Capture:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def read_pixels(self) -> dict[str, dict[str, np.ndarray]]:
        return {"arrays": {"data": self.data}}


def test_one_frame_transaction_applies_then_reads_inside_matching_end_frame() -> None:
    events: list[str] = []
    result = run_frame_transaction(
        _Instance(events),
        apply=lambda: events.append("apply"),
        readback=lambda: (events.append("read") or "frame"),
    )
    assert result == "frame"
    assert events == [
        "begin:enter",
        "apply",
        "begin:exit",
        "end:enter",
        "read",
        "end:exit",
    ]


def test_rgb_readback_copies_shared_memory_before_next_frame() -> None:
    source = np.arange(4 * 5 * 4, dtype=np.uint8).reshape(4, 5, 4)
    result = read_rgb_bgr(_Capture(source))
    source[:, :, :3] = 0
    assert result.shape == (4, 5, 3)
    assert int(result.sum()) > 0



def test_scene_capture_cleanup_uses_game_service_for_camera_and_preserves_order() -> None:
    events: list[str] = []
    camera = object()

    class _CaptureCleanup:
        def terminate_sp_funcs(self) -> None:
            events.append("terminate_sp_funcs")

        def Terminate(self) -> None:
            events.append("Terminate")

    class _UnrealService:
        def destroy_actor(self, *, actor: object) -> None:
            assert actor is camera
            events.append("destroy_actor")

    class _Game:
        unreal_service = _UnrealService()

    close_scene_capture(
        instance=_Instance(events),
        game=_Game(),
        camera=camera,
        capture=_CaptureCleanup(),
    )

    assert events == [
        "begin:enter",
        "begin:exit",
        "end:enter",
        "terminate_sp_funcs",
        "Terminate",
        "destroy_actor",
        "end:exit",
    ]


def test_neutral_actor_pose_readback_parses_direct_uppercase_components() -> None:
    class _Actor:
        def K2_GetActorLocation(self, *, as_dict: bool) -> dict[str, object]:
            assert as_dict is True
            return {"X": 1, "Y": 2.5, "Z": -3}

        def K2_GetActorRotation(self, *, as_dict: bool) -> dict[str, object]:
            assert as_dict is True
            return {"Roll": 4, "Pitch": -5.5, "Yaw": 6}

    assert read_actor_pose(_Actor()) == {
        "location_cm": [1.0, 2.5, -3.0],
        "rotation_deg": [4.0, -5.5, 6.0],
    }


def test_pose_readback_parses_nested_lowercase_and_fails_closed() -> None:
    class _Actor:
        def K2_GetActorLocation(self, *, as_dict: bool) -> dict[str, object]:
            assert as_dict is True
            return {"ReturnValue": {"x": 1, "y": 2.5, "z": -3}}

        def K2_GetActorRotation(self, *, as_dict: bool) -> dict[str, object]:
            assert as_dict is True
            return {"Envelope": {"roll": 4, "pitch": -5.5, "yaw": 6}}

    assert read_actor_pose(_Actor()) == {
        "location_cm": [1.0, 2.5, -3.0],
        "rotation_deg": [4.0, -5.5, 6.0],
    }

    malformed_rotation = {"Envelope": {"roll": 4, "pitch": -5.5}}

    class _MalformedActor(_Actor):
        def K2_GetActorRotation(self, *, as_dict: bool) -> dict[str, object]:
            assert as_dict is True
            return malformed_rotation

    with pytest.raises(SpearResearchRuntimeError, match="Yaw") as caught:
        read_actor_pose(_MalformedActor())
    assert repr(malformed_rotation) in str(caught.value)


def test_scene_capture_warmup_discards_exposure_transition_until_stable() -> None:
    frames = [
        np.full((2, 3, 4), value, dtype=np.uint8)
        for value in (0, 20, 40, 40, 40, 40)
    ]

    class _SequenceCapture:
        def read_pixels(self) -> dict[str, dict[str, np.ndarray]]:
            return {"arrays": {"data": frames.pop(0)}}

    result = warm_scene_capture_until_stable(
        _Instance([]),
        _SequenceCapture(),
        maximum_frames=6,
        minimum_frames=3,
        required_consecutive_stable_frames=3,
        mean_absolute_change_threshold=0.0,
    )
    assert result["discarded_frame_count"] == 6
    assert result["final_mean_absolute_change"] == 0.0
    assert result["formal_frame_zero_follows_warmup"] is True


def test_attached_visual_actor_enables_skeletal_tick_and_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class _Root:
        uobject = 17

        def SetMobility(self, **kwargs: object) -> None:
            events.append(("mobility", kwargs))

        def K2_AttachToComponent(self, **kwargs: object) -> bool:
            events.append(("attach", kwargs))
            return True

        def GetAttachParent(self, *, as_handle: bool) -> int:
            assert as_handle is True
            return 17

    anchor_root = _Root()
    visual_root = _Root()

    class _Actor:
        def __init__(self, *, root: _Root | None = None) -> None:
            self.root = root

        def SetActorEnableCollision(self, **kwargs: object) -> None:
            events.append(("collision", kwargs))

        def SetActorTickEnabled(self, **kwargs: object) -> None:
            events.append(("actor_tick", kwargs))

        def K2_SetActorLocationAndRotation(self, **kwargs: object) -> None:
            events.append(("pose", kwargs))

        def K2_GetRootComponent(self) -> _Root:
            assert self.root is not None
            return self.root

    anchor = _Actor()
    visual = _Actor(root=visual_root)

    class _Component:
        def SetComponentTickEnabled(self, **kwargs: object) -> None:
            events.append(("component_tick", kwargs))

        def SetCastShadow(self, **kwargs: object) -> None:
            events.append(("cast_shadow", kwargs))

    component = _Component()

    class _Service:
        def __init__(self) -> None:
            self.spawned = 0

        def spawn_actor(self, **_kwargs: object) -> _Actor:
            self.spawned += 1
            return anchor if self.spawned == 1 else visual

        def create_scene_component_for_actor(self, **_kwargs: object) -> _Root:
            return anchor_root

        def load_class(self, **_kwargs: object) -> object:
            return object()

    game = type("_Game", (), {"unreal_service": _Service()})()
    monkeypatch.setattr(
        research_runtime,
        "select_skeletal_mesh_component",
        lambda **_kwargs: component,
    )
    result = spawn_attached_visual_actor(
        game,
        actor_id="actor0",
        blueprint_class_path="/Game/Actor.Actor_C",
        position_ue_cm=[1.0, 2.0, 3.0],
        yaw_ue_degrees=4.0,
    )
    assert result["component"] is component
    assert ("component_tick", {"bEnabled": True}) in events
    assert ("cast_shadow", {"NewCastShadow": True}) in events
