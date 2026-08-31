from __future__ import annotations

import json
from pathlib import Path

import pytest

import avengine.cli as cli
import avengine.timeline.current_apartment_visual as apartment_visual


REPOSITORY = Path(__file__).resolve().parents[2]
REGISTRY = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"


def test_closure_selection_uses_the_resolved_timeline_map(tmp_path: Path) -> None:
    debug_map = "/Game/SPEAR/Scenes/debug_0000/Maps/debug_0000"
    packages = {
        debug_map: "/tmp/debug.umap",
        "/SpContent/Blueprints/BP_CameraSensor": "/tmp/camera.uasset",
        "/Game/Test/BP_Dog": "/tmp/dog.uasset",
        "/Game/Test/DogMesh": "/tmp/mesh.uasset",
        "/Game/Test/Idle": "/tmp/idle.uasset",
        "/Game/Test/Walk": "/tmp/walk.uasset",
    }
    report = tmp_path / "closure.json"
    report.write_text(json.dumps({
        "variants": {"debug": {
            "mapping_complete": True,
            "physical_mappings": [
                {"package": package,
                 "status": "unique_authorized_external_input",
                 "source_file": source}
                for package, source in packages.items()
            ],
        }},
    }))
    bindings = {
        "source1": {
            "blueprint_class_path": "/Game/Test/BP_Dog.BP_Dog_C",
            "graph_mesh_package": "/Game/Test/DogMesh",
            "idle_animation": "/Game/Test/Idle.Idle",
            "walking_animation": "/Game/Test/Walk.Walk",
        },
    }
    _, mappings = apartment_visual._closure_mappings(
        closure_report_path=report, bindings=bindings, native_map=debug_map)
    assert (debug_map, ".umap") in mappings
    with pytest.raises(apartment_visual.CurrentApartmentVisualError,
                       match="no complete variant"):
        apartment_visual._closure_mappings(
            closure_report_path=report, bindings=bindings,
            native_map=apartment_visual.NATIVE_APARTMENT_MAP)


def _profile_selection(
    tmp_path: Path, *, asset_authorization: str = "unverified"
) -> Path:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    records = {
        (record["asset_id"], record["revision"]): record
        for record in registry["assets"]
    }
    graph_meshes = {
        "legacy_human": (
            "/Game/MyAssets/Audioset/Meshes/"
            "gate_rocketbox_male_adult_01_original_ue_v3/runtime"
        ),
        "legacy_beagle": (
            "/Game/MyAssets/Audioset/Meshes/gate_m2_beagle_v7_world_contact_r5/visual"
        ),
    }
    slots = {"legacy_human": "source1", "legacy_beagle": "source2"}
    actors = []
    for alias in ("legacy_human", "legacy_beagle"):
        reference = registry["aliases"][alias]
        record = records[(reference["asset_id"], reference["revision"])]
        binding = record["runtime_backends"]["spear_unreal"]
        mesh_package = graph_meshes[alias]
        actors.append(
            {
                "source_slot_id": slots[alias],
                "asset_id": record["asset_id"],
                "revision": record["revision"],
                "ue_binding": {
                    "blueprint_object_path": binding["blueprint_class_path"],
                    "profile_skeletal_mesh_binding": binding["skeletal_mesh_binding"],
                    "profile_skeletal_mesh_path": binding["skeletal_mesh_path"],
                    "idle_object_path": binding["idle_animation"],
                    "walking_object_path": binding["walking_animation"],
                    "graph_derived_mesh": {
                        "package": mesh_package,
                        "object_path": mesh_package
                        + "."
                        + mesh_package.rsplit("/", 1)[1],
                    },
                },
            }
        )
    value = {
        "document_type": "test_current_apartment_actor_selection",
        "asset_authorization": asset_authorization,
        "actors": actors,
    }
    path = tmp_path / "actor_selection.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _author_timeline(tmp_path: Path, *, authorization: str = "unverified") -> Path:
    selection = _profile_selection(tmp_path, asset_authorization=authorization)
    path = tmp_path / "timeline.json"
    timeline = apartment_visual.author_current_apartment_visual_timeline(
        actor_selection_path=selection,
        source_asset_registry_path=REGISTRY,
        output_path=path,
        camera_position_ue_cm=(0.0, -600.0, 240.0),
        camera_yaw_deg=0.0,
        human_start_ue_cm=(-150.0, 70.0, 0.0),
        human_end_ue_cm=(150.0, 70.0, 0.0),
        beagle_start_ue_cm=(150.0, -70.0, 0.0),
        beagle_end_ue_cm=(-150.0, -70.0, 0.0),
    )
    assert timeline["asset_authorization"] == authorization
    return path


def test_author_writes_free_75_frame_research_timeline(tmp_path: Path) -> None:
    timeline_path = _author_timeline(tmp_path)
    value = json.loads(timeline_path.read_text(encoding="utf-8"))

    assert value["status"] == "research_only"
    assert value["research_only"] is True
    assert value["episode_counted"] is False
    assert value["qualification_claim"] is False
    assert value["asset_authorization"] == "unverified"
    assert value["render"] == {
        "frame_count": 75,
        "frame_rate_hz": 15,
        "ticks_per_frame": 3200,
        "resolution_hw": [720, 1280],
        "hfov_degrees": 105.0,
        "walk_start_frame": 0,
    }
    assert [frame["frame_index"] for frame in value["frames"]] == list(range(75))
    assert [frame["pts_ticks"] for frame in value["frames"]] == [
        index * 3200 for index in range(75)
    ]
    assert all(
        [state["source_slot_id"] for state in frame["actor_states"]]
        == ["source1", "source2"]
        for frame in value["frames"]
    )
    declarations = {
        actor["source_slot_id"]: actor["walk_phase_period_frames"]
        for actor in value["actors"]
    }
    assert declarations == {"source1": 16, "source2": 25}
    anatomical_forward_yaws = {
        actor["source_slot_id"]: actor["ue_anatomical_forward_yaw_deg"]
        for actor in value["actors"]
    }
    assert anatomical_forward_yaws == {"source1": 90.0, "source2": 180.0}
    starts = {
        "source1": [-150.0, 70.0, 0.0],
        "source2": [150.0, -70.0, 0.0],
    }
    ends = {
        "source1": [150.0, 70.0, 0.0],
        "source2": [-150.0, -70.0, 0.0],
    }
    states_at_walk_start = {
        state["source_slot_id"]: state for state in value["frames"][0]["actor_states"]
    }
    for slot, start in starts.items():
        assert states_at_walk_start[slot]["translation_ue_cm"] == start
        assert states_at_walk_start[slot]["action_id"] == "walk"
        assert states_at_walk_start[slot]["action_phase"] == 0.0
    states_at_walk_end = {
        state["source_slot_id"]: state for state in value["frames"][74]["actor_states"]
    }
    for slot, end in ends.items():
        assert states_at_walk_end[slot]["translation_ue_cm"] == end
        assert states_at_walk_end[slot]["action_id"] == "walk"
    assert all(
        value["frames"][frame_index]["actor_states"][0]["translation_ue_cm"]
        == pytest.approx(
            [
                starts["source1"][axis]
                + (ends["source1"][axis] - starts["source1"][axis])
                * frame_index
                / 74
                for axis in range(3)
            ]
        )
        for frame_index in (1, 31, 74)
    )
    frame_31 = {
        state["source_slot_id"]: state for state in value["frames"][31]["actor_states"]
    }
    assert frame_31["source1"]["action_phase"] == pytest.approx(15.0 / 16.0)
    assert frame_31["source2"]["action_phase"] == pytest.approx(6.0 / 25.0)
    frame_40 = {
        state["source_slot_id"]: state for state in value["frames"][40]["actor_states"]
    }
    assert frame_40["source1"]["action_phase"] == pytest.approx(8.0 / 16.0)
    assert frame_40["source2"]["action_phase"] == pytest.approx(15.0 / 25.0)
    assert all(
        state["walk_phase_period_frames"] == declarations[state["source_slot_id"]]
        for frame in value["frames"]
        for state in frame["actor_states"]
    )
    assert all(
        frame["actor_states"][0]["yaw_ue_deg"] == pytest.approx(-90.0)
        and frame["actor_states"][1]["yaw_ue_deg"] == pytest.approx(0.0)
        for frame in value["frames"]
    )
    encoded = json.dumps(value, sort_keys=True)
    assert "schema" not in encoded
    assert "sha256" not in encoded
    assert "audio" not in value
    assert "rlr" not in value
    assert "m6_m7_bundle" not in value


def test_timeline_state_keeps_static_actor_idle_at_start() -> None:
    binding = {
        "actor_id": "source1_actor",
        "source_slot_id": "source1",
        "asset_id": "asset",
        "revision": "v1",
        "walk_phase_period_frames": 16,
        "ue_anatomical_forward_yaw_deg": 90.0,
    }
    start = (25.0, -40.0, 0.0)
    for frame_index in (0, 14, 15, 74):
        state = apartment_visual._timeline_state(
            binding=binding,
            start=start,
            end=start,
            frame_index=frame_index,
        )
        assert state["translation_ue_cm"] == list(start)
        assert state["action_id"] == "idle"
        assert state["action_phase"] == 0.0


def test_timeline_state_starts_moving_actor_at_formal_frame_zero() -> None:
    binding = {
        "actor_id": "source1_actor",
        "source_slot_id": "source1",
        "asset_id": "asset",
        "revision": "v1",
        "walk_phase_period_frames": 16,
        "ue_anatomical_forward_yaw_deg": 90.0,
    }
    state0 = apartment_visual._timeline_state(
        binding=binding,
        start=(0.0, 0.0, 0.0),
        end=(740.0, 0.0, 0.0),
        frame_index=0,
    )
    state1 = apartment_visual._timeline_state(
        binding=binding,
        start=(0.0, 0.0, 0.0),
        end=(740.0, 0.0, 0.0),
        frame_index=1,
    )
    assert state0["action_id"] == "walk"
    assert state0["action_phase"] == 0.0
    assert state0["translation_ue_cm"] == [0.0, 0.0, 0.0]
    assert state1["action_phase"] == pytest.approx(1.0 / 16.0)
    assert state1["translation_ue_cm"] == pytest.approx([10.0, 0.0, 0.0])


def test_spawned_apartment_actors_bind_initial_animation_before_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []

    class _Animation:
        def __init__(self, name: str) -> None:
            self.name = name

        def GetPlayLength(self) -> float:
            return 2.0

    class _Component:
        def __init__(self, slot: str) -> None:
            self.slot = slot
            self.position = 0.0

        def GetSkeletalMeshAsset(self, *, as_handle: bool) -> int:
            assert as_handle is True
            return 41

        def PlayAnimation(self, **kwargs: object) -> None:
            events.append((self.slot + ".play", kwargs["NewAnimToPlay"]))

        def Stop(self) -> None:
            events.append((self.slot + ".stop", None))

        def SetPosition(self, **kwargs: object) -> None:
            self.position = float(kwargs["InPos"])
            events.append((self.slot + ".position", self.position))

        def GetPosition(self) -> float:
            return self.position

    class _Anchor:
        def K2_SetActorLocationAndRotation(self, **kwargs: object) -> None:
            events.append(("anchor.pose", kwargs))

    components = {slot: _Component(slot) for slot in ("source1", "source2")}

    def spawn(_game: object, **kwargs: object) -> dict[str, object]:
        slot = str(kwargs["actor_id"]).removesuffix("_actor")
        return {
            "anchor": _Anchor(),
            "visual_root": object(),
            "component": components[slot],
        }

    monkeypatch.setattr(apartment_visual, "spawn_attached_visual_actor", spawn)
    monkeypatch.setattr(
        apartment_visual, "apply_ue_component_frame_delta", lambda *_args: None
    )

    class _Service:
        def load_object(self, *, uclass: str, name: str, as_handle: bool = False):
            if uclass == "USkeletalMesh":
                assert as_handle is True
                return 41
            assert uclass == "UAnimationAsset"
            return _Animation(name)

    game = type("_Game", (), {"unreal_service": _Service()})()
    bindings = {
        slot: {
            "actor_id": slot + "_actor",
            "blueprint_class_path": "/Game/" + slot,
            "component_frame_delta": {},
            "graph_mesh_object_path": "/Game/mesh.mesh",
            "idle_animation": "/Game/Idle.Idle",
            "walking_animation": "/Game/Walking.Walking",
        }
        for slot in ("source1", "source2")
    }
    initial_frame = {
        "actor_states": [
            {
                "source_slot_id": slot,
                "action_id": "walk",
                "action_phase": 0.25,
                "translation_ue_cm": [1.0, 2.0, 3.0],
                "yaw_ue_deg": 4.0,
            }
            for slot in ("source1", "source2")
        ]
    }

    runtimes = apartment_visual._spawn_runtime_actors(
        game=game, bindings=bindings, initial_frame=initial_frame
    )

    for slot in ("source1", "source2"):
        assert runtimes[slot]["current_action"] == "walk"
        assert (slot + ".stop", None) in events
        assert (slot + ".position", 0.5) in events
        assert any(name == slot + ".play" for name, _value in events)


def test_unverified_myassets_returns_not_run_without_stage_or_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _profile_selection(tmp_path, asset_authorization="unverified")
    timeline = _author_timeline(tmp_path, authorization="unverified")
    monkeypatch.setattr(
        apartment_visual,
        "launch_external_game_instance",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("unverified MyAssets launched SPEAR")
        ),
    )

    result = apartment_visual.capture_current_apartment_visual(
        actor_selection_path=selection,
        source_asset_registry_path=REGISTRY,
        timeline_path=timeline,
        closure_report_path=tmp_path / "must_not_be_read.json",
        stage_root=tmp_path / "must_not_be_read_stage",
        spear_executable=tmp_path / "must_not_be_read_stage/SpearSim.sh",
        output_directory=tmp_path / "not_run",
    )

    assert result["status"] == "not_run"
    assert result["research_only"] is True
    assert result["episode_counted"] is False
    assert result["asset_authorization"] == "unverified"
    assert "no SPEAR launch" in result["reason"]
    assert (tmp_path / "not_run/research_receipt.json").is_file()


def test_timeline_rejects_selection_drift(tmp_path: Path) -> None:
    selection = _profile_selection(tmp_path)
    timeline = _author_timeline(tmp_path)
    value = json.loads(timeline.read_text(encoding="utf-8"))
    value["actors"][0]["revision"] = "wrong"
    timeline.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        apartment_visual.CurrentApartmentVisualError,
        match="differs from actor selection",
    ):
        apartment_visual.capture_current_apartment_visual(
            actor_selection_path=selection,
            source_asset_registry_path=REGISTRY,
            timeline_path=timeline,
            closure_report_path=tmp_path / "unused.json",
            stage_root=tmp_path / "unused",
            spear_executable=tmp_path / "unused/SpearSim.sh",
            output_directory=tmp_path / "output",
        )


def test_cli_exposes_separate_research_commands_without_audio_inputs() -> None:
    parser = cli.build_parser()
    author = parser.parse_args(
        [
            "m5",
            "author-current-apartment-visual-timeline",
            "--actor-selection",
            "selection.json",
            "--source-asset-registry",
            "registry.json",
            "--camera-position-ue-cm",
            "0",
            "0",
            "100",
            "--camera-yaw-deg",
            "0",
            "--human-start-ue-cm",
            "0",
            "0",
            "0",
            "--human-end-ue-cm",
            "100",
            "0",
            "0",
            "--beagle-start-ue-cm",
            "0",
            "100",
            "0",
            "--beagle-end-ue-cm",
            "100",
            "100",
            "0",
            "--output",
            "/tmp/current-apartment-timeline.json",
        ]
    )
    capture = parser.parse_args(
        [
            "m5",
            "capture-current-apartment-visual",
            "--actor-selection",
            "selection.json",
            "--source-asset-registry",
            "registry.json",
            "--timeline",
            "timeline.json",
            "--closure-report",
            "closure.json",
            "--stage-root",
            "/external/stage",
            "--spear-executable",
            "/external/stage/SpearSim.sh",
            "--output",
            "/tmp/current-apartment-capture",
        ]
    )
    assert author.m5_command == "author-current-apartment-visual-timeline"
    assert capture.m5_command == "capture-current-apartment-visual"
    assert not hasattr(author, "hrtf")
    assert not hasattr(capture, "beagle_dry")
    assert not hasattr(capture, "m4_request")


def test_capture_failure_writes_honest_partial_receipt_and_always_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = {
        "source1": {"actor_id": "source1_actor"},
        "source2": {"actor_id": "source2_actor"},
    }
    frames = []
    for frame_index in range(apartment_visual.FRAME_COUNT):
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * apartment_visual.TICKS_PER_FRAME,
                "camera": {
                    "translation_ue_cm": [0.0, 0.0, 0.0],
                    "yaw_ue_deg": 0.0,
                },
                "actor_states": [
                    {
                        "source_slot_id": "source1",
                        "actor_id": "source1_actor",
                        "action_id": "idle",
                        "action_phase": 0.0,
                        "walk_phase_period_frames": 16,
                        "translation_ue_cm": [0.0, 0.0, 0.0],
                        "yaw_ue_deg": 0.0,
                    },
                    {
                        "source_slot_id": "source2",
                        "actor_id": "source2_actor",
                        "action_id": "idle",
                        "action_phase": 0.0,
                        "walk_phase_period_frames": 25,
                        "translation_ue_cm": [0.0, 0.0, 0.0],
                        "yaw_ue_deg": 0.0,
                    },
                ],
            }
        )
    timeline = {
        "render": {
            "resolution_hw": [2, 3],
            "hfov_degrees": 105.0,
        },
        "frames": frames,
    }
    monkeypatch.setattr(
        apartment_visual,
        "_selection_bindings",
        lambda **_kwargs: (tmp_path / "selection.json", bindings, "verified_internal"),
    )
    monkeypatch.setattr(
        apartment_visual,
        "_load_timeline",
        lambda **_kwargs: (tmp_path / "timeline.json", timeline),
    )
    monkeypatch.setattr(
        apartment_visual,
        "_closure_mappings",
        lambda **_kwargs: (tmp_path / "closure.json", []),
    )
    monkeypatch.setattr(
        apartment_visual,
        "_validate_stage",
        lambda **_kwargs: (tmp_path, tmp_path / "SpearSim.sh"),
    )

    class _Frame:
        def __init__(self, events: list[str], name: str) -> None:
            self.events = events
            self.name = name

        def __enter__(self) -> "_Frame":
            self.events.append(self.name + ":enter")
            return self

        def __exit__(self, *_args: object) -> None:
            self.events.append(self.name + ":exit")

    events: list[str] = []

    class _GameplayStatics:
        def SetGamePaused(self, *, bPaused: bool) -> None:
            events.append(f"game.paused:{bPaused}")

    class _Game:
        def get_unreal_object(self, *, uclass: str) -> _GameplayStatics:
            assert uclass == "UGameplayStatics"
            return _GameplayStatics()

    game = _Game()

    class _Instance:
        def begin_frame(self) -> _Frame:
            return _Frame(events, "begin")

        def end_frame(self) -> _Frame:
            return _Frame(events, "end")

        def get_game(self) -> object:
            return game

        def close(self, *, force: bool) -> None:
            events.append(f"instance.close:{force}")

    monkeypatch.setattr(
        apartment_visual,
        "launch_external_game_instance",
        lambda **_kwargs: _Instance(),
    )

    class _Camera:
        def K2_SetActorLocationAndRotation(self, **_kwargs: object) -> None:
            events.append("camera.pose")

    monkeypatch.setattr(
        apartment_visual,
        "spawn_scene_capture",
        lambda *_args, **_kwargs: (_Camera(), object()),
    )
    monkeypatch.setattr(
        apartment_visual,
        "warm_scene_capture_until_stable",
        lambda *_args, **_kwargs: {"status": "pass"},
    )
    monkeypatch.setattr(
        apartment_visual,
        "_spawn_runtime_actors",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        apartment_visual,
        "run_frame_transaction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("frame boom")),
    )
    monkeypatch.setattr(
        apartment_visual,
        "_destroy_runtime_actors",
        lambda *_args, **_kwargs: (
            events.append("actor.cleanup")
            or (_ for _ in ()).throw(RuntimeError("actor cleanup boom"))
        ),
    )
    monkeypatch.setattr(
        apartment_visual,
        "close_scene_capture",
        lambda **_kwargs: (
            events.append("camera.cleanup")
            or (_ for _ in ()).throw(RuntimeError("camera cleanup boom"))
        ),
    )

    output = tmp_path / "failed-capture"
    with pytest.raises(RuntimeError, match="frame boom"):
        apartment_visual.capture_current_apartment_visual(
            actor_selection_path=tmp_path / "selection.json",
            source_asset_registry_path=REGISTRY,
            timeline_path=tmp_path / "timeline.json",
            closure_report_path=tmp_path / "closure.json",
            stage_root=tmp_path,
            spear_executable=tmp_path / "SpearSim.sh",
            output_directory=output,
        )

    receipt = json.loads((output / "research_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "fail"
    assert receipt["partial"] is True
    assert receipt["research_only"] is True
    assert receipt["episode_counted"] is False
    assert receipt["qualification_claim"] is False
    assert receipt["capture"]["completed_frame_count"] == 0
    assert receipt["error_type"] == "RuntimeError"
    assert receipt["error_text"] == "frame boom"
    encoded = json.dumps(receipt, sort_keys=True).lower()
    assert all(word not in encoded for word in ("schema", "evidence", "hash", "gate"))
    assert "actor.cleanup" not in events
    assert "camera.cleanup" not in events
    assert "game.paused:False" in events
    assert events[-1] == "instance.close:True"
