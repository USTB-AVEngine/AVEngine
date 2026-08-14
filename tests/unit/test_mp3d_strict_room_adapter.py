from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

TEST_FILE = Path(__file__).resolve()
ROOT = (
    TEST_FILE.parents[1]
    if (TEST_FILE.parents[1] / "reference").is_dir()
    else TEST_FILE.parents[2]
)
TOOLS = ROOT / "tools/qa"
sys.path.insert(0, str(TOOLS))

import build_strict_two_human_mp3d_room_preflight as builder  # noqa: E402
import capture_spear_imported_glb_strict_two_human_episode as capture  # noqa: E402
import spear_imported_glb_room_adapter as adapter_module  # noqa: E402
from avengine.sensor_rig_trajectory import (  # noqa: E402
    materialize_sensor_rig_trajectory,
)


class FakeObject:
    def __init__(self, uobject: int) -> None:
        self.uobject = uobject


class FakeActor(FakeObject):
    def __init__(self, uobject: int, component: FakeComponent) -> None:
        super().__init__(uobject)
        self.component = component


class FakeComponent(FakeObject):
    def __init__(self, uobject: int, *, mismatch: bool = False) -> None:
        super().__init__(uobject)
        self.mesh = None
        self.mismatch = mismatch
        self.mobility = None
        self.cast_shadow = None
        self.collision = None

    def SetMobility(self, *, NewMobility: str) -> None:
        self.mobility = NewMobility

    def SetStaticMesh(self, *, NewMesh: FakeObject) -> None:
        self.mesh = NewMesh

    def SetCastShadow(self, *, NewCastShadow: bool) -> None:
        self.cast_shadow = NewCastShadow

    def SetCollisionEnabled(self, *, NewType: str) -> None:
        self.collision = NewType

    def GetStaticMesh(self, *, as_handle: bool) -> int:
        if not as_handle or self.mesh is None:
            raise AssertionError("unexpected mesh readback")
        return self.mesh.uobject + int(self.mismatch)


class FakeNonCallableGetterComponent(FakeComponent):
    @property
    def GetStaticMesh(self) -> FakeObject:
        return FakeObject(9001)

    def get_property_value(self, *, property_name: str, as_handle: bool) -> int:
        if property_name != "StaticMesh" or not as_handle or self.mesh is None:
            raise AssertionError("unexpected property readback")
        return self.mesh.uobject


class FakePropertyOnlyComponent(FakeComponent):
    GetStaticMesh = None

    def get_property_value(self, *, property_name: str, as_handle: bool) -> int:
        if property_name != "StaticMesh" or not as_handle or self.mesh is None:
            raise AssertionError("unexpected property readback")
        return self.mesh.uobject


class FakeMissingReadbackComponent(FakeComponent):
    GetStaticMesh = None


class FakeCaptureComponent(FakeObject):
    def __init__(self, uobject: int, *, drift_handle: bool = False) -> None:
        super().__init__(uobject)
        self.fov_angle = 0.0
        self.drift_handle = drift_handle

    def set_property_value(self, *, property_name: str, property_value: float) -> None:
        if property_name != "FOVAngle":
            raise AssertionError("unexpected scene-capture property write")
        self.fov_angle = float(property_value)
        if self.drift_handle:
            self.uobject += 1

    def get_property_value(self, *, property_name: str) -> float:
        if property_name != "FOVAngle":
            raise AssertionError("unexpected scene-capture property read")
        return self.fov_angle


class FakeUnrealService:
    def __init__(
        self, *, mismatch_at: int | None = None, duplicate_loads: bool = False
    ) -> None:
        self.mismatch_at = mismatch_at
        self.duplicate_loads = duplicate_loads
        self.paths: dict[str, int] = {}
        self.objects: dict[int, FakeObject] = {}
        self.actors: list[FakeActor] = []
        self.stable_names: list[str] = []

    def load_object(self, *, uclass: str, name: str, as_handle: bool) -> int:
        if uclass != "UStaticMesh" or not as_handle:
            raise AssertionError("unexpected load")
        handle = self.paths.setdefault(
            name, 1000 if self.duplicate_loads else 1000 + len(self.paths)
        )
        self.objects.setdefault(handle, FakeObject(handle))
        return handle

    def spawn_actor(self, **_: object) -> FakeActor:
        index = len(self.actors)
        component = FakeComponent(3000 + index, mismatch=index == self.mismatch_at)
        actor = FakeActor(2000 + index, component)
        self.actors.append(actor)
        return actor

    def get_component_by_class(self, *, actor: FakeActor, uclass: str) -> FakeComponent:
        if uclass != "UStaticMeshComponent":
            raise AssertionError("unexpected component class")
        return actor.component

    def set_stable_name_for_actor(self, *, actor: FakeActor, stable_name: str) -> None:
        del actor
        self.stable_names.append(stable_name)


class FakeGame:
    def __init__(
        self, *, mismatch_at: int | None = None, duplicate_loads: bool = False
    ) -> None:
        self.unreal_service = FakeUnrealService(
            mismatch_at=mismatch_at, duplicate_loads=duplicate_loads
        )

    def get_unreal_object(self, *, uobject: int) -> FakeObject:
        return self.unreal_service.objects[uobject]


def _declared_inputs() -> tuple[Path, dict[str, Path]]:
    request_path = ROOT / "examples/qa/native_strict_two_human_mp3d_room_atom_v1.json"
    request = json.loads(request_path.read_text())
    staging_reference = ROOT / "reference"
    if staging_reference.is_dir():
        inputs = {
            "template_suite": staging_reference
            / "strict_two_human_template_suite.json",
            "ue_import_manifest": staging_reference / "mp3d_ue_import_result.json",
            "ue_runtime_evidence": staging_reference / "mp3d_ue_runtime_evidence.json",
            "fresh_navmesh_probe": staging_reference / "fresh_navmesh_probe.json",
            "acoustic_manifest": staging_reference / "mp3d_soundspaces2_manifest.json",
            "room_registry": staging_reference / "room_registry.json",
            "acoustic_profiles": staging_reference / "acoustic_profiles.json",
        }
    else:
        inputs = {
            "template_suite": Path(request["template_suite"]),
            "ue_import_manifest": Path(request["room"]["ue_import_manifest"]),
            "ue_runtime_evidence": Path(request["room"]["ue_runtime_evidence"]),
            "fresh_navmesh_probe": Path(request["room"]["fresh_navmesh_probe"]),
            "acoustic_manifest": Path(request["acoustics"]["package_manifest"]),
            "room_registry": Path(request["acoustics"]["room_registry"]),
            "acoustic_profiles": Path(
                request["acoustics"]["acoustic_profile_registry"]
            ),
        }
    return request_path, inputs


def _build_output(output: Path) -> None:
    request_path, inputs = _declared_inputs()
    builder.build(
        argparse.Namespace(
            request=request_path,
            **inputs,
            output=output,
        )
    )


def _v2_args(output: Path) -> argparse.Namespace:
    request = ROOT / "examples/qa/native_strict_two_human_mp3d_room_atom_v2.json"
    value = json.loads(request.read_text())
    return argparse.Namespace(
        request=request,
        template_suite=Path(value["template_suite"]),
        ue_import_manifest=Path(value["room"]["ue_import_manifest"]),
        ue_runtime_evidence=Path(value["room"]["ue_runtime_evidence"]),
        fresh_navmesh_probe=Path(value["room"]["fresh_navmesh_probe"]),
        acoustic_manifest=Path(value["acoustics"]["package_manifest"]),
        room_registry=Path(value["acoustics"]["room_registry"]),
        acoustic_profiles=Path(value["acoustics"]["acoustic_profile_registry"]),
        output=output,
    )


class RoomAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, inputs = _declared_inputs()
        cls.import_manifest = json.loads(inputs["ue_import_manifest"].read_text())
        cls.adapter = adapter_module.build_room_adapter_record(
            cls.import_manifest,
            execution_manifest_path="/execution/ue_import_result.json",
        )

    def test_manifest_binds_exact_71_unique_objects_and_shared_camera(self) -> None:
        adapter_module.validate_room_adapter(self.adapter)
        paths = self.adapter["static_mesh_object_paths"]
        self.assertEqual(len(paths), 71)
        self.assertEqual(len(set(paths)), 71)
        camera = self.adapter["camera_contract"]
        self.assertTrue(camera["one_camera_actor_for_all_passes"])
        components = camera["components"]
        self.assertEqual(
            {
                components["normal_metric_depth"],
                components["source1_target_only_metric_depth"],
                components["source2_target_only_metric_depth"],
            },
            {adapter_module.DEPTH_COMPONENT},
        )

    def test_mp3d_suite_projects_a_production_template_to_comparison(self) -> None:
        request_path, inputs = _declared_inputs()
        request = json.loads(request_path.read_text())
        template = json.loads(inputs["template_suite"].read_text())
        template["backend_role"] = "production_visual"
        template_scenario = template["scenarios"][0]
        template_scenario["backend_role"] = "production_visual"
        template_scenario["plan"]["backend_role"] = "production_visual"

        suite, _ = builder._build_suite(request, template, self.adapter)

        scenario = suite["scenarios"][0]
        self.assertEqual(suite["backend_role"], "comparison_visual")
        self.assertEqual(scenario["backend_role"], "comparison_visual")
        self.assertEqual(
            scenario["plan"]["backend_role"],
            "comparison_visual",
        )
        self.assertFalse(
            scenario["plan"]["authority"]["backend_may_replan"]
        )

    def test_hfov_uses_exact_named_scene_capture_components(self) -> None:
        camera = FakeObject(5000)
        components = {
            "rgb": FakeCaptureComponent(5001),
            "depth": FakeCaptureComponent(5002),
            "object_ids": FakeCaptureComponent(5003),
        }
        evidence = capture._set_camera_hfov(camera, components, 90.0)
        self.assertEqual(evidence["status"], "pass")
        self.assertEqual(evidence["camera_actor_handle"], 5000)
        self.assertEqual(
            evidence["component_handles"],
            {"rgb": 5001, "depth": 5002, "object_ids": 5003},
        )
        self.assertEqual(
            evidence["observed_horizontal_fov_deg_by_component"],
            {"rgb": 90.0, "depth": 90.0, "object_ids": 90.0},
        )
        self.assertEqual(
            evidence["write_method"],
            "named_USpSceneCaptureComponent2D.FOVAngle_property",
        )

    def test_hfov_rejects_missing_named_component(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing named camera components"):
            capture._set_camera_hfov(
                FakeObject(5000),
                {
                    "rgb": FakeCaptureComponent(5001),
                    "depth": FakeCaptureComponent(5002),
                },
                90.0,
            )

    def test_hfov_rejects_component_alias_and_handle_drift(self) -> None:
        shared = FakeCaptureComponent(5001)
        with self.assertRaisesRegex(RuntimeError, "distinct live handles"):
            capture._set_camera_hfov(
                FakeObject(5000),
                {"rgb": shared, "depth": shared, "object_ids": shared},
                90.0,
            )
        with self.assertRaisesRegex(RuntimeError, "handle drift"):
            capture._set_camera_hfov(
                FakeObject(5000),
                {
                    "rgb": FakeCaptureComponent(5001),
                    "depth": FakeCaptureComponent(5002, drift_handle=True),
                    "object_ids": FakeCaptureComponent(5003),
                },
                90.0,
            )

    def test_fake_runtime_fresh_load_spawn_readback_closes_71(self) -> None:
        game = FakeGame()
        actors, evidence = adapter_module.spawn_scene_meshes_with_readback(
            game, self.adapter
        )
        self.assertEqual(len(actors), 71)
        self.assertEqual(evidence["spawned_static_mesh_count"], 71)
        self.assertTrue(evidence["all_expected_handles_match_components"])
        self.assertEqual(evidence["unique_loaded_object_handle_count"], 71)
        self.assertEqual(evidence["unique_component_mesh_handle_count"], 71)
        self.assertEqual(len(set(game.unreal_service.stable_names)), 71)

    def test_static_mesh_readback_prefers_callable_component_getter(self) -> None:
        component = FakeComponent(3000)
        component.SetStaticMesh(NewMesh=FakeObject(1234))
        self.assertEqual(
            adapter_module._static_mesh_handle(component),
            (1234, "UStaticMeshComponent.GetStaticMesh"),
        )

    def test_static_mesh_readback_uses_property_for_noncallable_unreal_object(
        self,
    ) -> None:
        component = FakeNonCallableGetterComponent(3000)
        component.SetStaticMesh(NewMesh=FakeObject(2345))
        self.assertEqual(
            adapter_module._static_mesh_handle(component),
            (2345, "UStaticMeshComponent.StaticMesh_property"),
        )

    def test_static_mesh_readback_uses_property_when_getter_is_absent(self) -> None:
        component = FakePropertyOnlyComponent(3000)
        component.SetStaticMesh(NewMesh=FakeObject(3456))
        self.assertEqual(
            adapter_module._static_mesh_handle(component),
            (3456, "UStaticMeshComponent.StaticMesh_property"),
        )

    def test_static_mesh_readback_rejects_missing_getter_and_property(self) -> None:
        component = FakeMissingReadbackComponent(3000)
        component.SetStaticMesh(NewMesh=FakeObject(4567))
        with self.assertRaisesRegex(
            RuntimeError, "neither callable GetStaticMesh nor a readable"
        ):
            adapter_module._static_mesh_handle(component)

    def test_fake_runtime_rejects_component_mesh_mismatch(self) -> None:
        game = FakeGame(mismatch_at=9)
        with self.assertRaisesRegex(RuntimeError, "readback differs at index 9"):
            adapter_module.spawn_scene_meshes_with_readback(game, self.adapter)

    def test_fake_runtime_rejects_nonunique_loaded_objects(self) -> None:
        game = FakeGame(duplicate_loads=True)
        with self.assertRaisesRegex(RuntimeError, "71 unique live objects"):
            adapter_module.spawn_scene_meshes_with_readback(game, self.adapter)


class ActorFramingConsumerTests(unittest.TestCase):
    def _request(self) -> dict[str, object]:
        return {
            "episode_id": "episode0",
            "actor_framing": {
                "actor_bindings": [{"actor_id": "source1_actor"}],
                "sample_rate_hz": 120.0,
                "padding_m": 0.02,
            },
            "camera_framing": {
                "candidates": [
                    {
                        "candidate_id": "declared0",
                        "position_m": [0.0, 1.5, 0.0],
                        "yaw_deg": 0.0,
                        "room_gate": {
                            "status": "pass",
                            "authority_id": "request/declared0/room-gate",
                            "provenance": "declared_cpu_planning",
                            "native_habitat_validation_status": "pending",
                            "line_of_sight_validation_status": "pending",
                            "full_body_clearance_status": "pending",
                            "hard_gates": {"declared_position": {"status": "pass"}},
                        },
                    }
                ],
                "calibration": {"resolution_hw": [720, 1280]},
                "ordered_actor_ids": ["source1_actor", "source2_actor"],
                "minimum_order_gap_px": 8.0,
            },
        }

    def _suite(self) -> dict[str, object]:
        return {
            "scenarios": [
                {
                    "plan": {
                        "frames": [
                            {
                                "frame_index": index,
                                "actor_states": [{"actor_id": "source1_actor"}],
                            }
                            for index in range(75)
                        ]
                    }
                }
            ]
        }

    @patch.object(builder, "solve_static_camera_candidates")
    @patch.object(builder, "build_actor_framing_frames")
    def test_helper_binds_full75_inputs_and_selected_hold_rig(
        self, build_actor: object, solve_camera: object
    ) -> None:
        build_actor.return_value = {  # type: ignore[attr-defined]
            "frames": [
                {
                    "frame_index": index,
                    "actor_aabbs": {
                        "source1_actor": {
                            "minimum_m": [-0.2, 0.0, -2.2],
                            "maximum_m": [0.2, 1.8, -1.8],
                        },
                        "source2_actor": {
                            "minimum_m": [0.8, 0.0, -2.2],
                            "maximum_m": [1.2, 1.8, -1.8],
                        },
                    },
                }
                for index in range(75)
            ]
        }
        solve_camera.return_value = {  # type: ignore[attr-defined]
            "selected_candidate_id": "declared0",
            "sensor_rig_binding": {
                "source": "materialized_hold",
                "trajectory": {
                    "frames": [{"frame_index": index} for index in range(75)]
                },
            },
        }

        actor_inputs, solution = builder._solve_full75_actor_framing(
            self._request(), self._suite()
        )

        self.assertEqual(len(actor_inputs["frames"]), 75)
        self.assertEqual(solution["selected_candidate_id"], "declared0")
        self.assertEqual(
            len(solution["sensor_rig_binding"]["trajectory"]["frames"]), 75
        )
        self.assertEqual(build_actor.call_args.kwargs["expected_frame_count"], 75)
        self.assertEqual(
            solve_camera.call_args.kwargs["trajectory_id"], "episode0__sensor_rig"
        )
        room_gate = solve_camera.call_args.kwargs["candidates"][0]["room_gate"]
        self.assertEqual(room_gate["provenance"], "declared_cpu_planning")
        self.assertEqual(room_gate["line_of_sight_validation_status"], "pending")

    def test_helper_fails_closed_without_explicit_pending_room_gate(self) -> None:
        request = self._request()
        del request["camera_framing"]["candidates"][0]["room_gate"]  # type: ignore[index]
        with self.assertRaisesRegex(RuntimeError, "camera room_gate"):
            builder._solve_full75_actor_framing(request, self._suite())

    @patch.object(builder, "solve_static_camera_candidates")
    @patch.object(builder, "build_actor_framing_frames")
    def test_helper_fails_closed_when_solver_selects_nothing(
        self, build_actor: object, solve_camera: object
    ) -> None:
        build_actor.return_value = {"frames": []}  # type: ignore[attr-defined]
        solve_camera.return_value = {  # type: ignore[attr-defined]
            "selected_candidate_id": None
        }
        with self.assertRaisesRegex(RuntimeError, "no explicit CPU planning"):
            builder._solve_full75_actor_framing(self._request(), self._suite())

    @patch.object(builder, "solve_static_camera_candidates")
    @patch.object(builder, "build_actor_framing_frames")
    @patch.object(builder, "evaluate_camera_candidates")
    def test_runtime_helper_forwards_only_runtime_admitted_candidates(
        self, evaluate: object, build_actor: object, solve_camera: object
    ) -> None:
        request = self._request()
        request["actor_framing"]["actor_bindings"] = [  # type: ignore[index]
            {"actor_id": "source1_actor", "asset_id": "male", "asset_revision": "r1"},
            {"actor_id": "source2_actor", "asset_id": "female", "asset_revision": "r1"},
        ]
        request["camera_framing"]["floor_height_m"] = 0.0  # type: ignore[index]
        request["camera_framing"]["candidate_generation"] = {  # type: ignore[index]
            "eye_height_m": 1.5,
            "offsets_xz_m": [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]],
        }
        suite = self._suite()
        suite["scenarios"][0]["plan"]["actors"] = [  # type: ignore[index]
            {
                "actor_id": "source1_actor",
                "asset_id": "male",
                "asset_revision": "r1",
                "emitter_offset_m": [0.0, 1.6, 0.0],
            },
            {
                "actor_id": "source2_actor",
                "asset_id": "female",
                "asset_revision": "r1",
                "emitter_offset_m": [0.0, 1.55, 0.0],
            },
        ]
        for frame in suite["scenarios"][0]["plan"]["frames"]:  # type: ignore[index]
            frame["actor_states"][0].update(  # type: ignore[index]
                {"asset_id": "male", "translation_m": [0.0, 0.0, -2.0]}
            )
            frame["actor_states"].append(  # type: ignore[index]
                {
                    "actor_id": "source2_actor",
                    "asset_id": "female",
                    "translation_m": [1.0, 0.0, -2.0],
                }
            )
        evaluate.return_value = [  # type: ignore[attr-defined]
            {
                "candidate_id": "midpoint_grid_000",
                "status": "pass",
                "room_gate": {
                    "status": "pass",
                    "authority_id": "runtime/declared0",
                    "provenance": "habitat_cpu_runtime",
                    "native_habitat_validation_status": "pass",
                    "line_of_sight_validation_status": "pass",
                    "full_body_clearance_status": "pending_live_ue",
                    "hard_gates": {"listener_navmesh": {"status": "pass"}},
                },
            },
            {
                "candidate_id": "midpoint_grid_001",
                "status": "fail",
                "room_gate": None,
            },
        ]
        build_actor.return_value = {  # type: ignore[attr-defined]
            "frames": [
                {
                    "frame_index": index,
                    "actor_aabbs": {
                        "source1_actor": {
                            "minimum_m": [-0.2, 0.0, -2.2],
                            "maximum_m": [0.2, 1.8, -1.8],
                        },
                        "source2_actor": {
                            "minimum_m": [0.8, 0.0, -2.2],
                            "maximum_m": [1.2, 1.8, -1.8],
                        },
                    },
                }
                for index in range(75)
            ]
        }
        solve_camera.return_value = {  # type: ignore[attr-defined]
            "selected_candidate_id": "midpoint_grid_000",
            "sensor_rig_binding": {"trajectory": {"frames": []}},
        }

        _, solution, gates = builder._runtime_gate_and_solve_full75_actor_framing(
            request, suite, runtime_provider=object()
        )

        self.assertEqual(solution["selected_candidate_id"], "midpoint_grid_000")
        self.assertEqual(len(gates), 2)
        self.assertEqual(
            build_actor.return_value["actor_orientation_policy"],
            "frozen_suite_actor_states_not_retargeted_to_selected_camera",
        )
        self.assertEqual(
            list(evaluate.call_args.kwargs["actor_visibility_anchors_m"]),
            ["source1_actor", "source2_actor"],
        )
        anchors = evaluate.call_args.kwargs["actor_visibility_anchors_m"][
            "source1_actor"
        ]
        self.assertEqual(
            set(anchors), {"torso_envelope_center", "declared_emitter_proxy"}
        )
        forwarded = solve_camera.call_args.kwargs["candidates"]
        self.assertEqual(
            [item["candidate_id"] for item in forwarded], ["midpoint_grid_000"]
        )
        self.assertEqual(forwarded[0]["room_gate"]["provenance"], "habitat_cpu_runtime")
        self.assertTrue(forwarded[0]["room_gate"]["hard_gates"])

    def test_selected_hold_rig_replaces_plan_and_all_camera_states(self) -> None:
        suite = self._suite()
        suite["scenarios"][0]["scenario_id"] = "episode0"  # type: ignore[index]
        suite["scenarios"][0]["plan"]["camera"] = {  # type: ignore[index]
            "listener_id": "listener0",
            "horizontal_fov_deg": 90.0,
        }
        for frame in suite["scenarios"][0]["plan"]["frames"]:  # type: ignore[index]
            frame["pts_ticks"] = frame["frame_index"] * 3200  # type: ignore[index]
        rig = {
            "trajectory_id": "episode0__sensor_rig",
            "frames": [
                {
                    "frame_index": index,
                    "pts_ticks": index * 3200,
                    "pose_hash": "selected-pose",
                    "world_from_rig": {
                        "translation_m": [2.0, 1.5, -3.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                }
                for index in range(75)
            ],
        }
        solution = {
            "selected_camera_pose": {"position_m": [2.0, 1.5, -3.0], "yaw_deg": 0.0},
            "sensor_rig_binding": {"trajectory": rig},
        }

        applied, selected = builder._apply_selected_sensor_rig(suite, solution)

        camera = applied["scenarios"][0]["plan"]["camera"]
        self.assertEqual(camera["habitat_position_m"], [2.0, 1.5, -3.0])
        self.assertEqual(camera["ue_position_cm"], [200.0, -300.0, 150.0])
        self.assertEqual(camera["ue_yaw_deg"], -90.0)
        self.assertEqual(camera["sensor_rig_trajectory_id"], selected["trajectory_id"])
        for frame, rig_frame in zip(
            applied["scenarios"][0]["plan"]["frames"], selected["frames"]
        ):
            self.assertEqual(
                frame["camera_state"]["world_from_rig"], rig_frame["world_from_rig"]
            )
            self.assertEqual(frame["camera_state"]["pose_hash"], "selected-pose")

    def test_canonical_rir_uses_selected_rig_and_both_source_paths(self) -> None:
        suite = self._suite()
        plan = suite["scenarios"][0]["plan"]  # type: ignore[index]
        suite["scenarios"][0]["scenario_id"] = "episode0"  # type: ignore[index]
        plan["actors"] = [
            {"actor_id": "source1_actor", "emitter_offset_m": [0.0, 1.6, 0.0]},
            {"actor_id": "source2_actor", "emitter_offset_m": [0.0, 1.5, 0.0]},
        ]
        for frame in plan["frames"]:
            frame["actor_states"] = [
                {"actor_id": "source1_actor", "translation_m": [0.0, 0.0, -2.0]},
                {"actor_id": "source2_actor", "translation_m": [1.0, 0.0, -2.0]},
            ]
        rig = {
            "frames": [
                {
                    "world_from_rig": {
                        "translation_m": [2.0, 1.5, -3.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                }
                for _ in range(75)
            ]
        }

        bank, rir = builder._build_canonical_rir_plan(suite, rig, stride_frames=3)

        self.assertEqual(bank["episode_count"], 1)
        self.assertEqual(rir["listener_pose_mode"], "per_episode_frame")
        self.assertEqual(rir["requested_pair_state_count"], 50)
        self.assertEqual(rir["unique_listener_pose_count"], 1)
        self.assertEqual(
            {use["source_slot_id"] for job in rir["jobs"] for use in job["uses"]},
            {"source1", "source2"},
        )


class PreflightTests(unittest.TestCase):
    def _v2_solution(self) -> tuple[dict[str, object], dict[str, object]]:
        position = [0.0, 1.572447, -0.5]
        yaw_deg = 60.62595999388584
        rig = materialize_sensor_rig_trajectory(
            trajectory_id=("mp3d_17DRP5sb8fy_male_female_static_0002__sensor_rig"),
            program={"kind": "HOLD", "position_m": position, "yaw_deg": yaw_deg},
        )
        actor = {
            "actor_orientation_policy": (
                "frozen_suite_actor_states_not_retargeted_to_selected_camera"
            ),
            "frames": [{"frame_index": index} for index in range(75)],
        }
        solution = {
            "status": "pass_cpu_declared_bounds_framing",
            "selected_candidate_id": "midpoint_grid_000",
            "selected_camera_pose": {"position_m": position, "yaw_deg": yaw_deg},
            "sensor_rig_binding": {"trajectory": rig},
        }
        return actor, solution

    @patch.object(builder, "_runtime_gate_and_solve_full75_actor_framing")
    def test_v2_build_binds_framing_rig_and_canonical_rir(
        self, runtime_solve: object
    ) -> None:
        actor, solution = self._v2_solution()
        runtime_solve.return_value = (  # type: ignore[attr-defined]
            actor,
            solution,
            [
                {
                    "candidate_id": "midpoint_grid_000",
                    "status": "pass",
                    "room_gate": {"status": "pass"},
                }
            ],
        )
        factory = MagicMock()
        factory.return_value.__enter__.return_value = object()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "v2"
            builder.build(_v2_args(output), runtime_provider_factory=factory)
            suite = json.loads((output / "suite_execution_plan.json").read_text())
            rig = json.loads((output / "sensor_rig_trajectory.json").read_text())
            rir = json.loads((output / "rir_job_plan.json").read_text())
            execution = json.loads((output / "execution_plan.json").read_text())
            preflight = json.loads((output / "preflight.json").read_text())

            self.assertTrue((output / "actor_framing.json").is_file())
            self.assertTrue((output / "camera_framing.json").is_file())
            self.assertTrue((output / "runtime_camera_gates.json").is_file())
            camera = suite["scenarios"][0]["plan"]["camera"]
            self.assertEqual(camera["sensor_rig_trajectory_id"], rig["trajectory_id"])
            for suite_frame, rig_frame in zip(
                suite["scenarios"][0]["plan"]["frames"], rig["frames"]
            ):
                self.assertEqual(
                    suite_frame["camera_state"]["world_from_rig"],
                    rig_frame["world_from_rig"],
                )
            self.assertEqual(rir["listener_pose_mode"], "per_episode_frame")
            self.assertEqual(rir["requested_pair_state_count"], 50)
            self.assertEqual(
                execution["schema"],
                "avengine_native_strict_two_human_mp3d_execution_plan_v2",
            )
            self.assertEqual(execution["remote_target_root"], str(output.parent))
            self.assertEqual(
                execution["cpu_steps"][0]["environment"]["CUDA_VISIBLE_DEVICES"],
                "",
            )
            compile_argv = execution["cpu_steps"][1]["argv"]
            self.assertIn(
                "habitat_mp3d_example_17DRP5sb8fy_soundspaces2_strict_two_human_v2",
                compile_argv,
            )
            self.assertIn(
                str(output.parent / "fresh_soundspaces2_package_v2"),
                compile_argv,
            )
            self.assertIn(
                str(output / "suite_execution_plan.json"),
                execution["gpu_steps"][0]["argv"],
            )
            probe_index = compile_argv.index("--probe-origin")
            self.assertEqual(
                compile_argv[probe_index + 1 : probe_index + 4],
                [
                    str(component)
                    for component in solution["selected_camera_pose"]["position_m"]
                ],
            )
            self.assertIn(
                str(output.parent / "exact_rir_cache_v1"),
                execution["cpu_steps"][2]["argv"],
            )
            legacy_rir_argv = execution["cpu_steps"][2]["argv"]
            self.assertNotIn("--semantic-no-file-evidence", legacy_rir_argv)
            for option in (
                "--room-id",
                "--room-revision",
                "--room-registry",
                "--acoustic-profile-registry",
                "--simulation-profile",
            ):
                self.assertIn(option, legacy_rir_argv)
            self.assertEqual(
                preflight["runtime_camera_framing"]["selected_candidate_id"],
                "midpoint_grid_000",
            )
            self.assertEqual(preflight["status"], "pending_remaining_evidence")
            self.assertEqual(preflight["cpu_planning_status"], "pass")
            self.assertFalse(preflight["episode_ready"])
            self.assertFalse(preflight["capture_ready"])
            self.assertFalse(preflight["formal_ready"])
            self.assertGreaterEqual(
                preflight["planned_projection"]["horizontal_mouth_separation_px"],
                preflight["planned_projection"][
                    "minimum_horizontal_mouth_separation_px"
                ],
            )
            self.assertEqual(
                preflight["planned_projection"][
                    "minimum_horizontal_mouth_separation_px"
                ],
                24.0,
            )
            factory.assert_called_once()

    def test_v3_request_selects_semantic_rir_and_prioritizes_fresh_rig(self) -> None:
        request_path = (
            ROOT / "examples/qa/native_strict_two_human_mp3d_room_atom_v3.json"
        )
        request = json.loads(request_path.read_text())

        builder._validate_request(request)

        self.assertEqual(request["schema"], builder.REQUEST_SCHEMA_V2)
        self.assertEqual(
            request["request_id"],
            "mp3d_17DRP5sb8fy_strict_two_human_static_rig_v3",
        )
        self.assertEqual(
            request["episode_id"],
            "mp3d_17DRP5sb8fy_male_female_static_rig_0003",
        )
        self.assertEqual(
            request["camera_framing"]["candidate_generation"]["offsets_xz_m"][0],
            [4.175, 0.0, 2.35],
        )
        self.assertEqual(request["acoustics"]["rir_stride_frames"], 1)
        self.assertEqual(
            request["acoustics"]["rir_execution_mode"],
            builder.SEMANTIC_RIR_EXECUTION_MODE,
        )

    def test_v3_execution_plan_uses_only_explicit_semantic_rir_inputs(self) -> None:
        request = json.loads(
            (
                ROOT / "examples/qa/native_strict_two_human_mp3d_room_atom_v3.json"
            ).read_text()
        )
        _, solution = self._v2_solution()
        rig = solution["sensor_rig_binding"]["trajectory"]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cpu_preflight_v1"
            with patch.object(
                builder,
                "_canonical_sha256",
                side_effect=AssertionError("semantic execution plan must not hash"),
            ):
                execution = builder._execution_plan(request, output, rig=rig)

            argv = execution["cpu_steps"][2]["argv"]
            self.assertIn("--semantic-no-file-evidence", argv)
            self.assertEqual(
                argv[argv.index("--acoustic-package-manifest") + 1],
                str(Path(temporary) / "fresh_soundspaces2_package_v2/manifest.json"),
            )
            self.assertEqual(
                argv[argv.index("--simulation-request") + 1],
                request["acoustics"]["simulation_request"],
            )
            self.assertEqual(
                argv[argv.index("--hrtf") + 1], request["acoustics"]["hrtf"]
            )
            for option in (
                "--room-id",
                "--room-revision",
                "--room-registry",
                "--acoustic-profile-registry",
                "--simulation-profile",
                "--job-offset",
                "--job-limit",
            ):
                self.assertNotIn(option, argv)

    def test_v3_rejects_unknown_rir_execution_mode(self) -> None:
        request = json.loads(
            (
                ROOT / "examples/qa/native_strict_two_human_mp3d_room_atom_v3.json"
            ).read_text()
        )
        request["acoustics"]["rir_execution_mode"] = "semantic_typo"
        with self.assertRaisesRegex(RuntimeError, "RIR execution mode is invalid"):
            builder._validate_request(request)

    @patch.object(builder, "_runtime_gate_and_solve_full75_actor_framing")
    def test_v2_runtime_gate_failure_creates_no_output(
        self, runtime_solve: object
    ) -> None:
        runtime_solve.side_effect = RuntimeError("runtime gate rejected")  # type: ignore[attr-defined]
        factory = MagicMock()
        factory.return_value.__enter__.return_value = object()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "v2"
            with self.assertRaisesRegex(RuntimeError, "runtime gate rejected"):
                builder.build(_v2_args(output), runtime_provider_factory=factory)
            self.assertFalse(output.exists())

    def test_cpu_preflight_closes_safer_pair_but_keeps_live_review_pending(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "preflight"
            _build_output(output)
            preflight = json.loads((output / "preflight.json").read_text())
            suite = json.loads((output / "suite_execution_plan.json").read_text())
            rir = json.loads((output / "rir_job_plan.json").read_text())
            execution = json.loads((output / "execution_plan.json").read_text())
            room = json.loads((output / "room_adapter.json").read_text())

            self.assertEqual(preflight["status"], "pass")
            self.assertFalse(preflight["gpu_f15_request_ready"])
            self.assertEqual(
                preflight["gpu_f15_request_status"],
                "pending_live_male_female_bbox_and_mouth_review",
            )
            adult_gate = preflight["navigation"]["adult_static_pair_gate"]
            self.assertEqual(adult_gate["status"], "pass")
            self.assertEqual(adult_gate["minimum_each_root_clearance_m"], 0.5)
            self.assertEqual(adult_gate["minimum_horizontal_separation_m"], 1.3)
            self.assertGreaterEqual(
                preflight["navigation"]["horizontal_source_separation_m"], 1.3
            )
            for source in preflight["navigation"]["selected_positions"].values():
                self.assertGreaterEqual(source["fresh_clearance_m"], 0.5)
            self.assertEqual(preflight["formal_dataset_count"], 0)
            self.assertEqual(rir["unique_rir_job_count"], 2)
            self.assertEqual(len(rir["jobs"]), 2)
            self.assertEqual(len(suite["scenarios"][0]["plan"]["frames"]), 75)
            self.assertEqual(len(room["static_mesh_object_paths"]), 71)
            for record in preflight["inputs"].values():
                self.assertEqual(set(record), {"path"})
            self.assertNotIn("fact_sha256", json.dumps(suite, sort_keys=True))
            self.assertNotIn("byte_size", json.dumps(preflight, sort_keys=True))
            sparse = execution["gpu_steps"][0]
            self.assertEqual(sparse["step_id"], "sparse_f15_probe")
            self.assertTrue(
                sparse["cpu_preconditions"][
                    "fresh_navmesh_each_root_clearance_at_least_0_5m"
                ]
            )

            remote_root = "/data/jzy/code/AVEngine-lead-a"
            spear_python = "/data/jzy/miniconda3/envs/spear-env/bin/python"
            fresh_package = (
                f"{remote_root}/tmp/lead_a_mp3d_strict_two_human_room_atom_v1/"
                "fresh_soundspaces2_package_v1"
            )
            habitat_python = (
                "/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python"
            )
            runtime_step, compile_step, rir_step = execution["cpu_steps"]
            self.assertEqual(
                runtime_step["step_id"],
                "probe_authoritative_habitat_rir_runtime",
            )
            self.assertEqual(runtime_step["argv"][0], habitat_python)
            self.assertFalse(runtime_step["expected"]["cuda_initialized"])
            self.assertEqual(runtime_step["expected"]["compute_device"], "CPU")
            self.assertEqual(
                compile_step["step_id"], "fresh_compile_mp3d_rlr_materials"
            )
            self.assertEqual(compile_step["working_directory"], remote_root)
            self.assertEqual(
                compile_step["argv"][:5],
                [
                    habitat_python,
                    "-m",
                    "avengine.cli",
                    "m3",
                    "compile-mp3d-rlr-materials",
                ],
            )
            self.assertNotIn("compile-registered-scene", compile_step["argv"])
            self.assertEqual(
                compile_step["expected"]["manifest"],
                f"{fresh_package}/manifest.json",
            )
            self.assertEqual(
                compile_step["expected"]["semantic_material_coverage"],
                f"{fresh_package}/semantic_material_coverage.json",
            )
            self.assertEqual(
                compile_step["expected"]["package_mode"], "research_candidate"
            )
            self.assertFalse(compile_step["expected"]["qualification_claim"])

            self.assertEqual(rir_step["step_id"], "render_two_exact_rirs")
            self.assertEqual(rir_step["attempt_id"], "exact_rir_cache_v4")
            self.assertEqual(
                rir_step["supersedes_failed_attempts"],
                [
                    "exact_rir_cache_v1",
                    "exact_rir_cache_v2",
                    "exact_rir_cache_v3",
                ],
            )
            self.assertEqual(
                rir_step["argv"][:2],
                [
                    habitat_python,
                    f"{remote_root}/tools/m6x/render_rir_cache.py",
                ],
            )
            self.assertNotIn("--job-limit", rir_step["argv"])
            expected_environment = {
                "AVENGINE_HABITAT_RUNTIME_ROOT": (
                    "/data/jzy/code/habitat-sim-AVEngine"
                ),
                "AVENGINE_SOUNDSPACES_ROOT": "/data/jzy/code/sound-spaces",
                "AVENGINE_MP3D_SOUNDSPACES2_PACKAGE_ROOT": fresh_package,
                "PATH": (
                    "/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin:"
                    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                ),
                "PYTHONPATH": f"{remote_root}/src",
                "SKBUILD_EDITABLE_SKIP": (
                    "/data/jzy/code/habitat-sim-AVEngine/build/cp312-cp312-linux_x86_64"
                ),
                "NUMBA_DISABLE_JIT": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
            self.assertEqual(rir_step["environment"], expected_environment)
            builder.validate_rir_runtime_binding(
                habitat_python, rir_step["environment"]
            )
            for missing_name in expected_environment:
                mutation = dict(expected_environment)
                mutation.pop(missing_name)
                with self.assertRaisesRegex(RuntimeError, missing_name):
                    builder.validate_rir_execution_environment(mutation)
            with self.assertRaisesRegex(RuntimeError, "runtime interpreter"):
                builder.validate_rir_runtime_binding(
                    spear_python,
                    expected_environment,
                )
            self.assertEqual(rir_step["expected"]["selected_job_count"], 2)
            self.assertTrue(rir_step["expected"]["full_plan_complete"])
            self.assertEqual(rir_step["expected"]["layout"], "binaural")
            self.assertEqual(
                rir["acoustic_state_sha256_authority"],
                "avengine.m6x.room_feasibility.rir_acoustic_state_sha256",
            )
            option_values = dict(zip(rir_step["argv"][2::2], rir_step["argv"][3::2]))
            self.assertEqual(
                option_values["--room-id"], "habitat_mp3d_example_17DRP5sb8fy"
            )
            self.assertEqual(
                option_values["--room-revision"],
                "raw_v1_plus_declared_proxy_v2_research",
            )
            self.assertEqual(option_values["--layout"], "binaural")

            sparse_step, full75_step = execution["gpu_steps"]
            self.assertEqual(sparse_step["argv"][0], spear_python)
            self.assertEqual(full75_step["argv"][0], spear_python)
            self.assertNotEqual(sparse_step["argv"][0], habitat_python)
            self.assertNotEqual(full75_step["argv"][0], habitat_python)
            forbidden_local_environment = "/" + "." + "venv/"
            self.assertNotIn(
                forbidden_local_environment, json.dumps(execution, sort_keys=True)
            )
            self.assertEqual(
                {step["argv"][0] for step in execution["cpu_steps"]},
                {habitat_python},
            )
            self.assertIn(
                "/cpu_preflight_v5/",
                next(
                    value
                    for value in sparse_step["argv"]
                    if value.endswith("suite_execution_plan.json")
                ),
            )

            try:
                from avengine.m6x.rir_cache import (
                    RIRCacheError,
                    validate_rir_job_plan,
                )
            except ModuleNotFoundError:
                pass
            else:
                validated_jobs = validate_rir_job_plan(rir)
                self.assertEqual(len(validated_jobs), 2)
                self.assertEqual(
                    list(validated_jobs[0]["source_position_m"]),
                    rir["jobs"][0]["source_position_m"],
                )
                self.assertEqual(
                    list(validated_jobs[0]["listener_position_m"]),
                    rir["jobs"][0]["listener_position_m"],
                )
                self.assertEqual(
                    list(validated_jobs[0]["listener_orientation_wxyz"]),
                    rir["jobs"][0]["listener_orientation_wxyz"],
                )
                mutation = deepcopy(rir)
                mutation["jobs"][0]["source_position_m"][0] += 0.001
                with self.assertRaisesRegex(
                    RIRCacheError, "acoustic-state SHA-256 differs from its pose"
                ):
                    validate_rir_job_plan(mutation)

            scenario, frames = capture.validate_capture_contract(
                suite,
                scenario_id=suite["scenarios"][0]["scenario_id"],
                room_adapter=room,
                requested_frame_indices=[15],
            )
            self.assertEqual(scenario["plan"]["camera"]["listener_id"], "listener0")
            self.assertEqual([item["frame_index"] for item in frames], [15])


if __name__ == "__main__":
    unittest.main()
