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


def _v4_args(output: Path) -> argparse.Namespace:
    request = ROOT / "examples/qa/native_strict_two_human_mp3d_room_atom_v4.json"
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

    def test_v4_projects_selected_suite_state_into_existing_m1_shape(self) -> None:
        request = json.loads(
            (
                ROOT / "examples/qa/native_strict_two_human_mp3d_room_atom_v4.json"
            ).read_text()
        )
        template = json.loads(
            (ROOT / "examples/rooms/requests/habitat_mp3d_example.json").read_text()
        )
        original_template = deepcopy(template)
        pose = {
            "translation_m": [0.0, 1.572447, -0.5],
            "rotation_xyzw": [0.0, 0.5, 0.0, 0.8660254037844386],
        }
        source1_rotation = [0.0, 0.25, 0.0, 0.9682458365518543]
        source2_rotation = [0.0, -0.25, 0.0, 0.9682458365518543]
        suite_frames = [
            {
                "frame_index": frame_index,
                "camera_state": {"world_from_rig": deepcopy(pose)},
                "actor_states": [
                    {
                        "actor_id": "source1_actor",
                        "rotation_xyzw": source1_rotation,
                    },
                    {
                        "actor_id": "source2_actor",
                        "rotation_xyzw": source2_rotation,
                    },
                ],
            }
            for frame_index in range(75)
        ]
        suite = {
            "scenarios": [
                {
                    "scenario_id": request["episode_id"],
                    "render": {
                        "frame_count": 75,
                        "frame_rate_hz": 15,
                        "height": 720,
                        "width": 1280,
                        "horizontal_fov_deg": 90.0,
                    },
                    "plan": {"frames": suite_frames},
                }
            ]
        }
        rig = {
            "frames": [
                {
                    "frame_index": frame_index,
                    "world_from_rig": deepcopy(pose),
                }
                for frame_index in range(75)
            ]
        }
        source1 = [[-4.6, 1.682447, -2.35] for _ in range(75)]
        source2 = [[-3.75, 1.641459451171875, -3.35] for _ in range(75)]
        trajectory = {
            "episodes": [
                {
                    "episode_id": request["episode_id"],
                    "source_center_paths_m": {
                        "source1": source1,
                        "source2": source2,
                    },
                }
            ]
        }

        projected = builder._project_habitat_m1_capture_request(
            request, template, suite, rig, trajectory
        )

        self.assertEqual(template, original_template)
        self.assertEqual(
            projected["request_id"], f"{request['request_id']}__habitat_m1_capture"
        )
        self.assertEqual(projected["room_id"], request["room"]["room_id"])
        self.assertEqual(projected["primary_camera_rig"]["world_from_rig"], pose)
        self.assertEqual(
            projected["primary_camera_rig"]["shared_calibration"]["resolution_hw"],
            [720, 1280],
        )
        self.assertEqual(
            [source["source_id"] for source in projected["sources"]],
            ["source1", "source2"],
        )
        self.assertEqual(
            projected["sources"][0]["world_from_source"],
            {"translation_m": source1[0], "rotation_xyzw": source1_rotation},
        )
        self.assertEqual(
            projected["sources"][1]["world_from_source"],
            {"translation_m": source2[0], "rotation_xyzw": source2_rotation},
        )
        self.assertEqual(projected["qa_views"], template["qa_views"])

        rotation_drift = deepcopy(suite)
        rotation_drift["scenarios"][0]["plan"]["frames"][74]["actor_states"][1][
            "rotation_xyzw"
        ] = [0.0, 0.0, 0.0, 1.0]
        with self.assertRaisesRegex(RuntimeError, "rotations must remain frozen"):
            builder._project_habitat_m1_capture_request(
                request, template, rotation_drift, rig, trajectory
            )

        malformed_template = deepcopy(template)
        malformed_template["primary_camera_rig"]["modalities"] = [None]
        with self.assertRaisesRegex(RuntimeError, "camera/listener template drift"):
            builder._project_habitat_m1_capture_request(
                request, malformed_template, suite, rig, trajectory
            )

        missing_sensor_uuid = deepcopy(template)
        missing_sensor_uuid["primary_camera_rig"]["modalities"][0].pop("sensor_uuid")
        with self.assertRaisesRegex(RuntimeError, "camera/listener template drift"):
            builder._project_habitat_m1_capture_request(
                request, missing_sensor_uuid, suite, rig, trajectory
            )

        invalid_center = deepcopy(trajectory)
        invalid_center["episodes"][0]["source_center_paths_m"]["source1"] = [
            ["not-a-number", 1.0, 2.0] for _ in range(75)
        ]
        with self.assertRaisesRegex(RuntimeError, "source1 source center"):
            builder._project_habitat_m1_capture_request(
                request, template, suite, rig, invalid_center
            )

        nonunit_rotation = deepcopy(suite)
        for frame in nonunit_rotation["scenarios"][0]["plan"]["frames"]:
            frame["actor_states"][0]["rotation_xyzw"] = [0.0, 0.0, 0.0, 2.0]
        with self.assertRaisesRegex(RuntimeError, "unit normalized"):
            builder._project_habitat_m1_capture_request(
                request, template, nonunit_rotation, rig, trajectory
            )

    def test_v3_is_not_implicitly_promoted_to_habitat_production(self) -> None:
        request = json.loads(
            (
                ROOT / "examples/qa/native_strict_two_human_mp3d_room_atom_v3.json"
            ).read_text()
        )
        self.assertFalse(builder._is_habitat_native_production(request))
        builder._validate_request(request)

        request["visual_execution_mode"] = "habitat_native_typo"
        with self.assertRaisesRegex(RuntimeError, "visual execution mode is invalid"):
            builder._validate_request(request)

        request["visual_execution_mode"] = None
        with self.assertRaisesRegex(RuntimeError, "visual execution mode is invalid"):
            builder._validate_request(request)

    def test_v4_execution_plan_routes_habitat_and_keeps_spear_comparison(self) -> None:
        request_path = (
            ROOT / "examples/qa/native_strict_two_human_mp3d_room_atom_v4.json"
        )
        request = json.loads(request_path.read_text())
        builder._validate_request(request)
        self.assertTrue(builder._is_habitat_native_production(request))
        _, solution = self._v2_solution()
        rig = solution["sensor_rig_binding"]["trajectory"]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cpu_preflight_v1"
            execution = builder._execution_plan(
                request,
                output,
                rig=rig,
                request_path=request_path,
            )
            legacy_request = deepcopy(request)
            legacy_request.pop("visual_execution_mode")
            legacy = builder._execution_plan(legacy_request, output, rig=rig)

            self.assertEqual(execution["comparison_gpu_steps"], legacy["gpu_steps"])
            self.assertNotIn("comparison_gpu_steps", legacy)
            self.assertEqual(len(execution["gpu_steps"]), 1)
            production = execution["gpu_steps"][0]
            self.assertEqual(production["step_id"], "full75_habitat_production_episode")
            self.assertEqual(production["backend_role"], "production_visual")
            self.assertEqual(
                production["environment"],
                {
                    "AVENGINE_HABITAT_RUNTIME_ROOT": builder.HABITAT_RUNTIME_ROOT,
                    "CUDA_VISIBLE_DEVICES": "1",
                    "NUMBA_DISABLE_JIT": "1",
                    "PATH": builder.HABITAT_PATH,
                    "PYTHONPATH": str(builder.REMOTE_REPOSITORY / "src"),
                    "SKBUILD_EDITABLE_SKIP": builder.HABITAT_EDITABLE_BUILD,
                },
            )
            self.assertEqual(
                production["argv"],
                [
                    str(builder.HABITAT_PYTHON),
                    str(builder.HABITAT_TWO_HUMAN_CAPTURE),
                    "--atom-request",
                    str(request_path),
                    "--suite-plan",
                    str(output / "suite_execution_plan.json"),
                    "--sensor-rig",
                    str(output / "sensor_rig_trajectory.json"),
                    "--trajectory-bank",
                    str(output / "trajectory_bank.json"),
                    "--rir-plan",
                    str(output / "rir_job_plan.json"),
                    "--room-manifest",
                    str(builder.HABITAT_M1_ROOM_MANIFEST),
                    "--m1-request",
                    str(output / "habitat_m1_capture_request.json"),
                    "--output",
                    str(Path(temporary) / "native_full75_habitat_v1"),
                    "--runtime-root",
                    builder.HABITAT_RUNTIME_ROOT,
                ],
            )
            for forbidden in (
                "--room-adapter",
                "--spear-root",
                "--graphics-adapter",
                "--rpc-port",
            ):
                self.assertNotIn(forbidden, production["argv"])
            self.assertEqual(production["expected"]["formal_dataset_count"], 0)
            self.assertTrue(production["expected"]["research_only"])

            (Path(temporary) / "native_full75_habitat_v1").mkdir()
            with self.assertRaisesRegex(
                RuntimeError, "Habitat full75 production capture target already exists"
            ):
                builder._execution_plan(
                    request,
                    output,
                    rig=rig,
                    request_path=request_path,
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


if __name__ == "__main__":
    unittest.main()
