from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

TEST_FILE = Path(__file__).resolve()
ROOT = (
    TEST_FILE.parents[1]
    if (TEST_FILE.parents[1] / "reference").is_dir()
    else TEST_FILE.parents[2]
)
TOOLS = ROOT / "tools/qa"
sys.path.insert(0, str(TOOLS))

import build_strict_two_human_mp3d_room_preflight as builder
import capture_spear_imported_glb_strict_two_human_episode as capture
import spear_imported_glb_room_adapter as adapter_module


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
        component = FakeComponent(
            3000 + index, mismatch=index == self.mismatch_at
        )
        actor = FakeActor(2000 + index, component)
        self.actors.append(actor)
        return actor

    def get_component_by_class(
        self, *, actor: FakeActor, uclass: str
    ) -> FakeComponent:
        if uclass != "UStaticMeshComponent":
            raise AssertionError("unexpected component class")
        return actor.component

    def set_stable_name_for_actor(
        self, *, actor: FakeActor, stable_name: str
    ) -> None:
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
    request_path = (
        ROOT / "examples/qa/native_strict_two_human_mp3d_room_atom_v1.json"
    )
    request = json.loads(request_path.read_text())
    staging_reference = ROOT / "reference"
    if staging_reference.is_dir():
        inputs = {
            "template_suite": staging_reference
            / "strict_two_human_template_suite.json",
            "ue_import_manifest": staging_reference / "mp3d_ue_import_result.json",
            "ue_runtime_evidence": staging_reference
            / "mp3d_ue_runtime_evidence.json",
            "fresh_navmesh_probe": staging_reference / "fresh_navmesh_probe.json",
            "acoustic_manifest": staging_reference
            / "mp3d_soundspaces2_manifest.json",
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


class RoomAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, inputs = _declared_inputs()
        cls.import_manifest = json.loads(
            inputs["ue_import_manifest"].read_text()
        )
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

    def test_fake_runtime_rejects_component_mesh_mismatch(self) -> None:
        game = FakeGame(mismatch_at=9)
        with self.assertRaisesRegex(RuntimeError, "readback differs at index 9"):
            adapter_module.spawn_scene_meshes_with_readback(game, self.adapter)

    def test_fake_runtime_rejects_nonunique_loaded_objects(self) -> None:
        game = FakeGame(duplicate_loads=True)
        with self.assertRaisesRegex(RuntimeError, "71 unique live objects"):
            adapter_module.spawn_scene_meshes_with_readback(game, self.adapter)


class PreflightTests(unittest.TestCase):
    def test_cpu_preflight_closes_safer_pair_but_keeps_live_review_pending(self) -> None:
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
            sparse = execution["gpu_steps"][0]
            self.assertEqual(sparse["step_id"], "sparse_f15_probe")
            self.assertTrue(
                sparse["cpu_preconditions"][
                    "fresh_navmesh_each_root_clearance_at_least_0_5m"
                ]
            )

            remote_root = "/data/jzy/code/AVEngine-lead-a"
            python = f"{remote_root}/.venv/bin/python"
            fresh_package = (
                f"{remote_root}/tmp/lead_a_mp3d_strict_two_human_room_atom_v1/"
                "fresh_soundspaces2_package_v1"
            )
            compile_step, rir_step = execution["cpu_steps"]
            self.assertEqual(
                compile_step["step_id"], "fresh_compile_mp3d_rlr_materials"
            )
            self.assertEqual(compile_step["working_directory"], remote_root)
            self.assertEqual(compile_step["argv"][:5], [
                python,
                "-m",
                "avengine.cli",
                "m3",
                "compile-mp3d-rlr-materials",
            ])
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
            self.assertEqual(rir_step["argv"][:2], [
                python,
                f"{remote_root}/tools/m6x/render_rir_cache.py",
            ])
            self.assertNotIn("--job-limit", rir_step["argv"])
            self.assertEqual(
                rir_step["environment"][
                    "AVENGINE_MP3D_SOUNDSPACES2_PACKAGE_ROOT"
                ],
                fresh_package,
            )
            self.assertEqual(rir_step["expected"]["selected_job_count"], 2)
            self.assertTrue(rir_step["expected"]["full_plan_complete"])
            self.assertEqual(rir_step["expected"]["layout"], "binaural")

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
