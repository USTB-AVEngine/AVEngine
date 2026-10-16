from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

LAUNCHER_NAME = "tools/qa/run_strict_two_human_mp3d_f15_probe.py"
LAUNCHER_PATH = next(
    candidate / LAUNCHER_NAME
    for candidate in Path(__file__).resolve().parents
    if (candidate / LAUNCHER_NAME).is_file()
)
CAPTURE_PATH = LAUNCHER_PATH.with_name(
    "capture_spear_imported_glb_strict_two_human_episode.py"
)


def _load_launcher() -> ModuleType:
    name = "avengine_test_mp3d_f15_launcher"
    spec = importlib.util.spec_from_file_location(name, LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LAUNCHER = _load_launcher()


def _load_capture() -> ModuleType:
    name = "avengine_test_mp3d_f15_capture"
    spec = importlib.util.spec_from_file_location(name, CAPTURE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CAPTURE = _load_capture()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_v7_evidence_fixture(root: Path) -> dict[str, Path]:
    preflight_root = root / "cpu_preflight_v5"
    package_root = root / "package"
    cache_root = root / "cache"
    paths = {
        "execution_plan": preflight_root / "execution_plan.json",
        "preflight": preflight_root / "preflight.json",
        "room_adapter": preflight_root / "room_adapter.json",
        "suite_plan": preflight_root / "suite_execution_plan.json",
        "rir_runtime_probe": preflight_root / "rir_runtime_probe.json",
        "package_manifest": package_root / "manifest.json",
        "package_material_coverage": package_root / "coverage.json",
        "rir_cache_receipt": cache_root / "receipt.json",
        "rir_cache_index": cache_root / "index.json",
        "capture_output": root / "native_sparse_f15_v1",
    }
    episode_id = "episode"
    scene_id = "scene"
    mesh_paths = [f"/Game/mesh_{index:03d}" for index in range(71)]
    roots = {"source1": [1.0, 1.0, 1.0], "source2": [4.0, 4.0, 4.0]}
    source_positions = {"source1": [1.0, 2.0, 1.0], "source2": [4.0, 5.0, 4.0]}
    listener = [7.0, 8.0, 9.0]
    actor_assets = {"source1": "male_asset", "source2": "female_asset"}
    plan = {
        "schema": "avengine_native_strict_two_human_mp3d_execution_plan_v1",
        "status": "planned_not_run",
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "cpu_steps": [
            {
                "step_id": "fresh_compile_mp3d_rlr_materials",
                "argv": ["compile", "--package-id", "package"],
            }
        ],
        "gpu_steps": [
            {
                "step_id": "sparse_f15_probe",
                "argv": ["capture", "--scenario-id", episode_id],
            }
        ],
    }
    room = {
        "schema": "avengine_spear_imported_glb_room_adapter_v1",
        "scene_id": scene_id,
        "expected_static_mesh_count": 71,
        "static_mesh_object_paths": mesh_paths,
        "camera_contract": {
            "one_camera_actor_for_all_passes": True,
            "pass_order": [
                "normal",
                "source1_target_only",
                "source2_target_only",
            ],
        },
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    _write(paths["execution_plan"], plan)
    _write(
        paths["preflight"],
        {
            "schema": "avengine_native_strict_two_human_mp3d_room_preflight_v1",
            "status": "pass",
            "episode_id": episode_id,
            "gpu_started": False,
            "gpu_f15_request_materialized": True,
            "gpu_f15_request_ready": False,
            "qualification_claim": False,
            "formal_dataset_count": 0,
            "episode_contract": {
                "frame_count": 75,
                "frame_rate_hz": 15,
                "sparse_probe_frame_indices": [15],
                "static_distinct_human_pair": [
                    actor_assets["source1"],
                    actor_assets["source2"],
                ],
            },
            "navigation": {
                "status": "pass",
                "fresh_pathfinder_replay_status": "pass",
                "shared_island_id": 1,
                "horizontal_source_separation_m": 1.5,
                "adult_static_pair_gate": {
                    "clearance_gate_passed": True,
                    "separation_gate_passed": True,
                },
                "selected_positions": {
                    slot: {
                        "all_frames_navigable": True,
                        "island_id": 1,
                        "fresh_clearance_m": 0.8,
                        "habitat_root_m": roots[slot],
                    }
                    for slot in ("source1", "source2")
                },
            },
            "rir": {
                "status": "planned_not_run",
                "compute_device": "CPU",
                "unique_rir_job_count": 2,
                "source_positions_m": source_positions,
                "listener_position_m": listener,
            },
        },
    )
    _write(paths["room_adapter"], room)
    _write(
        paths["suite_plan"],
        {
            "schema": "avengine_optional_spear_imported_glb_suite_v1",
            "native_map": "/Engine/Maps/Entry",
            "qualification_claim": False,
            "formal_dataset_count": 0,
            "scenarios": [
                {
                    "scenario_id": episode_id,
                    "plan": {
                        "frames": [
                            {
                                "frame_index": index,
                                "actor_states": [
                                    {
                                        "actor_id": f"{slot}_actor",
                                        "asset_id": actor_assets[slot],
                                        "translation_m": roots[slot],
                                    }
                                    for slot in ("source1", "source2")
                                ],
                                "camera_state": {"habitat_position_m": listener},
                            }
                            for index in range(75)
                        ],
                        "actors": [
                            {
                                "actor_id": "source1_actor",
                                "template_id": "male",
                                "asset_id": actor_assets["source1"],
                                "emitter_offset_m": [0.0, 1.0, 0.0],
                            },
                            {
                                "actor_id": "source2_actor",
                                "template_id": "female",
                                "asset_id": actor_assets["source2"],
                                "emitter_offset_m": [0.0, 1.0, 0.0],
                            },
                        ],
                        "camera": {"habitat_position_m": listener},
                        "room": {"scene_id": scene_id, "room_adapter": room},
                    },
                }
            ],
        },
    )
    _write(
        paths["rir_runtime_probe"],
        {
            "schema": "avengine_mp3d_rir_runtime_probe_v1",
            "status": "pass",
            "compute_device": "CPU",
            "gpu_required": False,
            "cuda_initialized": False,
            "qualification_claim": False,
            "formal_dataset_count": 0,
        },
    )
    _write(
        paths["package_manifest"],
        {
            "schema": "avengine_acoustic_scene_package_v1",
            "package_id": "package",
            "package_mode": "research_candidate",
            "room_kind": "habitat_native",
            "geometry": {"triangle_count": 3, "vertex_count": 4},
        },
    )
    _write(
        paths["package_material_coverage"],
        {
            "schema": "avengine_m3_rlr_semantic_material_coverage_v1",
            "status": "research_candidate",
            "qualification_claim": False,
            "compiled_triangle_count": 3,
            "triangle_coverage": {"triangle_count": 3},
            "runtime_one_to_one": {"passed": True},
        },
    )
    _write(
        paths["rir_cache_receipt"],
        {
            "schema": "avengine_rlr_rir_cache_receipt_v1",
            "status": "pass",
            "compute_device": "CPU",
            "layout_type": "binaural",
            "layout_id": "rlr_binaural_lr_v1",
            "channel_count": 2,
            "sample_rate_hz": 16000,
            "selected_job_count": 2,
            "full_plan_job_count": 2,
            "full_plan_complete": True,
            "producer_backend": "RLR Audio Propagation",
            "dry_audio_independent": True,
            "qualification_claim": False,
        },
    )
    _write(
        paths["rir_cache_index"],
        {
            "schema": "avengine_rlr_rir_cache_index_v1",
            "status": "pass",
            "selected_job_count": 2,
            "full_plan_complete": True,
            "entries": [
                {
                    "job_index": index,
                    "job_id": slot,
                    "sample_count": 8,
                    "source_position_m": source_positions[slot],
                    "listener_position_m": listener,
                    "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                }
                for index, slot in enumerate(("source1", "source2"))
            ],
        },
    )
    return paths


def _idle_snapshot() -> dict[str, object]:
    return {
        "captured_at_utc": "2026-08-12T00:00:00Z",
        "gpus": [
            {
                "physical_index": 1,
                "uuid": LAUNCHER.GPU1_UUID,
                "name": "GPU",
                "memory_used_mib": 19,
                "utilization_percent": 0,
            }
        ],
        "compute_apps": [],
    }


def _request(attempt_root: Path) -> dict[str, object]:
    return {
        "attempt_root": str(attempt_root),
        "capture_output": str(
            attempt_root.parent / "diagnostic_f15_capture_attempt_01"
        ),
        "required_repo_commit": "a" * 40,
        "rpc_port": 39631,
    }


def _request_v2(attempt_root: Path) -> dict[str, object]:
    return {
        "attempt_root": str(attempt_root),
        "capture_output": str(attempt_root.parent / LAUNCHER.V2_CAPTURE_DIRECTORY),
        "capture_stdout": str(attempt_root / "capture_stdout.log"),
        "capture_stderr": str(attempt_root / "capture_stderr.log"),
        "required_repo_commit": "b" * 40,
        "rpc_port": LAUNCHER.V2_RPC_PORT,
    }


def _write_packaged_readback(atom: Path) -> tuple[dict[str, object], Path, Path]:
    mesh_paths = [
        f"/Game/MyAssets/Audioset/Scenes/17DRP5sb8fy/mesh_{index:03d}.mesh_{index:03d}"
        for index in range(LAUNCHER.EXPECTED_MESH_COUNT)
    ]
    room_adapter: dict[str, object] = {"static_mesh_object_paths": mesh_paths}
    root = atom / LAUNCHER.PACKAGED_ROOM_READBACK_DIRECTORY
    result_path = root / "RESULT.json"
    exit_path = root / "EXIT.json"
    meshes = [
        {
            "mesh_index": index,
            "object_path": object_path,
            "stable_actor_name": (f"AVEngine/ImportedGLB/17DRP5sb8fy/mesh_{index:03d}"),
            "expected_object_handle": 10_000 + index,
            "observed_component_mesh_handle": 10_000 + index,
            "readback_method": "UStaticMeshComponent.StaticMesh_property",
            "status": "pass",
        }
        for index, object_path in enumerate(mesh_paths)
    ]
    _write(
        result_path,
        {
            "schema": "avengine_packaged_imported_glb_room_readback_v1",
            "status": "pass",
            "readiness_status": "packaged_71_mesh_readback_pass_gpu_f15_pending",
            "scene_id": "17DRP5sb8fy",
            "entry_map": "/Engine/Maps/Entry",
            "nullrhi": True,
            "rendering_or_capture_called": False,
            "qualification_claim": False,
            "formal_dataset_count": 0,
            "room_live_readback": {
                "schema": "avengine_spear_imported_glb_live_readback_v1",
                "status": "pass",
                "scene_id": "17DRP5sb8fy",
                "entry_map": "/Engine/Maps/Entry",
                "expected_static_mesh_count": 71,
                "spawned_static_mesh_count": 71,
                "unique_loaded_object_handle_count": 71,
                "unique_component_mesh_handle_count": 71,
                "all_expected_handles_match_components": True,
                "meshes": meshes,
                "qualification_claim": False,
                "formal_dataset_count": 0,
            },
        },
    )
    _write(
        exit_path,
        {
            "schema": "avengine_packaged_imported_glb_room_probe_exit_v1",
            "status": "pass",
            "worker_exit_code": 0,
            "timed_out": False,
            "exact_packaged_process_exit_closed": True,
            "exact_packaged_processes_before": [],
            "exact_packaged_processes_after": [],
            "nullrhi": True,
            "rendering_or_capture_called": False,
            "result_exists": True,
            "result_status": "pass",
            "semantic_error": None,
            "qualification_claim": False,
            "formal_dataset_count": 0,
        },
    )
    return room_adapter, result_path, exit_path


class Mp3dF15LauncherTests(unittest.TestCase):
    def test_capture_argv_is_exactly_one_f15_on_adapter1(self) -> None:
        request = {
            "episode_id": "dynamic_episode_0002",
            "capture_python": "/runtime/python",
            "capture_script": "/repo/capture.py",
            "suite_plan": "/evidence/suite.json",
            "room_adapter": "/evidence/room.json",
            "spear_root": "/runtime/SPEAR",
            "capture_output": "/evidence/capture",
            "rpc_port": 39631,
        }
        argv = LAUNCHER._capture_argv(request)
        self.assertEqual(argv.count("--frame-index"), 1)
        self.assertEqual(argv[argv.index("--frame-index") + 1], "15")
        self.assertEqual(argv.count("--graphics-adapter"), 1)
        self.assertEqual(argv[argv.index("--graphics-adapter") + 1], "1")
        self.assertEqual(argv[argv.index("--scenario-id") + 1], "dynamic_episode_0002")

    def test_gpu_gate_rejects_uuid_drift_and_busy_gpu1(self) -> None:
        snapshot = _idle_snapshot()
        self.assertEqual(
            LAUNCHER._validate_gpu1_idle(snapshot)["uuid"], LAUNCHER.GPU1_UUID
        )
        wrong = _idle_snapshot()
        wrong["gpus"][0]["uuid"] = "GPU-wrong"
        with self.assertRaisesRegex(RuntimeError, "UUID drift"):
            LAUNCHER._validate_gpu1_idle(wrong)
        busy = _idle_snapshot()
        busy["compute_apps"] = [
            {"gpu_uuid": LAUNCHER.GPU1_UUID, "pid": 7, "process_name": "python"}
        ]
        with self.assertRaisesRegex(RuntimeError, "not idle"):
            LAUNCHER._validate_gpu1_idle(busy)

    def test_artifact_binding_is_path_only_and_ignores_legacy_digest_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text("{}\n", encoding="utf-8")
            record = LAUNCHER._file_record(path)
            self.assertEqual(record, {"path": str(path.resolve())})
            self.assertEqual(
                LAUNCHER._validate_file_record(record, owner="evidence"),
                path.resolve(),
            )
            path.write_text("{ }\n", encoding="utf-8")
            legacy_record = {**record, "legacy_metadata": "ignored"}
            self.assertEqual(
                LAUNCHER._validate_file_record(legacy_record, owner="evidence"),
                path.resolve(),
            )

    def test_execution_plan_resolver_rejects_suite_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            preflight = repo / "tmp/atom/cpu_preflight_v3"
            preflight.mkdir(parents=True)
            plan_path = preflight / "execution_plan.json"
            atom = preflight.parent
            cpu_steps = [
                {
                    "step_id": "probe_authoritative_habitat_rir_runtime",
                    "expected": {"receipt": str(preflight / "runtime.json")},
                },
                {
                    "step_id": "fresh_compile_mp3d_rlr_materials",
                    "expected": {
                        "manifest": str(atom / "package/manifest.json"),
                        "semantic_material_coverage": str(
                            atom / "package/coverage.json"
                        ),
                    },
                },
                {
                    "step_id": "render_two_exact_rirs",
                    "expected": {
                        "receipt": str(atom / "cache/receipt.json"),
                        "index": str(atom / "cache/index.json"),
                    },
                },
            ]
            sparse_argv = [
                "python",
                "capture.py",
                "--suite-plan",
                str(repo.parent / "escaped_suite.json"),
                "--room-adapter",
                str(preflight / "room_adapter.json"),
                "--output",
                str(atom / "capture"),
                "--frame-index",
                "15",
                "--graphics-adapter",
                "1",
            ]
            _write(
                plan_path,
                {
                    "schema": "avengine_native_strict_two_human_mp3d_execution_plan_v2",
                    "qualification_claim": False,
                    "formal_dataset_count": 0,
                    "local_staging_output": str(preflight),
                    "remote_target_root": str(atom),
                    "cpu_steps": cpu_steps,
                    "gpu_steps": [
                        {"step_id": "sparse_f15_probe", "argv": sparse_argv},
                        {"step_id": "full75_episode", "argv": []},
                    ],
                },
            )
            with (
                mock.patch.object(LAUNCHER, "REPOSITORY", repo),
                self.assertRaisesRegex(RuntimeError, "suite plan escapes"),
            ):
                LAUNCHER._execution_plan_artifact_paths(plan_path)

    def test_execution_plan_package_id_mismatch_fails_closed(self) -> None:
        plan = {
            "cpu_steps": [
                {
                    "step_id": "fresh_compile_mp3d_rlr_materials",
                    "argv": ["compile", "--package-id", "package_from_plan"],
                }
            ]
        }
        package = {
            "schema": "avengine_acoustic_scene_package_v1",
            "package_id": "different_package",
            "package_mode": "research_candidate",
            "room_kind": "habitat_native",
            "geometry": {"triangle_count": 10, "vertex_count": 9},
        }
        coverage = {
            "schema": "avengine_m3_rlr_semantic_material_coverage_v1",
            "status": "research_candidate",
            "qualification_claim": False,
            "compiled_triangle_count": 10,
            "triangle_coverage": {"triangle_count": 10},
            "runtime_one_to_one": {"passed": True},
        }
        with self.assertRaisesRegex(RuntimeError, "fresh acoustic package drift"):
            LAUNCHER._validate_execution_plan_package(plan, package, coverage)

    def test_v5_prepare_emits_path_only_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = root / "atom/cpu_preflight_v3/execution_plan.json"
            plan.parent.mkdir(parents=True)
            plan.write_text("{}\n", encoding="utf-8")
            validation = {
                "episode_id": "episode_0002",
                "scene_id": "scene_dynamic",
                "execution_plan": str(plan),
                "evidence_paths": {
                    "preflight": str(plan.with_name("preflight.json")),
                    "suite_plan": str(plan.with_name("suite_execution_plan.json")),
                    "room_adapter": str(plan.with_name("room_adapter.json")),
                },
                "capture_output": str(root / "atom/native_sparse_f15_v1"),
                "capture_argv": ["python", "capture.py", "--frame-index", "15"],
            }
            with (
                mock.patch.object(
                    LAUNCHER,
                    "offline_validate_execution_plan",
                    return_value=validation,
                ),
                mock.patch.object(LAUNCHER, "_git_head", return_value="c" * 40),
            ):
                request_path = LAUNCHER.prepare_request_v5(execution_plan_path=plan)
            request = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(request["schema"], LAUNCHER.REQUEST_SCHEMA_V5)
            self.assertEqual(request["scene_id"], "scene_dynamic")
            self.assertIn("evidence_paths", request)
            self.assertEqual(
                request["suite_plan"], validation["evidence_paths"]["suite_plan"]
            )
            self.assertEqual(
                request["room_adapter"], validation["evidence_paths"]["room_adapter"]
            )

    def test_v5_offline_validate_and_dry_run_never_query_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / LAUNCHER.V5_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = {
                "attempt_root": str(attempt),
                "episode_id": "episode_0002",
                "scene_id": "scene_dynamic",
                "required_repo_commit": "d" * 40,
                "execution_plan": "/evidence/execution_plan.json",
                "capture_output": str(attempt.parent / "capture"),
            }
            argv = ["python", "capture.py", "--rpc-port", "39631"]
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v5", return_value=(request, argv)
                ),
                mock.patch.object(LAUNCHER, "_gpu_snapshot") as snapshot,
            ):
                self.assertEqual(
                    LAUNCHER.run_v5(
                        request_path,
                        offline_validate=True,
                        dry_run=False,
                        authorize_gpu_capture=False,
                    ),
                    0,
                )
                self.assertEqual(
                    LAUNCHER.run_v5(
                        request_path,
                        offline_validate=False,
                        dry_run=True,
                        authorize_gpu_capture=False,
                    ),
                    0,
                )
            snapshot.assert_not_called()
            receipt = json.loads(
                (attempt / "dry_run_receipt.json").read_text(encoding="utf-8")
            )
            self.assertFalse(receipt["gpu_query_started"])
            self.assertFalse(receipt["gpu_started"])

    def test_v6_nullrhi_result_and_exit_tamper_fail_closed(self) -> None:
        for mutation, error in (
            ("result", "71-mesh live readback drift"),
            ("exit", "readback EXIT boundary drift"),
        ):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                atom = Path(directory).resolve() / "tmp/atom"
                room, result_path, exit_path = _write_packaged_readback(atom)
                LAUNCHER._validate_packaged_room_readback(
                    atom_root=atom,
                    result_path=result_path,
                    exit_path=exit_path,
                    room_adapter=room,
                    scene_id="17DRP5sb8fy",
                )
                tampered_path = result_path if mutation == "result" else exit_path
                tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
                if mutation == "result":
                    tampered["room_live_readback"]["spawned_static_mesh_count"] = 70
                else:
                    tampered["exact_packaged_processes_after"] = [1491774]
                _write(tampered_path, tampered)
                with self.assertRaisesRegex(RuntimeError, error):
                    LAUNCHER._validate_packaged_room_readback(
                        atom_root=atom,
                        result_path=result_path,
                        exit_path=exit_path,
                        room_adapter=room,
                        scene_id="17DRP5sb8fy",
                    )

    def test_v6_nullrhi_accepts_getter_and_rejects_duplicate_handles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            atom = Path(directory).resolve() / "tmp/atom"
            room, result_path, exit_path = _write_packaged_readback(atom)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["room_live_readback"]["meshes"][0]["readback_method"] = (
                "UStaticMeshComponent.GetStaticMesh"
            )
            _write(result_path, result)
            LAUNCHER._validate_packaged_room_readback(
                atom_root=atom,
                result_path=result_path,
                exit_path=exit_path,
                room_adapter=room,
                scene_id="17DRP5sb8fy",
            )

            meshes = result["room_live_readback"]["meshes"]
            meshes[1]["expected_object_handle"] = meshes[0]["expected_object_handle"]
            meshes[1]["observed_component_mesh_handle"] = meshes[0][
                "observed_component_mesh_handle"
            ]
            _write(result_path, result)
            with self.assertRaisesRegex(RuntimeError, "identities are not unique"):
                LAUNCHER._validate_packaged_room_readback(
                    atom_root=atom,
                    result_path=result_path,
                    exit_path=exit_path,
                    room_adapter=room,
                    scene_id="17DRP5sb8fy",
                )

    def test_v6_prepare_and_dry_run_bind_paths_without_gpu_or_ue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            atom = root / "tmp/atom"
            plan = atom / "cpu_preflight_v3/execution_plan.json"
            plan.parent.mkdir(parents=True)
            plan.write_text("{}\n", encoding="utf-8")
            evidence_paths = {
                "execution_plan": str(plan),
                "preflight": str(plan.with_name("preflight.json")),
                "suite_plan": str(plan.with_name("suite_execution_plan.json")),
                "room_adapter": str(plan.with_name("room_adapter.json")),
                "packaged_room_readback_result": str(
                    atom / "packaged_room_readback_v1/RESULT.json"
                ),
                "packaged_room_readback_exit": str(
                    atom / "packaged_room_readback_v1/EXIT.json"
                ),
            }
            validation = {
                "episode_id": "episode_0002",
                "scene_id": "17DRP5sb8fy",
                "execution_plan": str(plan),
                "evidence_paths": evidence_paths,
                "capture_output": str(atom / "native_sparse_f15_v1"),
                "capture_argv": [
                    "python",
                    "capture.py",
                    "--rpc-port",
                    "39631",
                    "--frame-index",
                    "15",
                ],
            }
            bound_commit = "1" * 40
            with (
                mock.patch.object(
                    LAUNCHER,
                    "offline_validate_execution_plan_v6",
                    return_value=validation,
                ),
                mock.patch.object(LAUNCHER, "_git_head", return_value=bound_commit),
            ):
                request_path = LAUNCHER.prepare_request_v6(execution_plan_path=plan)
            request = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(request["schema"], LAUNCHER.REQUEST_SCHEMA_V6)
            self.assertEqual(request["required_repo_commit"], bound_commit)
            self.assertEqual(request["execution_plan"], str(plan))
            self.assertEqual(request["evidence_paths"], evidence_paths)
            self.assertEqual(
                request["capture_output"], str(atom / "native_sparse_f15_v1")
            )
            argv = validation["capture_argv"]
            with (
                mock.patch.object(
                    LAUNCHER,
                    "_validate_request_v6",
                    return_value=(request, argv),
                ),
                mock.patch.object(LAUNCHER, "_gpu_snapshot") as gpu_snapshot,
                mock.patch.object(LAUNCHER.subprocess, "run") as child,
            ):
                self.assertEqual(
                    LAUNCHER.run_v6(
                        request_path,
                        offline_validate=True,
                        dry_run=False,
                        authorize_gpu_capture=False,
                    ),
                    0,
                )
                self.assertEqual(
                    LAUNCHER.run_v6(
                        request_path,
                        offline_validate=False,
                        dry_run=True,
                        authorize_gpu_capture=False,
                    ),
                    0,
                )
            gpu_snapshot.assert_not_called()
            child.assert_not_called()
            receipt = json.loads(
                (request_path.parent / "dry_run_receipt.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(receipt["schema"], LAUNCHER.RECEIPT_SCHEMA_V6)
            self.assertEqual(receipt["evidence_paths"], evidence_paths)
            self.assertFalse(receipt["gpu_query_started"])
            self.assertFalse(receipt["gpu_started"])
            self.assertFalse(receipt["attempt_consumed"])


    def test_v7_rejects_missing_external_capture_runtime(self) -> None:
        for missing in ("python", "runner", "spear"):
            with (
                self.subTest(missing=missing),
                tempfile.TemporaryDirectory() as directory,
            ):
                repo = Path(directory).resolve()
                atom = repo / "tmp/atom"
                plan = atom / "cpu_preflight_v5/execution_plan.json"
                plan.parent.mkdir(parents=True)
                plan.write_text("{}\n", encoding="utf-8")
                capture_python = repo / "runtime/python"
                capture_runner = (
                    repo
                    / "tools/qa/capture_spear_imported_glb_strict_two_human_episode.py"
                )
                spear_root = repo / "external/SPEAR"
                if missing != "python":
                    capture_python.parent.mkdir(parents=True)
                    capture_python.write_text("runtime\n", encoding="utf-8")
                if missing != "runner":
                    capture_runner.parent.mkdir(parents=True)
                    capture_runner.write_text("runner\n", encoding="utf-8")
                if missing != "spear":
                    spear_root.mkdir(parents=True)
                argv = [
                    str(capture_python),
                    str(capture_runner),
                    "--spear-root",
                    str(spear_root),
                    "--graphics-adapter",
                    "1",
                    "--rpc-port",
                    "39631",
                    "--output",
                    str(atom / "native_sparse_f15_v1"),
                    "--frame-index",
                    "15",
                ]
                with (
                    mock.patch.object(LAUNCHER, "REPOSITORY", repo),
                    mock.patch.object(
                        LAUNCHER, "CAPTURE_PYTHON_LOGICAL", capture_python
                    ),
                    mock.patch.object(LAUNCHER, "SPEAR_ROOT", spear_root),
                    mock.patch.object(
                        LAUNCHER,
                        "_validate_v7_raw_evidence_paths",
                        return_value=plan,
                    ),
                    mock.patch.object(
                        LAUNCHER,
                        "_execution_plan_artifact_paths",
                        return_value={"execution_plan": plan},
                    ),
                    mock.patch.object(
                        LAUNCHER,
                        "_validate_v7_execution_plan_evidence",
                        return_value={
                            "episode_id": "episode",
                            "scene_id": "scene",
                            "capture_argv": argv,
                        },
                    ),
                    self.assertRaisesRegex(RuntimeError, "runtime missing or drifted"),
                ):
                    LAUNCHER.offline_validate_execution_plan_v7(plan)


    def test_v7_prepare_and_dry_run_bind_head_without_gpu_or_ue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            atom = root / "tmp/atom"
            plan = atom / "cpu_preflight_v5/execution_plan.json"
            plan.parent.mkdir(parents=True)
            plan.write_text("{}\n", encoding="utf-8")
            capture_output = atom / LAUNCHER.V7_CAPTURE_DIRECTORY
            evidence_paths = {
                "suite_plan": str(plan.with_name("suite_execution_plan.json")),
                "room_adapter": str(plan.with_name("room_adapter.json")),
            }
            argv = [
                "python",
                "capture.py",
                "--graphics-adapter",
                "1",
                "--rpc-port",
                str(LAUNCHER.V7_RPC_PORT),
                "--output",
                str(capture_output),
                "--frame-index",
                "15",
            ]
            validation = {
                "episode_id": "episode",
                "scene_id": "scene",
                "execution_plan": str(plan),
                "evidence_paths": evidence_paths,
                "capture_output": str(capture_output),
                "capture_argv": argv,
            }
            bound_commit = "7" * 40
            with (
                mock.patch.object(
                    LAUNCHER,
                    "offline_validate_execution_plan_v7",
                    return_value=validation,
                ),
                mock.patch.object(LAUNCHER, "_git_head", return_value=bound_commit),
                mock.patch.object(
                    LAUNCHER, "_git_tracked_and_index_clean", return_value=True
                ),
                mock.patch.object(LAUNCHER, "_assert_port_available") as port_check,
            ):
                request_path = LAUNCHER.prepare_request_v7(execution_plan_path=plan)
            port_check.assert_called_once_with(LAUNCHER.V7_RPC_PORT)
            request = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(request["schema"], LAUNCHER.REQUEST_SCHEMA_V7)
            self.assertEqual(request["required_repo_commit"], bound_commit)
            self.assertEqual(
                request["attempt_root"], str(atom / LAUNCHER.V7_ATTEMPT_DIRECTORY)
            )
            self.assertEqual(request["capture_output"], str(capture_output))
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v7", return_value=(request, argv)
                ),
                mock.patch.object(LAUNCHER, "_gpu_snapshot") as gpu_snapshot,
                mock.patch.object(LAUNCHER.subprocess, "run") as child,
            ):
                self.assertEqual(
                    LAUNCHER.run_v7(
                        request_path,
                        offline_validate=False,
                        dry_run=True,
                        authorize_gpu_capture=False,
                    ),
                    0,
                )
            gpu_snapshot.assert_not_called()
            child.assert_not_called()
            receipt = json.loads(
                (request_path.parent / "dry_run_receipt.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(receipt["schema"], LAUNCHER.RECEIPT_SCHEMA_V7)
            self.assertFalse(receipt["gpu_query_started"])
            self.assertFalse(receipt["gpu_started"])
            self.assertFalse(receipt["attempt_consumed"])

    def test_v7_semantic_evidence_ignores_legacy_digest_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_v7_evidence_fixture(Path(directory))
            source = inspect.getsource(LAUNCHER._validate_v7_execution_plan_evidence)
            for legacy_field in (
                "retained_payload_hash_verified",
                "request_identity_sha256",
                "acoustic_selection_binding_sha256",
                "ir_sha256",
            ):
                self.assertNotIn(legacy_field, source)
            evidence = LAUNCHER._validate_v7_execution_plan_evidence(paths)
            self.assertEqual(evidence["episode_id"], "episode")
            self.assertEqual(evidence["scene_id"], "scene")

    def test_v7_semantic_evidence_rejects_structural_drift(self) -> None:
        for mutation, message in (
            ("layout", "binaural RIR semantic/pose closure drift"),
            ("status", "execution-plan boundary drift"),
        ):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                paths = _write_v7_evidence_fixture(Path(directory))
                target = (
                    paths["rir_cache_receipt"]
                    if mutation == "layout"
                    else paths["execution_plan"]
                )
                document = json.loads(target.read_text(encoding="utf-8"))
                if mutation == "layout":
                    document["layout_type"] = "mono"
                else:
                    document["status"] = "failed"
                _write(target, document)
                with self.assertRaisesRegex(RuntimeError, message):
                    LAUNCHER._validate_v7_execution_plan_evidence(paths)

    def test_v7_rejects_rir_pose_drift_even_when_index_matches(self) -> None:
        for mutation in ("source", "listener"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                paths = _write_v7_evidence_fixture(Path(directory))
                preflight = json.loads(paths["preflight"].read_text(encoding="utf-8"))
                index = json.loads(paths["rir_cache_index"].read_text(encoding="utf-8"))
                if mutation == "source":
                    preflight["rir"]["source_positions_m"]["source1"][0] += 0.5
                    index["entries"][0]["source_position_m"][0] += 0.5
                else:
                    preflight["rir"]["listener_position_m"][2] += 0.5
                    for entry in index["entries"]:
                        entry["listener_position_m"][2] += 0.5
                _write(paths["preflight"], preflight)
                _write(paths["rir_cache_index"], index)
                with self.assertRaisesRegex(
                    RuntimeError, "binaural RIR semantic/pose closure drift"
                ):
                    LAUNCHER._validate_v7_execution_plan_evidence(paths)

    def test_v7_prepare_rejects_symlinked_execution_plan_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            preflight = repo / "tmp/atom/cpu_preflight_v5"
            preflight.mkdir(parents=True)
            real_plan = preflight / "real_execution_plan.json"
            real_plan.write_text("{}\n", encoding="utf-8")
            linked_plan = preflight / "execution_plan.json"
            linked_plan.symlink_to(real_plan.name)
            with (
                mock.patch.object(LAUNCHER, "REPOSITORY", repo),
                mock.patch.object(LAUNCHER, "_load") as load,
                self.assertRaisesRegex(RuntimeError, "symlink component"),
            ):
                LAUNCHER.prepare_request_v7(execution_plan_path=linked_plan)
            load.assert_not_called()
            self.assertFalse(
                (repo / "tmp/atom" / LAUNCHER.V7_ATTEMPT_DIRECTORY).exists()
            )

    def test_v7_rejects_raw_symlinked_declared_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            preflight = repo / "tmp/atom/cpu_preflight_v5"
            preflight.mkdir(parents=True)
            suite_real = preflight / "suite_real.json"
            suite_real.write_text("{}\n", encoding="utf-8")
            suite_link = preflight / "suite_execution_plan.json"
            suite_link.symlink_to(suite_real.name)
            room = preflight / "room_adapter.json"
            runtime = preflight / "rir_runtime_probe.json"
            package = repo / "tmp/atom/package"
            cache = repo / "tmp/atom/cache"
            plan_path = preflight / "execution_plan.json"
            _write(
                plan_path,
                {
                    "cpu_steps": [
                        {
                            "step_id": "probe_authoritative_habitat_rir_runtime",
                            "expected": {"receipt": str(runtime)},
                        },
                        {
                            "step_id": "fresh_compile_mp3d_rlr_materials",
                            "expected": {
                                "manifest": str(package / "manifest.json"),
                                "semantic_material_coverage": str(
                                    package / "coverage.json"
                                ),
                            },
                        },
                        {
                            "step_id": "render_two_exact_rirs",
                            "expected": {
                                "receipt": str(cache / "receipt.json"),
                                "index": str(cache / "index.json"),
                            },
                        },
                    ],
                    "gpu_steps": [
                        {
                            "step_id": "sparse_f15_probe",
                            "argv": [
                                "capture",
                                "--suite-plan",
                                str(suite_link),
                                "--room-adapter",
                                str(room),
                            ],
                        }
                    ],
                },
            )
            with (
                mock.patch.object(LAUNCHER, "REPOSITORY", repo),
                self.assertRaisesRegex(RuntimeError, "symlink component"),
            ):
                LAUNCHER._validate_v7_raw_evidence_paths(plan_path)

    def test_v7_clean_check_covers_tracked_index_and_untracked_files(self) -> None:
        cases = (
            ((0, 0), "", True),
            ((1, 0), "", False),
            ((0, 1), "", False),
            ((0, 0), "?? tools/qa/cv2.py\n", False),
        )
        for returncodes, untracked, expected in cases:
            with self.subTest(returncodes=returncodes, untracked=untracked):
                calls: list[list[str]] = []

                def fake_run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
                    calls.append(argv)
                    if argv[:2] == ["git", "status"]:
                        return SimpleNamespace(returncode=0, stdout=untracked)
                    return SimpleNamespace(
                        returncode=returncodes[len(calls) - 1], stdout=""
                    )

                with mock.patch.object(
                    LAUNCHER.subprocess, "run", side_effect=fake_run
                ):
                    self.assertEqual(
                        LAUNCHER._git_tracked_and_index_clean(Path("/repo")),
                        expected,
                    )
                self.assertEqual(
                    calls,
                    [
                        ["git", "diff", "--quiet", "--"],
                        ["git", "diff", "--cached", "--quiet", "--"],
                        [
                            "git",
                            "status",
                            "--porcelain=v1",
                            "--untracked-files=all",
                        ],
                    ],
                )

    def test_v7_rejects_dangling_and_parent_symlinks_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            dangling = root / "dangling"
            dangling.symlink_to(root / "missing", target_is_directory=True)
            self.assertFalse(dangling.exists())
            self.assertTrue(dangling.is_symlink())
            with self.assertRaisesRegex(RuntimeError, "symlink component"):
                LAUNCHER._require_v7_nonsymlink_path(
                    dangling, root, owner="v7 capture output"
                )

            real = root / "real"
            real.mkdir()
            parent_link = root / "linked"
            parent_link.symlink_to(real, target_is_directory=True)
            request_path = parent_link / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch.object(LAUNCHER, "REPOSITORY", root),
                mock.patch.object(LAUNCHER, "_load") as load,
                self.assertRaisesRegex(RuntimeError, "symlink component"),
            ):
                LAUNCHER._validate_request_v7(request_path)
            load.assert_not_called()

    def test_v7_prepare_rejects_dangling_capture_leaf_before_attempt_write(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            atom = Path(directory).resolve()
            plan = atom / "cpu_preflight_v5/execution_plan.json"
            plan.parent.mkdir()
            plan.write_text("{}\n", encoding="utf-8")
            capture = atom / LAUNCHER.V7_CAPTURE_DIRECTORY
            capture.symlink_to(atom / "missing", target_is_directory=True)
            validation = {
                "execution_plan": str(plan),
                "episode_id": "episode",
                "scene_id": "scene",
            }
            with (
                mock.patch.object(
                    LAUNCHER,
                    "offline_validate_execution_plan_v7",
                    return_value=validation,
                ),
                self.assertRaisesRegex(RuntimeError, "symlink component"),
            ):
                LAUNCHER.prepare_request_v7(execution_plan_path=plan)
            self.assertFalse((atom / LAUNCHER.V7_ATTEMPT_DIRECTORY).exists())

    def test_v7_capture_closes_hfov_passes_alignment_and_mesh_handles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            capture.mkdir()
            suite = root / "suite.json"
            room_adapter = root / "room.json"
            _write(
                suite,
                {"scenarios": [{"plan": {"camera": {"horizontal_fov_deg": 90.0}}}]},
            )
            mesh_paths = [f"/Game/mesh_{index:03d}" for index in range(71)]
            _write(room_adapter, {"static_mesh_object_paths": mesh_paths})
            handles = {"rgb": 501, "depth": 502, "object_ids": 503}
            identities = [
                {
                    "pass_id": pass_id,
                    "camera_actor_handle": 500,
                    "rgb_component_handle": 501,
                    "metric_depth_component_handle": 502,
                    "object_id_component_handle": 503,
                }
                for pass_id in (
                    "normal",
                    "source1_target_only",
                    "source2_target_only",
                )
            ]
            manifest = {
                "camera_contract": {
                    "pass_identities": identities,
                    "runtime_alignment": {
                        "normal_frame_count": 1,
                        "target_pass_count": 2,
                        "maximum_location_drift_cm": 0.0,
                        "maximum_rotation_drift_deg": 0.0,
                    },
                    "hfov_readback": {
                        "status": "pass",
                        "camera_actor_handle": 500,
                        "component_handles": handles,
                        "requested_horizontal_fov_deg": 90.0,
                        "observed_horizontal_fov_deg_by_component": {
                            name: 90.0 for name in handles
                        },
                        "write_method": (
                            "named_USpSceneCaptureComponent2D.FOVAngle_property"
                        ),
                    },
                }
            }
            meshes = [
                {
                    "mesh_index": index,
                    "object_path": path,
                    "status": "pass",
                    "readback_method": "UStaticMeshComponent.StaticMesh_property",
                    "stable_actor_name": f"AVEngine/ImportedGLB/scene/mesh_{index:03d}",
                    "expected_object_handle": 1000 + index,
                    "observed_component_mesh_handle": 1000 + index,
                }
                for index, path in enumerate(mesh_paths)
            ]
            readback = {
                "schema": "avengine_spear_imported_glb_live_readback_v1",
                "status": "pass",
                "scene_id": "scene",
                "entry_map": "/Engine/Maps/Entry",
                "qualification_claim": False,
                "formal_dataset_count": 0,
                "meshes": meshes,
            }
            _write(capture / "manifest.json", manifest)
            _write(capture / "room_live_readback.json", readback)
            request = {
                "capture_output": str(capture),
                "suite_plan": str(suite),
                "room_adapter": str(room_adapter),
                "scene_id": "scene",
            }
            with mock.patch.object(
                LAUNCHER, "_validate_capture", return_value={"status": "pass"}
            ):
                result = LAUNCHER._validate_v7_capture(request)
                self.assertEqual(
                    result["per_mesh_live_readback_status"], "pass_exact_71_of_71"
                )
                manifest["camera_contract"]["hfov_readback"]["camera_actor_handle"] = (
                    "500"
                )
                _write(capture / "manifest.json", manifest)
                with self.assertRaisesRegex(RuntimeError, "HFOV evidence drift"):
                    LAUNCHER._validate_v7_capture(request)
                manifest["camera_contract"]["hfov_readback"]["camera_actor_handle"] = (
                    500
                )
                manifest["camera_contract"]["hfov_readback"][
                    "observed_horizontal_fov_deg_by_component"
                ]["depth"] = 89.0
                _write(capture / "manifest.json", manifest)
                with self.assertRaisesRegex(RuntimeError, "HFOV values"):
                    LAUNCHER._validate_v7_capture(request)
                manifest["camera_contract"]["hfov_readback"][
                    "observed_horizontal_fov_deg_by_component"
                ]["depth"] = 90.0
                meshes[3]["observed_component_mesh_handle"] = 9999
                _write(capture / "manifest.json", manifest)
                _write(capture / "room_live_readback.json", readback)
                with self.assertRaisesRegex(
                    RuntimeError, "per-mesh live readback drift"
                ):
                    LAUNCHER._validate_v7_capture(request)
                meshes[3]["observed_component_mesh_handle"] = 1003
                for field, value in (
                    ("scene_id", "foreign"),
                    ("formal_dataset_count", 1),
                ):
                    with self.subTest(field=field):
                        original = readback[field]
                        readback[field] = value
                        _write(capture / "room_live_readback.json", readback)
                        with self.assertRaisesRegex(
                            RuntimeError, "per-mesh live readback drift"
                        ):
                            LAUNCHER._validate_v7_capture(request)
                        readback[field] = original

    def test_capture_python_symlink_resolves_to_only_pinned_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "python3.11"
            real.write_text("runtime", encoding="utf-8")
            logical = root / "python"
            logical.symlink_to(real.name)
            wrong = root / "other-python"
            wrong.write_text("wrong", encoding="utf-8")
            with mock.patch.object(LAUNCHER, "CAPTURE_PYTHON_LOGICAL", logical):
                self.assertTrue(LAUNCHER._is_authoritative_capture_python(logical))
                self.assertTrue(LAUNCHER._is_authoritative_capture_python(real))
                self.assertFalse(LAUNCHER._is_authoritative_capture_python(wrong))

    def test_prepare_failure_archive_preserves_request_without_consuming_attempt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            atom = repo / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
            attempt = atom / "diagnostic_f15_launch_attempt_01"
            _write(attempt / "request.json", {"required_repo_commit": "a" * 40})
            with mock.patch.object(LAUNCHER, "REPOSITORY", repo):
                receipt_path = LAUNCHER.archive_preparation_failure(
                    atom_root=atom,
                    error="canonical interpreter symlink mismatch",
                )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertFalse(receipt["gpu_query_started"])
            self.assertFalse(receipt["gpu_started"])
            self.assertFalse(receipt["attempt_consumed"])
            self.assertTrue(receipt_path.with_name("request.json").is_file())
            self.assertFalse(attempt.exists())

    def test_dry_run_writes_only_non_consuming_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "diagnostic_f15_launch_attempt_01"
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = _request(attempt)
            argv = ["python", "capture.py", "--frame-index", "15"]
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request", return_value=(request, argv)
                ),
                mock.patch.object(
                    LAUNCHER, "_gpu_snapshot", return_value=_idle_snapshot()
                ),
                mock.patch.object(LAUNCHER, "_assert_port_available"),
            ):
                self.assertEqual(LAUNCHER.run(request_path, dry_run=True), 0)
            receipt = json.loads(
                (attempt / "dry_run_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "dry_run_pass_not_launched")
            self.assertEqual(receipt["frame_indices"], [15])
            self.assertFalse(receipt["full75_allowed"])
            self.assertFalse((attempt / "running_receipt.json").exists())
            self.assertFalse((attempt / "final_receipt.json").exists())

    def test_real_attempt_has_immutable_running_and_separate_final_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "diagnostic_f15_launch_attempt_01"
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = _request(attempt)
            argv = ["python", "capture.py", "--frame-index", "15"]
            validation = {
                "status": "pass_diagnostic_f15_review_ready",
                "qualification_claim": False,
                "formal_dataset_count": 0,
            }
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request", return_value=(request, argv)
                ),
                mock.patch.object(
                    LAUNCHER,
                    "_gpu_snapshot",
                    side_effect=[_idle_snapshot(), _idle_snapshot()],
                ),
                mock.patch.object(LAUNCHER, "_assert_port_available"),
                mock.patch.object(
                    LAUNCHER,
                    "subprocess",
                    wraps=LAUNCHER.subprocess,
                ) as subprocess_module,
                mock.patch.object(
                    LAUNCHER, "_validate_capture", return_value=validation
                ),
            ):
                subprocess_module.run.return_value = SimpleNamespace(returncode=0)
                self.assertEqual(LAUNCHER.run(request_path, dry_run=False), 0)
            running = json.loads(
                (attempt / "running_receipt.json").read_text(encoding="utf-8")
            )
            final = json.loads(
                (attempt / "final_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(running["status"], "running")
            self.assertIsNone(running["capture_process_exit_code"])
            self.assertEqual(final["status"], "pass_diagnostic_f15_review_ready")
            self.assertEqual(final["capture_process_exit_code"], 0)
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request", return_value=(request, argv)
                ),
                self.assertRaisesRegex(RuntimeError, "final receipt"),
            ):
                LAUNCHER.run(request_path, dry_run=False)

    def test_attempt01_failure_ledger_freezes_only_observed_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            atom = repo / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
            attempt = atom / "diagnostic_f15_launch_attempt_01"
            capture = atom / "diagnostic_f15_capture_attempt_01"
            capture.mkdir(parents=True)
            _write(
                attempt / "request.json",
                {
                    "schema": LAUNCHER.REQUEST_SCHEMA,
                    "capture_output": str(capture),
                },
            )
            _write(
                attempt / "dry_run_receipt.json",
                {
                    "schema": LAUNCHER.RECEIPT_SCHEMA,
                    "status": "dry_run_pass_not_launched",
                },
            )
            _write(
                attempt / "running_receipt.json",
                {"schema": LAUNCHER.RECEIPT_SCHEMA, "status": "running"},
            )
            _write(
                attempt / "final_receipt.json",
                {
                    "schema": LAUNCHER.RECEIPT_SCHEMA,
                    "status": "failed",
                    "capture_process_exit_code": 1,
                },
            )
            spear_log = repo / "SpearSim_rpc_39631.log"
            spear_log.write_text(
                "LogInit: Display: Game Engine Initialized.\n"
                "LogGlobalStatus: LoadMap Load map complete /Engine/Maps/Entry\n"
                "LogInit: Display: Engine is initialized. "
                "Leaving FEngineLoop::Init()\n",
                encoding="utf-8",
            )
            with mock.patch.object(LAUNCHER, "REPOSITORY", repo):
                ledger_path = LAUNCHER.record_attempt01_failure_ledger(
                    atom_root=atom,
                    spear_log=spear_log,
                )
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(ledger["status"], LAUNCHER.ATTEMPT01_FAILURE_STATUS)
            self.assertEqual(ledger["root_cause"], "undetermined")
            self.assertTrue(ledger["attempt_consumed"])
            self.assertTrue(ledger["retry_same_candidate_forbidden"])
            self.assertEqual(ledger["captured_frame_count"], 0)
            self.assertEqual(ledger["capture_artifact_count"], 0)
            self.assertEqual(ledger["first_capture_artifact_count"], 0)
            self.assertFalse(ledger["causal_exclusions"]["mesh_failure_claimed"])
            with (
                mock.patch.object(LAUNCHER, "REPOSITORY", repo),
                self.assertRaises(FileExistsError),
            ):
                LAUNCHER.record_attempt01_failure_ledger(
                    atom_root=atom,
                    spear_log=spear_log,
                )

    def test_capture_failure_journal_covers_all_runtime_phases(self) -> None:
        phases = (
            "preconnect",
            "post-entry",
            "mesh",
            "lighting",
            "camera",
            "actor",
            "capture",
        )
        for phase in phases:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "capture"
                output.mkdir()

                def fail_at_phase(
                    _args: SimpleNamespace,
                    journal: object,
                    *,
                    selected_phase: str = phase,
                ) -> Path:
                    journal.enter(selected_phase)
                    raise RuntimeError(f"fake runtime failure at {selected_phase}")

                with (
                    mock.patch.object(CAPTURE, "_run_impl", side_effect=fail_at_phase),
                    self.assertRaisesRegex(RuntimeError, phase),
                ):
                    CAPTURE.run(SimpleNamespace(output=output))
                failure = json.loads(
                    (output / "capture_failure.json").read_text(encoding="utf-8")
                )
                self.assertEqual(failure["phase"], phase)
                self.assertEqual(failure["exception_type"], "RuntimeError")
                self.assertIn(
                    f"RuntimeError: fake runtime failure at {phase}",
                    failure["traceback"],
                )
                markers = list(output.glob(f"capture_phase_*_{phase}.json"))
                self.assertEqual(len(markers), 1)

    def test_capture_runtime_wires_required_phases_in_order(self) -> None:
        source = inspect.getsource(CAPTURE._run_impl)
        phases = (
            "preconnect",
            "post-entry",
            "mesh",
            "lighting",
            "camera",
            "actor",
            "capture",
        )
        offsets = [source.index(f'journal.enter("{phase}")') for phase in phases]
        self.assertEqual(offsets, sorted(offsets))

    def test_revision_v2_dry_run_writes_no_child_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / LAUNCHER.V2_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = _request_v2(attempt)
            argv = ["python", "capture.py", "--frame-index", "15"]
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v2", return_value=(request, argv)
                ),
                mock.patch.object(
                    LAUNCHER, "_gpu_snapshot", return_value=_idle_snapshot()
                ),
                mock.patch.object(LAUNCHER, "_assert_port_available"),
            ):
                self.assertEqual(
                    LAUNCHER.run_v2(
                        request_path,
                        dry_run=True,
                        authorize_gpu_capture=False,
                    ),
                    0,
                )
            receipt = json.loads(
                (attempt / "dry_run_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "dry_run_pass_not_launched")
            self.assertFalse(receipt["gpu_started"])
            self.assertFalse(receipt["attempt_consumed"])
            self.assertFalse((attempt / "capture_stdout.log").exists())
            self.assertFalse((attempt / "capture_stderr.log").exists())

    def test_revision_v2_real_launch_requires_explicit_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / LAUNCHER.V2_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = _request_v2(attempt)
            argv = ["python", "capture.py", "--frame-index", "15"]
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v2", return_value=(request, argv)
                ),
                mock.patch.object(LAUNCHER, "_gpu_snapshot") as snapshot,
                self.assertRaisesRegex(RuntimeError, "explicit launch authorization"),
            ):
                LAUNCHER.run_v2(
                    request_path,
                    dry_run=False,
                    authorize_gpu_capture=False,
                )
            snapshot.assert_not_called()
            self.assertFalse((attempt / "running_receipt.json").exists())

    def test_revision_v2_rejects_preexisting_child_log_before_gpu_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / LAUNCHER.V2_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = _request_v2(attempt)
            Path(str(request["capture_stdout"])).write_text(
                "do not replace\n", encoding="utf-8"
            )
            argv = ["python", "capture.py", "--frame-index", "15"]
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v2", return_value=(request, argv)
                ),
                mock.patch.object(LAUNCHER, "_gpu_snapshot") as snapshot,
                self.assertRaisesRegex(RuntimeError, "already exists"),
            ):
                LAUNCHER.run_v2(
                    request_path,
                    dry_run=False,
                    authorize_gpu_capture=True,
                )
            snapshot.assert_not_called()
            self.assertEqual(
                Path(str(request["capture_stdout"])).read_text(encoding="utf-8"),
                "do not replace\n",
            )
            self.assertFalse((attempt / "running_receipt.json").exists())

    def test_revision_v2_failure_persists_observability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / LAUNCHER.V2_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            request_path.write_text("{}\n", encoding="utf-8")
            request = _request_v2(attempt)
            capture_output = Path(str(request["capture_output"]))
            argv = ["python", "capture.py", "--frame-index", "15"]

            def fake_child(*_args: object, **kwargs: object) -> SimpleNamespace:
                stdout = kwargs["stdout"]
                stderr = kwargs["stderr"]
                stdout.write(b"exclusive child stdout\n")
                stderr.write(b"Traceback: exclusive child stderr\n")
                capture_output.mkdir()
                _write(
                    capture_output / "capture_phase_00_mesh.json",
                    {
                        "schema": "avengine_mp3d_f15_capture_phase_v1",
                        "status": "entered",
                        "phase": "mesh",
                        "sequence": 0,
                        "qualification_claim": False,
                        "formal_dataset_count": 0,
                    },
                )
                _write(
                    capture_output / "capture_failure.json",
                    {
                        "schema": LAUNCHER.CAPTURE_FAILURE_SCHEMA,
                        "status": "failed",
                        "phase": "mesh",
                        "exception_type": "RuntimeError",
                        "exception_message": "fake mesh failure",
                        "traceback": (
                            "Traceback (most recent call last):\n"
                            "RuntimeError: fake mesh failure\n"
                        ),
                        "qualification_claim": False,
                        "formal_dataset_count": 0,
                    },
                )
                return SimpleNamespace(returncode=23)

            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v2", return_value=(request, argv)
                ),
                mock.patch.object(
                    LAUNCHER,
                    "_gpu_snapshot",
                    side_effect=[_idle_snapshot(), _idle_snapshot()],
                ),
                mock.patch.object(LAUNCHER, "_assert_port_available"),
                mock.patch.object(LAUNCHER.subprocess, "run", side_effect=fake_child),
            ):
                self.assertEqual(
                    LAUNCHER.run_v2(
                        request_path,
                        dry_run=False,
                        authorize_gpu_capture=True,
                    ),
                    23,
                )
            final = json.loads(
                (attempt / "final_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(final["status"], "failed")
            self.assertEqual(final["child_exit_code"], 23)
            self.assertEqual(final["capture_process_exit_code"], 23)
            self.assertEqual(final["child_exit"], {"observed": True, "returncode": 23})
            self.assertEqual(
                final["failure_observability_status"],
                "phase_and_complete_traceback_persisted",
            )
            self.assertEqual(
                final["capture_observability"]["capture_failure_detail"]["phase"],
                "mesh",
            )
            self.assertIn(
                "RuntimeError: fake mesh failure",
                final["capture_observability"]["capture_failure_detail"]["traceback"],
            )
            self.assertEqual(
                (attempt / "capture_stdout.log").read_text(encoding="utf-8"),
                "exclusive child stdout\n",
            )
            self.assertEqual(
                (attempt / "capture_stderr.log").read_text(encoding="utf-8"),
                "Traceback: exclusive child stderr\n",
            )
            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v2", return_value=(request, argv)
                ),
                self.assertRaisesRegex(RuntimeError, "final receipt"),
            ):
                LAUNCHER.run_v2(
                    request_path,
                    dry_run=False,
                    authorize_gpu_capture=True,
                )

    def test_v8_real_semantic_evidence_and_review_mutations(self) -> None:
        plan = (
            LAUNCHER.REPOSITORY
            / "tmp/lead_a_strict_two_human_mp3d_room_v4"
            / "cpu_preflight_v1/execution_plan.json"
        )
        if not plan.is_file():
            self.skipTest("fresh v4 semantic CPU evidence is unavailable")
        paths = LAUNCHER._v8_execution_plan_paths(plan)
        evidence = LAUNCHER._validate_v8_execution_plan_evidence(paths)
        self.assertEqual(
            evidence["episode_id"],
            "mp3d_17DRP5sb8fy_male_female_static_rig_0003",
        )
        self.assertEqual(evidence["scene_id"], "17DRP5sb8fy")

        def mutated(name: str, mutate: object, message: str) -> None:
            with tempfile.TemporaryDirectory() as directory:
                source = paths[name]
                changed = json.loads(source.read_text(encoding="utf-8"))
                assert callable(mutate)
                mutate(changed)
                replacement = Path(directory) / source.name
                _write(replacement, changed)
                replacement_paths = {**paths, name: replacement}
                with self.assertRaisesRegex(RuntimeError, message):
                    LAUNCHER._validate_v8_execution_plan_evidence(replacement_paths)

        mutated(
            "execution_plan",
            lambda value: next(
                step
                for step in value["cpu_steps"]
                if step["step_id"] == "render_two_exact_rirs"
            )["argv"].extend(["--job-limit", "1"]),
            "semantic RIR execution argv drift",
        )

        def replace_all75_listener_rotations(value: object) -> None:
            assert isinstance(value, dict)
            frames = value["scenarios"][0]["plan"]["frames"]
            self.assertEqual(len(frames), 75)
            for frame in frames:
                frame["camera_state"]["world_from_rig"]["rotation_xyzw"] = [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]

        mutated(
            "suite_plan",
            replace_all75_listener_rotations,
            "listener orientation",
        )

        def replace_both_los_anchor_receipts(value: object) -> None:
            assert isinstance(value, dict)
            selected = value["results"][0]
            for hard_gates in (
                selected["evidence"]["hard_gates"],
                selected["room_gate"]["hard_gates"],
            ):
                hard_gates["line_of_sight_source1_actor"]["anchor_ids"] = [
                    "declared_emitter_proxy"
                ]

        mutated(
            "runtime_camera_gates",
            replace_both_los_anchor_receipts,
            "listener runtime/nav/LOS closure drift",
        )
        mutated(
            "camera_framing",
            lambda value: value["candidate_evaluations"][0]["frame_evaluations"][0][
                "actors"
            ][0]["projection"]["projected_bbox_px"].update({"left": 0.0}),
            "ordinary CPU projection",
        )
        mutated(
            "package_manifest",
            lambda value: value["source_room"].update({"room_id": "foreign"}),
            "selected room identity differ",
        )

    def test_v8_offline_retargets_fresh_output_without_gpu_or_write(self) -> None:
        plan = (
            LAUNCHER.REPOSITORY
            / "tmp/lead_a_strict_two_human_mp3d_room_v4"
            / "cpu_preflight_v1/execution_plan.json"
        )
        if not plan.is_file():
            self.skipTest("fresh v4 semantic CPU evidence is unavailable")
        capture = plan.parent.parent / LAUNCHER.V8_CAPTURE_DIRECTORY
        if capture.exists() or capture.is_symlink():
            self.skipTest("v8 capture path has been consumed")
        with mock.patch.object(
            LAUNCHER, "_gpu_snapshot", side_effect=AssertionError("GPU queried")
        ):
            result = LAUNCHER.offline_validate_execution_plan_v8(plan)
        self.assertEqual(result["status"], "pass_offline_no_write_no_gpu_query")
        self.assertEqual(result["rpc_port"], 39638)
        self.assertEqual(
            result["capture_output"],
            str(plan.parent.parent / LAUNCHER.V8_CAPTURE_DIRECTORY),
        )
        self.assertFalse(result["gpu_started"])
        self.assertFalse(result["writes_performed"])

    def test_v8_prepare_and_dry_run_share_fresh_lifecycle_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            atom = repo / "tmp/fresh_v8_atom"
            plan = atom / "cpu_preflight_v1/execution_plan.json"
            _write(plan, {"schema": "fixture_only"})
            suite = plan.parent / "suite.json"
            room = plan.parent / "room.json"
            _write(suite, {})
            _write(room, {})
            capture = atom / LAUNCHER.V8_CAPTURE_DIRECTORY
            argv = [
                "python",
                "capture.py",
                "--rpc-port",
                str(LAUNCHER.V8_RPC_PORT),
                "--output",
                str(capture),
            ]
            validation = {
                "execution_plan": str(plan),
                "episode_id": "episode_v8",
                "scene_id": "scene_v8",
                "evidence_paths": {
                    "suite_plan": str(suite),
                    "room_adapter": str(room),
                },
                "capture_argv": argv,
            }
            with (
                mock.patch.object(LAUNCHER, "REPOSITORY", repo),
                mock.patch.object(
                    LAUNCHER,
                    "offline_validate_execution_plan_v8",
                    return_value=validation,
                ) as offline,
                mock.patch.object(
                    LAUNCHER, "_git_tracked_and_index_clean", return_value=True
                ),
                mock.patch.object(LAUNCHER, "_git_head", return_value="v8-commit"),
                mock.patch.object(LAUNCHER, "_assert_port_available") as port,
            ):
                request_path = LAUNCHER.prepare_request_v8(execution_plan_path=plan)
            offline.assert_called_once_with(plan)
            port.assert_called_once_with(LAUNCHER.V8_RPC_PORT)
            request = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(request["schema"], LAUNCHER.REQUEST_SCHEMA_V8)
            self.assertEqual(request["required_repo_commit"], "v8-commit")
            self.assertEqual(
                Path(request["attempt_root"]).name,
                LAUNCHER.V8_ATTEMPT_DIRECTORY,
            )
            self.assertEqual(request["capture_output"], str(capture))
            self.assertFalse(capture.exists())

            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v8", return_value=(request, argv)
                ),
                mock.patch.object(
                    LAUNCHER, "_gpu_snapshot", side_effect=AssertionError("GPU queried")
                ) as snapshot,
            ):
                self.assertEqual(
                    LAUNCHER.run_v8(
                        request_path,
                        offline_validate=False,
                        dry_run=True,
                        authorize_gpu_capture=False,
                    ),
                    0,
                )
            snapshot.assert_not_called()
            receipt = json.loads(
                (request_path.parent / "dry_run_receipt.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(receipt["schema"], LAUNCHER.RECEIPT_SCHEMA_V8)
            self.assertEqual(receipt["status"], "dry_run_pass_not_launched")
            self.assertFalse(receipt["gpu_query_started"])
            self.assertFalse(receipt["gpu_started"])
            self.assertFalse(receipt["attempt_consumed"])
            self.assertFalse((request_path.parent / "running_receipt.json").exists())
            self.assertFalse((request_path.parent / "final_receipt.json").exists())

            with (
                mock.patch.object(
                    LAUNCHER, "_validate_request_v8", return_value=(request, argv)
                ),
                mock.patch.object(LAUNCHER, "_gpu_snapshot") as snapshot,
                self.assertRaisesRegex(RuntimeError, "explicit launch authorization"),
            ):
                LAUNCHER.run_v8(
                    request_path,
                    offline_validate=False,
                    dry_run=False,
                    authorize_gpu_capture=False,
                )
            snapshot.assert_not_called()
            self.assertFalse((request_path.parent / "running_receipt.json").exists())

    def test_v8_cli_subcommands_are_wired_to_v8_only(self) -> None:
        plan = Path("plan.json")
        request = Path("request.json")
        prepared = LAUNCHER.parse_args(["prepare-v8", "--execution-plan", str(plan)])
        launched = LAUNCHER.parse_args(
            ["launch-v8", "--request", str(request), "--dry-run"]
        )
        offline = LAUNCHER.parse_args(
            ["offline-validate-v8", "--execution-plan", str(plan)]
        )
        validation_only = LAUNCHER.parse_args(
            [
                "validate-v8-capture",
                "--request",
                str(request),
                "--output-receipt",
                "validation.json",
            ]
        )
        self.assertEqual(
            (prepared.command, prepared.execution_plan), ("prepare-v8", plan)
        )
        self.assertEqual((launched.command, launched.request), ("launch-v8", request))
        self.assertTrue(launched.dry_run)
        self.assertEqual(
            (offline.command, offline.execution_plan),
            ("offline-validate-v8", plan),
        )
        self.assertEqual(validation_only.command, "validate-v8-capture")
        self.assertEqual(validation_only.request, request)
        self.assertEqual(validation_only.output_receipt, Path("validation.json"))

    def test_v8_validation_only_publishes_separate_receipt_without_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            attempt = root / LAUNCHER.V8_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            original_final = attempt / "final_receipt.json"
            original_final.write_bytes(b"original final bytes\n")
            original_running = attempt / "running_receipt.json"
            original_running.write_bytes(b"original running bytes\n")
            capture = root / LAUNCHER.V8_CAPTURE_DIRECTORY
            capture.mkdir()
            request = {
                "episode_id": "episode",
                "scene_id": "scene",
            }
            context = {
                "request": request,
                "request_path": request_path,
                "attempt_root": attempt,
                "capture_root": capture,
                "running_receipt": original_running,
                "original_final_receipt": original_final,
                "capture_required_repo_commit": "capture-commit",
                "validator_repo_commit": "validator-commit",
            }
            output = attempt / "validation_only_receipt_v1.json"
            validation = {
                "status": "pass_diagnostic_f15_review_ready",
                "visibility": {
                    "status": "pass",
                    "pixel_visibility_truth_status": "compiled_in_memory",
                },
            }
            with (
                mock.patch.object(
                    LAUNCHER,
                    "_validate_consumed_request_v8_for_revalidation",
                    return_value=context,
                ),
                mock.patch.object(
                    LAUNCHER, "_validate_v7_capture", return_value=validation
                ) as validate,
                mock.patch.object(
                    LAUNCHER, "_gpu_snapshot", side_effect=AssertionError("GPU queried")
                ) as gpu,
                mock.patch.object(
                    LAUNCHER.subprocess,
                    "run",
                    side_effect=AssertionError("capture subprocess started"),
                ) as child,
            ):
                self.assertEqual(
                    LAUNCHER.validate_v8_capture_only(
                        request_path, output_receipt=output
                    ),
                    output,
                )
            validate.assert_called_once_with(request, publish_visibility_truth=False)
            gpu.assert_not_called()
            child.assert_not_called()
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["schema"], LAUNCHER.VALIDATION_ONLY_RECEIPT_SCHEMA_V8
            )
            self.assertEqual(receipt["status"], "pass_diagnostic_f15_review_ready")
            self.assertEqual(receipt["capture_required_repo_commit"], "capture-commit")
            self.assertEqual(receipt["validator_repo_commit"], "validator-commit")
            self.assertFalse(receipt["gpu_query_started"])
            self.assertFalse(receipt["capture_subprocess_started"])
            self.assertFalse(receipt["full75_allowed"])
            self.assertEqual(receipt["formal_dataset_count"], 0)
            self.assertEqual(original_final.read_bytes(), b"original final bytes\n")
            self.assertEqual(original_running.read_bytes(), b"original running bytes\n")
            self.assertEqual(list(capture.iterdir()), [])
            with mock.patch.object(
                LAUNCHER,
                "_validate_consumed_request_v8_for_revalidation",
                return_value=context,
            ):
                with self.assertRaisesRegex(RuntimeError, "fresh attempt-root child"):
                    LAUNCHER.validate_v8_capture_only(
                        request_path, output_receipt=output
                    )

    def test_v8_validation_only_failure_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            attempt = root / LAUNCHER.V8_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            final = attempt / "final_receipt.json"
            final.write_bytes(b"immutable final\n")
            running = attempt / "running_receipt.json"
            running.write_bytes(b"immutable running\n")
            capture = root / LAUNCHER.V8_CAPTURE_DIRECTORY
            capture.mkdir()
            output = attempt / "validation_only_receipt_v1.json"
            context = {
                "request": {"episode_id": "episode", "scene_id": "scene"},
                "request_path": request_path,
                "attempt_root": attempt,
                "capture_root": capture,
                "running_receipt": running,
                "original_final_receipt": final,
                "capture_required_repo_commit": "capture-commit",
                "validator_repo_commit": "validator-commit",
            }
            with (
                mock.patch.object(
                    LAUNCHER,
                    "_validate_consumed_request_v8_for_revalidation",
                    return_value=context,
                ),
                mock.patch.object(
                    LAUNCHER,
                    "_validate_v7_capture",
                    side_effect=RuntimeError("bad capture"),
                ),
                mock.patch.object(LAUNCHER, "_gpu_snapshot") as gpu,
                mock.patch.object(LAUNCHER.subprocess, "run") as child,
                self.assertRaisesRegex(RuntimeError, "without publishing"),
            ):
                LAUNCHER.validate_v8_capture_only(request_path, output_receipt=output)
            gpu.assert_not_called()
            child.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(final.read_bytes(), b"immutable final\n")
            self.assertEqual(running.read_bytes(), b"immutable running\n")
            self.assertEqual(list(capture.iterdir()), [])

    def test_v8_consumed_admission_rejects_symlink_capture_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            atom = repo / "tmp/atom"
            attempt = atom / LAUNCHER.V8_ATTEMPT_DIRECTORY
            capture = atom / LAUNCHER.V8_CAPTURE_DIRECTORY
            attempt.mkdir(parents=True)
            capture.mkdir()
            request_path = attempt / "request.json"
            request = {
                "schema": LAUNCHER.REQUEST_SCHEMA_V8,
                "status": "prepared_not_launched",
                "frame_indices": [LAUNCHER.FRAME_INDEX],
                "full75_allowed": False,
                "physical_gpu_index": 1,
                "physical_gpu_uuid": LAUNCHER.GPU1_UUID,
                "graphics_adapter_argument": 1,
                "required_idle_compute_process_count": 0,
                "explicit_gpu_capture_authorization_required": True,
                "gpu_capture_authorized_at_prepare": False,
                "manual_review_required": True,
                "qualification_claim": False,
                "formal_dataset_count": 0,
                "attempt_policy": {
                    **LAUNCHER.ATTEMPT_POLICY,
                    "candidate_revision": "fresh_schema_v2_cpu_semantic_sparse_f15_v8",
                },
                "repo_root": str(repo),
                "required_repo_commit": "capture-commit",
                "atom_root": str(atom),
                "attempt_root": str(attempt),
                "execution_plan": str(atom / "cpu_preflight_v1/execution_plan.json"),
                "capture_output": str(capture),
                "episode_id": "episode",
                "scene_id": "scene",
                "evidence_paths": {"suite_plan": "suite", "room_adapter": "room"},
                "suite_plan": "suite",
                "room_adapter": "room",
                "capture_argv": ["capture"],
                "rpc_port": LAUNCHER.V8_RPC_PORT,
            }
            _write(request_path, request)
            outside = atom / "outside.json"
            _write(outside, {})
            (capture / "manifest.json").symlink_to(outside)
            projection = {
                "episode_id": "episode",
                "scene_id": "scene",
                "execution_plan": request["execution_plan"],
                "evidence_paths": request["evidence_paths"],
                "capture_output": str(capture),
                "capture_argv": ["capture"],
            }
            with (
                mock.patch.object(LAUNCHER, "REPOSITORY", repo),
                mock.patch.object(
                    LAUNCHER, "_git_tracked_and_index_clean", return_value=True
                ),
                mock.patch.object(LAUNCHER, "_git_head", return_value="validator"),
                mock.patch.object(
                    LAUNCHER,
                    "_validate_execution_plan_v8_projection",
                    return_value=projection,
                ),
                self.assertRaisesRegex(RuntimeError, "symlink component"),
            ):
                LAUNCHER._validate_consumed_request_v8_for_revalidation(request_path)

    def test_v8_validation_only_rejects_symlink_output_before_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            attempt = root / LAUNCHER.V8_ATTEMPT_DIRECTORY
            attempt.mkdir()
            request_path = attempt / "request.json"
            capture = root / LAUNCHER.V8_CAPTURE_DIRECTORY
            capture.mkdir()
            context = {
                "request": {"episode_id": "episode", "scene_id": "scene"},
                "request_path": request_path,
                "attempt_root": attempt,
                "capture_root": capture,
                "running_receipt": attempt / "running_receipt.json",
                "original_final_receipt": attempt / "final_receipt.json",
                "capture_required_repo_commit": "capture-commit",
                "validator_repo_commit": "validator-commit",
            }
            output = attempt / "validation_only_receipt_v1.json"
            output.symlink_to(attempt / "missing")
            with (
                mock.patch.object(
                    LAUNCHER,
                    "_validate_consumed_request_v8_for_revalidation",
                    return_value=context,
                ),
                mock.patch.object(LAUNCHER, "_validate_v7_capture") as validate,
                self.assertRaisesRegex(RuntimeError, "symlink component"),
            ):
                LAUNCHER.validate_v8_capture_only(request_path, output_receipt=output)
            validate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
