from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

LAUNCHER_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/qa/run_strict_two_human_skokloster_f15_probe.py"
)


def _load_launcher() -> ModuleType:
    name = "avengine_test_skokloster_f15_launcher"
    spec = importlib.util.spec_from_file_location(name, LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LAUNCHER = _load_launcher()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_stereo_pcm16(path: Path, *, active: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [0] * (80_000 * 2)
    if active:
        samples[2 * 7467] = 1234
        samples[2 * 7467 + 1] = -4321
    payload = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(payload)


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


class Layout:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.repo = self.root / "repo"
        self.atom = self.repo / LAUNCHER.ATOM_DIRECTORY
        self.archive = self.root / "Standalone-Skokloster-Development"
        self.executable = self.archive / "Linux/SpearSim.sh"
        self.binary = self.archive / "Linux/SpearSim/Binaries/Linux/SpearSim"
        self.pak = self.archive / "Linux/SpearSim/Content/Paks/SpearSim-Linux.pak"
        self.capture_python = self.root / "runtime/python"
        self.spear_root = self.root / "SPEAR"
        self.upstream = (
            self.repo
            / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
            / "diagnostic_f15_revision_v2_launch_attempt_01/final_receipt.json"
        )

    @contextlib.contextmanager
    def patched(self):
        with (
            mock.patch.object(LAUNCHER, "REPOSITORY", self.repo),
            mock.patch.object(LAUNCHER, "ARCHIVE_ROOT", self.archive),
            mock.patch.object(LAUNCHER, "PACKAGED_EXECUTABLE", self.executable),
            mock.patch.object(LAUNCHER, "PACKAGED_BINARY", self.binary),
            mock.patch.object(LAUNCHER, "PACKAGED_PAK", self.pak),
            mock.patch.object(LAUNCHER, "CAPTURE_PYTHON_LOGICAL", self.capture_python),
            mock.patch.object(LAUNCHER, "SPEAR_ROOT", self.spear_root),
            mock.patch.object(LAUNCHER, "MP3D_V2_TERMINAL_RECEIPT", self.upstream),
            mock.patch.object(LAUNCHER, "_git_head", return_value="a" * 40),
            mock.patch.object(LAUNCHER, "_git_status_porcelain", return_value=""),
        ):
            yield self

    def materialize(self) -> None:
        for path, payload in (
            (self.executable, b"#!/bin/sh\n"),
            (self.binary, b"fake packaged executable"),
            (self.pak, b"fake packaged pak with exact map"),
            (self.capture_python, b"fake python"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        self.spear_root.mkdir(parents=True)
        for relative in (
            "tools/qa/capture_skokloster_strict_two_human_episode.py",
            "tools/qa/capture_spear_native_pixel_episode.py",
        ):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# bound source\n", encoding="utf-8")

        preflight = self.atom / "cpu_preflight_v4"
        rir = self.atom / "exact_rir_cache_v3"
        binaural = self.atom / "binaural_v4"
        frames = [
            {
                "frame_index": index,
                "camera_state": {
                    "habitat_position_m": LAUNCHER.CAMERA_LISTENER_M,
                },
            }
            for index in range(75)
        ]
        suite = {
            "schema": "avengine_optional_spear_skokloster_suite_v1",
            "native_map": LAUNCHER.PACKAGED_MAP,
            "packaged_executable": str(self.executable),
            "scenarios": [
                {
                    "scenario_id": LAUNCHER.EPISODE_ID,
                    "native_scene": {"map": LAUNCHER.PACKAGED_MAP},
                    "render": {
                        "frame_count": 75,
                        "frame_rate_hz": 15,
                        "width": 1280,
                        "height": 720,
                    },
                    "plan": {
                        "actors": [
                            {"actor_id": "source1_actor"},
                            {"actor_id": "source2_actor"},
                        ],
                        "frames": frames,
                        "camera": {
                            "habitat_position_m": LAUNCHER.CAMERA_LISTENER_M,
                            "habitat_yaw_deg": LAUNCHER.CAMERA_YAW_DEG,
                        },
                    },
                }
            ],
            "qualification_claim": False,
            "formal_dataset_count": 0,
        }
        _write_json(
            preflight / "preflight.json",
            {
                "schema": "avengine_skokloster_strict_two_human_cpu_preflight_v1",
                "status": "cpu_plan_pass_gpu_sparse_pending",
                "attempt_id": "cpu_preflight_v4",
                "episode_id": LAUNCHER.EPISODE_ID,
                "camera_listener_habitat_m": LAUNCHER.CAMERA_LISTENER_M,
                "camera_habitat_yaw_deg": LAUNCHER.CAMERA_YAW_DEG,
                "gpu_capture_authorized": False,
                "qualification_claim": False,
                "formal_dataset_count": 0,
            },
        )
        _write_json(preflight / "suite_execution_plan.json", suite)
        _write_json(
            preflight / "execution_plan.json", {"status": "cpu_ready_gpu_blocked"}
        )
        _write_json(
            preflight / "rir_job_plan.json",
            {
                "schema": "avengine_room_rir_job_plan_v2",
                "unique_rir_job_count": 2,
                "requested_pair_state_count": 150,
                "cache_reuse_count": 148,
                "listener_position_m": LAUNCHER.CAMERA_LISTENER_M,
                "jobs": [
                    {
                        "job_id": "source1",
                        "uses": [{"frame_index": index} for index in range(75)],
                    },
                    {
                        "job_id": "source2",
                        "uses": [{"frame_index": index} for index in range(75)],
                    },
                ],
                "qualification_claim": False,
                "formal_dataset_count": 0,
            },
        )
        _write_json(preflight / "sensor_rig_trajectory.json", {"status": "pass"})
        _write_json(
            preflight / "audio_program_binding.json",
            {
                "schema": "avengine_skokloster_strict_audio_program_binding_v1",
                "source1": {
                    "role": "target",
                    "start_sample": 7467,
                    "end_sample_exclusive": 33093,
                    "source_start_sample": 0,
                    "source_end_sample_exclusive": 25626,
                    "linear_gain": 0.18,
                    "fade_samples": 80,
                },
                "source2": {"role": "distractor", "event_count": 0},
                "qualification_claim": False,
                "formal_dataset_count": 0,
            },
        )
        identity = "1" * 64
        acoustic = "2" * 64
        _write_json(
            rir / "receipt.json",
            {
                "schema": "avengine_rlr_rir_cache_receipt_v1",
                "status": "pass",
                "compute_device": "CPU",
                "layout_type": "binaural",
                "sample_rate_hz": 16000,
                "selected_job_count": 2,
                "full_plan_job_count": 2,
                "full_plan_complete": True,
                "retained_payload_hash_verified": True,
                "request_identity_sha256": identity,
                "acoustic_selection_binding_sha256": acoustic,
                "qualification_claim": False,
            },
        )
        _write_json(
            rir / "index.json",
            {
                "schema": "avengine_rlr_rir_cache_index_v1",
                "status": "pass",
                "selected_job_count": 2,
                "full_plan_complete": True,
                "request_identity_sha256": identity,
                "acoustic_selection_binding_sha256": acoustic,
                "entries": [{"job_id": "source1"}, {"job_id": "source2"}],
            },
        )
        shard = rir / "shards/shard_000000.npz"
        shard.parent.mkdir(parents=True)
        shard.write_bytes(b"bound two exact RIRs")
        _write_json(
            binaural / "delivery.json",
            {
                "status": "pass",
                "episode_count": 1,
                "sample_count": 1,
                "both_sources_active": False,
                "source_activity_contract": "m6_audio_program_event_windows_v1",
                "sensor_rig_rir_alignment": {"checked_use_count": 150},
                "qualification_claim": False,
            },
        )
        _write_json(binaural / "episodes.json", {"status": "pass"})
        _write_json(binaural / "timing.json", {"status": "pass"})
        _write_json(
            binaural / "samples.json",
            {
                "status": "pass",
                "samples": [
                    {
                        "sample_id": f"{LAUNCHER.EPISODE_ID}__v00",
                        "episode_id": LAUNCHER.EPISODE_ID,
                        "both_sources_active": False,
                        "source_activity_summary": {
                            "active_source_slots": ["source1"],
                            "silent_source_slots": ["source2"],
                            "active_sample_count_by_source_slot": {
                                "source1": 25626,
                                "source2": 0,
                            },
                        },
                        "audio": {
                            "channel_count": 2,
                            "sample_rate_hz": 16000,
                            "sample_count": 80000,
                            "stems": {
                                "source1": {"peak_absolute": 0.25},
                                "source2": {"peak_absolute": 0.0},
                            },
                        },
                    }
                ],
            },
        )
        audio_root = binaural / "audio/binaural"
        sample_id = f"{LAUNCHER.EPISODE_ID}__v00.wav"
        _write_stereo_pcm16(audio_root / sample_id, active=True)
        _write_stereo_pcm16(audio_root / "stems/source1" / sample_id, active=True)
        _write_stereo_pcm16(audio_root / "stems/source2" / sample_id, active=False)

        _write_json(
            self.repo
            / "examples/acoustics/skokloster_castle/skokloster_room_runtime_profile.json",
            {
                "schema": "avengine_skokloster_imported_room_runtime_profile_v1",
                "scene_id": LAUNCHER.SCENE_ID,
                "visual": {
                    "packaged_runtime_map": LAUNCHER.PACKAGED_MAP,
                    "expected_static_mesh_count": 1,
                    "packaged_readback": {
                        "nullrhi": True,
                        "actor_count": 1,
                        "mesh_handle_match": True,
                    },
                },
                "readiness": {
                    "cook": "pass",
                    "packaged_mesh_readback": "pass",
                    "strict_gpu_capture": "not_run",
                },
                "qualification_claim": False,
                "formal_dataset_count": 0,
            },
        )
        _write_json(
            self.repo / "examples/acoustics/skokloster_castle/editor_import_cook_plan.json",
            {
                "schema": "avengine_skokloster_editor_import_cook_plan_v1",
                "execution_history": {
                    "uat_development_v3": {
                        "status": "pass",
                        "exit_code": 0,
                        "pak_count": 1,
                    },
                    "packaged_object_readback_v2": {
                        "status": "pass",
                        "nullrhi": True,
                        "rendering_or_capture_called": False,
                    },
                },
            },
        )

    def prepare(self) -> Path:
        return LAUNCHER.prepare_request(
            atom_root=self.atom,
            capture_python=self.capture_python,
            spear_root=self.spear_root,
            rpc_port=LAUNCHER.RPC_PORT,
        )

    def write_upstream_terminal(self) -> None:
        _write_json(
            self.upstream,
            {
                "schema": LAUNCHER.MP3D_V2_RECEIPT_SCHEMA,
                "status": "failed",
                "attempt_consumed": True,
                "ended_at_utc": "2026-08-12T01:00:00Z",
            },
        )

    def make_capture(self) -> Path:
        capture = self.atom / LAUNCHER.CAPTURE_DIRECTORY
        capture.mkdir(parents=True)
        rgb = capture / "rgb_frames"
        rgb.mkdir()
        (rgb / "frame_000000.png").write_bytes(b"png")
        files = {
            "native_rgb_visual_only": capture / "native_rgb_visual_only.mp4",
            "native_rgb_binaural": capture / "native_rgb_binaural.mp4",
            "metric_depth": capture / "metric_depth_native.npz",
            "normal_object_ids": capture / "normal_object_ids_uint32.npz",
            "pixel_masks": capture / "native_pixel_masks_depth_authority_v1.npz",
            "pixel_visibility_truth": capture / "pixel_visibility_truth.json",
            "runtime_readbacks": capture / "runtime_readbacks.json",
            "runtime_asset_readbacks": capture / "runtime_asset_readbacks.json",
            "object_id_descriptors": capture / "normal_object_id_descriptors.json",
        }
        for name, path in files.items():
            if path.suffix != ".json":
                path.write_bytes(name.encode("utf-8"))
        truth = {
            "schema": "avengine_qa_pixel_visibility_truth_v1",
            "per_instance": {
                "source1": {
                    "frames": [
                        {
                            "frame_index": 15,
                            "visible_fraction": 0.91,
                            "visible_pixels": 12000,
                            "target_bbox_xyxy_px": [100, 80, 500, 680],
                        }
                    ]
                },
                "source2": {
                    "frames": [
                        {
                            "frame_index": 15,
                            "visible_fraction": 0.72,
                            "visible_pixels": 9000,
                            "target_bbox_xyxy_px": [700, 90, 1100, 675],
                        }
                    ]
                },
            },
        }
        assets = {
            "status": "pass",
            "per_instance": {
                "source1": {"status": "pass"},
                "source2": {"status": "pass"},
            },
        }
        _write_json(files["pixel_visibility_truth"], truth)
        _write_json(files["runtime_readbacks"], {"normal": [{}], "target_only": {}})
        _write_json(files["runtime_asset_readbacks"], assets)
        _write_json(files["object_id_descriptors"], {"descriptors": []})

        def file_record(path: Path) -> dict[str, object]:
            return {
                "kind": "file",
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }

        rgb_member = rgb / "frame_000000.png"
        records: dict[str, object] = {
            "rgb_frames": {
                "kind": "directory",
                "path": str(rgb.resolve()),
                "inventory": [
                    {
                        "relative_path": rgb_member.name,
                        "size_bytes": rgb_member.stat().st_size,
                        "sha256": _sha256(rgb_member),
                    }
                ],
            }
        }
        records.update({name: file_record(path) for name, path in files.items()})
        audio = (
            self.atom
            / "binaural_v4/audio/binaural"
            / (f"{LAUNCHER.EPISODE_ID}__v00.wav")
        )
        _write_json(
            capture / "manifest.json",
            {
                "schema": "avengine_qa_native_spear_pixel_episode_v1",
                "status": "pass",
                "scenario_id": LAUNCHER.EPISODE_ID,
                "native_map": LAUNCHER.PACKAGED_MAP,
                "benchmark_qualification_claim": False,
                "native_pixel_fact_binding_claim": True,
                "frame_contract": {
                    "frame_count": 1,
                    "formal_episode_frame_count": 75,
                    "captured_frame_indices": [15],
                    "frame_rate_hz": 15,
                    "resolution_hw": [720, 1280],
                },
                "runtime_alignment": {
                    "target_pass_count": 2,
                    "maximum_location_drift_cm": 0.0,
                    "maximum_rotation_drift_deg": 0.0,
                },
                "runtime_assets": {
                    "status": "pass",
                    "per_instance_status": {
                        "source1": "pass",
                        "source2": "pass",
                    },
                },
                "audio": {
                    "authoritative_wav": str(audio.resolve()),
                    "sha256": _sha256(audio),
                },
                "artifact_records": records,
            },
        )
        return capture


class SkoklosterF15LauncherTests(unittest.TestCase):
    def test_capture_argv_is_exactly_f15_adapter1_and_wrapper_authorized(self) -> None:
        request = {
            "capture_python": "/runtime/python",
            "capture_script": "/repo/capture.py",
            "suite_plan": "/evidence/suite.json",
            "audio_wav": "/evidence/audio.wav",
            "spear_root": "/runtime/SPEAR",
            "packaged_executable": "/archive/SpearSim.sh",
            "capture_output": "/evidence/capture",
            "rpc_port": 39831,
        }
        argv = LAUNCHER._capture_argv(request)
        self.assertEqual(argv.count("--frame-index"), 1)
        self.assertEqual(argv[argv.index("--frame-index") + 1], "15")
        self.assertEqual(argv[argv.index("--graphics-adapter") + 1], "1")
        self.assertEqual(argv.count("--authorize-gpu-capture"), 1)
        self.assertEqual(
            argv[argv.index("--spear-executable") + 1], "/archive/SpearSim.sh"
        )

    def test_gpu_gate_rejects_uuid_drift_and_busy_gpu1(self) -> None:
        self.assertEqual(
            LAUNCHER._validate_gpu1_idle(_idle_snapshot())["uuid"],
            LAUNCHER.GPU1_UUID,
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

    def test_clean_head_gate_rejects_any_porcelain_entry(self) -> None:
        with (
            mock.patch.object(
                LAUNCHER, "_git_status_porcelain", return_value=" M x.py\n"
            ),
            self.assertRaisesRegex(RuntimeError, "not clean"),
        ):
            LAUNCHER._require_clean_head(Path("/repo"))

    def test_prepare_binds_cpu_archive_sources_and_formal_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = Layout(Path(directory))
            with layout.patched():
                layout.materialize()
                request_path = layout.prepare()
                request = json.loads(request_path.read_text(encoding="utf-8"))
                self.assertEqual(request["frame_indices"], [15])
                self.assertFalse(request["full75_allowed"])
                self.assertFalse(request["gpu_capture_authorized_at_prepare"])
                self.assertEqual(request["formal_dataset_count"], 0)
                self.assertEqual(
                    set(request["package_records"]),
                    {"archive_launcher", "archive_binary", "archive_pak"},
                )
                self.assertIn("rir_shard", request["artifact_records"])
                self.assertIn("base_capture_runner", request["source_records"])
                validated, argv = LAUNCHER._validate_request(request_path)
                self.assertEqual(validated["required_repo_commit"], "a" * 40)
                self.assertEqual(argv.count("--frame-index"), 1)

    def test_bound_artifact_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = Layout(Path(directory))
            with layout.patched():
                layout.materialize()
                request_path = layout.prepare()
                shard = layout.atom / "exact_rir_cache_v3/shards/shard_000000.npz"
                shard.write_bytes(shard.read_bytes() + b"mutation")
                with self.assertRaisesRegex(RuntimeError, "binding drift"):
                    LAUNCHER._validate_request(request_path)

    def test_dry_run_does_not_require_upstream_or_consume_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = Layout(Path(directory))
            with layout.patched():
                layout.materialize()
                request_path = layout.prepare()
                with (
                    mock.patch.object(
                        LAUNCHER, "_gpu_snapshot", return_value=_idle_snapshot()
                    ),
                    mock.patch.object(LAUNCHER, "_assert_port_available"),
                ):
                    code = LAUNCHER.run(
                        request_path,
                        dry_run=True,
                        authorize_gpu_capture=False,
                        mp3d_v2_terminal_receipt=layout.upstream,
                    )
                self.assertEqual(code, 0)
                receipt = json.loads(
                    (request_path.parent / "dry_run_receipt.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertFalse(receipt["gpu_started"])
                self.assertFalse(receipt["attempt_consumed"])
                self.assertFalse(receipt["mp3d_revision_v2_terminal_checked"])
                self.assertFalse(
                    (request_path.parent / "running_receipt.json").exists()
                )

    def test_real_launch_requires_both_authorization_and_mp3d_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = Layout(Path(directory))
            with layout.patched():
                layout.materialize()
                request_path = layout.prepare()
                with self.assertRaisesRegex(
                    RuntimeError, "explicit launch authorization"
                ):
                    LAUNCHER.run(
                        request_path,
                        dry_run=False,
                        authorize_gpu_capture=False,
                        mp3d_v2_terminal_receipt=layout.upstream,
                    )
                with (
                    mock.patch.object(
                        LAUNCHER, "_gpu_snapshot", return_value=_idle_snapshot()
                    ),
                    mock.patch.object(LAUNCHER, "_assert_port_available"),
                    self.assertRaises(FileNotFoundError),
                ):
                    LAUNCHER.run(
                        request_path,
                        dry_run=False,
                        authorize_gpu_capture=True,
                        mp3d_v2_terminal_receipt=layout.upstream,
                    )
                self.assertFalse(
                    (request_path.parent / "running_receipt.json").exists()
                )

    def test_success_persists_child_logs_ordered_phases_and_final_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = Layout(Path(directory))
            with layout.patched():
                layout.materialize()
                request_path = layout.prepare()
                layout.write_upstream_terminal()

                def child(*_args, **kwargs):
                    kwargs["stdout"].write(b"capture stdout\n")
                    kwargs["stderr"].write(b"")
                    layout.make_capture()
                    return SimpleNamespace(returncode=0)

                with (
                    mock.patch.object(
                        LAUNCHER,
                        "_gpu_snapshot",
                        side_effect=[_idle_snapshot(), _idle_snapshot()],
                    ),
                    mock.patch.object(LAUNCHER, "_assert_port_available"),
                    mock.patch.object(LAUNCHER.subprocess, "run", side_effect=child),
                ):
                    code = LAUNCHER.run(
                        request_path,
                        dry_run=False,
                        authorize_gpu_capture=True,
                        mp3d_v2_terminal_receipt=layout.upstream,
                    )
                self.assertEqual(code, 0)
                final = json.loads(
                    (request_path.parent / "final_receipt.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    final["status"], "pass_diagnostic_f15_manual_review_pending"
                )
                self.assertEqual(final["child_exit_code"], 0)
                self.assertEqual(
                    [item["phase"] for item in final["launcher_phases"]],
                    [
                        "prelaunch_closed",
                        "child_invocation_started",
                        "child_exit_observed",
                        "capture_validation_started",
                        "complete",
                    ],
                )
                self.assertEqual(
                    (request_path.parent / "capture_stdout.log").read_text(
                        encoding="utf-8"
                    ),
                    "capture stdout\n",
                )
                self.assertEqual(final["formal_dataset_count"], 0)

    def test_child_failure_persists_stderr_traceback_and_forbids_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = Layout(Path(directory))
            with layout.patched():
                layout.materialize()
                request_path = layout.prepare()
                layout.write_upstream_terminal()

                def child(*_args, **kwargs):
                    kwargs["stdout"].write(b"before failure\n")
                    kwargs["stderr"].write(
                        b"Traceback (most recent call last):\nRuntimeError: fake\n"
                    )
                    return SimpleNamespace(returncode=1)

                with (
                    mock.patch.object(
                        LAUNCHER,
                        "_gpu_snapshot",
                        side_effect=[_idle_snapshot(), _idle_snapshot()],
                    ),
                    mock.patch.object(LAUNCHER, "_assert_port_available"),
                    mock.patch.object(LAUNCHER.subprocess, "run", side_effect=child),
                ):
                    code = LAUNCHER.run(
                        request_path,
                        dry_run=False,
                        authorize_gpu_capture=True,
                        mp3d_v2_terminal_receipt=layout.upstream,
                    )
                self.assertEqual(code, 1)
                final = json.loads(
                    (request_path.parent / "final_receipt.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(final["status"], "failed")
                self.assertEqual(final["child_exit_code"], 1)
                self.assertTrue(final["launcher_traceback"].strip())
                self.assertIn(
                    "Traceback",
                    (request_path.parent / "capture_stderr.log").read_text(
                        encoding="utf-8"
                    ),
                )
                with self.assertRaisesRegex(RuntimeError, "final receipt"):
                    LAUNCHER.run(
                        request_path,
                        dry_run=False,
                        authorize_gpu_capture=True,
                        mp3d_v2_terminal_receipt=layout.upstream,
                    )

    def test_capture_validator_rejects_subthreshold_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = Layout(Path(directory))
            with layout.patched():
                layout.materialize()
                request_path = layout.prepare()
                request = json.loads(request_path.read_text(encoding="utf-8"))
                capture = layout.make_capture()
                truth_path = capture / "pixel_visibility_truth.json"
                truth = json.loads(truth_path.read_text(encoding="utf-8"))
                truth["per_instance"]["source1"]["frames"][0]["visible_fraction"] = 0.79
                _write_json(truth_path, truth)
                manifest_path = capture / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                record = manifest["artifact_records"]["pixel_visibility_truth"]
                record["size_bytes"] = truth_path.stat().st_size
                record["sha256"] = _sha256(truth_path)
                _write_json(manifest_path, manifest)
                with self.assertRaisesRegex(RuntimeError, "visibility/pixel gate"):
                    LAUNCHER._validate_capture(request)


if __name__ == "__main__":
    unittest.main()
