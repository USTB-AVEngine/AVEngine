from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

THIS_FILE = Path(__file__).resolve()
STAGING_LAYOUT = (THIS_FILE.parents[1] / "config").is_dir()
ROOT = THIS_FILE.parents[1] if STAGING_LAYOUT else THIS_FILE.parents[2]
SCRIPT = (
    ROOT / "tools/build_skokloster_strict_two_human_preflight.py"
    if STAGING_LAYOUT
    else ROOT / "tools/qa/build_skokloster_strict_two_human_preflight.py"
)
SPEC = importlib.util.spec_from_file_location("skok_strict_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def layout_path(staging: str, repository: str) -> Path:
    return ROOT / (staging if STAGING_LAYOUT else repository)


def fixtures() -> tuple[dict, dict, dict]:
    request = load(
        layout_path(
            "config/runtime/native_strict_two_human_skokloster_room_atom_v1.json",
            "examples/qa/native_strict_two_human_skokloster_room_atom_v1.json",
        )
    )
    search = load(
        layout_path(
            "artifacts/skokloster_strict_listener_search_v1.json",
            "examples/qa/native_strict_two_human_skokloster_listener_search_v1.json",
        )
    )
    evidence = MODULE._validate_external_evidence(
        request=request,
        search=search,
        rejection=load(
            layout_path(
                "artifacts/strict_f15_near_listener_cpu_rejected_v1.json",
                "examples/qa/native_strict_two_human_skokloster_near_listener_rejected_v1.json",
            )
        ),
        runtime_profile=load(
            layout_path(
                "config/runtime/skokloster_room_runtime_profile.json",
                "examples/m3/skokloster_castle/skokloster_room_runtime_profile.json",
            )
        ),
        acoustic_profile=load(
            layout_path(
                "config/acoustics/skokloster_acoustic_profile.json",
                "examples/m3/skokloster_castle/skokloster_acoustic_profile.json",
            )
        ),
        package={
            "schema": "avengine_acoustic_scene_package_v1",
            "package_id": MODULE.PACKAGE_ID,
            "package_mode": "research_candidate",
            "geometry": {"triangle_count": 999935},
        },
        simulation=load(
            layout_path(
                "config/acoustics/skokloster_native_rlr_load_test_request.json",
                "examples/m3/skokloster_castle/skokloster_native_rlr_load_test_request.json",
            )
        ),
        audio_program={
            "schema": "avengine_m6_audio_program_v1",
            "mode": "one_active_of_n",
            "events": [
                {
                    "source_endpoint_id": "lead_d_source1_mouth",
                    "sound_asset_id": "speech_cremad_1001_ieo_neu_v1",
                    "start_sample": 7467,
                    "end_sample_exclusive": 33093,
                    "source_start_sample": 0,
                    "source_end_sample_exclusive": 25626,
                    "linear_gain": 0.18,
                    "fade_samples": 80,
                }
            ],
        },
        audio_binding={
            "schema": "avengine_native_strict_two_human_audio_binding_v1",
            "target_event_count": 1,
            "distractor_event_count": 0,
            "controlled_content": {"source1": {}, "source2": None},
        },
    )
    return request, search, evidence


def walk(value: object):
    yield value
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def test_materializes_static_75f_two_rir_and_silent_distractor() -> None:
    request, _, evidence = fixtures()
    MODULE._validate_request(request)
    documents = MODULE._build_documents(request, evidence)
    suite = documents["suite_execution_plan.json"]
    scenario = suite["scenarios"][0]
    frames = scenario["plan"]["frames"]
    assert len(frames) == 75
    assert (
        len({tuple(frame["camera_state"]["habitat_position_m"]) for frame in frames})
        == 1
    )
    assert all(
        frame["camera_state"]["habitat_position_m"]
        == evidence["camera_listener_habitat_m"]
        for frame in frames
    )
    assert [item["asset_id"] for item in scenario["plan"]["actors"]] == [
        MODULE.MALE_ASSET,
        MODULE.FEMALE_ASSET,
    ]
    rir = documents["rir_job_plan.json"]
    assert rir["unique_rir_job_count"] == 2
    assert len(rir["jobs"]) == 2
    assert [len(job["uses"]) for job in rir["jobs"]] == [75, 75]
    assert rir["listener_position_m"] == evidence["camera_listener_habitat_m"]
    assert "listener_pose_mode" not in rir
    assert all("listener_position_m" not in job for job in rir["jobs"])
    audio = documents["audio_program_binding.json"]
    assert audio["source1"]["sound_class"] == "human_speech"
    assert audio["source2"]["sound_class"] == "silent_human"
    assert audio["source2"]["event_count"] == 0


def test_spear_capture_steps_use_origin_main_environment_contract() -> None:
    request, _, _ = fixtures()
    official_python = "/data/jzy/miniconda3/envs/spear-env/bin/python"
    assert request["execution"]["python"] == official_python
    assert "/.venv/" not in request["execution"]["python"]

    output = Path(
        "/data/jzy/code/AVEngine-lead-a/tmp/"
        "lead_a_skokloster_strict_two_human_v1/cpu_preflight_v4"
    )
    execution = MODULE._execution_plan(request, output)
    assert len(execution["gpu_steps"]) == 2
    assert all(step["argv"][0] == official_python for step in execution["gpu_steps"])
    audio_step = execution["cpu_steps"][2]
    assert audio_step["argv"][0] == official_python


def test_real_rir_validator_accepts_legacy_fixed_listener_plan() -> None:
    rir_cache = pytest.importorskip("avengine.m6x.rir_cache")
    request, _, evidence = fixtures()
    documents = MODULE._build_documents(request, evidence)
    normalized = rir_cache.validate_rir_job_plan(documents["rir_job_plan.json"])
    assert len(normalized) == 2
    assert all(
        item["listener_position_m"] == evidence["camera_listener_habitat_m"]
        for item in normalized
    )


def test_rir_execution_uses_authoritative_runtime_and_path_closed_v3_cache() -> None:
    request, _, _ = fixtures()
    output = Path(
        "/data/jzy/code/AVEngine-lead-a/tmp/"
        "lead_a_skokloster_strict_two_human_v1/cpu_preflight_v4"
    )
    execution = MODULE._execution_plan(request, output)
    runtime_step, rir_step, audio_step = execution["cpu_steps"]
    expected_environment = {
        "AVENGINE_HABITAT_RUNTIME_ROOT": "/data/jzy/code/habitat-sim-AVEngine",
        "AVENGINE_SOUNDSPACES_ROOT": "/data/jzy/code/sound-spaces",
        "AVENGINE_SKOKLOSTER_RLR48_PACKAGE_ROOT": (
            "/tmp/skokloster_strict_room_atom_run/clean_package"
        ),
        "PATH": (
            "/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        "PYTHONPATH": "/data/jzy/code/AVEngine-lead-a/src",
        "SKBUILD_EDITABLE_SKIP": (
            "/data/jzy/code/habitat-sim-AVEngine/build/cp312-cp312-linux_x86_64"
        ),
        "NUMBA_DISABLE_JIT": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    habitat_python = "/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python"
    assert runtime_step["step_id"] == "probe_authoritative_habitat_rir_runtime"
    assert runtime_step["argv"][0] == habitat_python
    assert runtime_step["environment"] == expected_environment
    assert runtime_step["expected"]["cuda_initialized"] is False
    assert rir_step["attempt_id"] == "exact_rir_cache_v3"
    assert rir_step["supersedes_failed_attempts"] == ["exact_rir_cache_v1"]
    assert rir_step["argv"][:2] == [
        habitat_python,
        "/data/jzy/code/AVEngine-lead-a/tools/m6x/render_rir_cache.py",
    ]
    assert rir_step["environment"] == expected_environment
    assert rir_step["argv"][rir_step["argv"].index("--output") + 1].endswith(
        "/exact_rir_cache_v3"
    )
    assert audio_step["argv"][audio_step["argv"].index("--output") + 1].endswith(
        "/binaural_v4"
    )
    MODULE.validate_rir_runtime_binding(habitat_python, expected_environment)
    for missing_name in expected_environment:
        mutation = dict(expected_environment)
        mutation.pop(missing_name)
        with pytest.raises(RuntimeError, match=missing_name):
            MODULE.validate_rir_execution_environment(mutation)


def test_rejects_wrong_rir_interpreter_even_with_valid_environment() -> None:
    valid_environment = {
        "AVENGINE_HABITAT_RUNTIME_ROOT": MODULE.HABITAT_RUNTIME_ROOT,
        "AVENGINE_SOUNDSPACES_ROOT": MODULE.SOUNDSPACES_ROOT,
        "AVENGINE_SKOKLOSTER_RLR48_PACKAGE_ROOT": str(
            MODULE.SKOKLOSTER_RLR48_PACKAGE_ROOT
        ),
        "PATH": MODULE.HABITAT_PATH,
        "PYTHONPATH": str(MODULE.REMOTE_REPOSITORY / "src"),
        "SKBUILD_EDITABLE_SKIP": MODULE.HABITAT_EDITABLE_BUILD,
        "NUMBA_DISABLE_JIT": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    with pytest.raises(RuntimeError, match="runtime interpreter"):
        MODULE.validate_rir_runtime_binding(
            "/data/jzy/code/AVEngine-lead-a/.venv/bin/python",
            valid_environment,
        )


def test_canonical_sensor_rig_validates_and_aligns_all_rir_uses() -> None:
    sensor_rig = pytest.importorskip("avengine.sensor_rig_trajectory")
    m7_sensor_rig = pytest.importorskip("avengine.m7.sensor_rig")
    request, _, evidence = fixtures()
    documents = MODULE._build_documents(request, evidence)
    rig = documents["sensor_rig_trajectory.json"]
    assert sensor_rig.validate_sensor_rig_trajectory(rig) == []
    binding = m7_sensor_rig.m7_sensor_rig_binding(rig)
    assert binding["dynamic"] is False
    assert rig["trajectory_id"].endswith("__sensor_rig_v3")
    assert len(rig["frames"]) == 75
    assert all(
        frame["world_from_rig"]["translation_m"]
        == evidence["camera_listener_habitat_m"]
        for frame in rig["frames"]
    )
    alignment = m7_sensor_rig.validate_m7_rir_listener_alignment(
        rir_job_plan=documents["rir_job_plan.json"],
        sensor_rig_trajectory=rig,
    )
    assert alignment["listener_pose_mode"] == "fixed"
    assert alignment["checked_use_count"] == 150


def test_only_canonical_sensor_rig_uses_internal_pose_identity() -> None:
    sensor_rig = pytest.importorskip("avengine.sensor_rig_trajectory")
    request, _, evidence = fixtures()
    documents = MODULE._build_documents(request, evidence)
    rig = documents.pop("sensor_rig_trajectory.json")
    keys = [value.lower() for value in walk(documents) if isinstance(value, str)]
    assert not any("sha256" in value or "hash" in value for value in keys)
    assert rig["pose_hash_algorithm"] == sensor_rig.POSE_HASH_ALGORITHM
    assert all(
        set(frame) == {"frame_index", "pts_ticks", "world_from_rig", "pose_hash"}
        for frame in rig["frames"]
    )


def test_rejects_decoupled_camera_listener() -> None:
    request, search, _ = fixtures()
    bad = deepcopy(search)
    bad["selected"]["coupled_camera_listener"] = False
    with pytest.raises(RuntimeError, match="camera/listener"):
        MODULE._validate_external_evidence(
            request=request,
            search=bad,
            rejection=load(
                layout_path(
                    "artifacts/strict_f15_near_listener_cpu_rejected_v1.json",
                    "examples/qa/native_strict_two_human_skokloster_near_listener_rejected_v1.json",
                )
            ),
            runtime_profile=load(
                layout_path(
                    "config/runtime/skokloster_room_runtime_profile.json",
                    "examples/m3/skokloster_castle/skokloster_room_runtime_profile.json",
                )
            ),
            acoustic_profile=load(
                layout_path(
                    "config/acoustics/skokloster_acoustic_profile.json",
                    "examples/m3/skokloster_castle/skokloster_acoustic_profile.json",
                )
            ),
            package={
                "schema": "avengine_acoustic_scene_package_v1",
                "package_id": MODULE.PACKAGE_ID,
                "package_mode": "research_candidate",
                "geometry": {"triangle_count": 999935},
            },
            simulation=load(
                layout_path(
                    "config/acoustics/skokloster_native_rlr_load_test_request.json",
                    "examples/m3/skokloster_castle/skokloster_native_rlr_load_test_request.json",
                )
            ),
            audio_program={
                "schema": "avengine_m6_audio_program_v1",
                "mode": "one_active_of_n",
                "events": [
                    {
                        "source_endpoint_id": "lead_d_source1_mouth",
                        "sound_asset_id": "speech_cremad_1001_ieo_neu_v1",
                        "start_sample": 7467,
                        "end_sample_exclusive": 33093,
                        "source_start_sample": 0,
                        "source_end_sample_exclusive": 25626,
                        "linear_gain": 0.18,
                        "fade_samples": 80,
                    }
                ],
            },
            audio_binding={
                "schema": "avengine_native_strict_two_human_audio_binding_v1",
                "target_event_count": 1,
                "distractor_event_count": 0,
                "controlled_content": {"source2": None},
            },
        )
