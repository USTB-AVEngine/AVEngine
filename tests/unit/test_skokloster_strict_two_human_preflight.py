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


def test_generated_documents_do_not_add_digest_fields() -> None:
    request, _, evidence = fixtures()
    documents = MODULE._build_documents(request, evidence)
    keys = [value.lower() for value in walk(documents) if isinstance(value, str)]
    assert not any("sha256" in value or "hash" in value for value in keys)


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
