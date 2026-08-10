from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = (
    REPOSITORY
    / "tools/qa/build_strict_two_human_expansion_acoustic_batch.py"
)
TOOL_SPEC = importlib.util.spec_from_file_location(
    "build_strict_two_human_expansion_acoustic_batch", TOOL_PATH
)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
TOOL = importlib.util.module_from_spec(TOOL_SPEC)
TOOL_SPEC.loader.exec_module(TOOL)

PLAN = REPOSITORY / "examples/qa/native_strict_two_human_expansion_v1.json"
REGISTRY = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"
SOURCE_SUITE = (
    REPOSITORY
    / "tmp/lead_a_native_paper_balance_v1/stationary_finalized_gpu1_v3"
    / "suite_execution_plan.json"
)
CONTROLLED_REGISTRY = Path(
    "/data/jzy/code/SPEAR-lead-b/outputs/lead_b/audio_candidates_v1/"
    "controlled_sound_content_registry_v1.json"
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _prepare(tmp_path: Path) -> tuple[Path, dict]:
    preflight_path = TOOL.PREFLIGHT.build(PLAN, tmp_path / "strict8_preflight")
    output = tmp_path / "strict8_acoustic"
    manifest_path = TOOL.prepare(
        plan_path=PLAN,
        cpu_preflight_path=preflight_path,
        registry_path=REGISTRY,
        source_suite_path=SOURCE_SUITE,
        controlled_registry_path=CONTROLLED_REGISTRY,
        output=output,
    )
    return output, _load(manifest_path)


def test_prepare_builds_seven_identity_specific_cpu_recipes(tmp_path: Path) -> None:
    output, manifest = _prepare(tmp_path)

    assert manifest["status"] == "prepared_cpu_pending_per_row_rir_cache_binaural"
    assert manifest["row_count"] == 7
    assert manifest["cross_row_rir_reuse_allowed"] is False
    assert manifest["gpu_executed"] is False
    assert manifest["formal_scene_count"] == 0
    assert manifest["retained_row1_canary"]["status"] == (
        "pass_existing_sparse_canary"
    )

    expected = {
        "strict_02_male_female_left": (
            "speech_cremad_1001_ieo_neu_v1",
            [7, 31],
        ),
        "strict_03_female_male_right": (
            "speech_cremad_1002_mti_neu_v1",
            [7, 50],
        ),
        "strict_04_female_male_left": (
            "speech_cremad_1002_mti_neu_v1",
            [7, 50],
        ),
        "strict_05_male_construction_right": (
            "speech_cremad_1001_ieo_neu_v1",
            [7, 31],
        ),
        "strict_06_construction_male_left": (
            "speech_cremad_1005_tie_neu_v1",
            [7, 50],
        ),
        "strict_07_female_construction_right": (
            "speech_cremad_1002_mti_neu_v1",
            [7, 50],
        ),
        "strict_08_construction_female_left": (
            "speech_cremad_1005_tie_neu_v1",
            [7, 50],
        ),
    }
    assert {
        row["row_id"]: (
            row["target_sound_asset_id"],
            row["target_event_frame_window_inclusive"],
        )
        for row in manifest["rows"]
    } == expected

    for row in manifest["rows"]:
        recipe_root = output / row["row_id"] / "recipe_v1"
        recipe = _load(recipe_root / "recipe.json")
        program = _load(recipe_root / "controlled_audio_program/audio_program.json")
        suite = _load(recipe_root / "suite_execution_plan.pending_fact.json")
        trajectory = _load(recipe_root / "trajectory_bank.json")
        request = _load(recipe_root / "sparse_native_gate_request.json")
        preflight = _load(recipe_root / "preflight.json")

        assert len(program["events"]) == 1
        event = program["events"][0]
        assert event["source_endpoint_id"] == "lead_d_source1_mouth"
        assert event["start_sample"] == 7467
        assert event["source_start_sample"] == 0
        assert event["source_end_sample_exclusive"] == (
            event["end_sample_exclusive"] - event["start_sample"]
        )
        assert preflight["distractor_event_count"] == 0
        assert preflight["f15_target_speaking"] is True
        assert recipe["recipe_identity_sha256"] == row[
            "recipe_identity_sha256"
        ]
        assert {
            record["role"] for record in recipe["inputs"].values()
        } == {
            "strict8_plan",
            "strict8_cpu_preflight",
            "runtime_registry",
            "source_suite_template",
            "controlled_sound_registry",
        }
        assert all(
            len(record["sha256"]) == 64
            and Path(record["path"]).is_file()
            and Path(record["path"]).stat().st_size == record["size_bytes"]
            for record in recipe["inputs"].values()
        )
        assert set(preflight["conservative_vertical_envelope_fraction"]) == {
            "source1",
            "source2",
        }

        scenario = suite["scenarios"][0]
        assert scenario["scenario_id"] == row["episode_id"]
        assert len(scenario["plan"]["frames"]) == 75
        assert [actor["actor_id"] for actor in scenario["plan"]["actors"]] == [
            "source1_actor",
            "source2_actor",
        ]
        assert sorted(
            scenario["authoritative_capture_request"][
                "runtime_asset_expectations"
            ]
        ) == ["source1", "source2"]
        assert trajectory["episode_count"] == 1
        assert set(
            trajectory["episodes"][0]["source_center_paths_m"]
        ) == {"source1", "source2"}
        assert all(
            len(points) == 75
            for points in trajectory["episodes"][0][
                "source_center_paths_m"
            ].values()
        )

        assert request["status"] == "blocked_pending_exact_rir_cache_binaural"
        assert request["frame_indices"] == [15]
        assert request["target_only_actor_map"] == {
            "source1": "source1_actor",
            "source2": "source2_actor",
        }
        assert request["physical_gpu_index"] == 1
        assert request["forbidden_physical_gpu_indices"] == [0, 3]
        assert set(request["required_live_gates"]) == {
            "stable_actor_tag",
            "exact_blueprint_class",
            "skeletal_mesh",
            "skeleton",
            "standing_idle",
            "native_actor_root_plus_declared_profile_offset",
        }


def test_prepare_is_no_clobber_and_rejects_preflight_drift(
    tmp_path: Path,
) -> None:
    output, _ = _prepare(tmp_path)
    preflight = TOOL.PREFLIGHT.build(PLAN, tmp_path / "second_preflight")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        TOOL.prepare(
            plan_path=PLAN,
            cpu_preflight_path=preflight,
            registry_path=REGISTRY,
            source_suite_path=SOURCE_SUITE,
            controlled_registry_path=CONTROLLED_REGISTRY,
            output=output,
        )

    invalid = _load(preflight)
    invalid["status"] = "pass"
    invalid_path = tmp_path / "invalid_preflight.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(RuntimeError, match="CPU preflight status mismatch"):
        TOOL.prepare(
            plan_path=PLAN,
            cpu_preflight_path=invalid_path,
            registry_path=REGISTRY,
            source_suite_path=SOURCE_SUITE,
            controlled_registry_path=CONTROLLED_REGISTRY,
            output=tmp_path / "invalid_output",
        )

    stale = _load(preflight)
    stale["plan_record"]["sha256"] = "0" * 64
    stale_path = tmp_path / "stale_preflight.json"
    stale_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not bind this exact plan"):
        TOOL.prepare(
            plan_path=PLAN,
            cpu_preflight_path=stale_path,
            registry_path=REGISTRY,
            source_suite_path=SOURCE_SUITE,
            controlled_registry_path=CONTROLLED_REGISTRY,
            output=tmp_path / "stale_output",
        )


def test_full_dry_window_rejects_silent_fifteen_or_overflow() -> None:
    assert TOOL._speech_window(start_sample=7467, source_sample_count=25626) == (
        7467,
        33093,
        [7, 31],
    )
    assert TOOL._speech_window(start_sample=7467, source_sample_count=45912) == (
        7467,
        53379,
        [7, 50],
    )
    with pytest.raises(RuntimeError, match="exceeds five seconds"):
        TOOL._speech_window(start_sample=7467, source_sample_count=80000)


def test_prepare_rejects_identity_speech_window_drift(tmp_path: Path) -> None:
    plan = _load(PLAN)
    invalid = deepcopy(plan)
    invalid["approved_identity_catalog"]["F"][
        "expected_speech_frame_window_inclusive"
    ] = [7, 31]
    invalid_plan = tmp_path / "invalid_plan.json"
    invalid_plan.write_text(json.dumps(invalid), encoding="utf-8")
    preflight = {
        "status": "pass_cpu_plan_pending_exact_rir_and_seven_sparse_native_gates",
        "row_count": 8,
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    with pytest.raises(RuntimeError, match="identity F speech window mismatch"):
        TOOL.prepare(
            plan_path=invalid_plan,
            cpu_preflight_path=preflight_path,
            registry_path=REGISTRY,
            source_suite_path=SOURCE_SUITE,
            controlled_registry_path=CONTROLLED_REGISTRY,
            output=tmp_path / "invalid_output",
        )
