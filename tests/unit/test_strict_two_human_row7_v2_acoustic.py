from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/build_strict_two_human_row7_v2_acoustic.py"
TOOL_SPEC = importlib.util.spec_from_file_location(
    "build_strict_two_human_row7_v2_acoustic", TOOL_PATH
)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
TOOL = importlib.util.module_from_spec(TOOL_SPEC)
TOOL_SPEC.loader.exec_module(TOOL)

OVERLAY = (
    REPOSITORY
    / "examples/qa/native_strict_two_human_row7_v2_revision_overlay.json"
)
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


def _prepare(tmp_path: Path) -> tuple[Path, dict, Path]:
    overlay_preflight = TOOL.OVERLAY.build(
        OVERLAY, tmp_path / "overlay_preflight"
    )
    output = tmp_path / "row7_v2_acoustic"
    manifest_path = TOOL.prepare(
        overlay_path=OVERLAY,
        overlay_preflight_path=overlay_preflight,
        registry_path=REGISTRY,
        source_suite_path=SOURCE_SUITE,
        controlled_registry_path=CONTROLLED_REGISTRY,
        output=output,
    )
    return output, _load(manifest_path), overlay_preflight


def test_row7_v2_prepare_binds_overlay_full_1002_and_zero_distractor(
    tmp_path: Path,
) -> None:
    output, manifest, overlay_preflight = _prepare(tmp_path)
    assert manifest["status"] == TOOL.PREPARED_STATUS
    assert manifest["row_count"] == 1
    assert manifest["cross_attempt_rir_cache_audio_reuse_allowed"] is False
    assert manifest["forbidden_v1_acoustic_root"] == str(
        TOOL.OLD_V1_ACOUSTIC_ROOT.resolve()
    )
    assert manifest["gpu_executed"] is False
    assert manifest["formal_scene_count"] == 0
    row = manifest["rows"][0]
    assert row["row_id"] == TOOL.ROW_ID
    assert row["episode_id"] == TOOL.EPISODE_ID
    assert row["target_sound_asset_id"] == TOOL.SOUND_ASSET_ID
    assert row["target_event_frame_window_inclusive"] == [7, 50]

    recipe_root = output / TOOL.ROW_ID / "recipe_v1"
    recipe = _load(recipe_root / "recipe.json")
    preflight = _load(recipe_root / "preflight.json")
    program = _load(recipe_root / "controlled_audio_program/audio_program.json")
    trajectory = _load(recipe_root / "trajectory_bank.json")
    request = _load(recipe_root / "sparse_native_gate_request.json")
    assert Path(recipe["inputs"]["plan"]["path"]) == Path(
        _load(overlay_preflight)["effective_plan"]
    )
    assert Path(recipe["inputs"]["cpu_preflight"]["path"]) == overlay_preflight
    assert preflight["target_sound_asset_id"] == TOOL.SOUND_ASSET_ID
    assert preflight["target_event_sample_window"] == [7467, 53379]
    assert preflight["target_event_frame_window_inclusive"] == [7, 50]
    assert preflight["target_event_count"] == 1
    assert preflight["distractor_event_count"] == 0
    assert preflight["f15_target_speaking"] is True
    assert len(program["events"]) == 1
    assert program["events"][0]["source_endpoint_id"] == "lead_d_source1_mouth"
    assert program["events"][0]["source_start_sample"] == 0
    assert program["events"][0]["source_end_sample_exclusive"] == 45912

    episode = trajectory["episodes"][0]
    assert episode["episode_id"] == TOOL.EPISODE_ID
    assert episode["source_center_paths_m"]["source1"][15] == pytest.approx(
        [-4.001039245644131, 1.969012451171875, -2.231611274384164]
    )
    assert episode["source_center_paths_m"]["source2"][15] == pytest.approx(
        [-4.403688933398272, 2.064033031463623, -0.9725009525144421]
    )
    assert request["status"] == "blocked_pending_exact_rir_cache_binaural"
    assert request["frame_indices"] == [15]
    assert request["projection_and_native_thresholds"][
        "target_visible_fraction_minimum"
    ] == 0.8
    assert request["projection_and_native_thresholds"][
        "distractor_visible_fraction_minimum"
    ] == 0.5


def test_row7_v2_prepare_is_no_clobber_and_rejects_stale_overlay_preflight(
    tmp_path: Path,
) -> None:
    output, _, overlay_preflight = _prepare(tmp_path)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        TOOL.prepare(
            overlay_path=OVERLAY,
            overlay_preflight_path=overlay_preflight,
            registry_path=REGISTRY,
            source_suite_path=SOURCE_SUITE,
            controlled_registry_path=CONTROLLED_REGISTRY,
            output=output,
        )

    invalid = _load(overlay_preflight)
    invalid["status"] = "pass"
    invalid_path = tmp_path / "invalid_overlay_preflight.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(RuntimeError, match="overlay preflight status mismatch"):
        TOOL.prepare(
            overlay_path=OVERLAY,
            overlay_preflight_path=invalid_path,
            registry_path=REGISTRY,
            source_suite_path=SOURCE_SUITE,
            controlled_registry_path=CONTROLLED_REGISTRY,
            output=tmp_path / "invalid_output",
        )


def test_batch_finalizer_defaults_remain_backward_compatible() -> None:
    parameters = inspect.signature(TOOL.BATCH.finalize).parameters
    assert parameters["expected_row_count"].default == 7
    assert parameters["delivery_schema"].default == TOOL.BATCH.DELIVERY_SCHEMA
    assert parameters["delivery_status"].default == (
        "pass_cpu_rows2_to8_ready_for_sequential_f15_sparse"
    )
