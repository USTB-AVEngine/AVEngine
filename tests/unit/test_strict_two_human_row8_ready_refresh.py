from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/refresh_strict_two_human_row8_ready.py"
SPEC = importlib.util.spec_from_file_location("row8_ready_refresh", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)

ROOT = REPOSITORY / "tmp/lead_d_strict_two_human_expansion_v1"
PLAN = REPOSITORY / "examples/qa/native_strict_two_human_expansion_v1.json"
PREFLIGHT = ROOT / "cpu_preflight_contract_v5/preflight.json"
REGISTRY = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"
OLD_READY = (
    ROOT
    / "cpu_delivery_v3/strict_08_construction_female_left"
    / "sparse_native_gate_request.ready.json"
)
OLD_DELIVERY = ROOT / "cpu_delivery_v3/delivery.json"
BATCH_ROOT = ROOT / "acoustic_batch_v1"


def _build(output: Path, *, plan: Path = PLAN) -> Path:
    return TOOL.build(
        plan_path=plan,
        current_preflight_path=PREFLIGHT,
        registry_path=REGISTRY,
        old_ready_path=OLD_READY,
        old_delivery_path=OLD_DELIVERY,
        batch_root=BATCH_ROOT,
        output=output,
    )


def test_refresh_row8_ready_uses_split_contract_without_recomputing_audio(
    tmp_path: Path,
) -> None:
    ready_path = _build(tmp_path / "delivery")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    thresholds = ready["projection_and_native_thresholds"]
    assert thresholds["target_visible_fraction_minimum"] == 0.8
    assert thresholds["distractor_visible_fraction_minimum"] == 0.5
    assert "visible_fraction_minimum_per_actor" not in thresholds
    assert ready["row_id"] == TOOL.ROW_ID
    assert ready["episode_id"] == TOOL.EPISODE_ID
    assert ready["frame_indices"] == [15]
    assert ready["audio_record"]["sample_count"] == 80000
    assert ready["contract_refresh"][
        "exact_rir_cache_binaural_reused_without_recompute"
    ] is True
    assert ready["contract_refresh"]["cross_row_or_cross_attempt_acoustic_reuse"] is False
    assert ready["gpu_executed"] is False
    assert ready["formal_scene_count"] == 0
    receipt = json.loads(
        (ready_path.parent / "refresh_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "pass_ready_for_single_row8_f15_native_sparse"
    assert receipt["exact_rir_job_count"] == 2
    assert receipt["source1_peak_absolute"] > 0.0
    assert receipt["source2_peak_absolute"] == 0.0


def test_refresh_rejects_mutated_current_contract(tmp_path: Path) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["projection_and_native_thresholds"][
        "target_visible_fraction_minimum"
    ] = 0.5
    invalid = tmp_path / "invalid_plan.json"
    invalid.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(RuntimeError, match="current strict8 plan failed|split visibility"):
        _build(tmp_path / "invalid_delivery", plan=invalid)


def test_refresh_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "delivery"
    _build(output)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        _build(output)

_RETAINED_TMP_WORKSPACE = Path(__file__).resolve().parents[2] / "tmp"
if not _RETAINED_TMP_WORKSPACE.exists():
    pytest.skip(
        "retained strict-two-human evidence workspace (repository tmp "
        "symlink) is not present in this checkout",
        allow_module_level=True,
    )
