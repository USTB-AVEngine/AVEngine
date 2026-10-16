from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/adapt_strict_two_human_both_move_v1_preflight.py"
INPUT_PATH = (
    REPOSITORY
    / "examples/qa/native_strict_two_human_both_move_v1_geometry_preflight_v1.json"
)
SPEC = importlib.util.spec_from_file_location("both_move_adapter", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_both_move_adapter_binds_cross_scenario_paths_and_controlled_audio(
    tmp_path: Path,
) -> None:
    output = TOOL.adapt(INPUT_PATH, tmp_path / "preflight.json")
    adapted = json.loads(output.read_text(encoding="utf-8"))
    row = adapted["canaries"][0]

    assert adapted["status"] == "pass_cpu_adapter_ready_for_materialization"
    assert adapted["integration_adapter"] == {
        "schema": "avengine_both_move_cross_scenario_adapter_v1",
        "status": "pass",
        "candidate_revision": "both_move_v1_0304_0990_equal_arc_v1",
        "counterfactual_pair": True,
        "native_same_scene_pair": False,
        "native_human_anchor_scenario_ids": TOOL.EXPECTED_SCENARIOS,
        "interior_root_authority": (
            "derived_equal_arc_interpolation_not_native_readback"
        ),
        "audio_contract": "source1_controlled_speech_source2_exact_silence",
        "gpu_launch_authorized": False,
        "formal": False,
        "qualification_claim": False,
    }
    assert row["source_suite"] == str(TOOL.SOURCE_SUITE)
    assert row["target"]["sound_asset_id"] == "speech_cremad_1001_ieo_neu_v1"
    assert row["target"]["speech_frame_window_inclusive"] == [7, 31]
    assert row["target"]["speech_sample_count"] == 25_626
    assert row["distractor"]["voice_policy"] == "silent"
    assert row["native_source_scenario_ids"] == TOOL.EXPECTED_SCENARIOS
    for role in ("target", "distractor"):
        provenance = row[role]["path_provenance"]
        assert provenance["method"] == TOOL.PATH_METHOD
        assert provenance["counterfactual_cross_scenario_pair"] is True
        assert provenance["endpoints_exact_native_readbacks"] is True
        assert provenance["interior_output_roots_exact_native_frame_readbacks"] is False


def test_both_move_adapter_is_no_clobber(tmp_path: Path) -> None:
    output = TOOL.adapt(INPUT_PATH, tmp_path / "preflight.json")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        TOOL.adapt(INPUT_PATH, output)
