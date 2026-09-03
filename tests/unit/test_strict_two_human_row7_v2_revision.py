from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/build_strict_two_human_row7_v2_preflight.py"
TOOL_SPEC = importlib.util.spec_from_file_location(
    "build_strict_two_human_row7_v2_preflight", TOOL_PATH
)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
TOOL = importlib.util.module_from_spec(TOOL_SPEC)
TOOL_SPEC.loader.exec_module(TOOL)

OVERLAY_PATH = (
    REPOSITORY
    / "examples/qa/native_strict_two_human_row7_v2_revision_overlay.json"
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _inputs() -> tuple[dict, dict, dict, dict]:
    overlay = _load(OVERLAY_PATH)
    base = _load(REPOSITORY / overlay["base_plan"])
    rejection = _load(REPOSITORY / overlay["v1_rejection"])
    registry = _load(REPOSITORY / base["evidence"]["runtime_registry"])
    return overlay, base, rejection, registry


def test_row7_v2_overlay_materializes_only_allowed_fields() -> None:
    overlay, base, rejection, registry = _inputs()
    before = deepcopy(base)
    assert TOOL.validate_overlay(overlay, base, rejection, registry) == []

    revised, row_index = TOOL.apply_overlay(base, overlay)
    assert base == before
    assert row_index == 6
    assert revised["rows"][:6] == base["rows"][:6]
    assert revised["rows"][7:] == base["rows"][7:]
    old = base["rows"][row_index]
    new = revised["rows"][row_index]
    assert old["episode_id"].endswith("_v1")
    assert new["episode_id"].endswith("_v2")
    assert old["identity_pair"] == new["identity_pair"] == "F/C"
    assert old["target_expected_screen_side"] == new[
        "target_expected_screen_side"
    ] == "right"
    assert [TOOL._actor_contract(actor) for actor in old["actors"]] == [
        TOOL._actor_contract(actor) for actor in new["actors"]
    ]


def test_row7_v2_preflight_binds_native_points_and_projection(
    tmp_path: Path,
) -> None:
    result_path = TOOL.build(OVERLAY_PATH, tmp_path / "row7_v2_preflight")
    result = _load(result_path)
    assert result["status"] == (
        "pass_cpu_geometry_revision_pending_exact_rir_and_single_sparse_gate"
    )
    assert result["target_row_id"] == "strict_07_female_construction_right"
    assert result["replacement_episode_id"].endswith("_v2")
    assert result["base_plan_rows_unchanged"] is True
    assert len(result["unchanged_row_ids"]) == 7
    assert result["minimum_camera_cluster_distance_m"] > 2.0
    assert all(
        item["status"] == "pass_native_occupied_floor_point"
        and item["maximum_location_drift_m"] == 0.0
        for item in result["native_occupied_point_readbacks"]
    )
    projections = {
        item["source_slot_id"]: item for item in result["projections"]
    }
    assert projections["source1"]["mouth_xy_fraction"] == pytest.approx(
        [0.6093409809709169, 0.3535306478874262]
    )
    assert projections["source2"]["mouth_xy_fraction"] == pytest.approx(
        [0.3900148815903279, 0.3245568548893039]
    )
    assert result["immutable_contract"][
        "target_visible_fraction_minimum"
    ] == 0.8
    assert result["immutable_contract"][
        "distractor_visible_fraction_minimum"
    ] == 0.5
    assert result["formal_scene_count"] == 0
    assert result["qualification_claim"] is False
    assert result["gpu_or_rir_executed"] is False

    effective = _load(Path(result["effective_plan"]))
    assert effective["rows"][6]["episode_id"].endswith("_v2")
    expansion_preflight = _load(Path(result["expansion_preflight"]))
    assert expansion_preflight["row_count"] == 8
    assert expansion_preflight["camera_translation_cluster_count"] == 8
    assert expansion_preflight["formal_scene_count"] == 0

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        TOOL.build(OVERLAY_PATH, tmp_path / "row7_v2_preflight")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("target_threshold", "target visibility threshold must remain 0.8"),
        ("distractor_threshold", "distractor visibility threshold must remain 0.5"),
        ("identity_injection", "source1 replacement fields drift"),
        ("used_provenance", "revised provenance must be unused"),
        ("episode_reuse", "replacement Episode"),
    ],
)
def test_row7_v2_overlay_rejects_scope_or_contract_mutation(
    mutation: str,
    message: str,
) -> None:
    overlay, base, rejection, registry = _inputs()
    invalid = deepcopy(overlay)
    if mutation == "target_threshold":
        invalid["immutable_contract"]["target_visible_fraction_minimum"] = 0.5
    elif mutation == "distractor_threshold":
        invalid["immutable_contract"][
            "distractor_visible_fraction_minimum"
        ] = 0.49
    elif mutation == "identity_injection":
        invalid["replacement"]["actors"][0]["identity_key"] = "M"
    elif mutation == "used_provenance":
        invalid["replacement"]["camera_floor_point_provenance"] = deepcopy(
            base["rows"][1]["camera_floor_point_provenance"]
        )
    elif mutation == "episode_reuse":
        invalid["replacement"]["episode_id"] = base["rows"][6]["episode_id"]
    errors = TOOL.validate_overlay(invalid, base, rejection, registry)
    assert any(message in error for error in errors)


def test_row7_v2_overlay_rejects_v1_history_rewrite() -> None:
    overlay, base, rejection, registry = _inputs()
    invalid_rejection = deepcopy(rejection)
    invalid_rejection["status"] = "pass"
    invalid_rejection["decision"] = "pass"
    errors = TOOL.validate_overlay(
        overlay, base, invalid_rejection, registry
    )
    assert "v1 must remain rejected" in errors
    assert "v1 decision must remain fail" in errors

    invalid_base = deepcopy(base)
    invalid_base["rows"][1]["episode_id"] = "mutated_unrelated_row"
    revised, _ = TOOL.apply_overlay(invalid_base, overlay)
    assert revised["rows"][1] == invalid_base["rows"][1]

_RETAINED_TMP_WORKSPACE = Path(__file__).resolve().parents[2] / "tmp"
# Guarding on tmp/ existing was wrong: running the engine in a
# checkout creates tmp/spear_instance_*, which made this look
# mounted and sent 49 tests into a run without their data.  The
# evidence mount signature is a lead_* workspace.
if not any(_RETAINED_TMP_WORKSPACE.glob("lead_*")):
    pytest.skip(
        "no lead_* evidence workspace under the repository tmp "
        "directory, so this checkout does not carry the retained "
        "strict-two-human evidence",
        allow_module_level=True,
    )
