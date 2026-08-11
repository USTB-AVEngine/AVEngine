from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = (
    REPO_ROOT / "tools" / "qa" / "build_strict_two_human_camera_pan_v2_candidate.py"
)


def _load_builder():
    sys.path.insert(0, str(BUILDER_PATH.parent))
    spec = importlib.util.spec_from_file_location("camera_pan_v2_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_camera_pan_has_75_unique_yaws_and_required_span() -> None:
    builder = _load_builder()
    yaws = builder.yaw_path()
    assert len(yaws) == 75
    assert len({round(value, 4) for value in yaws}) == 75
    assert max(yaws) - min(yaws) == 6.0
    assert max(yaws) - min(yaws) >= builder.MINIMUM_YAW_SPAN_DEG


def test_both_actors_are_exactly_static() -> None:
    builder = _load_builder()
    target = [builder.TARGET_ROOT[:] for _ in range(builder.FRAME_COUNT)]
    distractor = [builder.DISTRACTOR_ROOT[:] for _ in range(builder.FRAME_COUNT)]
    assert builder.static_drift(target) == 0.0
    assert builder.static_drift(distractor) == 0.0
    assert builder.static_drift(target) <= builder.MAXIMUM_STATIC_DRIFT_M
    assert builder.static_drift(distractor) <= builder.MAXIMUM_STATIC_DRIFT_M


def test_all75_human_envelopes_stay_in_frame_and_on_opposite_sides() -> None:
    builder = _load_builder()
    target = [builder.TARGET_ROOT[:] for _ in range(builder.FRAME_COUNT)]
    distractor = [builder.DISTRACTOR_ROOT[:] for _ in range(builder.FRAME_COUNT)]
    projection = builder.projection_metrics(target, distractor, builder.yaw_path())
    assert projection["target_all75_2m_cylinder_x_fraction_range"][0] > 0.5
    assert projection["distractor_all75_2m_cylinder_x_fraction_range"][1] < 0.5
    assert (
        projection["target_right_midline_dead_zone_fraction"]
        >= builder.MINIMUM_MIDLINE_DEAD_ZONE_FRACTION
    )
    assert (
        projection["distractor_left_midline_dead_zone_fraction"]
        >= builder.MINIMUM_MIDLINE_DEAD_ZONE_FRACTION
    )
    assert (
        projection["minimum_frame_edge_margin_fraction"]
        >= builder.MINIMUM_FRAME_EDGE_MARGIN_FRACTION
    )
    for key in (
        "target_all75_2m_cylinder_x_fraction_range",
        "target_all75_2m_cylinder_y_fraction_range",
        "distractor_all75_2m_cylinder_x_fraction_range",
        "distractor_all75_2m_cylinder_y_fraction_range",
    ):
        assert 0.0 <= projection[key][0] <= projection[key][1] <= 1.0
    assert projection["minimum_actor_horizontal_separation_m"] > 1.4


def test_build_documents_keeps_fresh_pixels_and_gpu_pending(
    monkeypatch,
) -> None:
    builder = _load_builder()

    def fake_load(*_args):
        return object(), object(), {"status": "test_authority"}

    def fake_corridor(roots, *_args):
        return {
            "root_frame_count": len(roots),
            "minimum_depth_clearance_m": 0.4,
            "status": "pass",
        }

    monkeypatch.setattr(builder.base, "load_depth_authority", fake_load)
    monkeypatch.setattr(builder.base, "corridor_metrics", fake_corridor)
    preflight, receipt = builder.build_documents(
        Path("depth.npz"), Path("masks.npz"), Path("readbacks.json")
    )
    row = preflight["canaries"][0]
    assert receipt["candidate_decision"] == "GO_CPU_GEOMETRY_ONLY"
    assert row["mechanism"] == "camera_pan_both_static"
    assert row["target_side"] == "right"
    assert row["target"]["identity_key"] == "M"
    assert row["target"]["source_slot_id"] == "source1"
    assert row["target"]["sound_asset_id"] == "speech_cremad_1001_ieo_neu_v1"
    assert row["target"]["speech_frame_window_inclusive"] == [7, 31]
    assert row["target"]["speech_sample_count"] == 25626
    assert row["distractor"]["identity_key"] == "F"
    assert row["distractor"]["source_slot_id"] == "source2"
    assert row["distractor"]["voice_policy"] == "silent"
    assert row["gpu_launch_authorized"] is False
    assert row["motion_preflight"]["target"]["maximum_root_displacement_m"] == 0.0
    assert row["motion_preflight"]["distractor"]["maximum_root_displacement_m"] == 0.0
    assert row["motion_preflight"]["camera"]["yaw_span_deg"] == 6.0
    assert (
        receipt["strict_native_acceptance_gate"]["status"]
        == "pending_fresh_native_pan_capture"
    )
    assert receipt["acoustic_state_expectation"]["status"] == "not_executed"
    assert receipt["acoustic_state_expectation"]["total_unique_rir_states"] == 150
    assert receipt["dynamic_canary_side_balance_if_accepted"]["counts"] == {
        "left": 2,
        "right": 2,
    }
