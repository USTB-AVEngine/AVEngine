import numpy as np
import pytest

from avengine_v43.labels import (
    LegacyV4AudioError,
    caption_for_asset,
    circular_gaussian_targets,
    deterministic_split_samples,
    legacy_rows_for_native_360_bins,
    native_azimuth_to_bin360,
    native_to_legacy_azimuth_deg,
    resample_position_track,
)


def test_native_to_legacy_mapping_exposes_front_back_fold():
    native = np.asarray([0.0, 90.0, -90.0, 180.0, -180.0])
    assert native_to_legacy_azimuth_deg(native).tolist() == [
        90.0,
        0.0,
        180.0,
        90.0,
        90.0,
    ]


def test_native_360_bins_and_legacy_head_lift_are_explicit():
    assert native_azimuth_to_bin360(
        np.asarray([0.0, 90.0, -90.0, 180.0, -180.0])
    ).tolist() == [0.0, 90.0, 270.0, 180.0, 180.0]
    rows = legacy_rows_for_native_360_bins()
    assert rows.shape == (360,)
    assert [int(rows[index]) for index in (0, 90, 180, 270)] == [
        90,
        0,
        90,
        179,
    ]
    assert rows[0] == rows[180]


def test_circular_gaussian_wraps_across_zero_degrees():
    targets = circular_gaussian_targets(np.asarray([359.0]), sigma_deg=2.0)
    assert targets.shape == (1, 360)
    assert targets[0, 359] == pytest.approx(1.0)
    assert targets[0, 0] == pytest.approx(targets[0, 358])
    assert targets[0, 180] < 1.0e-20


def test_position_crop_uses_only_first_four_seconds():
    source = np.column_stack(
        [
            np.arange(75, dtype=np.float64) / 15.0,
            np.ones(75),
            np.zeros(75),
        ]
    )
    sampled = resample_position_track(
        source,
        source_frame_rate_hz=15.0,
        target_duration_seconds=4.0,
        target_frame_count=75,
    )
    assert sampled.shape == (75, 3)
    assert sampled[0, 0] == pytest.approx(0.0)
    assert sampled[-1, 0] == pytest.approx(74 * 4.0 / 75.0)
    assert sampled[-1, 0] < source[-1, 0]


def test_split_selection_is_ordered_and_never_random():
    index = {
        "status": "pass",
        "samples": [
            {"sample_id": "train0", "split": "train"},
            {"sample_id": "test0", "split": "test"},
            {"sample_id": "test1", "split": "test"},
            {"sample_id": "test2", "split": "test"},
        ],
    }
    assert [
        value["sample_id"]
        for value in deterministic_split_samples(
            index,
            split="test",
            offset=1,
            limit=2,
        )
    ] == ["test1", "test2"]


def test_caption_mapping_is_asset_family_scoped_and_unknown_fails():
    assert (
        caption_for_asset(
            "generated_abyssinian_ruddy_medium_standard_adult_research_v1"
        )
        == "cat meowing"
    )
    assert (
        caption_for_asset(
            "generated_border_collie_black_white_medium_standard_adult_research_v1"
        )
        == "dog barking"
    )
    assert (
        caption_for_asset("rocketbox_human_male_adult_01_m5_1_candidate")
        == "human speech"
    )
    with pytest.raises(LegacyV4AudioError, match="caption"):
        caption_for_asset("unknown_asset")
