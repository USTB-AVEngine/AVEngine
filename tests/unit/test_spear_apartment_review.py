import json

import numpy as np
import pytest

from tools.dataset.build_spear_apartment_review import (
    SpearApartmentReviewError,
    _validate_visual_binding,
    source_center_paths_from_spec,
    ssot_points_to_habitat,
)


def test_apartment_ssot_transform_matches_retained_listener():
    transformed = ssot_points_to_habitat([[0.5, 0.15, 1.2]])

    np.testing.assert_allclose(transformed[0], [-0.7, 1.471, 0.65])


def test_source_paths_remain_generic_and_apply_each_emitter_height():
    spec = {
        "sources": [
            {
                "trajectory_m": [[-1.5, 1.3, 0.0], [-1.6, 1.4, 0.0]],
                "audio_source_height_offset_m": 1.55,
            },
            {
                "trajectory_m": [[-3.2, 1.8, 0.0], [-3.1, 1.7, 0.0]],
                "audio_source_height_offset_m": 1.56,
            },
        ]
    }

    paths = source_center_paths_from_spec(spec)

    assert tuple(paths) == ("source1", "source2")
    np.testing.assert_allclose(paths["source1"][0], [-2.7, 1.821, -0.5])
    np.testing.assert_allclose(paths["source2"][0], [-4.4, 1.831, -1.0])


def test_visual_binding_rejects_wrong_actor_combination(tmp_path):
    spec = {
        "sources": [
            {"tag": "rocketbox_male"},
            {"tag": "rocketbox_female"},
        ]
    }
    metadata = {
        "capture_warmup": {"status": "passed"},
        "sources": [
            {"tag": "rocketbox_male"},
            {"tag": "labrador_wrong_asset"},
        ],
        "rig_direction_evidence": {
            "rocketbox_male": {"status": "passed"},
            "labrador_wrong_asset": {"status": "passed"},
        },
    }
    path = tmp_path / "actor_visual_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(
        SpearApartmentReviewError, match="UE runtime source tags differ"
    ):
        _validate_visual_binding(spec, path)
