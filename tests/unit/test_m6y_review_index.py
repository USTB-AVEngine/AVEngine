from pathlib import Path

from tools.m6y.build_review_index import _replicacad_section, build_page


def _media(name: str) -> dict[str, object]:
    return {
        "path": name,
        "status": "pass",
        "width": 1280,
        "height": 720,
        "frame_count": 270,
    }


def test_lighting_review_reports_corrected_profiles(tmp_path: Path) -> None:
    habitat_apartment = (
        tmp_path / "habitat_apartment",
        {
            "status": "pass",
            "clip": {"frame_count": 75},
            "review_visual_profile": {
                "profile_id": "spear_apartment_habitat_review_720p_natural_v3"
            },
            "variants": [
                {
                    "scenario_id": "S3",
                    "variant_id": "A",
                    "status": "pass",
                    "clean_video": "clean.mp4",
                    "diagnostic_video": "topdown.mp4",
                }
            ],
        },
    )
    apartment = (
        tmp_path / "ue_apartment",
        {
            "status": "pass",
            "scenarios": [
                {"scenario_id": item, "status": "pass", "media": {}}
                for item in ("S0", "S3", "S4")
            ],
        },
    )
    mp3d = (
        tmp_path / "mp3d",
        {
            "status": "pass",
            "clock": {"frame_count": 270, "timeline_v2_applicable": False},
            "exposure_qa": {"status": "pass"},
            "color_fidelity_qa": {
                "status": "pass",
                "mean_chroma_ratio_ue_to_habitat": 1.006,
            },
            "media": {"ue_clean_binaural": _media("mp3d.mp4")},
        },
    )
    replicacad = (
        tmp_path / "replicacad",
        {
            "status": "pass",
            "clock": {"frame_count": 270, "timeline_v2_applicable": False},
            "exposure_qa": {"status": "pass"},
            "runtime": {
                "scene_and_lighting_readback": {
                    "tagged_comparison_visual_actor_count": 171,
                    "active_positive_point_light_count": 3,
                    "lighting_profile_application": {
                        "active_positive_light_ids": ["0", "1", "2"],
                        "excluded_positive_light_ids": ["3", "4"],
                        "ue_intensity_scale": 2.0,
                    },
                }
            },
            "media": {"ue_clean_binaural": _media("replicacad.mp4")},
            "claim_boundary": "bounded comparison",
        },
    )

    page = build_page(
        habitat_apartment=habitat_apartment,
        apartment=apartment,
        mp3d=mp3d,
        replicacad=replicacad,
        output=tmp_path / "REVIEW_INDEX.html",
    )

    assert "natural_v3" in page
    assert "mean-chroma ratio 1.006" in page
    assert "white/grey result was an import-color-space error" in page
    assert "active room-local lights (IDs 0, 1, 2; excluded 3, 4" in page
    assert "no_lights + HBAO" in page


def test_replicacad_review_reports_generated_fill_without_claiming_source_light(
    tmp_path: Path,
) -> None:
    root = tmp_path / "replicacad"
    value = {
        "status": "pass",
        "runtime": {
            "scene_and_lighting_readback": {
                "active_positive_point_light_count": 3,
                "generated_review_point_light_count": 1,
                "lighting_profile_application": {
                    "active_positive_light_ids": ["0", "1", "2"],
                    "excluded_positive_light_ids": ["3", "4"],
                    "ue_intensity_scale": 1.0,
                    "generated_interior_fill": {"intensity_lumens": 1400.0},
                },
            }
        },
        "media": {},
    }

    section = _replicacad_section((root, value), tmp_path / "REVIEW_INDEX.html")

    assert "one neutral route-center ceiling fill at 1400.0 lm" in section
    assert "not a dataset-authored light or acoustic truth" in section
