from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from avengine.qa.design_viewpoints import (
    DesignViewpointError,
    load_design_viewpoint_bank,
    select_design_viewpoints,
)
from tools.qa import design_qa_batch


REPOSITORY = Path(__file__).resolve().parents[2]
LEGACY_REQUEST = (
    REPOSITORY / "examples/m6x/fixed_apartment/m1_capture_request_review_720p.json"
)
CORRIDOR_BANK = REPOSITORY / "examples/qa_v2/straight_corridor_bank_v1.json"
REGISTRY = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _bank_fixture(tmp_path: Path, *, count: int = 4) -> tuple[Path, dict]:
    base_request = json.loads(LEGACY_REQUEST.read_text(encoding="utf-8"))
    segments = json.loads(CORRIDOR_BANK.read_text(encoding="utf-8"))["segments"]
    evidence = tmp_path / "evidence"
    evidence.mkdir(parents=True)
    viewpoints = []
    for index in range(count):
        request = deepcopy(base_request)
        request["request_id"] = f"authenticated_view_{index}_m1"
        request_path = tmp_path / "requests" / f"view-{index}.json"
        _write_json(request_path, request)
        for kind in ("authentication", "corridor", "offscreen"):
            (evidence / f"view-{index}-{kind}.md").write_text(
                f"view-{index} {kind}: pass\n", encoding="utf-8"
            )
        viewpoints.append(
            {
                "viewpoint_id": f"view-{index}",
                "native_pixel_validation_status": "pass",
                "authentication": {
                    "status": "pass",
                    "evidence_path": f"evidence/view-{index}-authentication.md",
                },
                "camera": {
                    "ue_location_cm": [-70.0, 65.0, 147.1],
                    "ue_yaw_deg": -145.0,
                },
                "m1_request": {
                    "status": "pass",
                    "path": f"requests/view-{index}.json",
                },
                "corridor_subset": {
                    "status": "pass",
                    "evidence_path": f"evidence/view-{index}-corridor.md",
                    "segments": segments,
                },
                "offscreen_evidence": {
                    "status": "pass",
                    "evidence_path": f"evidence/view-{index}-offscreen.md",
                    "point_ue_cm": [-152.0 - index, 38.0, 27.1],
                },
            }
        )
    document = {
        "native_pixel_validation_status": "pass",
        "viewpoints": viewpoints,
    }
    path = tmp_path / "viewpoint-bank.json"
    _write_json(path, document)
    return path, document


def test_pending_prefilter_is_explicitly_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prefilter.json"
    _write_json(
        path,
        {
            "artifact_status": "deterministic_2d_raster_prefilter_only",
            "native_pixel_validation_status": "pending",
            "viewpoints": [],
        },
    )

    with pytest.raises(
        DesignViewpointError,
        match="native_pixel_validation_status=pending.*cannot be used",
    ):
        load_design_viewpoint_bank(path)


@pytest.mark.parametrize("count", [3, 9])
def test_bank_requires_complete_four_to_eight_views(
    tmp_path: Path, count: int
) -> None:
    path, _document = _bank_fixture(tmp_path, count=count)
    with pytest.raises(DesignViewpointError, match="complete 4-8 viewpoints"):
        load_design_viewpoint_bank(path)


def test_bank_rejects_duplicate_id_and_missing_evidence(tmp_path: Path) -> None:
    path, document = _bank_fixture(tmp_path)
    document["viewpoints"][1]["viewpoint_id"] = "view-0"
    _write_json(path, document)
    with pytest.raises(DesignViewpointError, match="duplicate viewpoint_id"):
        load_design_viewpoint_bank(path)

    path, document = _bank_fixture(tmp_path / "missing")
    document["viewpoints"][0]["offscreen_evidence"]["evidence_path"] = "missing.md"
    _write_json(path, document)
    with pytest.raises(DesignViewpointError, match="evidence_path does not resolve"):
        load_design_viewpoint_bank(path)


def test_bank_camera_must_match_corresponding_m1_request(tmp_path: Path) -> None:
    path, document = _bank_fixture(tmp_path)
    document["viewpoints"][0]["camera"]["ue_location_cm"][0] += 1.0
    _write_json(path, document)

    with pytest.raises(DesignViewpointError, match="does not match its M1 request"):
        load_design_viewpoint_bank(path)


def test_rotation_uses_each_view_corridors_offscreen_and_m1(tmp_path: Path) -> None:
    path, _document = _bank_fixture(tmp_path)
    viewpoints = load_design_viewpoint_bank(path)
    assert len(viewpoints[0].corridors) == len(
        {
            tuple(sorted((corridor.start_ue_cm, corridor.end_ue_cm)))
            for corridor in viewpoints[0].corridors
        }
    )
    assert len(viewpoints[0].corridors) < len(
        json.loads(CORRIDOR_BANK.read_text(encoding="utf-8"))["segments"]
    )
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assets = {item["asset_id"]: item for item in registry["assets"]}

    specs, twins = design_qa_batch.build_specs(
        viewpoints=viewpoints,
        assets=assets,
        cell_size=16,
        twins_per_cell=8,
    )

    assert len(specs) == 192
    assert len(twins) == 64
    assert [item["viewpoint_id"] for item in specs[:5]] == [
        "view-0",
        "view-1",
        "view-2",
        "view-3",
        "view-0",
    ]
    first_offscreen = next(item for item in specs if item["point_id"] == "b016")
    assert first_offscreen["viewpoint_id"] == "view-3"
    assert first_offscreen["s2_start"] == [-155.0, 38.0, 27.1]
    assert first_offscreen["s2_end"] == [-155.0, 38.0, 27.1]
    first_twin = next(item for item in twins if item["point_id"] == "bt001")
    assert first_twin["twin_of"] == "b001"
    assert first_twin["viewpoint_id"] == specs[0]["viewpoint_id"]
    assert first_twin["m1_request"] == specs[0]["m1_request"]
    assert specs[0]["viewpoint_authority"] == "declared_certified_viewpoint_input"

    explicit = select_design_viewpoints(viewpoints, "view-2")
    explicit_specs, _ = design_qa_batch.build_specs(
        viewpoints=explicit,
        assets=assets,
        cell_size=16,
        twins_per_cell=0,
    )
    assert {item["viewpoint_id"] for item in explicit_specs} == {"view-2"}


def test_side_and_timeline_camera_are_viewpoint_derived(tmp_path: Path) -> None:
    path, document = _bank_fixture(tmp_path)
    opposite_request = json.loads(
        (tmp_path / "requests/view-1.json").read_text(encoding="utf-8")
    )
    opposite_request["primary_camera_rig"]["world_from_rig"]["rotation_xyzw"] = [
        0.0,
        -0.8870108331782217,
        0.0,
        0.4617486132350339,
    ]
    _write_json(tmp_path / "requests/view-1.json", opposite_request)
    document["viewpoints"][1]["camera"]["ue_yaw_deg"] = 35.0
    _write_json(path, document)
    viewpoints = load_design_viewpoint_bank(path)
    point = (-200.0, 200.0, 27.1)
    assert viewpoints[0].side(point)[0] != viewpoints[1].side(point)[0]

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assets = {item["asset_id"]: item for item in registry["assets"]}
    specs, _ = design_qa_batch.build_specs(
        viewpoints=(viewpoints[1],),
        assets=assets,
        cell_size=16,
        twins_per_cell=0,
    )
    command = design_qa_batch.timeline_author_command(
        avengine_cli="avengine",
        point_directory=tmp_path / "point",
        spec=specs[0],
        viewpoint=viewpoints[1],
        registry_path=REGISTRY,
    )
    camera_index = command.index("--camera-position-ue-cm")
    yaw_index = command.index("--camera-yaw-deg")
    assert [float(item) for item in command[camera_index + 1 : camera_index + 4]] == (
        pytest.approx([-70.0, 65.0, 147.1])
    )
    assert float(command[yaw_index + 1]) == pytest.approx(35.0)
    assert specs[0]["m1_request"]["request_id"] == "authenticated_view_1_m1"


def test_continuous_minimum_separation_catches_between_sample_collision() -> None:
    first = ((-100.0, 0.0, 27.1), (900.0, 0.0, 27.1))
    stationary = ((0.0, 0.0, 27.1), (0.0, 0.0, 27.1))

    assert design_qa_batch.minimum_separation_cm(first, stationary) == pytest.approx(0.0)


def test_cell_size_rejects_overlapping_motion_composition(tmp_path: Path) -> None:
    path, _document = _bank_fixture(tmp_path)
    viewpoints = load_design_viewpoint_bank(path)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assets = {item["asset_id"]: item for item in registry["assets"]}

    with pytest.raises(design_qa_batch.DesignBatchError, match="at least 13"):
        design_qa_batch.build_specs(
            viewpoints=viewpoints,
            assets=assets,
            cell_size=12,
            twins_per_cell=0,
        )


def test_production_has_no_silent_legacy_fallback(tmp_path: Path) -> None:
    args = design_qa_batch.build_parser().parse_args(
        ["--output-root", str(tmp_path / "output")]
    )
    with pytest.raises(design_qa_batch.DesignBatchError, match="requires --viewpoint-bank"):
        design_qa_batch._resolve_viewpoints(args)

    legacy_args = design_qa_batch.build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path / "legacy-output"),
            "--legacy-view0-report-only",
            "--legacy-offscreen-point-ue-cm",
            "-152",
            "38",
            "27.1",
        ]
    )
    legacy = design_qa_batch._resolve_viewpoints(legacy_args)
    assert len(legacy) == 1
    assert legacy[0].viewpoint_id == "apartment_view0"
    assert legacy[0].authority == "legacy_view0_report_only"
