from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from avengine.qa.design_viewpoints import (
    DesignViewpointError,
    load_design_viewpoint_bank,
    select_design_viewpoints,
)
from tools.qa import design_qa_batch
from tools.qa import generate_qa_v2_questions


REPOSITORY = Path(__file__).resolve().parents[2]
LEGACY_REQUEST = (
    REPOSITORY / "examples/m6x/fixed_apartment/m1_capture_request_review_720p.json"
)
CORRIDOR_BANK = REPOSITORY / "examples/qa_v2/straight_corridor_bank_v1.json"
REGISTRY = REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"
PROGRAMS = REPOSITORY / "examples/m7/current_apartment/audio_programs/qa_v2"


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
def test_bank_requires_complete_four_to_eight_views(tmp_path: Path, count: int) -> None:
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


def test_explicit_input_modalities_flow_from_design_to_question_mask(
    tmp_path: Path,
) -> None:
    path, _document = _bank_fixture(tmp_path)
    viewpoints = load_design_viewpoint_bank(path)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assets = {item["asset_id"]: item for item in registry["assets"]}
    specs, twins = design_qa_batch.build_specs(
        viewpoints=viewpoints,
        assets=assets,
        cell_size=16,
        twins_per_cell=8,
        input_modalities=("audio", "video"),
    )

    assert all(spec["input_modalities"] == ["audio", "video"] for spec in specs)
    assert all(
        twin["input_modalities"] == source["input_modalities"]
        for twin in twins
        for source in specs
        if source["point_id"] == twin["twin_of"]
    )
    mask = generate_qa_v2_questions.resolve_modality_mask(
        {
            "type_id": "QV2-T2-ATTR",
            "modality_expectation": "dual_required",
        },
        spec=specs[0],
        point_id=specs[0]["point_id"],
        audio_available=True,
    )
    assert mask == {"audio": True, "video": True}

    refusal_mask = generate_qa_v2_questions.resolve_modality_mask(
        {
            "type_id": "QV2-TR-REFUSAL",
            "provided_modalities": ["audio"],
            "withheld_modalities": ["video"],
            "modality_expectation": "must_refuse_audio_only_visual_attribute",
            "refusal_gold": True,
        },
        spec=specs[0],
        point_id=specs[0]["point_id"],
        audio_available=True,
    )
    assert refusal_mask == {"audio": True, "video": False}


@pytest.mark.parametrize(
    "input_modalities",
    (("video",), ("audio", "audio"), ("audio", "text")),
)
def test_design_input_modalities_reject_invalid_facts(
    tmp_path: Path, input_modalities: tuple[str, ...]
) -> None:
    path, _document = _bank_fixture(tmp_path)
    viewpoints = load_design_viewpoint_bank(path)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assets = {item["asset_id"]: item for item in registry["assets"]}

    with pytest.raises(design_qa_batch.DesignBatchError, match="input-modalities"):
        design_qa_batch.build_specs(
            viewpoints=viewpoints,
            assets=assets,
            cell_size=16,
            twins_per_cell=0,
            input_modalities=input_modalities,
        )


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
    assert "--human-start-ue-cm" in command
    assert "--human-end-ue-cm" in command
    assert "--beagle-start-ue-cm" in command
    assert "--beagle-end-ue-cm" in command
    assert "--human-waypoint-ue-cm" not in command
    assert "--beagle-waypoint-ue-cm" not in command


def test_timeline_command_prefers_explicit_spec_waypoints(tmp_path: Path) -> None:
    path, _document = _bank_fixture(tmp_path)
    viewpoint = load_design_viewpoint_bank(path)[0]
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assets = {item["asset_id"]: item for item in registry["assets"]}
    specs, _ = design_qa_batch.build_specs(
        viewpoints=(viewpoint,),
        assets=assets,
        cell_size=16,
        twins_per_cell=0,
    )
    spec = dict(specs[0])
    spec["s1_waypoints_ue_cm"] = [
        [-300.0, 0.0, 27.1],
        [-100.0, 0.0, 27.1],
        [-100.0, 200.0, 27.1],
    ]
    spec["s2_waypoints_ue_cm"] = [
        [200.0, -100.0, 27.1],
        [200.0, 100.0, 27.1],
    ]

    command = design_qa_batch.timeline_author_command(
        avengine_cli="avengine",
        point_directory=tmp_path / "point",
        spec=spec,
        viewpoint=viewpoint,
        registry_path=REGISTRY,
    )

    assert "--human-start-ue-cm" not in command
    assert "--human-end-ue-cm" not in command
    assert "--beagle-start-ue-cm" not in command
    assert "--beagle-end-ue-cm" not in command
    human_indices = [
        index
        for index, value in enumerate(command)
        if value == "--human-waypoint-ue-cm"
    ]
    beagle_indices = [
        index
        for index, value in enumerate(command)
        if value == "--beagle-waypoint-ue-cm"
    ]
    assert [
        [float(item) for item in command[index + 1 : index + 4]]
        for index in human_indices
    ] == spec["s1_waypoints_ue_cm"]
    assert [
        [float(item) for item in command[index + 1 : index + 4]]
        for index in beagle_indices
    ] == spec["s2_waypoints_ue_cm"]


def test_continuous_minimum_separation_catches_between_sample_collision() -> None:
    first = ((-100.0, 0.0, 27.1), (900.0, 0.0, 27.1))
    stationary = ((0.0, 0.0, 27.1), (0.0, 0.0, 27.1))

    assert design_qa_batch.minimum_separation_cm(first, stationary) == pytest.approx(
        0.0
    )


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
    with pytest.raises(
        design_qa_batch.DesignBatchError, match="requires --viewpoint-bank"
    ):
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


def test_motion_case_sequence_covers_both_moving_and_both_static_balanced() -> None:
    sequence = design_qa_batch.motion_case_sequence(16)

    assert len(sequence) == 16
    assert sequence[-1] == "s1_moving_s2_static"
    assert Counter(sequence) == {
        "s1_moving_s2_static": 7,
        "s1_static_s2_moving": 5,
        "both_moving": 2,
        "both_static": 2,
    }


def test_both_static_specs_are_real_zero_length_tracks(tmp_path: Path) -> None:
    path, _document = _bank_fixture(tmp_path)
    viewpoints = load_design_viewpoint_bank(path)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assets = {item["asset_id"]: item for item in registry["assets"]}

    specs, _twins = design_qa_batch.build_specs(
        viewpoints=viewpoints,
        assets=assets,
        cell_size=16,
        twins_per_cell=0,
    )

    assert {spec["motion_case"] for spec in specs} == {
        "s1_moving_s2_static",
        "s1_static_s2_moving",
        "both_moving",
        "both_static",
    }
    both_static = [spec for spec in specs if spec["motion_case"] == "both_static"]
    assert both_static
    assert all(spec["s1_start"] == spec["s1_end"] for spec in both_static)
    assert all(spec["s2_start"] == spec["s2_end"] for spec in both_static)


def test_default_specs_keep_t7_ta_keys_balanced_with_source_identity_twins(
    tmp_path: Path,
) -> None:
    path, _document = _bank_fixture(tmp_path)
    viewpoints = load_design_viewpoint_bank(path)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assets = {item["asset_id"]: item for item in registry["assets"]}
    specs, _twins = design_qa_batch.build_specs(
        viewpoints=viewpoints,
        assets=assets,
        cell_size=16,
        twins_per_cell=8,
    )
    ta_answers = Counter()
    t7_answers = Counter()
    motion_cases_by_type = {"TA": set(), "T7": set()}
    for spec in specs:
        program = json.loads(
            (PROGRAMS / f"{spec['program_id']}.json").read_text(encoding="utf-8")
        )
        endpoints = program["candidate_source_endpoint_ids"]
        endpoint_to_slot = {endpoints[0]: "source1", endpoints[1]: "source2"}
        events = sorted(program["events"], key=lambda event: event["start_sample"])
        first_slot = endpoint_to_slot[events[0]["source_endpoint_id"]]
        second_slot = endpoint_to_slot[events[1]["source_endpoint_id"]]
        movers = {
            "s1_moving_s2_static": {"source1"},
            "s1_static_s2_moving": {"source2"},
            "both_moving": {"source1", "source2"},
            "both_static": set(),
        }[spec["motion_case"]]
        ta_answers["moving" if first_slot in movers else "staying still"] += 1
        t7_answers["moving" if second_slot in movers else "staying still"] += 1
        motion_cases_by_type["TA"].add(spec["motion_case"])
        motion_cases_by_type["T7"].add(spec["motion_case"])

    assert ta_answers == {"moving": 96, "staying still": 96}
    assert t7_answers == {"moving": 96, "staying still": 96}
    assert motion_cases_by_type == {
        "TA": {
            "s1_moving_s2_static",
            "s1_static_s2_moving",
            "both_moving",
            "both_static",
        },
        "T7": {
            "s1_moving_s2_static",
            "s1_static_s2_moving",
            "both_moving",
            "both_static",
        },
    }


def test_point_audio_schedules_vary_both_onsets_and_interval() -> None:
    base = json.loads(
        (PROGRAMS / "qa_v2_two_human_blue_green_turn_taking_v1.json").read_text(
            encoding="utf-8"
        )
    )
    materialized = [
        design_qa_batch.materialize_point_audio_program(
            base,
            point_id=f"b{seed:03d}",
            schedule_seed=seed,
        )
        for seed in range(32)
    ]
    schedules = [item[1] for item in materialized]

    assert len({item["first_event_frame"] for item in schedules}) > 1
    assert len({item["second_event_frame"] for item in schedules}) > 1
    assert len({item["first_second_onset_interval_samples"] for item in schedules}) > 1
    for program, schedule in materialized:
        assert design_qa_batch.validate_audio_program(program) == []
        assert schedule["event_start_samples"] == [
            event["start_sample"] for event in program["events"]
        ]
        for base_event, event in zip(base["events"], program["events"], strict=True):
            assert event["end_sample_exclusive"] - event["start_sample"] == (
                base_event["end_sample_exclusive"] - base_event["start_sample"]
            )
            for field in schedule["unchanged_event_fields"]:
                assert event[field] == base_event[field]
        assert program["events"][2:] == base["events"][2:]


def test_twins_share_exact_schedule_seed_and_event_windows(tmp_path: Path) -> None:
    path, _document = _bank_fixture(tmp_path)
    viewpoints = load_design_viewpoint_bank(path)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assets = {item["asset_id"]: item for item in registry["assets"]}
    specs, twins = design_qa_batch.build_specs(
        viewpoints=viewpoints,
        assets=assets,
        cell_size=16,
        twins_per_cell=8,
    )
    by_id = {spec["point_id"]: spec for spec in specs}
    for twin in twins:
        source = by_id[twin["twin_of"]]
        assert twin["schedule_seed"] == source["schedule_seed"]
        source_program, source_schedule = (
            design_qa_batch.materialize_point_audio_program(
                json.loads((PROGRAMS / f"{source['program_id']}.json").read_text()),
                point_id=source["point_id"],
                schedule_seed=source["schedule_seed"],
            )
        )
        twin_program, twin_schedule = design_qa_batch.materialize_point_audio_program(
            json.loads((PROGRAMS / f"{twin['program_id']}.json").read_text()),
            point_id=twin["point_id"],
            schedule_seed=twin["schedule_seed"],
        )
        assert (
            twin_schedule["event_start_samples"]
            == source_schedule["event_start_samples"]
        )
        assert (
            twin_schedule["event_end_samples_exclusive"]
            == source_schedule["event_end_samples_exclusive"]
        )
        assert [event["source_endpoint_id"] for event in twin_program["events"]] != [
            event["source_endpoint_id"] for event in source_program["events"]
        ]
