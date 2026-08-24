from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from avengine.m5_1.source_contracts import sample_boundary
from avengine.m6.audio_program import bind_audio_program_hash


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY / "tools/qa/generate_qa_v2_questions.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "generate_qa_v2_questions", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    registry = tmp_path / "registry.json"
    _write_json(
        registry,
        {
            "assets": [
                {
                    "asset_id": "blue_asset",
                    "realized_attributes": {"top_color": "blue"},
                    "identity": {"breed_id": None},
                },
                {
                    "asset_id": "green_asset",
                    "realized_attributes": {"top_color": "green"},
                    "identity": {"breed_id": None},
                },
            ]
        },
    )
    programs = tmp_path / "programs"
    _write_json(
        programs / "p.json",
        {
            "candidate_source_endpoint_ids": ["source1-mouth", "source2-mouth"],
            "events": [
                {
                    "source_endpoint_id": "source1-mouth",
                    "start_sample": 4000,
                    "end_sample_exclusive": 18000,
                },
                {
                    "source_endpoint_id": "source2-mouth",
                    "start_sample": 24000,
                    "end_sample_exclusive": 38000,
                },
            ],
        },
    )
    viewpoint_bank = tmp_path / "viewpoints.json"
    _write_json(
        viewpoint_bank,
        {
            "viewpoints": [
                {
                    "viewpoint_id": "view-left",
                    "camera": {
                        "ue_location_cm": [0.0, 0.0, 100.0],
                        "ue_yaw_deg": 0.0,
                    },
                },
                {
                    "viewpoint_id": "view-reverse",
                    "camera": {
                        "ue_location_cm": [0.0, 0.0, 120.0],
                        "yaw": 180.0,
                    },
                },
            ]
        },
    )
    inputs = tmp_path / "inputs"
    point = inputs / "p001"
    _write_json(
        point / "spec.json",
        {
            "pair_kind": "human",
            "source1_asset": "blue_asset",
            "source2_asset": "green_asset",
            "motion_case": "s1_moving_s2_static",
            "twin_of": None,
            "offscreen_candidate": False,
            "program_id": "p",
            "viewpoint_id": "view-left",
        },
    )
    frames = []
    for _frame_index in range(75):
        frames.append(
            {
                "actor_states": [
                    {
                        "source_slot_id": "source1",
                        "action_id": "walk",
                        "translation_ue_cm": [10.0, -10.0, 27.1],
                    },
                    {
                        "source_slot_id": "source2",
                        "action_id": "idle",
                        "translation_ue_cm": [40.0, 40.0, 27.1],
                    },
                ]
            }
        )
    _write_json(point / "timeline.json", {"frames": frames})
    return inputs, programs, registry, viewpoint_bank


def _duplicate_point(inputs: Path, *, viewpoint_id: str = "view-left") -> None:
    source = inputs / "p001"
    duplicate = inputs / "p002"
    duplicate.mkdir()
    spec = json.loads((source / "spec.json").read_text(encoding="utf-8"))
    spec["viewpoint_id"] = viewpoint_id
    _write_json(duplicate / "spec.json", spec)
    (duplicate / "timeline.json").write_bytes((source / "timeline.json").read_bytes())


def _write_native_capture(root: Path, point_id: str) -> None:
    frames = []
    for index in range(75):
        anchors = {
            "source1": {"location_cm": [10.0, -10.0, 27.1]},
            "source2": {"location_cm": [40.0, 40.0, 27.1]},
        }
        frames.append(
            {
                "frame_index": index,
                "pts_ticks": index * 3200,
                "camera": {
                    "translation_ue_cm": [0.0, 0.0, 100.0],
                    "yaw_ue_deg": 0.0,
                },
                "camera_pose": {
                    "location_cm": [0.0, 0.0, 100.0],
                    "rotation_deg": [0.0, 0.0, 0.0],
                },
                "actor_anchor_poses": anchors,
                "actor_states": [
                    {
                        "source_slot_id": slot,
                        "translation_ue_cm": record["location_cm"],
                    }
                    for slot, record in anchors.items()
                ],
            }
        )
    point = root / point_id
    _write_json(point / "frame_records.json", {"frames": frames})
    _write_json(
        point / "research_receipt.json",
        {
            "research_only": True,
            "episode_counted": False,
            "capture": {
                "completed_frame_count": 75,
                "frame_count": 75,
                "frame_rate_hz": 15,
                "ticks_per_frame": 3200,
                "root_readback_summary": {
                    "camera": {"status": "pass"},
                    "source1_actor": {"status": "pass"},
                    "source2_actor": {"status": "pass"},
                },
            },
        },
    )


def _write_pixel_truth(root: Path, point_id: str) -> None:
    def source_frames(slot: str):
        rows = []
        for index in range(75):
            out = slot == "source1" and 4 <= index <= 17
            rows.append(
                {
                    "frame_index": index,
                    "state": "out_of_view" if out else "visible_clear",
                    "visible_pixels": 0 if out else 100,
                    "target_pixels": 0 if out else 100,
                }
            )
        return rows

    _write_json(
        root / point_id / "pixel_visibility_truth.json",
        {
            "schema": "avengine_qa_pixel_visibility_truth_v1",
            "status": "computed_modal_target_only_v1",
            "frame_indices": list(range(75)),
            "resolution_hw": [720, 1280],
            "per_instance": {
                "source1": {"frames": source_frames("source1")},
                "source2": {"frames": source_frames("source2")},
            },
        },
    )


def _cli_args(
    *,
    inputs: Path,
    programs: Path,
    registry: Path,
    output: Path,
    viewpoint_bank: Path | None,
) -> list[str]:
    args = [
        "--inputs-root",
        str(inputs),
        "--programs-dir",
        str(programs),
        "--registry",
        str(registry),
        "--output",
        str(output),
    ]
    if viewpoint_bank is not None:
        args.extend(("--viewpoint-bank", str(viewpoint_bank)))
    return args


def test_report_only_is_required_for_legacy_global_camera(tmp_path: Path) -> None:
    module = _load_script()
    inputs, programs, registry, _viewpoint_bank = _fixture(tmp_path)

    legacy_yaw = module.generate_question_document(
        inputs_root=inputs,
        programs_dir=programs,
        registry_path=registry,
        report_only=True,
        camera_position_ue_cm=(0.0, 0.0, 100.0),
        camera_yaw_deg=0.0,
        generated_at="fixed",
    )
    reverse_yaw = module.generate_question_document(
        inputs_root=inputs,
        programs_dir=programs,
        registry_path=registry,
        report_only=True,
        camera_position_ue_cm=(0.0, 0.0, 100.0),
        camera_yaw_deg=180.0,
        generated_at="fixed",
    )

    legacy_side = next(
        question
        for question in legacy_yaw["questions"]
        if question["type_id"] == "QV2-T4-SIDE"
    )
    reverse_side = next(
        question
        for question in reverse_yaw["questions"]
        if question["type_id"] == "QV2-T4-SIDE"
    )
    assert legacy_side["answer"] == "left"
    assert reverse_side["answer"] == "right"
    assert legacy_side["viewpoint_id"] == "view-left"
    assert legacy_yaw["viewpoint_resolution"]["points"]["p001"] == {
        "viewpoint_id": "view-left",
        "camera_authority": "legacy_global_pose_report_only",
        "position_ue_cm": [0.0, 0.0, 100.0],
        "yaw_ue_deg": 0.0,
    }
    with pytest.raises(module.QuestionGenerationError, match="viewpoint-bank"):
        module.generate_question_document(
            inputs_root=inputs,
            programs_dir=programs,
            registry_path=registry,
        )


def test_mixed_viewpoints_resolve_per_point_and_are_written_on_every_question(
    tmp_path: Path,
) -> None:
    module = _load_script()
    inputs, programs, registry, viewpoint_bank = _fixture(tmp_path)
    _duplicate_point(inputs, viewpoint_id="view-reverse")

    document = module.generate_question_document(
        inputs_root=inputs,
        programs_dir=programs,
        registry_path=registry,
        viewpoint_bank_path=viewpoint_bank,
        generated_at="fixed",
    )

    by_point = {}
    for question in document["questions"]:
        by_point.setdefault(question["point_id"], set()).add(question["viewpoint_id"])
        assert question["answer_arity"] == "binary"
    assert by_point == {"p001": {"view-left"}, "p002": {"view-reverse"}}
    side_answers = {
        question["point_id"]: question["answer"]
        for question in document["questions"]
        if question["type_id"] == "QV2-T4-SIDE"
    }
    assert side_answers == {"p001": "left", "p002": "right"}
    points = document["viewpoint_resolution"]["points"]
    assert points["p001"]["position_ue_cm"] == [0.0, 0.0, 100.0]
    assert points["p002"]["position_ue_cm"] == [0.0, 0.0, 120.0]


def test_default_cli_is_strict_without_legacy_require_flag(tmp_path: Path) -> None:
    module = _load_script()
    inputs, programs, registry, viewpoint_bank = _fixture(tmp_path)
    output = tmp_path / "missing" / "out"
    args = _cli_args(
        inputs=inputs,
        programs=programs,
        registry=registry,
        output=output,
        viewpoint_bank=viewpoint_bank,
    )

    assert module.main(args) == 0
    document = json.loads((output / "questions.json").read_text(encoding="utf-8"))
    assert document["answer_balance"]["status"] == "pass"
    assert document["viewpoint_resolution"]["mode"] == "strict_production"
    assert module.main(args) == 2


def test_default_strict_rejects_imbalance_before_output_and_report_only_allows_audit(
    tmp_path: Path,
) -> None:
    module = _load_script()
    inputs, programs, registry, viewpoint_bank = _fixture(tmp_path)
    _duplicate_point(inputs)
    strict_output = tmp_path / "strict-failure"
    strict_args = _cli_args(
        inputs=inputs,
        programs=programs,
        registry=registry,
        output=strict_output,
        viewpoint_bank=viewpoint_bank,
    )

    assert module.main(strict_args) == 2
    assert not strict_output.exists()

    report_output = tmp_path / "report-only"
    report_args = _cli_args(
        inputs=inputs,
        programs=programs,
        registry=registry,
        output=report_output,
        viewpoint_bank=None,
    )
    report_args.append("--report-only")
    assert module.main(report_args) == 0
    report = json.loads((report_output / "questions.json").read_text(encoding="utf-8"))
    assert report["answer_balance"]["status"] == "fail"


def test_strict_requires_every_spec_id_to_cross_reference_bank(tmp_path: Path) -> None:
    module = _load_script()
    inputs, programs, registry, viewpoint_bank = _fixture(tmp_path)
    spec_path = inputs / "p001/spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["viewpoint_id"] = "not-in-bank"
    _write_json(spec_path, spec)
    output = tmp_path / "bad-viewpoint"

    assert (
        module.main(
            _cli_args(
                inputs=inputs,
                programs=programs,
                registry=registry,
                output=output,
                viewpoint_bank=viewpoint_bank,
            )
        )
        == 2
    )
    assert not output.exists()


def test_atomic_writer_removes_staging_on_serialization_failure(tmp_path: Path) -> None:
    module = _load_script()
    output = tmp_path / "new-parent" / "out"

    with pytest.raises(TypeError):
        module.write_question_document({"not_json": object()}, output)

    assert not output.exists()
    assert list(output.parent.glob(f".{output.name}.staging-*")) == []


def test_atomic_writer_loses_no_replace_race_without_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    output = tmp_path / "race-output"
    original_publish = module.atomic_publish_directory

    def racing_publish(policy, staging, destination):
        destination.mkdir()
        (destination / "sentinel").write_text("keep", encoding="utf-8")
        return original_publish(policy, staging, destination)

    monkeypatch.setattr(module, "atomic_publish_directory", racing_publish)
    with pytest.raises(FileExistsError):
        module.write_question_document({"ok": True}, output)

    assert (output / "sentinel").read_text(encoding="utf-8") == "keep"
    assert list(tmp_path.glob(f".{output.name}.staging-*")) == []


def test_t5_and_offscreen_use_native_ue_and_pixel_evidence(tmp_path: Path) -> None:
    module = _load_script()
    inputs, programs, registry, viewpoint_bank = _fixture(tmp_path)
    spec_path = inputs / "p001/spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["offscreen_candidate"] = True
    _write_json(spec_path, spec)
    captures = tmp_path / "captures"
    visibility = tmp_path / "visibility"
    _write_native_capture(captures, "p001")
    _write_pixel_truth(visibility, "p001")

    document = module.generate_question_document(
        inputs_root=inputs,
        programs_dir=programs,
        registry_path=registry,
        viewpoint_bank_path=viewpoint_bank,
        native_capture_root=captures,
        pixel_visibility_root=visibility,
        enable_t5_doa=True,
        enable_offscreen=True,
        generated_at="fixed",
    )

    t5 = next(q for q in document["questions"] if q["type_id"] == "QV2-T5-DOA")
    assert t5["answer"] == pytest.approx(45.0)
    assert t5["answer_arity"] == "numeric_banded"
    assert t5["format"] == "numeric_banded"
    assert t5["answer_numeric"] == pytest.approx(45.0)
    assert t5["scoring_bands"] == [15.0, 30.0]
    assert t5["scoring"]["full_credit_within_deg"] == 15.0
    assert t5["scoring"]["partial_credit_within_deg"] == 30.0
    assert t5["gold_authority"] == "ue_actor_anchor_and_camera_pose_readback"

    offscreen = next(
        q for q in document["questions"] if q["type_id"] == "QV2-TO-OFFSCREEN"
    )
    assert offscreen["answer"] == "left"
    assert offscreen["event_frame"] == 4
    assert offscreen["native_pixel_state"] == "out_of_view_entire_event"
    assert offscreen["event_frame_window"] == list(range(4, 18))
    assert offscreen["native_target_pixels_max"] == 0
    assert document["answer_balance"]["by_type"]["QV2-TO-OFFSCREEN"]["status"] == "pass"


def test_refusal_questions_meet_ten_to_fifteen_percent_without_deleting_base(
    tmp_path: Path,
) -> None:
    module = _load_script()
    inputs, programs, registry, viewpoint_bank = _fixture(tmp_path)
    _duplicate_point(inputs)
    source = inputs / "p001"
    third = inputs / "p003"
    third.mkdir()
    (third / "spec.json").write_bytes((source / "spec.json").read_bytes())
    (third / "timeline.json").write_bytes((source / "timeline.json").read_bytes())

    baseline = module.generate_question_document(
        inputs_root=inputs,
        programs_dir=programs,
        registry_path=registry,
        viewpoint_bank_path=viewpoint_bank,
        generated_at="fixed",
    )
    document = module.generate_question_document(
        inputs_root=inputs,
        programs_dir=programs,
        registry_path=registry,
        viewpoint_bank_path=viewpoint_bank,
        enable_refusal=True,
        generated_at="fixed",
    )

    refusal_rows = [
        q for q in document["questions"] if q["type_id"] == "QV2-TR-REFUSAL"
    ]
    assert len(document["questions"]) - len(refusal_rows) == len(baseline["questions"])
    assert all(
        row["answer"] == "cannot determine from audio alone" for row in refusal_rows
    )
    assert all(row["provided_modalities"] == ["audio"] for row in refusal_rows)
    quota = document["extended_types"]["refusal_quota"]
    assert quota["status"] == "pass"
    assert 0.10 <= quota["final_fraction"] <= 0.15
    assert len(refusal_rows) == quota["selected_count"]


def test_extended_types_fail_closed_without_native_roots(tmp_path: Path) -> None:
    module = _load_script()
    inputs, programs, registry, viewpoint_bank = _fixture(tmp_path)

    with pytest.raises(module.QuestionGenerationError, match="native-capture-root"):
        module.generate_question_document(
            inputs_root=inputs,
            programs_dir=programs,
            registry_path=registry,
            viewpoint_bank_path=viewpoint_bank,
            enable_t5_doa=True,
        )
    with pytest.raises(module.QuestionGenerationError, match="pixel-visibility-root"):
        module.generate_question_document(
            inputs_root=inputs,
            programs_dir=programs,
            registry_path=registry,
            viewpoint_bank_path=viewpoint_bank,
            native_capture_root=tmp_path / "captures",
            enable_offscreen=True,
        )


@pytest.mark.parametrize(
    "program_update",
    [
        {"events": []},
        {"candidate_source_endpoint_ids": ["source1-mouth"]},
        {"events": [{"source_endpoint_id": [], "start_sample": 0}, {}]},
    ],
)
def test_malformed_audio_program_fails_closed(
    tmp_path: Path,
    program_update: dict[str, object],
) -> None:
    module = _load_script()
    inputs, programs, registry, viewpoint_bank = _fixture(tmp_path)
    program_path = programs / "p.json"
    program = json.loads(program_path.read_text(encoding="utf-8"))
    program.update(program_update)
    _write_json(program_path, program)

    with pytest.raises(module.QuestionGenerationError, match="audio"):
        module.generate_question_document(
            inputs_root=inputs,
            programs_dir=programs,
            registry_path=registry,
            viewpoint_bank_path=viewpoint_bank,
        )


def test_non_object_audio_program_fails_closed(tmp_path: Path) -> None:
    module = _load_script()
    inputs, programs, registry, viewpoint_bank = _fixture(tmp_path)
    _write_json(programs / "p.json", [])

    with pytest.raises(module.QuestionGenerationError, match="must be an object"):
        module.generate_question_document(
            inputs_root=inputs,
            programs_dir=programs,
            registry_path=registry,
            viewpoint_bank_path=viewpoint_bank,
        )


def test_t7_and_ta_cover_all_four_motion_cases_with_balanced_keys(
    tmp_path: Path,
) -> None:
    module = _load_script()
    inputs, programs, registry, viewpoint_bank = _fixture(tmp_path)
    source = inputs / "p001"
    modes = (
        "s1_moving_s2_static",
        "s1_static_s2_moving",
        "both_moving",
        "both_static",
    )
    for index, mode in enumerate(modes, start=1):
        point = inputs / f"p{index:03d}"
        if point != source:
            point.mkdir()
            (point / "timeline.json").write_bytes(
                (source / "timeline.json").read_bytes()
            )
        spec = json.loads((source / "spec.json").read_text(encoding="utf-8"))
        spec["motion_case"] = mode
        _write_json(point / "spec.json", spec)
        timeline_path = point / "timeline.json"
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        movers = {
            "s1_moving_s2_static": {"source1"},
            "s1_static_s2_moving": {"source2"},
            "both_moving": {"source1", "source2"},
            "both_static": set(),
        }[mode]
        for frame in timeline["frames"]:
            for state in frame["actor_states"]:
                state["action_id"] = (
                    "walk" if state["source_slot_id"] in movers else "idle"
                )
        _write_json(timeline_path, timeline)

    document = module.generate_question_document(
        inputs_root=inputs,
        programs_dir=programs,
        registry_path=registry,
        viewpoint_bank_path=viewpoint_bank,
        generated_at="fixed",
    )
    expected_ta = {
        "s1_moving_s2_static": "moving",
        "s1_static_s2_moving": "staying still",
        "both_moving": "moving",
        "both_static": "staying still",
    }
    expected_t7 = {
        "s1_moving_s2_static": "staying still",
        "s1_static_s2_moving": "moving",
        "both_moving": "moving",
        "both_static": "staying still",
    }
    ta = {
        row["motion_case"]: row["answer"]
        for row in document["questions"]
        if row["type_id"] == "QV2-TA-MOTION"
    }
    t7 = {
        row["motion_case"]: row["answer"]
        for row in document["questions"]
        if row["type_id"] == "QV2-T7-DURING"
    }
    assert ta == expected_ta
    assert t7 == expected_t7
    assert document["answer_balance"]["by_type"]["QV2-TA-MOTION"]["status"] == "pass"
    assert document["answer_balance"]["by_type"]["QV2-T7-DURING"]["status"] == "pass"
    assert all(
        row["motion_gold_authority"] == "authored_timeline_action_id"
        for row in document["questions"]
        if row["type_id"] in {"QV2-TA-MOTION", "QV2-T7-DURING"}
    )


def test_question_generator_prefers_validated_point_audio_program(
    tmp_path: Path,
) -> None:
    module = _load_script()
    inputs, programs, registry, viewpoint_bank = _fixture(tmp_path)
    point = inputs / "p001"
    first_start = sample_boundary(2)
    second_start = sample_boundary(26)
    event_duration = 4_800
    events = []
    for index, (endpoint, start_sample) in enumerate(
        (
            ("source1-mouth", first_start),
            ("source2-mouth", second_start),
        )
    ):
        end_sample = start_sample + event_duration
        events.append(
            {
                "event_id": f"event_{index}",
                "source_endpoint_id": endpoint,
                "sound_asset_id": "unit_sound",
                "start_tick": start_sample * 3,
                "end_tick_exclusive": end_sample * 3,
                "start_sample": start_sample,
                "end_sample_exclusive": end_sample,
                "source_start_sample": 0,
                "source_end_sample_exclusive": event_duration,
                "linear_gain": 0.2,
                "fade_samples": 80,
                "normalization_policy": "use_sound_asset_policy",
                "render_source_stem": True,
            }
        )
    program = bind_audio_program_hash(
        {
            "schema": "avengine_m6_audio_program_v1",
            "program_id": "p.p001.schedule_v1",
            "revision": "v1",
            "mode": "sequential_sources",
            "timeline": {
                "time_base_hz": 48_000,
                "ticks_per_frame": 3_200,
                "video_fps": 15,
                "frame_count": 75,
                "sample_rate_hz": 16_000,
                "ticks_per_sample": 3,
                "sample_count": 80_000,
            },
            "candidate_source_endpoint_ids": ["source1-mouth", "source2-mouth"],
            "events": events,
            "source_specific_stems": True,
            "admission_state": "research",
        }
    )
    _write_json(point / "audio_program.json", program)
    spec_path = point / "spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["audio_program_path"] = "audio_program.json"
    spec["audio_schedule"] = {
        "base_program_id": "p",
        "materialized_program_id": program["program_id"],
        "event_start_samples": [event["start_sample"] for event in events],
        "event_end_samples_exclusive": [
            event["end_sample_exclusive"] for event in events
        ],
    }
    _write_json(spec_path, spec)

    document = module.generate_question_document(
        inputs_root=inputs,
        programs_dir=programs,
        registry_path=registry,
        viewpoint_bank_path=viewpoint_bank,
        generated_at="fixed",
    )

    assert document["audio_program_resolution"]["materialized_point_count"] == 1
    assert document["audio_program_resolution"]["points"]["p001"] == {
        "program_id": "p.p001.schedule_v1",
        "path": str(point / "audio_program.json"),
        "authority": "point_materialized_audio_program",
    }
    for question in document["questions"]:
        assert question["audio_program_id"] == "p.p001.schedule_v1"
        assert question["audio_program_authority"] == "point_materialized_audio_program"
        assert question["first_frame"] == 2
        assert question["second_frame"] == 26
        assert question["first_event_start_sample"] == first_start
        assert question["second_event_start_sample"] == second_start
        assert question["first_second_onset_interval_samples"] == (
            second_start - first_start
        )


def test_materialized_audio_schedule_mismatch_fails_closed(tmp_path: Path) -> None:
    module = _load_script()
    inputs, programs, registry, viewpoint_bank = _fixture(tmp_path)
    point = inputs / "p001"
    spec_path = point / "spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["audio_program_path"] = "audio_program.json"
    spec["audio_schedule"] = {}
    _write_json(spec_path, spec)
    _write_json(point / "audio_program.json", {})

    with pytest.raises(
        module.QuestionGenerationError, match="materialized AudioProgram"
    ):
        module.generate_question_document(
            inputs_root=inputs,
            programs_dir=programs,
            registry_path=registry,
            viewpoint_bank_path=viewpoint_bank,
        )


def test_motion_case_timeline_disagreement_fails_closed(tmp_path: Path) -> None:
    module = _load_script()
    inputs, programs, registry, viewpoint_bank = _fixture(tmp_path)
    timeline_path = inputs / "p001/timeline.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    for frame in timeline["frames"]:
        for state in frame["actor_states"]:
            if state["source_slot_id"] == "source1":
                state["action_id"] = "idle"
    _write_json(timeline_path, timeline)

    with pytest.raises(
        module.QuestionGenerationError,
        match="motion_case disagrees with authored Timeline",
    ):
        module.generate_question_document(
            inputs_root=inputs,
            programs_dir=programs,
            registry_path=registry,
            viewpoint_bank_path=viewpoint_bank,
        )
