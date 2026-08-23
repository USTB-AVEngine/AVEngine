from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY / "tools/qa/generate_qa_v2_questions.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("generate_qa_v2_questions", SCRIPT_PATH)
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
                {"source_endpoint_id": "source1-mouth", "start_sample": 4000},
                {"source_endpoint_id": "source2-mouth", "start_sample": 24000},
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
                        "translation_ue_cm": [10.0, -10.0, 27.1],
                    },
                    {
                        "source_slot_id": "source2",
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
