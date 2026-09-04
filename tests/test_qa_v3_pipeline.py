"""Focused tests for the single QA-v3 pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import run_qa_v3_pipeline as pipeline  # noqa: E402


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def test_request_keeps_budget_and_pool_declarations(tmp_path: Path) -> None:
    scene = _write(tmp_path / "scene.json", {"scene_id": "room_a"})
    profiles = _write(tmp_path / "profiles.json", [{"id": "speech_profile"}])
    _write(tmp_path / "pool.json", {"events": []})
    params = _write(
        tmp_path / "params.json",
        {
            "PAIR_KIND": "human",
            "SOUND_SOURCE_MODE": "event_pool",
            "SOUND_EVENT_POOL": "pool.json",
            "ITEMS_PER_ROOM_DEFAULT": 4,
            "ANSWER_FORMS_DEFAULT": ["open"],
        },
    )
    request = _write(
        tmp_path / "request.json",
        {
            "scene_configs": ["scene.json"],
            "profiles": "profiles.json",
            "params": "params.json",
            "requested_profiles": ["speech_profile"],
            "question_budget": 3,
            "answer_forms": ["open"],
            "audio_variants": ["main"],
        },
    )
    loaded = pipeline._load_request(request)
    assert loaded["question_budget"] == 3
    assert loaded["answer_forms"] == ["open"]
    assert loaded["scene_configs"] == [scene.resolve()]
    assert loaded["profiles"] == profiles.resolve()
    assert loaded["params"] == params.resolve()
    assert pipeline._params_summary(params)["PAIR_KIND"] == "human"
    assert pipeline._params_summary(params)["SOUND_SOURCE_MODE"] == "event_pool"


def test_profile_params_keep_common_budget_forms_but_allow_pair_specific_clock(tmp_path: Path) -> None:
    dog_params = _write(
        tmp_path / "dog_params.json",
        {
            "PAIR_KIND": "dog",
            "SOUND_SOURCE_MODE": "dry_canvas_window",
            "SOUND_ASSET": "dog_bark",
            "ITEMS_PER_ROOM_DEFAULT": 4,
            "ANSWER_FORMS_DEFAULT": ["mcq", "open"],
            "FRAME_COUNT": 75,
            "VIDEO_FPS": 15,
            "SAMPLE_RATE_HZ": 16000,
            "SAMPLE_COUNT": 80000,
        },
    )
    human_params = _write(
        tmp_path / "human_params.json",
        {
            "PAIR_KIND": "human",
            "SOUND_SOURCE_MODE": "event_pool",
            "SOUND_EVENT_POOL": "human_pool.json",
            "ITEMS_PER_ROOM_DEFAULT": 99,
            "ANSWER_FORMS_DEFAULT": ["open"],
            "FRAME_COUNT": 90,
            "VIDEO_FPS": 15,
            "SAMPLE_RATE_HZ": 16000,
            "SAMPLE_COUNT": 96000,
        },
    )
    _write(tmp_path / "human_pool.json", {"events": []})
    scene = _write(tmp_path / "scene.json", {"scene_id": "room_a"})
    profiles = _write(
        tmp_path / "profiles.json",
        [{"id": "dog_profile"}, {"id": "human_profile"}],
    )
    request = _write(
        tmp_path / "request.json",
        {
            "scene_configs": [str(scene)],
            "profiles": str(profiles),
            "params": str(dog_params),
            "profile_params": {
                "dog_profile": str(dog_params),
                "human_profile": {
                    "params": str(human_params),
                    "overrides": {
                        "FRAME_COUNT": 150,
                        "SAMPLE_COUNT": 160000,
                        "CLIP_SECONDS": 10.0,
                    },
                },
            },
            "requested_profiles": ["dog_profile", "human_profile"],
            "question_budget": 4,
            "answer_forms": ["mcq", "open"],
        },
    )
    loaded = pipeline._load_request(request)
    forms, budget, cells = pipeline._common_request_plan(loaded)
    assert forms == ["mcq", "open"]
    assert budget == 4
    assert cells == {"dog_profile": 1, "human_profile": 1}
    dog_snapshot, dog_summary = pipeline._effective_profile_params(
        loaded, "dog_profile", common_forms=forms,
        common_budget=budget, output_root=tmp_path / "out",
    )
    human_snapshot, human_summary = pipeline._effective_profile_params(
        loaded, "human_profile", common_forms=forms,
        common_budget=budget, output_root=tmp_path / "out",
    )
    assert dog_snapshot.is_file() and human_snapshot.is_file()
    assert dog_summary["PAIR_KIND"] == "dog"
    assert human_summary["PAIR_KIND"] == "human"
    assert human_summary["SOUND_SOURCE_MODE"] == "event_pool"
    assert human_summary["SOUND_EVENT_POOL"].endswith("human_pool.json")
    human_value = json.loads(human_snapshot.read_text())
    assert human_value["ITEMS_PER_ROOM_DEFAULT"] == 4
    assert human_value["FRAME_COUNT"] == 150
    assert human_value["SAMPLE_COUNT"] == 160000
    assert human_value["CLIP_SECONDS"] == 10.0


def test_resume_only_plans_independent_profile_params_without_launching_design(tmp_path: Path) -> None:
    dog = _write(
        tmp_path / "dog.json",
        {
            "PAIR_KIND": "dog",
            "SOUND_SOURCE_MODE": "dry_canvas_window",
            "ITEMS_PER_ROOM_DEFAULT": 4,
            "ANSWER_FORMS_DEFAULT": ["open"],
        },
    )
    human = _write(
        tmp_path / "human.json",
        {
            "PAIR_KIND": "human",
            "SOUND_SOURCE_MODE": "event_pool",
            "SOUND_EVENT_POOL": "human_pool.json",
            "ITEMS_PER_ROOM_DEFAULT": 4,
            "ANSWER_FORMS_DEFAULT": ["open"],
        },
    )
    _write(tmp_path / "human_pool.json", {"events": []})
    scene = _write(tmp_path / "scene.json", {"scene_id": "room_a"})
    profiles = _write(
        tmp_path / "profiles.json",
        [{"id": "dog_profile"}, {"id": "human_profile"}],
    )
    request = _write(
        tmp_path / "request.json",
        {
            "scene_configs": [str(scene)],
            "profiles": str(profiles),
            "params": str(dog),
            "profile_params": {
                "dog_profile": str(dog),
                "human_profile": str(human),
            },
            "requested_profiles": ["dog_profile", "human_profile"],
            "question_budget": 4,
            "answer_forms": ["open"],
        },
    )
    runtime = _write(tmp_path / "runtime.json", {})
    result = pipeline.run_pipeline(
        request, runtime, tmp_path / "pipeline", resume_only=True)
    assert result["status"] == "partial"
    assert result["stages"]["design"]["common_question_budget"] == 4
    assert result["stages"]["design"]["cells_by_profile"] == {
        "dog_profile": 2,
        "human_profile": 2,
    }
    assert [group["params_summary"]["PAIR_KIND"]
            for group in result["stages"]["design"]["groups"]] == ["dog", "human"]
    assert all(group["status"] == "pending"
               for group in result["stages"]["design"]["groups"])
    assert result["stages"]["assembly"]["status"] == "pending"


def _make_existing_design(root: Path) -> tuple[Path, Path, Path, Path]:
    design = root / "design"
    batch = design / "rooms" / "room_a" / "profiles" / "speech_profile" / "batch"
    point = batch / "speech_profile_001"
    point.mkdir(parents=True)
    fact = {
        "scene_id": "room_a",
        "profile_id": "speech_profile",
        "point_id": "speech_profile_001",
        "answer_forms": ["open"],
        "mcq": {
            "stem": "which?",
            "options_space": ["a", "b"],
            "truth_option": "a",
        },
        "open": {
            "stem": "which?",
            "truth_value": "a",
            "scoring": "closed_set",
        },
        "camera": {"height_m": 1.5, "clearance": {"fallback_used": False}},
        "audio": {"program_id": "speech_profile_001"},
    }
    _write(point / "fact_record.json", fact)
    _write(point / "fact_record_gateA.json", fact)
    _write(point / "actor_selection.json", {"actors": []})
    _write(point / "m1_capture_request.json", {"request": "synthetic"})
    _write(point / "audio_program.json", {"program_id": "speech_profile_001"})
    _write(point / "audio_program_gateA.json", {"program_id": "speech_profile_001_gateA"})
    frames = []
    for index in range(75):
        frames.append(
            {
                "camera": {
                    "translation_ue_cm": [0.0, 0.0, 150.0],
                    "yaw_ue_deg": 0.0,
                },
                "actor_states": [
                    {
                        "source_slot_id": "source1",
                        "translation_ue_cm": [100.0 + index, 0.0, 0.0],
                    },
                    {
                        "source_slot_id": "source2",
                        "translation_ue_cm": [200.0, 0.0, 0.0],
                    },
                ],
            }
        )
    _write(point / "timeline.json", {"frames": frames})
    (batch / "questions.jsonl").write_text(
        json.dumps(
            {
                "point_id": point.name,
                "profile_id": "speech_profile",
                "variant": "main",
                "form": "open",
                "question": "which?",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (batch / "questions_gateA.jsonl").write_text(
        json.dumps(
            {
                "point_id": point.name,
                "profile_id": "speech_profile",
                "variant": "gateA",
                "form": "open",
                "question": "which?",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    batch_manifest = _write(
        batch / "batch_manifest.json",
        {
            "status": "research_candidate",
            "records": [{"point_id": point.name}],
            "counts": {
                "cells_requested": 1,
                "geometry_candidates": 1,
                "rejected": 0,
            },
            "question_request": {"answer_forms": ["open"]},
        },
    )
    matrix = _write(
        design / "scene_profile_matrix.json",
        {
            "status": "completed",
            "scenes": [{"scene_id": "room_a"}],
            "requested_profiles": ["speech_profile"],
            "matrix": [
                {
                    "scene_id": "room_a",
                    "profile_id": "speech_profile",
                    "attempt_status": "generated",
                    "requested_cells": 1,
                    "geometry_candidates": 1,
                    "quota_shortfall": 0,
                    "batch_manifest": str(batch_manifest),
                }
            ],
        },
    )
    profiles = _write(root / "profiles.json", [{"id": "speech_profile"}])
    scene = _write(root / "scene.json", {"scene_id": "room_a"})
    params = _write(
        root / "params.json",
        {
            "PAIR_KIND": "human",
            "SOUND_SOURCE_MODE": "event_pool",
            "SOUND_EVENT_POOL": "events.json",
            "ITEMS_PER_ROOM_DEFAULT": 1,
            "ANSWER_FORMS_DEFAULT": ["open"],
        },
    )
    _write(root / "events.json", {"events": []})
    return design, profiles, scene, params


def test_resume_only_assembles_but_keeps_missing_media_pending(tmp_path: Path) -> None:
    design, profiles, scene, params = _make_existing_design(tmp_path)
    request = _write(
        tmp_path / "request.json",
        {
            "scene_configs": [str(scene)],
            "profiles": str(profiles),
            "params": str(params),
            "existing_design_root": str(design),
            "requested_profiles": ["speech_profile"],
            "answer_forms": ["open"],
            "audio_variants": ["main"],
        },
    )
    runtime = _write(tmp_path / "runtime.json", {})
    output = tmp_path / "pipeline"
    result = pipeline.run_pipeline(
        request, runtime, output, resume_only=True)
    assert result["status"] == "partial"
    assert result["stages"]["design"]["run"]["status"] == "reused_existing"
    assert result["stages"]["assembly"]["run"]["status"] == "complete"
    pair = result["stages"]["pairs"][0]
    assert pair["capture"]["status"] == "pending"
    assert pair["audio"]["status"] == "pending"
    assert pair["media"]["status"] == "pending"
    assert pair["verification"]["status"] == "pending"
    assert result["stages"]["questions"]["status"] == "pending"
    assert result["evaluation"]["status"] == "pending"
    saved = json.loads((output / "pipeline_manifest.json").read_text())
    assert saved["status"] == "partial"


def test_missing_capture_resources_are_recorded_as_failure(tmp_path: Path) -> None:
    design, profiles, scene, params = _make_existing_design(tmp_path)
    request = _write(
        tmp_path / "request.json",
        {
            "scene_configs": [str(scene)],
            "profiles": str(profiles),
            "params": str(params),
            "existing_design_root": str(design),
            "requested_profiles": ["speech_profile"],
            "answer_forms": ["open"],
            "audio_variants": ["main"],
        },
    )
    runtime = _write(tmp_path / "runtime.json", {"python": sys.executable})
    result = pipeline.run_pipeline(
        request, runtime, tmp_path / "pipeline-execute")
    assert result["status"] == "failed"
    assert result["stages"]["pairs"][0]["capture"]["status"] == "failed"
    assert any(failure["stage"] == "capture"
               for failure in result["failures"])


def test_invalid_runtime_timeout_fails_before_stage_launch(tmp_path: Path) -> None:
    scene = _write(tmp_path / "scene.json", {"scene_id": "room_a"})
    profiles = _write(tmp_path / "profiles.json", [{"id": "p"}])
    params = _write(
        tmp_path / "params.json",
        {
            "PAIR_KIND": "dog",
            "SOUND_SOURCE_MODE": "dry_canvas_window",
        },
    )
    request = _write(
        tmp_path / "request.json",
        {
            "scene_configs": [str(scene)],
            "profiles": str(profiles),
            "params": str(params),
        },
    )
    runtime = _write(
        tmp_path / "runtime.json",
        {"timeouts": {"design": 0}},
    )
    with pytest.raises(pipeline.PipelineError, match="finite and positive"):
        pipeline.run_pipeline(
            request, runtime, tmp_path / "output", resume_only=True)


def test_through_stage_assemble_leaves_runtime_and_questions_pending(tmp_path: Path) -> None:
    design, profiles, scene, params = _make_existing_design(tmp_path)
    request = _write(
        tmp_path / "request.json",
        {
            "scene_configs": [str(scene)],
            "profiles": str(profiles),
            "params": str(params),
            "existing_design_root": str(design),
            "requested_profiles": ["speech_profile"],
            "answer_forms": ["open"],
            "audio_variants": ["main"],
            "seed": "through-stage-test",
        },
    )
    runtime = _write(tmp_path / "runtime.json", {"python": sys.executable})
    output = tmp_path / "pipeline"
    result = pipeline.run_pipeline(
        request, runtime, output, through_stage="assemble")
    assert result["status"] == "partial"
    assert result["through_stage"] == "assemble"
    assert result["run_control"]["through_stage"] == "assemble"
    assert "through_stage" not in result["request"]
    assert result["stages"]["design"]["run"]["status"] == "reused_existing"
    assert result["stages"]["assembly"]["status"] == "complete"
    pair = result["stages"]["pairs"][0]
    assert pair["capture"]["status"] == "pending"
    assert "through-stage=assemble" in pair["capture"]["detail"]
    assert pair["audio"]["status"] == "pending"
    assert pair["media"]["status"] == "pending"
    assert pair["verification"]["status"] == "pending"
    assert result["stages"]["questions"]["status"] == "pending"
    assert "through-stage=assemble" in result["stages"]["questions"]["detail"]
    saved = json.loads((output / "pipeline_manifest.json").read_text())
    assert saved["through_stage"] == "assemble"


def test_profile_object_catalog_can_declare_offscreen_backend(tmp_path: Path) -> None:
    scene = _write(tmp_path / "scene.json", {"scene_id": "room_a"})
    profile = _write(
        tmp_path / "profile.json",
        {
            "schema": "avengine_qa_v3_offscreen_identity_profile_v1",
            "id": "offscreen_profile",
            "execution_backend": "offscreen_identity",
        },
    )
    params = _write(
        tmp_path / "params.json",
        {
            "PAIR_KIND": "human",
            "SOUND_SOURCE_MODE": "event_pool",
            "ITEMS_PER_ROOM_DEFAULT": 1,
            "ANSWER_FORMS_DEFAULT": ["mcq"],
        },
    )
    request = _write(
        tmp_path / "request.json",
        {
            "scene_configs": [str(scene)],
            "profiles": str(profile),
            "params": str(params),
            "requested_profiles": ["offscreen_profile"],
            "question_budget": 1,
            "answer_forms": ["mcq"],
        },
    )
    loaded = pipeline._load_request(request)
    assert pipeline._request_profile_ids(loaded) == ["offscreen_profile"]
    assert pipeline._profile_catalog(profile)[0]["execution_backend"] == "offscreen_identity"


def test_verifiers_receive_each_pair_effective_params(monkeypatch, tmp_path: Path) -> None:
    runtime = {"python": sys.executable}
    request = {"audio_variants": ["main"]}
    captured = []

    monkeypatch.setattr(
        pipeline,
        "_capture_states",
        lambda batch_root, capture_root, point_ids: ("complete", {}, None),
    )
    monkeypatch.setattr(
        pipeline,
        "_audio_states",
        lambda audio_root, point_ids, variants: ("complete", {}),
    )

    def fake_run(label, command, log_path, *, timeout, env=None):
        del timeout, env
        captured.append((label, list(command)))
        output = Path(command[command.index("--out") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        report = (
            {
                "schema": "qa_v3_audio_batch_verification_v1",
                "status": "research_candidate",
                "checked_renders": 1,
                "failures": [],
            }
            if label.endswith("/audio")
            else {
                "schema": "qa_v3_visual_batch_verification_v1",
                "status": "pass",
                "counts": {"failures": 0},
            }
        )
        output.write_text(json.dumps(report), encoding="utf-8")
        return {"status": "complete", "label": label}

    monkeypatch.setattr(pipeline, "_run_logged", fake_run)
    params_by_profile = {}
    for profile_id, frame_count in (("dog_profile", 75), ("human_profile", 150)):
        params = tmp_path / f"{profile_id}.json"
        params.write_text(json.dumps({"FRAME_COUNT": frame_count}), encoding="utf-8")
        params_by_profile[profile_id] = params
        result = pipeline._run_verifications(
            runtime,
            request,
            "room_a",
            profile_id,
            tmp_path / profile_id / "batch",
            tmp_path / profile_id,
            ["point_001"],
            params_path=params,
            resume_only=False,
        )
        assert result["status"] == "complete"

    audio_commands = {
        label.split("/")[1]: command
        for label, command in captured
        if label.startswith("verification:") and label.endswith("/audio")
    }
    assert set(audio_commands) == {"dog_profile", "human_profile"}
    for profile_id, params in params_by_profile.items():
        command = audio_commands[profile_id]
        assert command[command.index("--params") + 1] == str(params.resolve())


def test_resume_extends_stage_scope_and_rejects_backward_scope(tmp_path: Path) -> None:
    design, profiles, scene, params = _make_existing_design(tmp_path)
    request = _write(
        tmp_path / "request.json",
        {
            "scene_configs": [str(scene)],
            "profiles": str(profiles),
            "params": str(params),
            "existing_design_root": str(design),
            "requested_profiles": ["speech_profile"],
            "answer_forms": ["open"],
            "audio_variants": ["main"],
            "seed": "resume-scope-test",
        },
    )
    runtime = _write(tmp_path / "runtime.json", {"python": sys.executable})
    output = tmp_path / "pipeline"
    first = pipeline.run_pipeline(
        request, runtime, output, through_stage="assemble")
    assert first["stages"]["assembly"]["status"] == "complete"

    with pytest.raises(pipeline.PipelineError, match="cannot resume"):
        pipeline.run_pipeline(
            request, runtime, output, resume=True, resume_only=True,
            through_stage="design")

    second = pipeline.run_pipeline(
        request, runtime, output, resume=True, resume_only=True,
        through_stage="questions")
    assert second["status"] == "partial"
    assert second["through_stage"] == "questions"
    assert second["run_control"]["mode"] == "resume_only"
    assert second["stages"]["design"]["run"]["status"] == "reused_existing"
    assert second["stages"]["assembly"]["run"]["status"] == "reused_existing"
    request_snapshot = json.loads((output / "request.json").read_text())
    assert "through_stage" not in request_snapshot
    assert "mode" not in request_snapshot


def test_resume_rejects_budget_and_snapshot_content_changes(tmp_path: Path) -> None:
    design, profiles, scene, params = _make_existing_design(tmp_path)
    content_a = tmp_path / "content_a"
    content_b = tmp_path / "content_b"
    content_a.mkdir()
    content_b.mkdir()
    request_value = {
        "scene_configs": [str(scene)],
        "profiles": str(profiles),
        "params": str(params),
        "existing_design_root": str(design),
        "requested_profiles": ["speech_profile"],
        "question_budget": 1,
        "answer_forms": ["open"],
        "audio_variants": ["main"],
        "snapshot_content": str(content_a),
    }
    request = _write(tmp_path / "request.json", request_value)
    runtime = _write(tmp_path / "runtime.json", {"python": sys.executable})
    output = tmp_path / "pipeline"
    pipeline.run_pipeline(request, runtime, output, through_stage="assemble")

    request_value["question_budget"] = 2
    request.write_text(json.dumps(request_value), encoding="utf-8")
    with pytest.raises(pipeline.PipelineError, match="snapshot"):
        pipeline.run_pipeline(
            request, runtime, output, resume=True, resume_only=True,
            through_stage="questions")

    request_value["question_budget"] = 1
    request_value["snapshot_content"] = str(content_b)
    request.write_text(json.dumps(request_value), encoding="utf-8")
    with pytest.raises(pipeline.PipelineError, match="snapshot"):
        pipeline.run_pipeline(
            request, runtime, output, resume=True, resume_only=True,
            through_stage="questions")


def test_resume_rejects_scene_and_profile_content_changes(tmp_path: Path) -> None:
    design, profiles, scene, params = _make_existing_design(tmp_path)
    request_value = {
        "scene_configs": [str(scene)],
        "profiles": str(profiles),
        "params": str(params),
        "existing_design_root": str(design),
        "requested_profiles": ["speech_profile"],
        "answer_forms": ["open"],
        "audio_variants": ["main"],
    }
    request = _write(tmp_path / "request.json", request_value)
    runtime = _write(tmp_path / "runtime.json", {"python": sys.executable})
    output = tmp_path / "pipeline"
    pipeline.run_pipeline(request, runtime, output, through_stage="assemble")

    scene.write_text(json.dumps({"scene_id": "room_changed"}), encoding="utf-8")
    with pytest.raises(pipeline.PipelineError, match="snapshot"):
        pipeline.run_pipeline(
            request, runtime, output, resume=True, resume_only=True,
            through_stage="questions")
    scene.write_text(json.dumps({"scene_id": "room_a"}), encoding="utf-8")

    profiles.write_text(
        json.dumps([{"id": "different_profile"}]),
        encoding="utf-8",
    )
    with pytest.raises(pipeline.PipelineError, match="snapshot"):
        pipeline.run_pipeline(
            request, runtime, output, resume=True, resume_only=True,
            through_stage="questions")


def test_audio_command_uses_point_local_bindings_without_batch_fallbacks(
    tmp_path: Path,
) -> None:
    cfg = {
        "python": sys.executable,
        "repo": str(pipeline.REPO),
        "simulation_request": "simulation.json",
        "package_manifest": "package.json",
        "sound_asset_registry": "sounds.json",
        "hrtf": "hrtf.sofa",
        "runtime_prefix": "runtime",
        "rlr_sdk_root": "rlr",
        "magnum_python_site": "magnum",
        "source_asset_registry": "source_assets.json",
        "_snapshot_path": str(tmp_path / "audio_config.json"),
    }
    command = pipeline._audio_command(
        cfg,
        tmp_path / "batch",
        tmp_path / "capture",
        tmp_path / "audio",
        ["main"],
        ["point_001"],
        resume=False,
    )
    assert "--config" in command
    assert "m1_request" not in cfg
    assert "source_endpoint_registry" not in cfg


def test_pair_failure_stops_later_stages(monkeypatch, tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "batch_manifest.json").write_text("{}")
    monkeypatch.setattr(pipeline, "batch_point_ids", lambda root: ["point_001"])
    monkeypatch.setattr(
        pipeline, "_run_capture",
        lambda *args, **kwargs: {"status": "complete", "root": "capture"})
    monkeypatch.setattr(
        pipeline, "_run_audio",
        lambda *args, **kwargs: {"status": "complete", "root": "audio"})
    monkeypatch.setattr(
        pipeline, "_run_media",
        lambda *args, **kwargs: {
            "status": "failed", "root": "media", "detail": "missing codec"})
    verification_called = []
    monkeypatch.setattr(
        pipeline, "_run_verifications",
        lambda *args, **kwargs: verification_called.append(True)
        or {"status": "complete"})
    pair = pipeline._pair_record(
        {"audio_variants": ["main"]},
        {},
        {
            "scene_id": "room_a",
            "profile_id": "profile_a",
            "attempt_status": "generated",
            "batch_manifest": str(batch / "batch_manifest.json"),
        },
        tmp_path,
        tmp_path / "output",
        params_path=tmp_path / "params.json",
        resume_only=False,
        through_stage="questions",
    )
    assert pair["status"] == "failed"
    assert pair["media"]["status"] == "failed"
    assert pair["verification"]["status"] == "pending"
    assert "fail-fast" in pair["verification"]["detail"]
    assert verification_called == []


def test_failed_pair_prevents_later_pair_launch(monkeypatch, tmp_path: Path) -> None:
    scene = _write(tmp_path / "scene.json", {"scene_id": "room_a"})
    profiles = _write(
        tmp_path / "profiles.json",
        [{"id": "first"}, {"id": "second"}],
    )
    params = _write(
        tmp_path / "params.json",
        {
            "PAIR_KIND": "dog",
            "SOUND_SOURCE_MODE": "dry_canvas_window",
            "ITEMS_PER_ROOM_DEFAULT": 2,
            "ANSWER_FORMS_DEFAULT": ["mcq"],
        },
    )
    request = _write(
        tmp_path / "request.json",
        {
            "scene_configs": [str(scene)],
            "profiles": str(profiles),
            "params": str(params),
            "requested_profiles": ["first", "second"],
            "cells_per_pair": 1,
            "answer_forms": ["mcq"],
            "audio_variants": ["main"],
            "seed": "fail-fast-test",
        },
    )
    runtime = _write(tmp_path / "runtime.json", {"python": sys.executable})
    rows = [
        {
            "scene_id": "room_a",
            "profile_id": profile,
            "attempt_status": "generated",
            "batch_manifest": str(tmp_path / f"{profile}.json"),
        }
        for profile in ("first", "second")
    ]
    design = {
        "status": "complete",
        "groups": [{
            "root": str(tmp_path / "design"),
            "matrix": {"matrix": rows},
            "params_path": str(params),
        }],
    }
    monkeypatch.setattr(pipeline, "_design", lambda *args, **kwargs: design)
    monkeypatch.setattr(
        pipeline,
        "_assemble",
        lambda *args, **kwargs: {
            "status": "partial",
            "root": str(tmp_path / "assembly"),
            "pilot": {"rooms": {}},
        },
    )
    launched = []

    def fake_pair(request, runtime, row, design_root, out_root, **kwargs):
        del request, runtime, design_root, out_root, kwargs
        launched.append(row["profile_id"])
        return {
            "scene_id": row["scene_id"],
            "profile_id": row["profile_id"],
            "status": "failed",
            "design": dict(row),
            "capture": {"status": "failed", "detail": "fixture failure"},
            "audio": {"status": "pending"},
            "media": {"status": "pending"},
            "verification": {"status": "pending"},
            "points": [],
        }

    monkeypatch.setattr(pipeline, "_pair_record", fake_pair)
    monkeypatch.setattr(
        pipeline,
        "_run_questions",
        lambda *args, **kwargs: {"status": "pending", "root": "questions"},
    )
    result = pipeline.run_pipeline(
        request, runtime, tmp_path / "output", through_stage="questions")
    assert launched == ["first"]
    assert result["status"] == "failed"
    assert result["stages"]["pairs"][0]["status"] == "failed"
    second = result["stages"]["pairs"][1]
    assert second["profile_id"] == "second"
    assert second["status"] == "pending"
    assert "fail-fast stopped after room_a/first failed" in second["detail"]


def test_verification_report_contracts_distinguish_visual_and_audio_statuses() -> None:
    visual = {
        "schema": "qa_v3_visual_batch_verification_v1",
        "status": "pass",
        "counts": {"failures": 0},
    }
    audio = {
        "schema": "qa_v3_audio_batch_verification_v1",
        "status": "research_candidate",
        "checked_renders": 2,
        "failures": [],
    }
    assert pipeline._verification_report_passed("visual", visual)
    assert pipeline._verification_report_passed("audio", audio)
    assert not pipeline._verification_report_passed(
        "audio", {**audio, "failures": ["bad onset"]})
    assert not pipeline._verification_report_passed(
        "audio", {**audio, "checked_renders": 0})
    assert not pipeline._verification_report_passed(
        "visual", {**visual, "status": "research_candidate"})
