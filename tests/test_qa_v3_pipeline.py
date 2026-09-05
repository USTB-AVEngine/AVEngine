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
        lambda audio_root, point_ids, variants, layouts: ("complete", {}),
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
                "expected_variants": ["main"],
                "expected_layouts": ["binaural"],
                "complete_render_point_ids": {
                    "main": ["point_001"], "gateA": []},
                "complete_pair_count": 0,
                "checked_renders": 1,
                "failures": [],
                "audio_root": command[
                    command.index("--audio-root") + 1
                ],
            }
            if label.endswith("/audio")
            else {
                "schema": "qa_v3_visual_batch_verification_v1",
                "status": "pass",
                "counts": {"failures": 0},
                "inputs": {
                    "selection_manifest": command[
                        command.index("--selection-manifest") + 1
                    ],
                    "visual_root": command[
                        command.index("--visual-root") + 1
                    ],
                },
                "points": [{"point_id": "point_001"}],
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
        assert command[command.index("--variants") + 1] == "main"
        assert command[command.index("--layouts") + 1] == "binaural"


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
    assert command[command.index("--layouts") + 1] == "binaural"
    assert "m1_request" not in cfg
    assert "source_endpoint_registry" not in cfg

    foa_cfg = dict(cfg, layouts=["ambisonics"])
    foa_cfg.pop("hrtf")
    foa_command = pipeline._audio_command(
        foa_cfg,
        tmp_path / "batch",
        tmp_path / "capture",
        tmp_path / "audio_foa",
        ["main"],
        ["point_001"],
        resume=False,
    )
    assert foa_command[foa_command.index("--layouts") + 1] == "ambisonics"


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
        "expected_variants": ["main"],
        "expected_layouts": ["binaural"],
        "complete_render_point_ids": {"main": ["p1"], "gateA": []},
        "complete_pair_count": 0,
        "checked_renders": 1,
        "failures": [],
    }
    pair_audio = {
        **audio,
        "expected_variants": ["main", "gateA"],
        "complete_render_point_ids": {"main": ["p1"], "gateA": ["p1"]},
        "complete_pair_count": 1,
        "checked_renders": 2,
        "audio_variant_waveform_nonidentity_pairs": 1,
        "gatea_semantic_flip_pairs": 1,
        "execution_variant_verification": {"status": "verified"},
    }
    assert pipeline._verification_report_passed("visual", visual)
    assert pipeline._verification_report_passed(
        "audio", audio, expected_audio_variants=["main"],
        expected_audio_layouts=["binaural"])
    assert pipeline._verification_report_passed(
        "audio", pair_audio, expected_audio_variants=["main", "gateA"],
        expected_audio_layouts=["binaural"])
    assert not pipeline._verification_report_passed(
        "audio", pair_audio, expected_audio_variants=["main"],
        expected_audio_layouts=["binaural"])
    assert not pipeline._verification_report_passed(
        "audio", pair_audio, expected_audio_variants=["main", "gateA"],
        expected_audio_layouts=["ambisonics"])
    assert not pipeline._verification_report_passed(
        "audio", {**audio, "failures": ["bad onset"]})
    assert not pipeline._verification_report_passed(
        "audio", {**audio, "checked_renders": 0})
    assert not pipeline._verification_report_passed(
        "audio", {key: value for key, value in audio.items()
                  if key != "expected_variants"})
    assert not pipeline._verification_report_passed(
        "visual", {**visual, "status": "research_candidate"})


def test_source_state_records_current_avengine_without_becoming_a_gate() -> None:
    state = pipeline._source_state()
    assert Path(state["repository"]) == pipeline.REPO.resolve()
    assert Path(state["entrypoint"]) == Path(pipeline.__file__).resolve()
    assert isinstance(state["git_commit"], str) and state["git_commit"]
    assert state["git_branch"] is None or (
        isinstance(state["git_branch"], str) and state["git_branch"]
    )
    assert isinstance(state["tracked_worktree_changes"], list)
    assert Path(state["python_executable"]).is_file()



def _make_runtime_artifact_point(
    root: Path,
    fact: dict,
    *,
    extra_files: tuple[str, ...] = (),
) -> tuple[Path, Path]:
    """Create the smallest real candidate directory understood by the resolver."""

    batch = root / "batch"
    point = batch / "point_001"
    point.mkdir(parents=True)
    _write(point / "actor_selection.json", {"actors": []})
    _write(point / "timeline.json", {"frames": []})
    for name in extra_files:
        _write(point / name, {"frames": []})
    _write(point / "fact_record.json", fact)
    return batch, point


def _capture_runtime_for_test(tmp_path: Path) -> dict[str, object]:
    """Use existing files so _stage_config exercises normal path handling."""

    spear_ext = tmp_path / "spear_ext"
    spear_ext.mkdir()
    closure = _write(tmp_path / "closure.json", {})
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    executable = _write(tmp_path / "spear", "#!/bin/sh\n")
    return {
        "capture": {
            "python": sys.executable,
            "spear_ext": str(spear_ext),
            "closure_report": str(closure),
            "stage_root": str(stage_root),
            "spear_executable": str(executable),
        }
    }


def test_legacy_candidate_without_descriptors_keeps_main_capture_behavior(
    monkeypatch, tmp_path: Path,
) -> None:
    batch, _ = _make_runtime_artifact_point(
        tmp_path,
        {"profile_id": "legacy_profile"},
    )
    pair_root = tmp_path / "pair"
    monkeypatch.setattr(
        pipeline,
        "_run_logged",
        lambda *args, **kwargs: pytest.fail(
            "legacy candidate must not launch a second capture"
        ),
    )

    result = pipeline._run_declared_captures(
        _capture_runtime_for_test(tmp_path),
        "room_a",
        "legacy_profile",
        batch,
        pair_root,
        ["point_001"],
        resume_only=False,
    )

    assert result["status"] == "complete"
    assert result["records"] == []
    assert not (pair_root / "declared_visuals").exists()


def test_declared_capture_forwards_descriptor_identity_to_registered_runner(
    monkeypatch, tmp_path: Path,
) -> None:
    fact = {
        "visual_variants": [
            {
                "id": "main",
                "kind": "qa_v3_current_apartment_visual",
                "actor_selection": "actor_selection.json",
                "timeline": "timeline.json",
            },
            {
                "id": "gateB",
                "kind": "qa_v3_current_apartment_visual",
                "actor_selection": "actor_selection_gateB.json",
                "timeline": "timeline_gateB.json",
            },
        ]
    }
    batch, _ = _make_runtime_artifact_point(
        tmp_path,
        fact,
        extra_files=("actor_selection_gateB.json", "timeline_gateB.json"),
    )
    pair_root = tmp_path / "pair"
    calls = []
    states = iter(("missing", "complete"))
    monkeypatch.setattr(
        pipeline,
        "_declared_capture_state",
        lambda *args, **kwargs: next(states),
    )

    def fake_run(label, command, log_path, *, timeout, env=None):
        del log_path, timeout, env
        calls.append((label, list(command)))
        return {"status": "complete", "label": label}

    monkeypatch.setattr(pipeline, "_run_logged", fake_run)
    result = pipeline._run_declared_captures(
        _capture_runtime_for_test(tmp_path),
        "room_a",
        "profile_a",
        batch,
        pair_root,
        ["point_001"],
        resume_only=False,
    )

    assert result["status"] == "complete"
    assert len(calls) == 1
    label, command = calls[0]
    assert label.endswith("/point_001/visual_variant/gateB")
    assert command[command.index("--points") + 1] == "point_001"
    assert command[command.index("--descriptor-id") + 1] == "gateB"
    assert command[command.index("--descriptor-kind") + 1] == "visual_variant"


def test_declared_media_without_audio_variant_stays_pending_without_reuse(
    monkeypatch, tmp_path: Path,
) -> None:
    fact = {
        "visual_variants": [{
            "id": "main",
            "kind": "qa_v3_current_apartment_visual",
            "actor_selection": "actor_selection.json",
            "timeline": "timeline.json",
        }],
        "segments": [
            {"id": "segment1", "variant": "main"},
            {
                "id": "segment2",
                "variant": "main",
                "timeline": "timeline_segment2.json",
            },
        ],
        "release_media": [{
            "id": "segment2_release",
            "variant": "main",
            "segment": "segment2",
            "kind": "qa_v3_review_clip",
            "release": True,
        }],
    }
    batch, _ = _make_runtime_artifact_point(
        tmp_path, fact, extra_files=("timeline_segment2.json",),
    )
    pair_root = tmp_path / "pair"
    # A main audio file exists to prove that its presence cannot satisfy the
    # later segment whose descriptor has no audio consumer.
    main_audio = pair_root / "audio" / "point_001" / "audio" / "binaural" / "mixture.wav"
    main_audio.parent.mkdir(parents=True)
    main_audio.write_bytes(b"segment1-audio")
    monkeypatch.setattr(
        pipeline,
        "_run_logged",
        lambda *args, **kwargs: pytest.fail(
            "segment2 without an audio consumer must not launch media"
        ),
    )

    result = pipeline._run_declared_media(
        {},
        "room_a",
        "profile_a",
        batch,
        pair_root,
        ["point_001"],
        resume_only=False,
    )

    assert result["status"] == "partial"
    assert len(result["records"]) == 1
    record = result["records"][0]
    assert record["release_id"] == "segment2_release"
    assert record["audio_variant"] is None
    assert record["status"] == "pending"
    assert "no declared audio consumer" in record["detail"]
    assert not (pair_root / "declared_media").exists()


def test_missing_native_pixel_truth_stays_pending(monkeypatch, tmp_path: Path) -> None:
    fact = {"pixel_evidence": [{"id": "front", "kind": "qa_v3_extended_pixel"}]}
    batch, _ = _make_runtime_artifact_point(tmp_path, fact)
    params = _write(tmp_path / "params.json", {})
    monkeypatch.setattr(
        pipeline,
        "_run_logged",
        lambda *args, **kwargs: pytest.fail(
            "missing native pixel truth must not invoke the joiner"
        ),
    )

    result = pipeline._run_declared_pixels(
        {},
        "room_a",
        "profile_a",
        batch,
        tmp_path / "pair",
        ["point_001"],
        params_path=params,
        resume_only=False,
    )

    assert result["status"] == "partial"
    assert result["records"][0]["status"] == "pending"
    assert "pixel truth" in result["records"][0]["detail"]


def test_runtime_pixel_paths_invoke_only_the_registered_joiner(
    monkeypatch, tmp_path: Path,
) -> None:
    fact = {"pixel_evidence": [{"id": "front", "kind": "qa_v3_extended_pixel"}]}
    batch, point = _make_runtime_artifact_point(tmp_path, fact)
    params = _write(tmp_path / "params.json", {})
    truth = _write(tmp_path / "pixel_truth.json", {"truth": []})
    output = tmp_path / "pixel_result.json"
    runtime = {
        "python": sys.executable,
        "_path": str(tmp_path / "runtime.json"),
        "pixel": {
            "by_point": {
                "point_001": {
                    "front": {
                        "pixel_truth": str(truth),
                        "params": str(params),
                        "output": str(output),
                    }
                }
            }
        },
    }
    calls = []

    def fake_run(label, command, log_path, *, timeout, env=None):
        del log_path, timeout, env
        calls.append((label, list(command)))
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
        return {"status": "complete", "label": label}

    monkeypatch.setattr(pipeline, "_run_logged", fake_run)
    result = pipeline._run_declared_pixels(
        runtime,
        "room_a",
        "profile_a",
        batch,
        tmp_path / "pair",
        [point.name],
        params_path=params,
        resume_only=False,
    )

    assert result["status"] == "complete"
    assert len(calls) == 1
    _, command = calls[0]
    registered = pipeline.registered_pixel_consumer("qa_v3_extended_pixel")
    assert registered is not None
    assert Path(command[1]).resolve() == registered.resolve()
    assert command[command.index("--pixel-truth") + 1] == str(truth.resolve())
    assert command[command.index("--params") + 1] == str(params.resolve())


def test_unknown_pixel_kind_stays_pending_without_execution(
    monkeypatch, tmp_path: Path,
) -> None:
    fact = {"pixel_evidence": [{"id": "future", "kind": "future_pixel_consumer"}]}
    batch, _ = _make_runtime_artifact_point(tmp_path, fact)
    monkeypatch.setattr(
        pipeline,
        "_run_logged",
        lambda *args, **kwargs: pytest.fail(
            "unknown pixel kinds must not be executable"
        ),
    )

    result = pipeline._run_declared_pixels(
        {},
        "room_a",
        "profile_a",
        batch,
        tmp_path / "pair",
        ["point_001"],
        params_path=tmp_path / "params.json",
        resume_only=False,
    )

    assert result["status"] == "partial"
    assert result["records"][0]["status"] == "pending"
    assert "no registered consumer" in result["records"][0]["detail"]


def test_pair_status_remains_partial_when_declared_artifact_is_pending(
    monkeypatch, tmp_path: Path,
) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    manifest = _write(batch / "batch_manifest.json", {})
    monkeypatch.setattr(pipeline, "batch_point_ids", lambda root: ["point_001"])
    monkeypatch.setattr(
        pipeline,
        "_run_capture",
        lambda *args, **kwargs: {"status": "complete", "root": "capture"},
    )
    monkeypatch.setattr(
        pipeline,
        "_run_declared_captures",
        lambda *args, **kwargs: {"status": "complete", "root": "visuals"},
    )
    monkeypatch.setattr(
        pipeline,
        "_run_audio",
        lambda *args, **kwargs: {"status": "complete", "root": "audio"},
    )
    monkeypatch.setattr(
        pipeline,
        "_run_media",
        lambda *args, **kwargs: {"status": "complete", "root": "media"},
    )
    monkeypatch.setattr(
        pipeline,
        "_run_declared_media",
        lambda *args, **kwargs: {"status": "complete", "root": "declared_media"},
    )
    monkeypatch.setattr(
        pipeline,
        "_run_verifications",
        lambda *args, **kwargs: {"status": "complete", "root": "verification"},
    )
    monkeypatch.setattr(
        pipeline,
        "_run_declared_pixels",
        lambda *args, **kwargs: {
            "status": "partial",
            "root": "declared_pixels",
            "records": [{"status": "pending", "evidence_id": "front"}],
        },
    )

    pair = pipeline._pair_record(
        {"audio_variants": ["main"]},
        {},
        {
            "scene_id": "room_a",
            "profile_id": "profile_a",
            "attempt_status": "generated",
            "batch_manifest": str(manifest),
        },
        tmp_path,
        tmp_path / "output",
        params_path=tmp_path / "params.json",
        resume_only=False,
        through_stage="questions",
    )

    assert pair["status"] == "partial"
    assert pair["declared_pixels"]["status"] == "partial"


def test_questions_wait_for_pending_declared_artifact_before_release(
    monkeypatch, tmp_path: Path,
) -> None:
    point_id = "point_001"
    audio_root = tmp_path / "audio"
    media_root = tmp_path / "media"
    mixture = audio_root / point_id / "audio" / "binaural" / "mixture.wav"
    mixture.parent.mkdir(parents=True)
    mixture.write_bytes(b"audio")
    base_video = media_root / f"{point_id}.base.mp4"
    media_root.mkdir(parents=True)
    base_video.write_bytes(b"video")
    params = _write(tmp_path / "params.json", {})
    pilot_path = _write(tmp_path / "pilot.json", {})
    pilot = {
        "question_count": 1,
        "rooms": {
            "room_a": {
                "profiles": {
                    "profile_a": {
                        "status": "selected",
                        "candidates": [{
                            "source_point_id": point_id,
                            "pilot_id": "pilot_001",
                        }],
                    }
                }
            }
        },
    }
    pair = {
        "scene_id": "room_a",
        "profile_id": "profile_a",
        "status": "partial",
        "capture": {"status": "complete"},
        "audio": {"status": "complete", "root": str(audio_root)},
        "media": {"status": "complete", "root": str(media_root)},
        "declared_capture": {"status": "complete"},
        "declared_media": {"status": "complete"},
        "declared_pixels": {"status": "pending"},
    }
    released_path = (
        tmp_path / "pipeline" / "questions" / "released_items.json"
    )
    released_path.parent.mkdir(parents=True)
    released_path.write_text(
        json.dumps([{"task_type": "stale"}]), encoding="utf-8"
    )
    launched = []

    def fake_run(label, command, log_path, *, timeout, env=None):
        del label, log_path, timeout, env
        launched.append(list(command))
        released = Path(command[command.index("--output") + 1])
        released.parent.mkdir(parents=True, exist_ok=True)
        released.write_text(json.dumps([{"task_type": "open"}]), encoding="utf-8")
        return {"status": "complete"}

    monkeypatch.setattr(pipeline, "_run_logged", fake_run)
    result = pipeline._run_questions(
        {"params": str(params)},
        {},
        {"pilot": pilot, "pilot_path": str(pilot_path)},
        {("room_a", "profile_a"): pair},
        tmp_path / "pipeline",
        resume_only=False,
        through_stage="questions",
    )

    assert result["status"] == "pending"
    assert "runtime artifacts" in result["detail"]
    assert launched == []
    assert json.loads(released_path.read_text()) == [{"task_type": "stale"}]



@pytest.mark.parametrize(
    ("backend", "expected_capture", "expected_audio"),
    [
        (None, "ue_capture", "ue_audio"),
        ("habitat_native", "mp3d_capture", "mp3d_audio"),
    ],
)
def test_pair_dispatches_runtime_stages_from_backend_metadata(
    monkeypatch,
    tmp_path: Path,
    backend: str | None,
    expected_capture: str,
    expected_audio: str,
) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    manifest = _write(batch / "batch_manifest.json", {})
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(pipeline, "batch_point_ids", lambda root: ["point_001"])

    def stage(name):
        def run(*args, **kwargs):
            calls.append((name, kwargs.get("backend_id")))
            return {"status": "complete", "root": name}
        return run

    monkeypatch.setattr(pipeline, "_run_capture", stage("ue_capture"))
    monkeypatch.setattr(pipeline, "_run_mp3d_capture", stage("mp3d_capture"))
    monkeypatch.setattr(pipeline, "_run_audio", stage("ue_audio"))
    monkeypatch.setattr(pipeline, "_run_mp3d_audio", stage("mp3d_audio"))
    monkeypatch.setattr(
        pipeline, "_run_declared_captures", stage("declared_capture")
    )
    monkeypatch.setattr(pipeline, "_run_media", stage("media"))
    monkeypatch.setattr(
        pipeline, "_run_declared_media", stage("declared_media")
    )
    monkeypatch.setattr(pipeline, "_run_verifications", stage("verification"))
    monkeypatch.setattr(
        pipeline, "_run_declared_pixels", stage("declared_pixels")
    )
    row = {
        "scene_id": "room_a",
        "profile_id": "profile_a",
        "attempt_status": "generated",
        "batch_manifest": str(manifest),
    }
    if backend is not None:
        row["backend"] = backend

    pair = pipeline._pair_record(
        {"audio_variants": ["main"]},
        {},
        row,
        tmp_path,
        tmp_path / "output",
        params_path=tmp_path / "params.json",
        resume_only=False,
        through_stage="questions",
    )

    expected_backend = backend or "ue_spear"
    assert pair["status"] == "complete"
    assert pair["backend_id"] == expected_backend
    names = [name for name, _ in calls]
    assert expected_capture in names
    assert expected_audio in names
    assert ("media", expected_backend) in calls
    assert ("verification", expected_backend) in calls
    assert ("ue_capture" if backend else "mp3d_capture") not in names
    assert ("ue_audio" if backend else "mp3d_audio") not in names


def test_unknown_backend_fails_before_any_runtime_stage(
    monkeypatch, tmp_path: Path,
) -> None:
    launched = []
    monkeypatch.setattr(
        pipeline,
        "_run_capture",
        lambda *args, **kwargs: launched.append("capture"),
    )
    pair = pipeline._pair_record(
        {"audio_variants": ["main"]},
        {},
        {
            "scene_id": "room_a",
            "profile_id": "profile_a",
            "backend": "unregistered_backend",
            "attempt_status": "generated",
            "batch_manifest": str(tmp_path / "missing.json"),
        },
        tmp_path,
        tmp_path / "output",
        params_path=tmp_path / "params.json",
        resume_only=False,
        through_stage="questions",
    )
    assert pair["status"] == "failed"
    assert "unknown backend_id" in pair["detail"]
    assert launched == []


def _mp3d_pipeline_runtime_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, object]]:
    batch = tmp_path / "batch"
    point = batch / "candidate_a"
    point.mkdir(parents=True)
    _write(point / "fact_record.json", {"backend_id": "habitat_native"})
    _write(point / "case_manifest.json", {"clock": {"frame_count": 150}})
    _write(point / "m1_capture_request.json", {"room_id": "room"})
    room = _write(tmp_path / "room.json", {"room_id": "room"})
    runtime_prefix = tmp_path / "runtime_prefix"
    runtime_prefix.mkdir()
    mp3d_root = tmp_path / "mp3d"
    mp3d_root.mkdir()
    magnum = tmp_path / "magnum"
    magnum.mkdir()
    runtime = {
        "_path": str(tmp_path / "runtime.json"),
        "python": sys.executable,
        "capture": {
            "python": sys.executable,
            "room_manifest": str(room),
            "runtime_prefix": str(runtime_prefix),
            "mp3d_root": str(mp3d_root),
            "magnum_python_site": str(magnum),
            "gpu_device_id": 2,
        },
    }
    return batch, runtime


@pytest.mark.parametrize("frame_count", [75, 150])
def test_mp3d_capture_plan_uses_point_owned_case_and_m1(
    tmp_path: Path,
    frame_count: int,
) -> None:
    batch, runtime = _mp3d_pipeline_runtime_fixture(tmp_path)
    case_path = batch / "candidate_a" / "case_manifest.json"
    case_path.write_text(
        json.dumps({"clock": {"frame_count": frame_count}}),
        encoding="utf-8",
    )
    plan = pipeline._mp3d_capture_plan(
        runtime,
        "any_scene_name",
        "any_profile_name",
        batch,
        "candidate_a",
    )
    assert plan["case_manifest"] == (
        batch / "candidate_a" / "case_manifest.json"
    ).resolve()
    assert plan["m1_request"] == (
        batch / "candidate_a" / "m1_capture_request.json"
    ).resolve()
    assert plan["expected_frames"] == frame_count
    assert plan["gpu_device_id"] == 2


def test_mp3d_capture_does_not_reuse_shared_case_for_missing_point_input(
    tmp_path: Path,
) -> None:
    batch, runtime = _mp3d_pipeline_runtime_fixture(tmp_path)
    local_case = batch / "candidate_a" / "case_manifest.json"
    shared_case = _write(tmp_path / "shared_case.json", {
        "clock": {"frame_count": 150}
    })
    local_case.unlink()
    runtime["capture"]["case_manifest"] = str(shared_case)

    with pytest.raises(pipeline.PipelineError, match="case_manifest is not declared"):
        pipeline._mp3d_capture_plan(
            runtime,
            "scene",
            "profile",
            batch,
            "candidate_a",
        )


def test_mp3d_capture_runner_uses_current_repo_wrapper_and_point_inputs(
    monkeypatch, tmp_path: Path,
) -> None:
    batch, runtime = _mp3d_pipeline_runtime_fixture(tmp_path)
    states = iter([("missing", None), ("complete", None)])
    monkeypatch.setattr(
        pipeline,
        "_mp3d_capture_point_state",
        lambda *args, **kwargs: next(states),
    )
    calls = []

    def fake_run(label, command, log_path, *, timeout, env=None):
        del log_path, timeout, env
        calls.append((label, list(command)))
        return {"status": "complete"}

    monkeypatch.setattr(pipeline, "_run_logged", fake_run)
    result = pipeline._run_mp3d_capture(
        {"audio_variants": ["main"]},
        runtime,
        "scene",
        "profile",
        batch,
        tmp_path / "pair",
        ["candidate_a"],
        resume_only=False,
    )

    assert result["status"] == "complete"
    assert len(calls) == 1
    label, command = calls[0]
    assert label.endswith(":habitat_native")
    assert Path(command[1]).resolve() == pipeline.MP3D_CAPTURE.resolve()
    assert command[command.index("--case-manifest") + 1] == str(
        (batch / "candidate_a" / "case_manifest.json").resolve()
    )
    assert command[command.index("--gpu-device-id") + 1] == "2"



def _add_mp3d_audio_runtime(
    tmp_path: Path,
    batch: Path,
    runtime: dict[str, object],
) -> tuple[Path, Path]:
    point = batch / "candidate_a"
    program = _write(point / "audio_program.json", {
        "program_id": "program-a",
        "events": [],
    })
    _write(point / "source_endpoints.json", {"endpoints": []})
    simulation = _write(tmp_path / "simulation.json", {})
    package = _write(tmp_path / "package.json", {})
    sounds = _write(tmp_path / "sounds.json", {})
    hrtf = tmp_path / "hrtf.sofa"
    hrtf.write_bytes(b"hrtf")
    rlr = tmp_path / "rlr"
    rlr.mkdir()
    dry = tmp_path / "voice.wav"
    dry.write_bytes(b"RIFF")
    runtime["audio"] = {
        "python": sys.executable,
        "simulation_request": str(simulation),
        "package_manifest": str(package),
        "sound_asset_registry": str(sounds),
        "runtime_prefix": runtime["capture"]["runtime_prefix"],
        "rlr_sdk_root": str(rlr),
        "magnum_python_site": runtime["capture"]["magnum_python_site"],
        "layouts": ["binaural", "ambisonics"],
        "hrtf": str(hrtf),
        "sound_asset_paths": {"voice": str(dry)},
    }
    return program, dry


def test_mp3d_audio_runner_uses_configured_program_layouts_and_sound_map(
    monkeypatch, tmp_path: Path,
) -> None:
    batch, runtime = _mp3d_pipeline_runtime_fixture(tmp_path)
    program, dry = _add_mp3d_audio_runtime(tmp_path, batch, runtime)
    states = iter([("missing", None), ("complete", None)])
    monkeypatch.setattr(
        pipeline,
        "_mp3d_audio_point_state",
        lambda *args, **kwargs: next(states),
    )
    calls = []

    def fake_run(label, command, log_path, *, timeout, env=None):
        del log_path, timeout, env
        calls.append((label, list(command)))
        return {"status": "complete"}

    monkeypatch.setattr(pipeline, "_run_logged", fake_run)
    result = pipeline._run_mp3d_audio(
        {"audio_variants": ["main"]},
        runtime,
        "scene",
        "profile",
        batch,
        tmp_path / "pair",
        ["candidate_a"],
        resume_only=False,
    )

    assert result["status"] == "complete"
    assert len(calls) == 1
    label, command = calls[0]
    assert label.endswith("candidate_a:main:habitat_native")
    assert Path(command[0]).resolve() == Path(sys.executable).resolve()
    assert command[1:5] == [
        "-m",
        "avengine.cli",
        "m5",
        "render-current-mp3d-dynamic-audio",
    ]
    assert command[command.index("--audio-program") + 1] == str(
        program.resolve()
    )
    assert command[command.index("--layouts") + 1] == "binaural,ambisonics"
    assert command[command.index("--execution-variant") + 1] == "main"
    assert command[command.index("--sound-asset-path") + 1] == (
        f"voice={dry.resolve()}"
    )
    assert "--beagle-audio" not in command


def test_mp3d_audio_does_not_reuse_shared_program_for_missing_point_input(
    tmp_path: Path,
) -> None:
    batch, runtime = _mp3d_pipeline_runtime_fixture(tmp_path)
    program, _ = _add_mp3d_audio_runtime(tmp_path, batch, runtime)
    program.unlink()
    shared = _write(tmp_path / "shared_program.json", {"events": []})
    runtime["audio"]["audio_program"] = str(shared)

    with pytest.raises(pipeline.PipelineError, match="audio_program is not declared"):
        pipeline._mp3d_audio_plan(
            runtime,
            "scene",
            "profile",
            batch,
            tmp_path / "pair",
            "candidate_a",
            "main",
        )


def test_mp3d_audio_state_rejects_wrong_execution_identity(
    monkeypatch, tmp_path: Path,
) -> None:
    output = tmp_path / "audio"
    output.mkdir()
    program = _write(tmp_path / "program.json", {})
    m1 = _write(tmp_path / "m1.json", {})
    capture = tmp_path / "capture"
    capture.mkdir()
    _write(capture / "frame_records.json", {})
    endpoint = _write(tmp_path / "endpoints.json", {})
    _write(output / "research_receipt.json", {
        "execution_variant": "different",
        "audio_program": {"path": str(program)},
        "inputs": {
            "m1_request": {"path": str(m1)},
            "visual_capture_frame_records": {
                "path": str(capture / "frame_records.json")
            },
            "source_endpoint_registry": {"path": str(endpoint)},
        },
    })
    monkeypatch.setattr(
        pipeline, "audio_point_state", lambda *args, **kwargs: "complete"
    )
    state, detail = pipeline._mp3d_audio_point_state({
        "output": output,
        "layouts": ("binaural",),
        "variant": "main",
        "audio_program": program,
        "m1_request": m1,
        "capture": capture,
        "source_endpoint_registry": endpoint,
    })
    assert state == "partial"
    assert "execution_variant" in detail



def test_mp3d_verifier_receives_point_endpoint_map_without_id_encoding(
    monkeypatch, tmp_path: Path,
) -> None:
    point_ids = ["candidate=one", "candidate_two"]
    endpoints = {}
    for point_id in point_ids:
        endpoint = _write(tmp_path / f"{point_id}.json", {})
        endpoints[point_id] = endpoint
    monkeypatch.setattr(
        pipeline,
        "_capture_states_for_backend",
        lambda *args, **kwargs: (
            "complete",
            {point_id: "complete" for point_id in point_ids},
            None,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_audio_states",
        lambda *args, **kwargs: (
            "complete",
            {point_id: "complete" for point_id in point_ids},
        ),
    )
    program = _write(tmp_path / "program.json", {})
    m1 = _write(tmp_path / "m1.json", {})
    case = _write(tmp_path / "case.json", {})
    room = _write(tmp_path / "room.json", {})
    simulation = _write(tmp_path / "simulation.json", {})
    package = _write(tmp_path / "package.json", {})
    sounds = _write(tmp_path / "sounds.json", {})
    runtime_prefix = tmp_path / "runtime"
    runtime_prefix.mkdir()
    rlr = tmp_path / "rlr"
    rlr.mkdir()
    monkeypatch.setattr(
        pipeline,
        "_mp3d_audio_plan",
        lambda runtime, scene, profile, batch, pair, point_id, variant: {
            "source_endpoint_registry": endpoints[point_id],
            "audio_program": program,
            "program_variant": "A",
            "case_manifest": case,
            "room_manifest": room,
            "m1_request": m1,
            "expected_frames": 150,
            "simulation_request": simulation,
            "package_manifest": package,
            "sound_asset_registry": sounds,
            "runtime_prefix": runtime_prefix,
            "rlr_sdk_root": rlr,
        },
    )
    commands = []

    def fake_run(label, command, log_path, *, timeout, env=None):
        del log_path, timeout, env
        commands.append(list(command))
        output = Path(command[command.index("--out") + 1])
        if label.endswith("/visual"):
            payload = {
                "schema": "qa_v3_visual_batch_verification_v1",
                "status": "pass",
                "counts": {"failures": 0},
                "inputs": {
                    "selection_manifest": command[
                        command.index("--selection-manifest") + 1
                    ],
                    "visual_root": command[
                        command.index("--visual-root") + 1
                    ],
                },
                "points": [
                    {"point_id": point_id} for point_id in point_ids
                ],
            }
        else:
            payload = {
                "schema": "qa_v3_audio_batch_verification_v1",
                "status": "research_candidate",
                "failures": [],
                "expected_variants": ["main"],
                "expected_layouts": ["binaural"],
                "complete_render_point_ids": {"main": point_ids, "gateA": []},
                "checked_renders": len(point_ids),
                "complete_pair_count": 0,
                "audio_root": command[
                    command.index("--audio-root") + 1
                ],
            }
        output.write_text(json.dumps(payload), encoding="utf-8")
        return {"status": "complete"}

    monkeypatch.setattr(pipeline, "_run_logged", fake_run)
    params = _write(tmp_path / "params.json", {})
    result = pipeline._run_verifications(
        {
            "_path": str(tmp_path / "runtime.json"),
            "python": sys.executable,
            "audio": {"layouts": ["binaural"]},
        },
        {"audio_variants": ["main"]},
        "scene",
        "profile",
        tmp_path / "batch",
        tmp_path / "pair",
        point_ids,
        backend_id="habitat_native",
        params_path=params,
        resume_only=False,
    )

    assert result["status"] == "complete"
    audio_command = next(
        command for command in commands
        if Path(command[1]).name == "verify_qa_v3_audio_batch.py"
    )
    endpoint_map = Path(
        audio_command[audio_command.index("--source-endpoint-map") + 1]
    )
    assert json.loads(endpoint_map.read_text()) == {
        point_id: str(path.resolve())
        for point_id, path in endpoints.items()
    }



def test_audio_output_names_reject_point_variant_collision():
    with pytest.raises(pipeline.PipelineError, match="output name collision"):
        pipeline._validate_audio_output_names(
            ["candidate", "candidate_alternate"],
            ["main", "alternate"],
        )



def test_media_state_rejects_receipt_without_verifiable_clock(
    tmp_path: Path,
) -> None:
    media = tmp_path / "media"
    capture = tmp_path / "capture"
    audio = tmp_path / "audio"
    media.mkdir()
    (media / "point.mp4").write_bytes(b"uninspected")
    point_capture = capture / "point"
    point_capture.mkdir(parents=True)
    _write(point_capture / "research_receipt.json", {
        "capture": {"frame_rate_hz": 15}
    })
    mixture = audio / "point" / "audio" / "binaural" / "mixture.wav"
    mixture.parent.mkdir(parents=True)
    mixture.write_bytes(b"audio")

    state, detail = pipeline._media_state(
        media, capture, audio, "point"
    )

    assert state == "failed"
    assert "cannot be verified" in detail



def test_mp3d_audio_state_rejects_wrong_program_variant(
    monkeypatch, tmp_path: Path,
) -> None:
    output = tmp_path / "audio"
    output.mkdir()
    program = _write(tmp_path / "program.json", {})
    m1 = _write(tmp_path / "m1.json", {})
    capture = tmp_path / "capture"
    capture.mkdir()
    _write(capture / "frame_records.json", {})
    endpoint = _write(tmp_path / "endpoints.json", {})
    _write(output / "research_receipt.json", {
        "execution_variant": "main",
        "audio_program": {
            "path": str(program),
            "variant_id": "B",
        },
        "inputs": {
            "m1_request": {"path": str(m1)},
            "visual_capture_frame_records": {
                "path": str(capture / "frame_records.json")
            },
            "source_endpoint_registry": {"path": str(endpoint)},
        },
    })
    monkeypatch.setattr(
        pipeline, "audio_point_state", lambda *args, **kwargs: "complete"
    )
    state, detail = pipeline._mp3d_audio_point_state({
        "output": output,
        "layouts": ("binaural",),
        "variant": "main",
        "program_variant": "A",
        "audio_program": program,
        "m1_request": m1,
        "capture": capture,
        "source_endpoint_registry": endpoint,
    })
    assert state == "partial"
    assert "AudioProgram variant" in detail



def test_verification_resume_rejects_reports_without_matching_context(
    tmp_path: Path,
) -> None:
    pair = tmp_path / "pair"
    verification = pair / "verification"
    verification.mkdir(parents=True)
    point_ids = ["point_001"]
    _write(verification / "visual.json", {
        "schema": "qa_v3_visual_batch_verification_v1",
        "status": "pass",
        "counts": {"failures": 0},
        "inputs": {
            "selection_manifest": str(verification / "selection.json"),
            "visual_root": str(pair / "capture"),
        },
        "points": [{"point_id": "point_001"}],
    })
    _write(verification / "audio.json", {
        "schema": "qa_v3_audio_batch_verification_v1",
        "status": "research_candidate",
        "failures": [],
        "expected_variants": ["main"],
        "expected_layouts": ["binaural"],
        "complete_render_point_ids": {"main": point_ids},
        "checked_renders": 1,
        "audio_root": str(pair / "audio"),
    })

    result = pipeline._run_verifications(
        {
            "_path": str(tmp_path / "runtime.json"),
            "python": sys.executable,
            "audio": {"layouts": ["binaural"]},
        },
        {"audio_variants": ["main"]},
        "scene",
        "profile",
        tmp_path / "batch",
        pair,
        point_ids,
        params_path=tmp_path / "params.json",
        resume_only=False,
    )

    assert result["status"] == "failed"
    assert "no runtime context" in result["detail"]



def test_registered_backend_without_pipeline_adapter_fails_explicitly(
    monkeypatch,
):
    class Handler:
        backend_id = "future_backend"

    monkeypatch.setattr(
        pipeline, "get_backend_handler", lambda row: Handler()
    )
    with pytest.raises(
        pipeline.PipelineError, match="no QA pipeline adapter"
    ):
        pipeline._backend_id({"backend": "future_backend"})



def test_assembly_keeps_runtime_only_candidate_pending_without_failure(
    monkeypatch, tmp_path: Path,
) -> None:
    design_root = tmp_path / "design"
    batch = design_root / "batch"
    point = batch / "point_001"
    point.mkdir(parents=True)
    _write(point / "fact_record.json", {
        "runtime_consumer_status": "pending_question_facts"
    })
    _write(point / "timeline.json", {"frames": []})
    batch_manifest = _write(batch / "batch_manifest.json", {
        "status": "research_candidate",
        "records": [{"point_id": "point_001"}],
    })
    matrix = {
        "matrix": [{
            "scene_id": "room",
            "profile_id": "profile",
            "attempt_status": "generated",
            "batch_manifest": str(batch_manifest),
        }]
    }
    _write(design_root / "scene_profile_matrix.json", matrix)
    monkeypatch.setattr(
        pipeline,
        "_run_logged",
        lambda *args, **kwargs: pytest.fail(
            "question assembler must not run for pending runtime facts"
        ),
    )

    result = pipeline._assemble(
        {"profiles": tmp_path / "profiles.json"},
        {},
        {
            "status": "complete",
            "root": str(design_root),
            "matrix": matrix,
        },
        tmp_path / "output",
        resume_only=False,
        through_stage="questions",
    )

    assert result["status"] == "pending"
    assert result["pending_question_facts"] == [
        "room/profile/point_001"
    ]
