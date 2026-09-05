"""Engine-completion runtime: request paths, variants, resume, capture maps."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools" / "qa"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(REPO / "src"))

from avengine.contracts.json_io import sha256_file  # noqa: E402
from avengine.qa.pixel_visibility import (  # noqa: E402
    PIXEL_VISIBILITY_DEPTH_AUTHORITY,
    PIXEL_VISIBILITY_SCHEMA,
)
from avengine.qa.runtime_artifacts import declared_audio_variants, load_runtime_artifacts  # noqa: E402
from qa_v3_request import QARequestError, resolve_request_resource  # noqa: E402
import build_qa_v3_released_probe_items as released  # noqa: E402
import run_qa_v3_audio_batch as audio_batch  # noqa: E402
import run_qa_v3_pipeline as pipeline  # noqa: E402


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (bytes, bytearray)):
        path.write_bytes(value)
        return path
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
        return path
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _clock_timeline() -> dict:
    return {"render": {"frame_count": 75}, "frames": [{}] * 75}


def _card17_files(point: Path) -> None:
    point.mkdir(parents=True, exist_ok=True)
    for name in (
        "timeline.json",
        "timeline_segment2.json",
        "actor_selection.json",
        "actor_selection_gateB.json",
        "timeline_gateB.json",
    ):
        _write(point / name, _clock_timeline() if "timeline" in name else {"actors": []})


def _card17_release_list() -> list[dict]:
    return [
        {
            "id": "segment1",
            "variant": "main",
            "segment": "segment1",
            "audio_variant": "main",
            "release": True,
        },
        {
            "id": "segment2",
            "variant": "main",
            "segment": "segment2",
            "audio_variant": "segment2",
            "release": True,
        },
    ]


def _card17_release_mapping() -> dict:
    return {
        "segment1": {
            "variant": "main",
            "segment": "segment1",
            "audio_variant": "main",
            "release": True,
        },
        "segment2": {
            "variant": "main",
            "segment": "segment2",
            "audio_variant": "segment2",
            "release": True,
        },
    }


def _card17_fact(release_media) -> dict:
    return {
        "profile_id": "card17",
        "visual_variants": [
            {
                "id": "main",
                "actor_selection": "actor_selection.json",
                "timeline": "timeline.json",
            },
            {
                "id": "gateB",
                "actor_selection": "actor_selection_gateB.json",
                "timeline": "timeline_gateB.json",
            },
        ],
        "segments": [
            {
                "id": "segment1",
                "variant": "main",
                "actor_selection": "actor_selection.json",
                "timeline": "timeline.json",
            },
            {
                "id": "segment2",
                "variant": "main",
                "actor_selection": "actor_selection.json",
                "timeline": "timeline_segment2.json",
            },
        ],
        "release_media": release_media,
    }


def test_example_requests_use_repo_relative_source_paths() -> None:
    checkout = "/data/jzy/tmp/wt-qa-v3-engine-completion/"
    for path in sorted((REPO / "examples/qa/requests").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("profiles", "params"):
            assert not str(data[key]).startswith(checkout), path
            assert not Path(data[key]).is_absolute(), path
        for item in data["scene_configs"]:
            assert not str(item).startswith(checkout), path
            assert not Path(item).is_absolute(), path
        assert Path(data["snapshot_content"]).is_absolute()
        loaded = pipeline._load_request(path)
        assert loaded["profiles"].is_file()
        assert loaded["params"].is_file()
        assert loaded["scene_configs"][0].is_file()


def test_card17_default_request_declares_segment2() -> None:
    path = REPO / "examples/qa/requests/card17_segment2_default.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "segment2" in data["audio_variants"]
    loaded = pipeline._load_request(path)
    assert "segment2" in loaded["audio_variants"]


def test_request_resource_resolves_from_repo_root_or_request_file(tmp_path: Path) -> None:
    target = REPO / "examples/qa/qa_v3_current_profiles_v1.json"
    request = tmp_path / "nested" / "request.json"
    request.parent.mkdir(parents=True)
    request.write_text("{}", encoding="utf-8")
    resolved = resolve_request_resource(
        "examples/qa/qa_v3_current_profiles_v1.json",
        request_file=request,
        owner="profiles",
        repo_root=REPO,
    )
    assert resolved == target.resolve()
    local = _write(request.parent / "local.json", {"ok": True})
    assert resolve_request_resource(
        "local.json", request_file=request, owner="local", repo_root=REPO
    ) == local.resolve()
    with pytest.raises(QARequestError, match="missing"):
        resolve_request_resource(
            "does_not_exist.json", request_file=request, owner="gone", repo_root=REPO
        )


def test_verification_context_unions_declared_segment2(tmp_path: Path) -> None:
    point = tmp_path / "card17_001"
    _card17_files(point)
    _write(point / "fact_record.json", _card17_fact(_card17_release_list()))
    request = {"audio_variants": ["main", "gateA"]}
    variants = pipeline._audio_variants_for_pair(request, tmp_path, ["card17_001"])
    assert variants == ["main", "gateA", "segment2"]
    context = pipeline._verification_context(
        {}, request, "apartment_0000", "card17", "spear_ue",
        tmp_path, tmp_path / "pair", ["card17_001"], ["binaural"],
    )
    assert context["audio_variants"] == ["main", "gateA", "segment2"]


def test_list_and_mapping_release_media_yield_the_same_segment2_variants(
    tmp_path: Path,
) -> None:
    list_point = tmp_path / "list" / "card17_001"
    map_point = tmp_path / "map" / "card17_001"
    _card17_files(list_point)
    _card17_files(map_point)
    list_fact = _card17_fact(_card17_release_list())
    map_fact = _card17_fact(_card17_release_mapping())
    _write(list_point / "fact_record.json", list_fact)
    _write(map_point / "fact_record.json", map_fact)
    request = {"audio_variants": ["main", "gateA"]}
    list_variants = pipeline._audio_variants_for_pair(
        request, tmp_path / "list", ["card17_001"]
    )
    map_variants = pipeline._audio_variants_for_pair(
        request, tmp_path / "map", ["card17_001"]
    )
    assert list_variants == ["main", "gateA", "segment2"]
    assert map_variants == list_variants
    list_plan = load_runtime_artifacts(list_point)
    map_plan = load_runtime_artifacts(map_point)
    assert declared_audio_variants(list_plan) == declared_audio_variants(map_plan)
    assert released.extra_audio_variants_from_fact(
        list_fact, point_dir=list_point
    ) == ["segment2"]
    assert released.extra_audio_variants_from_fact(
        map_fact, point_dir=map_point
    ) == ["segment2"]


def test_raw_mapping_release_media_is_not_silently_dropped(tmp_path: Path) -> None:
    point = tmp_path / "card17_001"
    _card17_files(point)
    fact = _card17_fact(_card17_release_mapping())
    _write(point / "fact_record.json", fact)
    # Direct fact.get("release_media") iteration used to see mapping keys, not rows.
    assert not isinstance(next(iter(fact["release_media"])), dict)
    variants = pipeline._audio_variants_for_pair(
        {"audio_variants": ["main"]}, tmp_path, ["card17_001"]
    )
    assert variants == ["main", "segment2"]


def test_capture_by_variant_rejects_unknown_missing_and_duplicate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown point"):
        audio_batch.load_capture_by_variant_map(
            {"other": {"main": "/cap"}},
            point_ids=["card17_001"],
            variants=["main"],
        )
    with pytest.raises(ValueError, match="missing point"):
        audio_batch.load_capture_by_variant_map(
            {},
            point_ids=["card17_001"],
            variants=["main"],
        )
    with pytest.raises(ValueError, match="unknown variant"):
        audio_batch.load_capture_by_variant_map(
            {"card17_001": {"main": "/cap", "ghost": "/cap2"}},
            point_ids=["card17_001"],
            variants=["main"],
        )
    with pytest.raises(ValueError, match="missing variant"):
        audio_batch.load_capture_by_variant_map(
            {"card17_001": {"main": "/cap"}},
            point_ids=["card17_001"],
            variants=["main", "segment2"],
        )
    mapping = audio_batch.load_capture_by_variant_map(
        {"card17_001": {"main": "/cap/main", "segment2": "/cap/seg2"}},
        point_ids=["card17_001"],
        variants=["main", "segment2"],
    )
    assert mapping["card17_001"]["segment2"] == "/cap/seg2"

    point = tmp_path / "card17_001"
    _card17_files(point)
    fact = _card17_fact(_card17_release_list())
    fact["release_media"] = [
        {
            "id": "a",
            "variant": "main",
            "segment": "segment1",
            "audio_variant": "segment2",
            "release": True,
        },
        {
            "id": "b",
            "variant": "main",
            "segment": "segment2",
            "audio_variant": "segment2",
            "release": True,
        },
    ]
    _write(point / "fact_record.json", fact)
    with pytest.raises(pipeline.PipelineError, match="duplicate mappings"):
        pipeline._audio_capture_by_variant(
            tmp_path, tmp_path / "pair", ["card17_001"], ["main", "segment2"]
        )


def test_capture_by_variant_json_rejects_duplicate_keys() -> None:
    text = '{"card17_001": {"main": "/a"}, "card17_001": {"main": "/b"}}'
    with pytest.raises(ValueError, match="duplicate JSON key"):
        audio_batch.load_json_rejecting_duplicate_keys(text)
    nested = '{"card17_001": {"main": "/a", "main": "/b"}}'
    with pytest.raises(ValueError, match="duplicate JSON key"):
        audio_batch.load_json_rejecting_duplicate_keys(nested)


def _complete_pixel_output(
    output: Path,
    *,
    actor: Path,
    timeline: Path,
    frames: list[int],
    slot: str = "source1",
    height: int = 4,
    width: int = 6,
    drop: str | None = None,
) -> None:
    rgb = output / "rgb_frames"
    rgb.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        (rgb / f"frame_{frame:06d}.png").write_bytes(b"png")
    count = len(frames)
    arrays = {
        "normal_depth_m": np.zeros((count, height, width), dtype=np.float32),
        "normal_object_ids_uint32": np.zeros((count, height, width), dtype=np.uint32),
        f"target_only_{slot}_depth_m": np.zeros((count, height, width), dtype=np.float32),
    }
    if drop == "target":
        arrays.pop(f"target_only_{slot}_depth_m")
    if drop == "frame_axis":
        arrays["normal_depth_m"] = np.zeros((count + 1, height, width), dtype=np.float32)
    np.savez_compressed(output / "native_depth_and_object_ids.npz", **arrays)
    truth = {
        "schema": PIXEL_VISIBILITY_SCHEMA,
        "status": "computed_modal_target_only_v1",
        "authority": PIXEL_VISIBILITY_DEPTH_AUTHORITY,
        "frame_indices": list(frames),
        "resolution_hw": [height, width],
        "per_instance": {slot: {"semantic_id": 1}},
    }
    _write(output / "pixel_visibility_truth.json", truth)
    _write(
        output / "evidence.json",
        {
            "schema": "qa_v3_current_timeline_native_pixel_probe_v1",
            "status": "pass",
            "inputs": {
                "actor_selection": str(actor.resolve()),
                "timeline": str(timeline.resolve()),
            },
            "frame_indices": list(frames),
            "artifacts": {
                "rgb_frames": "rgb_frames",
                "arrays": "native_depth_and_object_ids.npz",
                "truth": "pixel_visibility_truth.json",
            },
        },
    )


def _pixel_producer_point(tmp_path: Path) -> tuple[Path, Path, Path]:
    point = tmp_path / "card11_001"
    point.mkdir()
    actor = _write(point / "actor_selection.json", {"actors": []})
    timeline = _write(point / "timeline.json", _clock_timeline())
    _write(
        point / "fact_record.json",
        {
            "pixel_producers": [{
                "id": "main",
                "kind": "qa_v3_timeline_native_pixel",
                "actor_selection": "actor_selection.json",
                "timeline": "timeline.json",
                "binding_frames": [30],
            }],
        },
    )
    return point, actor, timeline


def _capture_runtime(tmp_path: Path) -> dict:
    spear_ext = tmp_path / "spear_ext"
    spear_ext.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    return {
        "python": sys.executable,
        "capture": {
            "python": sys.executable,
            "spear_ext": str(spear_ext),
            "closure_report": str(_write(tmp_path / "closure.json", {})),
            "stage_root": str(stage),
            "spear_executable": str(_write(tmp_path / "spear", "#!/bin/sh\n")),
            "source_asset_registry": str(_write(tmp_path / "registry.json", {"assets": []})),
        },
    }


def test_pixel_producer_resume_fails_closed_on_incomplete_or_stale_products(
    tmp_path: Path, monkeypatch
) -> None:
    _pixel_producer_point(tmp_path)
    actor = tmp_path / "card11_001" / "actor_selection.json"
    timeline = tmp_path / "card11_001" / "timeline.json"
    output = tmp_path / "pair" / "declared_pixels" / "card11_001" / "main"
    output.mkdir(parents=True)
    _write(output / "pixel_visibility_truth.json", {"status": "computed"})
    (output / "rgb_frames").mkdir()
    monkeypatch.setattr(
        pipeline,
        "_run_logged",
        lambda *args, **kwargs: pytest.fail("incomplete pixel products must not relaunch"),
    )
    result = pipeline._run_declared_pixel_producers(
        {}, "apartment_0000", "card11", tmp_path, tmp_path / "pair",
        ["card11_001"], resume_only=True,
    )
    assert result["status"] == "failed"
    assert "evidence.json is missing" in result["records"][0]["detail"]

    _complete_pixel_output(output, actor=actor, timeline=timeline, frames=[12])
    result = pipeline._run_declared_pixel_producers(
        {}, "apartment_0000", "card11", tmp_path, tmp_path / "pair",
        ["card11_001"], resume_only=True,
    )
    assert result["status"] == "failed"
    assert "binding_frames" in result["records"][0]["detail"]

    _complete_pixel_output(output, actor=actor, timeline=timeline, frames=[30])
    result = pipeline._run_declared_pixel_producers(
        {}, "apartment_0000", "card11", tmp_path, tmp_path / "pair",
        ["card11_001"], resume_only=True,
    )
    assert result["status"] == "complete"
    assert result["records"][0]["run"]["status"] == "reused_existing"


def test_pixel_producer_run_that_only_writes_truth_is_failed(
    tmp_path: Path, monkeypatch
) -> None:
    _pixel_producer_point(tmp_path)

    def fake_run(label, command, log_path, timeout):
        del label, log_path, timeout
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        _write(output / "pixel_visibility_truth.json", {"status": "computed"})
        return {"status": "complete"}

    monkeypatch.setattr(pipeline, "_run_logged", fake_run)
    result = pipeline._run_declared_pixel_producers(
        _capture_runtime(tmp_path),
        "apartment_0000",
        "card11",
        tmp_path,
        tmp_path / "pair",
        ["card11_001"],
        resume_only=False,
    )
    assert result["status"] == "failed"
    assert result["records"][0]["status"] == "failed"
    assert "evidence.json is missing" in result["records"][0]["detail"]


def test_pixel_producer_npz_requires_named_arrays_and_matching_frame_axis(
    tmp_path: Path, monkeypatch
) -> None:
    _pixel_producer_point(tmp_path)
    actor = tmp_path / "card11_001" / "actor_selection.json"
    timeline = tmp_path / "card11_001" / "timeline.json"
    output = tmp_path / "pair" / "declared_pixels" / "card11_001" / "main"
    monkeypatch.setattr(
        pipeline,
        "_run_logged",
        lambda *args, **kwargs: pytest.fail("invalid NPZ must not relaunch"),
    )
    _complete_pixel_output(
        output, actor=actor, timeline=timeline, frames=[30], drop="target"
    )
    result = pipeline._run_declared_pixel_producers(
        {}, "apartment_0000", "card11", tmp_path, tmp_path / "pair",
        ["card11_001"], resume_only=True,
    )
    assert result["status"] == "failed"
    assert "target_only_source1_depth_m" in result["records"][0]["detail"]

    _complete_pixel_output(
        output, actor=actor, timeline=timeline, frames=[30], drop="frame_axis"
    )
    result = pipeline._run_declared_pixel_producers(
        {}, "apartment_0000", "card11", tmp_path, tmp_path / "pair",
        ["card11_001"], resume_only=True,
    )
    assert result["status"] == "failed"
    assert "frame axis" in result["records"][0]["detail"]


def test_missing_or_unreadable_fact_records_pixel_producer_failed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        pipeline,
        "_run_logged",
        lambda *args, **kwargs: pytest.fail("missing fact must not launch"),
    )
    missing = pipeline._run_declared_pixel_producers(
        {}, "apartment_0000", "card11", tmp_path, tmp_path / "pair",
        ["gone"], resume_only=False,
    )
    assert missing["status"] == "failed"
    assert missing["records"][0]["status"] == "failed"
    assert "missing" in missing["records"][0]["detail"]

    point = tmp_path / "broken"
    point.mkdir()
    (point / "fact_record.json").write_text("{", encoding="utf-8")
    unreadable = pipeline._run_declared_pixel_producers(
        {}, "apartment_0000", "card11", tmp_path, tmp_path / "pair",
        ["broken"], resume_only=False,
    )
    assert unreadable["status"] == "failed"
    assert unreadable["records"][0]["status"] == "failed"
    assert "unreadable" in unreadable["records"][0]["detail"]


def _pixel_consumer_point(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    point = tmp_path / "card11_001"
    point.mkdir()
    _write(point / "actor_selection.json", {"actors": []})
    _write(point / "timeline.json", _clock_timeline())
    fact = _write(
        point / "fact_record.json",
        {
            "profile_id": "card11",
            "pixel_evidence": [{"id": "main", "kind": "qa_v3_extended_pixel"}],
        },
    )
    truth = _write(
        tmp_path / "truth.json",
        {
            "schema": PIXEL_VISIBILITY_SCHEMA,
            "status": "computed_modal_target_only_v1",
            "authority": PIXEL_VISIBILITY_DEPTH_AUTHORITY,
            "frame_indices": [30],
            "per_instance": {},
        },
    )
    params = _write(tmp_path / "params.json", {})
    return point, fact, truth, params


def _consumer_truth_and_output(tmp_path: Path, truth: Path) -> tuple[Path, Path]:
    produced_dir = tmp_path / "pair" / "declared_pixels" / "card11_001" / "main"
    produced_dir.mkdir(parents=True)
    produced_truth = produced_dir / "pixel_visibility_truth.json"
    produced_truth.write_bytes(truth.read_bytes())
    output = tmp_path / "pair" / "declared_pixels" / "card11_001" / "main.json"
    return produced_truth, output


def test_pixel_consumer_pass_without_inputs_is_not_reuse(
    tmp_path: Path, monkeypatch
) -> None:
    _pixel_consumer_point(tmp_path)
    truth = tmp_path / "truth.json"
    _produced_truth, output = _consumer_truth_and_output(tmp_path, truth)
    _write(output, {"status": "pass"})
    monkeypatch.setattr(
        pipeline,
        "_run_logged",
        lambda *args, **kwargs: pytest.fail("stale pixel output must not relaunch"),
    )
    result = pipeline._run_declared_pixels(
        {},
        "apartment_0000",
        "card11",
        tmp_path,
        tmp_path / "pair",
        ["card11_001"],
        params_path=tmp_path / "params.json",
        resume_only=False,
    )
    assert result["status"] == "failed"
    assert "missing inputs" in result["records"][0]["detail"]


def test_pixel_consumer_pass_with_changed_input_requires_fresh_output(
    tmp_path: Path, monkeypatch
) -> None:
    _point, fact, truth, params = _pixel_consumer_point(tmp_path)
    produced_truth, output = _consumer_truth_and_output(tmp_path, truth)
    _write(
        output,
        {
            "status": "pass",
            "inputs": {
                "fact": {"path": str(fact.resolve()), "sha256": sha256_file(fact)},
                "pixel_truth": {
                    "path": str(produced_truth.resolve()),
                    "sha256": sha256_file(produced_truth),
                },
                "params": {"path": str(params.resolve()), "sha256": sha256_file(params)},
            },
        },
    )
    params.write_text(json.dumps({"changed": True}), encoding="utf-8")
    monkeypatch.setattr(
        pipeline,
        "_run_logged",
        lambda *args, **kwargs: pytest.fail("changed inputs must not relaunch over old output"),
    )
    result = pipeline._run_declared_pixels(
        {},
        "apartment_0000",
        "card11",
        tmp_path,
        tmp_path / "pair",
        ["card11_001"],
        params_path=params,
        resume_only=False,
    )
    assert result["status"] == "failed"
    assert "sha256" in result["records"][0]["detail"]
    assert "fresh output" in result["records"][0]["detail"]


def test_pixel_consumer_reuses_when_joiner_inputs_match_current_files(
    tmp_path: Path, monkeypatch
) -> None:
    _point, fact, truth, params = _pixel_consumer_point(tmp_path)
    produced_truth, output = _consumer_truth_and_output(tmp_path, truth)
    _write(
        output,
        {
            "status": "pass",
            "inputs": {
                "fact": {"path": str(fact.resolve()), "sha256": sha256_file(fact)},
                "pixel_truth": {
                    "path": str(produced_truth.resolve()),
                    "sha256": sha256_file(produced_truth),
                },
                "params": {"path": str(params.resolve()), "sha256": sha256_file(params)},
            },
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_run_logged",
        lambda *args, **kwargs: pytest.fail("matching inputs must reuse"),
    )
    result = pipeline._run_declared_pixels(
        {},
        "apartment_0000",
        "card11",
        tmp_path,
        tmp_path / "pair",
        ["card11_001"],
        params_path=params,
        resume_only=False,
    )
    assert result["status"] == "complete"
    assert result["records"][0]["run"]["status"] == "reused_existing"
