"""Engine-completion runtime: request paths, variants, resume, capture maps."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools" / "qa"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(REPO / "src"))

from qa_v3_request import QARequestError, resolve_request_resource  # noqa: E402
import run_qa_v3_pipeline as pipeline  # noqa: E402
import run_qa_v3_audio_batch as audio_batch  # noqa: E402
from avengine.qa.pixel_visibility import (  # noqa: E402
    PIXEL_VISIBILITY_DEPTH_AUTHORITY,
    PIXEL_VISIBILITY_SCHEMA,
)


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
    point.mkdir()
    for name in (
        "timeline.json",
        "timeline_segment2.json",
        "actor_selection.json",
        "actor_selection_gateB.json",
        "timeline_gateB.json",
    ):
        _write(point / name, {"render": {"frame_count": 75}, "frames": [{}] * 75})
    fact = {
        "visual_variants": [
            {"id": "main", "actor_selection": "actor_selection.json", "timeline": "timeline.json"},
            {"id": "gateB", "actor_selection": "actor_selection_gateB.json", "timeline": "timeline_gateB.json"},
        ],
        "segments": [
            {"id": "segment1", "variant": "main", "actor_selection": "actor_selection.json", "timeline": "timeline.json"},
            {"id": "segment2", "variant": "main", "actor_selection": "actor_selection.json", "timeline": "timeline_segment2.json"},
        ],
        "release_media": [
            {"id": "segment1", "variant": "main", "segment": "segment1", "audio_variant": "main", "release": True},
            {"id": "segment2", "variant": "main", "segment": "segment2", "audio_variant": "segment2", "release": True},
        ],
    }
    _write(point / "fact_record.json", fact)
    request = {"audio_variants": ["main", "gateA"]}
    variants = pipeline._audio_variants_for_pair(request, tmp_path, ["card17_001"])
    assert variants == ["main", "gateA", "segment2"]
    context = pipeline._verification_context(
        {}, request, "apartment_0000", "card17", "spear_ue",
        tmp_path, tmp_path / "pair", ["card17_001"], ["binaural"],
    )
    assert context["audio_variants"] == ["main", "gateA", "segment2"]


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
    point.mkdir()
    for name in (
        "timeline.json",
        "timeline_segment2.json",
        "actor_selection.json",
        "actor_selection_gateB.json",
        "timeline_gateB.json",
    ):
        _write(point / name, {"render": {"frame_count": 75}, "frames": [{}] * 75})
    fact = {
        "visual_variants": [
            {"id": "main", "actor_selection": "actor_selection.json", "timeline": "timeline.json"},
        ],
        "segments": [
            {"id": "segment1", "variant": "main", "actor_selection": "actor_selection.json", "timeline": "timeline.json"},
            {"id": "segment2", "variant": "main", "actor_selection": "actor_selection.json", "timeline": "timeline_segment2.json"},
        ],
        "release_media": [
            {"id": "a", "variant": "main", "segment": "segment1", "audio_variant": "segment2", "release": True},
            {"id": "b", "variant": "main", "segment": "segment2", "audio_variant": "segment2", "release": True},
        ],
    }
    _write(point / "fact_record.json", fact)
    with pytest.raises(pipeline.PipelineError, match="duplicate mappings"):
        pipeline._audio_capture_by_variant(
            tmp_path, tmp_path / "pair", ["card17_001"], ["main", "segment2"]
        )


def _complete_pixel_output(output: Path, *, actor: Path, timeline: Path, frames: list[int]) -> None:
    rgb = output / "rgb_frames"
    rgb.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        (rgb / f"frame_{frame:06d}.png").write_bytes(b"png")
    npz = output / "native_depth_and_object_ids.npz"
    with zipfile.ZipFile(npz, "w") as archive:
        archive.writestr("normal_depth_m.npy", b"array")
    truth = {
        "schema": PIXEL_VISIBILITY_SCHEMA,
        "status": "computed_modal_target_only_v1",
        "authority": PIXEL_VISIBILITY_DEPTH_AUTHORITY,
        "frame_indices": list(frames),
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


def test_pixel_producer_resume_fails_closed_on_incomplete_or_stale_products(
    tmp_path: Path, monkeypatch
) -> None:
    point = tmp_path / "card11_001"
    point.mkdir()
    actor = _write(point / "actor_selection.json", {"actors": []})
    timeline = _write(
        point / "timeline.json",
        {"render": {"frame_count": 75}, "frames": [{}] * 75},
    )
    fact = {
        "pixel_producers": [{
            "id": "main",
            "kind": "qa_v3_timeline_native_pixel",
            "actor_selection": "actor_selection.json",
            "timeline": "timeline.json",
            "binding_frames": [30],
        }],
    }
    _write(point / "fact_record.json", fact)
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
