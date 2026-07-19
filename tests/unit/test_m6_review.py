from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from avengine.contracts.json_io import file_record, load_json, write_json
from avengine.m6.review import (
    CANONICAL_ROLES,
    M6ReviewError,
    build_six_case_review,
    plan_six_case_review,
    validate_six_case_review_request,
    verify_six_case_review,
)


ROOT = Path(__file__).resolve().parents[2]
REQUEST = ROOT / "examples" / "m6" / "review" / "six_case_review_request.json"


def _request() -> dict:
    return load_json(REQUEST)


def test_example_has_exact_six_case_semantics() -> None:
    request = _request()
    assert validate_six_case_review_request(request) == []
    assert tuple(case["role"] for case in request["cases"]) == CANONICAL_ROLES
    assert len(
        {
            case["room_lineage_id"]
            for case in request["cases"]
            if case["is_room"]
        }
    ) == 4
    raw, derived = request["cases"][3:5]
    assert derived["room_lineage_id"] == raw["room_lineage_id"]
    assert derived["visual"] == {
        "availability": "shared",
        "reuse_from_case_id": raw["case_id"],
    }
    assert request["cases"][5]["is_room"] is False
    assert "room_id" not in request["cases"][5]


def test_mp3d_derived_cannot_claim_qualified_subject_or_audio() -> None:
    request = _request()
    derived = request["cases"][4]
    derived["status"]["qualification"] = "qualified"
    derived["status"]["value"] = "pass"
    derived["audio"]["evidence_tier"] = "qualified"
    errors = validate_six_case_review_request(request)
    assert any("MP3D derived is research-only" in error for error in errors)
    assert any("subject status cannot be pass" in error for error in errors)
    assert any("derived audio cannot be labelled qualified" in error for error in errors)


def test_cross_case_validator_rejects_six_distinct_mp3d_visual_lineage() -> None:
    request = _request()
    derived = request["cases"][4]
    derived["room_lineage_id"] = "incorrect_independent_mp3d_lineage"
    derived["visual"] = {
        "availability": "available",
        "path": "tmp/incorrect.mp4",
        "start_seconds": 0,
    }
    errors = validate_six_case_review_request(request)
    assert any("exactly four room lineages" in error for error in errors)
    assert any("same room lineage" in error for error in errors)
    assert any("explicitly reuse" in error for error in errors)


def test_corrupted_fixture_must_remain_a_silent_non_room() -> None:
    request = _request()
    corrupted = request["cases"][5]
    corrupted["is_room"] = True
    corrupted["room_id"] = "fake_room"
    corrupted["room_lineage_id"] = "fake_lineage"
    corrupted["audio"] = {
        "availability": "available",
        "path": "tmp/fake.wav",
        "start_seconds": 0,
        "channel_count": 2,
        "presentation_format": "binaural",
        "evidence_tier": "unqualified",
    }
    errors = validate_six_case_review_request(request)
    assert any("is_room must be False" in error for error in errors)
    assert any("must not bind room fields" in error for error in errors)
    assert any("cannot provide room audio" in error for error in errors)


def test_dry_run_commands_add_silence_and_preserve_mp3d_claim_scope(
    tmp_path: Path,
) -> None:
    request = _request()
    plan = plan_six_case_review(
        request,
        repository_root=ROOT,
        staging_directory=tmp_path / "future_staging",
        check_media=False,
    )
    assert len(plan["segments"]) == 6
    raw = plan["segments"][3]
    derived = plan["segments"][4]
    replicacad = plan["segments"][1]
    corrupted = plan["segments"][5]
    assert "anullsrc=channel_layout=stereo:sample_rate=48000" in raw["command"]
    assert "anullsrc=channel_layout=stereo:sample_rate=48000" not in replicacad[
        "command"
    ]
    assert replicacad["resolved_audio"].name == (
        "replicacad_human_beagle_annotated_binaural.mp4"
    )
    assert any(
        "AUDIO=BINAURAL / RESEARCH ONLY" in line
        for line in replicacad["title_lines"]
    )
    assert "anullsrc=channel_layout=stereo:sample_rate=48000" in corrupted["command"]
    assert any("drawtext=" in argument for argument in raw["command"])
    assert any(
        line.startswith("AUDIO UNAVAILABLE") for line in raw["title_lines"]
    )
    assert any(
        "RESEARCH ONLY (NOT QUALIFIED)" in line
        for line in derived["title_lines"]
    )
    assert derived["resolved_visual"] == raw["resolved_visual"]
    assert derived["reused_visual_case_id"] == raw["case"]["case_id"]
    for segment in plan["segments"][:5]:
        filter_index = segment["command"].index("-filter_complex")
        filter_graph = segment["command"][filter_index + 1]
        match = re.search(r"drawbox=x=0:y=0:w=iw:h=(\d+)", filter_graph)
        assert match is not None
        assert int(match.group(1)) <= 120
    assert plan["concat_command"][plan["concat_command"].index("-f") + 1] == "concat"


def _synthetic_source(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is unavailable")
    path.parent.mkdir(parents=True)
    completed = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10:duration=0.6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=0.6",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-ar",
            "16000",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.media_readback
def test_builder_encodes_synthetic_six_case_package_and_is_no_clobber(
    tmp_path: Path,
) -> None:
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is unavailable")
    source = tmp_path / "media" / "source.mp4"
    _synthetic_source(source)
    request = deepcopy(_request())
    request["review_id"] = "synthetic_six_case_review"
    request["output_profile"] = {
        "width": 320,
        "height": 240,
        "frame_rate_hz": 10,
        "segment_duration_seconds": 0.3,
        "audio_sample_rate_hz": 16000,
    }
    for case in request["cases"]:
        if case["visual"]["availability"] == "available":
            case["visual"]["path"] = "media/source.mp4"
            case["visual"]["start_seconds"] = 0
        if case["audio"]["availability"] == "available":
            case["audio"]["path"] = "media/source.mp4"
            case["audio"]["start_seconds"] = 0
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(request, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output = tmp_path / "review_output"
    manifest_path, video_path = build_six_case_review(
        request_path=request_path,
        output_directory=output,
        repository_root=tmp_path,
    )
    assert manifest_path == output / "review_manifest.json"
    assert video_path == output / "m6_six_case_review.mp4"
    assert video_path.is_file()
    assert len(tuple((output / "segments").glob("*.mp4"))) == 6
    manifest = load_json(manifest_path)
    assert manifest["case_semantics"] == {
        "case_count": 6,
        "corrupted_fixture_is_room": False,
        "mp3d_raw_derived_share_visual": True,
        "real_room_lineage_count": 4,
    }
    assert manifest["segments"][3]["audio_input"]["render_mode"] == (
        "silent_unavailable"
    )
    assert manifest["segments"][4]["audio_input"]["evidence_tier"] == (
        "research_only"
    )
    assert manifest["segments"][4]["status"]["qualification"] == "unqualified"
    assert manifest["combined_video"]["probe"]["duration_seconds"] == pytest.approx(
        1.8, abs=0.1
    )
    verification = verify_six_case_review(
        manifest_path, repository_root=tmp_path
    )
    assert verification["status"] == "pass"
    assert [check["check_id"] for check in verification["checks"]] == [
        "manifest_json",
        "manifest_schema",
        "request_binding",
        "case_semantics",
        "source_media",
        "segment_media",
        "combined_media",
        "package_closure",
    ]

    original_manifest = deepcopy(manifest)
    wrong_source = file_record(output / "request.json", relative_to=tmp_path)
    wrong_source.update(
        {
            "availability": "available",
            "evidence_tier": "research_only",
            "render_mode": "source_audio",
        }
    )
    manifest["segments"][0]["audio_input"] = wrong_source
    write_json(manifest_path, manifest)
    wrong_audio = verify_six_case_review(manifest_path, repository_root=tmp_path)
    source_check = next(
        check for check in wrong_audio["checks"] if check["check_id"] == "source_media"
    )
    assert source_check["status"] == "fail"
    assert any("audio_input differs" in error for error in source_check["errors"])
    write_json(manifest_path, original_manifest)
    manifest = original_manifest

    extra = output / "undeclared.txt"
    extra.write_text("not in manifest\n", encoding="utf-8")
    extra_report = verify_six_case_review(manifest_path, repository_root=tmp_path)
    closure_check = next(
        check
        for check in extra_report["checks"]
        if check["check_id"] == "package_closure"
    )
    assert closure_check["status"] == "fail"
    assert any("undeclared file" in error for error in closure_check["errors"])
    extra.unlink()

    outside = tmp_path / "outside"
    outside.mkdir()
    directory_link = output / "extra_directory_link"
    directory_link.symlink_to(outside, target_is_directory=True)
    symlink_report = verify_six_case_review(
        manifest_path, repository_root=tmp_path
    )
    closure_check = next(
        check
        for check in symlink_report["checks"]
        if check["check_id"] == "package_closure"
    )
    assert closure_check["status"] == "fail"
    assert any("symlink" in error for error in closure_check["errors"])
    directory_link.unlink()

    first_segment = output / manifest["segments"][0]["rendered_segment"]["path"]
    first_segment.write_bytes(first_segment.read_bytes() + b"tamper")
    failed = verify_six_case_review(manifest_path, repository_root=tmp_path)
    assert failed["status"] == "fail"
    segment_check = next(
        check for check in failed["checks"] if check["check_id"] == "segment_media"
    )
    assert segment_check["status"] == "fail"
    assert any("byte size differs" in error for error in segment_check["errors"])
    with pytest.raises(FileExistsError, match="refusing to replace"):
        build_six_case_review(
            request_path=request_path,
            output_directory=output,
            repository_root=tmp_path,
        )


def test_load_rejects_invalid_request(tmp_path: Path) -> None:
    request = _request()
    request["cases"] = request["cases"][:5]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(request) + "\n", encoding="utf-8")
    from avengine.m6.review import load_six_case_review_request

    with pytest.raises(M6ReviewError):
        load_six_case_review_request(path)
