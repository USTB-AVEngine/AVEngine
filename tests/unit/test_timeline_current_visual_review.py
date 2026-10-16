from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import avengine.cli as cli
from avengine.timeline.current_visual_review import (
    CurrentVisualReviewError,
    generate_current_visual_review,
)


def _write_capture_output(
    root: Path,
    *,
    status: str = "research_only",
    include_semantic: bool = True,
    action_id: str = "walk",
) -> Path:
    root.mkdir()
    arrays = root / "arrays"
    arrays.mkdir()
    frame_count = 3
    rgb = np.arange(frame_count * 2 * 4 * 3, dtype=np.uint8).reshape(3, 2, 4, 3)
    depth = np.asarray(
        [
            [[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]],
            [[4.5, 5.0, 5.5, 6.0], [6.5, 7.0, 7.5, 8.0]],
            [[8.5, 9.0, 9.5, 10.0], [10.5, 11.0, 11.5, 12.0]],
        ],
        dtype=np.float32,
    )
    semantic = np.asarray(
        [
            [[0, 210, 210, 0], [211, 211, 0, 0]],
            [[0, 210, 0, 0], [211, 0, 211, 0]],
            [[210, 210, 0, 0], [0, 211, 211, 0]],
        ],
        dtype=np.int32,
    )
    actor = np.tile(np.eye(4, dtype=np.float64), (frame_count, 2, 1, 1))
    actor[:, 0, 0, 3] = (1.0, 2.0, 3.0)
    actor[:, 1, 2, 3] = (4.0, 5.0, 6.0)
    sources = np.asarray(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]],
            [[1.2, 2.2, 3.2], [4.2, 5.2, 6.2]],
        ],
        dtype=np.float64,
    )
    visibility = np.asarray([[3, 2], [1, 2], [2, 2]], dtype=np.int64)
    np.save(arrays / "rgb.npy", rgb)
    np.save(arrays / "depth.npy", depth)
    if include_semantic:
        np.save(arrays / "semantic.npy", semantic)
    np.save(arrays / "actor_world_matrices.npy", actor)
    np.save(arrays / "source_positions_m.npy", sources)
    np.save(arrays / "semantic_visibility_pixels.npy", visibility)
    frames = [
        {
            "frame_index": index,
            "pts_ticks": index * 3200,
            "action_id": action_id,
            "action_sample_index": index,
        }
        for index in range(frame_count)
    ]
    (root / "frame_records.json").write_text(
        json.dumps({"frames": frames}, indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        "rgb": "arrays/rgb.npy",
        "depth": "arrays/depth.npy",
        "semantic": "arrays/semantic.npy",
        "actor_world_matrices": "arrays/actor_world_matrices.npy",
        "source_positions_m": "arrays/source_positions_m.npy",
        "semantic_visibility_pixels": "arrays/semantic_visibility_pixels.npy",
        "frame_records": "frame_records.json",
    }
    if not include_semantic:
        artifacts.pop("semantic")
    receipt = {
        "status": status,
        "research_only": status == "research_only",
        "episode_counted": False,
        "capture": {"native_habitat_started": True, "frame_count": frame_count},
        "artifacts": artifacts,
    }
    (root / "research_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def test_generates_offline_synchronized_research_only_review(tmp_path: Path) -> None:
    output = _write_capture_output(tmp_path / "capture")
    receipt_path = output / "research_receipt.json"
    receipt_before = receipt_path.read_bytes()

    review = generate_current_visual_review(output)

    assert review.frame_count == 3
    assert review.html_path == output / "review" / "index.html"
    assert review.html_path.is_file()
    assert receipt_path.read_bytes() == receipt_before
    html = review.html_path.read_text(encoding="utf-8")
    assert 'type="range"' in html
    assert "RGB" in html
    assert "Depth" in html
    assert "Semantic" in html
    assert "localStorage" in html
    assert "Research-only visual review" in html
    assert "planned_actor_world_positions_m" in html
    assert "runtime_source_readback_positions_m" in html
    assert "semantic_visibility_pixels" in html
    assert "approve" not in html.lower()
    assert "formal/admission evidence" in html
    for modality in ("rgb", "depth", "semantic"):
        paths = sorted((review.review_directory / "frames" / modality).glob("*.png"))
        assert len(paths) == 3
        assert paths[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_escapes_case_insensitive_script_end_marker_in_frame_metadata(
    tmp_path: Path,
) -> None:
    action_id = "</SCRIPT><script>injected()</script>"
    output = _write_capture_output(tmp_path / "capture", action_id=action_id)

    review = generate_current_visual_review(output)

    html = review.html_path.read_text(encoding="utf-8")
    assert action_id not in html
    assert "\\u003c/SCRIPT\\u003e\\u003cscript\\u003e" in html
    assert html.lower().count("</script") == 1


def test_rejects_frame_record_index_misaligned_with_arrays(tmp_path: Path) -> None:
    output = _write_capture_output(tmp_path / "capture")
    records_path = output / "frame_records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records["frames"][1]["frame_index"] = 0
    records_path.write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CurrentVisualReviewError, match="must equal its array index"):
        generate_current_visual_review(output)
    assert not (output / "review").exists()


def test_cli_generates_external_offline_review(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = _write_capture_output(tmp_path / "capture")
    external_review = tmp_path / "external-review"

    assert (
        cli.main(
            [
                "m5",
                "review-current-visual",
                "--research-output",
                str(output),
                "--review-output",
                str(external_review),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "research_only"
    assert payload["episode_counted"] is False
    assert payload["notes_storage"] == "browser localStorage only"
    assert Path(payload["review"]) == external_review / "index.html"
    assert (external_review / "index.html").is_file()


@pytest.mark.parametrize("status", ("not_run", "fail"))
def test_rejects_missing_or_noncompleted_research_output(
    tmp_path: Path, status: str
) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(CurrentVisualReviewError, match="research receipt"):
        generate_current_visual_review(missing)

    noncompleted = _write_capture_output(tmp_path / status, status=status)
    with pytest.raises(CurrentVisualReviewError, match="completed research_only"):
        generate_current_visual_review(noncompleted)


def test_rejects_missing_arrays_and_never_replaces_review(tmp_path: Path) -> None:
    missing_array = _write_capture_output(
        tmp_path / "missing-array",
        include_semantic=False,
    )
    with pytest.raises(
        CurrentVisualReviewError, match="missing required review artifacts"
    ):
        generate_current_visual_review(missing_array)

    output = _write_capture_output(tmp_path / "capture")
    generate_current_visual_review(output)
    with pytest.raises(CurrentVisualReviewError, match="refusing to replace"):
        generate_current_visual_review(output)
