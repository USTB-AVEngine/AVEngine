"""Review UI must display declared capture lengths without a 75-frame gate."""

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))
import build_batch_review_page as TOOL  # noqa: E402


def test_review_page_accepts_any_positive_declared_frame_count():
    assert "Number.isInteger(p.receipt.frames)" in TOOL.HTML
    assert "p.receipt.frames!==75" not in TOOL.HTML


def _make_clip(path: Path, *, video_seconds: float, audio_seconds: float):
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i",
            f"color=c=black:s=2x2:r=15:d={video_seconds}",
            "-f", "lavfi", "-i",
            f"anullsrc=r=16000:cl=stereo:d={audio_seconds}",
            "-map", "0:v:0", "-map", "1:a:0", "-t", str(video_seconds),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "15",
            "-c:a", "aac", str(path),
        ],
        check=True,
    )


def test_review_page_rejects_a_stale_or_shortened_clip_clock(tmp_path):
    good = tmp_path / "good.mp4"
    shortened_audio = tmp_path / "shortened-audio.mp4"
    _make_clip(good, video_seconds=2.0, audio_seconds=2.0)
    _make_clip(shortened_audio, video_seconds=2.0, audio_seconds=1.0)
    receipt = {"frames": 30, "frame_rate_hz": 15.0}
    assert TOOL._clip_matches_receipt(good, receipt) is True
    assert TOOL._clip_matches_receipt(shortened_audio, receipt) is False
