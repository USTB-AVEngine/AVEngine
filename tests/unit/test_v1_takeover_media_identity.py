"""T01 regression checks for capture-scoped visual media and audio flags."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_audio_commands_forward_declared_layouts_and_foa_normalization(tmp_path: Path) -> None:
    from avengine.rooms.qa_delivery import (
        _build_habitat_audio_command,
        build_audio_command,
    )

    request = {
        "audio_layouts": [
            {"type": "binaural", "channel_count": 2, "role": "primary"},
            {"type": "ambisonics", "channel_count": 4, "role": "attached_view"},
        ],
        "foa_normalization": "sn3d",
        "runtime": {
            "runtime_prefix": "runtime",
            "rlr_sdk_root": "rlr",
            "magnum_python_site": "magnum",
        },
    }
    plan = {
        "resources": {"acoustic_package": str(tmp_path / "package.json")},
        "voice_bindings": [
            {"sound_asset_id": "event", "path": str(tmp_path / "event.wav")}
        ],
    }
    commands = [
        build_audio_command(
            request, plan, tmp_path, tmp_path / "audio", repository=tmp_path,
        ),
        _build_habitat_audio_command(
            request, plan, tmp_path, tmp_path / "capture", tmp_path / "audio",
            tmp_path / "program.json", repository=tmp_path,
        ),
    ]
    for command in commands:
        assert _argument(command, "--layouts") == "binaural,ambisonics"
        assert _argument(command, "--foa-normalization") == "sn3d"


def _decode_video(path: Path) -> bytes:
    return subprocess.check_output(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
            "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
    )


def test_distinct_capture_paths_get_distinct_masters_and_same_capture_reuses(tmp_path: Path) -> None:
    from avengine.rooms.qa_delivery import (
        _encode_rgb_frames_to_video,
        _prepare_visual_video,
        _publish_visual_master,
    )
    from avengine.rooms.qa_evidence import shared_visual_pack_root

    long_base = tmp_path
    for index in range(4):
        long_base = long_base / ("path_segment_" + str(index) + "_" + "x" * 48)
    capture_a = long_base / "v0_capture" / "capture" / "attempt_01" / "capture"
    capture_b = long_base / "v1_capture" / "capture" / "attempt_01" / "capture"
    capture_a.mkdir(parents=True)
    capture_b.mkdir(parents=True)
    np.save(
        capture_a / "rgb.npy",
        np.full((4, 16, 16, 3), [220, 30, 30], dtype=np.uint8),
    )
    np.save(
        capture_b / "rgb.npy",
        np.full((4, 16, 16, 3), [30, 30, 220], dtype=np.uint8),
    )

    shared_root = tmp_path / "shared_visual_evidence"
    master_a = shared_visual_pack_root(shared_root, capture_a) / "visual_rgb.mp4"
    master_b = shared_visual_pack_root(shared_root, capture_b) / "visual_rgb.mp4"
    assert master_a != master_b
    # The complete path stays as directory components, so this long-path
    # counterexample cannot collapse into one component.
    assert len(str(capture_a.resolve())) > 255
    assert master_a.parent.name == "capture"
    assert master_a.parent.parent.name == "attempt_01"
    assert master_a.parent.parent.parent.name == "capture"
    assert len(master_a.parent.name) < 255

    first_a, first_status = _prepare_visual_video(
        capture_a, clock={"frame_count": 4, "frame_rate_hz": 2.0},
        output_path=tmp_path / "private_a.mp4", shared_master_path=master_a,
    )
    second_a, second_status = _prepare_visual_video(
        capture_a, clock={"frame_count": 4, "frame_rate_hz": 2.0},
        output_path=tmp_path / "private_a_again.mp4", shared_master_path=master_a,
    )
    first_b, first_b_status = _prepare_visual_video(
        capture_b, clock={"frame_count": 4, "frame_rate_hz": 2.0},
        output_path=tmp_path / "private_b.mp4", shared_master_path=master_b,
    )

    assert first_a == second_a == master_a.resolve()
    assert first_b == master_b.resolve()
    assert first_status["published_shared_master"] is True
    assert second_status["source"] == "shared_visual_master"
    assert first_b_status["published_shared_master"] is True
    assert _decode_video(master_a) != _decode_video(master_b)

    # Simulate the second creator arriving after the first creator published:
    # the no-replace publication path must preserve the referenced master.
    candidate = tmp_path / "candidate.mp4"
    candidate_info = _encode_rgb_frames_to_video(
        np.full((4, 16, 16, 3), [220, 30, 30], dtype=np.uint8),
        output_path=candidate, frame_rate_hz=2.0, expected_frame_count=4,
    )
    existing, race_status = _publish_visual_master(
        candidate, master_a, candidate_info,
    )
    assert existing == master_a.resolve()
    assert race_status["published_shared_master"] is False
    assert race_status["reused"] is True
    assert not candidate.exists()
