from copy import deepcopy
import wave
import pytest
from avengine.rooms.qa_delivery import _ancillary_audio_outputs


def _report(tmp_path):
    path = tmp_path / "ambisonics.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(4)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(bytes(32 * 4 * 2))
    layout = {"layout_id": "foa_world", "layout_type": "ambisonics",
              "channel_count": 4, "channel_labels": ["W", "Y", "Z", "X"],
              "channel_order": "ACN", "normalization": "N3D",
              "coordinate_frame": "avengine_world", "sample_rate_hz": 16000,
              "sample_count": 32, "mixture": {"path": str(path)}}
    return {"clock": {"sample_count": 32, "sample_rate_hz": 16000},
            "audio": {"layout_delivery": {"ambisonics": layout, "binaural": {"layout_type": "binaural"}}}}


def test_actual_layout_delivery_is_registered_once_with_its_convention(tmp_path):
    report = _report(tmp_path)
    report["foa_path"] = report["audio"]["layout_delivery"]["ambisonics"]["mixture"]["path"]
    outputs = _ancillary_audio_outputs(report)
    assert len(outputs) == 1
    row = outputs[0]
    assert row["media_readback"]["channel_count"] == 4
    assert row["sample_count"] == 32
    assert row["channel_order"] == "ACN"
    assert row["normalization"] == "N3D"
    assert row["coordinate_frame"] == "avengine_world"
    assert row["canonical"] is False


@pytest.mark.parametrize("field,value", [("channel_count", 2), ("sample_count", 31), ("sample_rate_hz", 8000)])
def test_declared_layout_cannot_hide_a_wrong_wav_clock_or_shape(tmp_path, field, value):
    report = _report(tmp_path)
    report["audio"]["layout_delivery"]["ambisonics"][field] = value
    with pytest.raises(ValueError, match="differs"):
        _ancillary_audio_outputs(report)


def test_declared_missing_layout_is_not_silently_dropped(tmp_path):
    report = _report(tmp_path)
    report["audio"]["layout_delivery"]["ambisonics"]["mixture"]["path"] = str(tmp_path / "missing.wav")
    with pytest.raises(FileNotFoundError, match="ancillary mixture"):
        _ancillary_audio_outputs(report)
