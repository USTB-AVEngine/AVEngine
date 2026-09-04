from dataclasses import dataclass
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.contracts.json_io import load_json
from avengine.contracts.json_io import sha256_file
from avengine.spatial_audio.audio import write_float32_wav
from avengine.rooms.visual_profile import (
    M6XVisualProfileError,
    ReviewVisualProfile,
    aspect_preserving_downscale_no_upscale,
    configure_runtime_review_profile,
    encode_profiled_h264_base_video,
    letterbox_marker_coordinates,
    light_setup_records,
    load_review_visual_profile,
    mux_profiled_binaural_wav,
    validate_profile_capture_request,
    validate_realized_review_profile,
    validate_runtime_review_light_readback,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "examples/routes/fixed_apartment/review_visual_profile.json"
REQUEST_PATH = ROOT / "examples/routes/fixed_apartment/m1_capture_request_review_720p.json"


def test_review_profile_accepts_a_declared_non15_frame_rate(tmp_path: Path) -> None:
    value = load_json(PROFILE_PATH)
    value["capture"]["frame_rate_hz"] = 12.0
    path = tmp_path / "review_profile_12fps.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    profile = load_review_visual_profile(path)
    assert profile.capture_frame_rate_hz == pytest.approx(12.0)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg tools are unavailable",
)
def test_profile_encoder_uses_declared_non15_frame_rate(tmp_path: Path) -> None:
    frame_count = 90
    frame_rate_hz = 12.0
    sample_rate_hz = 12_000
    sample_count = 90_000
    profile = ReviewVisualProfile(
        path=tmp_path / "profile.json",
        raw={"capture": {"frame_rate_hz": frame_rate_hz}},
        profile_id="declared_12fps",
        capture_resolution_hw=(48, 64),
        diagnostic_panel_resolution_hw=(48, 64),
        capture_frame_rate_hz=frame_rate_hz,
    )
    frames = np.zeros((frame_count, 48, 64, 3), dtype=np.uint8)
    for index in range(frame_count):
        frames[index, :, :, 0] = index % 256
        frames[index, :, :, 1] = (2 * index) % 256
    rng = np.random.default_rng(20260905)
    samples = rng.normal(0.0, 0.05, (sample_count, 2)).astype(np.float64)
    wav = tmp_path / "declared.wav"
    write_float32_wav(wav, samples.T, sample_rate_hz)
    base = tmp_path / "declared.base.mp4"
    encoded = encode_profiled_h264_base_video(
        frames, base, profile=profile)
    muxed = tmp_path / "declared.mp4"
    mux = mux_profiled_binaural_wav(
        base, wav, muxed, profile=profile,
        sample_rate_hz=sample_rate_hz, sample_count=sample_count,
    )
    assert encoded["video"]["frame_count"] == frame_count
    assert encoded["video"]["frame_rate"] == "12/1"
    assert encoded["video"]["duration_seconds"] == pytest.approx(7.5)
    assert mux["video"]["duration_seconds"] == pytest.approx(7.5)
    assert mux["audio"]["sample_rate_hz"] == sample_rate_hz
    assert mux["audio"]["duration_seconds"] == pytest.approx(7.5)
    assert mux["audio"]["decoded_sample_count"] >= sample_count
    assert mux["audio"]["decoded_padding_samples"] >= 0


def test_review_profile_freezes_native_720p_and_non_upscaling_diagnostic() -> None:
    profile = load_review_visual_profile(PROFILE_PATH)

    assert profile.capture_resolution_hw == (720, 1280)
    assert profile.diagnostic_panel_resolution_hw == (480, 640)
    assert profile.raw["exterior_proxy"]["proxy_kind"] == (
        "inward_uv_sphere_with_direction_projected_window_panels"
    )
    assert all(
        panel["uv_projection"] == "listener_direction_equirectangular"
        and min(panel["grid_subdivisions_wh"]) >= 2
        and "uv_rect" not in panel
        for panel in profile.raw["exterior_proxy"]["window_panels"]
    )
    validate_profile_capture_request(profile, load_json(REQUEST_PATH))


def test_review_lighting_has_normalized_directional_key_and_fill() -> None:
    records = light_setup_records(load_review_visual_profile(PROFILE_PATH))

    directional = [record for record in records if record["type"] == "directional"]
    assert len(directional) == 2
    for record in directional:
        assert record["vector_xyzw"][3] == 0.0
        assert np.linalg.norm(record["vector_xyzw"][:3]) == pytest.approx(1.0)
    assert records[-1]["type"] == "point"
    assert records[-1]["vector_xyzw"][3] == 1.0


def test_review_lighting_uses_shallow_neutral_window_and_bounce_balance() -> None:
    profile = load_review_visual_profile(PROFILE_PATH)
    raw = {light["light_id"]: light for light in profile.raw["lighting"]["lights"]}
    records = light_setup_records(profile)

    key_direction = np.asarray(
        next(
            record["vector_xyzw"][:3]
            for record in records
            if record["light_id"] == "window_directional_key"
        )
    )
    assert abs(key_direction[1]) < 0.55
    assert (
        raw["interior_practical_fill"]["intensity"]
        < (raw["window_directional_key"]["intensity"])
    )
    total_rgb = np.sum(np.asarray([record["color_rgb"] for record in records]), axis=0)
    assert float(np.max(total_rgb) / np.min(total_rgb)) < 1.05
    assert profile.profile_id == "spear_apartment_habitat_review_720p_natural_v3"
    assert profile.raw["lighting"]["setup_id"] == (
        "spear_apartment_window_bounce_natural_v2"
    )


@dataclass(frozen=True)
class _FakeLightInfo:
    vector: tuple[float, float, float, float]
    color: tuple[float, float, float]
    model: str


class _FakeSimulator:
    def __init__(self, current: list[_FakeLightInfo], actor: list[_FakeLightInfo]):
        self.current = current
        self.actor = actor

    def get_current_light_setup(self) -> list[_FakeLightInfo]:
        return list(self.current)

    def get_light_setup(self, key: str) -> list[_FakeLightInfo]:
        assert key == "actor-review-key"
        return list(self.actor)


def test_final_scene_and_actor_light_keys_both_match_profile() -> None:
    profile = load_review_visual_profile(PROFILE_PATH)
    fake_habitat = SimpleNamespace(
        gfx=SimpleNamespace(
            LightInfo=_FakeLightInfo,
            LightPositionModel=SimpleNamespace(
                Global="global", Camera="camera", Object="object"
            ),
        )
    )
    expected = [
        _FakeLightInfo(
            vector=record["vector_xyzw"],
            color=record["color_rgb"],
            model=record["position_model"],
        )
        for record in light_setup_records(profile)
    ]
    simulator = _FakeSimulator(expected, expected)

    evidence = validate_runtime_review_light_readback(
        simulator,
        profile=profile,
        habitat_sim=fake_habitat,
        actor_light_setup_key="actor-review-key",
    )

    assert evidence["current_matches_profile"] is True
    assert evidence["actor_setup_matches_profile"] is True
    simulator.actor = expected[:-1]
    with pytest.raises(M6XVisualProfileError, match="scene/actor light setup"):
        validate_runtime_review_light_readback(
            simulator,
            profile=profile,
            habitat_sim=fake_habitat,
            actor_light_setup_key="actor-review-key",
        )


def test_review_configuration_uses_mutable_default_light_key() -> None:
    profile = load_review_visual_profile(PROFILE_PATH)
    configuration = SimpleNamespace(
        sim_cfg=SimpleNamespace(
            scene_light_setup="dataset-final-key",
            override_scene_light_defaults=False,
        )
    )
    fake_habitat = SimpleNamespace(
        gfx=SimpleNamespace(DEFAULT_LIGHTING_KEY="mutable-default")
    )

    evidence = configure_runtime_review_profile(
        configuration, profile=profile, habitat_sim=fake_habitat
    )

    assert configuration.sim_cfg.scene_light_setup == "mutable-default"
    assert configuration.sim_cfg.override_scene_light_defaults is True
    assert evidence["previous_scene_light_setup_key"] == "dataset-final-key"


def test_diagnostic_letterbox_downscales_without_aspect_distortion() -> None:
    source = np.full((2, 720, 1280, 3), 177, dtype=np.uint8)
    result = aspect_preserving_downscale_no_upscale(
        source, target_resolution_hw=(480, 640)
    )

    assert result.shape == (2, 480, 640, 3)
    assert np.all(result[:, :60] == 0)
    assert np.all(result[:, 60:420] == 177)
    assert np.all(result[:, 420:] == 0)


def test_diagnostic_refuses_low_resolution_upscale() -> None:
    with pytest.raises(M6XVisualProfileError, match="refuses to upscale"):
        aspect_preserving_downscale_no_upscale(
            np.zeros((1, 240, 320, 3), dtype=np.uint8),
            target_resolution_hw=(480, 640),
        )


def test_main_markers_follow_the_same_720p_to_letterbox_transform() -> None:
    markers = np.asarray(
        [[0.0, 0.0], [640.0, 360.0], [1279.0, 719.0], [np.nan, np.nan]]
    )
    transformed = letterbox_marker_coordinates(
        markers,
        source_resolution_hw=(720, 1280),
        target_resolution_hw=(480, 640),
    )

    assert np.allclose(transformed[0], [0.0, 60.0])
    assert np.allclose(transformed[1], [320.0, 240.0])
    assert np.allclose(transformed[2], [639.5, 419.5])
    assert np.all(np.isnan(transformed[3]))


def test_profile_rejects_capture_request_resolution_drift() -> None:
    request = load_json(REQUEST_PATH)
    request["primary_camera_rig"]["shared_calibration"]["resolution_hw"] = [
        480,
        640,
    ]
    with pytest.raises(M6XVisualProfileError, match="resolution differs"):
        validate_profile_capture_request(
            load_review_visual_profile(PROFILE_PATH), request
        )


def test_retained_capture_visual_evidence_binds_profile_and_proxy(
    tmp_path: Path,
) -> None:
    profile = load_review_visual_profile(PROFILE_PATH)
    proxy = tmp_path / "exterior.glb"
    proxy.write_bytes(b"prepared exterior")
    evidence = {
        "review_visual_profile": {
            "status": "pass",
            "profile_id": profile.profile_id,
            "profile_sha256": sha256_file(PROFILE_PATH),
            "native_capture_resolution_hw": [720, 1280],
            "configuration": {
                "status": "pass",
                "setup_id": profile.raw["lighting"]["setup_id"],
                "override_scene_light_defaults": True,
            },
            "final_light_readback": {
                "status": "pass",
                "expected_light_count": 3,
                "current_matches_profile": True,
                "actor_setup_matches_profile": True,
            },
            "capture_scene_objects": {
                "removed_handle_prefixes": ["source_marker_"],
                "removed_count": 2,
                "remaining_matching_count": 0,
                "logical_source_representation": ("topdown_timeline_and_audio_only"),
            },
            "exterior_proxy": {
                "prepared_glb_sha256": sha256_file(proxy),
                "proxy_kind": profile.raw["exterior_proxy"]["proxy_kind"],
                "window_panel_ids": [
                    panel["panel_id"]
                    for panel in profile.raw["exterior_proxy"]["window_panels"]
                ],
                "collidable_readback": False,
                "semantic_id_readback": 0,
                "semantic_behavior": "background_id_zero",
                "depth_behavior": "renders_as_far_exterior_surface",
                "excluded_from": profile.raw["exterior_proxy"]["excluded_from"],
                "scope": "transient_visual_capture_simulator_only",
            },
        }
    }

    validate_realized_review_profile(
        evidence, profile=profile, exterior_proxy_glb_path=proxy
    )
    evidence["review_visual_profile"]["exterior_proxy"]["collidable_readback"] = True
    with pytest.raises(M6XVisualProfileError, match="differs"):
        validate_realized_review_profile(
            evidence, profile=profile, exterior_proxy_glb_path=proxy
        )



def test_capture_color_order_comes_from_receipt_not_backend_name(tmp_path):
    import json
    from avengine.rooms.visual_profile import resolve_review_capture_channel_order
    (tmp_path / "research_receipt.json").write_text(json.dumps({
        "capture": {"rgb_channel_order": "bgr"}}))
    assert resolve_review_capture_channel_order(tmp_path) == "bgr"
    assert resolve_review_capture_channel_order(tmp_path, "bgr") == "bgr"
    with pytest.raises(ValueError, match="differs"):
        resolve_review_capture_channel_order(tmp_path, "rgb")


def test_legacy_color_order_can_be_explicit_without_rewriting_capture(tmp_path):
    from avengine.rooms.visual_profile import resolve_review_capture_channel_order
    assert resolve_review_capture_channel_order(tmp_path, "bgr") == "bgr"
    assert resolve_review_capture_channel_order(tmp_path) == "rgb"



def test_review_uses_declared_rgb_artifact_instead_of_assuming_one_layout(tmp_path):
    import json
    from avengine.rooms.visual_profile import resolve_review_capture_rgb_path
    data = tmp_path / "producer_frames.npy"
    data.write_bytes(b"fixture")
    (tmp_path / "research_receipt.json").write_text(json.dumps({
        "artifacts": {"rgb": data.name}}))
    assert resolve_review_capture_rgb_path(tmp_path) == data
    data.unlink()
    (tmp_path / "rgb.npy").write_bytes(b"stale legacy file")
    with pytest.raises(ValueError, match="declared RGB artifact is missing"):
        resolve_review_capture_rgb_path(tmp_path)


def test_review_accepts_historical_root_rgb_layout(tmp_path):
    from avengine.rooms.visual_profile import resolve_review_capture_rgb_path
    data = tmp_path / "rgb.npy"
    data.write_bytes(b"fixture")
    assert resolve_review_capture_rgb_path(tmp_path) == data
