"""V1 delivery contract for the binaural and native FOA audio layouts.

The unified V1 production route must be able to ask one real input for
binaural, for first-order ambisonics, or for both, and a consumer reading the
receipt must be able to tell the channel order, the normalization and the
coordinate frame it actually received. These checks pin that contract and the
two ways a four-channel buffer can be fake: a tiled binaural pair, and a
normalization relabelled without the conversion.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

from typing import Any

import numpy as np
import pytest

import avengine.timeline.current_mp3d_dynamic_audio as dynamic_audio
from avengine.acoustics.runtime import (
    CompiledAcousticScene,
    CompiledSceneUploadExpectation,
)
from avengine.spatial_audio.audio import read_float32_wav
from avengine.timeline.audio_program import bind_audio_program_hash
from avengine.timeline.current_mp3d_dynamic_audio import (
    CurrentMP3DDynamicAudioError,
    foa_normalization_record,
    foa_normalization_scale,
    layout_output_contract,
    render_dynamic_research_audio,
)

REPOSITORY = Path(__file__).resolve().parents[2]
PROGRAM_PATH = (
    REPOSITORY
    / "examples/timeline/current_mp3d/audio_programs"
    / "current_mp3d_two_beagle_turn_taking_v1.json"
)
FRAME_COUNT = 75
SAMPLE_COUNT = 80_000


def test_layout_contracts_expose_the_native_channel_and_frame_identity() -> None:
    binaural = layout_output_contract("binaural")
    assert binaural["channel_labels"] == ["left", "right"]
    assert binaural["channel_count"] == 2
    assert binaural["coordinate_frame"] == "listener_local"
    assert binaural["normalization"] == "not_applicable"
    assert binaural["output_directory"] == "binaural"

    foa = layout_output_contract("ambisonics")
    # ACN 0..3 is W, Y, Z, X; the native RLR encode is fully normalized (N3D)
    # in the world frame, not the listener frame.
    assert foa["channel_labels"] == ["W", "Y", "Z", "X"]
    assert foa["channel_count"] == 4
    assert foa["channel_order"] == "ACN"
    assert foa["normalization"] == "N3D"
    assert foa["coordinate_frame"] == "avengine_world"
    assert foa["output_directory"] == "foa"
    assert foa["layout_id"] == "rlr_foa_acn_n3d_world_v1"

    with pytest.raises(CurrentMP3DDynamicAudioError):
        layout_output_contract("stereo")


def test_sn3d_conversion_is_the_exact_per_degree_factor() -> None:
    assert foa_normalization_scale("native_n3d") == (1.0, 1.0, 1.0, 1.0)
    scale = foa_normalization_scale("sn3d")
    # SN3D = N3D / sqrt(2l + 1): degree 0 is unchanged, degree 1 divides by
    # sqrt(3). This is a convention change, not a loudness adjustment.
    assert scale[0] == 1.0
    for value in scale[1:]:
        assert value == pytest.approx(1.0 / math.sqrt(3.0), abs=0.0, rel=1e-15)

    record = foa_normalization_record("sn3d")
    assert record["native_normalization"] == "N3D"
    assert record["delivered_normalization"] == "SN3D"
    assert record["channel_order"] == "ACN"
    assert foa_normalization_record("native_n3d")["delivered_normalization"] == "N3D"

    with pytest.raises(CurrentMP3DDynamicAudioError):
        foa_normalization_scale("fuma")


def test_tiled_binaural_is_rejected_as_native_ambisonics() -> None:
    rng = np.random.default_rng(20260910)
    pair = rng.standard_normal((2, 64))

    # [L, R, L, R]: two channels duplicated into four.
    with pytest.raises(CurrentMP3DDynamicAudioError, match="tiled binaural"):
        dynamic_audio._assert_native_ambisonics(
            np.concatenate([pair, pair], axis=0), owner="tiled"
        )

    # [L, L, R, R]: the other way to pad a pair up to four channels.
    with pytest.raises(CurrentMP3DDynamicAudioError, match="tiled binaural"):
        dynamic_audio._assert_native_ambisonics(
            np.stack([pair[0], pair[0], pair[1], pair[1]]), owner="paired"
        )

    # Directional energy with no omni component cannot come from a real encode.
    broken = rng.standard_normal((4, 64))
    broken[0] = 0.0
    with pytest.raises(CurrentMP3DDynamicAudioError, match="silent W"):
        dynamic_audio._assert_native_ambisonics(broken, owner="no-omni")

    # A real four-channel encode passes, and an unused source stays silent
    # without making a claim about its encoding.
    dynamic_audio._assert_native_ambisonics(
        rng.standard_normal((4, 64)), owner="native"
    )
    dynamic_audio._assert_native_ambisonics(np.zeros((4, 64)), owner="silent")

    with pytest.raises(CurrentMP3DDynamicAudioError):
        dynamic_audio._assert_native_ambisonics(np.zeros((2, 64)), owner="two-channel")


def _install_layout_stubs(
    monkeypatch: pytest.MonkeyPatch, *, trajectories: dict, sample_count: int
) -> dict:
    """Drive the real clock, receipt and WAVE writer with tiny native fakes."""

    calls: dict = {"rir": [], "channels": {}}
    monkeypatch.setattr(
        dynamic_audio, "_load_simulation_request", lambda _path: (None, None)
    )
    monkeypatch.setattr(
        dynamic_audio, "load_compiled_acoustic_scene", lambda *_a, **_k: object()
    )
    monkeypatch.setattr(dynamic_audio, "_asset_bindings", lambda *_a, **_k: {})

    def fake_rir(_scene, _simulation, *, grid, layout_type, hrtf_file_path=None):
        calls["rir"].append((layout_type, hrtf_file_path))
        labels = (
            ("left", "right")
            if layout_type == "binaural"
            else ("W", "Y", "Z", "X")
        )
        return SimpleNamespace(
            layout_type=layout_type,
            layout_id=layout_output_contract(layout_type)["layout_id"],
            channel_labels=labels,
            keyframe_samples=(0,),
            trajectory_sha256=f"test-trajectory-{layout_type}",
        )

    monkeypatch.setattr(
        dynamic_audio, "render_research_review_rir_sequence", fake_rir
    )

    def fake_assembly(materialized_program, _variant_id, **_kwargs):
        return SimpleNamespace(
            materialized_program=materialized_program,
            dry_audio=SimpleNamespace(
                buses={
                    source_id: np.zeros(sample_count, dtype=np.float64)
                    for source_id in trajectories
                },
                placement_receipts=(),
            ),
        )

    monkeypatch.setattr(
        dynamic_audio, "assemble_audio_program_dry_buses", fake_assembly
    )

    def fake_layout_audio(dry_buses, sequence, *, grid):
        # Distinct, non-silent per-channel content so the delivered PCM
        # actually carries a directional pattern to inspect.
        expected = int(grid.episode_sample_count)
        channels = 2 if sequence.layout_type == "binaural" else 4
        ramp = np.linspace(0.0, 1.0, expected, dtype=np.float64)
        stems = {}
        for source_index, source_id in enumerate(sorted(dry_buses)):
            block = np.stack(
                [
                    0.02 * (channel + 1) * (source_index + 1) * ramp
                    for channel in range(channels)
                ]
            )
            stems[source_id] = SimpleNamespace(
                episode=block.astype(np.float32)
            )
        mixture = np.sum(
            [value.episode for value in stems.values()], axis=0
        ).astype(np.float32)
        calls["channels"][sequence.layout_type] = channels
        return stems, mixture

    monkeypatch.setattr(
        dynamic_audio, "render_research_review_audio", fake_layout_audio
    )
    return calls


def _render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs) -> tuple:
    rir_override = kwargs.pop("_rir_override", None)
    program = deepcopy(json.loads(PROGRAM_PATH.read_text(encoding="utf-8")))
    program["timeline"].update(
        {"frame_count": FRAME_COUNT, "sample_count": SAMPLE_COUNT}
    )
    program = bind_audio_program_hash(program)
    program_path = tmp_path / "program.json"
    program_path.write_text(json.dumps(program), encoding="utf-8")
    for name in ("simulation.json", "package.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    hrtf_path = tmp_path / "hrtf.sofa"
    hrtf_path.write_bytes(b"tiny test hrtf")

    trajectories = {
        "beagle_0_muzzle": [[float(i), 0.0, 0.0] for i in range(FRAME_COUNT)],
        "beagle_1_muzzle": [[float(i), 0.0, 1.0] for i in range(FRAME_COUNT)],
    }
    calls = _install_layout_stubs(
        monkeypatch, trajectories=trajectories, sample_count=SAMPLE_COUNT
    )
    if rir_override is not None:
        monkeypatch.setattr(
            dynamic_audio, "render_research_review_rir_sequence", rir_override
        )
    output = tmp_path / kwargs.pop("output_name", "rendered")
    receipt = render_dynamic_research_audio(
        source_trajectories_m=trajectories,
        listener_position_m=[0.0, 0.0, 0.0],
        listener_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        simulation_request_path=tmp_path / "simulation.json",
        package_manifest_path=tmp_path / "package.json",
        audio_program_path=program_path,
        source_endpoint_registry_path=(
            REPOSITORY / "examples/registry/registries/source_endpoints_v1.json"
        ),
        sound_asset_registry_path=(
            REPOSITORY / "examples/registry/registries/sound_assets_v1.json"
        ),
        external_sound_asset_paths={},
        hrtf_file_path=hrtf_path,
        output_path=output,
        position_authority="test",
        listener_authority="test",
        **kwargs,
    )
    return receipt, output, calls


def test_one_input_delivers_binaural_and_native_foa_with_a_full_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, output, calls = _render(
        tmp_path, monkeypatch, layouts=("binaural", "ambisonics")
    )

    delivery = receipt["audio"]["layout_delivery"]
    assert set(delivery) == {"binaural", "ambisonics"}

    foa = delivery["ambisonics"]
    assert foa["channel_order"] == "ACN"
    assert foa["channel_labels"] == ["W", "Y", "Z", "X"]
    assert foa["normalization"] == "N3D"
    assert foa["coordinate_frame"] == "avengine_world"
    assert foa["layout_id"] == "rlr_foa_acn_n3d_world_v1"
    assert foa["sample_rate_hz"] == 16_000
    assert foa["sample_count"] == SAMPLE_COUNT
    assert foa["rir_source"] == "fresh_native_render"
    assert foa["foa_normalization"]["delivered_normalization"] == "N3D"

    binaural = delivery["binaural"]
    assert binaural["coordinate_frame"] == "listener_local"
    assert binaural["foa_normalization"] is None

    # Each layout names its own files and reports its own measured activity.
    for layout, subdir in (("binaural", "binaural"), ("ambisonics", "foa")):
        mixture_path = Path(delivery[layout]["mixture"]["path"])
        assert mixture_path == (output / "audio" / subdir / "mixture.wav").resolve()
        assert mixture_path.is_file()
        assert delivery[layout]["mixture"]["active_interval_samples"] is not None
        for source_id, stem in delivery[layout]["stems"].items():
            assert Path(stem["path"]).is_file()
            assert stem["peak_dbfs"] is not None

    assert receipt["audio"]["activity_measurement_layout"] == "binaural"

    # The FOA files really carry four channels on the episode clock, and the
    # native renderer was asked for ambisonics without an HRTF.
    foa_wave = read_float32_wav(output / "audio" / "foa" / "mixture.wav")
    assert foa_wave.frame_count == SAMPLE_COUNT
    assert foa_wave.sample_rate_hz == 16_000
    assert calls["channels"]["ambisonics"] == 4
    assert ("ambisonics", None) in calls["rir"]
    assert ("binaural", str(tmp_path / "hrtf.sofa")) in calls["rir"]

    binaural_wave = read_float32_wav(output / "audio" / "binaural" / "mixture.wav")
    foa_samples = np.asarray(foa_wave.samples)
    binaural_samples = np.asarray(binaural_wave.samples)
    assert foa_samples.shape == (4, SAMPLE_COUNT)
    assert binaural_samples.shape == (2, SAMPLE_COUNT)
    # The four FOA channels are not a repeat of the two binaural ones.
    assert not np.array_equal(foa_samples[0:2], foa_samples[2:4])


def test_sn3d_delivery_scales_the_native_encode_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native, native_output, _ = _render(
        tmp_path, monkeypatch, layouts=("ambisonics",), output_name="native"
    )
    converted, converted_output, _ = _render(
        tmp_path,
        monkeypatch,
        layouts=("ambisonics",),
        foa_normalization="sn3d",
        output_name="sn3d",
    )

    assert native["audio"]["layout_delivery"]["ambisonics"]["normalization"] == "N3D"
    delivered = converted["audio"]["layout_delivery"]["ambisonics"]
    assert delivered["normalization"] == "SN3D"
    assert delivered["foa_normalization"]["conversion"] == (
        "n3d_to_sn3d_per_degree_1_over_sqrt_2l_plus_1"
    )
    assert converted["inputs"]["audio_render_config"]["foa_normalization"][
        "delivered_normalization"
    ] == "SN3D"

    native_pcm = np.asarray(
        read_float32_wav(native_output / "audio" / "foa" / "mixture.wav").samples
    )
    sn3d_pcm = np.asarray(
        read_float32_wav(converted_output / "audio" / "foa" / "mixture.wav").samples
    )
    assert native_pcm.shape == (4, SAMPLE_COUNT)
    assert sn3d_pcm.shape == (4, SAMPLE_COUNT)

    # W is untouched; every degree-1 channel is divided by sqrt(3).
    np.testing.assert_allclose(sn3d_pcm[0], native_pcm[0], rtol=1e-6, atol=1e-9)
    for channel in (1, 2, 3):
        np.testing.assert_allclose(
            sn3d_pcm[channel],
            native_pcm[channel] / math.sqrt(3.0),
            rtol=1e-5,
            atol=1e-9,
        )

    # The conversion is linear, so the mixture still equals the sum of stems.
    stems = [
        np.asarray(read_float32_wav(path).samples)
        for path in sorted((converted_output / "audio" / "foa").glob("*_stem.wav"))
    ]
    np.testing.assert_allclose(
        np.sum(stems, axis=0), sn3d_pcm, rtol=1e-5, atol=1e-9
    )


def test_reused_rir_sequence_must_declare_its_own_layout_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid = SimpleNamespace(
        keyframes=(SimpleNamespace(tick=0, sample_index=0),),
        source_ids=("a", "b"),
        sample_rate_hz=16_000,
    )
    monkeypatch.setattr(
        dynamic_audio,
        "research_review_trajectory_record",
        lambda _grid: {"trajectory": "test"},
    )
    payload = {
        "samples": np.zeros((1, 2, 2, 4), dtype="<f4"),
        "lengths": np.ones((1, 2), dtype="<u4"),
    }

    # A cache payload that never recorded which layout produced it is missing
    # the metadata that would make it attributable, so it is not a hit.
    with pytest.raises(CurrentMP3DDynamicAudioError, match="not a cache hit"):
        dynamic_audio._sequence_from_override(payload, grid=grid, layout="binaural")

    with pytest.raises(CurrentMP3DDynamicAudioError, match="requires"):
        dynamic_audio._sequence_from_override(
            {**payload, "layout_id": "rlr_foa_acn_n3d_world_v1"},
            grid=grid,
            layout="binaural",
        )

    sequence = dynamic_audio._sequence_from_override(
        {**payload, "layout_id": "rlr_binaural_lr_v1"}, grid=grid, layout="binaural"
    )
    assert sequence.layout_id == "rlr_binaural_lr_v1"
    assert sequence.channel_labels == ("left", "right")


def _minimal_compiled_scene(**overrides) -> CompiledAcousticScene:
    """One tiny but real compiled scene, enough to derive an upload report."""

    triangle = {
        "object_id": "obj0",
        "position": np.zeros(3),
        "orientation_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        "vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "triangles": np.array([[0, 1, 2]], dtype=np.int32),
        "triangle_material_ids": np.array([0], dtype=np.int32),
    }
    values = dict(
        manifest_path=Path("/pkg/a/manifest.json"),
        manifest_sha256="a" * 64,
        manifest={},
        package_id="pkg_test",
        package_content_sha256="b" * 64,
        material_database_path=Path("/pkg/a/materials.json"),
        material_database_bytes=b"{}",
        material_database_sha256="c" * 64,
        material_categories_document={},
        rlr_material_database={
            "materials": [
                {
                    "name": "Concrete",
                    "labels": ["wall"],
                    "absorption": [0.1] * 4,
                    "scattering": [0.1] * 4,
                    "transmission": [0.1] * 4,
                }
            ]
        },
        material_categories=("wall",),
        objects=(triangle,),
        geometry_records={},
        triangle_count_by_material={"wall": 1},
        qa_reports={},
    )
    values.update(overrides)
    return CompiledAcousticScene(**values)


def test_one_upload_expectation_is_shared_across_the_layout_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both layouts upload the same scene, so derive the expectation once.

    Deriving the expected upload report canonicalizes the whole world mesh and
    measured about 11 s on the current MP3D package, so re-deriving it per
    layout is pure waste. Each native context must still receive it and still
    run its own upload comparison.
    """

    scene = _minimal_compiled_scene()
    seen: list[Any] = []
    constructed: list[Any] = []

    real_expectation = dynamic_audio.CompiledSceneUploadExpectation

    def counting_expectation(bound_scene):
        instance = real_expectation(bound_scene)
        constructed.append(instance)
        return instance

    monkeypatch.setattr(
        dynamic_audio, "CompiledSceneUploadExpectation", counting_expectation
    )

    def capture_rir(_scene, _sim, *, grid, layout_type, hrtf_file_path=None,
                    upload_expectation=None):
        seen.append((layout_type, upload_expectation))
        return SimpleNamespace(
            layout_type=layout_type,
            layout_id=layout_output_contract(layout_type)["layout_id"],
            channel_labels=(
                ("left", "right")
                if layout_type == "binaural"
                else ("W", "Y", "Z", "X")
            ),
            keyframe_samples=(0,),
            trajectory_sha256=f"t-{layout_type}",
        )

    receipt, output, _ = _render(
        tmp_path,
        monkeypatch,
        layouts=("binaural", "ambisonics"),
        scene_override=scene,
        _rir_override=capture_rir,
    )

    assert [layout for layout, _ in seen] == ["binaural", "ambisonics"]
    # Derived once for the whole render...
    assert len(constructed) == 1
    # ...and the very same object reached both native renders.
    assert seen[0][1] is constructed[0]
    assert seen[1][1] is constructed[0]
    assert isinstance(seen[0][1], real_expectation)
    assert set(receipt["audio"]["layout_delivery"]) == {"binaural", "ambisonics"}


def test_upload_expectation_refuses_a_scene_it_was_not_built_for() -> None:
    scene = _minimal_compiled_scene()
    expectation = CompiledSceneUploadExpectation(scene)
    assert len(expectation.expected_report(scene)) == 16

    other = _minimal_compiled_scene(package_id="pkg_other")
    with pytest.raises(Exception):
        expectation.expected_report(other)
    with pytest.raises(Exception):
        expectation.verify(other, expectation.expected_report(scene))


def test_a_scene_without_an_upload_report_still_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scene object that cannot carry an expectation must not be blocked."""

    seen: list[Any] = []

    def capture_rir(_scene, _sim, *, grid, layout_type, hrtf_file_path=None):
        # No upload_expectation keyword at all: the renderer must not pass one.
        seen.append(layout_type)
        return SimpleNamespace(
            layout_type=layout_type,
            layout_id=layout_output_contract(layout_type)["layout_id"],
            channel_labels=(
                ("left", "right")
                if layout_type == "binaural"
                else ("W", "Y", "Z", "X")
            ),
            keyframe_samples=(0,),
            trajectory_sha256=f"t-{layout_type}",
        )

    receipt, _, _ = _render(
        tmp_path, monkeypatch, layouts=("ambisonics",), _rir_override=capture_rir
    )
    assert seen == ["ambisonics"]
    assert receipt["audio"]["layout_delivery"]["ambisonics"]["normalization"] == "N3D"
