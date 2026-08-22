from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_SPEC = importlib.util.spec_from_file_location(
    "make_paired_ablation", REPOSITORY / "tools/studio/make_paired_ablation.py"
)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
TOOL = importlib.util.module_from_spec(TOOL_SPEC)
TOOL_SPEC.loader.exec_module(TOOL)


def _write_bundle(tmp_path: Path) -> Path:
    audio_dir = tmp_path / "render"
    binaural = audio_dir / "audio" / "binaural"
    binaural.mkdir(parents=True)
    rng = np.random.default_rng(7)
    stem_a = rng.standard_normal((160, 2)).astype(np.float32) * 0.1
    stem_b = rng.standard_normal((160, 2)).astype(np.float32) * 0.1
    wavfile.write(binaural / "beagle_0_muzzle_stem.wav", 16000, stem_a)
    wavfile.write(binaural / "beagle_1_muzzle_stem.wav", 16000, stem_b)
    wavfile.write(binaural / "mixture.wav", 16000, stem_a + stem_b)
    return audio_dir


def test_variants_share_parent_and_transform_correctly(tmp_path: Path) -> None:
    audio_dir = _write_bundle(tmp_path)
    manifest = TOOL.build_paired_ablation(
        audio_dir,
        tmp_path / "pair",
        pair_id="pair_demo_v1",
        mute_stems=("beagle_1_muzzle",),
    )
    assert manifest["schema"] == TOOL.PAIR_SCHEMA
    assert manifest["research_only"] is True
    assert manifest["formal_dataset_count"] == 0
    assert set(manifest["variants"]) == {
        "left_zeroed", "right_zeroed", "mono_folded", "muted_beagle_1_muzzle",
    }

    _, mixture = TOOL.read_float32_stereo_wav(
        audio_dir / "audio" / "binaural" / "mixture.wav"
    )
    _, left = TOOL.read_float32_stereo_wav(tmp_path / "pair" / "left_zeroed.wav")
    assert np.all(left[:, 0] == 0.0)
    np.testing.assert_array_equal(left[:, 1], mixture[:, 1])

    _, mono = TOOL.read_float32_stereo_wav(tmp_path / "pair" / "mono_folded.wav")
    np.testing.assert_array_equal(mono[:, 0], mono[:, 1])
    np.testing.assert_allclose(mono[:, 0], mixture.mean(axis=1), rtol=1e-6)

    _, muted = TOOL.read_float32_stereo_wav(
        tmp_path / "pair" / "muted_beagle_1_muzzle.wav"
    )
    _, stem_a = TOOL.read_float32_stereo_wav(
        audio_dir / "audio" / "binaural" / "beagle_0_muzzle_stem.wav"
    )
    np.testing.assert_allclose(muted, stem_a, atol=1e-6)

    persisted = json.loads(
        (tmp_path / "pair" / "pair_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted["pair_id"] == "pair_demo_v1"
    assert persisted["parent"]["stems"] == ["beagle_0_muzzle", "beagle_1_muzzle"]


def test_rejects_unknown_stem_and_clobber(tmp_path: Path) -> None:
    audio_dir = _write_bundle(tmp_path)
    with pytest.raises(TOOL.PairedAblationError, match="unknown stems"):
        TOOL.build_paired_ablation(
            audio_dir, tmp_path / "pair", pair_id="x", mute_stems=("nope",)
        )
    TOOL.build_paired_ablation(audio_dir, tmp_path / "pair2", pair_id="x")
    with pytest.raises(TOOL.PairedAblationError, match="fresh/no-clobber"):
        TOOL.build_paired_ablation(audio_dir, tmp_path / "pair2", pair_id="x")
