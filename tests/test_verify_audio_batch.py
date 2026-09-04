"""Positive controls for the audio-batch verifier: every check layer must
catch its corresponding corruption (工单纪律:检查器先证明能抓坏样本)。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from verify_qa_v3_audio_batch import check_onsets, main  # noqa: E402

EP1, EP2 = "qa_v2_dog_1_collie_muzzle", "qa_v2_dog_2_labrador_muzzle"
EVENTS = [
    {"source_endpoint_id": EP1, "start_sample": 8000,
     "end_sample_exclusive": 12800},
    {"source_endpoint_id": EP2, "start_sample": 48000,
     "end_sample_exclusive": 52800},
]
PARAMS = {"TAIL_MIN_S": 1.5}


def good_wav(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wav = np.zeros((80000, 2), dtype=np.float32)
    for e in EVENTS:
        s0, s1 = e["start_sample"], e["end_sample_exclusive"]
        burst = rng.normal(0, 0.1, (s1 - s0, 2)).astype(np.float32)
        burst[:, 1] *= 0.6          # 通道差
        wav[s0:s1] = burst
    return wav


def build_batch(tmp_path: Path, wav_main: np.ndarray,
                wav_gatea: np.ndarray | None = None,
                registry_ok: bool = True) -> tuple[Path, Path]:
    design = tmp_path / "design"
    programs = design / "programs"
    programs.mkdir(parents=True)
    prog = {"program_id": "qa_v3_dog_v3b_001_rand_v1",
            "candidate_source_endpoint_ids": [EP1, EP2], "events": EVENTS}
    (programs / "qa_v3_dog_v3b_001_rand_v1.json").write_text(json.dumps(prog))
    (programs / "qa_v3_dog_v3b_001_gateA_rand_v1.json").write_text(
        json.dumps(dict(prog, program_id="qa_v3_dog_v3b_001_gateA_rand_v1")))
    (programs / "qa_v3_dog_v3b_001_rand_v1.plan.json").write_text(json.dumps(
        {"anchor_slot": "source2", "anchor_start_sample": 48000,
         "anchor_end_sample": 52800, "tail_silence_samples": 27200}))
    audio = tmp_path / "audio"
    reg = ("examples/qa_v2/source_endpoints_qa_v2_v1.json" if registry_ok
           else "examples/registry/registries/source_endpoints_v1.json")

    def emit(name: str, wav: np.ndarray, prog_name: str) -> None:
        d = audio / name / "audio" / "binaural"
        d.mkdir(parents=True)
        sf.write(d / "mixture.wav", wav, 16000)
        (audio / name / "research_receipt.json").write_text(json.dumps({
            "qualification_claim": False,
            "inputs": {"source_endpoint_registry": {"path": f"/x/{reg}"}},
            "audio_program": {"path": f"/x/programs/{prog_name}"},
        }))

    emit("v3b_001", wav_main, "qa_v3_dog_v3b_001_rand_v1.json")
    if wav_gatea is not None:
        emit("v3b_001_gateA", wav_gatea, "qa_v3_dog_v3b_001_gateA_rand_v1.json")
    params_p = tmp_path / "params.json"
    params_p.write_text(json.dumps(PARAMS))
    return design, audio


def run(tmp_path, design, audio, out_name="report.json"):
    out = tmp_path / out_name
    rc = main(["--design-root", str(design), "--audio-root", str(audio),
               "--params", str(tmp_path / "params.json"), "--out", str(out)])
    return rc, json.loads(out.read_text())


def test_clean_batch_passes(tmp_path):
    w = good_wav()
    design, audio = build_batch(tmp_path, w, wav_gatea=good_wav(seed=7))
    rc, rep = run(tmp_path, design, audio)
    assert rc == 0 and rep["failures"] == []
    assert rep["audio_variant_waveform_nonidentity_pairs"] == 1
    # 这一层只说明音频变体非同一;Gate A 语义要另行逐题核验
    assert "not established by this tool" in rep["gatea_semantic_flip"]


def test_mono_fold_caught(tmp_path):
    w = good_wav()
    w[:, 1] = w[:, 0]                      # 单声道折叠
    design, audio = build_batch(tmp_path, w)
    rc, rep = run(tmp_path, design, audio)
    assert rc == 1
    assert any("identical" in f for f in rep["failures"])


def test_wrong_registry_caught(tmp_path):
    design, audio = build_batch(tmp_path, good_wav(), registry_ok=False)
    rc, rep = run(tmp_path, design, audio)
    assert rc == 1
    assert any("wrong endpoint registry" in f for f in rep["failures"])


def test_silent_event_caught(tmp_path):
    w = good_wav()
    w[8000:12800] = 0.0                    # 第一事件静默(渲染丢事件)
    design, audio = build_batch(tmp_path, w)
    rc, rep = run(tmp_path, design, audio)
    assert rc == 1
    assert any("zero energy" in f for f in rep["failures"])


def test_gatea_identical_caught(tmp_path):
    w = good_wav()
    design, audio = build_batch(tmp_path, w, wav_gatea=w.copy())
    rc, rep = run(tmp_path, design, audio)
    assert rc == 1
    assert any("gateA identical" in f for f in rep["failures"])


def test_current_fact_binds_identity_purpose_back_to_program_event():
    fact = {
        "profile_id": "card1F",
        "audio": {"events": [
            {"start_sample": 48000, "purpose": "identity_anchor"}
        ]},
    }
    program = {"events": EVENTS}
    assert check_onsets(good_wav(), program, fact, "card1F", 1.5) == []


def test_no_clobber(tmp_path):
    design, audio = build_batch(tmp_path, good_wav())
    rc, _ = run(tmp_path, design, audio)
    assert rc == 0
    out = tmp_path / "report.json"
    assert main(["--design-root", str(design), "--audio-root", str(audio),
                 "--params", str(tmp_path / "params.json"),
                 "--out", str(out)]) == 2
