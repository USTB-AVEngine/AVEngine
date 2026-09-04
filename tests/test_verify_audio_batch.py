"""Positive controls for the audio-batch verifier: every check layer must
catch its corresponding corruption (工单纪律:检查器先证明能抓坏样本)。"""

from __future__ import annotations

import json
import shutil
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
                registry_ok: bool = True,
                include_execution_variant: bool = True) -> tuple[Path, Path]:
    design = tmp_path / "design"
    programs = design / "programs"
    programs.mkdir(parents=True)
    prog = {"program_id": "qa_v3_dog_v3b_001_rand_v1",
            "candidate_source_endpoint_ids": [EP1, EP2], "events": EVENTS}
    (programs / "qa_v3_dog_v3b_001_rand_v1.json").write_text(json.dumps(prog))
    gate_events = [dict(event) for event in EVENTS]
    gate_events[0]["source_endpoint_id"] = EP2
    gate_events[1]["source_endpoint_id"] = EP1
    (programs / "qa_v3_dog_v3b_001_gateA_rand_v1.json").write_text(
        json.dumps(dict(
            prog,
            program_id="qa_v3_dog_v3b_001_gateA_rand_v1",
            events=gate_events,
        )))
    (programs / "qa_v3_dog_v3b_001_rand_v1.plan.json").write_text(json.dumps(
        {"anchor_slot": "source2", "anchor_start_sample": 48000,
         "anchor_end_sample": 52800, "tail_silence_samples": 27200}))
    point = design / "v3b_001"
    point.mkdir()
    (point / "fact_record.json").write_text(json.dumps({
        "profile_id": "fixture",
        "mcq": {
            "stem": "which source?",
            "options_space": ["blue", "red"],
            "truth_option": "blue",
        },
        "open": {"stem": "which source?", "truth_value": "blue"},
    }))
    (point / "fact_record_gateA.json").write_text(json.dumps({
        "profile_id": "fixture",
        "mcq": {
            "stem": "which source?",
            "options_space": ["blue", "red"],
            "truth_option": "red",
        },
        "open": {"stem": "which source?", "truth_value": "red"},
    }))
    audio = tmp_path / "audio"
    reg = ("examples/qa_v2/source_endpoints_qa_v2_v1.json" if registry_ok
           else "examples/registry/registries/source_endpoints_v1.json")

    def emit(name: str, wav: np.ndarray, prog_name: str) -> None:
        d = audio / name / "audio" / "binaural"
        d.mkdir(parents=True)
        sf.write(d / "mixture.wav", wav, 16000)
        receipt = {
            "qualification_claim": False,
            "inputs": {"source_endpoint_registry": {"path": f"/x/{reg}"}},
            "audio_program": {"path": f"/x/programs/{prog_name}"},
        }
        if include_execution_variant:
            receipt["execution_variant"] = (
                "gateA" if name.endswith("_gateA") else "main"
            )
        (audio / name / "research_receipt.json").write_text(
            json.dumps(receipt)
        )

    emit("v3b_001", wav_main, "qa_v3_dog_v3b_001_rand_v1.json")
    if wav_gatea is not None:
        emit("v3b_001_gateA", wav_gatea, "qa_v3_dog_v3b_001_gateA_rand_v1.json")
    params_p = tmp_path / "params.json"
    params_p.write_text(json.dumps(PARAMS))
    return design, audio


def run(tmp_path, design, audio, out_name="report.json", variants=None):
    out = tmp_path / out_name
    argv = [
        "--design-root", str(design), "--audio-root", str(audio),
        "--params", str(tmp_path / "params.json"),
    ]
    if variants is not None:
        argv.extend(["--variants", variants])
    argv.extend(["--out", str(out)])
    rc = main(argv)
    return rc, json.loads(out.read_text())


def test_clean_batch_passes(tmp_path):
    w = good_wav()
    design, audio = build_batch(tmp_path, w, wav_gatea=good_wav(seed=7))
    rc, rep = run(tmp_path, design, audio)
    assert rc == 0 and rep["failures"] == []
    assert rep["audio_variant_waveform_nonidentity_pairs"] == 1
    assert rep["execution_variant_verification"]["status"] == "verified"
    assert rep["gatea_semantic_flip_pairs"] == 1
    assert "established from current paired fact records" in rep["gatea_semantic_flip"]


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
    assert rep["audio_variant_waveform_nonidentity_pairs"] == 0
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
    rc, _ = run(tmp_path, design, audio, variants="main")
    assert rc == 0
    out = tmp_path / "report.json"
    assert main(["--design-root", str(design), "--audio-root", str(audio),
                 "--params", str(tmp_path / "params.json"),
                 "--out", str(out)]) == 2


def test_onsets_tail_uses_declared_sample_rate():
    sample_rate = 8000
    events = [
        {"source_endpoint_id": EP1, "start_sample": 4000,
         "end_sample_exclusive": 6400},
        {"source_endpoint_id": EP2, "start_sample": 24000,
         "end_sample_exclusive": 26400},
    ]
    rng = np.random.default_rng(17)
    wav = np.zeros((40000, 2), dtype=np.float32)
    for event in events:
        burst = rng.normal(
            0.0, 0.1,
            (event["end_sample_exclusive"] - event["start_sample"], 2),
        ).astype(np.float32)
        burst[:, 1] *= 0.6
        wav[event["start_sample"]:event["end_sample_exclusive"]] = burst
    assert check_onsets(
        wav, {"events": events},
        {"anchor_end_sample": 26400},
        None, 1.5, sample_rate_hz=sample_rate,
    ) == []


def test_gatea_semantics_are_derived_from_programs_for_new_fact_shapes():
    from verify_qa_v3_audio_batch import _gatea_semantic_failures

    main_events = [
        {
            "event_id": "voice_1",
            "source_endpoint_id": "source_a",
            "sound_asset_id": "speech_a",
            "start_tick": 30,
            "end_tick_exclusive": 60,
            "start_sample": 10,
            "end_sample_exclusive": 20,
            "source_start_sample": 0,
            "source_end_sample_exclusive": 10,
            "linear_gain": 1.0,
        },
        {
            "event_id": "voice_2",
            "source_endpoint_id": "source_b",
            "sound_asset_id": "speech_b",
            "start_tick": 90,
            "end_tick_exclusive": 120,
            "start_sample": 30,
            "end_sample_exclusive": 40,
            "source_start_sample": 0,
            "source_end_sample_exclusive": 10,
            "linear_gain": 1.0,
        },
    ]
    gate_events = [dict(row) for row in main_events]
    gate_events[0]["source_endpoint_id"] = "source_b"
    gate_events[1]["source_endpoint_id"] = "source_a"
    main_fact = {
        "mcq": {"stem": "who?", "options_space": ["blue", "red"],
                "truth_option": "blue"},
        "open": {"stem": "who?", "truth_value": "blue"},
    }
    gate_fact = {
        "mcq": {"stem": "who?", "options_space": ["blue", "red"],
                "truth_option": "red"},
        "open": {"stem": "who?", "truth_value": "red"},
    }
    main_program = {
        "candidate_source_endpoint_ids": ["source_a", "source_b"],
        "events": main_events,
    }
    gate_program = {
        "candidate_source_endpoint_ids": ["source_a", "source_b"],
        "events": gate_events,
    }
    assert _gatea_semantic_failures(
        main_fact, gate_fact,
        main_program=main_program, gate_program=gate_program,
    ) == []

    corrupted = json.loads(json.dumps(gate_program))
    corrupted["events"][0]["start_sample"] += 1
    assert "event_times_preserved" in _gatea_semantic_failures(
        main_fact, gate_fact,
        main_program=main_program, gate_program=corrupted,
    )


def test_gatea_semantics_reject_unchanged_assignment_and_gold():
    from verify_qa_v3_audio_batch import _gatea_semantic_failures

    event = {
        "event_id": "voice",
        "source_endpoint_id": "source_a",
        "sound_asset_id": "speech",
        "start_sample": 0,
        "end_sample_exclusive": 10,
        "source_start_sample": 0,
        "source_end_sample_exclusive": 10,
    }
    fact = {
        "mcq": {"stem": "who?", "options_space": ["a", "b"],
                "truth_option": "a"},
    }
    program = {
        "candidate_source_endpoint_ids": ["source_a", "source_b"],
        "events": [event],
    }
    failures = _gatea_semantic_failures(
        fact, fact, main_program=program, gate_program=program,
    )
    assert "audio_assignment_changed" in failures
    assert "question_gold_changed" in failures


def test_execution_variant_mismatch_is_caught(tmp_path):
    design, audio = build_batch(
        tmp_path, good_wav(), wav_gatea=good_wav(seed=7)
    )
    gate_receipt = audio / "v3b_001_gateA" / "research_receipt.json"
    payload = json.loads(gate_receipt.read_text())
    payload["execution_variant"] = "main"
    gate_receipt.write_text(json.dumps(payload))
    rc, rep = run(tmp_path, design, audio)
    assert rc == 1
    assert rep["execution_variant_verification"]["status"] == "failed"
    assert any("wrong execution variant" in failure for failure in rep["failures"])


def test_legacy_execution_variant_receipt_is_compatible_but_unverified(tmp_path):
    design, audio = build_batch(
        tmp_path, good_wav(), include_execution_variant=False
    )
    rc, rep = run(tmp_path, design, audio, variants="main")
    assert rc == 0
    verification = rep["execution_variant_verification"]
    assert verification["status"] == "unverified"
    assert verification["verified_renders"] == []
    assert verification["unverified_renders"] == ["v3b_001"]


def test_requested_pair_rejects_missing_gatea_render(tmp_path):
    design, audio = build_batch(tmp_path, good_wav())
    rc, rep = run(tmp_path, design, audio, variants="main,gateA")
    assert rc == 1
    assert any("missing complete gateA render" in failure for failure in rep["failures"])
    assert rep["complete_pair_count"] == 0


def test_requested_pair_rejects_missing_main_render(tmp_path):
    design, audio = build_batch(
        tmp_path, good_wav(), wav_gatea=good_wav(seed=7)
    )
    shutil.rmtree(audio / "v3b_001")
    rc, rep = run(tmp_path, design, audio, variants="main,gateA")
    assert rc == 1
    assert any("missing complete main render" in failure for failure in rep["failures"])
    assert rep["complete_pair_count"] == 0


def test_empty_render_directory_is_not_a_complete_main(tmp_path):
    design, audio = build_batch(tmp_path, good_wav())
    (audio / "v3b_001" / "research_receipt.json").unlink()
    rc, rep = run(tmp_path, design, audio, variants="main")
    assert rc == 1
    assert rep["execution_variant_verification"]["status"] == "unverified"
    assert any("missing complete main render" in failure for failure in rep["failures"])


def test_gatea_semantic_pair_count_is_required(tmp_path):
    design, audio = build_batch(
        tmp_path, good_wav(), wav_gatea=good_wav(seed=7)
    )
    (design / "v3b_001" / "fact_record_gateA.json").unlink()
    rc, rep = run(tmp_path, design, audio, variants="main,gateA")
    assert rc == 1
    assert any("Gate A semantic pair count" in failure for failure in rep["failures"])
