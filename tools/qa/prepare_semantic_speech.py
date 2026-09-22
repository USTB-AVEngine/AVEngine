#!/usr/bin/env python3
"""Synthesize short everyday semantic dialogue pairs with explicit text and answer provenance."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
# Short stable tags keep sound IDs readable; the gender lets a caller give a
# rendered actor a voice that matches what the picture shows.
VOICE_TAG = {"中文男": "zh_m", "英文男": "en_m", "日语男": "ja_m", "中文女": "zh_f",
             "英文女": "en_f", "粤语女": "yue_f", "韩语女": "ko_f"}
VOICE_GENDER = {voice: ("female" if tag.endswith("_f") else "male") for voice, tag in VOICE_TAG.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dialogues", type=Path, default=REPOSITORY/"examples/qa/everyday_semantic_dialogues.json")
    p.add_argument("--cosyvoice-root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--voices", nargs="+", default=["中文男", "英文男", "中文女"])
    p.add_argument("--candidates", type=int, default=2,
                   help="recordings per utterance and voice; speech recognition later keeps the first "
                        "one whose answer is audible")
    p.add_argument("--only", nargs="*", default=None,
                   help="synthesize only these scenario:utterance[:voice] cells, e.g. a speech-recognition retry")
    p.add_argument("--first-candidate", type=int, default=0,
                   help="number and seed new candidates after the ones an earlier run made")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--memory-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20260922)
    args = p.parse_args()
    output, runtime, model = args.output.resolve(), args.cosyvoice_root.resolve(), args.model.resolve()
    if not 0 < args.memory_fraction <= 0.25:
        raise ValueError("this short-speech tool allows at most one quarter of one GPU")
    if not (model/"llm.pt").is_file():
        raise FileNotFoundError("CosyVoice model weights are absent")
    config = json.loads(args.dialogues.read_text())
    for scenario in config["scenarios"]:
        utterances = scenario["utterances"]
        if len(utterances) < 2 or len({u["answer"] for u in utterances}) < 2:
            raise ValueError("semantic examples need different answers to the same visual-conditioned question")
    output.mkdir(parents=True, exist_ok=False)
    # External CosyVoice code/model remain read-only. Its optional frontend
    # resources are exposed from a fresh working directory instead of changing
    # its checkout or installing packages into the shared environment.
    resources = runtime/"pretrained_models/CosyVoice-ttsfrd"
    if resources.exists():
        (output/"pretrained_models").mkdir()
        (output/"pretrained_models/CosyVoice-ttsfrd").symlink_to(resources, target_is_directory=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path[:0] = [str(runtime), str(runtime/"third_party/Matcha-TTS")]
    os.chdir(output)
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    engine = CosyVoice(str(model), load_jit=False, load_trt=False, fp16=False)
    available = engine.list_available_spks()
    if any(v not in available for v in args.voices):
        raise ValueError(f"requested voices absent; available presets: {available}")
    records = []
    wanted = set(args.only) if args.only is not None else None
    for scenario in config["scenarios"]:
        for number, utterance in enumerate(scenario["utterances"]):
            for vi, voice in enumerate(args.voices):
                if wanted is not None and not ({f"{scenario['id']}:{number}", f"{scenario['id']}:{number}:{voice}"} & wanted):
                    continue
                for candidate in range(args.first_candidate, args.first_candidate + args.candidates):
                    torch.manual_seed(args.seed + 1000 * candidate + len(records))
                    sound_id = f"semantic_{scenario['id']}_{number}_{VOICE_TAG[voice]}_c{candidate}_v2"
                    chunks = [x["tts_speech"].detach().cpu() for x in engine.inference_sft(utterance["text"], voice, stream=False)]
                    if not chunks:
                        raise RuntimeError(f"TTS returned no samples for {sound_id}")
                    pcm = torch.cat(chunks, dim=-1)
                    pcm = torchaudio.functional.resample(pcm, engine.sample_rate, 16000).numpy().reshape(-1)
                    if not np.isfinite(pcm).all() or not np.any(pcm):
                        raise ValueError(f"TTS returned silent/nonfinite samples for {sound_id}")
                    peak = float(np.max(np.abs(pcm)))
                    pcm = pcm * (0.7 / peak)
                    path = output/(sound_id+".wav")
                    sf.write(path, pcm, 16000, subtype="PCM_16")
                    info = sf.info(path)
                    record = {"sound_asset_id": sound_id, "path": str(path), "sample_rate_hz": info.samplerate,
                        "sample_count": info.frames, "duration_s": info.duration, "sound_class": "speech_playback",
                        "transcript": utterance["text"], "language": config.get("language", "zh"),
                        "voice_preset": voice, "voice_gender": VOICE_GENDER[voice], "candidate": candidate, "scenario_id": scenario["id"], "utterance_index": number,
                        "semantic_slot": scenario["slot"], "semantic_answer": {k:v for k,v in utterance.items() if k!="text"},
                        "peak_abs": float(np.max(np.abs(pcm))), "source_origin": "locally_synthesized_authored_dialogue",
                        "transcript_status": "tts_requested_text_pending_readback"}
                    records.append(record)
                    print(json.dumps({"sound_asset_id": sound_id, "duration_s": info.duration}, ensure_ascii=False), flush=True)
                    (output/"progress.json").write_text(json.dumps({"recordings": records}, ensure_ascii=False, indent=2))
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "dialogues": config, "sounds": records,
        "tts": {"engine": "CosyVoice-300M-SFT", "candidates": args.candidates, "model": str(model), "runtime": str(runtime),
                "runtime_git_head": subprocess.check_output(["git","-C",str(runtime),"rev-parse","HEAD"],text=True).strip(),
                "python": sys.executable, "voices": args.voices, "seed": args.seed},
        "claim_boundary": "Synthesized dry speech and authored semantic answers; not yet spatially rendered, ASR-verified, or a multimodal-necessity result."}
    (output/"speech_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"status":"completed","recordings":len(records),"manifest":str(output/"speech_manifest.json")}), flush=True)

if __name__ == "__main__":
    main()
