#!/usr/bin/env python3
"""Read synthesized semantic speech back with a local Whisper model and admit the audible recordings.

A recording is admitted only when recognition hears one of its utterance's
``audible`` tokens. Whole-sentence character error rate is reported but does
not decide: a clip whose answer word is garbled mislabels its question even at
a low error rate, and one that only differs in punctuation or script is fine.
The first admitted candidate of each utterance and voice goes into a separate
manifest; the synthesized manifest is left as it was.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re

_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
          "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _numerals(text):
    def tens(match):
        high = _DIGIT[match.group(1)] if match.group(1) else 1
        low = _DIGIT[match.group(2)] if match.group(2) else 0
        return str(high * 10 + low)
    text = re.sub(r"([一二两三四五六七八九])?十([一二三四五六七八九])?", tens, text)
    return "".join(str(_DIGIT[c]) if c in _DIGIT else c for c in text)


def normalized(text):
    # Whisper often answers Mandarin in traditional characters and writes
    # hours as digits; neither changes what was said.
    try:
        import zhconv
        text = zhconv.convert(text, "zh-cn")
    except ImportError:
        pass
    return re.sub(r"[^a-z0-9一-鿿]", "", _numerals(text.casefold()))


def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def audible(sound, recognized):
    tokens = sound["semantic_answer"].get("audible") or [sound["semantic_answer"]["label_zh"]]
    heard = normalized(recognized)
    return [t for t in tokens if normalized(t) and normalized(t) in heard]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="readback receipt")
    p.add_argument("--admitted", type=Path, required=True, help="manifest of admitted recordings")
    p.add_argument("--merge-with", type=Path, nargs="*", default=(),
                   help="earlier admitted manifests; a record is kept only while its text still matches the dialogues")
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()
    for path in (args.output, args.admitted):
        if path.exists():
            raise FileExistsError(path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.12, 0)
    processor = WhisperProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(args.model), local_files_only=True, torch_dtype=torch.float16).to("cuda").eval()
    try:
        prompt = processor.get_prompt_ids("以下是普通话的句子。", return_tensors="pt").to("cuda")
    except Exception:
        prompt = None
    config = json.loads(args.manifest.read_text())
    rows = []
    for sound in config["sounds"]:
        pcm, rate = sf.read(sound["path"], dtype="float32")
        if rate != 16000 or pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError("ASR expects finite mono 16 kHz audio")
        inputs = processor(pcm, sampling_rate=rate, return_tensors="pt", return_attention_mask=True)
        # The simplified-Chinese prompt helps most clips, but Whisper sometimes
        # returns the prompt itself as the transcript; either decode may count.
        texts = []
        for use_prompt in ((True, False) if prompt is not None else (False,)):
            options = dict(language="zh", task="transcribe", max_new_tokens=96)
            if use_prompt:
                options["prompt_ids"] = prompt
            with torch.inference_mode():
                ids = model.generate(inputs.input_features.to("cuda", dtype=torch.float16),
                                     attention_mask=inputs.attention_mask.to("cuda"), **options)
            decoded = processor.batch_decode(ids, skip_special_tokens=True)[0]
            texts.append(decoded.replace("以下是普通话的句子。", "").strip())
        heard_by = [audible(sound, t) for t in texts]
        best = next((i for i, h in enumerate(heard_by) if h), 0)
        text, heard = texts[best], heard_by[best]
        truth, observed = normalized(sound["transcript"]), normalized(text)
        row = {"sound_asset_id": sound["sound_asset_id"], "requested": sound["transcript"],
               "recognized": text, "decodes": texts, "character_error_rate": edit_distance(truth, observed) / max(1, len(truth)),
               "audible_tokens_heard": heard, "answer_audible": bool(heard)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    by_id = {r["sound_asset_id"]: r for r in rows}
    chosen, missing = {}, []
    current = {(sc["id"], i): u["text"] for sc in config["dialogues"]["scenarios"]
               for i, u in enumerate(sc["utterances"])}
    stale = 0
    for earlier in args.merge_with:
        for record in json.loads(Path(earlier).read_text())["sounds"]:
            key = (record["scenario_id"], record["utterance_index"], record["voice_preset"])
            if current.get(key[:2]) != record["transcript"]:
                stale += 1  # the dialogues changed this utterance; its old recording is void
                continue
            chosen.setdefault(key, record)
    for sound in config["sounds"]:
        key = (sound["scenario_id"], sound["utterance_index"], sound["voice_preset"])
        if key in chosen or not by_id[sound["sound_asset_id"]]["answer_audible"]:
            continue
        record = deepcopy(sound)
        record["transcript_status"] = "asr_answer_audible"
        record["asr_readback"] = {k: by_id[sound["sound_asset_id"]][k]
                                  for k in ("recognized", "character_error_rate", "audible_tokens_heard")}
        chosen[key] = record
    voices = sorted({s["voice_preset"] for s in config["sounds"]} | {k[2] for k in chosen})
    for key in [(sid, i, v) for (sid, i) in current for v in voices]:
        if key not in chosen and key not in missing:
            missing.append(key)
    admitted = deepcopy(config)
    admitted["sounds"] = list(chosen.values())
    admitted["admission"] = {"rule": "first candidate whose answer token is heard by speech recognition",
                             "asr_model": str(args.model), "readback": str(args.output),
                             "admitted": len(chosen), "merged_from": [str(x) for x in args.merge_with],
                             "stale_recordings_dropped": stale,
                             "utterance_voices_without_an_audible_candidate": [list(k) for k in missing]}
    receipt = {"model": str(args.model), "rows": rows,
               "answer_audible": sum(r["answer_audible"] for r in rows), "recordings": len(rows),
               "utterance_voices_admitted": len(chosen),
               "utterance_voices_without_an_audible_candidate": [list(k) for k in missing],
               "claim_boundary": "ASR readback is a content check on the answer word; it is not a speech-quality or human-admission verdict."}
    with args.output.open("x") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    with args.admitted.open("x") as f:
        json.dump(admitted, f, ensure_ascii=False, indent=2)
    print(json.dumps({"admitted": len(chosen), "missing": len(missing)}), flush=True)


if __name__ == "__main__":
    main()
