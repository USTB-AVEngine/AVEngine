"""Synthesize complete authored dialogue candidates on CPU without trim/repeat.

This creates research dry assets, not perceptually certified semantic questions.
The native producer must preserve these whole utterances and record their role,
actor, visual and rendered-source evidence before QA-26..28 can be collected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from avengine.timeline.continuous_pilot import FormantSpeech


def synthesize(script_path, output):
    script = json.loads(Path(script_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    engine = FormantSpeech()
    sounds = []
    for scenario in script['scenarios']:
        for utterance in scenario['utterances']:
            for preset in ('en-us+m1', 'en-us+m3'):
                samples = engine.synthesize(utterance['text'], preset)
                if len(samples) == 0 or not np.isfinite(samples).all():
                    raise ValueError('TTS produced an empty or invalid utterance')
                sid = scenario['id'] + '__' + utterance['answer'] + '__' + preset.replace('+', '_')
                path = output / (sid + '.wav')
                sf.write(path, samples, 16000, subtype='FLOAT')
                readback, rate = sf.read(path, always_2d=True)
                if len(readback) != len(samples) or rate != 16000:
                    raise ValueError('complete authored speech changed on disk')
                sounds.append({'sound_asset_id': sid, 'scenario_id': scenario['id'],
                               'script_author': 'project_author', 'transcript': utterance['text'],
                               'semantic_answer': {k: v for k, v in utterance.items() if k != 'text'},
                               'voice_preset': preset, 'voice_gender': 'unverified',
                               'emission_mechanism': 'human_live_formant',
                               'source_real_waveform': False,
                               'qualification': 'research_candidate',
                               'formal_blockers': ['speech_intelligibility_and_output_rights_not_verified',
                                                   'native_dialogue_role_and_actor_evidence_not_rendered'],
                               'synthesis_readback': {'path': str(path), 'sample_rate_hz': 16000,
                                                      'sample_count': len(samples), 'duration_s': len(samples) / 16000,
                                                      'transcript': utterance['text'], 'content_complete': True,
                                                      'trimmed_or_repeated': False,
                                                      'engine': engine.receipt}})
    manifest = {'sounds': sounds, 'dialogues': {'scenarios': script['scenarios']},
                'script_source': str(script_path), 'formal_usable': 0,
                'qualification': 'complete CPU speech candidates; no new native semantic question claim'}
    (output / 'semantic_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--script',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a = p.parse_args();result = synthesize(a.script,a.output)
    print(json.dumps({'complete_speech_clips':len(result['sounds']),'formal_usable':0}))


if __name__ == '__main__':
    main()
