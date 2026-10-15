"""TTS intake retains all raw native evidence and verifies the human subregistry."""
from copy import deepcopy
from avengine.timeline.v7_registered_semantics import dialogue_evidence


def visible_dialogue_evidence(scene,program,receipt,visual,manifest,start,retained_visual):
    sounds={s['sound_asset_id']:s for s in program['records']};bindings={b['event_id']:b for b in receipt['events']}
    ids={e['source_id'] for e in scene['events'] if sounds[bindings[e['event_id']]['sound_asset_id']].get('source_type')=='author_tts'}
    if not ids:raise ValueError('authored semantic scene has no actual TTS human')
    subset=deepcopy(visual)
    for f in subset['frames']:f['actors']=[a for a in f['actors'] if a['actor_id'] in ids]
    proof=dialogue_evidence(scene,program,receipt,subset,manifest,start,retained_visual)
    if proof.get('native_dialogue_evidence')is not None:
        proof['native_dialogue_evidence']['registry_scope']='actual authored-speaking humans only; other registered objects/animals remain in the complete raw native evidence and independent binding audit'
    return proof
