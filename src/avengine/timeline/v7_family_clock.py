"""Compile automatic source timing while retaining each real admitted binding."""
from copy import deepcopy
import math


def build_constructed_scene(row,house,records,catalog,semantic_manifest=None):
    from avengine.timeline.v7_phase2_sources import build_phase2_scene
    from avengine.timeline.v7_conditioned_post_motion import apply_authored_post_schedule
    working,scene,program,assets=build_phase2_scene(row,house,records,catalog,semantic_manifest)
    scene,program=apply_authored_post_schedule(row,scene,program,assets)
    scene['independent_parent_scene_id']=row.get('independent_parent_scene_id',row['scene_id'])
    scene['automatic_construction']=deepcopy(row.get('automatic_construction',{}))
    # Taxonomy calls start after the complete authored utterance. This is a
    # fresh program, not an edit to already rendered samples or truth records.
    clock=row.get('automatic_taxonomy_clock')
    if clock:
        asset=next(a for a in assets if a['asset_id']==clock['asset_id']);ep=asset['endpoint_id']
        events=[e for e in scene['events']if e['endpoint_id']==ep]
        if not events:raise ValueError('automatic taxonomy clock has no actual registered emitting event')
        byid={r['sound_asset_id']:r for r in program['records']}
        bindings={b['event_id']:b for b in program['events']};template=events[0];binding=bindings[template['event_id']]
        if byid[binding['sound_asset_id']].get('source_type')=='author_tts':
            raise ValueError('taxonomy clock cannot relabel authored speech')
        lengths=[r['header']['sample_count']/r['header']['sample_rate_hz']for r in program['records']if r.get('source_type')=='author_tts']
        if not lengths:raise ValueError('automatic taxonomy scene requires its complete authored companion')
        offset=math.ceil((max(lengths)+.5)*10)
        duration=template['end_tick']-template['start_tick']
        if offset+duration>95:raise ValueError('complete source event and authored companion need another legal timing candidate')
        doomed={e['event_id']for e in events}
        scene['events']=[e for e in scene['events']if e['event_id']not in doomed]
        program['events']=[b for b in program['events']if b['event_id']not in doomed]
        for i,epoch in enumerate((0,100,200)):
            eid='automatic_taxonomy_'+ep+'_'+str(i)
            scene['events'].append({**template,'event_id':eid,'start_tick':epoch+offset,'end_tick':epoch+offset+duration})
            program['events'].append({**binding,'event_id':eid})
        scene['automatic_taxonomy_clock']={'source':'actual admitted full binding and complete companion header',
            'offset_ticks':offset,'duration_ticks':duration,'body_or_sound_kind_removed':False}
    if row.get('qa21_taxonomy_schedule'):
        from avengine.construction.joint.families.qa21 import apply_emission_schedule
        scene,program=apply_emission_schedule(row,scene,program,assets)
    scene['events'].sort(key=lambda e:(e['start_tick'],e['end_tick'],e['event_id']))
    order={e['event_id']:i for i,e in enumerate(scene['events'])}
    program['events'].sort(key=lambda e:order.get(e['event_id'],len(order)))
    return working,scene,program,assets


def generate_constructed_phase2_parent(*args,**kwargs):
    """The committed producer dispatches the automatic builder explicitly."""
    from avengine.timeline.v7_phase2_generation import generate_phase2_parent
    return generate_phase2_parent(*args,**kwargs)
