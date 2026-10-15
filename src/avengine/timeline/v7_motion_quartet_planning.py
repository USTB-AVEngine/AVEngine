"""Connect a room-level motion constructor to actual independent demand gaps."""
from copy import deepcopy
from avengine.timeline.v7_demand import demand_snapshot
MOTION_FAMILIES=('QA-06','QA-15','QA-16','QA-17')


def validate_template_request(row,catalog):
    from avengine.assets.roster_motion.navmesh_execution import SPEED_RANGES
    assets={a['asset_id']:a for a in catalog};requests=row.get('motion_during_condition_requests',[])
    # owner 10-15: construct what the room supports instead of failing a scene that serves fewer families.
    if not set(MOTION_FAMILIES)&{r['qa_id']for r in requests}:raise ValueError('motion template compiled no motion family')
    moving=row.get('motion_requests',{})
    if not moving:raise ValueError('motion quartet has no moving physical source')
    for aid,request in moving.items():
        asset=assets[aid];species=asset.get('species_id')or asset.get('kind');mode=request.get('animal_motion_admission',{}).get('mode')
        key='slide'if mode=='slide_no_animation'else species
        lo,hi=SPEED_RANGES[key];speed=float(request['speed_m_s'])
        if not lo<=speed<=hi:raise ValueError(f'motion template {species} speed {speed} outside owner range {lo}..{hi}')
    for request in requests:
        family=request['qa_id'];aid=request['target_asset_id']
        if family=='QA-15'and float(request.get('minimum_distance_change_m',0))<.5:
            raise ValueError('QA-15 construction requires at least 0.5m during the complete sound')
        if family in {'QA-16','QA-17'}:
            if float(request.get('minimum_motion_s',0))<2.:
                raise ValueError('post-sound construction requires at least two silent moving seconds')
            if assets[aid]['kind']=='human'and row['source_classes'].get(aid)!=['speech_playback']:
                raise ValueError('post-sound walking human must speak without footsteps')
    if any(len(classes)!=1 for classes in row.get('source_classes',{}).values()):raise ValueError('one physical human/source keeps one sound type')
    if row.get('phase2_include_footsteps_asset_ids'):raise ValueError('speech motion quartet must not add automatic footsteps')


def plan_room_motion_quartet(base_row,state,room,catalog,semantic,records,pathfinder,*,attempt=0,constructor=None,contract=None):
    extra={}
    if constructor is None:
        from avengine.timeline.v7_motion_quartet_template import motion_quartet_template
        constructor=motion_quartet_template
        if contract is not None:extra['contract']=contract      # animal roles need the production authority
    requested=deepcopy(base_row);row=constructor(requested,room,catalog,semantic,records,pathfinder,attempt=attempt,**extra)
    validate_template_request(row,catalog)
    required=set(base_row.get('scene_demand',{}).get('required_asset_ids',[]))
    if not required<=set(row['asset_ids']):raise ValueError('constructor removed explicitly required physical identities')
    snapshot=demand_snapshot(state,next_questions=70);target=int(state.get('minimum_motion_independent_questions',30))
    row['scene_constructor']='motion_quartet_template'
    row['scene_demand']={**deepcopy(row.get('scene_demand',{})),**snapshot,'focus':'motion_quartet_template',
        'requested_conditions':['walking_human','distance_trend_during_sound','moving_after_source_and_wet_tail_stop','staggered_sources'],
        'required_actual_candidate_families':[f for f in MOTION_FAMILIES if f in {r['qa_id']for r in row.get('motion_during_condition_requests',[])}],
        'batch_minimum_independent_family_counts':{f:target for f in MOTION_FAMILIES},
        'independent_motion_deficits':{f:max(0,target-state.get('family_counts',{}).get(f,0))for f in MOTION_FAMILIES},
        'required_asset_ids':list(row['asset_ids']),'constructor_attempt':int(attempt),
        'proposal_is_not_native_admission':True}
    return row


def plan_native_room_motion_quartet(base_row,state,house,catalog,semantic,records,inputs,*,attempt=0):
    from avengine.acoustics.runtime import load_habitat_runtime
    room=next(r for r in house['checked_rooms']if r['room_label']==base_row['room_id'])
    habitat,_=load_habitat_runtime(runtime_mode='current-installed',**inputs['runtime']);pathfinder=habitat.PathFinder()
    if not pathfinder.load_nav_mesh(house['input']['navmesh_source']):raise ValueError('existing registered NavMesh failed to load read-only')
    contract=inputs.get('production_input_contract')
    if isinstance(contract,str):
        import json
        from pathlib import Path
        contract=json.loads(Path(contract).read_text())
    return plan_room_motion_quartet(base_row,state,room,catalog,semantic,records,pathfinder,attempt=attempt,contract=contract)


def native_template_candidate_counts(row,parent):
    """Measured legal candidates, never requested conditions, certify realization."""
    from pathlib import Path
    import json
    from collections import Counter
    from avengine.timeline.v7_question_independence import independence_key
    parent=Path(parent);unique={};counts=Counter()
    for marker in parent.glob('selected_windows_*_completed.json'):
        record=json.loads(marker.read_text());pool=Path(record['recipe']['pool_directory'])
        for folder in pool.iterdir():
            if not(folder/'questions.json').is_file():continue
            public=json.loads((folder/'questions.json').read_text());gold={g['question_id']:g for g in json.loads((folder/'private_answers.json').read_text())}
            for q in public:
                if q['qa_id']not in MOTION_FAMILIES or q['form']!='open':continue
                key,_=independence_key(row['scene_id'],q,gold[q['question_id']]);unique.setdefault(key,q['qa_id'])
    counts.update(unique.values());missing=[f for f in MOTION_FAMILIES if not counts[f]]
    if missing:raise ValueError('motion quartet missing measured legal candidates: '+','.join(missing))
    return {'status':'pass','independent_legal_family_counts':dict(counts),'original_QA_predicates_changed':False}
