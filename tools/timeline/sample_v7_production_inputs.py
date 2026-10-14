"""Prepare legal v7 candidate selections; run no native/GPU work or export."""
from pathlib import Path
from collections import Counter, defaultdict
from copy import deepcopy
import argparse, hashlib, json, random
from avengine.timeline.v7_production_inputs import (
    COLORS, asset_category, bind_animal_motion, fingerprint,
    sample_dialogue, sample_room_assets, validate_recipe)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['contract','plan','assets','ue-assets','ue-native-assets','output']:
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--silent-decor-count',type=int,choices=[0,1],default=0,help='explicit optional silent visual decor; still requires original native gates')
    a=p.parse_args()
    if a.output.exists():raise ValueError('candidate output is immutable; use a fresh filename')
    contract=json.loads(a.contract.read_text());plan=json.loads(a.plan.read_text())
    if plan['production_contract']['sha256']!=fingerprint(a.contract)['sha256']:
        raise ValueError('plan and sampler contract differ')
    def assets(path):
        x=json.loads(path.read_text());return x.get('assets',x) if isinstance(x,dict) else x
    cpu=assets(a.assets);human_ue=assets(a.ue_assets);native_ue=assets(a.ue_native_assets)
    by_id={x['asset_id']:x for x in cpu};ue=deepcopy(human_ue)
    for x in ue:
        x.setdefault('kind','human');x.setdefault('controlled_shirt',x.get('color') in COLORS)
    for x in deepcopy(native_ue):
        species=x.get('species_id',x.get('category'))
        if species in {'dog','cat','bird'}:x['species_id']=species
        x.setdefault('kind','animal' if species in {'dog','cat','bird'} else 'object')
        x.setdefault('allowed_classes',by_id.get(x['asset_id'],{}).get('allowed_classes',[]))
        ue.append(x)
    sources=json.loads(Path(contract['inputs']['source_admission']['path']).read_text())['records']
    recorded_vctk=[r for r in sources if r.get('source_type')=='recorded' and 'VCTK' in r.get('source_dataset','') and r['sound_class']=='speech_playback']
    source_pool=defaultdict(list)
    for r in sources:source_pool[r['sound_class']].append(r['sound_asset_id'])
    semantic=json.loads(Path(contract['inputs']['semantic_manifest']['path']).read_text())
    groups=defaultdict(list)
    for s in semantic['sounds']:groups[s['scenario_id']].append(s)
    scenarios=[key for key,rows in sorted(groups.items()) if len({r['role_id'] for r in rows})==2]
    rooms={r['key']:r for r in contract['rooms']};rows=[];voices=Counter();classes=Counter();motion=Counter();families=Counter()
    proof=fingerprint(a.contract);receipt={'manifest_path':proof['path'],'manifest_sha256':proof['sha256']}
    for planned in plan['room_allocation']:
        room=rooms[planned['room_key']];catalog=ue if room['family']=='kujiale' else cpu
        catalog=[item for item in catalog if item['asset_id'] not in contract['policy'].get('temporarily_excluded_asset_ids', {})]
        for index in range(planned['parent_scenes']):
            seed=int(hashlib.sha256((room['key']+':'+str(index)).encode()).hexdigest()[:12],16)
            rng=random.Random(seed);selected=sample_room_assets(catalog,room,seed,controlled_people=3,decor_count=a.silent_decor_count)
            if index%6==0 and selected:
                professionals=[s for s in catalog if s.get('kind')=='human' and not s.get('controlled_shirt')
                               and 'Profession' in str(s.get('source','')) and asset_category(s,{},room['family']) in room['categories']]
                human_slots=[i for i,s in enumerate(selected) if s.get('kind')=='human']
                if professionals and human_slots:selected[human_slots[-1]]=deepcopy(rng.choice(professionals))
            chair_pool=[s for s in catalog if s.get('kind')=='chair' and asset_category(s,{},room['family']) in room['categories']]
            seated_only=False
            if room['family'] != 'kujiale' and chair_pool and (index%6==2 or not selected) and not any(s.get('kind')=='human' for s in selected):
                pool=[s for s in catalog if s.get('kind')=='human' and s.get('controlled_shirt') and s.get('sit_allowed')]
                if pool:selected.append(deepcopy(rng.choice(pool)));seated_only=True
            if index%6==5:
                pool=[s for s in catalog if s.get('kind')=='performer' and asset_category(s,{},room['family']) in room['categories'] and set(s.get('allowed_classes',[])) & set(source_pool)]
                if pool:selected.append(deepcopy(rng.choice(pool)))
            if not selected:raise ValueError('legal room has no sampler-supported candidate assets: '+room['key'])
            row={'schema':'v7-production-candidate-inputs-v1','scene_id':f'v7_candidate_{len(rows):06d}',
                'house':room['house'],'room_id':room['room_id'],'family':room['family'],'split':'train','scene_seed':seed,
                'asset_ids':[s['asset_id'] for s in selected],'source_classes':{},'proposed_sound_asset_ids':{},
                'silent_asset_ids':[s['asset_id'] for s in selected if s.get('silent_decor')],
                'motion_requests':{},'production_sampling':receipt,
                'native_proposal_render_truth_and_question_gates':'not_run; pass these selected inputs to the original family producer',
                'training_table_exported':False}
            for item in selected:
                allowed=[{'telephone':'telephone_bell_ringing','phone_ring':'telephone_bell_ringing'}.get(cl,cl)
                         for cl in item.get('allowed_classes',[])]
                allowed=sorted(set(allowed)&set(source_pool))
                if allowed:
                    cl=rng.choice(allowed);row['source_classes'][item['asset_id']]=[cl]
                    row['proposed_sound_asset_ids'][item['asset_id']]=rng.choice(source_pool[cl])
                aid=item['asset_id']
                if room['family'] != 'kujiale' and aid in contract['animal_motion'] and index%6 in {3,4}:
                    row['motion_requests'][aid]=bind_animal_motion({'speed_m_s':.03},aid,contract)
                    motion[contract['animal_motion'][aid]['mode']]+=1
            humans=[s for s in selected if s.get('kind')=='human']
            if len(humans)>=2 and scenarios:
                scenario=rng.choice(scenarios);roles=sorted({r['role_id'] for r in groups[scenario]})
                genders=[next(r['voice_gender'].lower() for r in groups[scenario] if r['role_id']==role) for role in roles]
                genders=[{'f':'female','m':'male'}.get(g,g) for g in genders]
                # Select the same shirt color from a real matching registered body.
                replacements=[]
                for old,gender in zip(humans[:2],genders):
                    pool=[s for s in catalog if s.get('kind')=='human' and s.get('color')==old['color']
                          and s.get('visual_gender')==gender and asset_category(s,{},room['family']) in room['categories']]
                    if not pool:break
                    replacements.append(deepcopy(rng.choice(pool)))
                if len(replacements)==2:
                    for old,new in zip(humans[:2],replacements):
                        slot=selected.index(old);selected[slot]=new
                        row['asset_ids'][slot]=new['asset_id'];row['source_classes'].pop(old['asset_id'],None);row['proposed_sound_asset_ids'].pop(old['asset_id'],None)
                    sounds=sample_dialogue(semantic,scenario,genders,seed)
                    row['semantic_scenario_id']=scenario;row['semantic_sound_ids']=[s['sound_asset_id'] for s in sounds]
                    row['semantic_human_asset_ids']=[h['asset_id'] for h in replacements]
                    row['semantic_camera_sector_targets_deg']={aid:angle for aid,angle in zip(row['semantic_human_asset_ids'],[0.,31.5])}
                    for s in sounds:voices[s['voice_reference']['speaker_id']]+=1
            if room['family'] != 'kujiale':
                available=[s for s in selected if s.get('kind')=='human' and s['asset_id'] not in row.get('semantic_human_asset_ids',[])]
                if index%6==1 and source_pool.get('footsteps'):
                    available=[s for s in available if s.get('walk_allowed')]
                    if available:
                        human=rng.choice(available);aid=human['asset_id']
                        row['motion_requests'][aid]={'action':'walk','speed_m_s':.15,'intervals_s':[[1.,2.25],[16.,17.25]],'footstep_events':6}
                        row['source_classes'][aid]=['footsteps'];row['proposed_sound_asset_ids'][aid]=rng.choice(source_pool['footsteps'])
                elif (index%6==2 or seated_only) and chair_pool and recorded_vctk:
                    available=[s for s in available if s.get('sit_allowed')]
                    if available:
                        human=rng.choice(available);aid=human['asset_id']
                        gender=human.get('visual_gender');clips=[r for r in recorded_vctk if {'f':'female','m':'male'}.get(str(r.get('gender','')).lower(),str(r.get('gender','')).lower())==gender]
                        if not clips:raise ValueError('no registered complete VCTK clip matching selected seated body gender')
                        # Reuse the original seat-first compiler's admitted chair pool.
                        # A random fixed chair and zero offset discarded the
                        # phase-two physical fit search before native gates.
                        row['motion_requests'][aid]={'action':'sit',
                            'minimum_camera_distance_m':1.6,
                            'chair_horizontal_offset_proposals_m':[
                                [0.,0.,0.],[0.,0.,.04],[0.,0.,-.04],
                                [0.,0.,.08],[0.,0.,-.08],[.04,0.,0.],[-.04,0.,0.]],
                            'require_observed_chair_candidate':True}
                        row['source_classes'][aid]=['speech_playback'];row['proposed_sound_asset_ids'][aid]=rng.choice(clips)['sound_asset_id']
                        row['recorded_speech_constraints']={aid:'preserve complete registered VCTK clip; no authored TTS on moving or seated body'}
                        # Chairs are resolved by the original phase2 compiler,
                        # rather than becoming an audible entity in asset_ids.
                        selected.extend(deepcopy(chair_pool))
            validate_recipe(row,selected,contract)
            for selected_classes in row['source_classes'].values():
                classes.update(selected_classes)
            families[room['family']]+=1;rows.append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows))
    human_motion=Counter(req['action'] for row in rows for aid,req in row['motion_requests'].items() if aid not in contract['animal_motion'])
    summary={'schema':'v7-production-candidate-selection-summary-v1','output':str(a.output),'rows':len(rows),
             'families':dict(families),'actual_voice_counts':dict(voices),'voice_count':len(voices),
             'proposed_source_class_counts':dict(classes),'proposed_animal_motion_counts':dict(motion),
             'all_room_categories_test_exclusion_shirt_uniqueness_motion_authorities_checked':True,
             'proposed_human_motion_counts':dict(human_motion),'performer_candidates':sum(any(by_id.get(aid,{}).get('kind')=='performer' for aid in r['asset_ids']) for r in rows),
             'animal_motion_native_dispatch':'HM3D/MP3D private phase2 only; Kujiale registered producer is static and rejects these motion requests',
             'native_or_GPU_jobs_started':False,'formal_training_questions_claimed':False}
    a.output.with_suffix('.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
