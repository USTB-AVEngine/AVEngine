"""Produce R5 binding evidence from complete native quartets, never design arrays.

The standing speech profile certifies three applicable kinds. Instrument and
playback alternatives remain deferred until their native device intake exists.
The QA authority rechecks every pair, physical capability and public domain.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json,os,sqlite3
import numpy as np
import soundfile as sf
from avengine.timeline.continuous_pilot import write_json
from avengine.timeline.v7_candidate_generation import read_json,stage
from avengine.timeline.continuous_scene import partitioned_convolve

CONTENT_BRIDGE_REVISION='registered-native-content-quartet-frozen-v2'
MEMBERS=('v0_a0','v0_a1','v1_a0','v1_a1')


def retain_readback(path,value):
    path=Path(path)
    if path.exists():
        if read_json(path)!=value:raise ValueError('immutable content witness changed')
    else:write_json(path,value)


def _source_components(root,scene,factor):
    marker=read_json(root/'dry_completed.json');folder=root/marker['attempt']
    dry={ep:np.load(folder/(ep+'.npy'),allow_pickle=False) for ep in scene['sources']}
    if factor=='a1':dry['voice1'],dry['voice2']=dry['voice2'],dry['voice1']
    cache=root/'rir_cache';db=sqlite3.connect(f'file:{cache/"positions.sqlite3"}?mode=ro',uri=True)
    try:poses={pose:index for index,pose,complete in db.execute('select id,pose,completed from rirs') if complete}
    finally:db.close()
    stems={};supports={}
    for ep,spec in scene['sources'].items():
        if not np.any(dry[ep]):continue
        point=spec['trajectory'][0]['position_m']
        if any(k['position_m']!=point for k in spec['trajectory']):raise ValueError('content bridge supports registered static emitters only')
        key=json.dumps(point,separators=(',',':'));index=poses[key]
        rir=np.load(cache/f'rir_{index:06d}.npy',allow_pickle=False)
        stems[ep]=partitioned_convolve(dry[ep],[rir],[0])
        # Exact finite input/filter support proves silence without introducing
        # an audibility/SNR threshold for the other physical emitters.
        samples=np.flatnonzero(dry[ep]);supports[ep]=(samples,len(rir[0]))
    return stems,supports


def _verified_native(visual,folder,scene,assets,roster,namespace):
    from avengine.qa.v7_roster_intake import standing_mouth_world
    from avengine.timeline.v7_registered_generation import _verify_window
    media_clock=_verify_window(folder)
    if visual['frame_count']!=450 or len(visual['frames'])!=450:raise ValueError('complete native world required')
    align=visual['alignment']
    if any(align[k]!=0 for k in ('root_max_error','joint_max_error','camera_max_error','link_max_error')) or align['restored_modal_RGB_depth_semantic_frames']!=450:raise ValueError('native state restoration failed')
    if visual['source_material_checks']['status']!='pass' or not all(r['status']=='pass' and r['known_bad_detected'] for r in visual['standing_admission']):raise ValueError('native material or standing gate failed')
    first=round(scene['_content_start_s']*15);error=0.
    for frame in visual['frames'][first:first+150]:
        if len({a['actor_id'] for a in frame['actors']})!=len(frame['actors']):raise ValueError('duplicate actual native actor IDs')
        for actor in frame['actors']:
            spec=scene['sources'][actor['endpoint_id']]
            if spec['actor_id']!=actor['actor_id'] or np.linalg.norm(np.asarray(spec['trajectory'][0]['position_m'])-actor['emitter_position_m'])>.002:raise ValueError('event source identity or emitter differs from actual native frame')
            asset=assets[actor['asset_id']]
            if actor['source_registry_sha256']!=namespace['original_native_registry_sha256'] or actor['loaded_artifact_sha256']!=asset['actual_artifact_sha256'][asset['habitat_glb']]:
                raise ValueError('actual loaded actor differs from formal standing registry')
            expected=standing_mouth_world(asset,actor['body_to_world_m'])
            error=max(error,float(np.linalg.norm(np.asarray(actor['emitter_position_m'])-expected)))
    if error>.002:raise ValueError('native mouth attachment exceeds unchanged 2mm criterion')
    return {'status':'pass','actual_media_clock':media_clock,'complete_native_frames':150,'mouth_max_error_m':error,
            'standing_material_masks_status':'pass','source_registry_sha256':roster['source_registry_sha256']}


def augment_content_binding(row,members,root):
    """Return new immutable pools; the original window candidates stay intact."""
    from avengine.qa.v7_roster_intake import intake_roster
    from avengine.qa.v7_content_binding import collect_content_binding,content_truth,validate_modality_witness
    from avengine.capture.v7_alpha_native import rotate_world_foa
    root=Path(root);recipe=read_json(root/'input_recipe.json');program=recipe['recorded_program']
    if set(m['member'] for m in members)!=set(MEMBERS):return members,{'status':'not_applicable_single_parent'}
    if any(r.get('source_type')!='author_tts' for r in program['records']):
        return members,{'status':'deferred','reason':'standing content proof currently requires native authored speech without unregistered non-speech mechanisms'}
    from avengine.timeline.v7_audio_sources import source_blockers
    if any(source_blockers(record) for record in program['records']):raise ValueError('actual authored source admission failed')
    from avengine.timeline.v7_registry_snapshot import resolve_registry,qualify_namespace
    reference_path=Path(members[0]['visual_directory'])/'native_alpha_visual_evidence.json'
    registry,original_registry_sha,mapped_sha=resolve_registry(reference_path,read_json(reference_path),recipe['assets'][0]['source_registry'])
    roster=intake_roster(registry)
    # Reuse a fully published content stage after a later selection/seal
    # failure. The original source hashes and every required file still bind.
    cached=[]
    for member in members:
        name=member['member'];frozen=root/('frozen_dialogue_pools_'+name+'_completed.json')
        marker=root/('content_pools_'+name+'_completed.json')
        if not frozen.is_file() or not marker.is_file():break
        fp=read_json(frozen);cp=read_json(marker);frozen_directory=root/fp['attempt']
        wanted={'revision':CONTENT_BRIDGE_REVISION,'source_pool':str(frozen_directory)}
        if (fp['recipe']['original_native_registry_sha256']!=original_registry_sha
                or fp['recipe']['actual_registry_sha256']!=mapped_sha or cp['recipe']!=wanted):
            raise ValueError('completed content registry or recipe changed')
        if any(not (root/pointer['attempt']/file).is_file() for pointer in (fp,cp) for file in pointer['required_files']):
            raise ValueError('published content pool lost a required artifact')
        cached.append({**member,'windows_directory':str(root/cp['attempt']),
                       'windows':cp['result']['windows'],'content_pool_resumed':True})
    if len(cached)==len(members) and (root/'content_binding/summary.json').is_file():
        return cached,read_json(root/'content_binding/summary.json')
    # Re-evaluate dialogue questions against the exact retained registry. This
    # is a new immutable QA pool; old source facts, media and rejected pools stay
    # intact. Only the manifest file path changes; its bytes/hash and artifact root stay intact.
    from avengine.timeline.v7_full_question_delivery import collect_native_pool
    qualified=[]
    for member in members:
        source=Path(member['windows_directory']);name=member['member']
        def requalify(attempt):
            required=[];windows=[]
            visual=read_json(Path(member['visual_directory'])/'native_alpha_visual_evidence.json')
            visual['native_appearance_review']=read_json(Path(member['visual_directory'])/'native_appearance_review.json')['review']
            for wi,window in enumerate(member['windows']):
                wid=window['window_id'];old=source/wid;dest=attempt/wid;dest.mkdir()
                ev=read_json(old/'evidence.json');proof=deepcopy(ev['source_qa_evidence'])
                if not proof.get('native_dialogue_evidence'):raise ValueError('no actual native dialogue attachment to requalify')
                native_shas={a['standing_pose_readback']['source_registry_sha256'] for a in proof['native_dialogue_evidence']['registered_actors']}
                if not native_shas.issubset({original_registry_sha,mapped_sha}):raise ValueError('actual native dialogue registry version differs from retained snapshot')
                vp=Path(member['visual_directory'])/'native_alpha_visual_evidence.json'
                _,ns=qualify_namespace(visual,registry,original_registry_sha,vp.with_name('registry_namespace_readback.json'),vp)
                for actor in proof['native_dialogue_evidence']['registered_actors']:
                    stand=actor['standing_pose_readback'];stand['source']=ns['source']
                    stand['source_registry_sha256']=mapped_sha;stand['original_native_registry_sha256']=original_registry_sha
                proof['semantic_manifest']['source_registry']=registry
                proof['semantic_manifest']['retained_registry_sha256']=mapped_sha
                # Partial unknown occluder IDs remain measured/private. They
                # cannot be supplied as complete unique-occlusion evidence.
                if proof.get('occluder_interface_status',{}).get('status')=='measured_partial':
                    proof.pop('occluder_registry',None);proof.pop('occluder_evidence',None)
                media={k:str(old/n) for k,n in [('video','video.mp4'),('audio_foa','audio_foa.wav'),('audio_mono','audio_mono.wav')]}
                quota=row['question_quota_base']+row['members'].index(name)*row['window_count']+wi
                q,g,audit=collect_native_pool(ev['scene'],ev['start_s'],wid,visual,media,proof,quota,row['questions_per_window'])
                from avengine.qa.v7_dialogue_binding import audit_semantic_wording_exchange
                semantic=audit.get('semantic_manifest');facts=audit.get('family_fact_overrides',{}).get('QA-26',audit['facts'])
                wording=audit_semantic_wording_exchange(facts,semantic,seed=str(row['scene_seed']),variant_id=wid) if semantic else {'status':'blocked','reason':'no_registered_semantic_manifest'}
                if wording['status']!='pass':raise ValueError('actual dialogue standard/alternate wording exchange failed: '+str(wording))
                audit['native_semantic_wording_exchange']=wording;ev['source_qa_evidence']=proof
                ev['registry_resolution']={'status':'exact_snapshot','sha256':mapped_sha,'original_native_sha256':original_registry_sha,'snapshot':registry,'original_media_and_native_facts_unchanged':True}
                for n in ('video.mp4','audio_foa.wav','audio_mono.wav'):os.link(old/n,dest/n)
                for n,value in [('questions.json',q),('private_answers.json',g),('candidate_audit.json',audit),('evidence.json',ev)]:write_json(dest/n,value)
                required.extend(f'{wid}/{n}' for n in ('video.mp4','audio_foa.wav','audio_mono.wav','questions.json','private_answers.json','candidate_audit.json','evidence.json'))
                windows.append({'window_id':wid,'logical_candidates':len({x['logical_question_id'] for x in q})})
            return {'windows':windows},required
        directory,result,resumed=stage(root,'frozen_dialogue_pools_'+name,{'revision':'frozen-native-dialogue-qa-pool-v1','source_pool':str(source),'actual_registry_sha256':mapped_sha,'original_native_registry_sha256':original_registry_sha,'retained_snapshot':registry},requalify)
        qualified.append({**member,'windows_directory':str(directory),'windows':result['windows'],'frozen_dialogue_pool_resumed':resumed})
    members=qualified
    assets={r['asset_id']:r for r in roster['records']};by_member={m['member']:m for m in members}
    contexts={};visuals={};component_cache={};counts=Counter();deferrals=[];pools={m:[] for m in MEMBERS};proofroot=root/'content_binding';proofroot.mkdir(exist_ok=True)
    for member in MEMBERS:
        m=by_member[member];vd=Path(m['visual_directory']);visuals[member]=read_json(vd/'native_alpha_visual_evidence.json')
        for wi,w in enumerate(m['windows']):
            folder=Path(m['windows_directory'])/w['window_id'];ev=read_json(folder/'evidence.json');scene=ev['scene'];start=ev['start_s'];scene=deepcopy(scene);scene['_content_start_s']=start
            vp=Path(m['visual_directory'])/'native_alpha_visual_evidence.json'
            _,namespace=qualify_namespace(visuals[member],registry,original_registry_sha,vp.with_name('registry_namespace_readback.json'),vp)
            proof=_verified_native(visuals[member],folder,scene,assets,roster,namespace)
            audio=read_json(Path(m['audio_directory'])/'audio_evidence.json')
            if audio['status']!='pass' or any(not d['passed_5deg'] for d in audio['directions']) or audio['camera_rotation_energy_absolute_error']>1e-10:raise ValueError('native acoustic direction/rotation admission failed')
            factor=member.split('_')[1]
            if factor not in component_cache:
                stems,supports=_source_components(root,scene,factor);length=max(x.shape[1] for x in stems.values());world=np.zeros((4,length))
                for x in stems.values():world[:,:x.shape[1]]+=x
                component_cache[factor]=(stems,supports,rotate_world_foa(world,visuals[member]['camera_frame_readbacks']))
            stems,supports,expected=component_cache[factor];lo=round(start*16000)
            actual,_=sf.read(folder/'audio_foa.wav',dtype='float32',always_2d=True)
            if not np.array_equal(actual,expected[:,lo:lo+160000].T.astype('float32')):raise ValueError('content source/RIR reconstruction differs from delivered FOA')
            proof['FOA_reconstruction']='float32_samplewise_identical';proof['audio_gate']=audio['status']
            contexts[member,wi]={'folder':folder,'ev':ev,'scene':scene,'start':start,'proof':proof,'stems':stems,'supports':supports}
    for wi in range(row['window_count']):
        anchor=None
        for t in np.arange(.05,4.,.05):
            unique=[]
            for member in MEMBERS:
                c=contexts[member,wi];absolute=round((c['start']+t)*16000);active=[];causal=[]
                for ep,(samples,nrir) in c['supports'].items():
                    if np.any((samples>=absolute-nrir+1)&(samples<=absolute+1)):causal.append(ep)
                source= c['ev']['source_qa_evidence']['events']
                for event in c['scene']['events']:
                    if any(a<=absolute<b for a,b in source[event['event_id']]['activity_intervals_samples_global']):active.append(event['endpoint_id'])
                active=set(active)
                if len(active)!=1 or set(causal)!=active:break
                ep=next(iter(active))
                if ep not in {'voice1','voice2'} or abs(c['stems'][ep][0,absolute])<=64*np.finfo(float).eps*np.max(np.abs(c['stems'][ep][0])):break
                frame=round((c['start']+t)*15);vis=visuals[member]
                if any(vis['pixel_visibility_truth']['per_instance'][str(a['actor_id'])]['frames'][frame]['visible_pixels']<20 for a in vis['frames'][frame]['actors'] if a['endpoint_id'] in {'voice1','voice2'}):break
                unique.append(ep)
            if len(unique)==4:anchor=float(round(t,6));break
        if anchor is None:
            deferrals.append({'window_index':wi,'reason':'no_measured_visible_single_emitter_causal_anchor_shared_by_actual_quartet'})
            continue
        scenes={};queries={};domains={};media={}
        for member in MEMBERS:
            c=contexts[member,wi];v=visuals[member];fi=round((c['start']+anchor)*15);appearance=v['native_appearance_review']['actors'] if 'native_appearance_review' in v else read_json(Path(by_member[member]['visual_directory'])/'native_appearance_review.json')['review']['actors']
            actors=[]
            native=v['frames'][fi]['actors'];order=sorted((a for a in native if a['visible_pixels']>0),key=lambda a:a['mask_centroid_px'][0])
            rank={a['actor_id']:i+1 for i,a in enumerate(order)}
            for a in native:
                ar=appearance.get(str(a['actor_id']));visible=a['visible_pixels']>=20
                if ar and ar['status']=='reviewed':descriptor=ar['label_zh']
                elif visible:descriptor=f'画面中从左数第{rank[a["actor_id"]]}个人'
                else:descriptor=f'当前不可见的第{a["actor_id"]}个人'
                actors.append({'actor_id':a['actor_id'],'asset_id':a['asset_id'],'pose':'stand','descriptor_zh':descriptor,'visible':visible,
                    'capability':assets[a['asset_id']]['capability'],'body_to_world_m':a['body_to_world_m'],
                    'standing_pose_readback':{'status':'pass','source':str(Path(by_member[member]['visual_directory'])/'registry_namespace_readback.json'),
                        'actor_id':a['actor_id'],'asset_id':a['asset_id'],'source_registry_sha256':mapped_sha,'original_native_registry_sha256':a['source_registry_sha256']}})
            emissions=[];activity=c['ev']['source_qa_evidence']['events'];by_actor={a['actor_id']:a for a in native};source_onsets={};listener_onsets={}
            for event in c['scene']['events']:
                measured=[[max(0.,a/16000-c['start']),min(10.,b/16000-c['start'])] for a,b in activity[event['event_id']]['activity_intervals_samples_global'] if a<(c['start']+10)*16000 and b>c['start']*16000]
                if not measured:continue
                actor=event['source_id'];source_onsets[actor]=min(source_onsets.get(actor,10.),measured[0][0])
                emissions.append({'emission_id':event['event_id'],'mechanism':'actor_live','actor_id':actor,'content_class':'speech','emitter_point_kind':'mouth',
                    'native_emitter_world_m':by_actor[actor]['emitter_position_m'],'measured_activity_s':measured})
            for ep,stem in c['stems'].items():
                segment=stem[0,round(c['start']*16000):round((c['start']+10)*16000)];threshold=64*np.finfo(float).eps*np.max(np.abs(stem[0]));nz=np.flatnonzero(np.abs(segment)>threshold)
                if len(nz):listener_onsets[c['scene']['sources'][ep]['actor_id']]=int(nz[0])/16000
            first_source=min(source_onsets,key=source_onsets.get);first_listener=min(listener_onsets,key=listener_onsets.get)
            order_pass=first_source==first_listener and list(source_onsets.values()).count(source_onsets[first_source])==1 and list(listener_onsets.values()).count(listener_onsets[first_listener])==1
            receipt_path=proofroot/f'native_{member}_w{wi:02d}.json'
            native_receipt={**c['proof'],'single_emitter_anchor_s':anchor,'causal_other_emitter_supports_empty':True,
                'source_actor_onsets_s':source_onsets,'listener_component_onsets_s':listener_onsets,'source_order_listener_order_agree':order_pass,
                'listener_onset_numeric_floor':'64*float64_eps*full_component_max; machine roundoff, not an audibility gate',
                'original_QC_sources':[r['path'] for r in program['records']]}
            retain_readback(receipt_path,native_receipt)
            keys=['video','audio','pixel_actor_identity','measured_source_activity','asset_license_and_capability','physical_emitter_attachment','audible_emitter_uniqueness']
            readbacks={k:{'status':'pass','source':str(receipt_path)} for k in keys}
            readbacks['source_order_listener_order_agree']={'status':'pass' if order_pass else 'blocked','source':str(receipt_path)}
            scenes[member]={'window_s':[0.,10.],'actors':actors,'emissions':emissions,'native_readback':readbacks}
            blue=next(a for a in actors if a['descriptor_zh']=='蓝色上衣的人')
            queries[member]=[{'kind':'same_class_actor','time_s':anchor,'public_time_s':anchor,'content_class':'speech'},
                {'kind':'offscreen_origin','time_s':anchor,'public_time_s':anchor,'subject_actor_id':blue['actor_id'],'subject_descriptor_zh':blue['descriptor_zh']},
                {'kind':'first_actor'}]
            domains[member]=sorted(a['descriptor_zh'] for a in actors if a['visible']);media[member]={'video_path':str(c['folder']/'video.mp4'),'audio_path':str(c['folder']/'audio_foa.wav')}
        group_path=proofroot/f'group_w{wi:02d}.json';records={m:[] for m in MEMBERS};pair_audits=[]
        for qi in range(3):
            for member in MEMBERS:
                v,a=member.split('_');partners={'same_V_different_A_answer_flip':v+'_'+('a1' if a=='a0' else 'a0'),
                    'same_A_different_V_answer_flip':('v1' if v=='v0' else 'v0')+'_'+a};witness={}
                domain=['是','否'] if qi==1 else domains[member]
                for axis,other in partners.items():
                    other_domain=['是','否'] if qi==1 else domains[other]
                    pair={'scenes':[scenes[member],scenes[other]],'queries':[queries[member][qi],queries[other][qi]],
                        'media':[media[member],media[other]],'answer_domains':[domain,other_domain]}
                    check=validate_modality_witness(pair,axis);pair_audits.append({'member':member,'kind':queries[member][qi]['kind'],**check})
                    witness[axis]={'status':check['status'],'source':str(group_path),'native_pair':pair}
                records[member].append({'query':queries[member][qi],'binding_witness':witness,'answer_domain':domain})
        retain_readback(group_path,{'status':'measured','revision':CONTENT_BRIDGE_REVISION,'actual_media_pairs':pair_audits,
            'missing_kinds':['mixed_class_actor native heterogeneous capabilities','live_or_playback native loudspeaker device']})
        for member in MEMBERS:
            c=contexts[member,wi];wid=c['folder'].name;bundle={'source_registry':registry,'scene':scenes[member],'queries':records[member],
                'pairing_id':row['scene_id']+f'__w{wi:02d}','base_world_id':row['scene_id']}
            pub,priv,audit=collect_content_binding(bundle,wid,read_json(c['folder']/'questions.json')[0]['media'],seed=str(row['scene_seed']))
            alternate=deepcopy(bundle)
            for record in alternate['queries']:record['wording_variant']='alternate'
            ap,ag,aa=collect_content_binding(alternate,wid,pub[0]['media'] if pub else read_json(c['folder']/'questions.json')[0]['media'],seed=str(row['scene_seed']))
            signature=lambda qs,gs:([(q['question_id'],q['qa_id'],q['form'],q.get('choices'),q['answer_type']) for q in qs],[(g['question_id'],g['canonical_answer'],g['evidence']['answer_domain']) for g in gs])
            invariant=signature(pub,priv)==signature(ap,ag)
            if not invariant:raise ValueError('actual content wording exchange changes IDs, answers, domains or options')
            audit['wording_exchange']={'status':'pass','standard_logical':len(pub)//2,'alternate_logical':len(ap)//2,'facts_identical':True}
            counts.update(audit.get('logical_counts_by_kind',{}));deferrals.extend(audit['deferred']);pools[member].append((wi,pub,priv,audit,bundle))
    output=[]
    for member in MEMBERS:
        m=by_member[member];source=Path(m['windows_directory'])
        def publish(attempt):
            required=[];windows=[];extras={wi:(p,g,a,b) for wi,p,g,a,b in pools[member]}
            for wi,w in enumerate(m['windows']):
                wid=w['window_id'];dest=attempt/wid;dest.mkdir();old=source/wid
                for name in ('video.mp4','audio_foa.wav','audio_mono.wav'):os.link(old/name,dest/name)
                p,g,a,b=extras.get(wi,([],[],{'status':'deferred_no_common_native_anchor'},None))
                q=read_json(old/'questions.json')+p;gold=read_json(old/'private_answers.json')+g;audit=read_json(old/'candidate_audit.json');ev=read_json(old/'evidence.json')
                audit['content_binding']=a;ev['actual_content_binding_evidence']=b
                for name,value in [('questions.json',q),('private_answers.json',gold),('candidate_audit.json',audit),('evidence.json',ev)]:write_json(dest/name,value)
                required.extend(f'{wid}/{name}' for name in ('video.mp4','audio_foa.wav','audio_mono.wav','questions.json','private_answers.json','candidate_audit.json','evidence.json'));windows.append({'window_id':wid,'logical_candidates':len({r['logical_question_id'] for r in q})})
            return {'windows':windows},required
        dest,result,resumed=stage(root,'content_pools_'+member,{'revision':CONTENT_BRIDGE_REVISION,'source_pool':str(source)},publish)
        output.append({**m,'windows_directory':str(dest),'windows':result['windows'],'content_pool_resumed':resumed})
    audit={'status':'pass' if counts else 'deferred','logical_candidate_counts':dict(counts),'deferrals':deferrals,'formal_selected_count':'determined by unchanged group selector after this stage'}
    retain_readback(proofroot/'summary.json',audit)
    return output,audit
