"""Actual audible inventory coverage; a planned or silent asset never counts."""
from collections import Counter
from pathlib import Path
import json

REVISION='actual-delivered-window-source-inventory-v1'


def make_inventory(admission,semantic,catalog):
    recordings={}
    for r in admission['records']:
        recordings[r['sound_asset_id']]={'sound_class':r['sound_class'],'source_type':r.get('source_type','recorded'),
            'gender':r.get('gender'),'duration_s':r.get('duration_s')}
    for r in semantic['sounds']:
        recordings[r['sound_asset_id']]={'sound_class':r.get('sound_class','speech_playback'),'source_type':'author_tts',
            'gender':r.get('voice_gender',r.get('voice_reference',{}).get('gender')),'duration_s':r['synthesis_readback']['duration_s'],
            'voice_id':r['voice_reference']['speaker_id']}
    entities={a['asset_id']:{k:a.get(k)for k in ['kind','species_id','visual_gender','allowed_classes']}for a in catalog if a.get('allowed_classes')}
    return {'revision':REVISION,'recordings':recordings,'sounding_entities':entities,
        'registered_source_classes':sorted({r['sound_class']for r in recordings.values()}),
        'non_sounding_entity_ids':[a['asset_id']for a in catalog if not a.get('allowed_classes')]}


def collect(inventory,receipts,rows):
    selected={r['window_id']for r in rows};recordings=Counter();entities=Counter();classes=Counter();actual_classes=Counter();voices=Counter();placed=Counter();unknown=Counter();seen=set()
    for receipt in receipts:
        pointer_path=Path(receipt['export']['source_seal']);pointer=json.loads(pointer_path.read_text());parent=pointer_path.parent
        for win in pointer['result']['windows']:
            wid=win['window_id']
            if wid not in selected:continue
            member=win.get('member')or wid.rsplit('__',2)[-2];marker=json.loads((parent/f'selected_windows_{member}_completed.json').read_text());pool=Path(marker['recipe']['pool_directory'])/wid
            facts=json.loads((pool/'candidate_audit.json').read_text())['facts'];actors=facts['actors']
            for actor in actors.values():placed[actor['asset_id']]+=1
            for event in facts['events']:
                key=(receipt['scene_id'],member,event['event_id'])
                if key in seen:continue
                seen.add(key);record=event['sound_asset_id'];actor=actors.get(str(event['actor_id']),actors.get(event['actor_id'],{}));aid=actor.get('asset_id')
                if aid:entities[aid]+=1
                actual_classes[event.get('sound_class','unknown')]+=1
                if record in inventory['recordings']:
                    recordings[record]+=1;entry=inventory['recordings'][record];classes[entry['sound_class']]+=1
                    if entry.get('voice_id'):voices[entry['voice_id']]+=1
                else:unknown[record]+=1
    unused_records=sorted(set(inventory['recordings'])-recordings.keys());unused_entities=sorted(set(inventory['sounding_entities'])-entities.keys());unused_classes=sorted(set(inventory['registered_source_classes'])-classes.keys())
    return {'revision':REVISION,'counting_scope':'actual audible events in model-delivered windows; overlapping copies of the same parent/member/event counted once',
        'recording_counts':dict(recordings),'sounding_entity_counts':dict(entities),'placed_entity_counts':dict(placed),
        'registered_class_counts':dict(classes),'actual_event_class_counts':dict(actual_classes),'authored_voice_counts':dict(voices),
        'registered_recordings':len(inventory['recordings']),'used_recordings':len(set(recordings)&set(inventory['recordings'])),
        'registered_sounding_entities':len(inventory['sounding_entities']),'used_sounding_entities':len(set(entities)&set(inventory['sounding_entities'])),
        'registered_classes':len(inventory['registered_source_classes']),'used_registered_classes':len(set(classes)&set(inventory['registered_source_classes'])),
        'unused_recording_ids':unused_records,'unused_sounding_entity_ids':unused_entities,'unused_classes':unused_classes,
        'unmapped_native_sound_ids':dict(unknown),'all_sources_used_at_least_once':not(unused_records or unused_entities or unused_classes)}


def next_joint_requests(inventory,usage,limit=16):
    """Deterministic coverage debts; constructors must bind, then native gates prove."""
    byclass={}
    rc=usage.get('recording_counts',{});ec=usage.get('sounding_entity_counts',{});cc=usage.get('registered_class_counts',{})
    for rid,r in inventory['recordings'].items():byclass.setdefault(r['sound_class'],[]).append(rid)
    for ids in byclass.values():ids.sort(key=lambda rid:(rc.get(rid,0),rid))
    requests=[]
    for aid,a in inventory['sounding_entities'].items():
        candidates=[]
        for cls in a.get('allowed_classes',[]):
            for rid in byclass.get(cls,[])[:limit]:
                record=inventory['recordings'][rid];gender=record.get('gender')
                if record.get('source_type')=='author_tts'and a.get('kind')!='human':continue
                if a.get('kind')=='human'and gender and str(gender).lower()[0]!=str(a.get('visual_gender','')).lower()[:1]:continue
                candidates.append((rc.get(rid,0)>0,cc.get(cls,0),rc.get(rid,0),rid,cls))
        if candidates:
            _,_,_,rid,cls=min(candidates);score=(ec.get(aid,0)>0,rc.get(rid,0)>0,ec.get(aid,0),cc.get(cls,0),rc.get(rid,0),aid,rid)
            requests.append((score,{'asset_id':aid,'sound_asset_id':rid,'sound_class':cls,'actual_usage_before_request':{'entity':ec.get(aid,0),'recording':rc.get(rid,0),'class':cc.get(cls,0)},
                'requires_original_room_legality_and_native_admission':True,'planned_is_not_used':True}))
    return [r for _,r in sorted(requests,key=lambda p:p[0])[:limit]]


def for_manifest(manifest,receipts,rows):
    path=manifest.get('source_inventory_path')
    if not path:return None
    inventory=json.loads(Path(path).read_text());result=collect(inventory,receipts,rows)
    result['next_inventory_requests']=next_joint_requests(inventory,result)
    return result
