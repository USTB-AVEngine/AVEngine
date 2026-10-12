"""Balance two fresh voice variants for every authored dialogue line."""
from _dialogue_voice_runtime import *
from copy import deepcopy
from collections import Counter,defaultdict
from select_dialogue_voice_references import speaker_info
def main():
    refs=json.loads((STAGE/'references.json').read_text());assert refs['status']=='pass';source=json.loads((R/'natural_batch_v1/plan.json').read_text());old=json.loads((R/'natural_batch_ready/semantic_manifest.json').read_text())['sounds'];info=speaker_info()
    selected=refs['selected'];counts=Counter();used=defaultdict(set);tasks=[]
    # Every scenario uses each new speaker for at most one role/variant; old speakers are disjoint.
    for i,t in enumerate(source['tasks']):
     for variant in range(2):
      candidates=[r for r in selected[t['voice_gender']] if r['speaker_id'] not in used[t['scenario_id']]]
      assert candidates,('no distinct speaker remains',t['scenario_id'])
      candidates.sort(key=lambda r:(counts[r['speaker_id']],hashlib.sha256((t['sound_asset_id']+str(variant)+r['speaker_id']).encode()).hexdigest()))
      ref=candidates[0];speaker=ref['speaker_id'];counts[speaker]+=1;used[t['scenario_id']].add(speaker)
      new=deepcopy(t);new.update(authored_line_id=t['sound_asset_id'],sound_asset_id=t['sound_asset_id']+'__voice_'+speaker,voice_reference=ref,expansion_kind='new_voice',voice_variant=variant+2);new['synthesis_config'].update(seed=31000000+i*100+variant,model_license='Apache-2.0',model_weight_license_receipt=str(ROOT/'license_receipt.json'));tasks.append(new)
    # Five male/five female pilot lines, distinct new speakers, spanning all dialogue categories.
    pilot=[];seen=set();gcounts=Counter()
    for t in sorted(tasks,key=lambda t:hashlib.sha256(t['sound_asset_id'].encode()).hexdigest()):
     g=t['voice_gender'];sp=t['voice_reference']['speaker_id']
     if gcounts[g]<5 and sp not in seen:pilot.append(t['sound_asset_id']);seen.add(sp);gcounts[g]+=1
     if len(pilot)==10:break
    sid='daily_reminder_home_living_room__role_2';repair=deepcopy(next(t for t in source['tasks'] if t['sound_asset_id']==sid));repair.update(authored_line_id=sid,expansion_kind='old_language_repair',voice_variant=1);ref=repair['voice_reference'];ref.update(info[ref['speaker_id']]);ref['source_waveform_sha256']=digest(ref['source_waveform']);ref['source_transcript_sha256']=digest(ref['source_transcript_path']);repair['synthesis_config'].update(seed=51000000,model_license='Apache-2.0',model_weight_license_receipt=str(ROOT/'license_receipt.json'));repair['replaces_original_audio']={'path':next(s['path'] for s in old if s['sound_asset_id']==sid),'sha256':next(s['sha256'] for s in old if s['sound_asset_id']==sid)}
    for t in tasks+[repair]:assert t['voice_gender']==t['standing_actor_plan']['visual_gender']==t['voice_reference']['gender']
    plan={**deepcopy(source),'schema':'v7_dialogue_voice_expansion_plan_v2','tasks':tasks,'repair_task':repair,'pilot_ids':pilot,'planned_new_lines':len(tasks),'new_voice_counts':dict(counts),'max_attempts':3,'source_plan':str(R/'natural_batch_v1/plan.json'),'source_plan_sha256':digest(R/'natural_batch_v1/plan.json'),'language_gate':json.loads((STAGE/'config.json').read_text())['language_rule']}
    assert len(tasks)==1080 and len(counts)==16 and min(counts.values())>=40 and len(pilot)==10
    save(STAGE/'expansion_plan.json',plan);progress('扩音色计划已固定：540句×2，共1080条，16个新音色，计划每个音色'+str(min(counts.values()))+'–'+str(max(counts.values()))+'句；性别及情景内去重预检全部通过。');print('PLAN',len(tasks),dict(counts),'PILOT',pilot)

if __name__ == '__main__':
    main()
