"""Patch final registries with explicit mouth visibility and licensed pet-bird receipts."""
import argparse,copy,json,datetime
from pathlib import Path
from v7_human_roster_materials import json_write
from v7_roster_disk import check
ALLOWED_MODEL_LICENSES={'CC0-1.0','CC-BY-4.0','CC-BY-3.0'}
def mouth_visibility(review):
 habitat=review.get('status','not_verified');ue=review.get('ue_status','not_verified')
 occluded=any('occludes'in s for s in [habitat,ue])
 return {'habitat_status':habitat,'ue_status':ue,'uniform_occludes_lips':occluded,'helmet_or_lighting_limits_ue':'helmet_dark'in ue,'visible_lips_binding_allowed_habitat':habitat.startswith('pass_'),'visible_lips_binding_allowed_ue':ue.startswith('pass_'),'voice_emission_allowed':True,'physical_anchor_relocated':False,'policy':'Keep physical lip point following MUpperLip; require visible_lips_binding_allowed for lip-visible questions. Uniform occlusion is explicitly marked, never relabel a helmet surface as lips.','source_evidence':[review.get('evidence'),review.get('ue_evidence')]}
def pet_admitted(meta,habitat,ue,geometry):
 return meta.get('source_license')in ALLOWED_MODEL_LICENSES and meta.get('movement')=='static'and len(habitat)==2 and {r.get('house')for r in habitat}=={'mp3d','hm3d'}and all(r.get('status')=='pass'for r in habitat)and ue.get('qa_status')=='pass'and geometry.get('status')=='pass'
def run(T):
 check(T,4*1024**2);R=T/'round4';human=json.loads((T/'registrations/human_roster.json').read_text());roster=json.loads((T/'registrations/v7_asset_roster.json').read_text());review=json.loads((T/'evidence/mouth_visual_review.json').read_text());visibility={}
 for major in human['majors']:
  aid=major['major_id'];vis=mouth_visibility(review.get(aid,{}));visibility[aid]=vis;major['mouth_visibility']=vis
  for variant in major['minor_variants']:
   variant['mouth_visibility']=vis;variant['mouth']['visibility']=vis;variant['visible_lips_binding_allowed']={'habitat':vis['visible_lips_binding_allowed_habitat'],'ue':vis['visible_lips_binding_allowed_ue']}
 human['summary']['uniform_occluded_lips_major']=sum(v['uniform_occludes_lips']for v in visibility.values());human['summary']['ue_helmet_dark_lips_major']=sum(v['helmet_or_lighting_limits_ue']for v in visibility.values());human['followup_round4_evidence']='round4/evidence/mouth_visibility.json';json_write(R/'evidence/mouth_visibility.json',visibility)
 meta=json.loads((R/'assets/parrot/source.json').read_text());h=json.loads((R/'evidence/habitat_pet_parrot_results.json').read_text());ue=json.loads((R/'evidence/ue_parrot_qa.json').read_text());geometry=json.loads((R/'evidence/parrot_source_geometry.json').read_text());usable=pet_admitted(meta,h,ue,geometry);audio=json.loads((R/'evidence/animal_audio_binding.json').read_text())
 bird=next(g for g in roster['animals']if g['major_id']=='bird');retired=roster.setdefault('retired_candidates',[])
 for old in list(bird['minor_variants']):
  if old['asset_id']=='v7_pet_budgerigar_candidate_01':
   old['reason']='retired by explicit owner round4 instruction: existing licensed model route replaces generated 2D approval route';retired.append(old);bird['minor_variants'].remove(old)
  elif 'sparrow'in old['asset_id']:old['name_zh']='麻雀（野鸟）';old['pet_bird_requested_goal']='wild_bird_fallback_only'
 bird['minor_variants']=[v for v in bird['minor_variants']if v['asset_id']!=meta['asset_id']]
 import_receipt=json.loads((R/'evidence/ue_parrot_import.json').read_text())
 entry={'asset_id':meta['asset_id'],'name_zh':meta['name_zh'],'movement':['static'],'walk_allowed':False,'dual_backend_static_usable':bool(usable),'habitat_glb':'round4/assets/parrot/runtime.glb','ue_assets':import_receipt['assets'],'ue_blueprint_path':import_receipt['blueprint_path'],'ue_generated_class_path':import_receipt['generated_class_path'],'logical_actor_count':1,'ue_blueprint_actor_yaw_degrees':0,'ue_static_component_yaw_degrees':90,'height_bird_m':meta['height_bird_m'],'total_height_m':meta['total_height_m'],'source_license':meta['source_license'],'source_author':meta['source_author'],'source_url':meta['source_url'],'source_metadata':meta,'beak_local_glb_m':meta['beak_local_glb_m'],'beak_local_ue_blueprint_cm':[-100*meta['beak_local_glb_m'][2],100*meta['beak_local_glb_m'][0],100*meta['beak_local_glb_m'][1]],'beak_visual_status':'pass native semantic neighbourhood and UE component/readback; weighted beak surface point','perch_produces_sound':False,'audio_admission_owner':'v7-families','sound_class':'pet_bird_parrot','dry_recording_binding':'round4/evidence/animal_audio_binding.json#/classes/pet_bird_parrot','admitted_recorded_sound_count':audio['classes']['pet_bird_parrot']['count'],'source_clip_policy':'source rig and 15 clips retained in original; final frozen idle pose only; no walk/flight admission','receipts':{'habitat':'round4/evidence/habitat_pet_parrot_results.json','ue':'round4/evidence/ue_parrot_qa.json','source_geometry':'round4/evidence/parrot_source_geometry.json','license':'round4/evidence/parrot_provenance.json'}}
 entry['source_metadata']=dict(meta,habitat_status='pass_native_reference_houses'if usable else'not_admitted',ue_status='pass_native_reference_views'if usable else'not_admitted')
 bird['minor_variants'].append(entry)
 for group in roster['animals']:
  if group['major_id']=='dog':
   for v in group['minor_variants']:
    v['admitted_motion_clips']={'walk':['Walking_Contact_'+h for h in ['mp3d','hm3d','kujiale']],'stand':['Idle_Contact_'+h for h in ['mp3d','hm3d','kujiale']]};v['unadmitted_source_motion_loops']=['Idle','Walking'];v['static_pose_audit']='round4/evidence/static_dog_pose_audit.json'
    v['sound_class']='dog_bark';v['dry_recording_binding']='round4/evidence/animal_audio_binding.json#/classes/dog_bark';v['admitted_recorded_sound_count']=audio['classes']['dog_bark']['count'];v['recording_breed_not_verified']=True
 roster['humans']=human;roster['summary'].update(human['summary']);roster['summary']['pet_bird_dual_backend_usable']=int(usable);roster['summary']['wild_bird_fallback_dual_backend_usable']=sum(v.get('dual_backend_static_usable',False)for v in bird['minor_variants']if 'sparrow'in v['asset_id']);roster['generated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();roster['round4_scope']='Existing CC-BY pet parrot static/native intake, mouth visibility flags and readonly real-recording pointers; generator/spatial episodes not run.'
 json_write(T/'registrations/human_roster.json',human);json_write(T/'registrations/v7_asset_roster.json',roster);json_write(R/'registrations/animal_roster.json',roster['animals'])
 print('ROUND4_ROSTER',roster['summary']);return roster
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
