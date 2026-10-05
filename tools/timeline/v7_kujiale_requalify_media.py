"""Recompile the original QA and seal against unchanged, SHA-verified UE media.

Old questions and answers are never imported into the fresh QA root. The
frozen generator's inner seal function is compiled verbatim after an AST
comparison with 33b5b12, so its validation and delivery code remain identical.
"""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import ast,json,os,time,tarfile,hashlib,subprocess,traceback,shutil
ROOT=Path('/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003');RUN=ROOT/'ue_round14';REPO=Path('/data/jzy/code/AVEngine-codex-v7-kujiale-round14-20261005')
FROZEN='33b5b12df1a45ca265bf0c4fbe9b26b664577a7c'

def atomic(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def original_seal(namespace):
 from avengine.timeline import v7_registered_generation as gen
 current=(REPO/'src/avengine/timeline/v7_registered_generation.py').read_text()
 frozen=subprocess.check_output(['git','show',FROZEN+':src/avengine/timeline/v7_registered_generation.py'],cwd=REPO,text=True)
 def extract(source):
  tree=ast.parse(source);parent=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_registered_parent')
  return next(n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name=='seal')
 node=extract(current);assert ast.dump(node,include_attributes=False)==ast.dump(extract(frozen),include_attributes=False),'frozen seal code differs'
 globals_=dict(vars(gen));globals_.update(namespace);module=ast.Module(body=[node],type_ignores=[]);exec(compile(module,str(REPO/'src/avengine/timeline/v7_registered_generation.py'),'exec'),globals_)
 return globals_['seal']

def main():
 from avengine.timeline.v7_candidate_generation import stage,audit_parent_group
 from avengine.timeline.v7_registered_generation import qualify_licenses,PRODUCER_QA_REVISION,_cpu
 from avengine.timeline.v7_registered_sampling import choose_assets
 from avengine.timeline.v7_full_question_delivery import collect_native_pool,finalize_parent_question_pools
 from avengine.timeline.v7_storage_budget import configure_kujiale_production
 from avengine.timeline.v7_alpha_delivery import export_alpha
 from avengine.timeline.v7_s_batch_checks import check_critical_delivery,open_delivered_media
 from avengine.capture.v7_alpha_ue import assert_ue_alpha_visual
 archive=RUN/'evidence/CCI_requalification_source_direct.tar.gz';digest=hashlib.sha256(archive.read_bytes()).hexdigest();assert digest=='644e306c5fad68ff73623fd6ea285e9115b7b0323007faa94988877950db2838'
 phase=RUN/'canary_endpoint_fixed_v4';phase.mkdir(exist_ok=False);recovered=phase/'immutable_CCI_sources';recovered.mkdir();output=phase/'production';output.mkdir()
 with tarfile.open(archive) as tar:
  for m in tar.getmembers():
   assert (recovered/m.name).resolve().is_relative_to(recovered.resolve()) and (m.isdir() or m.isfile())
  tar.extractall(recovered)
 for release in recovered.iterdir():
  transfer=json.loads((release/'TRANSFER_SHA256.json').read_text())
  for item in transfer['files']:
   f=release/item['relative'];assert f.resolve().is_relative_to(release.resolve()) and f.stat().st_size==item['bytes'];assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'],str(f)
 configure_kujiale_production(phase,RUN,workers=1)
 houses={h['house']:h for h in json.loads((RUN/'inputs/houses.json').read_text())['houses']};runtime=json.loads((RUN/'inputs/runtime.json').read_text());catalog=json.loads((RUN/'inputs/assets.json').read_text())['assets'];human_admission=json.loads((RUN/'inputs/human_admission.json').read_text());allrows={r['scene_id']:r for r in map(json.loads,(RUN/'inputs/kujiale_v7_S_train.jsonl').read_text().splitlines())};completed=[];results=[];successful_timings=[];started=time.time()
 for release in sorted(recovered.iterdir()):
  for predecessor in sorted((release/'parents').iterdir()):
   timer=time.time();cpu_start=_cpu();old=json.loads((predecessor/'delivery_manifest.json').read_text());pid=predecessor.name;row=allrows[pid];assert row['members']==['v0_a0']
   root=output/'parents'/pid;root.mkdir(parents=True);visual_dirs={'v0':predecessor/'private/native/v0'};visual=json.loads((visual_dirs['v0']/'native_registered_visual_evidence.json').read_text());appearance=json.loads((visual_dirs['v0']/'native_appearance_review.json').read_text());visual['native_appearance_review']=appearance['review'];assert_ue_alpha_visual(visual)
   assert appearance['status']=='pass' and len(appearance['review']['actors'])>=2 and all(x['status']=='pass' and x['known_bad_detected'] for x in visual['standing_admission'])
   cache=predecessor/'private/rir_cache';
   if cache.is_dir():shutil.copytree(cache,root/'rir_cache')
   assets=choose_assets(catalog,row);working=deepcopy(houses[row['house']]);working['room']=deepcopy(next(r for r in working['checked_rooms'] if r['room_label']==row['room_id']));assert set(a['asset_id'] for a in assets)==set(a['asset_id'] for a in visual['frames'][0]['actors']);licenses=qualify_licenses(assets,human_admission)
   for a in assets:
    assert any(d['asset_id']==a['asset_id'] for d in visual['standing_admission'])
   source_windows=old['windows'];base=json.loads((predecessor/'private'/(source_windows[0]['window_id']+'_evidence.json')).read_text())['scene'];base['window_starts_s']=[x['start_s'] for x in source_windows]
   recipe={'profile':old['profile'],'assets':assets,'house':working,'runtime_inputs':runtime,'producer_qa_revision':PRODUCER_QA_REVISION,'seed_row':row,'immutable_CCI_media_archive_sha256':digest,'original_parent':str(predecessor),'native_media_regenerated':False,'old_QA_and_answers_imported':False};atomic(root/'input_recipe.json',recipe)
   try:
    def compile_windows(attempt):
     required=[];windows=[]
     for wi,record in enumerate(source_windows):
      wid=record['window_id'];src=predecessor/'windows'/wid;dest=attempt/wid;dest.mkdir()
      for file in ('video.mp4','audio_foa.wav','audio_mono.wav'):os.link(src/file,dest/file)
      evidence=json.loads((predecessor/'private'/(wid+'_evidence.json')).read_text());assert evidence['seed_row']==row
      proof=deepcopy(evidence['source_qa_evidence']);old_foa_path=proof['foa_readback']['path'];assert Path(old_foa_path).name=='audio_foa.wav';proof['foa_readback']['path']=str(dest/'audio_foa.wav');proof['foa_readback']['original_media_reference']=old_foa_path;media={key:str(dest/name) for key,name in [('video','video.mp4'),('audio_foa','audio_foa.wav'),('audio_mono','audio_mono.wav')]};quota=row['question_quota_base']+wi
      q,g,audit=collect_native_pool(evidence['scene'],record['start_s'],wid,visual,media,proof,quota,row['questions_per_window'])
      atomic(dest/'questions.json',q);atomic(dest/'private_answers.json',g);atomic(dest/'candidate_audit.json',audit)
      evidence['source_qa_evidence']=proof;evidence.pop('alpha_admission',None);evidence.pop('full_qa_selection',None);evidence.update(alpha_selection='pending',native_visual_evidence=str(visual_dirs['v0']/'native_registered_visual_evidence.json'),producer_qa_revision=PRODUCER_QA_REVISION,immutable_native_reuse={'archive_sha256':digest,'source_parent':str(predecessor),'old_answers_reused':False})
      atomic(dest/'evidence.json',evidence);windows.append({'window_id':wid,'logical_candidates':len({x['logical_question_id'] for x in q})});required.extend(f'{wid}/{f}' for f in ('video.mp4','audio_foa.wav','audio_mono.wav','questions.json','private_answers.json','candidate_audit.json','evidence.json'))
     return {'windows':windows,'native_media_regenerated':False},required
    directory,receipt_,_=stage(root,'windows_v0_a0',recipe,compile_windows)
    members=[{'member':'v0_a0','windows_directory':str(directory),'windows':receipt_['windows'],'visual_directory':str(visual_dirs['v0']),'audio_directory':str(predecessor/'private/native')}]
    members,selection=finalize_parent_question_pools(row,members,root,binding_items_per_window=5)
    group=audit_parent_group(row,base,members);atomic(root/'counterfactual_alpha_audit.json',group)
    audio=root/'audio_native_reuse';audio.mkdir()
    for f in ('audio_evidence.json','native_RLR_setup.json'):os.link(predecessor/'private/native'/('a0_'+f),audio/f)
    ar=json.loads((audio/'audio_evidence.json').read_text());program=json.loads((predecessor/'private/dry_program.json').read_text());receipt=json.loads((predecessor/'private/dry_emission_receipt.json').read_text())
    namespace=dict(root=root,licenses=licenses,program=program,receipt=receipt,visual_names=['v0'],visual_dirs=visual_dirs,audio_factors={'a0':(audio,ar)},row=row,base=base,recipe=recipe,content_audit={'status':'not_applicable_no_authored_quartet'},elapsed=time.time()-timer,cpu=_cpu()-cpu_start,timings=[{'stage':'native_media_reuse','resumed':True,'regenerated':False,'original_native_wall_seconds':old['wall_seconds_before_seal']}],members=members,group=group)
    seal=original_seal(namespace);delivery,result,_=stage(root,'alpha_seal',{'input_recipe':recipe},seal)
    successful_timings.append({**result,'delivery_directory':str(delivery),'wall_seconds':time.time()-timer,'cpu_seconds_including_children':_cpu()-cpu_start,'stage_timings':[{'stage':'visual_v0','resumed':True,'wall_seconds':0.,'cpu_seconds_including_children':0.,'native_media_regenerated':False,'original_native_wall_seconds':old['wall_seconds_before_seal']}],'native_media_regenerated':False});results.append({'scene_id':pid,'status':result['status'],'formal_windows':result['alpha_formal_windows'],'formal_questions':result['alpha_formal_logical_questions'],'QA_requalification_wall_seconds':time.time()-timer,'original_native_wall_seconds':old['wall_seconds_before_seal'],'old_answers_modified':False});completed.append(row)
   except Exception as e:
    result={'scene_id':pid,'status':'failed_original_gate' if isinstance(e,ValueError) else 'requalification_interface_error','error_type':type(e).__name__,'reason':str(e),'traceback':traceback.format_exc(),'wall_seconds':time.time()-timer};atomic(root/'requalification_failed.json',result);results.append(result)
   print(json.dumps(results[-1],ensure_ascii=False),flush=True)
 seeds=phase/'passed_seeds.jsonl';seeds.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in completed))
 if completed:
  atomic(output/'shard_requalified_media_v1.json',{'parents':successful_timings,'new_native_GPU_render':False})
  summary=export_alpha(seeds,[output],phase/'release',expected_parents=len(completed));coverage=check_critical_delivery(phase/'release',require_full=False);media=open_delivered_media(phase/'release');assert coverage['status']==media['status']=='pass'
  atomic(phase/'release/DELIVERY_MANIFEST.json',{**summary,'status':'sealed','generator_commit':FROZEN,'producer_qa_revision':PRODUCER_QA_REVISION,'native_backend_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'render_path':'ue_spear','coverage':coverage,'media_readback':media,'supersedes_withdrawn_batches':['batch_000','batch_001_passed'],'old_QA_modified':False,'native_media_reused_without_regeneration':True})
  if completed:
   oldcanary=json.loads((RUN/'evidence/canary_result.json').read_text());atomic(RUN/'evidence/canary_result.before_endpoint_fix.json',oldcanary)
   atomic(RUN/'evidence/canary_result.json',{'status':'pass','formal_windows':summary['formal_windows'],'formal_questions':summary['formal_logical_questions'],'wall_seconds':next(x['original_native_wall_seconds'] for x in results if x['status']=='formal_alpha_admitted'),'QA_requalification_wall_seconds':time.time()-started,'peak_mib':oldcanary['peak_mib'],'release':str(phase/'release'),'producer_qa_revision':PRODUCER_QA_REVISION,'native_media_reused':True,'parent_ids':[x['scene_id'] for x in completed]})
 out={'status':'completed','parents':results,'phase':str(phase),'formal_windows':sum(x.get('formal_windows',0) for x in results),'formal_questions':sum(x.get('formal_questions',0) for x in results),'wall_seconds':time.time()-started,'new_native_GPU_render':False};atomic(RUN/'evidence/sealed_media_requalification.json',out)
 with (ROOT/'PROGRESS_zh.md').open('a') as f:f.write('\n第14回合历史媒体重新准入：CCI封存源逐文件SHA复核，以原RGB/FOA/原生读回重新生成题池、原选题及原封存函数。旧题/旧答案未导入；当前 '+str(out['formal_windows'])+'窗口/'+str(out['formal_questions'])+'题，结果evidence/sealed_media_requalification.json；新GPU渲染为0，不冒称渲染机时。\n')
 print(json.dumps(out,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
