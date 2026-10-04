#!/usr/bin/env python3
"""Resume task-isolated native import/capture/QA without changing frozen rules."""
from __future__ import annotations
import argparse,csv,datetime,json,os,subprocess,time
from copy import deepcopy
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
 o=a.root.resolve();u=o/'ue_v1';r=o/'ue_round6';repo=Path(__file__).resolve().parents[2];py=os.sys.executable
 env=dict(os.environ,PYTHONPATH=f'{repo}/src:{repo}:{u}/python_pkgs',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',TMPDIR=str(u/'tmp'))
 houses=[x['house'] for x in read(o/'acoustic_pass_train_candidates.json')['houses']]
 if len(houses)!=21:raise ValueError(f'expected 21 frozen acoustic passes, got {len(houses)}')
 def progress(title,body):
  with (o/'PROGRESS_zh.md').open('a') as f:f.write('\n\n## '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+title+'\n\n'+body+'\n')
 def execute(argv,log):
  with log.open('a') as f:subprocess.run(argv,env=env,cwd=repo,stdout=f,stderr=subprocess.STDOUT,check=True)
 def guard(q,house,kind):
  q=deepcopy(q);q.update(gpu_index=a.gpu,previous_peak_mib=3876,storage_limit_bytes=80000000000,minimum_disk_free_bytes=150000000000,cpu_affinity=list(range(16,24)),run_directory=str(r/f'logs/{house}_{kind}'),maximum_seconds=7200,allowed_auxiliary_gpu_indices=[i for i in range(4) if i!=a.gpu],maximum_auxiliary_gpu_mib=64)
  existing=Path(q['run_directory'])/'result.json'
  if existing.exists() and read(existing)['status']=='pass':return read(existing)
  if Path(q['run_directory']).exists():q['run_directory']+='_'+str(time.time_ns())
  q['command']=[('-graphicsadapter='+str(a.gpu)) if s.startswith('-graphicsadapter=') else s for s in q['command']]
  qpath=r/f'scripts/{house}_{kind}_guard.json';write(qpath,q)
  execute([py,str(repo/'tools/rooms/v7_kujiale_ue_guard.py'),'--request',str(qpath)],r/f'logs/{house}_{kind}_coordinator.log')
  return read(Path(q['run_directory'])/'result.json')
 selected=[h for h in houses if not (u/f'houses/{h}/acceptance_plan_v6.json').exists()]
 if selected:
  execute([py,str(repo/'tools/rooms/prepare_v7_kujiale_ue.py'),'--output-root',str(o),'--thresholds',str(o.parent/'room_selection_merged_20261003/thresholds.frozen.yaml'),'--baseline-manifest',str(o.parent/'v7_batch1_20261003/acceptance_manifest_v2.json'),'--houses',*selected,'--version','6','--witness-count','11'],r/'logs/prepare_21_plans.log')
 progress('21套原生自动验收请求准备完成','- 保留冻结房间/门槛，最多11个同规则摆位候选、需要3个通过视角；本轮全部21套/44候选房请求已准备。\n- 输出ue_round6/houses，原生地图和资产另存NativeV6/ImportedNativeV6，原始数据不动。')
 results=[]
 for house in houses:
  d=r/'houses'/house;d.mkdir(parents=True,exist_ok=True)
  if (d/'result.json').exists():
   old_result=read(d/'result.json')
   if old_result['status']=='completed':results.append(old_result);continue
   write(d/f"attempt_failed_{time.time_ns()}.json",old_result)
  began=time.monotonic();receipt={'house':house,'status':'started','rooms':[],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  try:
   plan_path=u/f'houses/{house}/acceptance_plan_v6.json';plan=read(plan_path)
   # The two previously measured houses retain their established, identical
   # candidate selection; only the native map path is versioned anew.
   previous=u/f'houses/{house}/acceptance_plan_v5.json'
   if house in ('kujiale_0022','kujiale_0030') and previous.exists():
    old=read(previous);old['map_path']=plan['map_path'];plan=old
   plan_path=d/'acceptance_plan.json';write(plan_path,plan)
   request=read(u/f'houses/{house}/import_request_v6.json')
   baked=d/'source_world_baked.usdc'
   if not (d/'bake_receipt.json').exists():execute([py,str(repo/'tools/ue/bake_kujiale_native_usd.py'),'--source',request['source_usd'],'--output',str(baked),'--receipt',str(d/'bake_receipt.json')],d/'bake.log')
   request.update(source_usd=str(baked),acceptance_plan_path=str(plan_path),result_path=str(d/'import_result.json'),native_editor_audit_path=str(d/'native_editor_audit.json'),native_geometry_prefix=str(d/'native_geometry'),world_obj_path=str(d/'unused.obj'),correct_native_usd_y_reflection=True,audit_only=False)
   request.pop('expected_bounds_ue_cm',None);request_path=d/'import_request.json';write(request_path,request)
   imp=read(u/'scripts/e2e_0030_import_guard_request_v1.json');imp['environment']['AVENGINE_USD_LEVEL_IMPORT_REQUEST']=str(request_path)
   imp['command']=[('-abslog='+str(d/'import_editor.log')) if s.startswith('-abslog=') else s for s in imp['command']];imp['reserve_bytes']=1500000000
   receipt['import']=guard(imp,house,'import')
   cfg=read(u/'scripts/ue_runtime_production_0030_v5.json');cfg.update(graphics_adapter=a.gpu,rpc_port=0,acceptance_plan_path=str(plan_path));cfg_path=d/'ue_config.json';write(cfg_path,cfg)
   capture_path=d/'capture'
   if capture_path.exists() and not (capture_path/'result.json').exists():capture_path=d/('capture_retry_'+str(time.time_ns()))
   cap=read(u/'scripts/capture_0030_final_profile_guard_request_v2.json');cap['command']=[py,str(repo/'tools/rooms/check_v7_kujiale_ue.py'),'--plan',str(plan_path),'--editor-audit',str(d/'native_editor_audit.json'),'--ue-config',str(cfg_path),'--output',str(capture_path)];cap['reserve_bytes']=300000000
   receipt['capture']=guard(cap,house,'capture')
   started=time.monotonic()
   execute(['taskset','-c','16-23',py,str(repo/'tools/rooms/finalize_v7_kujiale_ue.py'),'--plan',str(plan_path),'--editor-audit',str(d/'native_editor_audit.json'),'--capture',str(capture_path),'--output',str(d/'qa')],d/'finalize.log')
   receipt['capture_directory']=str(capture_path);receipt['qa_directory']=str(d/'qa')
   receipt['cpu_qa_seconds']=time.monotonic()-started
   qa=read(d/'qa/result.json');receipt['rooms']=[{'room_id':x['room_id'],'status':x['automated_room_status'],'failures':x['failures']} for x in qa['rooms']]
   receipt.update(status='completed',passed_rooms=qa['room_geometry_visual_floor_pass_count'])
  except Exception as error:
   receipt.update(status='failed',error=f'{type(error).__name__}: {error}')
  receipt['machine_seconds']=time.monotonic()-began;write(d/'result.json',receipt);results.append(receipt)
  write(r/'room_batch_live.json',{'records':results,'expected_houses':21,'expected_rooms':44})
  progress(house+' 原生房间检查阶段结束',f"- 状态 {receipt['status']}，机时{receipt['machine_seconds']:.3f}秒，房间结果{json.dumps(receipt['rooms'],ensure_ascii=False)}。来源ue_round6/houses/{house}/result.json；未执行人工放行。")
  print(json.dumps(receipt,ensure_ascii=False),flush=True)
 write(r/'room_batch_result.json',{'status':'completed' if all(x['status']=='completed' for x in results) else 'completed_with_failures','records':results,'expected_houses':21,'expected_rooms':44})

if __name__=='__main__':main()
