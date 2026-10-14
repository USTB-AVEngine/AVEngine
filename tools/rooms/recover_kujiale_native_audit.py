#!/usr/bin/env python3
"""Read back an own saved map after interrupted import; rerun native QA."""
import argparse,datetime,json,os,subprocess,time
from pathlib import Path
from copy import deepcopy
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--house',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--output-tag',default='');p.add_argument('--fresh-import',action='store_true');a=p.parse_args()
 if a.output_tag and (not a.output_tag.replace('_','').isalnum()):raise ValueError('output tag must be alphanumeric/underscore')
 if a.fresh_import and not a.output_tag:raise ValueError('fresh import requires a new explicit output tag')
 o=a.root.resolve();u=o/'ue_v1';r=o/'ue_round6';repo=Path(__file__).resolve().parents[2];py=os.sys.executable
 read=lambda p:json.loads(Path(p).read_text())
 def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
 d=r/'recovered_houses'/(a.house+('_'+a.output_tag if a.output_tag else ''));d.mkdir(parents=True,exist_ok=False);old=r/'houses'/a.house
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=f'{repo}/src:{repo}:{u}/python_pkgs',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
 def run(argv,log):
  with log.open('x') as f:subprocess.run(argv,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 def guard(q,label):
  q=deepcopy(q);q.update(gpu_index=a.gpu,allowed_auxiliary_gpu_indices=[i for i in range(4) if i!=a.gpu],maximum_auxiliary_gpu_mib=64,storage_limit_bytes=80000000000,minimum_disk_free_bytes=150000000000,cpu_affinity=list(range(96,104)),run_directory=str(d/(label+'_guard')))
  q['command']=[('-graphicsadapter='+str(a.gpu)) if s.startswith('-graphicsadapter=') else s for s in q['command']]
  request=d/(label+'_guard_request.json');write(request,q);run([py,str(repo/'tools/rooms/v7_kujiale_ue_guard.py'),'--request',str(request)],d/(label+'_coordinator.log'));return read(d/(label+'_guard/result.json'))
 started=time.monotonic();receipt={'house':a.house,'status':'started','recovery':'fresh own import and actual render geometry/pixels remeasured' if a.fresh_import else 'own saved map loaded and actual render geometry/pixels remeasured; initial import return code is not treated as success'}
 try:
  plan=read(old/'acceptance_plan.json')
  if a.fresh_import:plan['map_path']=plan['map_path']+'_'+a.output_tag
  plan_path=d/'acceptance_plan.json';write(plan_path,plan)
  req=read(old/'import_request.json');req.update(audit_only=not a.fresh_import,acceptance_plan_path=str(plan_path),native_editor_audit_path=str(d/'native_editor_audit.json'),native_geometry_prefix=str(d/'native_geometry'));
  if a.fresh_import:req.update(map_path=plan['map_path'],destination_path=req['destination_path']+'_'+a.output_tag,result_path=str(d/'import_result.json'),world_obj_path=str(d/'unused.obj'))
  reqpath=d/'audit_request.json';write(reqpath,req)
  q=read(u/'scripts/e2e_0030_import_guard_request_v1.json');q['environment']['AVENGINE_USD_LEVEL_IMPORT_REQUEST']=str(reqpath);q['command']=[('-abslog='+str(d/'audit_editor.log')) if s.startswith('-abslog=') else s for s in q['command']];receipt['audit']=guard(q,'audit')
  cfg=read(old/'ue_config.json') if (old/'ue_config.json').exists() else read(u/'scripts/ue_runtime_production_0030_v5.json');cfg.update(graphics_adapter=a.gpu,rpc_port=0,acceptance_plan_path=str(plan_path));cfgpath=d/'ue_config.json';write(cfgpath,cfg)
  q=read(u/'scripts/capture_0030_final_profile_guard_request_v2.json');q['command']=[py,str(repo/'tools/rooms/check_v7_kujiale_ue.py'),'--plan',str(plan_path),'--editor-audit',str(d/'native_editor_audit.json'),'--ue-config',str(cfgpath),'--output',str(d/'capture')];receipt['capture']=guard(q,'capture')
  run(['taskset','-c','96-103',py,str(repo/'tools/rooms/finalize_v7_kujiale_ue.py'),'--plan',str(plan_path),'--editor-audit',str(d/'native_editor_audit.json'),'--capture',str(d/'capture'),'--output',str(d/'qa')],d/'finalize.log')
  qa=read(d/'qa/result.json');receipt.update(status='completed',passed_rooms=qa['room_geometry_visual_floor_pass_count'],rooms=[{'room_id':x['room_id'],'status':x['automated_room_status'],'failures':x['failures']} for x in qa['rooms']],qa_directory=str(d/'qa'),capture_directory=str(d/'capture'))
 except Exception as error:
  receipt.update(status='failed',error=f'{type(error).__name__}: {error}');raise
 finally:
  receipt['machine_seconds']=time.monotonic()-started;write(d/'result.json',receipt)
  with (o/'PROGRESS_zh.md').open('a') as f:f.write('\n\n## '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+a.house+' 中断导入读回恢复结束\n\n- '+json.dumps({k:v for k,v in receipt.items() if k not in ('audit','capture')},ensure_ascii=False)+'。来源ue_round6/recovered_houses/'+d.name+'/result.json；旧失败文件保留。\n')
if __name__=='__main__':main()
