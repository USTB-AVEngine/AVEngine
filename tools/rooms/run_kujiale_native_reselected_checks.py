#!/usr/bin/env python3
"""Re-render safe frozen witness subsets and publish overrides only after QA."""
from __future__ import annotations
import argparse,datetime,json,os,subprocess,time
from pathlib import Path
from avengine.capture.native_floor_reference import NativeFloorReference
from tools.rooms.check_kujiale_native_capsule_contact import query_capsule


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
 p.add_argument('--selection-file',type=Path);p.add_argument('--result-output',type=Path);a=p.parse_args()
 o=a.root.resolve();r=o/'ue_round6';u=o/'ue_v1';v=o/'ue_round7';repo=Path(__file__).resolve().parents[2];py=os.sys.executable
 env=dict(os.environ,PYTHONPATH=f'{repo}/src:{repo}:{u}/python_pkgs',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 records=[];override_path=r/'native_room_overrides.json'
 overrides=read(override_path)['records'] if override_path.exists() else []
 selection_file=(a.selection_file or r/'capsule_witness_reselection.json').resolve()
 result_output=(a.result_output or v/'reselected_room_results.json').resolve()
 if any(o not in p.parents for p in (selection_file,result_output)):raise ValueError('reselection receipt escaped task root')
 for selection in read(selection_file)['records']:
  if selection['status']!='ready_for_native_recapture':continue
  house=selection['house'];room_id=selection['room_id'];d=Path(selection.get('directory',r/'capsule_reselected_houses'/house));original=r/'houses'/house;started=time.monotonic();receipt={'house':house,'room_id':room_id,'status':'started'}
  try:
   config=read(d/'ue_config.json');config['graphics_adapter']=a.gpu;write(d/'ue_config.json',config)
   q=read(u/'scripts/capture_0030_final_profile_guard_request_v2.json');q['command']=[py,str(repo/'tools/rooms/check_v7_kujiale_ue.py'),'--plan',str(d/'acceptance_plan.json'),'--editor-audit',str(original/'native_editor_audit.json'),'--ue-config',str(d/'ue_config.json'),'--output',str(d/'capture')]
   q.update(gpu_index=a.gpu,previous_peak_mib=3876,run_directory=str(v/'logs'/ (house+'_'+room_id+'_reselected_capture')),cpu_affinity=list(range(96,104)),maximum_seconds=1800,storage_limit_bytes=80000000000,minimum_disk_free_bytes=150000000000,allowed_auxiliary_gpu_indices=[i for i in range(4) if i!=a.gpu],maximum_auxiliary_gpu_mib=64)
   qp=v/'scripts'/(house+'_'+room_id+'_reselected_guard.json');write(qp,q)
   with (d/'capture_coordinator.log').open('x') as log:subprocess.run([py,str(repo/'tools/rooms/v7_kujiale_ue_guard.py'),'--request',str(qp)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   guard=read(Path(q['run_directory'])/'result.json')
   if guard['status']!='pass':raise ValueError('native recapture guard did not pass')
   with (d/'finalize.log').open('x') as log:subprocess.run(['taskset','-c','96-103',py,str(repo/'tools/rooms/finalize_v7_kujiale_ue.py'),'--plan',str(d/'acceptance_plan.json'),'--editor-audit',str(original/'native_editor_audit.json'),'--capture',str(d/'capture'),'--output',str(d/'qa')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   qa=read(d/'qa/result.json');room=qa['rooms'][0];surface=NativeFloorReference(read(r/'native_floor_spatial_v1'/(house+'.json'))['native_floor_surface'])
   receipt['capsule_checks']=[query_capsule(surface.mesh,probe) for probe in room['floor_reference']['probes'] if probe.get('status')=='measured']
   receipt['native_floor_surface_validation']=surface.validate_probes(room['floor_reference'])
   receipt['status']='pass' if room['automated_room_status']=='pass' and len(receipt['capsule_checks'])>=9 and all(c['status']=='pass' for c in receipt['capsule_checks']) else 'fail'
   receipt['native_room_status']=room['automated_room_status'];receipt['qa_result']=str(d/'qa/result.json')
   if receipt['status']=='pass':
    overrides=[row for row in overrides if (row['house'],row['room_id'])!=(house,room_id)]
    overrides.append({'house':house,'room_id':room['room_id'],'qa_result':str(d/'qa/result.json'),'contact_result':str(d/'result.json'),'original_frozen_witness_indices':selection['selected_original_witness_indices']})
  except Exception as error:receipt.update(status='fail',error=f'{type(error).__name__}: {error}')
  receipt['machine_seconds']=time.monotonic()-started;write(d/'result.json',receipt);records.append(receipt)
  write(r/'native_room_overrides.json',{'schema':'kujiale_native_room_qa_overrides_v1','records':overrides})
  with (o/'PROGRESS_zh.md').open('a') as stream:stream.write('\n\n## '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+house+' 原冻结安全摆位重渲完成\n\n- 状态 '+receipt['status']+'；原生画面、地板九点、胶囊、同一几何对齐均重新验证。只选择原候选索引 '+str(selection['selected_original_witness_indices'])+'，未改门槛/几何/导航；证据 '+str(d/'result.json')+'。\n')
  print(house,receipt['status'],receipt.get('error'),flush=True)
 write(result_output,{'records':records})

if __name__=='__main__':main()
