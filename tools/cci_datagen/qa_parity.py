#!/usr/bin/env python3
"""Read-only package geometry replay and native BVH ray parity; no dataset media."""
import argparse,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,p)

def run_house(a):
 import numpy as np
 sys.path.insert(0,str(Path(a.code_root)/'src'));sys.path.insert(0,str(Path(a.code_root)/'tools/acoustics'))
 from avengine.acoustics.qa import geometry_report
 from avengine.acoustics.runtime import load_compiled_acoustic_scene,RLRSimulationConfig
 from avengine.acoustics.rir_cache import _NativeRIRBatchRenderer
 from accept_houses import ray_checks,alternate_listener
 refs=Path(a.references);p=Path(a.mirror_root)/'data/jzy/tmp/claude_hardbank_20260923/v7_batch1_20261003/measurement_final/houses'/a.house/'package'
 m=read(p/'manifest.json');old=read(refs/'batch1'/a.house/'package/qa/geometry_report.json');new=read(refs/'batch2'/a.house/'result.json')
 files={str(p/'manifest.json'):sha(p/'manifest.json')}
 for k,v in m['arrays'].items():
  path=p/v['path'];files[str(path)]=sha(path)
  if files[str(path)]!=v['sha256']:raise ValueError('CCI array SHA mismatch: '+str(path))
 if sha(p/'qa/geometry_report.json')!=sha(refs/'batch1'/a.house/'package/qa/geometry_report.json'):raise ValueError('CCI QA snapshot SHA mismatch')
 vertices=np.load(p/m['arrays']['vertices']['path'],allow_pickle=False);triangles=np.load(p/m['arrays']['triangles']['path'],allow_pickle=False)
 g=m['geometry'];started=time.monotonic()
 actual=geometry_report(vertices,triangles,source_sha256=old['source_geometry_sha256'],representation=g['representation'],source_to_canonical=g['source_to_canonical'],objects=m['objects'])
 geometry_seconds=time.monotonic()-started
 # Research exemption is explicit and confined to diagnostic ray replay.
 scene=load_compiled_acoustic_scene(p/'manifest.json',allow_nonpassing_research_qa=True)
 manifest_path=refs/'batch2_acceptance_manifest.json'
 if not manifest_path.exists():manifest_path=refs/'batch1_acceptance_manifest.json'
 mf=read(manifest_path);sim=dict(mf['simulation']);sim['thread_count']=1;sim['temporal_coherence']=False
 runtime={'runtime_prefix':str(Path(a.native_root)/'runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z'),'magnum_python_site':str(Path(a.native_root)/'runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages'),'rlr_sdk_root':str(Path(a.native_root)/'rlr-sdk/RLRAudioPropagationPkg')}
 rooms=new['input']['acoustic_rooms'];first=rooms[0]['placement'];started=time.monotonic()
 renderer=_NativeRIRBatchRenderer(scene,RLRSimulationConfig.from_mapping(sim),batch_size=1,initial_positions_m=[first['source_1_m']],listener_position_m=first['camera_m'],listener_orientation_wxyz=[1.,0.,0.,0.],layout_type='ambisonics',channel_count=4,hrtf_file_path='',source_radius_m=mf['source_radius_m'],listener_radius_m=mf['listener_radius_m'],**runtime)
 renderer.render([first['source_1_m']])  # BVH is available only after native simulation. RIR is discarded.
 origins=[(r['room_label'],k,r['placement'][k]) for r in rooms for k in ('camera_m','source_1_m','source_2_m')]
 if len(rooms)==1:origins.append((rooms[0]['room_label'],'camera_alt_m',alternate_listener(first)))
 rays=ray_checks(renderer.context,origins,np.stack((vertices.min(0),vertices.max(0))),mf['criteria']['ray_count_per_origin'])
 expected=new['leakage'];matches=[{'room':r['room_label'],'role':r['role'],'escape_count':r['escape_count'],'expected_escape_count':e['escape_count'],'escape_indices_match':[x['direction_index'] for x in r['escaped_rays']]==[x['direction_index'] for x in e['escaped_rays']],'near_surface_hits_match':r['near_surface_hits']==e['near_surface_hits']} for r,e in zip(rays,expected)]
 result={'house':a.house,'host':socket.gethostname(),'scope':'read-only geometry/native ray diagnosis, no dataset video/FOA/questions','geometry_seconds':geometry_seconds,'native_setup_and_rays_seconds':time.monotonic()-started,'geometry_equal':actual==old,'geometry_actual_canonical_sha256':canonical(actual),'geometry_source_canonical_sha256':canonical(old),'geometry_status':actual['status'],'topology':{k:v for k,v in actual['topology'].items() if k!='per_object'},'ray_count':sum(r['ray_count'] for r in rays),'max_escape_fraction':max(r['escape_fraction'] for r in rays),'ray_records_equal':rays==expected,'ray_canonical_sha256':canonical(rays),'expected_ray_canonical_sha256':canonical(expected),'ray_matches':matches,'thread_count':1,'input_file_sha256':files,'native_configuration':renderer.setup_report}
 save(Path(a.output)/a.house/'geometry_replayed.json',actual);save(Path(a.output)/a.house/'leakage_replayed.json',rays);save(Path(a.output)/a.house/'result.json',result)
 print(json.dumps({k:result[k] for k in ('house','geometry_equal','geometry_status','ray_count','ray_records_equal','max_escape_fraction')}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--references',required=True);p.add_argument('--mirror-root',default='/');p.add_argument('--native-root',default='/data/avengine_external');p.add_argument('--code-root',default=str(Path(__file__).resolve().parents[2]));p.add_argument('--output',required=True);p.add_argument('--house')
 a=p.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU diagnostic must set CUDA_VISIBLE_DEVICES empty')
 if a.house:return run_house(a)
 houses=read(Path(a.references)/'recommended_initial12.json')['houses'];results=[]
 for h in houses:
  r=Path(a.output)/h['house']/'result.json'
  if not r.exists():
   cmd=[sys.executable,'-B',__file__,*sys.argv[1:],'--house',h['house']]
   subprocess.run(cmd,check=True,stdin=subprocess.DEVNULL)
  results.append(read(r))
 summary={'houses':results,'geometry_equal_count':sum(r['geometry_equal'] for r in results),'native_ray_equal_count':sum(r['ray_records_equal'] for r in results),'ray_count':sum(r['ray_count'] for r in results),'geometry_fail_count':sum(r['geometry_status']=='fail' for r in results),'max_escape_fraction':max(r['max_escape_fraction'] for r in results),'formal_test_members_generated':0}
 save(Path(a.output)/'SUMMARY.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='houses'}))
if __name__=='__main__':main()
