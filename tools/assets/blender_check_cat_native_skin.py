#!/usr/bin/env python3
"""Check actual native-replay skins from a bounded private overwriteable stream."""
import argparse,json,pathlib,sys,runpy,hashlib,time
import numpy as np

def main():
 args=sys.argv[sys.argv.index('--')+1:]
 p=argparse.ArgumentParser();p.add_argument('--case',type=pathlib.Path,required=True);p.add_argument('--stream',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args(args)
 a.output.mkdir(parents=True,exist_ok=False);start=time.process_time()
 h=runpy.run_path(str(pathlib.Path(__file__).with_name('blender_check_cat_rig.py')),run_name='helpers');d,b=h['glb'](a.case/'motion.glb');parts=h['load_parts'](d,b);idle=next(x for x in d['animations'] if x['name']=='Idle');rest=h['skin'](parts,h['globals_at'](d,b,idle,0));v=np.concatenate(rest);f=np.concatenate([x['faces']+sum(len(y) for y in rest[:i]) for i,x in enumerate(parts)]);surface=runpy.run_path(str(pathlib.Path(__file__).with_name('cat_surface_checks.py')),run_name='surface_helpers');monitor=surface['RegionSurfaceChecks'](v,f,surface['anatomical_regions'](parts,d),h['crossings'],h['new_disjoint_contact_pairs']);baseline=monitor.baseline
 loaded=np.load(a.stream);assert np.array_equal(loaded['faces'],f),'native_skin_topology_changed'
 _,weld=np.unique(np.round(v/1e-7).astype(np.int64),axis=0,return_inverse=True);areas=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)*.5
 edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0);length=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1);valid=length>1e-5;rows=[];failures=[]
 for index,posed in zip(loaded['indices'],loaded['vertices']):
  posed=posed.astype(float);check=monitor.inspect(posed);stretch=float(np.max(np.linalg.norm(posed[edges[:,0]]-posed[edges[:,1]],axis=1)[valid]/length[valid]))
  lo=np.full((weld.max()+1,3),np.inf);hi=np.full_like(lo,-np.inf)
  np.minimum.at(lo,weld,posed);np.maximum.at(hi,weld,posed);seam=float(np.max(np.linalg.norm(hi-lo,axis=1)))
  issues=list(check['surface_failures'])
  if stretch>1.5:issues.append('mesh_stretch')
  if seam>1e-5:issues.append('coincident_seam_separation')
  failures.extend(issues);row={'frame_index':int(index),'status':'pass' if not issues else 'fail','failures':issues,'surface_check':check,'max_edge_stretch_ratio':stretch,'max_coincident_seam_separation_m':seam,'posed_vertices_sha256':hashlib.sha256(posed.astype(np.float32).tobytes()).hexdigest()};rows.append(row);print('NATIVE_SKIN_QA',row,flush=True)
 receipt={'status':'pass' if not failures else 'fail','failures':sorted(set(failures)),'baseline_pairs':len(baseline),'surface_check_rule':monitor.provenance(),'counts_participate_in_acceptance':False,'edge_stretch_limit':1.5,'seam_separation_limit_m':1e-5,'input_stream_sha256':hashlib.sha256(a.stream.read_bytes()).hexdigest(),'stream_storage':'private overwriteable stream, per-frame hashes and quantitative checks retained','frames':rows,'cpu_s':time.process_time()-start}
 (a.output/'skin_checks.json').write_text(json.dumps(receipt,indent=2)+'\n')
 if failures:raise SystemExit(2)
if __name__=='__main__':main()
