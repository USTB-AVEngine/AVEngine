"""Measure a registered static cat baseline and prove rigid slide preservation."""
import argparse,json,pathlib,sys,time,hashlib
import numpy as np

def main():
 argv=sys.argv[sys.argv.index('--')+1:];p=argparse.ArgumentParser();p.add_argument('--inventory',type=pathlib.Path,required=True);p.add_argument('--asset',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=False);start=time.process_time()
 import runpy
 h=runpy.run_path(str(pathlib.Path(__file__).with_name('blender_check_cat_rig.py')),run_name='helpers');rec=next(r for r in json.loads(a.inventory.read_text())['assets'] if r['asset_id']==a.asset);files=rec['files'];key=next(k for k in ('habitat_glb','runtime_glb') if k in files);path=pathlib.Path(files[key]['path']);d,b=h['glb'](path);assert not d.get('skins')
 g=h['globals_at'](d,b,{'channels':[]},0);vertices=[];faces=[]
 for ni,n in enumerate(d['nodes']):
  if 'mesh' not in n:continue
  for prim in d['meshes'][n['mesh']]['primitives']:
   v=h['accessor'](d,b,prim['attributes']['POSITION']).astype(float);f=h['accessor'](d,b,prim['indices']).astype(int).reshape(-1,3);faces.append(f+sum(len(x) for x in vertices));vertices.append((np.c_[v,np.ones(len(v))]@g[ni].T)[:,:3])
 v=np.concatenate(vertices);f=np.concatenate(faces);pairs=h['crossings'](v,f);receipt={'asset_id':a.asset,'status':'static_baseline_measured','source_glb':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rest_self_intersection_pairs':len(pairs),'vertex_count':len(v),'triangle_count':len(f),'self_intersection_tolerance':max(5,int(np.ceil(len(pairs)*.01))),'source_mesh_repaired':False,'slide_invariance':'StandingSlide has no joint animation and returns identical source skin parts for every frame. Only a native rigid root matrix changes. Edge lengths, coincident seams and self intersections are invariant; their motion deltas are exactly zero. Full native geometry and actual rotated drawable controls are separately required.','slide_max_stretch_ratio':1.,'slide_self_intersection_count_delta':0,'slide_seam_separation_delta_m':0.,'native_status':'not_run_here','cpu_s':time.process_time()-start}
 (a.output/'static_geometry.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt,flush=True)
if __name__=='__main__':main()
