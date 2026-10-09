"""Prepare compact original-USD semantics and real CPU overhead renders."""
from __future__ import annotations
import argparse, os, time, resource, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import shapely
from shapely.geometry import mapping, shape
from scipy.spatial import cKDTree
from tools.rooms.room_split_kujiale.adapter import read, dump
from tools.rooms.room_split_kujiale.usd_mesh import read_original
from tools.rooms.room_split_kujiale.raster import render_house
from tools.rooms.room_selection.measurements import layer_furniture


def prepare_house(h, out):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (20*1024**3, 20*1024**3))
    from pxr import Work
    Work.SetConcurrencyLimit(1)
    t=time.time(); out=Path(out)
    scene=read_original(h)
    old=np.load(Path(h['build'])/'surface/vertices.npy',mmap_mode='r')
    tree=cKDTree(scene.vertices)
    ids=np.linspace(0,len(old)-1,min(2000,len(old)),dtype=int)
    distances=tree.query(np.asarray(old[ids]),workers=1)[0]
    parity=dict(previous_vertices=str(Path(h['build'])/'surface/vertices.npy'),
                sample_count=len(ids), raw_original_vertex_count=len(scene.vertices),
                previous_vertex_count=len(old), maximum_nearest_raw_error_m=float(distances.max()),
                fraction_within_1mm=float((distances<=.001).mean()),
                matrix_trials=1, method='fixed-index existing surface vertices to original USD composed vertices, cKDTree; no transform search')
    if parity['fraction_within_1mm']<.999:
        raise ValueError('Original USD and existing placement surface coordinate mismatch: '+str(parity))
    furniture={}
    for y in sorted({r['floor_y_m'] for r in h['rooms']}):
        items=layer_furniture(scene,-1,y,1.5)
        furniture[str(y)]=[dict(x,geometry=mapping(x['geometry'])) for x in items]
    markers=[{k:v for k,v in m.items() if k!='triangles'} for m in scene.markers]
    dump(out/'scene_adapters_v1'/(h['house']+'.json'),dict(house=h['house'],furniture_by_height=furniture,
         markers=markers,raw_mesh_receipt=scene.receipt,existing_surface_parity=parity))
    renders=render_house(h,scene,out/'overhead_cpu_v1')
    receipt=dict(house=h['house'],pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),
                 elapsed_s=time.time()-t,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 address_space_limit_gib=20,raw_mesh=scene.receipt,surface_parity=parity,
                 render_paths=[r['path'] for r in renders],renderer='CPU raw USD',GPU_used=False)
    dump(out/'evidence'/('scene_prepare_'+h['house']+'.json'),receipt)
    print('USD_RENDER',h['house'],'seconds',round(time.time()-t,1),'triangles',len(scene.faces),'parity_max',parity['maximum_nearest_raw_error_m'],flush=True)
    return receipt


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',required=True);a.add_argument('--workers',type=int,default=4);a.add_argument('--houses',nargs='*')
    v=a.parse_args();out=Path(v.root);plan=read(out/'input_plan_v1.json');houses=plan['houses']
    if v.houses: houses=[h for h in houses if h['house'] in v.houses]
    if not 1<=v.workers<=4:raise ValueError('At most four 20GiB workers plus one 10GiB parent')
    resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,20*1024**3));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    receipts=[];started=time.time()
    if v.workers==1:
        for h in houses: receipts.append(prepare_house(h,out))
    else:
        with ProcessPoolExecutor(max_workers=v.workers,mp_context=multiprocessing.get_context('fork')) as pool:
            fs={pool.submit(prepare_house,h,out):h['house'] for h in houses}
            for f in as_completed(fs):
                try:receipts.append(f.result())
                except Exception as e:receipts.append(dict(house=fs[f],status='failed',error=repr(e)))
    dump(out/'evidence'/('scene_batch_'+('_'.join(v.houses) if v.houses else 'all')+'.json'),dict(receipts=receipts,workers=v.workers,elapsed_s=time.time()-started,
        max_compute_processes=v.workers+1,combined_address_space_limit_gib=20*v.workers+10,cpu_only=True))
    if any(r.get('status')=='failed' for r in receipts):raise RuntimeError('Some scene preparations failed; inspect receipts')

if __name__=='__main__':main()
