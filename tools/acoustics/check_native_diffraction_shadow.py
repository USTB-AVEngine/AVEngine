#!/usr/bin/env python3
"""CPU shadow controls verify diffraction behavior beyond configuration readback.

This supplemental control is outside the fixed 80-sample admission test.
Transmission is explicitly disabled in both shadow arms to isolate paths;
all four production comparisons and their references stay unchanged.
"""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from avengine.timeline.continuous_pilot import write_json
from tools.timeline.pilot_continuous_scenes import native_renderer

def read(path):return json.loads(Path(path).read_text())
def work(task):
    ip,layout_root,out,hid=task;inputs=read(ip);house=next(h for h in inputs['houses'] if h['house']==hid)
    layout=read(Path(layout_root)/hid/'native_layout.json');listener=np.asarray(layout['listener_m'])
    points=[p for p in layout['candidate_slots'] if p['native_LOS_measured'] and p['blocked']]
    if not points:
        from avengine.acoustics.runtime import load_habitat_runtime
        hs,_=load_habitat_runtime(runtime_mode='current-installed',**inputs['runtime'])
        pf=hs.PathFinder()
        if not pf.load_nav_mesh(house['input']['navmesh_source']):raise ValueError('read-only navmesh unavailable')
        control,_=native_renderer(inputs,house);control.render([house['room']['placement']['source_1_m']])
        floor=float(np.asarray(pf.snap_point(np.asarray(house['room']['placement']['source_1_m'])-[0.,1.3,0.]))[1])
        for x in np.arange(listener[0]-5,listener[0]+5.01,.65):
            for z in np.arange(listener[2]-5,listener[2]+5.01,.65):
                nominal=np.array([x,floor,z]);foot=np.asarray(pf.snap_point(nominal))
                if not np.isfinite(foot).all() or np.linalg.norm(foot-nominal)>.1:continue
                position=foot+[0.,1.3,0.];distance=np.linalg.norm(listener-position)
                if not 1.<distance<8.:continue
                hit=control.context.trace_ray_first_hit(position.tolist(),((listener-position)/distance).tolist(),.001,distance-.001)
                if hit.hit and float(hit.distance)>.2:
                    points.append({'position_m':position.tolist(),'native_LOS_measured':True,'blocked':True,
                        'first_hit_distance_m':float(hit.distance),'scope':'bounded same-house same-floor native shadow diagnostic; not room admission'})
    points.sort(key=lambda p:(np.linalg.norm(np.array(p['position_m'])-listener),p['position_m']))
    selected=[]
    for point in points:
        if all(np.linalg.norm(np.array(point['position_m'])-np.array(old['position_m']))>.65 for old in selected):selected.append(point)
        if len(selected)==3:break
    root=Path(out)/hid;root.mkdir(parents=True,exist_ok=False);rows=[]
    if not selected:
        result={'house':hid,'status':'not_run_no_native_shadow_slots','formal_dataset_count':0}
        write_json(root/'summary.json',result);return result
    for enabled in (False,True):
        requested=deepcopy(inputs);requested['simulation'].update(diffraction=enabled,max_diffraction_order=10 if enabled else 0,transmission=False)
        renderer,_=native_renderer(requested,house);write_json(root/f'diffraction_{int(enabled)}_setup.json',renderer.setup_report)
        for i,point in enumerate(selected):
            position=np.asarray(point['position_m']);distance=float(np.linalg.norm(listener-position));delay=distance/343
            hit=renderer.context.trace_ray_first_hit(position.tolist(),((listener-position)/distance).tolist(),.001,distance-.001) if i else None
            values=[]
            for repeat in range(3):
                batch=renderer.render([position.tolist()]);wave=batch.samples[0]
                np.save(root/f'diffraction_{int(enabled)}_point{i}_repeat{repeat}.npy',wave,allow_pickle=False)
                lo=max(0,round((delay-.001)*16000));hi=min(wave.shape[-1],round((delay+.020)*16000))
                values.append({'total_W_energy':float(np.square(wave[0]).sum()),
                    'first20ms_after_geometric_arrival_W_energy':float(np.square(wave[0,lo:hi]).sum()),
                    'native_CPU_seconds':batch.process_cpu_seconds,'native_wall_seconds':batch.wall_seconds})
            rows.append({'diffraction':enabled,'source_m':position.tolist(),'LOS_authority':point,'repeats':values})
    comparisons=[]
    for i in range(len(selected)):
        off=rows[i]['repeats'];on=rows[len(selected)+i]['repeats']
        def stats(items,key):
            vals=[r[key] for r in items];return {'min':min(vals),'max':max(vals),'mean':float(np.mean(vals))}
        details={}
        for key in ('total_W_energy','first20ms_after_geometric_arrival_W_energy'):
            a,b=stats(off,key),stats(on,key)
            details[key]={'off':a,'on':b,'mean_on_over_off_db':float(10*np.log10(max(b['mean'],1e-30)/max(a['mean'],1e-30))),
                          'repeat_ranges_disjoint':a['max']<b['min'] or b['max']<a['min']}
        comparisons.append({'source_m':selected[i]['position_m'],'metrics':details})
    result={'house':hid,'status':'control_complete','rows':rows,'comparisons':comparisons,
        'transmission':False,'indirect':True,'not_one_of_the_four_production_arms':True,'formal_dataset_count':0,
        'native_calls':len(rows)*3,'native_CPU_seconds':sum(r['native_CPU_seconds'] for row in rows for r in row['repeats'])}
    write_json(root/'summary.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--layout-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    inputs=read(a.inputs);a.output.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(work,[(str(a.inputs),str(a.layout_root),str(a.output),h['house']) for h in inputs['houses']]))
    write_json(a.output/'summary.json',{'houses':rows,'fixed80_admission_unchanged':True,'formal_dataset_count':0})
if __name__=='__main__':main()
