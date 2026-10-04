#!/usr/bin/env python3
"""CPU views for all training rooms and strict-zero geometry comparisons."""
from __future__ import annotations
import argparse,copy,datetime,faulthandler,json,resource,signal,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from shapely.geometry import shape
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'src')]
from tools.acoustics.train_house_followup import read,save,corrected_result
from tools.acoustics.derive_zero_area_glb import derive
from tools.acoustics.cpu_scan_rays import CPUScanRayIntersector
from tools.acoustics.verify_train_house_derivation import differences
from avengine.acoustics.house_acceptance_visual import load_raw_scan,render_probe
from avengine.acoustics.house_acceptance_geometry import valid_native_triangles
from tools.rooms.room_selection.cpu_overhead import CPUOverhead
from tools.rooms.room_selection.run import load_parameters

def directions_without_audio(room):
    """Visibility alone can still be checked when semantic acoustics fail."""
    p=room['placement'];l=np.asarray(p['camera_m'],float);a=np.asarray(p['source_1_m'],float);axis=a-l
    sources=[a,np.asarray(p['source_2_m'],float),l+axis*.65];ds=[]
    for name,s in zip(('source_1','source_2','range_middle'),sources,strict=True):
        forward=s-l;forward[1]=0;forward/=np.linalg.norm(forward)
        yaw=np.radians(-20 if name=='source_2' else 20)
        forward=np.array([forward[0]*np.cos(yaw)+forward[2]*np.sin(yaw),0,-forward[0]*np.sin(yaw)+forward[2]*np.cos(yaw)])
        ds.append(dict(probe=room['room_label']+'_'+name,listener_m=l.tolist(),source_m=s.tolist(),camera_forward=forward.tolist(),check=dict(foa_world_direction=None,angle_deg=None)))
    return ds

def memory_limit(maximum_gib=16):
    """Bound only this owned CPU worker; shared limits/environments stay intact."""
    limit=int(maximum_gib*1024**3)
    soft,hard=resource.getrlimit(resource.RLIMIT_AS)
    if soft!=resource.RLIM_INFINITY:limit=min(limit,soft)
    if hard!=resource.RLIM_INFINITY:limit=min(limit,hard)
    resource.setrlimit(resource.RLIMIT_AS,(limit,hard))
    return limit

def memory_stage(name):
    print('VISUAL_STAGE',name,'max_rss_mib',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,flush=True)

def one(output,house,destination=None):
    h=read(output/'inputs'/('cache_'+house+'.json'))['input']
    root=output/'houses'/house;dest=destination if destination is not None else output/'visuals'/house;dest.mkdir(parents=True,exist_ok=False)
    manifest=read(output/'inputs'/(h['family']+'_settings.json'))
    raw=Path(h['scene_directory'])/(h['scan_id']+'.glb')
    try:
        prior=h.get('previous_geometry') or {}
        memory_stage('raw_load_start');raw_mesh,raw_colors,modes=load_raw_scan(raw);memory_stage('raw_load_done')
        points=np.asarray(raw_mesh.vertices,dtype=np.float64)[raw_mesh.faces]
        exact=np.all(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])==0,axis=1)
        del points
        receipt=None;derived=None;native_rejected=int((~valid_native_triangles(raw_mesh.vertices,raw_mesh.faces)).sum())
        if exact.any():
            if prior.get('derived') and prior.get('maximum_omitted_area_m2')==0 and prior.get('float32_roundoff_zero_faces',0)==0 and Path(prior['derived']).exists():
                receipt=dict(prior,reused_previous_derivation=True);derived=Path(prior['derived'])
            else:
                derived=dest/'geometry'/(h['scan_id']+'.glb');receipt=derive(raw,derived,strict=True)
            save(dest/'geometry_derivation.json',receipt);memory_stage('strict_derivation_done')
            after_mesh,after_colors,after_modes=load_raw_scan(derived);memory_stage('derived_raw_load_done')
            engine=CPUScanRayIntersector(after_mesh);before_engine=CPUScanRayIntersector(raw_mesh)
        else:
            after_mesh,after_colors,after_modes=raw_mesh,raw_colors,modes;engine=CPUScanRayIntersector(raw_mesh);before_engine=None
        memory_stage('CPU_BVH_done');overheads=[];views=[];room_rows=[]
        before_overhead=after_overhead=None
        if receipt:
            _,params=load_parameters(output/'inputs/thresholds.frozen.yaml')
            memory_stage('overhead_load_start');texture_cache={};before_overhead=CPUOverhead(raw,texture_cache);memory_stage('before_overhead_load_done');after_overhead=CPUOverhead(derived,texture_cache);memory_stage('after_overhead_load_done')
        for room,rule in zip(h['acoustic_rooms'],h['room_rule_records'],strict=True):
            label=room['room_label'];memory_stage('room_'+label);result_path=root/'rooms'/label/'result.json';result=read(result_path)
            d=dest/'rooms'/label;d.mkdir(parents=True)
            directions=result.get('foa_checks') or directions_without_audio(room)
            checks=[render_probe(after_mesh,after_colors,engine,x,d,i+1) for i,x in enumerate(directions)]
            value=dict(status='pass' if len(checks)==3 and all(x['status']=='pass' for x in checks) else 'review',checks=checks,raw_visual_asset=str(raw),derived_visual_asset=str(derived) if derived else None,ray_engine='validated Numba CPU BVH, all positive surfaces retained',ray_receipt=engine.receipt,compute_device='CPU',native_habitat_rgb_verified=False,foa_measured=bool(result.get('foa_checks')),black_fraction_frozen=rule['metrics']['black_fraction'],black_threshold_unchanged=True,geometry_derivation=receipt,scope='CPU scan visibility, source projection and frozen room-overhead admission; no production Habitat RGB')
            compare=[]
            if receipt:
                before=d/'before';before.mkdir()
                before_checks=[render_probe(raw_mesh,raw_colors,before_engine,x,before,i+1) for i,x in enumerate(directions)]
                for a,b in zip(before_checks,checks,strict=True):
                    with Image.open(a['image_path']) as ia,Image.open(b['image_path']) as ib:diff=differences(ia,ib)
                    compare.append(dict(probe=a['probe'],before=a['image_path'],after=b['image_path'],**diff,projection_status_unchanged=a['status']==b['status'],marker_pixels_unchanged=a['marker_visible_pixels']==b['marker_visible_pixels']))
                m=rule['metrics'];poly=shape(m['floor_polygon']);floor=m['floor_y_m']
                _,_,a,ad=before_overhead.render(poly,floor,d/'overhead_before.png',params)
                _,_,b,bd=after_overhead.render(poly,floor,d/'overhead_after.png',params)
                overheads.append(dict(room_label=label,**differences(a,b),before_usable=ad['black_measurement_usable'],after_usable=bd['black_measurement_usable']))
                views+=compare
                fields=['rirs','leakage','distance_attenuation','foa_checks','foa_all_positions','rt60_mid_median_s','rt60_valid_fraction','max_escape_fraction','max_foa_angle_deg','distinct_listener_positions','status','reasons']
                if result.get('rirs'):
                    before_metrics=corrected_result(result,manifest);after_metrics=corrected_result(copy.deepcopy(result),manifest)
                    acoustic=dict(exactly_equal=all(before_metrics.get(k)==after_metrics.get(k) for k in fields),by_field={k:before_metrics.get(k)==after_metrics.get(k) for k in fields},comparison_method='visual GLB derivation changes no semantic acoustic array/material/RIR; e3dc246 deterministic same-waveform recomputation',native_propagation_rerun=False,waveform_paths=[x['rir_path'] for x in result['rirs']],acoustic_package_unchanged=result.get('acoustic_package_manifest'))
                else:acoustic=dict(exactly_equal=None,status='not_run',reason='Original semantic acoustic load or material conflict prevented waveform measurement; keep review')
                save(d/'acoustic_comparison.json',acoustic)
                value['derivation_comparison']=dict(cpu_views=compare,overhead=overheads[-1],acoustic=acoustic)
                if not all(x['different_pixels']==0 and x['projection_status_unchanged'] and x['marker_pixels_unchanged'] for x in compare) or overheads[-1]['different_pixels']!=0 or acoustic['exactly_equal'] is not True:
                    value['geometry_comparison_status']='review'
                else:value['geometry_comparison_status']='pass'
            else:value['geometry_comparison_status']='not_needed'
            if native_rejected and not receipt:
                value['status']='review';value['reasons']=['NATIVE_REJECTED_POSITIVE_AREA_FACES_RETAINED']
            if receipt and value['geometry_comparison_status']!='pass':
                value['status']='review';value['reasons']=['GEOMETRY_BEFORE_AFTER_COMPARISON_UNVERIFIED']
            save(d/'visual_checks.json',value);room_rows.append(dict(room_label=label,status=value['status'],geometry_status=value['geometry_comparison_status'],visual_path=str(d/'visual_checks.json')))
            print('ROOM_VISUAL',house,label,value['status'],flush=True)
        summary=dict(house=house,family=h['family'],status='pass' if all(x['status']=='pass' for x in room_rows) else 'review',rooms=room_rows,geometry_derivation=receipt,raw_native_invalid_faces=native_rejected,derived_native_invalid_faces=int((~valid_native_triangles(after_mesh.vertices,after_mesh.faces)).sum()),perspective_different_pixels=sum(x['different_pixels'] for x in views),all_perspective_pixels_identical=all(x['different_pixels']==0 for x in views),all_candidate_overheads=overheads,all_overhead_pixels_identical=all(x['different_pixels']==0 for x in overheads),compute_device='CPU',native_habitat_rgb_verified=False,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(dest/'finished.json',summary)
    except Exception as exc:
        failure=dict(house=house,status='review',reasons=[type(exc).__name__+': '+str(exc)],traceback=traceback.format_exc(),compute_device='CPU')
        save(dest/'failure.json',failure)
        for room in h['acoustic_rooms']:
            p=dest/'rooms'/room['room_label']/'visual_checks.json'
            if not p.exists():save(p,dict(failure,room_label=room['room_label']))
        save(dest/'finished.json',dict(failure,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print('VISUAL_HOUSE_FAILURE',house,failure['reasons'],flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--house',required=True);p.add_argument('--destination',type=Path);a=p.parse_args();limit=memory_limit();faulthandler.enable();faulthandler.register(signal.SIGUSR1,all_threads=True);print('VISUAL_ADDRESS_SPACE_LIMIT_BYTES',limit,flush=True);one(a.output,a.house,a.destination)
if __name__=='__main__':main()
