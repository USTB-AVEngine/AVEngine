#!/usr/bin/env python3
"""Verify authorized MP3D zero-surface derived copies with native CPU rays."""
from __future__ import annotations
import argparse,copy,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from shapely.geometry import shape
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'src')]
from tools.acoustics.derive_zero_area_glb import derive
from tools.acoustics.train_house_followup import corrected_result,read,save
from tools.rooms.room_selection.cpu_overhead import CPUOverhead
from tools.rooms.room_selection.run import load_parameters
from avengine.acoustics.house_acceptance_visual import load_raw_scan,NativeRawRayIntersector,render_probe
from avengine.acoustics.house_acceptance_geometry import valid_native_triangles
from tools.acoustics.cpu_scan_rays import CPUScanRayIntersector

def differences(a,b):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
    delta=np.abs(a.astype(np.int16)-b.astype(np.int16))
    return dict(shape=list(a.shape),different_pixels=int(np.any(delta!=0,axis=-1).sum()),maximum_channel_difference=int(delta.max(initial=0)),mean_absolute_channel_difference=float(delta.mean()),root_mean_square_channel_difference=float(np.sqrt(np.square(delta.astype(float)).mean())))

def visual(result,manifest,raw,destination):
    destination.mkdir(parents=True,exist_ok=False)
    mesh,colors,modes=load_raw_scan(raw);native_invalid=int((~valid_native_triangles(mesh.vertices,mesh.faces)).sum())
    native_error=None
    try:engine=NativeRawRayIntersector(mesh,result,manifest,destination)
    except ValueError as exc:
        if 'contains a zero-area triangle' not in str(exc):raise
        # Retain all original faces in a CPU reference BVH. This accelerates
        # first-batch render_probe without weakening the native upload guard.
        native_error=str(exc);engine=CPUScanRayIntersector(mesh)
    checks=[render_probe(mesh,colors,engine,d,destination,i+1) for i,d in enumerate(result['foa_checks'])]
    value=dict(status='pass' if len(checks)==3 and all(c['status']=='pass' for c in checks) else 'review',raw_visual_asset=str(raw),coordinate_transform='raw (x,y,z) -> Habitat (x,z,-y)',checks=checks,color_modes=modes,ray_bvh_setup=getattr(engine,'receipt',None),ray_engine='Numba CPU scan BVH reference' if native_error else 'native RLR CPU raw BVH',native_raw_upload_error=native_error,triangle_count=len(mesh.faces),native_invalid_faces_remaining=native_invalid,compute_device='CPU',native_habitat_rgb_verified=False,claim='unchanged first-batch render_probe; CPU reference first-hit rays when native raw upload rejects; no Habitat RGB production capture')
    save(destination/'visual_checks.json',value);return value

def one(output,work,h,manifest):
    house=h['house'];d=work/'mp3d_derived_v4'/house;d.mkdir(parents=True,exist_ok=False)
    raw=Path(h['scene_directory'])/(h['scan_id']+'.glb');clean=d/'geometry'/(h['scan_id']+'.glb')
    receipt=derive(raw,clean);save(d/'geometry_derivation.json',receipt)
    before=read(work/'mp3d_corrected/houses'/house/'result.json')
    # This task changes the raw visual GLB only. The semantic PLY acoustic
    # geometry, material package and saved propagation waveforms are identical.
    # Recompute every acoustic field on each side from those same waveforms;
    # do not claim an independent Monte Carlo propagation rerun is bit-identical.
    before_metrics=corrected_result(before,manifest)
    after=copy.deepcopy(before);after['input']=dict(h,derived_visual_source=str(clean))
    after_metrics=corrected_result(after,manifest)
    fields=['status','reasons','rirs','leakage','distance_attenuation','foa_checks','foa_all_positions','rt60_mid_median_s','rt60_valid_fraction','max_escape_fraction','max_foa_angle_deg','distinct_listener_positions']
    acoustic=dict(exactly_equal=all(before_metrics.get(k)==after_metrics.get(k) for k in fields),by_field={k:before_metrics.get(k)==after_metrics.get(k) for k in fields},semantic_source_unchanged=h['semantic_source'],acoustic_package_unchanged=before['acoustic_package_manifest'],waveforms_unchanged=[r['rir_path'] for r in before['rirs']],comparison_method='same unchanged semantic acoustic inputs and cached RIRs, deterministic e3dc246 recomputation before/after visual-only derivation',native_propagation_rerun=False)
    save(d/'acoustic_comparison.json',acoustic)
    original_view=visual(before,manifest,raw,d/'visual_before');derived_view=visual(after,manifest,clean,d/'visual_after')
    views=[]
    for a,b in zip(original_view['checks'],derived_view['checks'],strict=True):
        with Image.open(a['image_path']) as ia,Image.open(b['image_path']) as ib:diff=differences(ia,ib)
        views.append(dict(probe=a['probe'],before=a['image_path'],after=b['image_path'],**diff,projection_status_unchanged=a['status']==b['status'],marker_visible_pixels_before=a['marker_visible_pixels'],marker_visible_pixels_after=b['marker_visible_pixels']))
    _,p=load_parameters(work/'inputs/thresholds.frozen.yaml')
    rooms=read(output/'mp3d_rooms_final/houses'/(house+'.json'))['rows'];rooms=[r for r in rooms if r['stage1']['status']=='pass']
    before_overhead=CPUOverhead(raw);after_overhead=CPUOverhead(clean);overheads=[]
    for r in rooms:
        m=r['metrics'];scope=shape(m['floor_polygon']);floor_y=m['floor_y_m'];label=r['room_label']
        _,_,a,ad=before_overhead.render(scope,floor_y,d/'overheads_before'/(label+'.png'),p)
        _,_,b,bd=after_overhead.render(scope,floor_y,d/'overheads_after'/(label+'.png'),p)
        overheads.append(dict(room_label=label,**differences(a,b),before_measurement_usable=ad['black_measurement_usable'],after_measurement_usable=bd['black_measurement_usable']))
    compare=dict(house=house,removed_faces=receipt['removed_faces'],exact_zero_faces=receipt['exact_zero_faces'],float32_roundoff_zero_faces=receipt['float32_roundoff_zero_faces'],maximum_omitted_area_m2=receipt['maximum_omitted_area_m2'],candidate_rooms=len(rooms),acoustic_exactly_equal=acoustic['exactly_equal'],native_invalid_faces_after=derived_view['native_invalid_faces_remaining'],cpu_perspective_images=views,all_candidate_overhead_images=overheads,all_overhead_pixels_identical=all(x['different_pixels']==0 for x in overheads),perspective_different_pixels=sum(x['different_pixels'] for x in views),visual_status_before=original_view['status'],visual_status_after=derived_view['status'],source_raw_unchanged=True,native_habitat_rgb_verified=False,ray_engine_before=original_view['ray_engine'],ray_engine_after=derived_view['ray_engine'],native_raw_upload_error_before=original_view['native_raw_upload_error'],native_raw_upload_error_after=derived_view['native_raw_upload_error'])
    compare['status']='pass' if acoustic['exactly_equal'] and compare['native_invalid_faces_after']==0 and compare['all_overhead_pixels_identical'] and derived_view['status']=='pass' and all(v['projection_status_unchanged'] and v['marker_visible_pixels_before']==v['marker_visible_pixels_after'] for v in views) else 'review'
    compare['perspective_difference_interpretation']='nearest-face-centroid display coloring can change when empty face centroids are removed; full statistics retained; exact textured admission overheads and source projection must remain unchanged'
    save(d/'comparison.json',compare)
    current=work/'mp3d_current/houses'/house
    save(current/'result.json',dict(after,derived_geometry=receipt,derived_geometry_comparison=str(d/'comparison.json')))
    derived_view.update(derived_geometry=receipt,derivation_comparison=compare)
    if compare['status']!='pass':derived_view.update(status='review',reasons=['DERIVED_GEOMETRY_COMPARISON_REQUIRES_REVIEW'])
    save(current/'visual_checks.json',derived_view)
    return compare

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--house',action='append');a=p.parse_args()
    manifest=read(a.output/'mp3d_manifest.json');rows=[]
    for h in manifest['houses']:
        result=read(a.work/'mp3d_corrected/houses'/h['house']/'result.json')
        if result['status']!='pass' or (a.house and h['house'] not in a.house):continue
        previous=[a.work/f/h['house']/'comparison.json' for f in ['mp3d_derived','mp3d_derived_v2','mp3d_derived_v3']]
        previous=next((p for p in previous if p.exists() and read(p)['status']=='pass'),None)
        if previous:
            rows.append(read(previous));print('MP3D_DERIVATION_REUSED',h['house'],flush=True);continue
        try:r=one(a.output,a.work,h,manifest)
        except Exception as exc:
            import traceback
            r=dict(house=h['house'],status='review',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());save(a.work/'mp3d_derived_v4'/h['house']/'failure.json',r)
        rows.append(r);print('MP3D_DERIVATION',json.dumps({k:v for k,v in r.items() if k in ('house','status','removed_faces','candidate_rooms','perspective_different_pixels','all_overhead_pixels_identical','error')}),flush=True)
    save(a.work/'evidence/mp3d_derivation_summary_v4.json',dict(houses=len(rows),pass_houses=sum(r['status']=='pass' for r in rows),pass_rooms=sum(r.get('candidate_rooms',0) for r in rows if r['status']=='pass'),rows=rows))
if __name__=='__main__':main()
