#!/usr/bin/env python3
"""Evaluate acoustic-only repairs without changing original acceptance code."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO)]
import numpy as np
from avengine.acoustics.hole_patch import (RULES,loop_candidates,planar_cluster_candidates,write_json,
    shell_quads,escaped_rays,check_direct_paths,nav_prism_intersections,export_derived_glb)


def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frozen_sources():
    return [REPO/'src/avengine/acoustics/hole_patch.py',REPO/'src/avengine/acoustics/hole_patch_checks.py',Path(__file__)]


def assert_frozen(root):
    f=read(root/'rules.frozen.json')
    assert f['rules']==RULES,'Rule parameters changed after pilot'
    for r in f['decision_sources']:assert sha(REPO/r['path'])==r['sha256'],'Decision implementation changed after pilot'


def fresh_package(base, patches, h, out):
    from avengine.acoustics.compiler import compile_explicit_glb_research_scene
    from avengine.acoustics.runtime import load_compiled_acoustic_scene
    from tools.acoustics.accept_houses import room_manifest
    base=Path(base);m=read(base)
    out.mkdir(parents=True,exist_ok=False)
    if not patches:
        # A fresh reference version uses immutable source package files. All
        # paths are local to this run, while original arrays remain read-only.
        records=list(m['arrays'].values())+list(m['qa'].values())+[m['debug_mesh']]
        records.extend(value for value in m['materials'].values() if isinstance(value,dict) and 'path' in value)
        for record in records:
            src=base.parent/record['path'];dst=out/record['path'];dst.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['cp','--reflink=auto','--',str(src),str(dst)],check=True)
        shutil.copyfile(base,out/'manifest.json')
        write_json(out/'hole_patch_version.json',{'version':'acoustic_hole_patch_v1','operation':'no_op',
            'base_manifest':str(base),'original_assets_modified':False,'patch_count':0,'patch_area_m2':0})
        return out/'manifest.json'
    glb=export_derived_glb(base,patches,out/'acoustic_only_derived.glb')
    dataset=out/'scene_dataset_config.json';write_json(dataset,{'stages':{'paths':{'.glb':[str(glb)]}}})
    rm=room_manifest(h,dataset)
    for asset in rm['assets']:
        if asset['role']=='render_surface_mesh':asset['path']=str(glb)
    rm['scene']['scene_id']=str(glb)
    rm['provenance']['source_revision']='acoustic-only derived mesh; original visual and semantic assets unchanged'
    room=out/'derived_room_manifest.json';write_json(room,rm)
    package=compile_explicit_glb_research_scene(room_manifest=room,
        material_mapping=base.parent/m['materials']['source_mapping']['path'],
        material_database=base.parent/m['materials']['source_database']['path'],
        output=out/'compiled',package_id=h['house']+'_acoustic_hole_patch_v1')
    d=read(package)
    v=np.load(base.parent/m['arrays']['vertices']['path']);t=np.load(base.parent/m['arrays']['triangles']['path'])
    vi=np.load(package.parent/d['arrays']['vertices']['path']);ti=np.load(package.parent/d['arrays']['triangles']['path'])
    mid=np.load(base.parent/m['arrays']['triangle_material_ids']['path']);midi=np.load(package.parent/d['arrays']['triangle_material_ids']['path'])
    assert np.array_equal(v,vi[:len(v)]),'Base vertices changed by export'
    assert np.array_equal(t,ti[:len(t)]),'Base faces changed by export'
    assert np.array_equal(mid,midi[:len(mid)]),'Base acoustic materials changed'
    assert len(vi)==len(v)+4*len(patches) and len(ti)==len(t)+2*len(patches)
    load_compiled_acoustic_scene(package,allow_nonpassing_research_qa=True)
    write_json(out/'hole_patch_version.json',{'version':'acoustic_hole_patch_v1','operation':'append_planar_quads',
        'base_manifest':str(base),'base_vertices_exact':True,'base_faces_exact':True,'base_material_ids_exact':True,
        'original_assets_modified':False,'patch_count':len(patches),'patch_area_m2':sum(p['patch_area_m2'] for p in patches),
        'qualification':'research_candidate; same existing material authority; no production claim'})
    return package


def evaluate(a):
    if a.stage=='batch':assert_frozen(a.output)
    rows=read(a.output/'inputs/all_house_results.json');r=next(r for r in rows if r['house']==a.house);h=r['input']
    dest=a.output/a.stage/'houses'/a.house;dest.mkdir(parents=True,exist_ok=False)
    package=Path(r['acoustic_package_manifest']);m=read(package)
    vertices=np.load(package.parent/m['arrays']['vertices']['path'],mmap_mode='r')
    triangles=np.load(package.parent/m['arrays']['triangles']['path'],mmap_mode='r')
    material_ids=np.load(package.parent/m['arrays']['triangle_material_ids']['path'],mmap_mode='r')
    labels=np.empty(len(triangles),dtype=object)
    for o in m['objects']:
        start=o['triangle_offset'];labels[start:start+o['triangle_count']]=o['source_material_name']
    stamp=time.monotonic()
    candidates,rays,clusters,stats=loop_candidates(vertices,triangles,labels,r['leakage'],RULES)
    planar,plane_stats=planar_cluster_candidates(vertices,triangles,labels,r['leakage'],h['acoustic_rooms'],RULES)
    candidates.extend(planar);stats['horizontal_plane_exclusions']=plane_stats
    data={'house':a.house,'family':r['family'],'stage':a.stage,'base_result_directory':r['result_directory'],
        'base_package_manifest':str(package),'base_status':r['status'],'base_max_escape_fraction':r.get('max_escape_fraction'),
        'candidate_room_count':h['candidate_room_count'],'sampled_room_count':len(h['acoustic_rooms']),
        'holdout_reserved':h.get('holdout_reserved',False),'boundary_stats':stats,'escaped_rays':rays,
        'exit_clusters':clusters,'candidates':candidates,'patches':[],'rules':RULES,'started_at':now()}
    placement=None;raw=None
    try:
        if any(c['status']=='candidate' for c in candidates):
            from avengine.acoustics.runtime import load_habitat_runtime
            from avengine.acoustics.hole_patch_checks import placement_candidates,opening_boxes,raw_scan_rays,visible_opening_rays,opening_cone_overlaps
            selection=read(a.output/'inputs/selection.json')
            manifest=read(a.output/'inputs'/(h['family']+'_source_manifest.json'))
            habitat,runtime=load_habitat_runtime(runtime_mode='current-installed',**manifest['runtime'])
            placement=placement_candidates(h,selection['registry_snapshots'][h['registry_source']],a.output/'inputs/thresholds.frozen.yaml',habitat)
            _,nv,nt,cameras,sources,pairs,counts=placement
            data['placement_check']={'room_counts':counts,'camera_count':len(cameras),'source_count':len(sources),
                'pair_superset_count':len(pairs),'method':'original 0.25 m grid, native snap/geodesic components, original 128/256 farthest sampler',
                'all_pairs_checked':True,'native_runtime':runtime,'nav_triangle_count':len(nt)}
            write_json(dest/'placement_candidates.json',{'cameras':cameras,'sources':sources,'room_counts':counts})
            boxes=opening_boxes(vertices,triangles,labels)
            write_json(dest/'opening_boxes.json',boxes)
            for c in candidates:
                if c['status']!='candidate':continue
                direct=check_direct_paths(c['quad_m'],pairs+r.get('rirs',[]))
                nav=nav_prism_intersections(c['quad_m'],nv,nt,RULES['navigation_agent_height_m'],RULES['navigation_radius_m'])
                c['direct_consistency']={'blocked_pair_count':len(direct),'examples':direct[:8],'checked_pairs':len(pairs)+len(r.get('rirs',[]))}
                c['navigation_consistency']={'intersected_native_triangle_prisms':len(nav),'triangle_examples':nav[:16],
                    'agent_height_m':RULES['navigation_agent_height_m'],'agent_radius_m':RULES['navigation_radius_m'],'continuous_quad_clip':True}
                if direct:c['reasons'].append('PLACEMENT_DIRECT_PATH_BLOCKED')
                if nav:c['reasons'].append('NATIVE_WALKABLE_PRISM_INTERSECTION')
                cones=opening_cone_overlaps(c['quad_m'],cameras,boxes) if not c['reasons'] else []
                c['continuous_opening_cone_check']={'overlap_count':len(cones),'examples':cones[:16],'conservative_veto':True}
                if cones:c['reasons'].append('OPENING_VIEW_CONE_PRESERVATION_UNVERIFIED')
                if boxes and not c['reasons']:
                    if raw is None:raw=raw_scan_rays(Path(h['scene_directory'])/(h['scan_id']+'.glb'))
                    vis=visible_opening_rays(c['quad_m'],cameras,boxes,raw,RULES['visibility_grid_step_m'])
                else:vis={'tested_opening_rays':0,'visible_opening_violation_count':0,'status':'not_run' if c['reasons'] else 'no_semantic_opening_context'}
                c['opening_visibility']=vis
                if vis['visible_opening_violation_count']:c['reasons'].append('VISIBLE_REAL_OPENING_OCCLUDED')
                c['status']='review' if c['reasons'] else 'accepted'
                if c['status']=='accepted':
                    # Keep disjoint quads on each inferred plane. An overlap is
                    # withheld rather than merged into a larger unverified patch.
                    from shapely.geometry import Polygon
                    q=np.asarray(c['quad_m']);normal=np.asarray(c['plane_normal_outward'])
                    overlapping=False
                    for previous in data['patches']:
                        pq=np.asarray(previous['quad_m']);pn=np.asarray(previous['plane_normal_outward'])
                        if abs(normal@pn)>.99 and abs((q.mean(0)-pq.mean(0))@normal)<.05:
                            axis=int(np.argmax(abs(normal)));axes=[i for i in range(3) if i!=axis]
                            if Polygon(q[:,axes]).intersection(Polygon(pq[:,axes])).area>1e-8:overlapping=True;break
                    if overlapping:c['status']='review';c['reasons'].append('OVERLAPPING_ACCEPTED_PATCH')
                    else:
                        o=next(o for o in m['objects'] if o['source_material_name']==c['semantic_label'])
                        mid=int(material_ids[o['triangle_offset']]);c['material_id']=mid
                        categories=read(package.parent/m['materials']['categories']['path'])
                        c['material_category_record']=categories['categories'][mid]
                        mapping=read(package.parent/m['materials']['source_mapping']['path'])
                        c['material_mapping_record']=next(e for e in mapping['entries'] if e['source_material_name']==c['semantic_label'])
                        data['patches'].append(c)
            if raw:data['raw_visibility_backend']=raw.receipt
        else:data['placement_check']={'status':'no_patches; original placement cannot be changed'}
    except Exception as exc:
        data['consistency_error']=type(exc).__name__+': '+str(exc);data['consistency_traceback']=traceback.format_exc()
        data['patches']=[]
        for c in candidates:
            if c['status'] in ('candidate','accepted'):
                c['status']='review';c['reasons'].append('CONSISTENCY_UNVERIFIED')
    # The exterior-shell trial is generated, but never admitted without proof
    # that *every* escaped direction is a scan defect and openings are preserved.
    shell=shell_quads([vertices.min(0).tolist(),vertices.max(0).tolist()],RULES['shell_outside_offset_m'])
    covered=set(i for c in data['patches'] for i in c['ray_ids'])
    uncovered=sorted(set(range(len(rays)))-covered)
    shell_reasons=[]
    if uncovered:shell_reasons.append('ESCAPED_RAYS_WITHOUT_CONFIRMED_SCAN_HOLE')
    if any('door' in x or 'window' in x for x in labels):shell_reasons.append('EXTERIOR_DOOR_WINDOW_PRESERVATION_UNVERIFIED')
    if not data['patches']:shell_reasons.append('NO_CONFIRMED_HOLE; NO_SHELL_NEEDED')
    # No full shell is admitted on uncertainty. In the exceptional case of no
    # opening semantics, this trial still lacks a facade-by-facade material and
    # original-view certificate and therefore stays review.
    if not shell_reasons:shell_reasons.append('WHOLE_FACADE_MATERIAL_AND_VIEW_CERTIFICATE_UNVERIFIED')
    data['shell_trial']={'candidate_quads':shell,'candidate_patch_count':6,'candidate_area_m2':sum(c['patch_area_m2'] for c in shell),
        'unexplained_escape_ray_count':len(uncovered),'unexplained_ray_ids':uncovered,'status':'review_without_shell',
        'reasons':shell_reasons,'applied_patch_count':0,'native_full_shell_measurement':'not_run; would violate unverified-opening policy',
        'method':'closed scan AABB + 3 cm; gates checked before any acoustic upload'}
    data['patch_count']=len(data['patches']);data['patch_area_m2']=sum(c['patch_area_m2'] for c in data['patches'])
    data['local_package_manifest']=str(fresh_package(package,data['patches'],h,dest/'local_package'))
    write_json(dest/'patch_record.json',data)
    source_manifest=read(a.output/'inputs'/(h['family']+'_source_manifest.json'))
    source_manifest['houses']=[h]
    manifest=dest/'acceptance_manifest.json';write_json(manifest,source_manifest)
    runs={}
    for method,pack in (('baseline',package),('local',Path(data['local_package_manifest']))):
        log=dest/(method+'_acceptance.log');output=dest/method
        command=[sys.executable,str(REPO/'tools/acoustics/accept_houses.py'),'measure','--manifest',str(manifest),
                 '--output',str(output),'--house',a.house,'--reuse-package',str(pack)]
        with log.open('x') as f:
            proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
            write_json(dest/(method+'_process.json'),{'pid':proc.pid,'command':command,'log':str(log),'started_at':now()})
            try:rc=proc.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise RuntimeError('Owned acceptance process timed out: '+str(proc.pid))
        result=output/'houses'/a.house/'result.json'
        if rc!=0 or not result.exists():raise RuntimeError(f'{method} failed: returncode {rc}; see {log}')
        runs[method]={'result_path':str(result),'process_returncode':rc}
    summary={'house':a.house,'family':h['family'],'patch_record_path':str(dest/'patch_record.json'),'runs':runs,
        'wall_seconds':time.monotonic()-stamp,'finished_at':now(),'shell_status':data['shell_trial']['status']}
    write_json(dest/'complete.json',summary)
    print(json.dumps({'house':a.house,'patches':data['patch_count'],'area_m2':data['patch_area_m2'],
        'baseline_status':read(runs['baseline']['result_path'])['status'],'local_status':read(runs['local']['result_path'])['status'],
        'baseline_leak':read(runs['baseline']['result_path']).get('max_escape_fraction'),
        'local_leak':read(runs['local']['result_path']).get('max_escape_fraction'),'seconds':summary['wall_seconds']},ensure_ascii=False),flush=True)


def freeze(a):
    plan=read(a.output/'inputs/selection.json')
    for house in plan['pilot']:
        d=a.output/'pilot/houses'/house
        assert (d/'complete.json').exists(),'Pilot incomplete: '+house
        assert not read(d/'patch_record.json').get('consistency_error'),'Pilot consistency implementation failed: '+house
        for method in ('baseline','local'):
            r=read(d/method/'houses'/house/'result.json')
            assert not r.get('exception_traceback'),'Pilot native measurement incomplete: '+house
    frozen={'frozen_at':now(),'rules':RULES,'pilot_houses':plan['pilot'],
        'decision_sources':[{'path':str(p.relative_to(REPO)),'sha256':sha(p)} for p in frozen_sources()],
        'policy':'No decision parameter or implementation changes after these five pilot houses. Evaluation code fixes require a new trial version.',
        'pilot_summary':[read(a.output/'pilot/houses'/h/'complete.json') for h in plan['pilot']]}
    write_json(a.output/'rules.frozen.json',frozen)
    print(json.dumps({'frozen_at':frozen['frozen_at'],'pilot_houses':plan['pilot']}),flush=True)


def batch(a):
    assert_frozen(a.output)
    from concurrent.futures import ThreadPoolExecutor,as_completed
    plan=read(a.output/'inputs/selection.json');allocation=read(a.output/'evidence/cpu_allocation_batch.json')
    ids=plan['blocked']+plan['controls'];pilot=set(plan['pilot'])
    pending=[h for h in ids if h not in pilot and not (a.output/'batch/houses'/h/'complete.json').exists()]
    workers=allocation['workers'];groups=allocation['affinity_groups']
    import queue
    slots=queue.Queue()
    for g in groups:slots.put(g)
    def run(h):
        group=slots.get();log=a.output/'logs'/('batch_'+h+'.log')
        command=['taskset','-c',','.join(map(str,group)),sys.executable,str(Path(__file__).resolve()),'evaluate',
            '--output',str(a.output),'--stage','batch','--house',h,'--timeout',str(a.timeout)]
        try:
            with log.open('x') as f:
                proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
                write_json(a.output/'evidence'/('batch_process_'+h+'.json'),{'house':h,'pid':proc.pid,'command':command,'log':str(log),'started_at':now()})
                rc=proc.wait()
            return {'house':h,'returncode':rc,'log':str(log)}
        finally:slots.put(group)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(run,h) for h in pending]
        for f in as_completed(fs):print(json.dumps(f.result()),flush=True)
    missing=[h for h in pending if not (a.output/'batch/houses'/h/'complete.json').exists()]
    write_json(a.output/'evidence/batch_execution.json',{'finished_at':now(),'scheduled':pending,'missing':missing,'workers':workers,'pid':os.getpid()})
    if missing:raise ValueError('Incomplete houses: '+', '.join(missing))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    for cmd in ('evaluate','freeze','batch'):
        q=s.add_parser(cmd);q.add_argument('--output',type=Path,required=True)
        if cmd in ('evaluate','batch'):q.add_argument('--timeout',type=float,default=5400)
        if cmd=='evaluate':q.add_argument('--house',required=True);q.add_argument('--stage',choices=('pilot','batch'),required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only; CUDA_VISIBLE_DEVICES must be empty')
    {'evaluate':evaluate,'freeze':freeze,'batch':batch}[a.command](a)
if __name__=='__main__':main()
