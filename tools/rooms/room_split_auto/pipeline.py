"""CPU-only HM3D splitting with explicit scope, witnesses, and no-replace outputs."""
from __future__ import annotations
from pathlib import Path
import json,os,math,time,traceback,datetime,resource,subprocess
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter,defaultdict
import numpy as np
import shapely
from shapely.geometry import shape,mapping,GeometryCollection,LineString,Point
import trimesh
from tools.rooms.room_selection.geometry import canonical,short_side,infer_type
from tools.rooms.room_selection.navigation import sample_navigation,placement
from tools.rooms.room_selection.measurements import navmesh_triangles,load_scene
from tools.rooms.room_selection.media import black_metric
from tools.rooms.room_screening.geometry import union_projected_polygons
from .scene import structural_instances
from .contours import structural_partition,body_width
from .visibility import partition_visibility,assess_visibility,grid_atoms
from .cpu_rays import CPUScanRayIntersector
from .atlas import floor_overhead


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,allow_nan=False)


def prepare(root,base,training_csv=None):
    root=Path(root);base=Path(base)
    v20=json.loads((root/'reference_smy_readonly_copy/cut_planning_review_manifest_v20.json').read_text())
    split=json.loads((root/'house_hash_split_v1.json').read_text())
    origin=defaultdict(set)
    for group in ('uncut','matched'):
        for d in v20[group]:origin[(d['house'],d['label'])].add('smy_v20_181')
    for d in split['source_rooms']:origin[(d['house'],d['region'])].add('smy_hand_cut_33')
    rows={};counts=Counter();jobs={}
    for f in sorted((root/'inventory_v1').glob('*.json')):
        result=json.loads(f.read_text());counts[result['status']]+=1
        for r in result['rows']:
            key=(r['house'],r['room_label']);rows[key]=r
            if r['floor_area_sum_m2']>50+1e-8:origin[key].add('fresh_semantic_floor_gt50')
    # Existing status is informational; no original candidate or admission file is changed.
    for line in (base/'rooms_registry.jsonl').open():
        old=json.loads(line);key=(old['house'],old['room_label'])
        if key in rows:
            rows[key]['existing_stage1_status']=old.get('stage1',{}).get('status')
            rows[key]['existing_floor_area_m2']=(old.get('metrics') or {}).get('floor_area_m2')
            rows[key]['existing_room_type']=old.get('room_type')
    missing=[]
    fold={h['house']:h['fold'] for h in split['houses']}
    for key,sources in sorted(origin.items()):
        if key not in rows:missing.append({'house':key[0],'region':key[1],'origins':sorted(sources),'reason':'SOURCE_REGION_MISSING_FROM_FRESH_SEMANTIC_INVENTORY'});continue
        r=rows[key];r['origins']=sorted(sources);r['reference_fold']=fold.get(key[0]);r['requires_split']=r['floor_area_sum_m2']>50+1e-8
        if r['house'] not in jobs:
            job=json.loads((base/'jobs'/(r['house']+'.json')).read_text());jobs[r['house']]={k:v for k,v in job.items() if k!='rows'};jobs[r['house']]['rows']=[]
        jobs[r['house']]['rows'].append(r)
    plan={'created_at_utc':now(),'inventory_houses':len(list((root/'inventory_v1').glob('*.json'))),'inventory_status_counts':dict(counts),'target_sources':len(origin),'available_sources':sum(len(j['rows']) for j in jobs.values()),'target_houses':len(jobs),'source_counts':dict(Counter(s for a in origin.values() for s in a)),'missing':missing,'jobs':list(jobs.values()),'parameters':json.loads((base/'inputs.json').read_text())['parameters'],'reference_split':str(root/'house_hash_split_v1.json'),'overhead_cache':str(base/'media_reference'),'training_csv':str(training_csv) if training_csv else None,'baseline':'b2d625ff32258bd7fb19fdcc17764a752d774034','within_50_policy':'unchanged; no new rejection, regardless of diagnostic checks; not counted as a new room'}
    dump(root/'processing_plan_v1.json',plan)
    # This grid and tie-break are recorded before any tune score is computed.
    grid=[{'id':f'P{i:02d}','width_min':0.6,'width_max':w,'coverage':c,'max_distance':d} for i,(w,c,d) in enumerate(( (w,c,d) for w in (1.4,1.6,1.8) for c in (0.7,0.8,0.9) for d in (5.0,6.0,8.0)))]
    dump(root/'parameter_search_plan_v1.json',{'grid':grid,'selection':'maximum tune source-macro mean IoU; ties prefer default 0.6–1.6 m, 80%, 6 m, then stable parameter ID','allowed_parameters_only':True,'holdout_scores_unavailable_to_selection':True,'primary_iou':'source-semantic-floor clipped by manual bbox; Hungarian maximum one-to-one pairing; zero for unmatched/unresolved references; only dagger excluded','global_minimum_policy':'universal ceil(A/50) bound can certify a finite-camera construction; otherwise global optimality is explicitly unverified'})
    with (root/'PROGRESS_zh.md').open('a') as f:
        f.write('\n## 新鲜测量与处理并集\n'+json.dumps({k:v for k,v in plan.items() if k in ('inventory_houses','inventory_status_counts','target_sources','available_sources','target_houses','source_counts','missing')},ensure_ascii=False,indent=2)+'\n调参前已写 parameter_search_plan_v1.json：27 组，仅调窄口上界、可见比例、最远距离；平分优先冻结默认值。\n')
    return plan


def load_native():
    from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime
    runtime=prepare_installed_habitat_runtime(runtime_prefix=os.environ['AVENGINE_HABITAT_RUNTIME_PREFIX'],magnum_python_site=os.environ['AVENGINE_HABITAT_MAGNUM_PYTHON_SITE'],rlr_sdk_root=os.environ.get('AVENGINE_RLR_SDK_ROOT'),allow_mp3d_environment=False)
    return runtime.habitat_sim


def raw_collision(scene_dir):
    path=Path(scene_dir);sid=path.name.split('-',1)[1];raw=path/(sid+'.glb')
    # Trimesh preserves zero-area input faces; those cannot create a CPU ray hit.
    scene=trimesh.load(raw,force='scene',process=False,skip_materials=True)
    mesh=scene.to_geometry();mesh.vertices=canonical(mesh.vertices)
    mesh.ray=CPUScanRayIntersector(mesh)
    return mesh,dict(mesh.ray.receipt,raw_glb=str(raw),coordinate_transform='(X,Y,Z) glTF -> (X,Z,-Y) Habitat world; scene node transforms applied')


def overhead_for(row,cache):
    key=row['house']+'__'+row['room_label'];cache=Path(cache)
    meta=cache/(key+'.json');png=cache/(key+'.png')
    if not meta.exists() or not png.exists():return None
    from PIL import Image
    entry=json.loads(meta.read_text());im=next((x for x in entry['images'] if x['path'].endswith('_overview.png')),None)
    if im is None:return None
    return entry,im,Image.open(png).convert('RGB'),{'metadata_path':str(meta),'image_path':str(png),'orientation':entry.get('orientation'),'source':'existing orthographic overhead; readonly cache'}



def measured_black(g,floor_y,overhead,p):
    if overhead is None:return {'black_fraction':None,'status':'OVERHEAD_UNAVAILABLE'}
    entry,im,image=overhead[:3]
    clip=np.asarray(im['projection'])@np.asarray(entry['view'])
    if not np.allclose(clip[3],[0,0,0,1]):return {'black_fraction':None,'status':'OVERHEAD_NOT_ORTHOGRAPHIC'}
    A=clip[:2][:,[0,2]];offset=clip[:2,1]*floor_y+clip[:2,3]
    from shapely.geometry import Polygon
    frame=Polygon([np.linalg.solve(A,np.array(q)-offset) for q in ((-1,-1),(1,-1),(1,1),(-1,1))])
    coverage=g.intersection(frame).area/g.area if g.area else 0.0
    if entry.get('coverage_geometry_xz_m') is not None:
        coverage=min(coverage,g.intersection(shape(entry['coverage_geometry_xz_m'])).area/g.area if g.area else 0.0)
    if coverage<1-1e-6:return {'black_fraction':None,'status':'OVERHEAD_FRAME_DOES_NOT_COVER_FLOOR','world_floor_in_frame_fraction':coverage}
    result=black_metric(g,floor_y,entry,im,image,p);result['world_floor_in_frame_fraction']=coverage
    result['metadata_path']=overhead[3]['metadata_path'];result['image_path']=overhead[3]['image_path']
    return result


def nav_scope_at(nav_polys,nav_ys,floor_y,scope,p):
    selected=np.flatnonzero(abs(nav_ys-floor_y)<=p['floor_height_separation_m'])
    return union_projected_polygons([nav_polys[i] for i in selected]).intersection(scope)


def block_type(g,floor_y,objects,markers,row):
    members={}
    for instance in objects:
        if instance['region_id'] not in (row['region_id'],-1):continue
        tri=instance['triangles'];verts=tri.reshape(-1,3)
        if verts[:,1].min()>floor_y+2.5 or verts[:,1].max()<floor_y-0.3:continue
        centre=verts[:,[0,2]].mean(axis=0)
        if g.covers(Point(centre)):members[instance['instance_id']]=instance
    kind,evidence=infer_type(members)
    local=[]
    for marker in markers:
        centre=marker['centre_xz_m'];lo,hi=marker['height_range_m']
        if lo<=floor_y+1.8 and hi>=floor_y-0.3 and g.covers(Point(centre)):local.append(marker['category'])
    anchors={'bed','mattress','sofa','couch','dining table','desk','stove','oven'} & {x['category'] for x in members.values()}
    if not anchors and any(x in ('stairs','stair','staircase','steps','stair step') for x in local):return 'stairs',local
    if not anchors and any(any(t in x for t in ('outdoor','patio','balcony','terrace','yard','grass')) for x in local):return 'outdoor',local
    return kind,evidence


def stair_partition(scope,floor_y,markers,height_separation):
    """Carve only directly annotated stair mesh projections in this floor window."""
    from shapely.geometry import Polygon
    footprints=[];audit=[]
    for marker in markers:
        if 'stair' not in marker['category'] and marker['category']!='steps':continue
        tri=marker['triangles'];y=tri[:,:,1].mean(axis=1)
        chosen=tri[abs(y-floor_y)<=height_separation]
        polys=[Polygon(t[:,[0,2]]) for t in chosen]
        projection=union_projected_polygons(polys).intersection(scope)
        if projection.is_empty or projection.area<=1e-8:continue
        footprints.append(projection)
        audit.append({'semantic_instance_id':marker['instance_id'],'category':marker['category'],'semantic_source':marker['semantic_source'],'floor_y_m':floor_y,'area_m2':float(projection.area),'method':'direct semantic stair triangles in the existing floor-height window; horizontal projection clipped to source ground'})
    if not footprints:return scope,[],audit
    stairs=shapely.union_all(footprints)
    components=list(stairs.geoms) if hasattr(stairs,'geoms') else [stairs]
    return scope.difference(stairs),[g for g in components if g.area>1e-8],audit


def finalize_interfaces(blocks,cuts):
    """Keep applied-cut provenance, but expose only surviving final interfaces."""
    common=[]
    for i,a in enumerate(blocks):
        ga=shape(a['floor_polygon_xz_m'])
        for b in blocks[i+1:]:
            if a['floor_id']!=b['floor_id']:continue
            interface=ga.boundary.intersection(shape(b['floor_polygon_xz_m']).boundary)
            if interface.length>1e-6:common.append((a['floor_id'],interface))
    for cut in cuts:
        line=shape(cut['line_geometry_xz_m']) if cut.get('line_geometry_xz_m') else LineString(cut['line_xz_m']) if cut.get('line_xz_m') else GeometryCollection()
        surviving=shapely.union_all([g.intersection(line.buffer(1e-6)) for fid,g in common if fid==cut['floor_id']])
        cut['active_in_final_partition']=surviving.length>1e-6
        cut['final_interface_geometry_xz_m']=mapping(surviving)
    for block in blocks:
        boundary=shape(block['floor_polygon_xz_m']).boundary
        block['cut_ids']=[c['id'] for c in cuts if c['floor_id']==block['floor_id'] and c['active_in_final_partition'] and boundary.intersection(shape(c['final_interface_geometry_xz_m']).buffer(1e-6)).length>1e-6]


def adjacency(blocks,cuts):
    for b in blocks:b['adjacent_rooms']=[]
    for i,a in enumerate(blocks):
        ga=shape(a['floor_polygon_xz_m'])
        for b in blocks[i+1:]:
            if a['floor_id']!=b['floor_id']:continue
            gb=shape(b['floor_polygon_xz_m'])
            common=ga.boundary.intersection(gb.boundary)
            supporting=[]
            for cut in cuts:
                if cut.get('floor_id')!=a['floor_id'] or not cut.get('line_xz_m'):continue
                line=LineString(cut['line_xz_m'])
                if ga.distance(line)<0.01 and gb.distance(line)<0.01:supporting.append(cut)
            if common.length<1e-6 and not supporting:continue
            ctype='door' if any(c['type']=='door' for c in supporting) else 'narrow' if any(c['type']=='narrow' for c in supporting) else 'visibility'
            width=max([c.get('width_m',0) for c in supporting]+[float(common.length)])
            a['adjacent_rooms'].append({'id':b['id'],'opening_type':ctype,'opening_width_m':width,'cut_ids':[c['id'] for c in supporting]})
            b['adjacent_rooms'].append({'id':a['id'],'opening_type':ctype,'opening_width_m':width,'cut_ids':[c['id'] for c in supporting]})


def process_region(row,parameter,p,mesh,pf,hs,nav_polys,nav_ys,doors,markers,objects,door_info,overhead,cache=None,atlas_output=None):
    result={'house':row['house'],'source_region':row['room_label'],'source_region_id':row['region_id'],'origins':row['origins'],'reference_fold':row.get('reference_fold'),'source_floor_area_m2':row['floor_area_sum_m2'],'requires_split':row['requires_split'],'source_geometry':row,'cut_lines':[],'blocks':[],'door_diagnostics':door_info,'parameter':parameter,'status':'unchanged' if not row['requires_split'] else 'processed','merge_audit':[]}
    index=0
    for floor in row['floors']:
        scope=shape(floor['floor_polygon']);floor_y=floor['floor_y_m'];fid=floor['floor_id']
        floor_rgb=floor_overhead(row,floor,cache,atlas_output,overhead) if cache is not None else overhead
        result.setdefault('floor_overheads',{})[fid]=floor_rgb[3] if floor_rgb else None
        stair_parts=[]
        if row['requires_split'] and (row['ambiguous_semantic_palette'] or door_info['invalid_ground_zero_area_faces_by_region'].get(row['region_id'],0)):
            parts=[scope];cuts=[];audits=[];floor_error='SEMANTIC_PALETTE_AMBIGUOUS' if row['ambiguous_semantic_palette'] else 'ZERO_AREA_SEMANTIC_GROUND_FACES'
        else:
            floor_error=None
            if row['requires_split']:
                room_scope,stair_parts,stair_audit=stair_partition(scope,floor_y,markers,p['floor_height_separation_m'])
                result.setdefault('stair_separation_audit',[]).extend(dict(a,floor_id=fid) for a in stair_audit)
                parts,cuts,audits=structural_partition(room_scope,doors,floor_y,parameter['width_min'],parameter['width_max']) if not room_scope.is_empty else ([],[],[])
            else:parts,cuts,audits=[scope],[],[]
        for c in cuts:c['floor_id']=fid;c['id']=fid+'_'+c['id']
        result['cut_lines'].extend(cuts);result.setdefault('door_instance_audit',[]).extend(audits)
        nav_scope=nav_scope_at(nav_polys,nav_ys,floor_y,scope,p)
        _,all_points,all_clearance,all_adj,_=sample_navigation(pf,hs,scope,floor_y,p)
        final_parts=[(g,None,{'status':'excluded_stairs','minimum_piece_count_verified':False},None) for g in stair_parts]
        for part in parts:
            allowed=np.flatnonzero(shapely.contains_xy(part,all_points[:,0],all_points[:,2]))
            if part.area>50+1e-8 and not floor_error:
                children,witnesses,receipt=partition_visibility(mesh,part,nav_scope.intersection(part),all_points[allowed],p,parameter['coverage'],parameter['max_distance'])
                for k,g in enumerate(children):final_parts.append((g,witnesses[k] if k<len(witnesses) else None,receipt,None))
                if len(children)>1:
                    for i in range(len(children)):
                        for j in range(i+1,len(children)):
                            shared=children[i].boundary.intersection(children[j].boundary)
                            if not shared.is_empty:
                                result['cut_lines'].append({'id':fid+f'_V{len(result["cut_lines"]):03d}','floor_id':fid,'type':'visibility','line_geometry_xz_m':mapping(shared),'line_xz_m':list(shared.coords) if shared.geom_type=='LineString' else None,'geometry_source':'integer-owned exact structural floor intersections; raw-scan CPU visibility','minimum_piece_count_verified':receipt['minimum_piece_count_verified']})
            else:final_parts.append((part,None,None,floor_error))
        # Merge only wide, open contour/visibility interfaces. Door interfaces are protected.
        if row['requires_split'] and not floor_error:
            changed=True
            while changed:
                changed=False
                for i in range(len(final_parts)):
                    if changed:break
                    for j in range(i+1,len(final_parts)):
                        if any(final_parts[k][2] and final_parts[k][2].get('status')=='excluded_stairs' for k in (i,j)):continue
                        a,b=final_parts[i][0],final_parts[j][0]
                        if a.area+b.area>50+1e-8:continue
                        interface=a.boundary.intersection(b.boundary)
                        if interface.length<=1.6:continue
                        supported=shapely.line_merge(interface.intersection(nav_scope))
                        segments=list(supported.geoms) if hasattr(supported,'geoms') else [supported]
                        native_width=max([x.length for x in segments]+[0])+2*float(pf.nav_mesh_settings.agent_radius)
                        if native_width<=1.6:continue
                        if any(c['type']=='door' and a.distance(LineString(c['line_xz_m']))<0.01 and b.distance(LineString(c['line_xz_m']))<0.01 for c in cuts):continue
                        merged=a.union(b);ids=np.flatnonzero(shapely.contains_xy(merged,all_points[:,0],all_points[:,2]))
                        _,_,weights=grid_atoms(merged,nav_scope.intersection(merged),all_points[ids],p['grid_step_m'])
                        v=assess_visibility(mesh,all_points[ids],weights,p,parameter['coverage'],parameter['max_distance'])
                        result['merge_audit'].append({'floor_id':fid,'source_areas_m2':[a.area,b.area],'opening_width_proxy_m':float(interface.length),'native_supported_opening_width_m':native_width,'visibility':v,'merged':v['meets_visibility'],'method':'continuous common boundary; door-protected; wide >1.6 m; visibility verified'})
                        if not v['meets_visibility']:continue
                        final_parts[i]=(merged,v,{'status':'wide_open_merge','minimum_piece_count_verified':False},None);final_parts.pop(j);changed=True;break
        for g,visibility,solver,error in final_parts:
            bid=f"{row['house']}__{row['room_label']}__{fid}__S{index:03d}";index+=1
            allowed=np.flatnonzero(shapely.contains_xy(g,all_points[:,0],all_points[:,2]));nav=g.intersection(nav_scope)
            # All diagnostics are measured; an unchanged original cannot acquire a new rejection.
            if visibility is None:
                _,_,weights=grid_atoms(g,nav,all_points[allowed],p['grid_step_m'])
                visibility=assess_visibility(mesh,all_points[allowed],weights,p,parameter['coverage'],parameter['max_distance'])
            witness=placement(mesh,all_points,all_clearance,all_adj,p,allowed.tolist())
            black=measured_black(g,floor_y,floor_rgb,p)
            kind,evidence=block_type(g,floor_y,objects,markers,row)
            if solver and solver.get('status')=='excluded_stairs':kind,evidence='stairs',['direct semantic stair footprint separated before room cuts']
            width=body_width(g);rect=g.minimum_rotated_rectangle
            longest=max(np.linalg.norm(np.diff(np.array(rect.exterior.coords),axis=0),axis=1)) if rect.geom_type=='Polygon' else 0
            short=short_side(g);aspect=float(longest/short) if short else None
            reasons=[];unknown=[]
            if g.area<6-1e-8:reasons.append('FLOOR_AREA_BELOW_6')
            if g.area>50+1e-8:unknown.append('STILL_ABOVE_50_UNRESOLVED')
            if short<2.4-1e-8:reasons.append('SHORT_SIDE_BELOW_2_4')
            if width<1.5 and aspect is not None and aspect>=3:reasons.append('CORRIDOR_BODY_WIDTH_BELOW_1_5')
            if kind=='stairs':reasons.append('STAIRS')
            if kind=='outdoor':reasons.append('OUTDOOR_OR_BALCONY_SEMANTIC_AND_LOCAL_GEOMETRY')
            if black['black_fraction'] is None:unknown.append('SCAN_BLACK_FRACTION_UNVERIFIED')
            elif black['black_fraction']>0.15:reasons.append('SCAN_BLACK_FRACTION_ABOVE_15_PERCENT')
            if not witness['found']:reasons.append('PLACEMENT_NO_WITNESS_WITHIN_FROZEN_BUDGET')
            if error:unknown.append(error)
            if solver and solver.get('status')=='unresolved':unknown.append(solver['reason'])
            decision='unchanged' if not row['requires_split'] else 'unresolved' if error else 'discard' if reasons else 'unresolved' if unknown else 'retain'
            result['blocks'].append({'schema':'hm3d_auto_room_split_v1','id':bid,'house':row['house'],'source_region_id':row['region_id'],'source_region':row['room_label'],'source_region_origins':row['origins'],'floor_id':fid,'floor_y_m':floor_y,'floor_height_range_m':floor['height_range_m'],'floor_polygon_xz_m':mapping(g),'floor_area_m2':float(g.area),'short_side_m':short,'body_width_m_proxy':width,'aspect_ratio':aspect,'nav_walkable_area_m2':float(nav.area),'nav_grid_point_count':len(allowed),'black_fraction':black['black_fraction'],'black_measurement':black,'visibility_coverage_fraction':visibility['coverage_fraction'],'visibility':visibility,'visibility_partition_solver':solver,'room_type':kind,'type_evidence':evidence,'placement_witness':witness,'decision':decision,'discard_reasons':reasons if decision=='discard' else [],'unresolved_reasons':unknown if decision=='unresolved' else [],'unverified_diagnostics':unknown,'diagnostic_reasons_without_new_rejection':reasons+unknown if decision=='unchanged' else [],'cut_ids':[c['id'] for c in result['cut_lines'] if c['floor_id']==fid and (c.get('line_geometry_xz_m') and g.distance(shape(c['line_geometry_xz_m']))<.01 or c.get('line_xz_m') and g.distance(LineString(c['line_xz_m']))<.01)],'adjacent_rooms':[],'new_room':row['requires_split'],'acoustics':'not_run_per_task','measurement_source':row['semantic_source'],'area_method':row['ground_measurement'],'preservation_reason':'original <=50 m²; existing scope and decision preserved' if not row['requires_split'] else None})
    if not row['floors']:
        result['status']='unresolved';result['unresolved_reasons']=['NO_SEMANTIC_GROUND_FLOOR']
    if any(b['decision']=='unresolved' for b in result['blocks']):result['status']='partially_unresolved'
    result['floor_minimum_certificates']=[]
    if row['requires_split']:
        for floor in row['floors']:
            candidates=[b for b in result['blocks'] if b['floor_id']==floor['floor_id'] and b['room_type']!='stairs']
            area=sum(b['floor_area_m2'] for b in candidates);bound=int(math.ceil(max(0,area-1e-8)/50))
            feasible=bool(candidates) and all(6-1e-8<=b['floor_area_m2']<=50+1e-8 and b['short_side_m']>=2.4-1e-8 and b['visibility']['meets_visibility'] for b in candidates)
            verified=feasible and len(candidates)==bound
            certificate={'floor_id':floor['floor_id'],'room_floor_area_m2_excluding_stairs':area,'piece_count':len(candidates),'universal_area_lower_bound':bound,'minimum_piece_count_verified':verified,'proof':'final feasible construction attains ceil(floor area/50)' if verified else 'global minimum unverified','certification_scope':'geometric floor partition before placement/black/type discards; does not certify retained-room count'}
            result['floor_minimum_certificates'].append(certificate)
            if verified:
                for block in candidates:
                    if block.get('visibility_partition_solver'):
                        block['visibility_partition_solver']=dict(block['visibility_partition_solver'],minimum_piece_count_verified=True,minimum_proof=certificate['proof'],final_floor_certificate=certificate)
    result['retained_new_rooms']=sum(b['decision']=='retain' for b in result['blocks']);result['overhead']=overhead[3] if overhead else None
    finalize_interfaces(result['blocks'],result['cut_lines'])
    adjacency(result['blocks'],[c for c in result['cut_lines'] if c.get('active_in_final_partition',True)])
    result['area_partition_error_m2']=abs(sum(b['floor_area_m2'] for b in result['blocks'])-row['floor_area_sum_m2'])
    return result


def worker(job,parameters,p,cache,outdir,scope,memory_gib=12):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_AS,(memory_gib*1024**3,memory_gib*1024**3))
    started=time.time();outdir=Path(outdir)
    rows=[r for r in job['rows'] if scope=='all' or r.get('reference_fold')==scope or (scope=='reference' and 'smy_hand_cut_33' in r['origins'])]
    if scope in ('tune','holdout'):rows=[r for r in rows if 'smy_hand_cut_33' in r['origins']]
    if not rows:return {'house':job['house'],'status':'no_rows'}
    previous=job.get('precomputed_reference_regions',{})
    if rows and all(r['room_label'] in previous for r in rows):
        for param in parameters:dump(outdir/param['id']/(job['house']+'.json'),{'house':job['house'],'status':'reused_reference','parameter':param,'regions':[previous[r['room_label']] for r in rows],'reference_run_reused':job.get('reference_run_reused'),'code_commit':job.get('split_code_commit'),'native_navmesh':'native witnesses reused from the recorded reference batch; not executed again'})
        return {'house':job['house'],'status':'reused_reference','regions':len(rows),'seconds':time.time()-started,'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    try:
        hs=load_native();pf,nav_polys,nav_ys=navmesh_triangles(hs,Path(job['navmesh']))
        mesh,ray_receipt=raw_collision(job['scene_directory'])
        structural,door_info=structural_instances(job['scene_directory'])
        scene=load_scene(job['scene_directory']);objects=scene.instances
        doors=[d for d in structural if 'door' in d['category']];markers=[d for d in structural if 'door' not in d['category']]
        for param in parameters:
            results=[]
            for row in rows:
                try:
                    result=previous[row['room_label']] if row['room_label'] in previous else process_region(row,param,p,mesh,pf,hs,nav_polys,nav_ys,doors,markers,objects,door_info,overhead_for(row,cache),cache,outdir.parent/'overhead_cpu_atlas_v1')
                except Exception as e:result={'house':row['house'],'source_region':row['room_label'],'source_geometry':row,'origins':row['origins'],'requires_split':row['requires_split'],'source_floor_area_m2':row['floor_area_sum_m2'],'status':'unresolved','unresolved_reasons':[type(e).__name__+': '+str(e)],'traceback':traceback.format_exc(),'blocks':[],'cut_lines':[]}
                results.append(result)
            dump(outdir/param['id']/(job['house']+'.json'),{'house':job['house'],'status':'processed','parameter':param,'regions':results,'ray_receipt':ray_receipt,'code_commit':job.get('split_code_commit'),'native_navmesh':'PathFinder only; no Simulator or rendering context initialized','elapsed_s':time.time()-started})
        return {'house':job['house'],'status':'processed','regions':len(rows),'seconds':time.time()-started,'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    except Exception as e:
        failure={'house':job['house'],'status':'unresolved','error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc(),'regions':[{'house':r['house'],'source_region':r['room_label'],'source_geometry':r,'origins':r['origins'],'requires_split':r['requires_split'],'source_floor_area_m2':r['floor_area_sum_m2'],'status':'unresolved','unresolved_reasons':['HOUSE_NATIVE_OR_GEOMETRY_FAILURE'],'blocks':[],'cut_lines':[]} for r in rows]}
        for param in parameters:dump(outdir/param['id']/(job['house']+'.json'),failure)
        return {k:v for k,v in failure.items() if k not in ('regions','traceback')}


def run(root,scope,outname,workers=8,reuse_reference_run=None):
    resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
    root=Path(root);plan=json.loads((root/'processing_plan_v1.json').read_text());outdir=root/outname;outdir.mkdir(exist_ok=False)
    if scope=='tune':parameters=json.loads((root/'parameter_search_plan_v1.json').read_text())['grid']
    else:parameters=[json.loads((root/'selected_parameter_v1.json').read_text())['selected_parameter']]
    jobs=[j for j in plan['jobs'] if any('smy_hand_cut_33' in r['origins'] and (scope=='reference' or r.get('reference_fold')==scope) for r in j['rows'])] if scope!='all' else plan['jobs']
    if reuse_reference_run:
        if scope!='all' or Path(reuse_reference_run).name!=reuse_reference_run:raise ValueError('Reference reuse requires an output-root run name and all scope')
        previous={}
        for path in (root/reuse_reference_run/parameters[0]['id']).glob('*.json'):
            for region in json.loads(path.read_text()).get('regions',[]):previous[(region['house'],region['source_region'])]=region
        for job in jobs:
            reused={}
            for row in job['rows']:
                if 'smy_hand_cut_33' not in row['origins']:continue
                key=(row['house'],row['room_label'])
                if key not in previous:raise ValueError('Missing reusable reference: '+str(key))
                region=previous[key]
                if region['source_geometry']['region_id']!=row['region_id'] or abs(region['source_floor_area_m2']-row['floor_area_sum_m2'])>1e-8:raise ValueError('Reference source geometry identity differs: '+str(key))
                reused[row['room_label']]=region
            job['precomputed_reference_regions']=reused;job['reference_run_reused']=reuse_reference_run
    code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for job in jobs:job['split_code_commit']=code_commit
    memory_gib=6
    receipt={'started_at_utc':now(),'code_commit':code_commit,'pid':os.getpid(),'machine':os.uname().nodename,'scope':scope,'reference_run_reused':reuse_reference_run,'workers':workers,'cpu_only':True,'numpy_madvise_hugepage':os.environ.get('NUMPY_MADVISE_HUGEPAGE'),'per_worker_address_space_limit_gib':memory_gib,'parent_address_space_limit_gib':12,'tree_address_space_upper_bound_gib':12+workers*memory_gib,'parameters':parameters,'output':str(outdir),'results':[]}
    dump(root/(outname+'_start.json'),receipt)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(worker,j,parameters,plan['parameters'],plan['overhead_cache'],outdir,scope,memory_gib):j for j in jobs}
        for future in as_completed(fs):
            job=fs[future]
            try:r=future.result()
            except Exception as e:
                r={'house':job['house'],'status':'unresolved','error':type(e).__name__+': '+str(e),'reason':'OWNED_CPU_WORKER_FAILURE'}
                selected=[row for row in job['rows'] if scope=='all' or 'smy_hand_cut_33' in row['origins'] and (scope=='reference' or row.get('reference_fold')==scope)]
                failure=dict(r,code_commit=code_commit,regions=[{'house':row['house'],'source_region':row['room_label'],'source_geometry':row,'origins':row['origins'],'requires_split':row['requires_split'],'source_floor_area_m2':row['floor_area_sum_m2'],'status':'unresolved','unresolved_reasons':['OWNED_CPU_WORKER_FAILURE'],'blocks':[],'cut_lines':[]} for row in selected])
                for parameter in parameters:
                    path=outdir/parameter['id']/(job['house']+'.json')
                    if not path.exists():dump(path,failure)
            receipt['results'].append(r);print('HOUSE',json.dumps(r,ensure_ascii=False),flush=True)
    receipt['finished_at_utc']=now();dump(root/(outname+'_complete.json'),receipt)
    return receipt
