"""Apply HM3D v6 shape construction and frozen CPU placement to all CAD rooms."""
from __future__ import annotations
import argparse, math, os, resource, time, traceback, multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import numpy as np
import shapely, trimesh, yaml
from shapely.geometry import shape, mapping, Point, LineString, GeometryCollection
from shapely.ops import nearest_points
from tools.rooms.room_split_kujiale.adapter import read, dump
from tools.rooms.room_split_kujiale.resources import Resources
from tools.rooms.room_selection.media import polygons
from tools.rooms.room_selection.geometry import short_side
from tools.rooms.room_selection.navigation import sample_navigation, placement, ray_clear_batch
from tools.rooms.room_selection.measurements import navigation_geometry, navmesh_triangles
from tools.rooms.room_split_auto import capped_split as cap, connected_split as cs
from tools.rooms.room_split_auto.cpu_rays import CPUScanRayIntersector
from tools.rooms.room_split_auto.pipeline import load_native, finalize_interfaces, adjacency


def v6_api():
    # Formal v6 must be merged in this worktree before running this pipeline.
    from tools.rooms.room_split_auto import shape_quality_geometry as quality
    from tools.rooms.room_split_auto.shape_quality_repair import local_wrap_repair
    return quality, local_wrap_repair


def raw_disk(g):
    core=g.buffer(-1.2)
    pts=[p.representative_point() for p in polygons(core)] + [g.centroid]
    if not pts or max((g.boundary.distance(p) for p in pts if g.covers(p)), default=0)<1.2-1e-7:
        line=shapely.maximum_inscribed_circle(g,tolerance=.002)
        if not line.is_empty:pts.append(Point(line.coords[0]))
    good=[(g.boundary.distance(p),p) for p in pts if g.covers(p)]
    radius,pt=max(good,key=lambda t:t[0]) if good else (0.,None)
    return dict(fits=radius>=1.2-1e-7,diameter_m=2.4,radius_m=float(radius),
                centre_xz_m=[pt.x,pt.y] if pt is not None else None,
                shape_basis='exact delivered real polygon, including holes; no navigation enlargement')


def make_repair(furniture,axis):
    quality,_=v6_api()
    class StrictRepair(quality.Repair):
        phase = 'axis'

        def primitive_lines(self,g,angles=None,near_line=None):
            axial=all(min(abs((a-self.axis)%90),90-abs((a-self.axis)%90))<1e-6 for a in (angles or [self.axis,self.axis+90]))
            if (self.phase=='axis' and axial) or (self.phase=='tilt' and not axial):
                yield from super().primitive_lines(g,angles,near_line)

        def bent_lines(self,g,near_line=None):
            if self.phase=='two_leg':
                # The owner permits a two-leg fallback, rather than introducing
                # an extra turn merely because the shared v6 generator has one.
                for line in super().bent_lines(g,near_line):
                    if len(line.coords)-1<=2:yield line

        def choices(self,g,cause=None,near_line=None,neighbors=(),fallback=True):
            answer=[]
            phases=('axis','tilt','two_leg') if fallback else ('axis',)
            for phase in phases:
                self.phase=phase
                candidates=super().choices(g,cause,near_line,neighbors,phase!='axis')
                current=[]
                for score,line,_ in candidates:
                    m=cs.cut_measure(line,g,self.furniture)
                    if m['furniture_intersection_length_m']>.5:continue
                    cells=quality.raw_cells(g,line)
                    large=[q for q in cells if self.classify(q,cause) is None]
                    if not large or any(short_side(q)<2.4-1e-7 or not raw_disk(q)['fits'] for q in large):continue
                    actual=list(score);actual[2]=False;actual[6]=m['furniture_intersection_length_m']
                    current.append((tuple(actual),line,m))
                current.sort(key=lambda x:x[0])
                answer+=current
                if current and current[0][0][:4]==(0,0,False,0.):break
            self.phase='axis'
            if answer or near_line is not None or not fallback:return sorted(answer,key=lambda x:x[0])
            # Reuse the older orthogonal generator if v6's bounded bends fail.
            from tools.rooms.room_split_auto.walkable_split import two_leg_candidates
            mode='corridor' if cause=='CORRIDOR' else 'connectivity' if cause=='NECK' else 'rooms'
            for _,line,children,m in two_leg_candidates(g,self.furniture,self.cap,self.axis,mode):
                m=cs.cut_measure(line,g,self.furniture)
                if m['furniture_intersection_length_m']>.5:continue
                cells=quality.raw_cells(g,line)
                if len(cells)!=2:continue
                large=[q for q in cells if self.classify(q,cause) is None]
                if not large or any(short_side(q)<2.4-1e-7 or not raw_disk(q)['fits'] for q in large):continue
                bad=sum(quality.defects(q)['neck_count']+quality.defects(q)['corridor_count'] for q in large)
                wraps=quality.wrap_count(large+list(neighbors))
                over=sum(max(0,math.ceil((q.area-1e-7)/self.cap)-1) for q in large)
                score=(wraps,bad,False,0.,sum(q.area for q in cells if q not in large),over,
                       m['furniture_intersection_length_m'],abs(cells[0].area-cells[1].area),2)
                answer.append((score,line,m))
            return sorted(answer,key=lambda x:x[0])

    return StrictRepair(furniture,axis,max_nodes=48)


def door_partition(g,markers,axis,fy,furniture):
    quality,_=v6_api()
    from tools.rooms.room_split_auto.contours import chord, split_at_chord
    leaves=[g];cuts=[];audit=[]
    for door in sorted((m for m in markers if 'door' in m['category']),key=lambda m:m['instance_id']):
        row={k:v for k,v in door.items() if k not in ('geometry','triangles')}
        applied=False;row['status']='outside_shape_or_no_feasible_room_cut'
        if door['height_range_m'][0]>fy+1.8 or door['height_range_m'][1]<fy+.3:
            row['status']='other_floor';audit.append(row);continue
        for i,p in enumerate(leaves):
            hit=chord(quality.own_outline(p),door['centre_xz_m'],[axis,axis+90])
            if hit is None or hit['width_m']>max(door['extent_xz_m'])*1.7+.4:continue
            children=split_at_chord(p,hit['line'],6.)
            if not children or any(short_side(q)<2.4-1e-7 or not raw_disk(q)['fits'] for q in children):continue
            m=cs.cut_measure(hit['line'],p,furniture)
            if m['furniture_intersection_length_m']>.5:
                row['status']='furniture_crossing_exceeds_0p5m';continue
            leaves[i:i+1]=children;cuts.append((hit['line'],'DOOR',m,p));applied=True
            row.update(status='cut_applied',opening_chord_width_m=hit['width_m']);break
        audit.append(row)
    return leaves,cuts,audit


def connectivity_check(g,pf,hs,points,ctx,forbidden):
    """Confirm far-part paths stay in this room and only its pair bridges."""
    raw=sorted(polygons(g),key=lambda p:-p.area)
    if not raw:return dict(pass_=False,reason='EMPTY_GEOMETRY')
    main=[raw[0]];far=raw[1:];changed=True
    while changed:
        changed=False;blob=shapely.union_all(main)
        for p in list(far):
            if p.distance(blob)<=.05+1e-9:main.append(p);far.remove(p);changed=True
    blob=shapely.union_all(main);paths=[]
    links=ctx.record(g)['links']
    bridges=shapely.union_all([shape(x['bridge_geometry_xz_m']) for x in links])
    allowed=g.union(bridges).buffer(.05)
    all_ok=True
    for part in far:
        if part.area<1e-8:continue
        ids_a=np.flatnonzero(shapely.contains_xy(part,points[:,0],points[:,2]))
        ids_b=np.flatnonzero(shapely.contains_xy(blob,points[:,0],points[:,2]))
        a,b=nearest_points(part,blob)
        ids_a=sorted(ids_a,key=lambda i:Point(points[i,0],points[i,2]).distance(b))[:6]
        ids_b=sorted(ids_b,key=lambda i:Point(points[i,0],points[i,2]).distance(a))[:6]
        best=None
        for i in ids_a:
            for j in ids_b:
                sp=hs.ShortestPath();sp.requested_start=points[i];sp.requested_end=points[j]
                if not pf.find_path(sp) or not math.isfinite(sp.geodesic_distance):continue
                route=LineString(np.asarray(sp.points)[:,[0,2]])
                eu=float(np.linalg.norm(points[i,[0,2]]-points[j,[0,2]]))
                detour=float(sp.geodesic_distance)-eu
                escaped=route.difference(allowed).length
                other=route.intersection(forbidden).length if not forbidden.is_empty else 0.
                score=(escaped>1e-5,other>1e-5,detour)
                rec=dict(part_area_m2=float(part.area),gap_m=float(part.distance(blob)),
                         geodesic_m=float(sp.geodesic_distance),straight_m=eu,detour_m=detour,
                         route_outside_own_shape_and_pair_bridges_m=float(escaped),
                         route_in_other_room_floor_m=float(other),path_xz_m=list(map(list,route.coords)))
                if best is None or score<best[0]:best=(score,rec)
        ok=best is not None and not best[0][0] and not best[0][1] and best[0][2]<=1.+1e-8
        paths.append(dict(best[1] if best else dict(part_area_m2=float(part.area),reason='NO_DIRECT_SUPPORTED_PATH'),pass_=ok))
        all_ok &= ok
    return dict(pass_=bool(all_ok),raw_components=len(raw),seam_merged_parts=len(main),far_paths=paths,
                bridge_policy='only original distinct part pair overlap, clipped to navmesh and excluding all other CAD room floor; own narrow neck never widened',
                certificate=ctx.certificate(g))


def witness_check(witness,g,mesh,pf,p):
    if not witness.get('found'):return witness
    roles=[('camera_m',p['camera_height_m']),('source_1_m',p['source_height_m']),('source_2_m',p['source_height_m'])]
    checks=[];ok=True
    for role,height in roles:
        point=np.asarray(witness[role],float);support=point-[0,height,0]
        snapped=np.asarray(pf.snap_point(support),float)
        inside=bool(g.contains(Point(point[0],point[2])))
        nav=bool(np.isfinite(snapped).all() and np.linalg.norm(snapped-support)<1e-4)
        ok &= inside and nav
        checks.append(dict(role=role,inside_real_polygon=inside,navmesh_supported=nav,
                           support_m=support.tolist(),snap_error_m=float(np.linalg.norm(snapped-support)) if np.isfinite(snapped).all() else None))
    a,b,c=[witness[k] for k in ('camera_m','source_1_m','source_2_m')]
    clear=[bool(ray_clear_batch(mesh,a,[b],p['ray_endpoint_tolerance_m'])[0]),
           bool(ray_clear_batch(mesh,a,[c],p['ray_endpoint_tolerance_m'])[0]),
           bool(ray_clear_batch(mesh,b,[c],p['ray_endpoint_tolerance_m'])[0])]
    ok &= all(clear)
    witness.update(found=bool(ok),validation=dict(status='pass' if ok else 'fail',support_checks=checks,
        three_raw_mesh_segment_rays_clear=clear,ray_endpoint_tolerance_m=p['ray_endpoint_tolerance_m'],
        collision_source='unchanged unfiltered existing surface/vertices.npy and triangles.npy, verified against original USD',
        simulated_renderer_created=False))
    return witness


def source_region(row,h,scene_adapter,mesh,pf,hs,nav_polys,nav_ys,p):
    quality,wrap_repair=v6_api()
    g=shape(row['floor_polygon_xz_m']);fy=row['floor_y_m'];fid=row['floor_id'];axis=row['wall_axes']['primary_deg']
    excluded=row['room_type'] in ('unknown','outdoor','storage','garage','corridor','stairs')
    furniture=cs.FurnitureList([dict(x,geometry=shape(x['geometry'])) for x in scene_adapter['furniture_by_height'][str(fy)]])
    other=shapely.union_all([shape(r['floor_polygon_xz_m']) for r in h['rooms'] if r['room_label']!=row['room_label'] and abs(r['floor_y_m']-fy)<=.3])
    nav=shapely.union_all([np_ for np_,y in zip(nav_polys,nav_ys) if abs(y-fy)<=.3])
    ctx=quality.PartConnectivity(nav,other)
    dd=quality.defects(g)
    stair_record=scene_adapter['stair_masks_by_region'][row['room_label']]
    stair_parts=[shape(x) for x in stair_record['stair_parts']]
    terrain=g.difference(shapely.union_all(stair_parts))
    terrain_defects=quality.defects(terrain) if not terrain.is_empty else dict(neck_count=0,corridor_count=0)
    needs_repair=row['requires_split'] or bool(terrain_defects['neck_count'] or terrain_defects['corridor_count'])
    leaves=[];cuts=[];door_audit=[];repair=make_repair(furniture,axis)
    if excluded:
        leaves=[dict(g=g,forced='NON_RESIDENTIAL_OR_TRANSIT_OR_UNKNOWN_TYPE',error=None)]
    elif g.area<6-1e-8 or short_side(g)<2.4-1e-7:
        # Do not re-cut an ordinary room to evade its frozen short-side threshold.
        leaves=[dict(g=g,forced='FLOOR_AREA_BELOW_6' if g.area<6-1e-8 else 'SHORT_SIDE_BELOW_2_4',error=None)]
    else:
        leaves.extend(dict(g=q,forced='STAIRS',error=None) for q in stair_parts)
        groups=ctx.groups(terrain)
        for component in groups:
            if len(groups)>1 and component.area<6-1e-8:
                leaves.append(dict(g=component,forced='DETACHED_FRAGMENT',error=None));continue
            parts=[component]
            if needs_repair:
                parts,doorcuts,da=door_partition(component,scene_adapter['markers'],axis,fy,furniture)
                cuts+=doorcuts;door_audit+=da
            for part in parts:
                if needs_repair:
                    repair.nodes=0;ll,cc=repair.solve(part);leaves+=ll;cuts+=cc
                else:leaves.append(dict(g=part,forced=None,error=None))
        if needs_repair:
            repair.nodes=0;leaves,cc,wrap_audit=wrap_repair(leaves,repair);cuts+=cc
        else:wrap_audit=[]
    result=dict(schema='kujiale_auto_room_split_v1',house=h['house'],source_region=row['room_label'],
                source_region_id=int(row['room_label'][1:]),requires_split=row['requires_split'],
                shape_repair_applied=needs_repair and not excluded,source_floor_area_m2=float(g.area),
                cad_area_m2=row['cad_area_m2'],cad_without_measured_floor_m2=row['cad_without_measured_floor_m2'],
                source_geometry=row,cut_lines=[],blocks=[],door_instance_audit=door_audit,
                stair_partition_audit=stair_record,wall_axes={fid:row['wall_axes']},navmesh_source=h['navmesh'],matrix_source=h['matrix_source'],
                max_room_area_m2=35.,license='noncommercial research only; no redistribution',
                acoustics='not_run_per_task',production_modified=False)
    for line,kind,m,parent in cuts:
        result['cut_lines'].append(dict(id=h['house']+'__'+row['room_label']+'__C%03d'%len(result['cut_lines']),floor_id=fid,
            type='door' if kind=='DOOR' else 'narrow' if kind in ('NECK','CORRIDOR') else 'shape',
            stage='door_first' if kind=='DOOR' else 'own_outline_'+kind.lower()+'_shape_repair',
            line_xz_m=list(map(list,line.coords)),line_geometry_xz_m=mapping(line),segment_count=len(line.coords)-1,
            wall_axis_deg=axis,segment_angle_errors_deg=cap.angle_errors(line,axis),
            angle_fallback_deg=max(cap.angle_errors(line,axis),default=0.),applied_parent_geometry_xz_m=mapping(parent),**m))
    for i,item in enumerate(leaves):
        q=item['g'];reasons=[];pending=[];sampled={};navigation={};check={};circle=raw_disk(q) if q.area>=6 else dict(fits=False)
        if item.get('forced'):reasons.append(item['forced'])
        if q.area<6-1e-8:reasons.append('FLOOR_AREA_BELOW_6')
        if short_side(q)<2.4-1e-7:reasons.append('SHORT_SIDE_BELOW_2_4')
        if q.area>35+1e-8:pending.append('ABOVE_35_UNRESOLVED')
        if item.get('error'):pending.append(item['error'])
        defects=quality.defects(q)
        if not reasons and (defects['neck_count'] or defects['corridor_count']):pending.append('OWN_OUTLINE_SHAPE_UNRESOLVED')
        new_shape=len(leaves)!=1 or q.symmetric_difference(g).area>1e-6
        if not reasons and new_shape and not circle['fits']:pending.append('NO_2_4M_DISK_IN_REAL_SHAPE')
        witness=dict(found=False,not_run_reason='discard_or_unresolved_geometry',acoustics='not_run')
        if not reasons and not pending:
            sampled,points,clearance,adj,comps=sample_navigation(pf,hs,q,fy,p)
            navigation=navigation_geometry(nav_polys,nav_ys,fy,q,p,shape(sampled['nav_grid_main_polygon']))
            limit=p['bathroom_nav_main_area_min_m2'] if row['room_type']=='bathroom' else p['nav_main_area_min_m2']
            if navigation['nav_main_area_m2']<limit-1e-8:reasons.append('NAV_MAIN_AREA_BELOW_FROZEN_'+str(limit))
            dominant=row['legacy_measurement'].get('dominant_floor_area_fraction')
            if dominant is not None and dominant<p['dominant_floor_area_fraction_min']-1e-8:reasons.append('DOMINANT_FLOOR_BELOW_FROZEN_THRESHOLD')
            check=connectivity_check(q,pf,hs,points,ctx,other)
            if not check['pass_']:pending.append('DIRECT_NAV_CONNECTIVITY_UNRESOLVED')
            witness=witness_check(placement(mesh,points,clearance,adj,p),q,mesh,pf,p)
            if not witness['found']:reasons.append('PLACEMENT_NO_WITNESS_WITHIN_FROZEN_BUDGET')
        decision='discard' if reasons else 'unresolved' if pending else 'retain'
        bid=h['house']+'__'+row['room_label']+'__'+fid+'__K%03d'%i
        result['blocks'].append(dict(schema='kujiale_split_room_v1',id=bid,house=h['house'],source_region=row['room_label'],source_region_id=int(row['room_label'][1:]),
             floor_id=fid,floor_y_m=fy,floor_polygon_xz_m=mapping(q),floor_area_m2=float(q.area),short_side_m=short_side(q),
             decision=decision,placement_witness=witness,room_type=row['room_type'],source_room_type=row['source_room_type'],
             discard_reasons=sorted(set(reasons)) if decision=='discard' else [],unresolved_reasons=sorted(set(pending)) if decision=='unresolved' else [],
             unverified_diagnostics=sorted(set(pending)),new_room=new_shape,old_production_admitted=row['old_production_admitted'],
             inscribed_circle=circle,shape_quality=dict(neck_count=defects['neck_count'],corridor_count=defects['corridor_count']),
             navmesh_source=h['navmesh'],navigation_metrics=navigation,nav_grid_metrics=sampled,connectivity=check,
             black_fraction=None,black_measurement=dict(status='not_applicable',reason='CAD authored USD; scan black-area incompleteness threshold does not apply'),
             frozen_thresholds_source=p['_source'],dominant_floor_area_fraction=row['legacy_measurement'].get('dominant_floor_area_fraction'),
             visibility_admission_gate=False,visibility=dict(status='not_required',reason='no new visibility-only subcap admission gate'),
             acoustics='not_run_per_task',leakage=None,leakage_eligibility='pending_other_task',source_stage=h['source_stage'],matrix_source=h['matrix_source']))
    retained=[shape(b['floor_polygon_xz_m']) for b in result['blocks'] if b['decision']=='retain']
    if quality.wrap_count(retained):
        for b in result['blocks']:
            if b['decision']=='retain':b.update(decision='unresolved',unresolved_reasons=['WRAP_REPAIR_UNRESOLVED'])
    finalize_interfaces(result['blocks'],result['cut_lines']);adjacency(result['blocks'],result['cut_lines'])
    # A blocked candidate must never enter the list through an active bad design.
    badcuts=[c for c in result['cut_lines'] if c.get('active_in_final_partition',True) and (c['segment_count']>3 or c['furniture_intersection_length_m']>.5)]
    if badcuts:
        for b in result['blocks']:
            if b['decision']=='retain':b.update(decision='unresolved',unresolved_reasons=['CUT_SEGMENTS_OR_FURNITURE_UNRESOLVED'])
    result.update(retained_new_rooms=sum(b['decision']=='retain' for b in result['blocks']),
                  status='partially_unresolved' if any(b['decision']=='unresolved' for b in result['blocks']) else 'processed',
                  area_partition_error_m2=abs(sum(b['floor_area_m2'] for b in result['blocks'])-g.area))
    return result


def worker(h,out,p):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(20*1024**3,20*1024**3))
    started=time.time();out=Path(out);hs=load_native();pf,navp,navy=navmesh_triangles(hs,Path(h['navmesh']))
    adapter=read(out/'scene_adapters_v1'/(h['house']+'.json'))
    adapter['stair_masks_by_region']=read(out/'scene_stairs_v1'/(h['house']+'.json'))['rooms']
    if adapter['existing_surface_parity']['fraction_within_1mm']<.999:raise ValueError('Unaligned original USD surface')
    vertices=np.load(Path(h['build'])/'surface/vertices.npy',mmap_mode='r');faces=np.load(Path(h['build'])/'surface/triangles.npy',mmap_mode='r')
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False);mesh.ray=CPUScanRayIntersector(mesh)
    receipts=[]
    for row in h['rooms']:
        t=time.time()
        try:region=source_region(row,h,adapter,mesh,pf,hs,navp,navy,p)
        except Exception as e:
            region=dict(house=h['house'],source_region=row['room_label'],requires_split=row['requires_split'],source_floor_area_m2=row['floor_area_m2'],source_geometry=row,
                        status='unresolved',cut_lines=[],retained_new_rooms=0,blocks=[dict(id=h['house']+'__'+row['room_label']+'__F0__K000',house=h['house'],source_region=row['room_label'],
                        floor_id='F0',floor_y_m=row['floor_y_m'],floor_polygon_xz_m=row['floor_polygon_xz_m'],floor_area_m2=row['floor_area_m2'],short_side_m=row['short_side_m'],
                        room_type=row['room_type'],decision='unresolved',placement_witness=dict(found=False,not_run_reason=repr(e)),unresolved_reasons=['PROCESS_FAILURE',repr(e)],discard_reasons=[])],
                        error=repr(e),traceback=traceback.format_exc())
        dump(out/'kujiale_delivery_v1/final_v1/regions'/(h['house']+'__'+row['room_label']+'.json'),region)
        for b in region['blocks']:dump(out/'kujiale_delivery_v1/final_v1/rooms'/(b['id']+'.json'),b)
        receipts.append(dict(source_region=row['room_label'],seconds=time.time()-t,status=region['status'],retained=region['retained_new_rooms'],error=region.get('error')))
        print('ROOM',h['house'],row['room_label'],region['status'],'kept',region['retained_new_rooms'],'seconds',round(time.time()-t,2),flush=True)
    receipt=dict(house=h['house'],pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),seconds=time.time()-started,
                 peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,AS_limit_gib=20,rooms=receipts,ray_receipt=mesh.ray.receipt)
    dump(out/'evidence'/('selection_'+h['house']+'.json'),receipt);return receipt


def run(root,workers=4):
    v6_api();out=Path(root);plan=read(out/'input_plan_v1.json');p={k:x['value'] for k,x in yaml.safe_load(Path(plan['thresholds_source']).read_text())['parameters'].items()};p['_source']=plan['thresholds_source']
    if not 1<=workers<=4:raise ValueError('At most four workers; total AS at most 90GiB')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,20*1024**3))
    aligned=read(out/'evidence/coordinate_alignment_v1.json')
    if any(x['status']!='aligned_numeric' for x in aligned['houses']):raise ValueError('Coordinate mismatch: stop, do not guess')
    receipts=[];start=time.time()
    with Resources() as resources:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
            jobs={pool.submit(worker,h,out,p):h['house'] for h in plan['houses']}
            for f in as_completed(jobs):
                try:receipts.append(f.result())
                except Exception as e:receipts.append(dict(house=jobs[f],status='house_failed',error=repr(e),traceback=traceback.format_exc()))
    resources.save(out/'evidence/selection_resources_v1.json')
    dump(out/'evidence/selection_complete_v1.json',dict(receipts=receipts,seconds=time.time()-start,workers=workers,max_combined_AS_gib=20*workers+10,CPU_only=True,parent_pid=os.getpid()))
    if any(x.get('status')=='house_failed' for x in receipts):raise RuntimeError('House worker failed; delivery is incomplete')


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',required=True);a.add_argument('--workers',type=int,default=4);v=a.parse_args();run(v.root,v.workers)
