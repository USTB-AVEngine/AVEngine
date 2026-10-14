#!/usr/bin/env python3
"""Check measured native witness capsules against actual UE render LOD0 geometry."""
from __future__ import annotations
import argparse,datetime,json,math,time
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def capsule_rays(center,radius,half_height):
    """Sample retained native capsule tube and end caps; never deform geometry."""
    if not (0<radius<=half_height):raise ValueError('invalid native capsule dimensions')
    center=np.asarray(center,float);origins=[];directions=[]
    axis=half_height-radius
    horizontal=[np.array([math.cos(a),0.,math.sin(a)]) for a in np.linspace(0,2*math.pi,16,endpoint=False)]
    for y in (-axis,0.,axis):
        for direction in horizontal:origins.append(center+[0.,y,0.]);directions.append(direction)
    for sign in (-1,1):
        origin=center+[0.,sign*axis,0.]
        for elevation in (math.pi/6,math.pi/3):
            for d in horizontal[::2]:origins.append(origin);directions.append(d*math.cos(elevation)+[0.,sign*math.sin(elevation),0.])
        origins.append(origin);directions.append([0.,sign,0.])
    return np.asarray(origins),np.asarray(directions)


def query_capsule(mesh,probe,tolerance_m=.02):
    capsule=probe['capsule'];radius=float(capsule['radius_cm'])/100.;half=float(capsule['half_height_cm'])/100.
    raw=capsule['center_ue_cm'];center=[raw[0]/100.,raw[2]/100.,raw[1]/100.]
    origins,directions=capsule_rays(center,radius,half)
    points,rays,triangles=mesh.ray.intersects_location(origins,directions,multiple_hits=False)
    lengths=np.linalg.norm(points-origins[rays],axis=1)
    interior=lengths<radius-tolerance_m
    bad=[{'ray_index':int(ray),'triangle_index':int(tri),'distance_m':float(distance),'penetration_m':radius-float(distance),'world_hit_m':point.tolist()} for ray,tri,distance,point in zip(rays[interior],triangles[interior],lengths[interior],points[interior],strict=True)]
    return {'status':'fail' if bad else 'pass','witness_index':probe['witness_index'],'role':probe['role'],'center_m':center,'native_radius_m':radius,'native_half_height_m':half,'sampled_rays':len(origins),'retained_contact_tolerance_m':tolerance_m,'interior_intersections':bad}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args()
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    o=a.root.resolve();r=o/'ue_round6';a.output.mkdir(parents=True,exist_ok=True)
    houses=[x['house'] for x in read(o/'acoustic_pass_train_candidates.json')['houses']];records=[];started=time.monotonic()
    for house in houses:
        completed=a.output/(house+'.json')
        if completed.exists():records.append(read(completed));continue
        d=r/'houses'/house
        recovered=r/'recovered_houses'/(house+'_fresh_v2')
        if (recovered/'result.json').exists() and read(recovered/'result.json')['status']=='completed':d=recovered
        if not (d/'result.json').exists() or read(d/'result.json')['status']!='completed':
            if a.allow_incomplete:continue
            raise ValueError('native room QA incomplete: '+house)
        qa=read(d/'qa/result.json');audit=read(d/'native_editor_audit.json');geom=audit['native_render_geometry']
        raw=np.fromfile(geom['world_vertices_ue_cm_f32'],dtype='<f4').reshape(-1,3).astype(float)
        vertices=raw[:,[0,2,1]]/100.;faces=np.fromfile(geom['triangles_u32'],dtype='<u4').reshape(-1,3)
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False);mesh._cache['ray']=RayMeshIntersector(mesh)
        # trimesh exposes the Embree intersector through its lazy ray property.
        intersector=RayMeshIntersector(mesh);mesh.ray=intersector
        rows=[]
        for room in qa['rooms']:
            floor=room.get('floor_reference',{});probes=floor.get('probes',[]);checks=[]
            for probe in probes:
                if probe.get('status')=='measured' and probe.get('capsule',{}).get('status')=='pass':checks.append(query_capsule(mesh,probe))
            measured=[p.get('native_visual_floor_y_m') for p in probes if p.get('status')=='measured']
            row={'room_id':room['room_id'],'native_room_qa_status':room['automated_room_status'],'capsule_checks':checks,'status':'pass' if len(checks)>=9 and all(c['status']=='pass' for c in checks) else 'fail_or_incomplete','failures':[]}
            if len(checks)<9:row['failures'].append('fewer_than_nine_native_floor_capsule_probes')
            if any(c['status']=='fail' for c in checks):row['failures'].append('native_capsule_intersects_static_render_geometry')
            row['native_visual_floor_range_m']=[min(measured),max(measured)] if measured else None
            row['production_median_offset_support_errors_m']=[p['nav_point_m'][1]+floor['native_ue_floor_offset_m']-p['native_visual_floor_y_m'] for p in probes if p.get('status')=='measured'] if 'native_ue_floor_offset_m' in floor else []
            rows.append(row)
        result={'house':house,'status':'completed','rooms':rows,'geometry_origin':str(d/'native_editor_audit.json'),'scope':'sampled retained native capsule (0.20m radius, 0.75m half height) static room interpenetration; exact skinned hand/cloth contact with all furniture is not proven','source_or_frozen_thresholds_modified':False}
        write(completed,result);records.append(result);print(house,[(x['room_id'],x['status']) for x in rows],flush=True)
    result={'schema':'kujiale_native_capsule_contact_v1','status':'complete' if len(records)==21 else 'partial','records':records,'expected_houses':21,'expected_rooms':44,'elapsed_seconds':time.monotonic()-started}
    write(a.output/'result.json',result)
    with (o/'PROGRESS_zh.md').open('a') as f:f.write('\n\n## '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' 原生胶囊与实导出几何接触核对\n\n- 实际完成'+str(len(records))+'套，来源'+str(a.output/'result.json')+'；使用原生读回胶囊尺寸和UE实际LOD0三角形，不改冻门槛。采样胶囊静态碰撞检查不等于手、衣物与所有家具的精确动态接触。\n')

if __name__=='__main__':main()
