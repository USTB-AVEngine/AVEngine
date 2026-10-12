"""Native LOD0 support and conservative furniture-clearance checks for Kujiale."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np


def rotation(yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def triangle_box_overlap(triangles,low,high):
    """Exact triangle/AABB separating-axis test; touching is overlap."""
    t=np.asarray(triangles,float);lo=np.asarray(low,float);hi=np.asarray(high,float)
    if not len(t):return np.zeros(0,dtype=bool)
    center=(lo+hi)/2;half=(hi-lo)/2;v=t-center
    active=np.all(v.min(axis=1)<=half+1e-9,axis=1)&np.all(v.max(axis=1)>=-half-1e-9,axis=1)
    edges=np.stack([v[:,1]-v[:,0],v[:,2]-v[:,1],v[:,0]-v[:,2]],axis=1)
    axes=[np.cross(edges[:,0],edges[:,1])]
    for edge in range(3):
        for unit in np.eye(3):axes.append(np.cross(edges[:,edge],unit))
    for axis in axes:
        radius=np.sum(np.abs(axis)*half,axis=1);projection=np.einsum('nij,nj->ni',v,axis)
        active &= (projection.min(axis=1)<=radius+1e-9)&(projection.max(axis=1)>=-radius-1e-9)
    return active


class NativeSupportGeometry:
    def __init__(self,declaration):
        from avengine.capture.native_floor_reference import NativeFloorReference
        self.floor=NativeFloorReference(declaration);self.mesh=self.floor.mesh
        audit=json.loads(Path(declaration['native_editor_audit']).read_text())
        self.sections=audit['native_render_geometry']['sections'];self.face_sections=np.empty(len(self.mesh.faces),dtype=np.uint16);offset=0
        for i,s in enumerate(self.sections):
            n=int(s['triangle_count']);self.face_sections[offset:offset+n]=i;offset+=n
        if offset!=len(self.mesh.faces):raise ValueError('native section face ranges are incomplete')
        self.triangles=self.mesh.triangles;self.low=self.triangles.min(axis=1);self.high=self.triangles.max(axis=1)

    def ray(self,point,direction,length):
        hits,_,ids=self.mesh.ray.intersects_location(np.asarray(point,float)[None],np.asarray(direction,float)[None],multiple_hits=False)
        if len(hits)!=1:return None
        hit=hits[0];distance=float(np.linalg.norm(hit-point))
        if distance>length:return None
        face=int(ids[0]);section=self.sections[int(self.face_sections[face])]
        return {'point_m':hit.tolist(),'triangle_index':face,'normal':self.mesh.face_normals[face].tolist(),'material_path':section['material_path'],'section':section['section'],'distance_m':distance}

    def collision(self,vertices,root,yaw,normal,tolerance):
        """No native room triangle may enter the visual envelope above support.

        This conservative native-LOD test can reject a clear concave asset; it
        cannot grant clearance merely because a bounding-box vertex was clear.
        The support band alone uses the declared contact tolerance.
        """
        vertices=np.asarray(vertices,float);root=np.asarray(root,float);R=rotation(yaw);lo=vertices.min(axis=0);hi=vertices.max(axis=0)
        local_n=R.T@np.asarray(normal,float);axis=int(np.argmax(np.abs(local_n)))
        if local_n[axis]>0:lo[axis]+=tolerance
        else:hi[axis]-=tolerance
        if np.any(hi<=lo):return {'status':'fail','reason':'asset envelope narrower than support tolerance'}
        corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])@R.T+root
        world_lo,world_hi=corners.min(axis=0),corners.max(axis=0)
        ids=np.flatnonzero(np.all(self.low<=world_hi+1e-8,axis=1)&np.all(self.high>=world_lo-1e-8,axis=1))
        local=(self.triangles[ids]-root)@R
        overlaps=triangle_box_overlap(local,lo,hi);bad=ids[overlaps]
        return {'status':'pass' if len(bad)==0 else 'fail','method':'native UE LOD0 triangles vs complete conservative visual OBB, 13 separating axes',
                'intersecting_native_triangles':len(bad),'tested_native_triangles':len(ids),'support_band_m':tolerance,
                'intersection_materials':list(dict.fromkeys(self.sections[int(self.face_sections[i])]['material_path'] for i in bad[:10]))}

    def support_catalog(self,house,room):
        from shapely.geometry import shape,Point
        plan=json.loads(Path(house['input']['ue_config']['acceptance_plan_path']).read_text())
        declared=next(x for x in plan['rooms'] if x['room_id']==room['room_label']);polygon=shape(declared['floor_polygon'])
        self.floor.validate_probes(room['floor_reference']);baseline=self.floor.room_floor_y_m
        catalog={'floor':[],'table':[],'wall':[],'ceiling':[]};minx,minz,maxx,maxz=polygon.bounds
        points=[]
        for x in np.arange(minx+.15,maxx,.3):
            for z in np.arange(minz+.15,maxz,.3):
                if polygon.contains(Point(x,z)):
                    try:p=self.floor.sample([x,baseline+.01,z])
                    except ValueError:continue
                    points.append(np.asarray(p));catalog['floor'].append({'point_m':p,'normal':[0,1,0],'surface':'floor','floor_y_m':baseline,'support_identity':self.floor.last_support_readback})
                origin=[x,baseline+1.45,z]
                if not polygon.contains(Point(x,z)):continue
                hit=self.ray(origin,[0,-1,0],1.45)
                if hit and .50<=hit['point_m'][1]-baseline<=1.22 and abs(hit['normal'][1])>.96 and any(t in hit['material_path'].lower() for t in ['table_','desk_','counter','cabinet_']):
                    catalog['table'].append({**hit,'normal':[0,1,0],'surface':'table','floor_y_m':baseline})
        for p in points:
            hit=self.ray(p+[0,1.8,0],[0,1,0],2.2)
            if hit and '/Meshes/ceiling/' in hit['material_path'] and abs(hit['normal'][1])>.96 and 2.15<=hit['point_m'][1]-baseline<=3.9:
                catalog['ceiling'].append({**hit,'normal':[0,-1,0],'surface':'ceiling','floor_y_m':baseline})
            for height in [1.45,2.20,2.35,1.10]:
                for d in [np.array(x) for x in [[1,0,0],[-1,0,0],[0,0,1],[0,0,-1]]]:
                    hit=self.ray(p+[0,height,0],d,3.5)
                    if hit and '/Meshes/wall/' in hit['material_path'] and abs(float(np.dot(hit['normal'],d)))>.96:
                        catalog['wall'].append({**hit,'normal':(-d).tolist(),'surface':'wall','floor_y_m':baseline})
        target=np.asarray(room['placement']['source_1_m'])[[0,2]]
        for rows in catalog.values():rows.sort(key=lambda r:float(np.linalg.norm(np.asarray(r['point_m'])[[0,2]]-target)))
        return catalog,polygon

    def placement(self,vertices,support,yaw,tolerance):
        vertices=np.asarray(vertices,float);normal=np.asarray(support['normal'],float);point=np.asarray(support['point_m'],float)
        R=rotation(yaw);local_n=R.T@normal;values=vertices@local_n;extreme=float(values.min());band=min(.002,.05*float(np.ptp(values)))
        contact=vertices[values<=extreme+band+1e-9];anchor=(contact.min(axis=0)+contact.max(axis=0))/2
        anchor += local_n*(extreme-float(anchor@local_n));root=point-R@anchor
        target_samples=[q for q in contact[np.linspace(0,len(contact)-1,min(5,len(contact))).astype(int)]]
        gaps=[];readbacks=[]
        for q in target_samples:
            p=root+R@q;hit=self.ray(p+normal*.10,-normal,.22)
            if hit is None:return None
            gap=float(np.dot(p-np.asarray(hit['point_m']),normal));gaps.append(gap);readbacks.append(hit)
        if max(abs(x) for x in gaps)>tolerance:return None
        collision=self.collision(vertices,root,yaw,normal,tolerance)
        if collision['status']!='pass':return None
        buried=self.collision(vertices,root-normal*.03,yaw,normal,tolerance)
        if buried['status']!='fail':return None
        return {'status':'pass','root_m':root.tolist(),'yaw_rad':float(yaw),'normal':normal.tolist(),'contact_tolerance_m':tolerance,
                'contact_deltas_m':gaps,'support':support,'support_readbacks':readbacks,'collision':collision,
                'known_bad_3cm_buried_collision':buried,'attachment_points_local_m':[q.tolist() for q in target_samples]}


def import_vertices(receipt):
    parts=[]
    for obj in receipt['static']:
        if obj['class']=='StaticMesh':
            value=json.loads(Path(obj['geometry_path']).read_text());parts.append(np.asarray(value['vertices_ue_cm'],float)[:,[0,2,1]]/100.)
    if not parts:raise ValueError('no measured native static geometry')
    return np.concatenate(parts)


def camera_candidates(vertices,placement,polygon):
    from shapely.geometry import Point
    root=np.asarray(placement['root_m']);R=rotation(placement['yaw_rad']);world=np.asarray(vertices)@R.T+root;lo,hi=world.min(axis=0),world.max(axis=0);target=(lo+hi)/2
    normal=np.asarray(placement['normal']);is_wall=abs(normal[1])<.5;diameter=max(float(np.linalg.norm(hi-lo)),.15)
    distance=max(.50,diameter*1.6);base=math.atan2(normal[2],normal[0]) if is_wall else 0
    angles=np.deg2rad([-45,-20,0,20,45]) if is_wall else np.deg2rad([0,60,120,180,240,300,30,90,150,210,270,330])
    candidates=[]
    for factor in [1.,.8,1.2]:
        for angle in angles:
            eye=target+np.array([math.cos(base+angle)*distance*factor,max(.12,.3*diameter),math.sin(base+angle)*distance*factor])
            eye[1]=min(eye[1],placement['support'].get('floor_y_m',0)+2.4)
            if not polygon.contains(Point(eye[0],eye[2])):continue
            d=target-eye
            candidates.append({'position_m':eye.tolist(),'yaw_deg':math.degrees(math.atan2(d[2],d[0])),
                               'pitch_deg':math.degrees(math.atan2(d[1],np.linalg.norm(d[[0,2]]))),'target_m':target.tolist(),'hfov_deg':85})
    return candidates
