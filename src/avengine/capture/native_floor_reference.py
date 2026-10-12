"""Ground native actors on the audited UE render surface, without moving meshes."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


class NativeFloorReference:
    def __init__(self, declaration):
        import trimesh
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        audit_path=Path(declaration['native_editor_audit'])
        if file_sha256(audit_path)!=declaration['native_editor_audit_sha256']:
            raise ValueError('native floor audit identity changed')
        audit=json.loads(audit_path.read_text());geometry=audit['native_render_geometry']
        for key,expected in [('world_vertices_ue_cm_f32','native_vertices_sha256'),('triangles_u32','native_triangles_sha256')]:
            if file_sha256(geometry[key])!=declaration[expected]:raise ValueError('native floor geometry identity changed')
        raw=np.fromfile(geometry['world_vertices_ue_cm_f32'],dtype='<f4').reshape(-1,3)
        faces=np.fromfile(geometry['triangles_u32'],dtype='<u4').reshape(-1,3)
        self.mesh=trimesh.Trimesh(vertices=raw[:,[0,2,1]].astype(float)/100.,faces=faces,process=False)
        self.mesh.ray=RayMeshIntersector(self.mesh)
        self.declaration=declaration
        # Native LOD sections retain their source surface identity even when
        # every room object was baked into one StaticMesh. A bed material
        # section can never become floor merely by passing a downward ray.
        sections=geometry.get('sections',[])
        self.floor_face_mask=None
        if sections:
            self.floor_face_mask=np.zeros(len(faces),dtype=bool)
            offset=0
            for section in sections:
                count=int(section['triangle_count'])
                if '/Meshes/floor/' in section.get('material_path',''):
                    self.floor_face_mask[offset:offset+count]=True
                offset+=count
            if offset!=len(faces):raise ValueError('native section triangle ranges disagree with render geometry')
        self.room_floor_y_m=None
        self.last_support_readback=None

    def sample(self, nav_point_m):
        point=np.asarray(nav_point_m,dtype=float)
        if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('native floor nav point is invalid')
        origin=point+[0.,.18,0.]
        hits,_,triangles=self.mesh.ray.intersects_location(origin[None,:],[[0.,-1.,0.]],multiple_hits=False)
        if len(hits)!=1:raise ValueError('native floor support surface is missing')
        surface=hits[0]
        if abs(float(surface[1]-point[1]))>.15:
            raise ValueError('native floor support lies outside retained navigation snap tolerance')
        triangle=int(triangles[0])
        if self.floor_face_mask is not None and not self.floor_face_mask[triangle]:
            raise ValueError('native support face is not room floor')
        if self.room_floor_y_m is not None and abs(float(surface[1]-self.room_floor_y_m))>.02:
            raise ValueError('native support height differs from measured room floor baseline')
        self.last_support_readback={'triangle_index':triangle,'surface_y_m':float(surface[1]),
            'room_floor_y_m':self.room_floor_y_m,'surface_identity':'floor section' if self.floor_face_mask is not None else 'measured horizontal floor fixture'}
        return [float(point[0]),float(surface[1]),float(point[2])]

    def validate_probes(self, floor_reference):
        measured=[p['native_visual_floor_y_m'] for p in floor_reference['probes'] if p.get('status')=='measured']
        if len(measured)<9:raise ValueError('render surface does not agree with nine retained native floor probes')
        self.room_floor_y_m=float(np.median(measured))
        checks=[]
        for probe in floor_reference['probes']:
            if probe.get('status')!='measured':raise ValueError('native floor reference has an unmeasured witness')
            try:actual=self.sample(probe['nav_point_m'])
            except ValueError as error:raise ValueError('render surface does not agree with nine retained native floor probes: '+str(error)) from error
            error=abs(actual[1]-probe['native_visual_floor_y_m'])
            checks.append({'witness_index':probe['witness_index'],'role':probe['role'],'render_surface_y_m':actual[1],
                           'native_depth_floor_y_m':probe['native_visual_floor_y_m'],'absolute_error_m':error})
        if len(checks)<9 or max(c['absolute_error_m'] for c in checks)>.02:
            raise ValueError('render surface does not agree with nine retained native floor probes')
        return {'status':'pass','method':'actual audited UE render LOD0 downward ray at each actor nav point',
                'checks':checks,'max_native_depth_error_m':max(c['absolute_error_m'] for c in checks),
                'native_depth_tolerance_m':.02,'source_or_nav_geometry_modified':False,
                'floor_offset_is_spatial_not_a_constant_median':True,'room_floor_y_m':self.room_floor_y_m,
                'support_face_must_be_room_floor':True}
