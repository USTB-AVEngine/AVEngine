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


def _source_floor_lineage_faces(declaration, geometry, vertices, faces):
    """Verify exact source-floor faces when material deduplication lost identity.

    A material donor's name is not a geometry owner. The optional frozen
    receipt links source files, the audited Native LOD, and each floor face's
    actual coordinates; it never grants floor identity to a whole material.
    """
    path = declaration.get('source_floor_face_lineage_path')
    expected = declaration.get('source_floor_face_lineage_sha256')
    if not path and not expected:
        return [], None
    if not path or not expected or file_sha256(path) != expected:
        raise ValueError('source floor lineage receipt identity changed')
    receipt = json.loads(Path(path).read_text())
    if receipt.get('schema') != 'kujiale_native_source_floor_lineage_v1':
        raise ValueError('source floor lineage schema is invalid')
    bindings = [
        ('native_audit', declaration['native_editor_audit'], declaration['native_editor_audit_sha256']),
        ('native_vertices', geometry['world_vertices_ue_cm_f32'], declaration['native_vertices_sha256']),
        ('native_triangles', geometry['triangles_u32'], declaration['native_triangles_sha256']),
    ]
    for key, native_path, native_hash in bindings:
        record = receipt[key]
        if Path(record['path']).resolve() != Path(native_path).resolve() or record['sha256'] != native_hash:
            raise ValueError('source floor lineage is for different native geometry')
    source_files = [receipt['source_original'], receipt['source_expanded']]
    source_files += receipt['source_prim_layer_files']
    verified_files = set()
    for record in source_files:
        key = (str(Path(record['path']).resolve()), record['sha256'])
        if key not in verified_files and file_sha256(record['path']) != record['sha256']:
            raise ValueError('source floor lineage source geometry identity changed')
        verified_files.add(key)
    # Ten micrometres covers only export float rounding. Contact, navigation,
    # floor baseline and every UE readback threshold remain unchanged.
    tolerance = 1e-5
    if not 0 < float(receipt['match_tolerance_m']) <= tolerance:
        raise ValueError('source floor lineage matching tolerance is invalid')
    selected = set()
    import itertools
    for surface in receipt['floor_mesh_lineage']:
        prim = surface['source_prim']
        if not prim.startswith('/Root/Meshes/floor/') or not prim.rsplit('/', 1)[-1].startswith('SM_Floor'):
            raise ValueError('source floor lineage is not a source floor mesh')
        points = np.asarray(surface['source_points_canonical_m'], dtype=float)
        source_faces = np.asarray(surface['source_triangles'])
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('source floor lineage vertices are invalid')
        if source_faces.ndim != 2 or source_faces.shape[1] != 3 or not np.issubdtype(source_faces.dtype, np.integer):
            raise ValueError('source floor lineage triangles are invalid')
        if source_faces.size == 0 or source_faces.min() < 0 or source_faces.max() >= len(points):
            raise ValueError('source floor lineage triangle vertices are outside source geometry')
        mappings = surface['native_face_to_source_triangle']
        if len(mappings) != len(source_faces):
            raise ValueError('source floor lineage is not a complete floor mesh mapping')
        source_indices = set()
        for mapping in mappings:
            native_index = mapping['native_triangle_index']
            source_index = mapping['source_floor_triangle_index']
            if not isinstance(native_index, int) or not isinstance(source_index, int):
                raise ValueError('source floor lineage face indices are invalid')
            if not 0 <= native_index < len(faces) or not 0 <= source_index < len(source_faces):
                raise ValueError('source floor lineage faces are outside geometry')
            if native_index in selected or source_index in source_indices:
                raise ValueError('source floor lineage duplicates a face')
            source_triangle = points[source_faces[source_index]]
            native_triangle = vertices[faces[native_index]]
            error = min(float(np.linalg.norm(native_triangle - source_triangle[list(order)], axis=1).max())
                        for order in itertools.permutations(range(3)))
            if error > tolerance:
                raise ValueError('source floor lineage triangle geometry differs from native face')
            selected.add(native_index)
            source_indices.add(source_index)
    if not selected:
        raise ValueError('source floor lineage has no verified floor faces')
    return sorted(selected), str(Path(path).resolve())


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
        # Material paths identify ordinary sections. Deduplicated materials
        # may name another mesh; exact, frozen source triangle lineage can
        # prove those particular faces without accepting furniture surfaces.
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
        lineage_faces, self.floor_lineage_receipt = _source_floor_lineage_faces(
            declaration, geometry, np.asarray(self.mesh.vertices), faces)
        self.floor_lineage_face_mask = np.zeros(len(faces), dtype=bool)
        if lineage_faces:
            if self.floor_face_mask is None:
                self.floor_face_mask = np.zeros(len(faces), dtype=bool)
            self.floor_face_mask[lineage_faces] = True
            self.floor_lineage_face_mask[lineage_faces] = True
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
            'room_floor_y_m':self.room_floor_y_m,'surface_identity':('verified source-floor triangle lineage' if self.floor_lineage_face_mask[triangle] else 'floor section') if self.floor_face_mask is not None else 'measured horizontal floor fixture','source_floor_lineage_receipt':self.floor_lineage_receipt if self.floor_lineage_face_mask[triangle] else None}
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
