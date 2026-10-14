"""Exact vectorized vertical intersections against every original stage triangle.

This accelerates candidate-sole searches. Every selected support point is still
read back independently by the native collision ray and compared within 0.1 mm.
No plane approximation, grid quantization or triangle decimation is used.
"""
from pathlib import Path
import hashlib
import numpy as np

class OriginalVerticalStageRay:
    def __init__(self,path,*,mesh=None,transform_evidence=None):
        import shapely
        from shapely.strtree import STRtree
        from avengine.capture.v7_visible_mesh import indexed_static_mesh
        path=Path(path);mesh=indexed_static_mesh(path) if mesh is None else mesh
        self.native_transform_evidence=transform_evidence
        self.source=str(path);self.source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        self.tri=np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
        lo=self.tri[:,:,[0,2]].min(1);hi=self.tri[:,:,[0,2]].max(1)
        self.tree=STRtree(shapely.box(lo[:,0],lo[:,1],hi[:,0],hi[:,1]))
        self.requested_points=0

    def many(self,points):
        import shapely
        p=np.asarray(points,float)
        if p.ndim!=2 or p.shape[1]!=3:raise ValueError('vertical stage rays require N x 3 metres')
        pairs=self.tree.query(shapely.points(p[:,0],p[:,2]))
        result=np.full(len(p),-np.inf)
        if pairs.size:
            pi,ti=pairs;t=self.tri[ti];a=t[:,0];u=t[:,1]-a;v=t[:,2]-a
            x=p[pi,0]-a[:,0];z=p[pi,2]-a[:,2]
            denominator=u[:,0]*v[:,2]-u[:,2]*v[:,0]
            safe=np.abs(denominator)>1e-15
            alpha=np.zeros(len(pi));beta=np.zeros(len(pi))
            alpha[safe]=(x[safe]*v[safe,2]-z[safe]*v[safe,0])/denominator[safe]
            beta[safe]=(u[safe,0]*z[safe]-u[safe,2]*x[safe])/denominator[safe]
            y=a[:,1]+alpha*u[:,1]+beta*v[:,1]
            origin=p[pi,1]+.2
            valid=safe&(alpha>=-1e-8)&(beta>=-1e-8)&(alpha+beta<=1+1e-8)&(y<=origin+1e-7)&(y>=origin-.85-1e-7)
            np.maximum.at(result,pi[valid],y[valid])
        if not np.isfinite(result).all():raise ValueError('original whole stage has no vertical support for every candidate point')
        self.requested_points+=len(p)
        return result


def original_native_stage_mesh(video,path):
    """Resolve the full original GLB using the actual native stage attributes."""
    from avengine.capture.v7_visible_mesh import indexed_static_mesh
    path=Path(path)
    manager=video.sim.metadata_mediator.stage_template_manager
    attr=manager.get_template_by_handle(str(video.receipt['stage']))
    up=np.asarray(attr.orient_up,float);front=np.asarray(attr.orient_front,float)
    if abs(np.linalg.norm(up)-1)>1e-6 or abs(np.linalg.norm(front)-1)>1e-6 or abs(up@front)>1e-6:
        raise ValueError('native scan orientation is not an admitted orthonormal basis')
    if not np.allclose(attr.scale,[1,1,1],atol=1e-8) or not np.allclose(attr.origin,[0,0,0],atol=1e-8):
        raise ValueError('native stage scale/origin needs an explicit qualified adapter')
    right=np.cross(front,up);right/=np.linalg.norm(right)
    rotation=np.stack((right,up,-front),axis=0)
    if abs(np.linalg.det(rotation)-1)>1e-6:raise ValueError('native stage basis is not a rigid rotation')
    key=(str(path),rotation.tobytes())
    cache=getattr(video,'phase2_native_stage_geometry_cache',{})
    if key not in cache:
        raw=indexed_static_mesh(path);raw.vertices=np.asarray(raw.vertices)@rotation.T
        evidence={'revision':'native-stage-up-front-attributes-readback-v1','source':str(path),
            'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'native_stage_handle':str(video.receipt['stage']),
            'native_orient_up':up.tolist(),'native_orient_front':front.tolist(),
            'source_coordinate_to_native_world_rotation':rotation.tolist(),'native_scale':list(attr.scale),
            'native_origin':list(attr.origin),'original_vertices':len(raw.vertices),'original_triangles':len(raw.faces),
            'triangles_decimated':False,'axis_transform_is_rigid':True}
        cache[key]=(raw,evidence);video.phase2_native_stage_geometry_cache=cache
    return cache[key]
