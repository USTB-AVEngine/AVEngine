"""Use exact source-floor face identity with the original four-paw 12mm test.

Original native material names stay in the readback. A deduplicated material
can name furniture while its particular, SHA-bound source triangle is floor.
"""
from copy import deepcopy

def landmark_contacts(geometry,row,vertices,placement):
    from avengine.qa.kujiale_asset_paws import landmark_contacts as original
    result=original(geometry,row,vertices,placement)
    if result.get('status')in {'not_applicable','deferred_to_native_skeletal_motion'}:return result
    return _with_source_floor_identity(geometry,result)

def _with_source_floor_identity(geometry,result):
    result=deepcopy(result);floor=geometry.floor
    for check in result.get('checks',[]):
        hit=check.get('native_floor_ray');gap=check.get('surface_gap_m')
        if hit is None or gap is None:continue
        triangle=hit['triangle_index'];lineage=getattr(floor,'floor_lineage_face_mask',[])
        verified=0<=triangle<len(lineage)and bool(lineage[triangle])
        check['native_source_floor_identity']={'verified_source_floor_triangle':verified,'native_triangle_index':triangle,'source_floor_lineage_receipt':getattr(floor,'floor_lineage_receipt',None),'original_native_material_path':hit['material_path']}
        if verified:
            check['original_material_name_check_status']=check['status'];check['status']='pass'if abs(gap)<=.012 else'fail'
    result['status']='pass'if len(result.get('checks',[]))==4 and all(c['status']=='pass'for c in result['checks'])else'fail'
    result['floor_identity_method']='original native floor material owner or independently verified exact source-floor triangle lineage'
    result['paw_region_and_12mm_thresholds_changed']=False
    return result


class PreparedPawContacts:
    """Cache only immutable sole selection; measure every proposed world pose.

    The unchanged QA evaluator selects the four native sole vertices once.
    Every pose still casts four current native rays and uses its 12mm gate.
    Full episode admission independently runs the original evaluator again.
    """
    def __init__(self,row,vertices):
        from avengine.qa.kujiale_asset_paws import landmark_contacts as original
        class NoWorldRays:
            def ray(self,*args):return None
        self.template=original(NoWorldRays(),row,vertices,{'root_m':[0.,0.,0.],'yaw_rad':0.})

    def measure(self,geometry,placement):
        import numpy as np
        from avengine.qa.kujiale_asset_geometry import rotation
        result=deepcopy(self.template)
        if result.get('status')in {'not_applicable','deferred_to_native_skeletal_motion'}:return result
        root=np.asarray(placement['root_m']);matrix=rotation(placement['yaw_rad'])
        for check in result.get('checks',[]):
            if 'native_lod0_vertex_local_m'not in check:continue
            world=root+matrix@np.asarray(check['native_lod0_vertex_local_m']);hit=geometry.ray(world+[0,.10,0],[0,-1,0],.22);gap=None if hit is None else float(world[1]-hit['point_m'][1]);passed=hit is not None and '/Meshes/floor/'in hit['material_path']and abs(gap)<=.012
            check.update(world_m=world.tolist(),native_floor_ray=hit,surface_gap_m=gap,status='pass'if passed else'fail')
        result['status']='pass'if all(c['status']=='pass'for c in result['checks'])else'fail'
        return _with_source_floor_identity(geometry,result)
