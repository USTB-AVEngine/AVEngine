"""Use exact source-floor face identity with the original four-paw 12mm test.

Original native material names stay in the readback. A deduplicated material
can name furniture while its particular, SHA-bound source triangle is floor.
"""
from copy import deepcopy

def landmark_contacts(geometry,row,vertices,placement):
    from avengine.qa.kujiale_asset_paws import landmark_contacts as original
    result=original(geometry,row,vertices,placement)
    if result.get('status')in {'not_applicable','deferred_to_native_skeletal_motion'}:return result
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
