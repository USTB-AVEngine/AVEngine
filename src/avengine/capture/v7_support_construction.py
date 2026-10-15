"""Bounded same-surface reconstruction, independent of semantic scan labels.

A label is helpful placement evidence, not an answer. Every reconstructed site
still runs the actual native support rays, mesh contacts, visibility, emitter
and acoustic checks. Required bodies and their registered mounts stay intact.
"""
from copy import deepcopy
import math
from shapely.geometry import Point


def alternate_survey_origins(polygon,floor_y,original):
    centre=polygon.representative_point();x0,z0,x1,z1=polygon.bounds
    points=[(centre.x,centre.y)]+[(x0+(x1-x0)*x,z0+(z1-z0)*z)for x,z in ((.2,.2),(.8,.8),(.2,.8),(.8,.2))]
    unique={}
    for x,z in points:
        if polygon.covers(Point(x,z)) and math.hypot(x-original[0],z-original[2])>.35:
            unique.setdefault((round(x,4),round(z,4)),[float(x),float(floor_y),float(z)])
    return list(unique.values())[:4]


def retry_same_surface(scene,asset,key,supports,polygon,floor_y,original,survey_at):
    if asset.get('kind')!='object':return None
    aid=asset['asset_id'];state=scene.setdefault('_support_reconstruction',{}).setdefault(aid,{'next':0})
    origins=alternate_survey_origins(polygon,floor_y,original)
    stages=[None,*origins]
    while state['next']<len(stages):
        index=state['next'];state['next']+=1;origin=stages[index]
        candidates=(supports.get('_native_geometric_candidates',{}).get(key,[])if origin is None else survey_at(origin).get(key,[]))
        candidates=[deepcopy(p)for p in candidates if p.get('surface')==asset['attachment_surface']and abs(p.get('floor_y',floor_y)-floor_y)<=.08 and polygon.buffer(.01).covers(Point(*[p.get('originating_native_floor_m',p['point'])[i]for i in (0,2)]))]
        if not candidates:continue
        supports[key]=candidates
        scene.setdefault('native_physical_support_fallback_asset_ids',[]).append(aid)
        record={'asset_id':aid,'stage':'same_registered_surface_native_geometry','surface':key,
            'survey_origin_m':origin or list(original),'support_candidates':len(candidates),
            'semantic_label_required':False,'actual_support_contacts_and_acoustics_required':True}
        scene.setdefault('construction_fallbacks',[]).append(record)
        return record
    return None
