"""Owner-approved object-only retries; native truth and answer checks stay intact."""
from copy import deepcopy
from pathlib import Path
import gc
import re
import shutil

from avengine.timeline.continuous_pilot import write_json


def drop_failed_object(row, catalog, reason):
    """Return a marked recipe subset only for a named failed registered object."""
    match=re.search(r'source material never measured: ([A-Za-z0-9_,.\-]+)',reason)
    if match:
        failed=match.group(1).split(',')
    else:
        match=re.search(r'no strict native (?:floor|table|wall|ceiling) placement for ([A-Za-z0-9_.\-]+)',reason)
        failed=[match.group(1)]if match else []
    by_id={a['asset_id']:a for a in catalog}
    aid=next((a for a in failed if a in row['asset_ids']and by_id.get(a,{}).get('kind')=='object'),None)
    if aid is None:
        return None
    result=deepcopy(row)
    result['scene_id']=row['scene_id']+'__drop_object_1'
    result['independent_parent_scene_id']=result['scene_id']
    result['asset_ids'].remove(aid)
    for key in ('source_classes','proposed_sound_asset_ids','motion_requests'):
        result.get(key,{}).pop(aid,None)
    result['silent_asset_ids']=[a for a in result.get('silent_asset_ids',[])if a!=aid]
    for key in list(result):
        if key.startswith('reuse_'):
            result.pop(key)
    result.setdefault('element_degradations',[]).append({
        'asset_id':aid,'action':'removed_failed_object',
        'reason':reason,'source_parent_scene_id':row['scene_id'],
        'policy':'owner-single-element-fallback-20261014',
        'remaining_asset_sound_bindings_unchanged':True,
        'source_truth_recomputed_from_actual_remaining_native_scene':True})
    return result


def generate_parent_with_element_fallback(row,house,inputs,catalog,source_admission,human_admission,output):
    from avengine.timeline.v7_phase2_generation import generate_phase2_parent
    requested=row['scene_id']
    current=row
    for attempt in range(2):
        try:
            result=generate_phase2_parent(current,house,inputs,catalog,source_admission,human_admission,output)
            return {**result,'requested_scene_id':requested,
                    'element_fallbacks':current.get('element_degradations',[])}
        except ValueError as error:
            replacement=drop_failed_object(current,catalog,str(error))if attempt==0 else None
            if replacement is None:
                raise
            write_json(Path(output)/'parents'/current['scene_id']/'element_fallback.json',{
                'status':'retry_with_one_marked_object_removed','reason':str(error),
                'next_scene_id':replacement['scene_id'],
                'original_partial_outputs_retained':True,
                'maximum_object_fallbacks_per_parent':1})
            # The failed native context has closed in the producer's finally.
            # Do not retain its exception frame while constructing another one.
            error.__traceback__=None
        current=replacement
        gc.collect()
        if Path(output).resolve().is_relative_to(Path('/data')):
            available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:'))
            if available<40_000_000_000 or shutil.disk_usage('/data').free<150_000_000_000:
                raise RuntimeError('object fallback stopped by owner memory/disk headroom')
    raise AssertionError('bounded object fallback exhausted')
