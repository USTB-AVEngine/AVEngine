"""Keep only speed-compliant delivery windows; never change native measures.

Whole-parent camera, readback, invalidation and other strict source conditions
remain binding. The source seal and every QA answer remain unchanged.
"""
from copy import deepcopy

REVISION='unchanged-speed-range-window-subset-v1'


def scoped_window_policy(source,windows,*,enabled=True):
    result=deepcopy(source)
    if not enabled:return result
    ids=[w['window_id']for w in windows]
    metrics=source.get('native_views',{}).get('windows',{})
    dropped={wid:['window_measured_root_speed_out_of_range']for wid in ids
        if any(actor.get('status')=='fail'for actor in metrics.get(wid,{}).get('actors',[]))}
    if not dropped:return result
    speed_suffixes={'window_measured_root_speed_out_of_range','measured_root_speed_out_of_range'}
    strict=[r for r in source.get('reasons',[])if r.rsplit(':',1)[-1]not in speed_suffixes]
    kept=[wid for wid in ids if wid not in dropped]
    if not kept:strict.append('no_deliverable_window_meets_unchanged_speed_range')
    result.update(status='fail'if strict else'pass',reasons=strict,
        window_delivery_revision=REVISION,excluded_window_ids=sorted(dropped),
        excluded_window_reasons=dropped,delivered_window_ids=kept,
        source_parent_policy_status=source.get('status'),source_parent_reasons=source.get('reasons',[]),
        scope='unchanged global source conditions plus speed-compliant delivered windows',
        speed_ranges_changed=False,original_native_seal_changed=False,answers_changed=False)
    return result


def scope_receipt_windows(receipt,source_policy,*,enabled=True):
    result=deepcopy(receipt);export=result['export']
    windows=export.get('source_windows_before_exclusion',export['windows'])
    policy=scoped_window_policy(source_policy,windows,enabled=enabled)
    export['scene_delivery_policy']=policy
    if policy.get('excluded_window_ids'):
        excluded=set(policy['excluded_window_ids']);export['source_windows_before_exclusion']=windows
        export['windows']=[w for w in windows if w['window_id']not in excluded]
        export['excluded_window_ids']=sorted(excluded);export['window_delivery_revision']=REVISION
    return result,policy
