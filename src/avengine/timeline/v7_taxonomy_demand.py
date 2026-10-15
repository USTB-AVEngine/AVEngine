"""Choose an underproduced class supported by the real source and QA panel."""
def choose_unseen_class(asset,records,observed_class34_counts,*,emission_mechanisms=None):
    available={r['sound_class'] for r in records if r.get('sound_class')}
    classes=set(asset.get('allowed_classes',[]))&available
    if asset.get('kind')in {'human','performer'}:classes.discard('speech_playback')
    if not classes:raise ValueError('no admitted recording matches the actual source capability')
    if any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in observed_class34_counts.values()):
        raise ValueError('taxonomy feedback must count actual delivered positive answers')
    # Match the existing source intake and CLASS34 predicate; neither is changed.
    # A walking planner may supply its real registered-foot mechanism. Ordinary
    # mouth sources use the same mechanism as v7_visible_sources.build_scene.
    mechanisms={}
    for cls in classes:
        if asset.get('kind') in {'human','performer'}:mechanism='human_live_recorded_'+cls
        elif asset.get('kind')=='animal':mechanism='animal_live_recorded'
        elif asset.get('entity_class')=='loudspeaker':mechanism='loudspeaker_playback'
        else:mechanism='device_recorded_event'
        mechanisms[cls]=(emission_mechanisms or {}).get(cls,mechanism)
    excluded={cls:mechanism for cls,mechanism in mechanisms.items()
        if mechanism.startswith('human_live') and mechanism!='human_live_recorded_laughter'}
    classes-=excluded.keys()
    if not classes:raise ValueError('unchanged V7-CLASS34 predicate excludes the proposed source mechanisms: '+repr(excluded))
    chosen=min(classes,key=lambda cls:(observed_class34_counts.get(cls,0),cls))
    return chosen,{'actual_asset_id':asset['asset_id'],'eligible_classes':sorted(classes),
        'source_mechanisms_by_class':mechanisms,'excluded_by_unchanged_class34_predicate':excluded,
        'observed_class34_positive_counts':dict(observed_class34_counts),
        'selected_class':chosen,'selection':'least observed actual positive count after source capability, recording and original panel eligibility'}
