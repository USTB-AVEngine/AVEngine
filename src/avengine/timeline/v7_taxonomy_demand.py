"""Choose an actually supported underproduced class for a new physical scene."""
def choose_unseen_class(asset,records,observed_class34_counts):
    available={r['sound_class'] for r in records if r.get('sound_class')}
    classes=set(asset.get('allowed_classes',[]))&available
    classes.discard('speech_playback')
    if not classes:raise ValueError('no admitted non-speech recording matches the actual source capability')
    if any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in observed_class34_counts.values()):
        raise ValueError('taxonomy feedback must count actual delivered positive answers')
    chosen=min(classes,key=lambda cls:(observed_class34_counts.get(cls,0),cls))
    return chosen,{'actual_asset_id':asset['asset_id'],'eligible_classes':sorted(classes),
        'observed_class34_positive_counts':dict(observed_class34_counts),
        'selected_class':chosen,'selection':'least observed actual positive count; deterministic ties'}
