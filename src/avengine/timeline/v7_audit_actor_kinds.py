"""Read native species before appearance text in independent delivery audits.

An offscreen or silent person's display label can be a numeric instance ID.
That does not turn the physical human into a nonhuman object.
"""

def actor_kind(actor):
    species=str(actor.get('species_id')or'').lower()
    if 'human'in species:return 'human'
    mode=str((actor.get('source')or{}).get('motion_mode',''))
    if mode.startswith('slide'):return 'slide'
    if 'dog'in species:return 'dog'
    if 'cat'in species:return 'cat'
    if species:return 'other'
    label=str(actor.get('display_label')or actor.get('asset_id')or'').lower()
    return 'human'if any(word in label for word in ('person','woman','man ','adult','human'))or label.endswith('man')or not species else'other'


from functools import lru_cache


@lru_cache(maxsize=32)
def observed_motion_modes(visual_path):
    """Read actual complete native admission, never infer sliding from speed."""
    import json
    from pathlib import Path
    visual=json.loads(Path(visual_path).read_text());modes={}
    for record in visual.get('native_motion_admissions',{}).values():
        aid=record.get('asset_id');frames=record.get('native_30Hz_readbacks',[])
        if (not aid or record.get('status')!='pass'or len(frames)!=901
                or record.get('final_complete_scene_recheck',{}).get('status')!='pass'
                or any(f.get('asset_id')!=aid for f in frames)):
            continue
        mode=record.get('motion_mode')
        if mode in {'slide_no_animation','walk_animation'}:modes[aid]=mode
    return modes
