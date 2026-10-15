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
