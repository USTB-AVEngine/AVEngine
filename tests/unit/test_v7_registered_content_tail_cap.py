import json,sqlite3
import numpy as np
import pytest
from avengine.timeline.v7_registered_content import _source_components
pytestmark=pytest.mark.fast_unit


def test_registered_reconstruction_uses_actual_tail_cap_without_modifying_raw_rir(tmp_path):
    drydir=tmp_path/'dry';drydir.mkdir();cache=tmp_path/'rir_cache';cache.mkdir()
    x=np.zeros(16000);x[:100]=1.
    np.save(drydir/'voice1.npy',x);np.save(drydir/'voice2.npy',x*2)
    (tmp_path/'dry_completed.json').write_text(json.dumps({'attempt':'dry'}))
    points=[[0.,1.,2.],[2.,1.,2.]];h=np.ones((4,16001))*.01
    with sqlite3.connect(cache/'positions.sqlite3')as db:
        db.execute('create table rirs(id integer, pose text, completed integer)')
        for i,p in enumerate(points):
            db.execute('insert into rirs values (?,?,1)',(i,json.dumps(p,separators=(',',':'))));np.save(cache/f'rir_{i:06d}.npy',h)
    original=(cache/'rir_000000.npy').read_bytes()
    scene={'sources':{f'voice{i+1}':{'trajectory':[{'position_m':p}]}for i,p in enumerate(points)}}
    stems,_=_source_components(tmp_path,scene,'a0');swapped,_=_source_components(tmp_path,scene,'a1')
    assert np.any(stems['voice1'][:,:100])
    assert not np.any(stems['voice1'][:,8100:])
    assert np.max(stems['voice1'][:,6500:7500])>np.max(stems['voice1'][:,8000:8100])
    assert np.allclose(swapped['voice1'],2*stems['voice1'])
    assert (cache/'rir_000000.npy').read_bytes()==original


def test_counterfactual_subject_uses_actual_registered_color_and_can_swap_actor_id():
    from avengine.timeline.v7_registered_content import _subject_for_appearance
    assets={'yellow':{'color':'yellow'},'black':{'color':'black'}};row={'asset_ids':['yellow','black']}
    original=[{'actor_id':1,'asset_id':'yellow','visible':True},{'actor_id':2,'asset_id':'black','visible':True}]
    swapped=[{'actor_id':1,'asset_id':'black','visible':True},{'actor_id':2,'asset_id':'yellow','visible':True}]
    assert _subject_for_appearance(row,original,assets)['actor_id']==1
    assert _subject_for_appearance(row,swapped,assets)['actor_id']==2
