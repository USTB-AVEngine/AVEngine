import json
from pathlib import Path
import numpy as np
import pytest
from tools.rooms.alpha_storage_contract import native_seal
from tools.rooms.package_kujiale_alpha import publish_noreplace
from tools.rooms.plan_kujiale_room_quota import quota_rows,member_mix
from avengine.timeline.v7_ue_appearance import measure_ue_color


def fixture_seal(tmp_path):
    root=tmp_path/'task';parent=root/'ue_round8/run/generation/parents/test_scene';parent.mkdir(parents=True)
    recipe={'seed_row':{'scene_id':parent.name},'house':{'input':{'visual_backend':'ue_spear'}}}
    (parent/'input_recipe.json').write_text(json.dumps(recipe))
    seal=parent/'alpha_seal_attempt_test';seal.mkdir();(seal/'payload').write_text('unchanged')
    pointer={'attempt':seal.name,'result':{'status':'formal_alpha_admitted'},'required_files':['payload']}
    (parent/'alpha_seal_completed.json').write_text(json.dumps(pointer))
    return root,parent,pointer


def test_seal_rejects_required_path_escape(tmp_path):
    root,parent,pointer=fixture_seal(tmp_path);pointer['required_files']=['../../input_recipe.json']
    (parent/'alpha_seal_completed.json').write_text(json.dumps(pointer))
    with pytest.raises(ValueError,match='escapes'):native_seal(parent,root)


def test_seal_rejects_directory_escape(tmp_path):
    root,parent,pointer=fixture_seal(tmp_path);pointer['attempt']='../alpha_seal_attempt_other'
    (parent/'alpha_seal_completed.json').write_text(json.dumps(pointer))
    with pytest.raises(ValueError,match='escaped'):native_seal(parent,root)


def test_seal_rejects_symlink_payload(tmp_path):
    root,parent,pointer=fixture_seal(tmp_path);seal=parent/pointer['attempt'];(seal/'payload').unlink()
    (seal/'payload').symlink_to(parent/'input_recipe.json')
    with pytest.raises(ValueError,match='symlink'):native_seal(parent,root)


def test_seal_rejects_different_parent_identity(tmp_path):
    root,parent,pointer=fixture_seal(tmp_path);p=parent/'input_recipe.json';recipe=json.loads(p.read_text());recipe['seed_row']['scene_id']='other';p.write_text(json.dumps(recipe))
    with pytest.raises(ValueError,match='ownership'):native_seal(parent,root)


def test_atomic_publication_keeps_existing_delivery(tmp_path):
    old=tmp_path/'delivery';old.mkdir();(old/'payload').write_text('old');new=tmp_path/'partial';new.mkdir();(new/'payload').write_text('new')
    with pytest.raises(FileExistsError):publish_noreplace(new,old)
    assert (old/'payload').read_text()=='old' and (new/'payload').read_text()=='new'


def test_room_caps_and_exact_S_subset_of_full():
    admission={'split':'train','render_path':'ue_spear','houses':[{'house':'kujiale_0001','room_ids':['R0','R1']}],'house_count':1,'room_count':2}
    template={'seeds':{'layout':1},'window_count':2,'questions_per_window':14}
    small,full=member_mix(quota_rows(admission,template,4),quota_rows(admission,template,16))
    assert len(small)==8 and len(full)==32
    by_id={r['scene_id']:r for r in full}
    assert all(r==by_id[r['scene_id']] for r in small)
    assert all(sum(r['room_id']==rid for r in full)==16 for rid in ['R0','R1'])


def test_nontrain_kujiale_budget_is_refused():
    admission={'split':'test','render_path':'ue_spear'}
    with pytest.raises(ValueError,match='training'):quota_rows(admission,{'seeds':{}},4)


def test_zero_colored_ROI_cannot_create_color_evidence():
    rgb=np.full((32,32,3),80,dtype=np.uint8);mask=np.ones((32,32),dtype=np.uint32)
    result=measure_ue_color(rgb,mask,1,'blue')
    assert result['status']=='blocked' and result['saturated_pixels']==0


def test_same_UE_pixels_wrong_color_label_still_rejected():
    rgb=np.zeros((32,32,3),dtype=np.uint8);rgb[:]=[0,80,240];mask=np.ones((32,32),dtype=np.uint32)
    assert measure_ue_color(rgb,mask,1,'blue')['status']=='pass'
    assert measure_ue_color(rgb,mask,1,'yellow')['status']=='blocked'


def transfer_fixture(tmp_path):
    from tools.rooms.prepare_kujiale_cci_batch import prepare_batch,digest
    root=tmp_path/'task';b=root/'ue_round8/deliveries/probe';b.mkdir(parents=True);payload=b/'payload';payload.write_text('native derived data')
    (b/'BUNDLE_MANIFEST.json').write_text(json.dumps({'files':[{'path':'payload','bytes':payload.stat().st_size,'sha256':digest(payload)}]}))
    (root/'ue_round8/evidence').mkdir()
    result=prepare_batch(root,[b],'test');plan=Path(result['plan']);value=json.loads(plan.read_text());receipt=root/'ue_round8/evidence/receipt.json'
    base='/mnt/afs/xinyuanqian/jzy/hardbank/mirror'
    proof={'status':'all_SHA256_verified','cci_base':base,'files':[{'path':base+'/'+r['cci_relative'],'bytes':r['bytes'],'sha256':r['sha256']} for r in value['source_files']]}
    receipt.write_text(json.dumps(proof));return root,b,plan,receipt,proof


def test_reclaim_requires_independent_matching_CCI_hash(tmp_path):
    from tools.rooms.reclaim_kujiale_cci_batch import reclaim_batch
    root,b,plan,receipt,proof=transfer_fixture(tmp_path);proof['files'][0]['sha256']='0'*64;receipt.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='checksum'):reclaim_batch(plan,receipt,root)
    assert (b/'payload').read_text()=='native derived data'


def test_reclaim_refuses_new_unlisted_files(tmp_path):
    from tools.rooms.reclaim_kujiale_cci_batch import reclaim_batch
    root,b,plan,receipt,proof=transfer_fixture(tmp_path);(b/'unlisted').write_text('must retain')
    with pytest.raises(ValueError,match='unlisted'):reclaim_batch(plan,receipt,root)
    assert (b/'payload').exists() and (b/'unlisted').exists()


def test_reclaim_keeps_other_owned_data_after_checked_copy(tmp_path):
    from tools.rooms.reclaim_kujiale_cci_batch import reclaim_batch
    root,b,plan,receipt,proof=transfer_fixture(tmp_path);keep=root/'ue_round8/necessary_witness';keep.write_text('retain')
    result=reclaim_batch(plan,receipt,root)
    assert result['status']=='reclaimed' and not b.exists() and keep.read_text()=='retain'


def test_owned_metadata_publication_keeps_no_successful_temp_links(tmp_path):
    from tools.rooms.alpha_storage_contract import write_owned_json
    d=tmp_path/'ue_round8';d.mkdir();p=d/'receipt.json';write_owned_json(p,{'status':'pass'})
    assert [f.name for f in d.iterdir()]==['receipt.json']
    with pytest.raises(FileExistsError):write_owned_json(p,{'status':'changed'})
    assert json.loads(p.read_text())=={'status':'pass'}


def test_common_alpha_member_hook_keeps_room_based_single_rows():
    from avengine.timeline.v7_alpha_sampling import alpha_row
    original={'schema':'upstream','assignment_status':'admitted','house':'kujiale_0001','room_id':'R0','members':['v0_a0'],'window_start_ticks':[0,200],'question_quota_base':80,'counterfactual_plan':'single','allowed_member_changes':[]}
    layout={'house':'kujiale_0001','room_id':'R0','status':'pass'}
    kept=alpha_row(original,layout,3,force_quartet=False)
    assert kept['members']==['v0_a0'] and kept['question_quota_base']==80
    forced=alpha_row(original,layout,3)
    assert len(forced['members'])==4 and forced['question_quota_base']==24


def test_room_budget_cannot_inherit_canary_layout_from_another_house():
    admission={'split':'train','render_path':'ue_spear','houses':[{'house':'kujiale_0001','room_ids':['R0']}],'house_count':1,'room_count':1}
    rows=quota_rows(admission,{'seeds':{},'measured_static_layout':{'house':'kujiale_0022'},'upstream_room_id':'wrong'},4)
    assert all('measured_static_layout' not in r and 'upstream_room_id' not in r for r in rows)


def test_formal_binding_rejects_cross_house_or_missing_four_body_layout():
    from tools.rooms.bind_kujiale_formal_seed import bind_formal_seed
    row={'split':'train','render_path':'ue_spear','house':'kujiale_0001','room_id':'R0','room_parent_slot':0,'members':['v0_a0']}
    house={'family':'kujiale','split':'train','house':'kujiale_0001','input':{'visual_backend':'ue_spear'},'checked_room_ids':['R0']}
    with pytest.raises(ValueError,match='do not match'):bind_formal_seed(row,house,{'house':'kujiale_0022','room_id':'R0','status':'pass','valid':[1,2,3,4]})
    with pytest.raises(ValueError,match='not qualified'):bind_formal_seed(row,house,{'house':'kujiale_0001','room_id':'R0','status':'blocked','valid':[]})
