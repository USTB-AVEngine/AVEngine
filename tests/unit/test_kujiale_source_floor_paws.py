from types import SimpleNamespace
import pytest
from avengine.capture.kujiale_source_floor_paws import landmark_contacts
pytestmark=pytest.mark.fast_unit

def test_exact_floor_lineage_preserves_original_material_and_12mm_gate(monkeypatch):
    from avengine.qa import kujiale_asset_paws
    raw={'status':'fail','checks':[{'foot':str(i),'status':'fail','surface_gap_m':.003,'native_floor_ray':{'triangle_index':i,'material_path':'/Game/Meshes/cabinet/deduplicated_material'}}for i in range(4)]};monkeypatch.setattr(kujiale_asset_paws,'landmark_contacts',lambda *a:raw)
    floor=SimpleNamespace(floor_lineage_face_mask=[True]*4,floor_lineage_receipt='SHA-verified exact floor triangles');geometry=SimpleNamespace(floor=floor)
    result=landmark_contacts(geometry,{},[],{});assert result['status']=='pass'and raw['status']=='fail'
    assert all('cabinet'in c['native_floor_ray']['material_path']for c in result['checks'])
    raw['checks'][0]['surface_gap_m']=.013;assert landmark_contacts(geometry,{},[],{})['status']=='fail'
    raw['checks'][0]['surface_gap_m']=.003;floor.floor_lineage_face_mask[0]=False;assert landmark_contacts(geometry,{},[],{})['status']=='fail'

def test_missing_four_native_paws_cannot_gain_admission(monkeypatch):
    from avengine.qa import kujiale_asset_paws
    raw={'status':'deferred_to_native_skeletal_motion','reason':'no four landmarks'};monkeypatch.setattr(kujiale_asset_paws,'landmark_contacts',lambda *a:raw)
    assert landmark_contacts(SimpleNamespace(),{},[],{})==raw
