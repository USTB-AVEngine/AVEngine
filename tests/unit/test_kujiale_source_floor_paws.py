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


def test_prepared_sole_selection_matches_independent_original_on_every_pose():
    import numpy as np
    from avengine.capture.kujiale_source_floor_paws import PreparedPawContacts
    vertices=np.array([[x,y,z]for x in [-.2,.2]for z in [-.3,.3]for y in [0.,.03]])
    row={'category':'dog','landmarks_local_m':{f'foot_{i}':[x,0.,z]for i,(x,z)in enumerate([(-.2,-.3),(-.2,.3),(.2,-.3),(.2,.3)])}}
    class Geometry:
        floor=SimpleNamespace(floor_lineage_face_mask=[True],floor_lineage_receipt='verified exact floor')
        def ray(self,point,direction,length):return {'triangle_index':0,'material_path':'/Game/Meshes/cabinet/shared','point_m':[float(point[0]),0.,float(point[2])]}
    prepared=PreparedPawContacts(row,vertices);geometry=Geometry()
    for yaw in [0.,.37,-2.2]:
        for gap in [0.,.011,.013,-.013]:
            placement={'root_m':[1.,gap,2.],'yaw_rad':yaw};assert prepared.measure(geometry,placement)==landmark_contacts(geometry,row,vertices,placement)
    assert prepared.measure(geometry,{'root_m':[0.,.013,0.],'yaw_rad':0.})['status']=='fail'


def test_prepared_paws_retain_missing_native_sole_failure():
    import numpy as np
    from avengine.capture.kujiale_source_floor_paws import PreparedPawContacts
    row={'category':'dog','landmarks_local_m':{f'foot_{i}':[float(i),0.,0.]for i in range(4)}}
    geometry=SimpleNamespace(floor=SimpleNamespace(floor_lineage_face_mask=[],floor_lineage_receipt=None),ray=lambda *a:None)
    vertices=np.array([[0.,0.,0.],[0.,.1,0.]])
    assert PreparedPawContacts(row,vertices).measure(geometry,{'root_m':[0.,0.,0.],'yaw_rad':0.})==landmark_contacts(geometry,row,vertices,{'root_m':[0.,0.,0.],'yaw_rad':0.})
