from copy import deepcopy
import pytest
from avengine.capture.kujiale_registered_room_floor import bind_room_floor
pytestmark=pytest.mark.fast_unit

def test_same_geometry_lineage_is_carried_without_changing_native_probes():
    original={'native_editor_audit':'a','native_editor_audit_sha256':'a-sha','native_vertices_sha256':'v-sha','native_triangles_sha256':'t-sha'};house={'house':'h','input':{'ue_config':{'native_floor_surface':original}},'checked_rooms':[{'room_label':'R1','floor_reference':{'probes':['nine original native measurements']}}]};before=deepcopy(house);extended={**original,'source_floor_face_lineage_path':'exact-source-triangles.json','source_floor_face_lineage_sha256':'proof'};registration={'rooms':[{'house':'h','room_id':'R1','native_floor_surface':extended}]}
    result=bind_room_floor(house,'R1',registration);assert house==before and result['checked_rooms']==before['checked_rooms']and result['input']['ue_config']['native_floor_surface']==extended
    extended['native_triangles_sha256']='different mesh'
    with pytest.raises(ValueError,match='different native geometry'):bind_room_floor(house,'R1',registration)
