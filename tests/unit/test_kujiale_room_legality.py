from types import SimpleNamespace
import math
import numpy as np
import pytest
from tools.ue.check_kujiale_room_legality import (
    bounded_supports, category, groups, find_witness, corners)
from tools.ue.capture_kujiale_room_legality import measure_point

def view(**changes):
    value={"position_m":[0.,0.,0.],"yaw_deg":0.,"pitch_deg":0.,"hfov_deg":85.}
    value.update(changes);return value

def test_native_point_readback_rejects_intervening_furniture_at_unchanged_5mm():
    depth=np.full((72,128),2.)
    assert measure_point(depth,view(),[3.,0.,0.])["status"]=="fail"
    depth[:]=2.994
    assert measure_point(depth,view(),[3.,0.,0.])["status"]=="fail"
    depth[:]=2.996
    result=measure_point(depth,view(),[3.,0.,0.])
    assert result["status"]=="pass"
    assert result["occlusion_tolerance_m"]==.005

def test_native_point_projection_handles_yaw_pitch_and_rejects_back_or_crop():
    yaw,pitch=map(math.radians,[63.,21.])
    target=4*np.array([math.cos(pitch)*math.cos(yaw),math.sin(pitch),math.cos(pitch)*math.sin(yaw)])
    depth=np.full((72,128),4.1)
    v=view(yaw_deg=63.,pitch_deg=21.)
    assert measure_point(depth,v,target)["status"]=="pass"
    assert measure_point(depth,v,-target)["reason"]=="target_behind_native_camera"
    assert measure_point(depth,view(),[1.,0.,10.])["reason"]=="target_outside_native_frame"

def test_native_invalid_depth_cannot_grant_placement_visibility():
    for value in (np.nan,0.,-1.):
        assert measure_point(np.full((72,128),value),view(),[3.,0.,0.])["status"]=="fail"

def test_bounded_support_search_deduplicates_and_covers_entire_catalog():
    supports=[{"point_m":[x/10.,0.,0.],"normal":[0,1,0]} for x in range(30)]
    rows=bounded_supports(supports+supports,4)
    assert len(rows)==4 and rows[0]["point_m"]==supports[0]["point_m"]
    assert rows[-1]["point_m"]==supports[-1]["point_m"]

def test_object_type_keeps_table_and_wall_restrictions_separate():
    assets=[{"category":"object","object_type":"television","attachment_surface":s} for s in ("table","wall")]
    assert set(groups(assets))=={"object.television.table","object.television.wall"}
    assert [category(a) for a in assets]==["table_object","wall_object"]
    assert category({"category":"bird"})=="bird_with_perch"

def test_floor_witness_cannot_be_admitted_from_a_bed_support(monkeypatch):
    class Floor:
        room_floor_y_m=0.
        def sample(self,point):
            raise ValueError("native support face is not room floor")
    support={"point_m":[0.,.5,0.],"normal":[0,1,0],"floor_y_m":0.,"surface":"floor"}
    geometry=SimpleNamespace(floor=Floor(),placement=lambda *a:{
        "root_m":[0.,.5,0.],"yaw_rad":0.,"support":support})
    vertices=corners([-.1,0.,-.1],[.1,.3,.1])
    asset={"category":"object","attachment_surface":"floor"}
    placement,reason=find_witness(geometry,None,asset,vertices,{"floor":[support]},4,10.)
    assert placement is None
    assert reason["failures"]["support_is_not_measured_room_floor"]==2

def test_absent_support_is_bounded_search_failure_not_physical_impossibility():
    asset={"category":"object","attachment_surface":"ceiling"}
    placement,reason=find_witness(None,None,asset,corners([0,0,0],[1,1,1]),{"ceiling":[]},4,10.)
    assert placement is None
    assert reason["reason"]=="no_sampled_ceiling_support"
    assert "physical impossibility is not proven" in reason["claim"]


def test_room_evidence_publication_keeps_completed_record_and_no_partial_file(tmp_path):
    import json
    from tools.ue.check_kujiale_room_legality import publish
    target=tmp_path/"room.json"
    publish(target,{"status":"pass","room_id":"R0"})
    with pytest.raises(ValueError,match="completed evidence retained"):
        publish(target,{"status":"fail"})
    assert json.loads(target.read_text())=={"status":"pass","room_id":"R0"}
    assert list(tmp_path.iterdir())==[target]

def test_native_publication_never_replaces_existing_evidence(tmp_path):
    from tools.ue.capture_kujiale_room_legality import publish
    target=tmp_path/"native.json"
    publish(target,{"status":"pass"})
    with pytest.raises(FileExistsError):
        publish(target,{"status":"fail"})
    assert target.read_text().count('"pass"')==1
    assert list(tmp_path.iterdir())==[target]


def test_room_whitelist_does_not_upgrade_a_CPU_witness_without_native_readback():
    from tools.ue.report_kujiale_room_legality import combine_room
    cpu={"house":"H","room_id":"R0","split":"train","ue_map_path":"/Game/R",
         "floor_support_check":{"status":"pass"},"native_floor_surface":{},
         "original_native_floor_admission":{},"CPU_planning_floor_reference":{},
         "category_checks":[{"category_key":"animal.dog","coarse_category":"dog",
                            "representative_asset_id":"dog_a",
                            "status":"available_CPU_native_LOD_witness"}]}
    native={"house":"H","room_id":"R0","ue_map_path":"/Game/R",
            "native_room_render_status":"pass","category_views":[]}
    result=combine_room(cpu,native)
    assert result["categories"][0]["status"]=="not_verified"
    assert result["coarse_categories"]["dog"]["usable_category_keys"]==[]

def test_room_whitelist_rejects_native_readback_for_a_different_individual():
    from tools.ue.report_kujiale_room_legality import combine_room
    cpu={"house":"H","room_id":"R0","split":"train","ue_map_path":"/Game/R",
         "floor_support_check":{"status":"pass"},"native_floor_surface":{},
         "original_native_floor_admission":{},"CPU_planning_floor_reference":{},
         "category_checks":[{"category_key":"animal.dog","coarse_category":"dog",
                            "representative_asset_id":"dog_a",
                            "status":"available_CPU_native_LOD_witness"}]}
    native={"house":"H","room_id":"R0","ue_map_path":"/Game/R",
            "native_room_render_status":"pass","category_views":[
            {"category_key":"animal.dog","representative_asset_id":"dog_b","status":"pass"}]}
    with pytest.raises(ValueError,match="different representative asset"):
        combine_room(cpu,native)


def test_later_room_floor_lineage_reloads_geometry_instead_of_inheriting_donor_mask(monkeypatch):
    from tools.ue import check_kujiale_room_legality as tool
    class Floor:
        def __init__(self,declaration):
            self.declaration=declaration.copy()
        def sample(self,point):
            if not self.declaration.get("source_floor_face_lineage_path"):
                raise ValueError("native support face is not room floor")
            return point
    def construct(declaration):
        return SimpleNamespace(floor=Floor(declaration))
    monkeypatch.setattr(tool,"NativeSupportGeometry",construct)
    first=tool.geometry_for_room({"native_editor_audit":"same_mesh"})
    with pytest.raises(ValueError,match="not room floor"):
        first.floor.sample([0.,0.,0.])
    assert tool.geometry_for_room({"native_editor_audit":"same_mesh"},first) is first
    later=tool.geometry_for_room({"native_editor_audit":"same_mesh",
                                 "source_floor_face_lineage_path":"verified_source_floor"},first)
    assert later is not first
    assert later.floor.sample([0.,0.,0.])==[0.,0.,0.]
