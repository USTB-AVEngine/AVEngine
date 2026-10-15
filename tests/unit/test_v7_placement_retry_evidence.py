import json
import pytest
from avengine.capture.v7_placement_diagnostics import record_placement_rejection
from avengine.timeline.v7_construction_retry import spatial_construction_failure
pytestmark=pytest.mark.fast_unit


def test_three_reconstruction_passes_keep_distinct_immutable_evidence(tmp_path):
    paths=[record_placement_rejection(tmp_path,'voice1',i,{'pass':i,'reason':'no site yet'})for i in range(3)]
    assert len(set(paths))==3
    assert [json.loads(p.read_text())['pass']for p in paths]==[0,1,2]
    assert not(tmp_path/'attachment_rejection.json').exists()


def test_only_identified_layout_evidence_collision_is_automatically_reconstructed():
    assert spatial_construction_failure({'error_type':'FileExistsError','reason':'standing/.attachment_rejection.json.partial -> attachment_rejection.json'})
    assert not spatial_construction_failure({'error_type':'FileExistsError','reason':'unknown_shared_file'})
