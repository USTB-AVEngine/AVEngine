import pytest
from avengine.capture.v7_semantic_label_sampling_view import make_semantic_label_sampling_view
pytestmark=pytest.mark.fast_unit

def test_sampling_view_preserves_valid_lines_and_original_bytes(tmp_path):
    raw=b'object_id,color,label,region\r\n1,112233,"floor",0\r\n2,c,"radiator",0\r\n3,334455,"wall",0\r\n'
    source=tmp_path/'original.txt';source.write_bytes(raw);view=tmp_path/'private.txt'
    receipt=make_semantic_label_sampling_view(source,view)
    assert source.read_bytes()==raw
    assert view.read_bytes()==raw.replace(b'2,c,"radiator",0\r\n',b'')
    assert receipt['removed_irrelevant_rows']==[{'line_number':3,'object_id':'2','colour':'c','category':'radiator','original_support_kind':None}]
    assert receipt['geometry_modified']is False

def test_malformed_supported_floor_is_never_filtered(tmp_path):
    source=tmp_path/'original.txt';source.write_bytes(b'object_id,color,label,region\n1,c,"floor",0\n');view=tmp_path/'private.txt'
    with pytest.raises(ValueError,match='malformed support-surface colour'):
        make_semantic_label_sampling_view(source,view)
    assert not view.exists()
