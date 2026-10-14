"""Physical budget accounting must survive sparse files, links and nested outputs."""
import os
import pytest

pytestmark=pytest.mark.fast_unit

from avengine.timeline.v7_storage_budget import allocated_bytes


def test_allocated_blocks_nested_sparse_and_links(tmp_path):
    nested=tmp_path/"a"/"b"
    nested.mkdir(parents=True)
    ordinary=nested/"ordinary"
    ordinary.write_bytes(b"x"*8193)
    sparse=tmp_path/"sparse"
    with sparse.open("wb") as stream:
        stream.seek(4*1024*1024)
        stream.write(b"x")
    os.link(ordinary,tmp_path/"hardlink")
    outside=tmp_path.parent/(tmp_path.name+"_outside")
    outside.mkdir()
    (outside/"excluded").write_bytes(b"x"*16384)
    (tmp_path/"file_link").symlink_to(ordinary)
    (tmp_path/"directory_link").symlink_to(outside,target_is_directory=True)
    (tmp_path/"broken_link").symlink_to(outside/"missing")
    expected=sum(path.stat().st_blocks*512 for path in (ordinary,sparse))
    assert allocated_bytes(tmp_path)==expected
    assert allocated_bytes(tmp_path/"missing")==0
    assert allocated_bytes(nested)==ordinary.stat().st_blocks*512
