"""Regressions for seated triangle contact and bounded native floor samples."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/assets'))
from v7_roster_collision import intersections,clipped_surface_minimum
from v7_roster_floor_grid import FloorGrid
pytestmark=pytest.mark.fast_unit

def test_triangle_crossing_seat_is_detected_when_all_vertices_are_outside():
 tri=np.array([[[-2,-.01,0],[2,-.01,0],[0,-.01,2]]])
 assert np.all(np.any(abs(tri[0])>np.array([.5,.025,.5]),axis=1))
 assert intersections(tri,[0,-.025,0],[1,.05,1]).tolist()==[0]

def test_intended_contact_and_rotated_box():
 touch=np.array([[[-.2,0,-.2],[.2,0,-.2],[0,0,.2]]])
 assert not len(intersections(touch,[0,-.025,0],[1,.05,1]))
 assert len(intersections(touch-[0,.02,0],[0,-.025,0],[1,.05,1]))==1
 outside=np.array([[[.7,.02,-.05],[.9,.02,-.05],[.8,.02,.05]]])
 assert not len(intersections(outside,[0,0,0],[1,.2,.2],np.pi/2))

def test_clipped_contact_height_uses_edge_intersections():
 tri=np.array([[[-1,.3,0],[1,.7,0],[0,.5,1]]])
 point=clipped_surface_minimum(tri,[0,0],[.2,.2],.25)
 assert point is not None
 assert np.allclose(point,[-.1,.48,0],atol=1e-9)

def test_missing_seat_surface_is_not_a_contact():
 tri=np.array([[[2,.4,2],[3,.4,2],[2,.4,3]]])
 assert clipped_surface_minimum(tri,[0,0],[1,1])is None

def test_floor_grid_interpolates_actual_slope_and_rejects_extrapolation(tmp_path):
 p=tmp_path/'floor.npz';np.savez(p,x=[0,1],z=[0,1],y=[[1,3],[2,4]])
 f=FloorGrid(p)
 assert f([.25,9,.5])==pytest.approx(2.25)
 assert f.many(np.array([[.25,9,.5],[1,0,1]])).tolist()==pytest.approx([2.25,4])
 with pytest.raises(RuntimeError,match='outside'):f([1.01,0,.5])
 with pytest.raises(RuntimeError,match='outside'):f.many(np.array([[-.01,0,.5]]))

def test_missing_floor_sample_rejected(tmp_path):
 p=tmp_path/'floor.npz';np.savez(p,x=[0,1],z=[0,1],y=[[0,np.nan],[0,0]])
 f=FloorGrid(p)
 with pytest.raises(RuntimeError,match='Missing'):f([.5,0,.5])

def test_gpu_zero_is_rejected_before_reading_devices():
 import argparse
 from v7_human_roster_gpu import run
 with pytest.raises(AssertionError,match='only GPU 1'):run(argparse.Namespace(gpu=0))


def test_foot_support_rejects_furniture_surface_and_missing_ground():
    from v7_roster_floor_grid import require_physical_ground
    import pytest
    assert require_physical_ground(0.009,0.008)==0.009
    with pytest.raises(ValueError,match="furniture"):
        require_physical_ground(0.45,0.008)
    with pytest.raises(ValueError,match="non-finite"):
        require_physical_ground(float("nan"),0.008)


def test_disk_measurement_does_not_follow_external_symlinks_or_count_hardlinks_twice(tmp_path):
    import os
    from v7_roster_disk import task_bytes
    task=tmp_path/"task";task.mkdir()
    outside=tmp_path/"external";outside.mkdir();(outside/"large").write_bytes(b"x"*1000000)
    (task/"payload").write_bytes(b"abc");os.link(task/"payload",task/"same_inode")
    (task/"external_link").symlink_to(outside,target_is_directory=True)
    size=task_bytes(task)
    assert size==task.stat().st_size+3+(task/"external_link").lstat().st_size
    assert size<1000000


def test_native_animation_nodes_exclude_orphan_placeholder_and_reject_orphan_skin_joint():
    from v7_roster_contact_ik import animation_nodes
    import pytest
    d={"nodes":[{"children":[1]},{},{}],"scenes":[{"nodes":[0]}],"skins":[{"joints":[1]}]}
    assert animation_nodes(d)=={0,1}
    d["skins"][0]["joints"].append(2)
    with pytest.raises(ValueError,match="Skin joint"):
        animation_nodes(d)


def test_review_camera_rejects_wall_immediately_in_front_of_near_clip():
    from v7_roster_ue_camera_plan import visible
    tri=np.array([[[.0005,-1,-1],[.0005,1,-1],[.0005,0,1]]])
    assert not visible(np.array([0.,0,0]),np.array([1.,0,0]),tri,tri.min(1),tri.max(1))
    assert visible(np.array([0.,2,0]),np.array([1.,2,0]),tri,tri.min(1),tri.max(1))


def test_second_gpu_job_is_rejected_until_first_guardian_exits(tmp_path):
    from v7_human_roster_gpu import exclusive_gpu_job
    with exclusive_gpu_job(tmp_path):
        with pytest.raises(RuntimeError,match='not exited'):
            with exclusive_gpu_job(tmp_path):
                pytest.fail('second job must never enter')
    with exclusive_gpu_job(tmp_path):
        pass
