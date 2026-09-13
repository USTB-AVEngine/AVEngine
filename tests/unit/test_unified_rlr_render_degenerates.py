"""The GLB extractor matches the retained native RLR zero-area check."""
import numpy as np
from avengine.acoustics.gltf import _nonzero_area_triangle_mask


def test_tiny_render_triangle_is_rejected_before_native_upload():
    vertices=np.array([[0,0,0],[2e-6,0,0],[0,2e-6,0],[1,0,0],[0,1,0]],dtype=np.float32)
    triangles=np.array([[0,1,2],[0,3,4]],dtype=np.uint32)
    assert _nonzero_area_triangle_mask(vertices,triangles).tolist()==[False,True]


def test_collinear_triangle_and_empty_mesh_keep_native_semantics():
    vertices=np.array([[0,0,0],[1,0,0],[2,0,0]],dtype=np.float32)
    assert not _nonzero_area_triangle_mask(vertices,np.array([[0,1,2]],dtype=np.uint32))[0]
    assert _nonzero_area_triangle_mask(vertices,np.empty((0,3),dtype=np.uint32)).size==0
