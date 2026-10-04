"""Hermetic checks for the actual glTF pose and colour algorithms."""
import io,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/assets'))
from v7_human_roster_pose import pose,quat_matrix,slerp
from v7_human_roster_materials import srgb_to_linear,linear_to_srgb
from v7_human_roster_color_qa import lab
pytestmark=pytest.mark.fast_unit

def test_srgb_and_d65_known_values():
 rgb=np.arange(256,dtype=np.uint8)
 assert np.array_equal(linear_to_srgb(srgb_to_linear(rgb)),rgb)
 assert np.allclose(lab([255,255,255]),[100,0,0],atol=.02)
 assert np.allclose(lab([255,0,0]),[53.2408,80.0925,67.2032],atol=.02)

def test_shortest_path_quaternion_and_half_turn():
 q=np.array([0.,1.,0.,0.]);half=slerp(np.array([0.,0.,0.,1.]),q,.5)
 assert np.allclose(quat_matrix(half)@np.array([1,0,0]),[0,0,-1],atol=1e-8)
 assert np.allclose(quat_matrix(slerp(q,-q,.3)),quat_matrix(q))

def test_skin_world_transform_includes_inverse_bind_once():
 positions=np.array([[0,0,0],[1,1,0],[0,1,1]],np.float32);normal=np.tile([0,0,1],(3,1)).astype(np.float32);ibm=np.eye(4,dtype=np.float32);ibm[0,3]=-2
 arrays=[positions,normal,np.zeros((3,4),np.uint16),np.tile([1,0,0,0],(3,1)).astype(np.float32),ibm.T.reshape(1,16)]
 d={'nodes':[{'name':'Bone','translation':[3,0,0]},{'mesh':0,'skin':0}],'scenes':[{'nodes':[0,1]}],'scene':0,'meshes':[{'primitives':[{'attributes':{'POSITION':0,'NORMAL':1,'JOINTS_0':2,'WEIGHTS_0':3}}]}],'skins':[{'joints':[0],'inverseBindMatrices':4}],'accessors':[],'bufferViews':[]}
 data=bytearray()
 for a,kind in zip(arrays,['VEC3','VEC3','VEC4','VEC4','MAT4']):
  offset=len(data);data.extend(a.tobytes());view=len(d['bufferViews']);d['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':a.nbytes});d['accessors'].append({'bufferView':view,'componentType':5123 if a.dtype==np.uint16 else 5126,'count':len(a),'type':kind})
 result,evidence=pose(d,bytes(data),canonical_flip=False)
 assert np.allclose(result[0]['positions'],positions+[1,0,0])
 assert evidence['floor_y_m']==0
 assert np.allclose(result[0]['normals'],normal)


def test_rigid_export_preserves_normalized_nonwhite_colour(tmp_path):
 from v7_human_roster_pose import rigid_glb,accessor
 from v7_human_roster_materials import read_glb
 positions=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
 rgba=np.tile([255,0,128,255],(3,1)).astype(np.uint8)
 binary=positions.tobytes()+rgba.tobytes()
 d={'asset':{'version':'2.0'},'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':positions.nbytes},{'buffer':0,'byteOffset':positions.nbytes,'byteLength':rgba.nbytes}],'accessors':[{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3'},{'bufferView':1,'componentType':5121,'normalized':True,'count':3,'type':'VEC4'}],'nodes':[{'mesh':0}],'meshes':[{'primitives':[{'attributes':{'POSITION':0,'COLOR_0':1}}]}]}
 path=tmp_path/'pose.glb';rigid_glb(d,binary,[{'node':0,'primitive':0,'positions':positions,'normals':None}],path)
 out,b=read_glb(path);idx=out['meshes'][0]['primitives'][0]['attributes']['COLOR_0']
 assert out['accessors'][idx]['componentType']==5126
 assert np.allclose(accessor(out,b,idx),rgba.astype(float)/255,atol=1e-7)

def test_motion_sampling_includes_keys_after_constant_root_channel():
 from v7_human_roster_pose import animation_times
 root=np.array([0],np.float32);moving=np.array([0,.5,1],np.float32)
 d={'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':root.nbytes},{'buffer':0,'byteOffset':root.nbytes,'byteLength':moving.nbytes}],'accessors':[{'bufferView':0,'componentType':5126,'count':1,'type':'SCALAR'},{'bufferView':1,'componentType':5126,'count':3,'type':'SCALAR'}]}
 actual=animation_times(d,root.tobytes()+moving.tobytes(),{'samplers':[{'input':0},{'input':1}]})
 assert np.array_equal(actual,[0,.5,1])
