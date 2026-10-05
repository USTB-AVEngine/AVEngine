"""Explicit garment masks and single-attribute counterfactual GLBs.

Recolour uses the established linear luminance formula, within shirt_main_color
only. Normal/specular/opacity images, UVs, geometry and animation bytes persist.
"""
import argparse,hashlib,io,json,struct
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
SHIRT_ISLANDS={
 'female_adult_01':[1,3,5,6,7,8,9], 'female_adult_02':[0,5,6],
 'female_adult_05':[0,1,2,3,4,5,13,14], 'female_adult_08':[0],
 'female_adult_13':[8,9,10,11,12,13,14,15,17,18,19,23,26],
 'female_adult_14':[2,4,5,6,7,8,11,12,13,16,17,21,22],
 'female_adult_15':[12,13], 'female_adult_17':[0,1,2],
 'male_adult_01':[2,3,4], 'male_adult_02':[6,7,8],
 'male_adult_06':[2,3,4], 'male_adult_08':[2,3,5],
 'male_adult_10':[0,1,5], 'male_adult_14':[0,1,3],
 'male_adult_16':[6,7,8], 'male_adult_20':[0,4,5]}
def srgb_to_linear(x):
 x=np.asarray(x,dtype=float)/255;return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def linear_to_srgb(x):
 x=np.clip(x,0,1);return np.rint(255*np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)).astype('uint8')
def read_glb(p):
 raw=Path(p).read_bytes();assert raw[:4]==b'glTF';n,k=struct.unpack_from('<I4s',raw,12);doc=json.loads(raw[20:20+n]);off=20+n;binary=b''
 if off<len(raw):l,k=struct.unpack_from('<I4s',raw,off);assert k==b'BIN\0';binary=raw[off+8:off+8+l]
 return doc,binary

def write_glb(p,doc,binary):
 raw=json.dumps(doc,separators=(',',':')).encode();raw+=b' '*((-len(raw))%4);binary+=b'\0'*((-len(binary))%4);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(struct.pack('<4sII',b'glTF',2,28+len(raw)+len(binary))+struct.pack('<I4s',len(raw),b'JSON')+raw+struct.pack('<I4s',len(binary),b'BIN\0')+binary)

def viewbytes(doc,binary,i):
 v=doc['bufferViews'][i];s=v.get('byteOffset',0);return binary[s:s+v['byteLength']]

def image_payloads(d,b):return [hashlib.sha256(viewbytes(d,b,i['bufferView'])).hexdigest() for i in d.get('images',[])]
def append_image(d,b,index,payload):
 b+=b'\0'*((-len(b))%4);v=len(d['bufferViews']);d['bufferViews'].append({'buffer':0,'byteOffset':len(b),'byteLength':len(payload)});d['images'][index]['bufferView']=v;d['images'][index]['mimeType']='image/png';b+=payload;d['buffers'][0]['byteLength']=len(b);return b

def material_body_image(d,prefix):
 mat=next(m for m in d['materials'] if m.get('name','').split('.')[0]==prefix+'_body');idx=mat['pbrMetallicRoughness']['baseColorTexture']['index'];return d['textures'][idx]['source']
def json_write(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def geometry_animation_digest(d,b):
 idx=set()
 for m in d.get('meshes',[]):
  for p in m['primitives']:idx.update(p['attributes'].values());idx.add(p['indices'])
 for s in d.get('skins',[]):idx.add(s['inverseBindMatrices'])
 for a in d.get('animations',[]):
  for s in a['samplers']:idx.update([s['input'],s['output']])
 h=hashlib.sha256(json.dumps({k:d.get(k) for k in ['meshes','nodes','skins','animations']},sort_keys=True).encode())
 for i in sorted(idx):ac=d['accessors'][i];h.update(json.dumps(ac,sort_keys=True).encode());h.update(viewbytes(d,b,ac['bufferView']))
 return h.hexdigest()