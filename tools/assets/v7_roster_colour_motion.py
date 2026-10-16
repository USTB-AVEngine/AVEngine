"""Apply a selected shirt variant's PBR appearance to the unique motion master in memory."""
import copy,base64
from pathlib import Path
from v7_human_roster_materials import read_glb
def apply_appearance(master,blob,variant_path):
 v,vb=read_glb(Path(variant_path));d=copy.deepcopy(master);b=bytearray(blob);byname={m['name']:i for i,m in enumerate(v['materials'])}
 for mesh in d['meshes']:
  for primitive in mesh['primitives']:
   if 'material'in primitive:
    name=master['materials'][primitive['material']]['name']
    if name not in byname:raise ValueError('Variant has no corresponding material: '+name)
    primitive['material']=byname[name]
 d['materials']=copy.deepcopy(v['materials']);d['textures']=copy.deepcopy(v.get('textures',[]));d['samplers']=copy.deepcopy(v.get('samplers',[]));images=[]
 for im in v.get('images',[]):
  image=copy.deepcopy(im)
  if 'bufferView'in im:
   view=v['bufferViews'][im['bufferView']];off=view.get('byteOffset',0);data=vb[off:off+view['byteLength']]
  else:
   uri=im['uri'];data=base64.b64decode(uri.split(',',1)[1])if uri.startswith('data:')else(Path(variant_path).parent/uri).read_bytes();image.pop('uri',None)
  b.extend(b'\0'*(-len(b)%4));index=len(d['bufferViews']);d['bufferViews'].append({'buffer':0,'byteOffset':len(b),'byteLength':len(data)});b.extend(data);image['bufferView']=index;images.append(image)
 d['images']=images;d['buffers'][0]['byteLength']=len(b)
 for key in ['extensionsUsed','extensionsRequired']:
  if key in d or key in v:d[key]=sorted(set(d.get(key,[]))|set(v.get(key,[])))
 return d,bytes(b)
