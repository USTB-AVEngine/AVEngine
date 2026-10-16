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
def masks(T,selection):
 sheet=Image.new('RGB',(4*340,4*360),'white');draw=ImageDraw.Draw(sheet)
 for idx,row in enumerate(r for r in selection['avatars'] if r['controlled_color']):
  key=row['name'].lower();islands=json.loads((T/'uv'/row['asset_id']/'uv_islands.json').read_text())['islands'];ids=SHIRT_ISLANDS[key];src=np.asarray(Image.open(row['body_texture']).convert('RGB'));h,w=src.shape[:2];mask=Image.new('L',(w,h));d=ImageDraw.Draw(mask)
  for island in islands:
   if island['island_id'] in ids:
    for poly in island['uv_polygons']:d.polygon([(u*(w-1),(1-v)*(h-1)) for u,v in poly],fill=255)
  surface=np.asarray(mask.filter(ImageFilter.MinFilter(3)))>0
  # Adult 01 short sleeves share an island with exposed forearm UVs.
  skin=np.zeros((h,w),bool)
  if key=='female_adult_01':
   rgb=src.astype(float);skin=surface&(rgb[...,1]>rgb[...,2]*1.12)&(rgb[...,0]>rgb[...,1]*1.08)&(rgb[...,0]<rgb[...,1]*1.8);surface &= ~skin
  lin=srgb_to_linear(src);lum=lin@np.array([.2126,.7152,.0722]);median=np.median(lum[surface]);protect=surface&((lum<median*.22)|(lum>median*2.6))
  # Dark horizontal stripes and the three buttons are preserved exactly, as in
  # the original controlled-shirt-colour-pipeline for m002.
  if key=='male_adult_01':
   integer_luma=(54*src[...,0].astype('uint32')+183*src[...,1].astype('uint32')+19*src[...,2].astype('uint32')+128)>>8;protect|=surface&(integer_luma<=145)
   guard=Image.new('L',(w,h));gd=ImageDraw.Draw(guard)
   for x,y,r in [(968,952,18),(979,1010,18),(983,1064,18)]:gd.ellipse((x-r,y-r,x+r,y+r),fill=255)
   protect|=surface&(np.asarray(guard)>0)
  if key in ['female_adult_08','male_adult_16']:
   # Distinct printed motif is not a garment base colour.
   # The shirt-local residual separates the motif from broad shading folds.
   luma8=np.rint(lum*255).astype('uint8');blur=np.asarray(Image.fromarray(luma8).filter(ImageFilter.GaussianBlur(20))).astype(float)/255
   if key=='female_adult_08':protect|=surface&(lum<blur*.48)
   else:protect|=surface&(lum>blur*1.5)&(lum>median*1.2)
  main=surface&~protect;assert main.sum()>20000,row['asset_id'];dest=T/'masks'/row['asset_id'];dest.mkdir(parents=True,exist_ok=True)
  for name,m in [('shirt_surface',surface),('shirt_main_color',main),('stripe_detail_protect',protect),('skin_exclusion',skin)]:Image.fromarray(m.astype('uint8')*255).save(dest/(name+'.png'))
  json_write(dest/'mask_registry.json',{'schema':'avengine_v7_shirt_mask_registry_v1','asset_id':row['asset_id'],'source_body_color':row['body_texture'],'selected_uv_islands':ids,'review_method':'UV topology plus source texture atlas inspection by Codex','surface_pixels':int(surface.sum()),'main_color_pixels':int(main.sum()),'protected_pixels':int(protect.sum()),'semantic':'shirt_main_color','operation':'replace_base_color_preserve_pbr_detail_v1','formula':'linear sRGB palette times source luminance / median; clamp 0.30..2.25','skin_exclusion_pixels':int(skin.sum())})
  overlay=src.copy().astype(float);overlay[main]=overlay[main]*.4+[0,153,60];overlay[protect]=overlay[protect]*.4+[150,0,150];im=Image.fromarray(np.clip(overlay,0,255).astype('uint8'));im.thumbnail((330,330));x=idx%4*340;y=idx//4*360;sheet.paste(im,(x,y+25));draw.text((x+2,y+3),row['name'],fill='black')
 sheet.save(T/'evidence/mask_contact_sheet.jpg',quality=85)
def variants(T,selection):
 proof=[]
 for row in selection['avatars']:
  base=T/'assets/metric'/row['asset_id']/'runtime.glb'
  if not base.exists():continue
  srcd,srcb=read_glb(base);orig_digest=geometry_animation_digest(srcd,srcb);ii=material_body_image(srcd,row['prefix']);image=Image.open(io.BytesIO(viewbytes(srcd,srcb,srcd['images'][ii]['bufferView']))).convert('RGB');src=np.asarray(image)
  colors=selection['palette_srgb'].items() if row['controlled_color'] else [('original',None)]
  for name,rgb in colors:
   dest=T/'assets/runtime'/row['asset_id']/name/'runtime.glb';d=json.loads(json.dumps(srcd));b=srcb;info={}
   if rgb:
    mdir=T/'masks'/row['asset_id'];main=np.asarray(Image.open(mdir/'shirt_main_color.png'))>0;protect=np.asarray(Image.open(mdir/'stripe_detail_protect.png'))>0
    assert src.shape[:2]==main.shape
    lum=srgb_to_linear(src[main])@np.array([.2126,.7152,.0722]);scale=np.clip(lum/np.median(lum),.30,2.25);out=src.copy();out[main]=linear_to_srgb(srgb_to_linear(np.array(rgb))[None,:]*scale[:,None]);buf=io.BytesIO();Image.fromarray(out).save(buf,format='PNG');b=append_image(d,b,ii,buf.getvalue());texture=T/'textures'/row['asset_id']/(name+'.png');texture.parent.mkdir(parents=True,exist_ok=True);texture.write_bytes(buf.getvalue());changed=np.any(src!=out,axis=2)
    assert not np.any(changed&~main);assert np.array_equal(src[protect],out[protect]);info={'palette_srgb':rgb,'changed_pixels':int(changed.sum()),'outside_mask_changes':0,'protected_pixel_changes':0}
   assert geometry_animation_digest(d,b)==orig_digest
   changed_images=[i for i,(a,c) in enumerate(zip(image_payloads(srcd,srcb),image_payloads(d,b))) if a!=c];assert changed_images==([ii] if rgb else [])
   write_glb(dest,d,b);info.update(asset_id=row['asset_id'],color=name,runtime=str(dest.relative_to(T)),mesh_skin_action_contract_sha256=orig_digest,changed_image_indices=changed_images,geometry_uv_bones_actions_unchanged=True,status='pass');proof.append(info)
   json_write(dest.with_name('material_proof.json'),info)
 json_write(T/'evidence/material_variant_equivalence.json',proof);print('VARIANTS_BUILT',len(proof))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('mode',choices=['masks','variants']);a=p.parse_args();selection=json.loads((a.task_root/'registrations/selection.json').read_text());globals()[a.mode](a.task_root,selection)
