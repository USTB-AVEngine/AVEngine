"""Read-only source survey for the v7 human roster."""
import argparse,collections,hashlib,json,pathlib,subprocess
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--rocketbox',type=pathlib.Path,required=True);p.add_argument('--lx',type=pathlib.Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
records=[]
for cat in ('Adults','Professions','Children'):
 for d in sorted((a.rocketbox/'Assets/Avatars'/cat).iterdir()):
  fbx=list(d.glob('Export/*.fbx'));tex=sorted(d.glob('Textures/*_body_color.tga'))
  if not fbx:continue
  records.append(dict(category=cat,name=d.name,fbx=str(fbx[0]),textures=str(d/'Textures'),body_texture=str(tex[0]) if tex else None,prefix=tex[0].name.replace('_body_color.tga','') if tex else None))
(a.output/'rocketbox_source_inventory.json').write_text(json.dumps(dict(commit=subprocess.check_output(['git','-C',str(a.rocketbox),'rev-parse','HEAD'],text=True).strip(),category_counts=dict(collections.Counter(r['category'] for r in records)),avatars=records),ensure_ascii=False,indent=2)+'\n')
print('ROCKETBOX',dict(collections.Counter(r['category'] for r in records)),flush=True)
legacy=pathlib.Path('/data/datasets/avengine_workspaces/AVEngine/external/SPEAR/tmp/rocketbox_route1_inventory_v1/inventory.json')
if legacy.exists():
 data=json.loads(legacy.read_text());(a.output/'legacy_inventory_summary.json').write_text(json.dumps({k:v for k,v in data.items() if k!='avatars'},indent=2));print('LEGACY',legacy,'keys',list(data),'count',len(data.get('avatars',[])))
 if data.get('avatars'):print('LEGACY_FIRST',json.dumps(data['avatars'][0])[:2200])
lx=[]
for f in sorted(a.lx.glob('*/runtime/*/model.glb')):
 manifests={}
 for role,pattern in [('generation','generation/*/generation_manifest.json'),('pixal','pixal/*/*.manifest.json')]:
  manifests[role]=[]
  for m in sorted(f.parents[2].glob(pattern)):
   data=json.loads(m.read_text());manifests[role].append({'path':str(m),'model':data.get('model'),'pipeline':data.get('pipeline'),'code':data.get('code'),'runtime':data.get('runtime'),'source':data.get('source'),'all_keys':list(data)})
 lx.append(dict(asset_id=f.parents[2].name,path=str(f),sha256=sha(f),standardization=json.loads(f.with_suffix('.manifest.json').read_text()),provenance=manifests))
(a.output/'lx_source_inventory.json').write_text(json.dumps(lx,ensure_ascii=False,indent=2)+'\n')
for r in lx:print('LX',r['asset_id'],r['provenance'])
# Static texture contact sheet for selecting separable casual garments.
from PIL import Image,ImageDraw
adults=[r for r in records if r['category']=='Adults' and 'Adult' in r['name']]
out=Image.new('RGB',(8*220,5*270),'white');draw=ImageDraw.Draw(out)
for i,r in enumerate(adults):
 if not r['body_texture']:continue
 tex=Image.open(r['body_texture']).convert('RGB');tex.thumbnail((210,240));x=(i%8)*220;y=(i//8)*270;out.paste(tex,(x,y+25));draw.text((x+2,y+3),r['name'],fill='black')
out.save(a.output/'adult_texture_contact_sheet.jpg',quality=90)
