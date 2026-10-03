#!/usr/bin/env python3
"""Map existing reviewed source GLB PBR repairs to isolated native UE materials."""
from __future__ import annotations
import argparse,base64,hashlib,json,struct
from pathlib import Path


def glb(path):
    data=Path(path).read_bytes();magic,version,length=struct.unpack_from('<4sII',data)
    if magic!=b'glTF' or version!=2 or length!=len(data):raise ValueError('derived GLB is incomplete')
    offset=12;document=None;binary=None
    while offset<length:
        n,t=struct.unpack_from('<II',data,offset);payload=data[offset+8:offset+8+n];offset+=8+n
        if t==0x4e4f534a:document=json.loads(payload)
        elif t==0x004e4942:binary=payload
    if document is None or binary is None:raise ValueError('derived GLB lacks JSON/BIN')
    return document,binary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--native-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    inventory_bytes=args.inventory.read_bytes();inventory=json.loads(inventory_bytes)
    (args.output/'inventory_snapshot.json').write_bytes(inventory_bytes)
    native={r['asset_id']:r for r in json.loads(args.native_audit.read_text())['records']}
    repairs=[];unresolved=[]
    normalize=lambda n:''.join(c for c in n.lower() if c.isalnum()).removeprefix('mi')
    for asset in inventory['assets']:
        aid=asset['asset_id'];receipt=asset.get('material_receipt')
        row=native.get(aid)
        if not receipt or not row or row['status']!='audited':
            unresolved.append({'asset_id':aid,'reason':'derived_source_or_native_material_audit_missing'});continue
        derived=Path(receipt['derived_path'])
        if hashlib.sha256(derived.read_bytes()).hexdigest()!=receipt['derived_sha256']:raise ValueError('reviewed derived GLB hash changed')
        document,binary=glb(derived);materials=document.get('materials',[])
        for i,material in enumerate(materials):
            expected=normalize(material.get('name',''))
            candidates=[m for m in row['materials'] if normalize(m.get('path','').rsplit('.',1)[-1])==expected]
            if len(candidates)!=1:
                unresolved.append({'asset_id':aid,'source_material_index':i,'source_material_name':material.get('name'),'reason':'native_material_name_mapping_not_unique'});continue
            m=candidates[0]
            if m['class']!='MaterialInstanceConstant' or not m['path'].startswith('/Game/'):
                unresolved.append({'asset_id':aid,'source_material_index':i,'native_material':m['path'],'reason':'native_material_is_not_owned_game_MIC'});continue
            needed={'MetallicFactor','RoughnessFactor'}
            if not needed.issubset(m['scalar_parameter_names']):
                unresolved.append({'asset_id':aid,'source_material_index':i,'reason':'native_PBR_scalar_interface_missing'});continue
            pbr=material.get('pbrMetallicRoughness',{});mr=pbr.get('metallicRoughnessTexture');image_path=None
            if mr:
                texture=document['textures'][mr['index']];image=document['images'][texture['source']]
                if image.get('mimeType') not in ('image/png',None):raise ValueError('derived packed MR image must be PNG')
                if 'bufferView' in image:
                    view=document['bufferViews'][image['bufferView']];begin=view.get('byteOffset',0);png=binary[begin:begin+view['byteLength']]
                elif image.get('uri','').startswith('data:image/png;base64,'):png=base64.b64decode(image['uri'].split(',',1)[1])
                else:raise ValueError('external MR image requires explicit resolution')
                digest=hashlib.sha256(png).hexdigest();image_path=args.output/f'{digest}.png'
                if image_path.exists():
                    if image_path.read_bytes()!=png:raise ValueError('image digest collision')
                else:image_path.write_bytes(png)
            repairs.append({'asset_id':aid,'source_material_index':i,'source_material_name':material.get('name'),
                'native_material':m['path'],'source_derived_glb':str(derived),'source_derived_glb_sha256':receipt['derived_sha256'],
                'metallic_factor':float(pbr.get('metallicFactor',1.)),'roughness_factor':float(pbr.get('roughnessFactor',1.)),
                'packed_mr_png':str(image_path) if image_path else None,
                'packed_mr_sha256':image_path.stem if image_path else None,'basecolor_and_uv_modified':False})
    result={'schema':'kujiale_native_source_pbr_repair_request_v1','inventory_snapshot':str(args.output/'inventory_snapshot.json'),
        'inventory_sha256':hashlib.sha256(inventory_bytes).hexdigest(),'repairs':repairs,'unresolved':unresolved,
        'policy':'copy existing reviewed source GLB PBR factors and packed MR image; preserve native original basecolor, UV, normals, geometry and registered source identities'}
    (args.output/'request.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'material_repairs':len(repairs),'unresolved':len(unresolved),'input_bytes':sum(f.stat().st_size for f in args.output.iterdir() if f.is_file())}))


if __name__=='__main__':main()
