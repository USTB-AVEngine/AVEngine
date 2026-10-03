"""Decode embedded Basis images into an explicit derived GLB for native views.

Adapted from room_selection/demo_textures.py, commit 0465ecd.
The installed renderer lacks BasisImporter. Original assets remain read-only.
The entire original BIN prefix, mesh/accessor/UV/material/sampler/node records
are retained. Only image encoding and equivalent texture source references
change. This is texture decompression, not a new scene or synthetic texture.
"""
from __future__ import annotations
import copy, io, json, os, struct, subprocess
from pathlib import Path
from PIL import Image
from .common import write_json


def read_glb(path):
    data=Path(path).read_bytes()
    magic,version,total=struct.unpack_from('<III',data)
    if magic!=0x46546c67 or version!=2 or total!=len(data):raise ValueError('invalid GLB')
    offset=12;document=None;binary=None
    while offset<len(data):
        length,kind=struct.unpack_from('<II',data,offset);offset+=8
        payload=data[offset:offset+length];offset+=length
        if kind==0x4e4f534a:document=json.loads(payload)
        elif kind==0x004e4942:binary=payload
    if document is None or binary is None:raise ValueError('GLB requires JSON and BIN')
    return document,binary


def decode_rgba_payload(payload):
    width,height,y_flipped=struct.unpack_from('<III',payload)
    if width<=0 or height<=0 or len(payload)!=12+width*height*4:raise ValueError('invalid decoder output')
    image=Image.frombytes('RGBA',(width,height),payload[12:])
    # Magnum BasisImporter presents bottom-left pixel storage: it flips only
    # if !m_y_flipped. PNG import flips top-down storage to bottom-left, so
    # convert an already Y-up Basis payload back to top-down before PNG encoding.
    if y_flipped:image=image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return image,bool(y_flipped)


def prepare_texture_scene(source,output,decoder=None):
    source=Path(source).resolve(strict=True);output=Path(output)
    original,binary=read_glb(source)
    basis_images=[i for i,image in enumerate(original.get('images',[])) if image.get('mimeType')=='image/x-basis']
    if not basis_images:return source,dict(status='original_texture_encoding_supported',source=str(source))
    decoder=Path(decoder or os.environ.get('AVENGINE_DEMO_BASIS_DECODER',''))
    if not decoder.is_file():raise FileNotFoundError('supply explicit AVENGINE_DEMO_BASIS_DECODER for source image/x-basis')
    destination=output/'derived_scenes'/source.parent.name
    destination.mkdir(parents=True,exist_ok=True)
    target=destination/(source.stem+'.png-textures.glb');receipt_path=target.with_suffix('.receipt.json')
    stat=source.stat()
    if target.is_file() and receipt_path.is_file():
        receipt=json.loads(receipt_path.read_text())
        if receipt['source']==str(source) and receipt['source_size_bytes']==stat.st_size and receipt['source_mtime_ns']==stat.st_mtime_ns:
            return target,receipt
    document=copy.deepcopy(original);out_binary=bytearray(binary);images=[]
    for image_id in basis_images:
        image=document['images'][image_id];view=original['bufferViews'][image['bufferView']]
        if view.get('buffer',0)!=0:raise ValueError('external buffer not supported by explicit GLB adapter')
        start=view.get('byteOffset',0);payload=binary[start:start+view['byteLength']]
        compressed=destination/(f'image_{image_id:04d}.basis');raw=destination/(f'image_{image_id:04d}.rgba')
        compressed.write_bytes(payload)
        command=[str(decoder.resolve()),str(compressed.resolve()),str(raw.resolve())]
        result=subprocess.run(command,capture_output=True,text=True,timeout=120)
        if result.returncode:raise RuntimeError('Basis decoder failed: '+result.stdout+result.stderr)
        pixels,y_flipped=decode_rgba_payload(raw.read_bytes())
        stream=io.BytesIO();pixels.save(stream,format='PNG');png=stream.getvalue()
        out_binary.extend(b'\0'*((-len(out_binary))%4));offset=len(out_binary);out_binary.extend(png)
        image['mimeType']='image/png';image['bufferView']=len(document['bufferViews'])
        document['bufferViews'].append(dict(buffer=0,byteOffset=offset,byteLength=len(png)))
        images.append(dict(image_id=image_id,width=pixels.width,height=pixels.height,basis_y_flipped=y_flipped,png_top_down_flip=y_flipped,decoded_rgba=str(raw),command=command))
        print('DECODED_TEXTURE',source.parent.name,image_id,pixels.width,pixels.height,'y_up',y_flipped,flush=True)
    for texture in document.get('textures',[]):
        extensions=texture.get('extensions',{})
        if 'GOOGLE_texture_basis' in extensions:
            texture['source']=extensions.pop('GOOGLE_texture_basis')['source']
            if not extensions:texture.pop('extensions')
    for key in ['extensionsUsed','extensionsRequired']:
        if key in document:document[key]=[v for v in document[key] if v!='GOOGLE_texture_basis']
    out_binary.extend(b'\0'*((-len(out_binary))%4));document['buffers'][0]['byteLength']=len(out_binary)
    unchanged=['nodes','meshes','scenes','scene','accessors','materials','samplers','skins','animations']
    if any(document.get(k)!=original.get(k) for k in unchanged):raise ValueError('geometry, UV, material, or scene graph changed')
    if out_binary[:len(binary)]!=binary:raise ValueError('original BIN data changed')
    encoded=json.dumps(document,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%4)
    total=12+8+len(encoded)+8+len(out_binary)
    with target.open('wb') as f:
        f.write(struct.pack('<III',0x46546c67,2,total));f.write(struct.pack('<II',len(encoded),0x4e4f534a));f.write(encoded);f.write(struct.pack('<II',len(out_binary),0x004e4942));f.write(out_binary)
    receipt=dict(status='decoded_texture_copy',qualification='native visualization using decoded original texture pixels; source geometry and material records preserved',source=str(source),source_size_bytes=stat.st_size,source_mtime_ns=stat.st_mtime_ns,derived_scene=str(target.resolve()),decoder=str(decoder.resolve()),upstream='https://github.com/BinomialLLC/basis_universal/tree/v2_50',images=images,original_bin_prefix_equal=True,unchanged_document_fields=unchanged,orientation_authority='Magnum BasisImporter m_y_flipped semantics; PNG top-down rows converted accordingly')
    write_json(receipt_path,receipt)
    return target,receipt
