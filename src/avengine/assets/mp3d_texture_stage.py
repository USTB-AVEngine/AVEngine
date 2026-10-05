"""Lossless CPU texture decoding into a new MP3D visual input.

This preserves geometry, animation, navigation and acoustic packages. The
production PNG/JPEG precheck is retained. Raw scans and the external decoder
are read only. The explicit Basis Y-flip flag is applied to standard PNG rows, matching
9/9 actual textures from the three previously accepted visual inputs.
Actual raw Basis native RGB comparison remains a separate, unpassed control.
"""
from __future__ import annotations
import copy,hashlib,io,json,struct,subprocess,tempfile
from pathlib import Path

def geometry_digest(document,binary):
    views=set()
    for accessor in document.get('accessors',[]):
        if 'bufferView' in accessor:views.add(accessor['bufferView'])
        for value in accessor.get('sparse',{}).values():
            if isinstance(value,dict) and 'bufferView' in value:views.add(value['bufferView'])
    payload={}
    for i in sorted(views):
        view=document['bufferViews'][i]
        if view.get('buffer',0)!=0:raise ValueError('external geometry buffers unsupported')
        lo=int(view.get('byteOffset',0));hi=lo+int(view['byteLength'])
        if not 0<=lo<=hi<=len(binary):raise ValueError('geometry bufferView outside binary')
        payload[i]={'metadata':{k:v for k,v in view.items() if k not in ('byteOffset','buffer')},
                    'SHA256':hashlib.sha256(memoryview(binary)[lo:hi]).hexdigest()}
    fields=('meshes','nodes','skins','animations','scenes','scene','accessors')
    structure={k:document.get(k) for k in fields}
    return hashlib.sha256(json.dumps({'structure':structure,'geometry_views':payload},sort_keys=True,separators=(',',':')).encode()).hexdigest()

def decode_visual(source,decoder,output):
    from PIL import Image
    from avengine.assets.roster_motion.io import read_glb,write_glb
    from avengine.assets.source_materials import read_glb as read_plain_glb
    source,decoder,output=map(Path,(source,decoder,output))
    if output.exists() or output.resolve()==source.resolve():raise ValueError('new immutable derived visual path required')
    original,binary=read_glb(source);doc=copy.deepcopy(original);data=bytearray(binary);rows=[]
    before=geometry_digest(original,binary)
    for i,image in enumerate(doc.get('images',[])):
        if image.get('mimeType') in ('image/png','image/jpeg'):continue
        if image.get('mimeType')!='image/x-basis' or 'bufferView' not in image:raise ValueError('unsupported nonembedded MP3D texture')
        view=original['bufferViews'][image['bufferView']];lo=int(view.get('byteOffset',0));raw=memoryview(binary)[lo:lo+int(view['byteLength'])]
        output.parent.mkdir(parents=True,exist_ok=True)
        # The retained 20261003 executable has a declared file/file ABI.
        # A later source adapter uses stdin/stdout; it is a different binary.
        with tempfile.TemporaryDirectory(prefix='v7_basis_image_',dir=output.parent) as temporary:
            src=Path(temporary)/'source.basis';dst=Path(temporary)/'decoded.rgba';src.write_bytes(raw)
            subprocess.run([str(decoder),str(src),str(dst)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=60)
            decoded=dst.read_bytes()
        if len(decoded)<8:raise ValueError('truncated legacy Basis decoder response')
        width,height=struct.unpack_from('<II',decoded);count=4*width*height
        if not 0<width<=16384 or not 0<height<=16384:raise ValueError('invalid Basis decoder dimensions')
        if len(decoded)==8+count:header=8;flipped=None
        elif len(decoded)==12+count:
            header=12;flipped=struct.unpack_from('<I',decoded,8)[0]
            if flipped not in (0,1):raise ValueError('invalid Basis Y-flip flag')
        else:raise ValueError('legacy Basis payload length differs')
        if flipped is None:raise ValueError('Basis orientation is unknown; decoder must return the explicit Y-flip flag')
        rgba=decoded[header:];raster=Image.frombytes('RGBA',(width,height),rgba)
        if flipped:raster=raster.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        standard_rgba=raster.tobytes();png=io.BytesIO();raster.save(png,format='PNG',compress_level=3)
        if Image.open(io.BytesIO(png.getvalue())).convert('RGBA').tobytes()!=standard_rgba:raise ValueError('lossless PNG pixel roundtrip failed')
        while len(data)%4:data.append(0)
        start=len(data);encoded=png.getvalue();data.extend(encoded)
        image['bufferView']=len(doc['bufferViews']);image['mimeType']='image/png';image.pop('uri',None)
        doc['bufferViews'].append({'buffer':0,'byteOffset':start,'byteLength':len(encoded)})
        rows.append({'image_index':i,'size':[width,height],'Basis_Y_flipped':None if flipped is None else bool(flipped),'decoder_protocol':'file-input-file-output-RGBA-with-dimensions','decoded_RGBA_SHA256':hashlib.sha256(rgba).hexdigest(),'standard_PNG_RGBA_SHA256':hashlib.sha256(standard_rgba).hexdigest(),'Basis_Y_flip_applied_to_standard_PNG':bool(flipped),'PNG_SHA256':hashlib.sha256(encoded).hexdigest()})
    for texture in doc.get('textures',[]):
        ext=texture.get('extensions',{});basis=ext.pop('GOOGLE_texture_basis',None)
        if basis is not None:texture['source']=basis['source']
        if not ext:texture.pop('extensions',None)
    for key in ('extensionsUsed','extensionsRequired'):
        if key in doc:
            doc[key]=[x for x in doc[key] if x!='GOOGLE_texture_basis']
            if not doc[key]:doc.pop(key)
    doc['buffers']=[{'byteLength':len(data)}]
    if geometry_digest(doc,data)!=before:raise ValueError('decoded visual changed geometry or poses')
    output.parent.mkdir(parents=True,exist_ok=True)
    write_glb(output,doc,bytes(data));observed,observed_binary=read_plain_glb(output)
    if geometry_digest(observed,observed_binary)!=before:raise ValueError('serialized visual geometry changed')
    return {'status':'pass','source':str(source),'source_SHA256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'visual_source':str(output),'visual_SHA256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'geometry_SHA256':before,'geometry_exactly_preserved':True,'PNG_JPEG_production_gate_changed':False,
            'raw_modified':False,'decoder':str(decoder),'decoder_SHA256':hashlib.sha256(decoder.read_bytes()).hexdigest(),
            'images':rows,'actual_native_RGB_equivalence':'not_run'}
