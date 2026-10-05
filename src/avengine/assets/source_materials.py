"""PBR materials for inserted Habitat sources; scanned stages are untouched.

The production multi-actor path inherited Phong AO templates. In the installed
runtime that route produces white specular spots on texture-backed fur. Read
ordinary PNG/JPEG glTF materials through PBR, repair only source material fields
in a separate versioned GLB, and measure rendered source pixels. Geometry,
skinning, animations, the source package and its admission state stay intact.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np
from PIL import Image

MATERIAL_REVISION = 'source_materials_v1'
PBR_HANDLE = 'avengine_inserted_source_pbr_v1'
COMPRESSED_EXTENSIONS = {'GOOGLE_texture_basis', 'KHR_texture_basisu'}


class SourceMaterialError(ValueError):
    pass


def read_glb(path: str | Path) -> tuple[dict, bytes]:
    path = Path(path)
    if '.basis.' in path.name.lower():
        raise SourceMaterialError(f'compressed Basis GLB cannot enter Habitat RGB: {path}')
    raw = path.read_bytes()
    if len(raw) < 20 or struct.unpack_from('<4sII', raw) != (b'glTF', 2, len(raw)):
        raise SourceMaterialError(f'expected a complete GLB v2: {path}')
    n, tag = struct.unpack_from('<I4s', raw, 12)
    if tag != b'JSON' or 20+n > len(raw):
        raise SourceMaterialError(f'GLB JSON chunk is invalid: {path}')
    doc = json.loads(raw[20:20+n])
    forbidden = COMPRESSED_EXTENSIONS.intersection(doc.get('extensionsUsed', []))
    forbidden.update(COMPRESSED_EXTENSIONS.intersection(doc.get('extensionsRequired', [])))
    for texture in doc.get('textures', []):
        forbidden.update(COMPRESSED_EXTENSIONS.intersection(texture.get('extensions', {})))
    if forbidden or any(i.get('mimeType') in {'image/ktx2', 'image/basis'} for i in doc.get('images', [])):
        raise SourceMaterialError(f'unsupported compressed source textures {sorted(forbidden)}: {path}')
    offset = 20+n
    if offset == len(raw):
        return doc, b''
    if offset+8 > len(raw):
        raise SourceMaterialError('truncated GLB BIN chunk')
    size, tag = struct.unpack_from('<I4s', raw, offset)
    if tag != b'BIN\0' or offset+8+size != len(raw):
        raise SourceMaterialError('expected one complete GLB BIN chunk')
    return doc, raw[offset+8:]


def image_bytes(doc: Mapping, blob: bytes, index: int, path: Path) -> bytes:
    image = doc['images'][index]
    if 'bufferView' in image:
        v = doc['bufferViews'][image['bufferView']]
        if v.get('buffer', 0) != 0:
            raise SourceMaterialError('external GLB buffers are unsupported')
        s = v.get('byteOffset', 0); e = s+v['byteLength']
        if e > len(blob):
            raise SourceMaterialError('texture bufferView exceeds BIN')
        return blob[s:e]
    uri = image.get('uri', '')
    if uri.startswith('data:'):
        return base64.b64decode(uri.split(',', 1)[1], validate=True)
    p = (path.parent/uri).resolve()
    if not uri or not p.is_relative_to(path.parent.resolve()) or not p.is_file():
        raise SourceMaterialError(f'missing or escaping texture URI: {uri}')
    return p.read_bytes()


def texture_pixels(doc: Mapping, blob: bytes, texture: Mapping, path: Path) -> np.ndarray:
    source = doc['textures'][texture['index']].get('source')
    if not isinstance(source, int):
        raise SourceMaterialError('texture has no supported image source')
    try:
        with Image.open(io.BytesIO(image_bytes(doc, blob, source, path))) as image:
            if image.format not in {'PNG', 'JPEG'}:
                raise SourceMaterialError(f'unsupported source texture format: {image.format}')
            return np.array(image.convert('RGBA'))
    except (OSError, KeyError, IndexError) as exc:
        raise SourceMaterialError(f'cannot load source texture: {path}: {exc}') from exc


def material_profile(asset_id: str, entity_class: str, name: str = '') -> dict:
    label = name.lower()
    if entity_class == 'articulated_animal':
        return {'surface': 'fur', 'metallic': 0.0, 'roughness_min': .85, 'max_highlight_ratio': .02}
    if entity_class == 'articulated_human':
        if 'tools' in label:
            return {'surface': 'metal_tools', 'metallic': .65, 'roughness_min': .60, 'max_highlight_ratio': .12}
        floor = .65 if any(s in label for s in ['head', 'helmet']) else .78
        return {'surface': 'skin_or_cloth', 'metallic': 0.0, 'roughness_min': floor, 'max_highlight_ratio': .04}
    # These source IDs describe finishes, not breed-specific appearance edits.
    if 'floor_drain' in asset_id:
        return {'surface': 'brushed_metal', 'metallic': .65, 'roughness_min': .60, 'max_highlight_ratio': .12}
    floor = .82 if any(s in asset_id for s in ['fabric', 'walnut', 'black_ash', 'masonry', 'matte']) else .65
    return {'surface': 'dielectric_or_mixed', 'metallic': 0.0, 'roughness_min': floor, 'max_highlight_ratio': .08}


def _accessor(doc: Mapping, blob: bytes, index: int) -> np.ndarray:
    a = doc['accessors'][index]
    if 'sparse' in a:
        raise SourceMaterialError('sparse UV accessor requires explicit conversion')
    v = doc['bufferViews'][a['bufferView']]
    dtype = {5126: '<f4', 5121: 'u1', 5123: '<u2', 5125: '<u4'}[a['componentType']]
    channels = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[a['type']]
    item = np.dtype(dtype).itemsize
    result = np.ndarray((a['count'], channels), dtype=dtype, buffer=blob,
                        offset=v.get('byteOffset', 0)+a.get('byteOffset', 0),
                        strides=(v.get('byteStride', channels*item), item)).astype(float)
    if a.get('normalized') and a['componentType'] != 5126:
        result /= np.iinfo(np.dtype(dtype)).max
    return result


def source_palette(doc: Mapping, blob: bytes, path: Path) -> np.ndarray:
    """Sample source texels on actual mesh UVs, excluding unused atlas background."""
    colors = []
    for mesh in doc.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            if 'material' not in primitive:
                continue
            mat = doc['materials'][primitive['material']]
            pbr = mat.get('pbrMetallicRoughness', {})
            factor = np.array(pbr.get('baseColorFactor', [1,1,1,1]))[:3]
            texture = pbr.get('baseColorTexture')
            if texture is None:
                colors.append(factor[None, :]); continue
            key = f'TEXCOORD_{texture.get("texCoord", 0)}'
            if key not in primitive.get('attributes', {}):
                raise SourceMaterialError(f'base color texture has no {key}: {path}')
            uv = _accessor(doc, blob, primitive['attributes'][key])
            if 'indices' in primitive and primitive.get('mode', 4) == 4:
                ids = _accessor(doc, blob, primitive['indices']).astype(int).reshape(-1,3)
                ids = ids[::max(1, len(ids)//2048)]
                uv = uv[ids].mean(axis=1)
            else:
                uv = uv[::max(1,len(uv)//2048)]
            transform = texture.get('extensions', {}).get('KHR_texture_transform', {})
            if transform:
                angle = float(transform.get('rotation', 0)); c = np.cos(angle); s = np.sin(angle)
                uv = (uv*np.array(transform.get('scale',[1,1])))@np.array([[c,s],[-s,c]])+transform.get('offset',[0,0])
            pixels = texture_pixels(doc, blob, texture, path)
            uv = np.mod(uv, 1.0)
            x = np.minimum((uv[:,0]*pixels.shape[1]).astype(int),pixels.shape[1]-1)
            y = np.minimum((uv[:,1]*pixels.shape[0]).astype(int),pixels.shape[0]-1)
            sample = pixels[y,x,:3].astype(float)/255*factor
            sample = sample[pixels[y,x,3] > 128]
            if len(sample): colors.append(sample)
    if not colors:
        raise SourceMaterialError(f'no source color samples: {path}')
    result = np.concatenate(colors)
    # Compact palette; mesh-face sampling retains multi-material color coverage.
    return result[::max(1,len(result)//512)][:512]


def inspect_source_glb(path: str | Path) -> dict:
    path = Path(path); doc, blob = read_glb(path)
    decoded_images = []
    for index, image in enumerate(doc.get('images', [])):
        data = image_bytes(doc, blob, index, path)
        with Image.open(io.BytesIO(data)) as decoded:
            if decoded.format not in {'PNG', 'JPEG'}:
                raise SourceMaterialError(f'unsupported source texture format: {decoded.format}')
            decoded.load()
            decoded_images.append({'index': index, 'name': image.get('name'),
                                   'declared_mime': image.get('mimeType'),
                                   'decoded_format': decoded.format,
                                   'size_wh': list(decoded.size), 'bytes': len(data)})
    materials = []
    for mat in doc.get('materials', []):
        pbr = mat.get('pbrMetallicRoughness', {})
        for key in ['baseColorTexture', 'metallicRoughnessTexture']:
            if key in pbr: texture_pixels(doc, blob, pbr[key], path)
        materials.append({'name': mat.get('name'), 'base_color_texture': pbr.get('baseColorTexture'),
                          'base_color_factor': pbr.get('baseColorFactor',[1,1,1,1]),
                          'metallic_factor': pbr.get('metallicFactor',1),
                          'roughness_factor': pbr.get('roughnessFactor',1),
                          'metallic_roughness_texture': pbr.get('metallicRoughnessTexture'),
                          'alpha_mode': mat.get('alphaMode','OPAQUE'), 'alpha_cutoff': mat.get('alphaCutoff',.5),
                          'extensions': mat.get('extensions',{})})
    return {'path': str(path), 'format':'GLB v2', 'bytes':path.stat().st_size,
            'extensions_used':doc.get('extensionsUsed',[]), 'compressed':False,
            'images':doc.get('images',[]), 'decoded_images':decoded_images, 'materials':materials}


def prepare_source_glb(path: str | Path, *, asset_id: str, entity_class: str,
                       output_directory: str | Path) -> tuple[Path, dict]:
    path = Path(path).resolve(); doc, blob = read_glb(path)
    original = deepcopy(doc)
    profiles = [material_profile(asset_id,entity_class,m.get('name','')) for m in doc.get('materials',[])]
    if not profiles: raise SourceMaterialError(f'inserted source has no material: {path}')
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    external_images = [{'index': index, 'sha256': hashlib.sha256(image_bytes(doc,blob,index,path)).hexdigest()}
                       for index,image in enumerate(doc.get('images', []))
                       if 'uri' in image and not image['uri'].startswith('data:')]
    recipe_inputs = {'profiles':profiles,'external_images':external_images} if external_images else profiles
    recipe = hashlib.sha256(json.dumps(recipe_inputs,sort_keys=True).encode()).hexdigest()[:12]
    target_dir = Path(output_directory)/MATERIAL_REVISION/(identity[:20]+'_'+recipe)
    target_dir.mkdir(parents=True,exist_ok=True)
    target = target_dir/'visual.materials_v1.glb'; receipt_path = target_dir/'materials.json'
    if target.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['source_sha256'] != identity or receipt['profiles'] != profiles:
            raise SourceMaterialError('derived-material cache identity differs')
        if hashlib.sha256(target.read_bytes()).hexdigest() != receipt['derived_sha256']:
            raise SourceMaterialError('derived-material cache checksum differs')
        return target, receipt
    palette = source_palette(original,blob,path)
    image_changes = []
    for index,(mat,profile) in enumerate(zip(doc['materials'],profiles,strict=True)):
        pbr = mat.setdefault('pbrMetallicRoughness',{})
        pbr['metallicFactor'] = profile['metallic']
        texture = pbr.get('metallicRoughnessTexture')
        if texture:
            pixels = texture_pixels(original,blob,texture,path)
            factor = float(pbr.get('roughnessFactor',1))
            effective = pixels[:,:,1].astype(float)/255*factor
            pixels[:,:,1] = np.rint(np.maximum(effective,profile['roughness_min'])*255).clip(0,255).astype('uint8')
            # Metallic factor and B are independent. Uniform metal surfaces must
            # not inherit a paint-generation roughness image's incidental B.
            if profile['metallic'] > 0: pixels[:,:,2] = 255
            stream = io.BytesIO(); Image.fromarray(pixels).save(stream,format='PNG')
            data = stream.getvalue(); blob += b'\0'*(-len(blob)%4)
            view = len(doc['bufferViews']); doc['bufferViews'].append({'buffer':0,'byteOffset':len(blob),'byteLength':len(data)})
            blob += data; source = len(doc['images']); doc['images'].append({'bufferView':view,'mimeType':'image/png','name':f'material_{index}_roughness_v1'})
            tex = deepcopy(doc['textures'][texture['index']]); tex['source'] = source
            tex_index = len(doc['textures']); doc['textures'].append(tex)
            pbr['metallicRoughnessTexture']['index'] = tex_index; pbr['roughnessFactor'] = 1.0
            image_changes.append({'material_index':index,'roughness_min':profile['roughness_min']})
        else:
            pbr['roughnessFactor'] = max(float(pbr.get('roughnessFactor',1)),profile['roughness_min'])
    # Embed all external images in the derived GLB. Relative source paths must
    # remain usable when the new GLB lives in an episode/cache directory.
    for image_index, image in enumerate(doc.get('images',[])[:len(original.get('images',[]))]):
        if 'uri' in image:
            data = image_bytes(original,blob,image_index,path); blob += b'\0'*(-len(blob)%4)
            view = len(doc['bufferViews']); doc['bufferViews'].append({'buffer':0,'byteOffset':len(blob),'byteLength':len(data)})
            blob += data; image.pop('uri'); image['bufferView'] = view
            with Image.open(io.BytesIO(data)) as im: image['mimeType'] = 'image/png' if im.format=='PNG' else 'image/jpeg'
    doc['buffers'][0]['byteLength'] = len(blob)
    payload = json.dumps(doc,separators=(',',':')).encode(); payload += b' '*(-len(payload)%4)
    blob += b'\0'*(-len(blob)%4)
    raw = struct.pack('<4sII',b'glTF',2,28+len(payload)+len(blob))+struct.pack('<I4s',len(payload),b'JSON')+payload+struct.pack('<I4s',len(blob),b'BIN\0')+blob
    receipt = {'revision':MATERIAL_REVISION,'source_path':str(path),'source_sha256':identity,
               'derived_path':str(target),'derived_sha256':hashlib.sha256(raw).hexdigest(),
               'profiles':profiles,'material_before':original['materials'],'material_after':doc['materials'],
               'source_palette_srgb':palette.tolist(),'image_changes':image_changes,
               'geometry_skin_animation_unchanged':all(original.get(k)==doc.get(k) for k in ['meshes','nodes','skins','animations','accessors']),
               'original_bin_prefix_unchanged':blob.startswith(read_glb(path)[1]),
               'external_images':external_images,
               'shader_type':'pbr','environment_reflections':False}
    # Concurrent capture workers share a cache only when explicitly configured.
    # Same content/recipe emits identical bytes; atomic publication avoids readers
    # seeing a partial GLB. Never replace an original input.
    import os,tempfile
    for destination,data in [(target,raw),(receipt_path,json.dumps(receipt,indent=2).encode())]:
        fd,temp = tempfile.mkstemp(prefix='.materials-',dir=target_dir)
        try:
            with os.fdopen(fd,'wb') as stream: stream.write(data)
            try: os.link(temp,destination)
            except FileExistsError:
                if destination.read_bytes()!=data: raise SourceMaterialError('material cache publication differs')
        finally: Path(temp).unlink(missing_ok=True)
    return target,receipt


def configure_source_pbr(configuration: Any, habitat_sim: Any, *, metadata_mediator: Any = None) -> dict:
    # A direct-GLB stage can contain PBR scanning materials. Such callers load
    # the stage first and pass its actual mediator, then configure newly
    # inserted sources. Existing stage drawables retain their original handle.
    mediator = (metadata_mediator if metadata_mediator is not None else
                habitat_sim.metadata.MetadataMediator(configuration.sim_cfg))
    manager = mediator.pbr_shader_template_manager
    attributes = manager.create_template(PBR_HANDLE,register_template=False)
    attributes.enable_ibl = False
    attributes.enable_direct_lights = True
    attributes.map_mat_txtr_to_linear = True
    attributes.map_output_to_srgb = True
    # Native light intensities and exposure defaults are retained. No scene
    # light, postprocess gain, or source base color is dimmed to hide gloss.
    if int(manager.register_template(attributes,PBR_HANDLE)) < 0 or not mediator.set_curr_default_pbr_attributes_handle(PBR_HANDLE):
        raise SourceMaterialError('cannot configure source PBR shader')
    configuration.metadata_mediator = mediator
    return {'handle':PBR_HANDLE,'shader_type':'pbr','enable_ibl':False,
            'map_mat_txtr_to_linear':True,'map_output_to_srgb':True,
            'direct_light_intensity':float(attributes.direct_light_intensity)}


def prepare_texture_reference_glb(path: str | Path, output: str | Path) -> Path:
    """Diagnostic unlit original-color GLB, independent of PBR→Phong heuristics."""
    path = Path(path).resolve(); doc, blob = read_glb(path)
    for mat in doc.get('materials', []):
        pbr = mat.setdefault('pbrMetallicRoughness', {})
        pbr['metallicFactor'] = 0.0; pbr['roughnessFactor'] = 1.0
        pbr.pop('metallicRoughnessTexture', None)
        mat.setdefault('extensions', {})['KHR_materials_unlit'] = {}
    used = doc.setdefault('extensionsUsed', [])
    if 'KHR_materials_unlit' not in used: used.append('KHR_materials_unlit')
    for index, image in enumerate(doc.get('images', [])):
        if 'uri' not in image: continue
        data = image_bytes(doc, blob, index, path); blob += b'\0' * (-len(blob) % 4)
        view = len(doc['bufferViews']); doc['bufferViews'].append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': len(data)})
        blob += data; image.pop('uri'); image['bufferView'] = view
        with Image.open(io.BytesIO(data)) as decoded:
            if decoded.format not in {'PNG', 'JPEG'}: raise SourceMaterialError('unsupported reference texture')
            image['mimeType'] = 'image/png' if decoded.format == 'PNG' else 'image/jpeg'
    doc['buffers'][0]['byteLength'] = len(blob)
    payload = json.dumps(doc, separators=(',', ':')).encode(); payload += b' ' * (-len(payload) % 4)
    blob += b'\0' * (-len(blob) % 4)
    raw = struct.pack('<4sII', b'glTF', 2, 28 + len(payload) + len(blob)) + struct.pack('<I4s', len(payload), b'JSON') + payload + struct.pack('<I4s', len(blob), b'BIN\0') + blob
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    return output


def prepare_source_ao_config(config_path: str | Path, urdf_path: str | Path,
                             render_asset_path: str | Path, output_directory: str | Path) -> Path:
    """Declare the derived render asset with unchanged AO settings and URDF."""
    doc = json.loads(Path(config_path).read_text())
    doc['render_asset'] = str(Path(render_asset_path).resolve())
    doc['urdf_filepath'] = str(Path(urdf_path).resolve())
    data = json.dumps(doc, sort_keys=True, indent=2).encode()
    key = hashlib.sha256(data).hexdigest()[:20]
    folder = Path(output_directory); folder.mkdir(parents=True, exist_ok=True)
    destination = folder/f'actor_{key}.materials_v1.ao_config.json'
    import os, tempfile
    fd, name = tempfile.mkstemp(prefix='.ao-materials-', dir=folder)
    try:
        with os.fdopen(fd, 'wb') as stream: stream.write(data)
        try: os.link(name, destination)
        except FileExistsError:
            if destination.read_bytes() != data: raise SourceMaterialError('AO material cache differs')
    finally: Path(name).unlink(missing_ok=True)
    return destination


def measure_source_appearance(rgb: np.ndarray, mask: np.ndarray, *, palette: np.ndarray,
                              max_highlight_ratio: float, reference_rgb: np.ndarray | None = None,
                              metallic_surface: bool = False,
                              strict: bool = True) -> dict:
    rgb = np.asarray(rgb)[...,:3].astype(float)/255; mask = np.asarray(mask,dtype=bool)
    if mask.shape != rgb.shape[:2]: raise SourceMaterialError('appearance mask/image shapes differ')
    pixels = rgb[mask]; n = len(pixels)
    if n < 100:
        result = {'status':'not_run','visible_pixels':n,'reason':'fewer_than_100_source_pixels'}
        if strict: raise SourceMaterialError(str(result))
        return result
    palette = np.asarray(palette,dtype=float)
    if not palette.ndim==2 or palette.shape[1]!=3 or not len(palette): raise SourceMaterialError('source color palette is missing')
    chroma = pixels/np.maximum(pixels.sum(axis=1,keepdims=True),.03)
    if reference_rgb is not None:
        reference = np.asarray(reference_rgb)[...,:3].astype(float)/255
        if reference.shape != rgb.shape: raise SourceMaterialError('source reference/image shapes differ')
        expected = reference[mask]
        expected_chroma = expected/np.maximum(expected.sum(axis=1,keepdims=True),.03)
        difference = np.linalg.norm(chroma-expected_chroma,axis=1)
        # Hue is undefined at black. A neutral light contribution to black fur
        # must be judged by RGB magnitude, not by distance from (0,0,0) hue.
        # Normalizing tiny RGB values overstates hue changes caused by neutral
        # specular light on black/tinted-black fur. Preserve full sensitivity
        # above 0.1 sRGB. Use squared signal confidence below that level so a
        # small neutral light contribution cannot dominate the hue statistic.
        # Absolute RGB error still gates all nonmetal pixels, including black.
        difference *= np.clip(expected.mean(axis=1)/.1, 0.0, 1.0)**2
        expected_luma = expected.mean(axis=1)
        absolute = np.linalg.norm(pixels-expected,axis=1)/np.sqrt(3)
    else:
        pc = palette/np.maximum(palette.sum(axis=1,keepdims=True),.03)
        difference = np.concatenate([np.sqrt(((chunk[:,None]-pc[None])**2).sum(axis=2).min(axis=1)) for chunk in np.array_split(chroma,max(1,n//4096))])
        expected_luma = np.full(n,float(np.quantile(palette.mean(axis=1),.95)))
        absolute = None
        if expected_luma[0] <= .02:
            difference = np.zeros(n)
            absolute = np.linalg.norm(pixels,axis=1)/np.sqrt(3)
    luminance = pixels.mean(axis=1)
    # Hue is also undefined for rendered black pixels (e.g. hair, back faces
    # or deep shadows). Zero RGB has no normalized RGB color to compare with
    # a bright palette. Keep these in highlight/RGB statistics, and refuse a
    # wholly unobservable colored source instead of silently accepting black.
    dark_render = luminance <= .02
    difference[dark_render] = 0.0
    unobservable_color = (reference_rgb is None and float(np.quantile(palette.mean(axis=1),.95)) > .1
                          and float(np.quantile(luminance,.95)) <= .02)
    highlight = (pixels.min(axis=1)>.70)&(luminance>expected_luma*1.35+.05)&(luminance-expected_luma>.12)
    ratio = float(highlight.mean())
    p50,p95 = [float(v) for v in np.quantile(difference,[.5,.95])]
    failures = []
    if unobservable_color: failures.append('source_texture_color_unobservable')
    if ratio>max_highlight_ratio: failures.append('excess_source_highlights')
    if p50>.12 or p95>.28: failures.append('source_texture_color_mismatch')
    # A metal base-color map describes specular reflectance, not unlit diffuse
    # brightness. Its lit RGB can legitimately differ greatly in magnitude.
    # Retain raw RMSE and enforce hue/highlight gates; keep the absolute check
    # for dielectrics and black-source fallback detection.
    check_absolute = absolute is not None and not (metallic_surface and reference_rgb is not None)
    if check_absolute and float(np.median(absolute))>.35: failures.append('source_texture_rgb_error')
    result = {'status':'fail' if failures else 'pass','visible_pixels':n,
              'highlight_ratio':ratio,'texture_chromaticity_error_p50':p50,'texture_chromaticity_error_p95':p95,
              'texture_rgb_rmse':None if absolute is None else float(np.sqrt(np.mean(absolute**2))),
              'thresholds':{'max_highlight_ratio':max_highlight_ratio,'max_chromaticity_p50':.12,'max_chromaticity_p95':.28,'max_median_rgb_error':.35 if check_absolute else None},
              'metric_revision':'source_image_metrics_v3','metallic_surface':metallic_surface,
              'rendered_dark_pixel_ratio':float(dark_render.mean()),'color_observable':not unobservable_color,
              'reference_dark_color_full_weight_srgb':.1 if reference_rgb is not None else None,
              'reference_dark_confidence_exponent':2 if reference_rgb is not None else None,
              'reference_kind':'same_view_original_texture' if reference_rgb is not None else 'original_mesh_uv_palette',
              'failures':failures}
    if failures and strict: raise SourceMaterialError(json.dumps(result))
    return result
