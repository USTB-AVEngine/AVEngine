"""Regression: source texture loss, Phong-like highlights and unsafe imports."""
import io,json,struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import pytest
from avengine.assets.source_materials import (
    SourceMaterialError,read_glb,prepare_source_glb,texture_pixels,measure_source_appearance,
)

pytestmark=pytest.mark.fast_unit


def source_glb(tmp_path, *, external=False):
    # One colored textured triangle, low-roughness map, and retained skin/animation
    # declarations. The regression must preserve all geometry and color bytes.
    uv=np.array([[.15,.15],[.8,.15],[.4,.8]],dtype='<f4').tobytes()
    color=np.full((8,8,4),[150,40,20,255],dtype='uint8')
    rough=np.full((8,8,4),[255,50,255,255],dtype='uint8')
    streams=[]
    for arr in [color,rough]:
        b=io.BytesIO();Image.fromarray(arr).save(b,format='PNG');streams.append(b.getvalue())
    blob=uv;views=[{'buffer':0,'byteOffset':0,'byteLength':len(uv)}];images=[]
    for i,data in enumerate(streams):
        blob+=b'\0'*(-len(blob)%4);views.append({'buffer':0,'byteOffset':len(blob),'byteLength':len(data)});blob+=data
        images.append({'bufferView':i+1,'mimeType':'image/png'})
    if external:
        (tmp_path/'coat.png').write_bytes(streams[0]);images[0]={'uri':'coat.png'}
    doc={'asset':{'version':'2.0'},'buffers':[{'byteLength':len(blob)}],'bufferViews':views,
         'accessors':[{'bufferView':0,'componentType':5126,'count':3,'type':'VEC2'}],
         'images':images,'textures':[{'source':0},{'source':1}],
         'materials':[{'name':'fur','pbrMetallicRoughness':{'baseColorTexture':{'index':0},'metallicRoughnessTexture':{'index':1}}}],
         'meshes':[{'primitives':[{'material':0,'attributes':{'TEXCOORD_0':0}}]}],
         'nodes':[{'mesh':0}],'skins':[{'joints':[0]}],'animations':[{'name':'idle','samplers':[]}],
         'scenes':[{'nodes':[0]}],'scene':0}
    def write(doc,path):
        j=json.dumps(doc).encode();j+=b' '*(-len(j)%4);b=blob+b'\0'*(-len(blob)%4)
        path.write_bytes(struct.pack('<4sII',b'glTF',2,len(j)+len(b)+28)+struct.pack('<I4s',len(j),b'JSON')+j+struct.pack('<I4s',len(b),b'BIN\0')+b)
    path=tmp_path/'source.glb';write(doc,path)
    return path,write


@pytest.mark.parametrize('external',[False,True])
def test_repair_keeps_source_color_geometry_skin_and_animation(tmp_path,external):
    source,_=source_glb(tmp_path,external=external);original_bytes=source.read_bytes();before,bin0=read_glb(source)
    target,receipt=prepare_source_glb(source,asset_id='test_cat',entity_class='articulated_animal',output_directory=tmp_path/'derived')
    after,bin1=read_glb(target)
    assert source.read_bytes()==original_bytes and target!=source and bin1.startswith(bin0)
    for key in ['meshes','nodes','accessors','skins','animations']:assert before[key]==after[key]
    p0=texture_pixels(before,bin0,before['materials'][0]['pbrMetallicRoughness']['baseColorTexture'],source)
    p1=texture_pixels(after,bin1,after['materials'][0]['pbrMetallicRoughness']['baseColorTexture'],target)
    np.testing.assert_array_equal(p0,p1)
    pbr=after['materials'][0]['pbrMetallicRoughness'];assert pbr['metallicFactor']==0
    mr=texture_pixels(after,bin1,pbr['metallicRoughnessTexture'],target)
    assert mr[:,:,1].min()/255>=.84 and receipt['geometry_skin_animation_unchanged']
    assert 'uri' not in after['images'][0]


@pytest.mark.parametrize('extension',['GOOGLE_texture_basis','KHR_texture_basisu'])
def test_basis_extension_rejected_before_native_import(tmp_path,extension):
    source,write=source_glb(tmp_path);doc,_=read_glb(source);doc['extensionsUsed']=[extension];write(doc,source)
    with pytest.raises(SourceMaterialError,match='compressed'):read_glb(source)


def test_basis_filename_rejected_and_missing_color_texture_not_silently_gray(tmp_path):
    source,write=source_glb(tmp_path)
    p=tmp_path/'scene.basis.glb';p.write_bytes(source.read_bytes())
    with pytest.raises(SourceMaterialError,match='Basis'):read_glb(p)
    doc,_=read_glb(source);doc['images'][0]={'uri':'missing.png'};write(doc,source)
    with pytest.raises(SourceMaterialError,match='missing'):prepare_source_glb(source,asset_id='cat',entity_class='articulated_animal',output_directory=tmp_path/'out')


def test_colored_texture_gray_fallback_and_specular_spots_fail_acceptance():
    mask=np.ones((40,40),bool);ref=np.full((40,40,3),[150,40,20],dtype='uint8');palette=np.array([[150,40,20]])/255
    gray=np.full_like(ref,100)
    with pytest.raises(SourceMaterialError,match='color_mismatch'):measure_source_appearance(gray,mask,palette=palette,max_highlight_ratio=.02,reference_rgb=ref)
    spotted=ref.copy();spotted[:4]=255
    with pytest.raises(SourceMaterialError,match='highlights'):measure_source_appearance(spotted,mask,palette=palette,max_highlight_ratio=.02,reference_rgb=ref)
    assert measure_source_appearance(ref,mask,palette=palette,max_highlight_ratio=.02,reference_rgb=ref)['status']=='pass'


def test_white_fur_does_not_count_as_a_novel_highlight_and_offscreen_is_not_pass():
    ref=np.full((40,40,3),240,dtype='uint8');mask=np.ones((40,40),bool)
    assert measure_source_appearance(ref,mask,palette=np.array([[240]*3])/255,max_highlight_ratio=.02,reference_rgb=ref)['highlight_ratio']==0
    assert measure_source_appearance(ref,~mask,palette=np.array([[240]*3])/255,max_highlight_ratio=.02,strict=False)['status']=='not_run'


def test_black_hue_is_undefined_but_gray_fallback_is_still_rejected():
    mask=np.ones((40,40),bool);ref=np.zeros((40,40,3),dtype='uint8');palette=np.zeros((1,3))
    dim=np.full_like(ref,30)
    assert measure_source_appearance(dim,mask,palette=palette,max_highlight_ratio=.02,reference_rgb=ref)['status']=='pass'
    gray=np.full_like(ref,128)
    for reference in [ref,None]:
        with pytest.raises(SourceMaterialError,match='rgb_error'):
            measure_source_appearance(gray,mask,palette=palette,max_highlight_ratio=.02,reference_rgb=reference)


def test_tinted_black_fur_neutral_light_and_metal_shadow_have_valid_color():
    mask=np.ones((40,40),bool);ref=np.full((40,40,3),[5,12,25],dtype='uint8');lit=np.full_like(ref,[34,35,37])
    assert measure_source_appearance(lit,mask,palette=np.array([[5,12,25]])/255,max_highlight_ratio=.02,reference_rgb=ref)['status']=='pass'
    silver=np.full_like(ref,210);shadow=np.full_like(ref,85)
    assert measure_source_appearance(shadow,mask,palette=np.array([[210]*3])/255,max_highlight_ratio=.12,reference_rgb=silver,metallic_surface=True)['status']=='pass'
    copper=np.full_like(ref,[210,130,25])
    with pytest.raises(SourceMaterialError,match='color_mismatch'):
        measure_source_appearance(shadow,mask,palette=np.array([[210,130,25]])/255,max_highlight_ratio=.12,reference_rgb=copper,metallic_surface=True)


def test_black_shadow_has_no_hue_but_a_colored_source_cannot_disappear_into_black():
    mask=np.ones((40,40),bool);palette=np.array([[150,40,20]])/255
    rgb=np.full((40,40,3),[150,40,20],dtype='uint8');rgb[:10]=0
    measured=measure_source_appearance(rgb,mask,palette=palette,max_highlight_ratio=.04)
    assert measured['status']=='pass' and measured['rendered_dark_pixel_ratio']==.25
    with pytest.raises(SourceMaterialError,match='color_unobservable'):
        measure_source_appearance(np.zeros_like(rgb),mask,palette=palette,max_highlight_ratio=.04)


def test_shared_derived_cache_concurrent_publication(tmp_path):
    source,_=source_glb(tmp_path)
    def build(_):return prepare_source_glb(source,asset_id='cat',entity_class='articulated_animal',output_directory=tmp_path/'out')
    with ThreadPoolExecutor(max_workers=2) as executor:results=list(executor.map(build,[0,1]))
    assert results[0][0]==results[1][0] and results[0][1]==results[1][1]
    read_glb(results[0][0])


def test_versioned_ao_config_keeps_scale_and_changes_the_loaded_asset(tmp_path):
    from avengine.assets.source_materials import prepare_source_ao_config
    config=tmp_path/'original.ao_config.json'; original={'render_asset':'old.glb','urdf_filepath':'old.urdf',
                                                       'uniform_scale':.0123,'shader_type':'phong'}
    config.write_text(json.dumps(original));before=config.read_bytes()
    target=prepare_source_ao_config(config,tmp_path/'actor.urdf',tmp_path/'derived.glb',tmp_path/'cache')
    loaded=json.loads(target.read_text())
    assert config.read_bytes()==before and loaded['uniform_scale']==original['uniform_scale']
    assert loaded['render_asset']==str(tmp_path/'derived.glb') and target!=config


def test_cache_detects_a_changed_external_texture_and_corrupted_derived_glb(tmp_path):
    source,_=source_glb(tmp_path,external=True)
    def build():return prepare_source_glb(source,asset_id='cat',entity_class='articulated_animal',output_directory=tmp_path/'cache')
    first,_=build()
    Image.new('RGBA',(8,8),(20,80,160,255)).save(tmp_path/'coat.png')
    second,_=build();assert first!=second
    second.write_bytes(second.read_bytes()[:-1]+b'x')
    with pytest.raises(SourceMaterialError,match='checksum'):build()


def test_matrix_gate_rejects_a_missing_house_and_failed_materials(tmp_path):
    from tools.assets.check_source_material_matrix import check_results
    inventory = {'assets': [{'asset_id': 'colored_source','source':{'path':'original.glb'},'derived_path':'derived.glb'}]}
    metric = {'status': 'pass', 'visible_pixels': 400, 'failures': []}
    metric['runtime_palette_check'] = {'status': 'pass', 'visible_pixels': 400, 'failures': []}
    for house in ['mp3d', 'hm3d']:
        folder = tmp_path/house/'colored_source'; folder.mkdir(parents=True)
        result = {'asset_id': 'colored_source', 'house': house, 'status': 'pass',
                  'camera_position': [0,1,2], 'camera_rotation': [0,0,0,1],
                  'phase_readbacks':{'before':{'native_render_asset_fullpath':'original.glb'},
                                    'after':{'native_render_asset_fullpath':'derived.glb'}},
                  'scanned_background_changed_pixels': 0, 'metrics': {'after': metric}}
        (folder/'result.json').write_text(json.dumps(result))
        for phase in ['before', 'reference', 'after']:
            (folder/phase).mkdir()
            for name in ['rgb.png', 'mask.png', 'crop.png']:
                Image.fromarray(np.full((20,20),255,dtype='uint8')).save(folder/phase/name)
    assert not check_results(inventory,tmp_path,['mp3d','hm3d'])[1]
    extra = check_results(inventory,tmp_path,['mp3d','hm3d','missing_house'])[1]
    assert any(e['error']=='missing_native_result' for e in extra)
    folder=tmp_path/'hm3d/colored_source'; result=json.loads((folder/'result.json').read_text())
    result['metrics']['after'].update(status='fail',failures=['source_texture_color_mismatch'])
    (folder/'result.json').write_text(json.dumps(result))
    assert any(e['error']=='source_texture_color_mismatch' for e in check_results(inventory,tmp_path,['mp3d','hm3d'])[1])
