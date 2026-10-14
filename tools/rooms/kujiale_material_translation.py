"""Explicit extra InteriorAgent MDL inputs for research Preview Surface USD.

Source equations in M_BaseMaterial_v2.mdl distinguish constant, texture and
view-dependent falloff. Only constant and direct texture inputs are mapped;
falloff, rotated UVs and specular textures remain declared review items.
"""
from __future__ import annotations
from pathlib import Path


def family(values, source_id, mdl):
    if source_id == 'UsdPreviewSurface':return 'native_preview'
    if 'ColorIsTexType' in values or 'ColorColor' in values:return 'base_v2'
    if 'BaseColor_Color' in values or 'IsBaseColorTex' in values:return 'base_v1'
    if 'OmniGlass' in mdl:return 'omni_glass'
    if 'OmniPBR' in mdl:return 'omni_pbr'
    return 'unsupported'


def numeric_plan(values, source_id=None, mdl=''):
    f=family(values,source_id,mdl);review=[];textures=[];constants={}
    rgb=lambda x:tuple(float(x[i]) for i in range(3))
    scalar=lambda key,default:float(values.get(key,default) or 0.)
    if f=='base_v2':
        channels=[('diffuseColor','Color','ColorColor',True),('roughness','Roughness','RoughnessFloat',False),
                  ('metallic','Metallic','MetallicFloat',False),('emissiveColor','Emission','EmissionColor',True)]
        for target,prefix,constant,is_rgb in channels:
            toggle=scalar(prefix+'IsTexType',0.)
            if abs(toggle-1.)<1e-5:
                uv=values.get(prefix+'UVW',(1.,1.,0.,1.));offset=values.get(prefix+'Offset',(0.,0.,0.,1.))
                # The source MDL additionally rotates around the UV centre.
                # Nonzero rotations and flips require native review.
                if abs(float(uv[2]))>1e-7:review.append(prefix+':ROTATED_UV_NOT_TRANSLATED')
                uva=(float(uv[0]),float(uv[1]),float(offset[0]),1.-float(uv[1])-float(offset[1]))
                textures.append({'target':target,'key':prefix+'Tex','uva':uva,'raw':not is_rgb,'scale':1.,'bias':0.,'color':(1.,1.,1.)})
            elif toggle<1.:
                value=values.get(constant,(0.,0.,0.,1.) if is_rgb else 0.)
                constants[target]=rgb(value) if is_rgb else max(0.,min(1.,float(value)))
            else:review.append(prefix+':VIEW_DEPENDENT_FALLOFF_NOT_TRANSLATED')
        if scalar('IsNormal',0.)>.5:
            uv=values.get('NormalUVW',(1.,1.,0.,1.));offset=values.get('NormalOffset',(0.,0.,0.,1.))
            textures.append({'target':'normal','key':'NormalTex','uva':(float(uv[0]),float(uv[1]),float(offset[0]),1.-float(uv[1])-float(offset[1])),'raw':True,'scale':2.,'bias':-1.,'color':(1.,1.,1.)})
            if abs(float(uv[2]))>1e-7:review.append('Normal:ROTATED_UV_NOT_TRANSLATED')
            review.append('NORMAL_INTENSITY_AND_MDL_TANGENT_BASIS_REQUIRE_NATIVE_REVIEW')
        if scalar('IsFlip',0.)>.5:review.append('UV_FLIP_NOT_TRANSLATED')
        if scalar('SpecularIsTexType',0.)>0.:review.append('SPECULAR_TEXTURE_OR_FALLOFF_NOT_TRANSLATED')
    elif f=='omni_pbr':
        constants['diffuseColor']=rgb(values.get('diffuse_tint',values.get('diffuse_color_constant',(0.65,0.65,0.65))))
        for target,key in [('roughness','reflection_roughness_constant'),('metallic','metallic_constant')]:
            if key in values:constants[target]=max(0.,min(1.,float(values[key])))
        if values.get('diffuse_texture'):
            textures.append({'target':'diffuseColor','key':'diffuse_texture','uva':None,'raw':False,'scale':1.,'bias':0.,'color':constants['diffuseColor']})
        review.append('OMNIPBR_EXTRA_CONTROLS_REQUIRE_NATIVE_REVIEW')
    elif f=='base_v1':
        # pxr.Gf vectors implement indexing but are not collections.abc.Sequence.
        # Read channels explicitly, rather than the legacy helper's Sequence test.
        colour=rgb(values.get('BaseColor_Color',(.65,.65,.65)))
        constants['diffuseColor']=colour
        gloss=rgb(values.get('Gloss_Color',(.45,.45,.45)))
        metal=rgb(values.get('Metallic_Color',(0.,0.,0.)))
        constants['roughness']=max(.03,min(.97,1.-sum(gloss)/3.))
        constants['metallic']=max(0.,min(1.,sum(metal)/3.))
        if scalar('IsBaseColorTex',0.)>.5:
            textures.append({'target':'diffuseColor','key':'BaseColor_Tex','uva':values.get('BaseColor_UVA'),
                             'raw':False,'scale':1.,'bias':0.,'color':colour})
        if scalar('EmissiveIntensity',0.)>0.:
            emission=rgb(values.get('Emissive_Color',(1.,1.,1.)))
            constants['emissiveColor']=tuple(max(0.,min(1.,x*scalar('EmissiveIntensity',0.))) for x in emission)
        for toggle,key,target,prefix,invert in [('IsGlossTex','Gloss_Tex','roughness','Gloss',True),('IsMetallicTex','Metallic_Tex','metallic','Metallic',False)]:
            if scalar(toggle,0.)>.5:
                textures.append({'target':target,'key':key,'uva':values.get(prefix+'_UVA'),'raw':True,'scale':-1. if invert else 1.,'bias':1. if invert else 0.,'color':(1.,1.,1.)})
        if scalar('IsSpecularTex',0.)>.5:review.append('SPECULAR_TEXTURE_NOT_TRANSLATED')
    elif f=='unsupported':review.append('UNSUPPORTED_SOURCE_PARAMETER_SET; LEGACY_DEFAULT_REQUIRES_REVIEW')
    if f not in {'native_preview','unsupported'}:review.append('PREVIEW_APPROXIMATION_NOT_NATIVE_MDL_EQUIVALENCE')
    return {'family':f,'constants':constants,'textures':textures,'review':review}


def expand_stage(adapter_path:Path,source_path:Path):
    from pxr import Usd,UsdShade,Sdf,Gf
    from tools.rooms.prepare_interioragent_kujiale_adapter import _connect_uv_texture,_texture_path
    source=Usd.Stage.Open(str(source_path));stage=Usd.Stage.Open(str(adapter_path));stage.SetEditTarget(stage.GetRootLayer())
    records=[]
    types={'diffuseColor':Sdf.ValueTypeNames.Color3f,'emissiveColor':Sdf.ValueTypeNames.Color3f,
           'roughness':Sdf.ValueTypeNames.Float,'metallic':Sdf.ValueTypeNames.Float,'normal':Sdf.ValueTypeNames.Normal3f}
    for prim in list(stage.Traverse()):
        if not prim.IsA(UsdShade.Material):continue
        original=UsdShade.Material(source.GetPrimAtPath(prim.GetPath()))
        src=original.ComputeSurfaceSource() if original else None
        if not src or not src[0]:
            records.append({'material':str(prim.GetPath()),'family':'unresolved_surface','review':['SOURCE_SURFACE_UNRESOLVED']});continue
        shader=src[0];values={i.GetBaseName():i.Get() for i in shader.GetInputs()}
        mdl=str(shader.GetSourceAsset('mdl') or '')
        plan=numeric_plan(values,shader.GetIdAttr().Get(),mdl)
        row={'material':str(prim.GetPath()),'family':plan['family'],'review':list(plan['review']),'mapped_texture_inputs':[],'mapped_constants':list(plan['constants'])}
        material=UsdShade.Material(prim)
        if plan['family']=='native_preview':
            # Preserve the actual original Preview graph, including its UV nodes.
            inherited=UsdShade.Shader(stage.GetPrimAtPath(shader.GetPath()))
            material.CreateSurfaceOutput().ConnectToSource(inherited.ConnectableAPI(),src[1])
            records.append(row);continue
        preview=UsdShade.Shader(stage.GetPrimAtPath(prim.GetPath().AppendChild('AVEnginePreviewSurface')))
        if not preview:
            row['review'].append('LEGACY_ADAPTER_PREVIEW_MISSING');records.append(row);continue
        for target,value in plan['constants'].items():
            inp=preview.CreateInput(target,types[target]);inp.DisconnectSource()
            inp.Set(Gf.Vec3f(*value) if isinstance(value,tuple) else value)
        for item in plan['textures']:
            texture=_texture_path(values.get(item['key']))
            if texture is None:
                row['review'].append('ENABLED_TEXTURE_UNRESOLVED:'+item['key']);continue
            node=_connect_uv_texture(stage=stage,material_path=prim.GetPath(),name='AVEngineExpanded_'+item['target'],
                    texture=texture,color=item['color'],uva=item['uva'],raw=item['raw'])
            scale=item['scale'];bias=item['bias']
            if scale!=1. or bias!=0.:
                node.GetInput('scale').Set(Gf.Vec4f(scale,scale,scale,1.))
                node.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(bias,bias,bias,0.))
            target=item['target'];channel='rgb' if target in {'diffuseColor','emissiveColor','normal'} else 'r'
            if channel=='r':node.CreateOutput('r',Sdf.ValueTypeNames.Float)
            preview.CreateInput(target,types[target]).ConnectToSource(node.ConnectableAPI(),channel)
            row['mapped_texture_inputs'].append(item['key'])
        # Legacy normal textures are encoded 0..1; Preview's tangent normals are signed.
        legacy=UsdShade.Shader(stage.GetPrimAtPath(prim.GetPath().AppendChild('AVEngineNormalTexture')))
        if legacy:
            legacy.GetInput('scale').Set(Gf.Vec4f(2.,2.,2.,1.))
            legacy.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1.,-1.,-1.,0.))
            row['review'].append('NORMAL_INTENSITY_AND_MDL_TANGENT_BASIS_REQUIRE_NATIVE_REVIEW')
        records.append(row)
    stage.GetRootLayer().Save()
    return records
