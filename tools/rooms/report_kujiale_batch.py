#!/usr/bin/env python3
"""Report CPU Kujiale conversion, missing native evidence and internal CCI files."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

REPOSITORY=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPOSITORY),str(REPOSITORY/'src')]
from tools.rooms.kujiale_batch import write,type_of,completed_build

def read(p):return json.loads(Path(p).read_text())
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def material_audit(args):
    from pxr import Usd,UsdGeom,UsdShade,Sdf
    summary=[]
    for inventory in read(args.output/'inventory.json'):
        stage=Usd.Stage.Open(inventory['source']);used={}
        for prim in stage.Traverse():
            if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset)):continue
            if prim.IsA(UsdGeom.Mesh):
                counts=UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
                if counts is None or not len(counts):continue
            material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if material:used[str(material.GetPath())]=material
        flagged=[]
        for path,material in used.items():
            source=material.ComputeSurfaceSource()
            if not source or not source[0]:
                flagged.append({'material':path,'reason':'SOURCE_SURFACE_UNRESOLVED'});continue
            values={x.GetBaseName():x.Get() for x in source[0].GetInputs()}
            reasons=[]
            if not any(k in values for k in ['BaseColor_Color','IsBaseColorTex']):
                reasons.append('SOURCE_SURFACE_NOT_MDL_PARAMETER_SET; legacy adapter default colour requires review')
            for toggle,texture in [('IsBaseColorTex','BaseColor_Tex'),('IsNormalTex','Normal_Tex')]:
                if float(values.get(toggle) or 0.)<=0:continue
                value=values.get(texture)
                if not isinstance(value,Sdf.AssetPath) or not value.resolvedPath:
                    reasons.append('ENABLED_TEXTURE_UNRESOLVED:'+texture)
            for key in ['IsGlossTex','IsMetallicTex','IsSpecularTex']:
                if float(values.get(key) or 0.)>0:reasons.append('TEXTURE_CONTROL_NOT_TRANSLATED_BY_LEGACY_ADAPTER:'+key)
            if reasons:flagged.append({'material':path,'reasons':reasons})
        row={'house':inventory['house'],'source':inventory['source'],'used_materials':len(used),
             'source_materials_requiring_native_review':len(flagged),'flagged':flagged,
             'render_defect_fraction':None,'is_native_render_check':False}
        write(args.output/'houses'/inventory['house']/'material_review.json',row)
        summary.append(row)
        print('MATERIAL_REVIEW',row['house'],len(used),len(flagged),flush=True)
    write(args.output/'evidence/material_review_summary.json',summary)

def prepare_inputs(args):
    import math
    for original in read(args.output/'inventory.json'):
        name=original['house'];build=completed_build(args.output,name,args.build_name)
        conversion=read(build/'conversion_result.json');refresh=read(build/'material_refresh.json')
        rooms=read(build/'rooms_screened.json')
        package=Path(conversion['acoustic_package']).relative_to(build).as_posix() if conversion.get('acoustic_package') else None
        index={'schema':'avengine_kujiale_cpu_asset_index_v1','house':name,'split':'train','training_only':True,
               'qualification':'research_only_pending_native_UE','training_admitted':False,
               'room_geometry_manifest':'geometry_room_manifest.json','visual_usd':Path(refresh['derived_stage']).relative_to(build).as_posix(),
               'native_ue_map':None,'surface_mesh':'surface/surface.glb','vertices':'surface/vertices.npy','triangles':'surface/triangles.npy',
               'navigation':'navigation/navigation.navmesh','semantic_rooms':'semantic_rooms.json','floor_reference':'floor_reference.json',
               'screened_rooms':'rooms_screened.json','material_review':'material_refresh.json','acoustic_package':package,
               'visual_coordinates':'original InteriorAgent Z-up metres; UE importer converts to cm',
               'canonical_coordinates':'surface/nav/acoustics +Y up, -Z front; original [x,y,z] -> [x,z,y]',
               'path_rule':'All operational paths are relative to this file; 48g paths in old receipts are provenance only',
               'license':'InteriorAgent noncommercial internal research; attribution; no redistribution'}
        write(build/'engine_asset_index.json',index)
        pending=build/'native_ue_pending';pending.mkdir(exist_ok=False)
        request={'source_usd':str(Path(refresh['derived_stage'])),
                 'map_path':f'/Game/AVEngine/V7Kujiale20261003/{name}',
                 'destination_path':f'/Game/AVEngine/V7Kujiale20261003/Imported/{name}',
                 'result_path':str(pending/'import_result.json'),'visual_lights':[]}
        write(pending/'import_request.json',request)
        profile=read(build/'adapter/profile.json');views=[]
        for room in rooms:
            placement=room.get('metrics',{}).get('placement',{})
            if not room.get('geometry_eligible') or not placement.get('found'):continue
            camera=placement['camera_m'];a=placement['source_1_m'];b=placement['source_2_m']
            target=[(x+y)/2 for x,y in zip(a,b)];dx=target[0]-camera[0];dy=target[2]-camera[2]
            views.append({'view_id':name+'_'+room['room_label'],'position_m':[camera[0],camera[2],camera[1]],
                          'yaw_deg':math.degrees(math.atan2(dy,dx)),
                          'source_room_label':room['room_label'],'camera_basis':'actual CPU three-point witness; native floor/collision pending'})
        if not views:raise ValueError('No measured placement for native review profile: '+name)
        selected=[]
        for i in range(4):
            view=dict(views[i%len(views)])
            view['view_id']=name+'_native_proposal_'+str(i)
            view['yaw_deg']+=90.*(i//len(views))
            selected.append(view)
        profile.update(camera_views=selected,map_path=request['map_path'])
        from avengine.optional_backends.interioragent_kujiale import build_kujiale_review_plan
        build_kujiale_review_plan(profile,source_stage=Path(refresh['derived_stage']))
        write(pending/'profile.json',profile)
        write(pending/'pending_checks.json',{'house':name,'status':'not_run','checks':[
            'isolated UE project and map import; no writes through shared Content symlinks',
            'native collision/depth floor offset before actor calibration; do not assume 0020 offset',
            'native RGB missing surfaces/intersections/black <= frozen 15% criterion',
            'material/UV/normal/opacity/exposure manual review from material_refresh.json',
            'capsule and foot-contact checks on each admitted room',
            'native UE export parity with CPU USD acoustic/navigation surfaces'],
            'gpu_start_condition':'free memory >= same-workload measured peak + 4GB; no eligible peak record found',
            'import_tool':'tools/ue/import_usd_stage_to_level_editor.py','cci_native_runtime_status':'not_run'})
        print('PREPARED_INPUTS',name,flush=True)

def report(args):
    out=args.output;inventory=read(out/'inventory.json')
    conversions=[];results=[];file_rows=[];scenes=[];room_rows=[]
    build_by_house={}
    for original in inventory:
        name=original['house'];build=completed_build(out,name,args.build_name)
        build_by_house[name]=build
        if (build/'conversion_result.json').exists():
            conversion=read(build/'conversion_result.json');conversions.append(conversion)
            rooms=read(build/'rooms_screened.json')
            entry=read(build/'acceptance_house.json')
            mr=read(build/'material_refresh.json') if (build/'material_refresh.json').exists() else {}
        else:
            conversion={};rooms=[];entry={'house':name,'family':'kujiale','original_split':'train','candidate_room_count':0,'candidate_area_sum_m2':0.,'acoustic_rooms':[]};mr={}
        rp=out/'acoustics_batch/houses'/name/'result.json'
        result=read(rp) if rp.exists() else {'house':name,'family':'kujiale','status':'review','known_exposure':name=='kujiale_0020',
                   'rirs':[],'reasons':['ACOUSTICS_NOT_RUN'],'input':entry}
        acoustic=result.get('status','review') if rp.exists() else 'not_run'
        eligible_rooms=[r for r in rooms if r.get('geometry_eligible')]
        result.update(native_status=acoustic,acoustic_status=acoustic,
                      visual={'status':'not_run','reasons':['NATIVE_UE_RGB_NOT_RUN'],
                              'render_missing_intersection_black_fraction':None},
                      status='fail' if acoustic=='fail' else 'review',batch_eligible=False,
                      training_only=True,training_admitted=False,qualified_room_count=0,
                      result_directory=str(rp.parent),conversion=conversion,
                      geometry_candidate_room_count=len(eligible_rooms),
                      material_review=mr,floor_readback_status='not_run',actor_foot_contact_status='not_run')
        result['reasons']=list(result.get('reasons',[]))+['NATIVE_UE_RGB_NOT_RUN','NATIVE_UE_FLOOR_AND_FOOT_CONTACT_NOT_RUN','NATIVE_UE_EXPORT_VS_USD_ACOUSTIC_ALIGNMENT_UNVERIFIED']
        if not conversion:result['reasons'].append('CPU_CONVERSION_NOT_COMPLETE')
        result['input']=entry;results.append(result)
        for r in rooms:
            room_rows.append({'house':name,'room_label':r['room_label'],'room_type':r['room_type'],
                    'geometry_eligible':r['geometry_eligible'],'native_visual_status':'not_run','training_admitted':False,
                    'floor_area_m2':r.get('metrics',{}).get('floor_area_m2'),
                    'short_side_m':r.get('metrics',{}).get('short_side_m'),
                    'nav_main_area_m2':r.get('metrics',{}).get('nav_main_area_m2'),
                    'floor_height_m':r.get('floor_height_m'),
                    'dominant_floor_area_fraction':r.get('dominant_floor_area_fraction'),
                    'placement':r.get('metrics',{}).get('placement'),'reason_codes':r.get('reason_codes',[])})
        n=len(eligible_rooms)
        scenes.append({'house':name,'qualified_scenes':0,'conditional_independent_60s_scene_range':[8*n,16*n],
                       'assumption':'8–16 scenes per geometric candidate room after native admission; not a measured capacity',
                       'geometry_candidate_rooms':n})
        if not conversion:continue
        # Operational closures only. Source layers, originals and acoustic snapshots are omitted.
        includes=[Path(mr['engine_directory']) if mr else build/'engine',build/'surface',build/'navigation',Path(conversion['acoustic_package']).parent if conversion.get('acoustic_package') else None,
                  build/'semantic_rooms.json',build/'floor_reference.json',build/'rooms_screened.json',build/'geometry_room_manifest.json',build/'engine_asset_index.json',build/'material_refresh.json']
        files=[]
        for p in includes:
            if p is None:continue
            files.extend([q for q in p.rglob('*') if q.is_file() and q.name!='render_before_native_scalar_fix.usdc'] if p.is_dir() else [p])
        for p in sorted(set(files)):
            file_rows.append({'house':name,'path':str(p),'relative_path':str(p.relative_to(out)),
                              'bytes':p.stat().st_size,'sha256':sha256(p),'training_admitted':False,
                              'use':'internal_research_candidate','redistribution':'prohibited; InteriorAgent terms'})
    if len(conversions)!=25 and not args.allow_partial:
        raise ValueError(f'Only {len(conversions)}/25 conversions completed')
    rirs=[q for r in results for q in r['rirs']]
    cpu=np.array([q['native_simulation_cpu_seconds'] for q in rirs if 'native_simulation_cpu_seconds' in q])
    wall=np.array([r['wall_seconds'] for r in conversions])
    export_wall=np.array([r['conversion'].get('wall_seconds',0.)+r['material_review'].get('wall_seconds',0.) for r in results if r['conversion']])
    refresh_wall=np.array([r['material_review']['wall_seconds'] for r in results if r['material_review']])
    median_cpu=float(np.median(cpu)) if len(cpu) else None
    for row in scenes:
        measured=[q['native_simulation_cpu_seconds'] for r in results if r['house']==row['house'] for q in r['rirs'] if 'native_simulation_cpu_seconds' in q]
        row['measured_house_rir_cpu_seconds_median']=float(np.median(measured)) if measured else None
        # Two moving sources, 60s, RIR stride 0.5s -> 242 IR calls; rendering is excluded.
        row['rir_calls_per_scene_assumption']=242
        unit=row['measured_house_rir_cpu_seconds_median']
        row['rir_cpu_hours_range']=[x*242*unit/3600 for x in row['conditional_independent_60s_scene_range']] if unit is not None else None
        row['generation_cpu_total_hours']=None
        row['budget_excludes']='dry audio convolution, mixing, video, actors, encoding, labels and CCI hardware differences; total generation time unverified'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPOSITORY,text=True).strip()
    total_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    stats={'houses':25,'metadata_rooms':sum(len(h['rooms']) for h in inventory),'converted_houses':len(conversions),
          'geometry_candidate_houses':sum(r['geometry_candidate_room_count']>0 for r in results),
          'geometry_candidate_rooms':sum(r['geometry_candidate_room_count'] for r in results),
          'acoustic_status_counts':dict(Counter(r['native_status'] for r in results)),
          'acoustic_pass_houses':[r['house'] for r in results if r['native_status']=='pass'],
          'qualified_train_houses':0,'qualified_train_rooms':0,'rirs':len(rirs),'valid_rirs':sum(q.get('status')=='valid' for q in rirs),
          'spherical_rays':sum(sum(x['ray_count'] for x in r.get('leakage',[])) for r in results),
          'native_scalar_fixed_houses':sum(bool(r['material_review'].get('native_scalar_correction')) for r in results),
          'native_scalar_corrected_materials':sum(r['material_review'].get('native_scalar_correction',{}).get('materials_corrected',0) for r in results),
          'material_refresh_houses':sum(bool(r['material_review']) for r in results),
          'material_refresh_wall_seconds_sum':sum(r['material_review'].get('wall_seconds',0.) for r in results),
          'raw_usd_complete_houses':sum(h.get('usd_complete',False) for h in inventory),
          'conversion_driver_cpu_seconds_sum':sum(r.get('driver_cpu_seconds',0.) for r in conversions),
          'material_refresh_driver_cpu_seconds_sum':sum(r['material_review'].get('driver_cpu_seconds',0.) for r in results),
          'cpu_conversion_wall_seconds':{'min':float(wall.min()),'median':float(np.median(wall)),'max':float(wall.max()),'sum':float(wall.sum())} if len(wall) else None,
          'cpu_export_stage_wall_seconds':{'min':float(export_wall.min()),'median':float(np.median(export_wall)),'max':float(export_wall.max()),'sum':float(export_wall.sum())} if len(export_wall) else None,
          'material_refresh_wall_seconds':{'min':float(refresh_wall.min()),'median':float(np.median(refresh_wall)),'max':float(refresh_wall.max())} if len(refresh_wall) else None,
          'native_rir_cpu_seconds_median':median_cpu,'all_task_output_bytes_including_attempts':total_bytes,
          'cci_research_candidate_files':len(file_rows),'cci_research_candidate_bytes':sum(r['bytes'] for r in file_rows),
          'cci_admitted_training_files':0,'git_commit':commit,'gpu_jobs_started':0,'datasets_generated':0}
    if stats['material_refresh_houses']!=25 and not args.allow_partial:
        raise ValueError(f"Only {stats['material_refresh_houses']}/25 expanded material exports completed")
    write(out/'all_house_results.json',results);write(out/'summary.json',stats)
    write(out/'rooms_results.json',room_rows);write(out/'scene_cpu_budget.json',scenes)
    write(out/'train_houses.json',{'split':'train','houses':[],'rooms':[],'reason':'native visual, floor and actor-foot checks missing'})
    write(out/'acoustic_pass_train_candidates.json',{'training_only':True,'qualified':False,'houses':[r['input'] for r in results if r['native_status']=='pass']})
    write(out/'CCI_DERIVED_ASSETS.json',{'transfer_performed':False,'admitted_train_files':[],
          'candidate_files':file_rows,'candidate_bytes':sum(r['bytes'] for r in file_rows),'asset_status':'research_only_pending_native_UE',
          'runtime_path_authority':'engine_asset_index.json paths are relative to each house build; 48g absolute paths in navigation/build_result.json are provenance and must not be used as CCI operational paths',
          'native_CCI_runtime_status':'not_run','original_interioragent_package_included':False,'license':'noncommercial internal research; attribution; no redistribution'})
    with (out/'CCI_DERIVED_ASSETS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(file_rows[0]) if file_rows else ['house','relative_path','bytes','sha256']);writer.writeheader();writer.writerows(file_rows)
    columns=['house','family','original_split','candidate_area_sum_m2','candidate_room_count','native_status','visual_status','status','batch_eligible','rt60_mid_median_s','max_escape_fraction','max_foa_angle_deg','reasons','result_directory']
    with (out/'house_results.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in results:
            record={k:r.get(k) for k in columns};record.update({k:r['input'].get(k) for k in ['original_split','candidate_area_sum_m2','candidate_room_count']})
            record.update(visual_status='not_run',reasons='; '.join(r['reasons']));writer.writerow(record)
    w=stats['cpu_conversion_wall_seconds'];ew=stats['cpu_export_stage_wall_seconds']
    lines=[f"结论：25 套中正式可用于训练 0 套、0 间；已完成 {len(conversions)} 套 CPU 派生转换及原生数值修正，留下 {stats['geometry_candidate_rooms']} 间几何候选，声学诊断通过 {len(stats['acoustic_pass_houses'])} 套；每套成功阶段 CPU 转换加材质更新中位 {ew['median']/60:.1f} 分钟，范围 {ew['min']/60:.1f}–{ew['max']/60:.1f} 分钟（0009 中止尝试另计），原生画面、地板偏移和人物脚底仍需人工核验（预算 0.75–2.25 人时/套，未实测）。",'',
           f"48g 独立 worktree：{REPOSITORY}；分支 codex/v7-kujiale-20261003；提交 `{commit}`；未 push。产物根：{out}。",'',
           '盘点依据为 inventory.json、逐套 inventory.json 及 evidence/download_manifest_check.json。25 套共有 '+str(stats['metadata_rooms'])+' 个设计语义房间。13,469 个文件的大小与 HF 清单一致；未逐文件复算原始内容哈希。revision ac599fe2d2285ed0ddfafb70ba56e489462e0650。原始与既有检出、checkpoint 保持只读。', '',
           '原始 USD 不是完整的可移植材质闭包：存在旧 UE 材质路径提示、MDL 模块路径以及少量 PreviewSurfaceTextures 缺图。原始组合错误、缺 asset、空 Mesh、退化面均在逐套 inventory.json 保留。最终派生 USD 的依赖只在各 build/engine_expanded/ 内；扩展转换保留原有 Preview 图，并按索引读取原生 Gf 向量的 v1 数值、修复旧默认色/光泽/金属/发光问题（native_scalar_fix.json），映射 BaseMaterial v2 常量/纹理、OmniPBR、v1 光泽/金属纹理并修正编码法线范围；复杂 falloff、旋转 UV、法线强度、specular/玻璃/曝光仍需原生核验。源缺图只清除纹理路径并保留作者写的 fallback，操作逐项记在 material_refresh.json；不能把默认颜色当作正确材质。', '',
           '逐套结果（面积是 rooms.json 设计多边形之和，含被排除空间；楼层是朝上地板面支持的房间高度簇代理，源数据未提供可靠的房产楼层标签；原始 USD 缺项数依次是未解析资产/组合错误/非空网格错误；声学 pass 仅为 CPU 诊断）：','',
           '| 房屋 | 房间数/房型 | 设计面积 m² | 地板高度簇 m | 原始USD缺项数 | 几何候选 | 基础/材质 min | 声学 | RIR/有效 | 漏声% | FOA° | RT60 s |',
           '|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for h,r in zip(inventory,results):
        heights=sorted({round(q.get('metrics',{}).get('floor_y_m',q.get('floor_height_m',0.)),3) for q in read(build_by_house[h['house']]/'rooms_screened.json') if q.get('metrics',{}).get('floor_area_m2') is not None}) if r['conversion'] else []
        number=lambda x,d=3:'—' if x is None else f'{x:.{d}f}'
        lines.append(f"| {h['house']} | {len(h['rooms'])}/"+', '.join(f'{k}:{v}' for k,v in h['room_type_counts'].items())+f" | {h['metadata_area_sum_m2']:.2f} | {heights} | {len(h.get('unresolved_assets',[]))}/{len(h.get('composition_errors',[]))}/{len(h.get('mesh_errors',[]))} | {r['geometry_candidate_room_count']} | "+number(r['conversion'].get('wall_seconds',0.)/60 if r['conversion'] else None,1)+' / '+number(r['material_review'].get('wall_seconds',0.)/60 if r['material_review'] else None,1)+f" | {r['native_status']} | {len(r['rirs'])}/{sum(q.get('status')=='valid' for q in r['rirs'])} | "+number(r.get('max_escape_fraction')*100 if r.get('max_escape_fraction') is not None else None,2)+' | '+number(r.get('max_foa_angle_deg'),4)+' | '+number(r.get('rt60_mid_median_s'))+' |')
    lines += ['',f"结果来源：houses/<house>/<build>/conversion_result.json、material_refresh.json、rooms_screened.json（0024 使用 build_v5，其余 build_v4，映射见 evidence/build_selection.json）；acoustics_batch/houses/<house>/result.json。详细原因和原生直达延迟、距离指数、听者数都保留在 all_house_results.json。每套最多三间，实际总计 {len(rirs)} RIR、有效 {stats['valid_rirs']} 条、{stats['spherical_rays']} 球面射线。",'',
      '能演：直接复用冻结门槛与原生 navmesh/三段原网格视线搜索；可走面积用连续三角形与主连通网格支持相交，不用设计面积替代。封闭地板底面不算新楼层。能看：没有原生 UE RGB，缺面/穿模/黑画面比例为 null，未填 0。能听：原封调用 3414629 的验收代码、同一 runtime、v6 RLR 配置和判据，固定来源字段见 baseline_identity.json，语义声学材料是研究近似，未从 STARSS 拟合。', '',
      '自动产物：自包含 Preview Surface USD/纹理、同几何的 GLB/数组、房间语义、5cm/1cm navmesh、逐房地板记录、声学包、RLR 兼容派生、最多三房实测。空 Mesh 容器仅转为 Xform 并保留子网格与变换；不补洞，不删除原始表面。', '',
      '必须补的人工/原生步骤：材质、玻璃、缺图和曝光的真实 UE 检查（预算 20–60 min/套）；以低相机深度或碰撞量 UE 地板偏移、胶囊和脚底（15–45 min/套）；门洞、声源/听者与 UE 导出/声学几何对齐抽查（10–30 min/套）。这些均未执行，工时是未验证预算，失败修复另算。声学失败/待审不得靠调门槛或补洞改成 pass。', '',
      'GPU 限制：现场显存快照只证明查询时可用，未找到同负载可核对的显存峰值；因此不满足“空闲显存≥实测峰值+4GB”的启动条件，没有起 GPU 作业。未烘焙原生 UE 地图，没有原生截图、正式 floor offset、人物动态或完整 Episode。缺少可核对的同负载峰值记录或明确的受控峰值测量安排，不能继续占用 GPU。', '',
      '条件场景/CPU 预算见 scene_cpu_budget.json：通过原生验收后，每间先规划 8–16 个独立 60 秒场景，故每套为 8N–16N；这不是已证明的摆放容量。按两个移动声源、0.5秒 RIR 步长计算每场景 242 次 RIR，用各套实测 native_simulation_cpu_seconds 的中位数外推。完整生成 CPU 小时未验证：干声混音、时变卷积、人物、视频/编码/出题和 CCI 硬件差异尚未计入；数据集未生成。', '',
      f"CCI 清单：CCI_DERIVED_ASSETS.csv/json，候选 {len(file_rows)} 文件、{stats['cci_research_candidate_bytes']/1e9:.3f} GB，每个文件有 sha256；正式已准入训练资产 0 文件。只列扩展 engine、表面、导航、语义/地板、相对路径资产索引与 RLR 包，不列原始 InteriorAgent 包、原始 USD 引用层或 snapshot。engine_asset_index.json 的相对路径是 CCI 使用依据；导航旧 receipt 的 48g 绝对路径只作来源记录。0022 声学包换目录后通过既有 package 校验，详见 evidence/relocation_validation.json；CCI 本机运行未验证。未传到 CCI。派生资产同样禁止再分发，只限自有 48g/CCI 非商业研究并须引用。",'',
      f"磁盘依据：evidence/storage_estimate.json 原预估 44.623GB；evidence/storage_pilot_estimate.json 实测外推 22.773GB；材质更新预估见 evidence/storage_material_refresh_estimate.json（49.865GB 上限内）；所有尝试合计当前 {total_bytes/1e9:.3f}GB。旧尝试保留，未删缓存或别人的文件。",'',
      '最终定向回归 14 passed（logs/tests_kujiale_scalar_fix.log）；仓库 fast_unit 以 logs/tests_fast_unit_scalar_fix.log 和 evidence/tests_fast_unit_scalar_fix_exit.json 为最终来源（此前 277 passed、36 skipped、5890 deselected，28.44s 的日志仍保留）。跳过依赖的保留 facts 等材料在此独立检出中缺失，未把 skip 当作通过。基线验收和共用筛选的 21 个源码文件逐字未改（baseline_identity.json）。其余验证见 logs/tests_kujiale_v3.log、logs/tests_kujiale_expanded.log、logs/tests_fast_unit.log、evidence/final_verification.json；不同日志的 pass/skip/失败如实单列，不能用 Python 测试代替原生画面。停止方式和最终进程核验见 JOBS_zh.md。脚本执行目录为 scripts/；README 为代码仓库 docs/rooms/KUJIALE_BATCH_zh.md。']
    lines += ['', '逐套条件场景与 RIR CPU 预算（正式准入前场景数均为 0；以下是假设每间 8–16 个独立 60 秒场景的研究规划）：', '',
              '| 房屋 | 几何候选房 | 条件场景数 | 仅 RIR 的 CPU 小时 | 完整生成 CPU 小时 |',
              '|---|---:|---:|---:|---|']
    for row in scenes:
        n=row['conditional_independent_60s_scene_range'];h=row['rir_cpu_hours_range']
        hours='未验证' if h is None else f'{h[0]:.3f}–{h[1]:.3f}'
        lines.append(f"| {row['house']} | {row['geometry_candidate_rooms']} | {n[0]}–{n[1]} | {hours} | 未验证 |")
    lines += ['', f"全批 RIR 每调用 CPU 秒数的样本中位值={median_cpu}（预算实际用各套中位值），来自 acoustics_batch/houses/*/result.json 的 rirs[].native_simulation_cpu_seconds；保留每套不同房间/几何耗时，不把本机实测当作 CCI 性能。", '',
              '0009 包含一次为启用等价家具缓存而停止的本任务转换，初次已用时见 evidence/resume_0009.json；表内基础时间为恢复过程的实测，不含初次中止用时，故不是冷启动保证。其他旧失败、首轮 EXR 不支持和第一次 refresh CLI 缺 manifest 的失败日志都保留；EXR 开关仅设在本任务进程，不改公共环境。', '',
              '原生 UE 导入的具体 request、实际三点摆放相机及待验项目位于各 build/native_ue_pending/；这些仅是输入文件，不表示地图已导入。执行所需的新隔离 UE 项目、磁盘预算以及合规的实测显存峰值仍缺；不要对共享 Content 或既有项目写入。', '',
              '命令与改动范围见 COMMANDS_zh.md，逐阶段事实见 PROGRESS_zh.md；提交只含代码、测试和说明，未包含受许可限制的原始或派生资产。']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(stats,ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['material-audit','prepare-inputs','report'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--build-name',default='build_v4')
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args();{'material-audit':material_audit,'prepare-inputs':prepare_inputs,'report':report}[args.command](args)
if __name__=='__main__':main()
