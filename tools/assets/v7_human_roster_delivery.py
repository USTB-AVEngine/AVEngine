"""Build the deliverable hierarchy from saved native evidence, with qualified states."""
import argparse,base64,csv,html,io,json,re,shutil
from pathlib import Path
from PIL import Image
from v7_human_roster_materials import json_write

ZH={'red':'红','green':'绿','blue':'蓝','yellow':'黄','white':'白','black':'黑','original':'原色'}
ROLE={'Fire':'消防员','Medical':'医护','Police':'警察','Construction':'建筑工人','Business':'商务人士','Pilot':'飞行员','Chef':'厨师','Gardener':'园丁'}
LX={'clapping_person':'鼓掌人物','coughing_person':'咳嗽人物','flute_player':'长笛演奏者','guitar_player':'吉他演奏者','hand_drum_player':'手鼓演奏者','megaphone_speaker':'喇叭喊话人物','phone_call':'打电话人物','pianist':'钢琴演奏者','saxophone_player':'萨克斯演奏者','violin_player':'小提琴演奏者'}
def read(p,default=None):return json.loads(p.read_text())if p.exists()else default
def title(aid,selected):
 if aid.startswith('v7_lx_'):return LX[aid.removeprefix('v7_lx_')]
 r=selected[aid];parts=r['name'].split('_');sex='女性'if r['gender']=='female'else'男性'
 if r['controlled_color']:return '成年'+sex+' '+parts[-1]
 return ROLE[parts[0]]+('女'if r['gender']=='female'else'男')+' '+parts[-1]
def embed(p):
 if not p.exists():return '<div class="missing">未完成该视图</div>'
 im=Image.open(p).convert('RGB');im.thumbnail((480,480));b=io.BytesIO();im.save(b,format='JPEG',quality=82);return '<img loading="lazy" src="data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()+'">'
def run(T):
 selection=read(T/'registrations/selection.json');selected={r['asset_id']:r for r in selection['avatars']};hab={(r['house'],r['asset_id'],r['color']):r for r in read(T/'evidence/habitat_native_results.json',[])};ue={(r['asset_id'],r['color']):r for r in read(T/'evidence/ue_import_results.json',[])}
 unative={(r['asset_id'],r['color']):r for r in read(T/'evidence/ue_native_results.json',[])};skeleton={r['asset_id']:r for r in read(T/'evidence/ue_skeletal_root_results.json',read(T/'evidence/ue_skeletal_results.json',[]))};animation={r['asset_id']:r for r in read(T/'evidence/ue_animation_root_readback.json',read(T/'evidence/ue_animation_readback.json',[]))};colors=read(T/'evidence/rendered_color_check.json',{});manual=read(T/'evidence/mouth_visual_review.json',{})
 blueprints={(r['asset_id'],r['color']):r for r in read(T/'evidence/ue_blueprint_results.json',[])};overrides={(r['asset_id'],r['color']):r for r in read(T/'evidence/ue_skeletal_instance_materials.json',[])};action_qa=read(T/'evidence/ue_animation_root_equivalence.json',{});hab_actions=read(T/'evidence/habitat_action_native_results.json',[]);ue_actions=read(T/'evidence/ue_action_native_results.json',[])
 majors={};flat=[]
 for path in sorted((T/'assets/runtime').glob('*/*/poses.json')):
  p=read(path);aid=p['asset_id'];color=p['color'];r=selected.get(aid);class_name=title(aid,selected);static=aid.startswith('v7_lx_');ht=[hab.get((h,aid,color))for h in ['mp3d','hm3d']];u=ue.get((aid,color),{});n=unative.get((aid,color),{})
  hp=all(x and x['status']=='pass'for x in ht);up=u.get('status')=='pass' and n.get('status')=='pass' and n.get('visible_pixels',1)>1500;review=manual.get(aid,{});mouth=p['poses']['stand']['mouth_m']
  item={'variant_id':aid+'_'+color+'_v1','major_id':aid,'major_name_zh':class_name,'minor_name_zh':ZH[color] if r and r['controlled_color']else'静态'if static else'独立身份原色','color':color,'controlled_shirt':bool(r and r['controlled_color']),'static_only':static,'walk_allowed':not static,'walk_requires_runtime_adapter':not static,'ue_native_clip_playback_requires_ground_correction':not static,'units':{'glb':'metres +Y up -Z forward','ue':'centimetres +Z up; static parts common actor yaw +90'},'height_m':p['poses']['stand']['bounds_max_m'][1],'mouth_local_glb_m':mouth,'mouth_local_ue_cm':[-mouth[2]*100,mouth[0]*100,mouth[1]*100]if mouth else None,'mouth':{'source':'MUpperLip bone'if not static else'annotated visible surface','review_status':review.get('status','not_run'),'review_evidence':review.get('evidence'),'ue_native_animation_readback':animation.get(aid,{}).get('status','not_run')},'habitat':{'status':'pass'if hp else'not_run','glb':p['habitat_glb'],'stand':{'status':'pass'if hp else'not_run'},'walk':{'status':'not_supported'if static else'qualified','implementation':'geometry-only cached source-motion frames materialized to rigid GLB; actor route translation remains external'if not static else None},'sit':{'status':'not_supported'if static else'qualified_requires_seat_support','glb':str(path.with_name('sitting.glb').relative_to(T))if not static else None},'native_evidence':[x['rgb']for x in ht if x]},'ue':{'status':'pass'if up else'qualified_import_only'if u.get('status')=='pass'else'not_run','mesh_paths':[x['path']for x in u.get('assets',[])if x['class']=='StaticMesh'],'materials':[x['path']for x in u.get('assets',[])if x['class']=='MaterialInstanceConstant'],'skeletal_mesh':[x['path']for x in skeleton.get(aid,{}).get('assets',[])if x['class']=='SkeletalMesh'],'animations':[x['path']for x in skeleton.get(aid,{}).get('assets',[])if x['class']=='AnimSequence'],'native_evidence':n.get('rgb'),'static_actor_yaw_degrees':90},'artifact_path':str(path.parent.relative_to(T)),'motion_manifest':p.get('motion_manifest'),'sound_class_requested':p.get('sound_class','human_speech'),'audio_admission_owner':'v7-families','state_classification':'research_candidate','episode_admission':'not_run','dual_backend_static_usable':bool(hp and up),'automatic_qa':{'habitat_floor_m':[x['native_floor_delta_m']for x in ht if x],'habitat_highlight_ratio':[x['highlight_ratio_rgb_ge_245']for x in ht if x],'ue_floor_m':n.get('floor_delta_m'),'ue_highlight_ratio':n.get('highlight_ratio_rgb_ge_245')}}
  item['mouth']['ue_visual_review']=review.get('ue_status','not_run')
  bp=blueprints.get((aid,color),{});ov=overrides.get((aid,color),{})
  item['ue']['blueprint_path']=bp.get('blueprint_path')if bp.get('status')=='pass'else None
  item['ue']['generated_class_path']=bp.get('generated_class_path')if bp.get('status')=='pass'else None
  item['ue']['logical_actor_count']=1 if bp.get('status')=='pass'else None
  item['ue']['blueprint_readback_status']=bp.get('status','not_run')
  item['ue']['skeletal_material_overrides']=ov.get('slots',[])
  item['ue']['skeletal_material_match_status']=ov.get('status','not_supported'if static else'not_run')
  item['ue']['motion_source_equivalence_status']=action_qa.get('status','not_run')if not static else'not_supported'
  item['ue']['skeletal_component_yaw_degrees']=-90
  item['ue']['skeletal_actor_root_yaw_offset_degrees']=-90
  item['ue']['skeletal_actor_root_component_note']='SkeletalMeshActor root is its skeletal component: compose actor yaw with -90 and add ground correction to scene actor position; do not reset root relative location to zero XY.'
  item['ue']['native_action_evidence']=[x['rgb']for x in ue_actions if x['asset_id']==aid]
  item['ue']['native_action_color_scope']='representative red or original; all colour geometry and native material slot equivalence checked separately'if not static else'not_supported'
  item['ue']['walk']={'implementation':'one SkeletalMeshActor, matched material overrides, Walking clip, per-frame ground correction from motion cache, FOA attached to native MUpperLip bone'if not static else None,'status':'not_supported'if static else'qualified_native_clip_source_checked'if action_qa.get('status')=='pass'else'qualified_source_equivalence_pending'}
  item['ue']['sit']={'status':'not_supported'if static else'qualified_requires_seat_support'}
  walk_cases=[x for x in hab_actions if x['asset_id']==aid and x['motion_family']=='walk']
  item['habitat']['walk']['native_sampled_frame_status']='pass'if not static and len(walk_cases)==6 and all(x['status']=='pass'for x in walk_cases)else'not_supported'if static else'not_run'
  sit_cases=[x for x in hab_actions if x['asset_id']==aid and x['motion_family']=='sit']
  item['habitat']['sit']['native_sampled_frame_status']='pass'if not static and len(sit_cases)==2 and all(x['status']=='pass'for x in sit_cases)else'not_supported'if static else'not_run'
  native_ue_walk=[x for x in ue_actions if x['asset_id']==aid and x['motion_family']=='walk']
  native_ue_sit=[x for x in ue_actions if x['asset_id']==aid and x['motion_family']=='sit']
  for family,cases,expected in [('walk',native_ue_walk,3),('sit',native_ue_sit,1)]:
   item['ue'][family]['native_sampled_frame_status']='not_supported'if static else'pass'if len(cases)==expected and all(x['status']=='pass'and x.get('capture_qa_status')=='pass'for x in cases)else'not_run'
  if item['ue']['walk']['native_sampled_frame_status']=='pass':item['ue']['walk']['status']='qualified_native_scene_samples_checked'
  flat.append(item);majors.setdefault(aid,{'major_id':aid,'name_zh':class_name,'source':'lx'if static else'Rocketbox Adults'if r['controlled_color']else'Rocketbox Professions','gender':r['gender']if r else'not_annotated','minor_variants':[]})['minor_variants'].append(item)
 summary={'major_count':len(majors),'minor_count':len(flat),'habitat_usable_minor':sum(x['habitat']['status']=='pass'for x in flat),'ue_native_usable_minor':sum(x['ue']['status']=='pass'for x in flat),'dual_backend_usable_minor':sum(x['dual_backend_static_usable']for x in flat),'dual_backend_usable_major':sum(any(x['dual_backend_static_usable']for x in m['minor_variants'])for m in majors.values()),'controlled_base_count':16,'controlled_color_variants':96,'profession_identities':14,'lx_static_identities':10}
 roster={'schema':'avengine_v7_human_roster_v1','artifact_root':str(T),'ue_project':'ue_project/V7Humans.uproject','summary':summary,'majors':list(majors.values()),'notes':['walk and sit are separately qualified; static uasset presence does not imply native animation admission','lx must not be scheduled to walk','audio category mapping only; v7-families owns audio admission','UE colour separation is not an acceptance gate']}
 json_write(T/'registrations/human_roster.json',roster)
 with (T/'registrations/human_roster.csv').open('w',newline='',encoding='utf-8-sig')as f:
  cols=['大类','小类','variant_id','Habitat','UE','动作','Habitat嘴部','UE嘴部','GLB','UE蓝图'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for x in flat:w.writerow({'大类':x['major_name_zh'],'小类':x['minor_name_zh'],'variant_id':x['variant_id'],'Habitat':x['habitat']['status'],'UE':x['ue']['status'],'动作':'仅静态'if x['static_only']else'站；走缓存帧；坐需要匹配支撑面','Habitat嘴部':x['mouth']['review_status'],'UE嘴部':x['mouth']['ue_visual_review'],'GLB':x['habitat']['glb'],'UE蓝图':x['ue']['generated_class_path']})
 profiles=[]
 for r in selected.values():
  if not r['controlled_color']:continue
  for i,domain in enumerate(selection['color_domains']):profiles.append({'base_id':r['asset_id'],'profile_group':r['asset_id']+'_shirt_set_'+str(i+1),'attribute':'shirt_main_color','domain':domain,'value_labels':{c:ZH[c]for c in domain},'operation':'replace_base_color_preserve_pbr_detail_v1','mask':str((T/'masks'/r['asset_id']/'shirt_main_color.png').relative_to(T))})
 json_write(T/'registrations/controlled_shirt_domains.json',{'schema':'avengine_v7_human_color_domains_v1','profiles':profiles,'reason':'Each existing strict profile attribute permits at most three values; the six variants still share one major identity.'})
 json_write(T/'registrations/lx_dry_audio_mapping.json',{'schema':'avengine_v7_lx_dry_audio_mapping_v1','admission_owner':'v7-families','entries':[{'major_id':x['major_id'],'human_action':x['major_name_zh'],'requested_dry_class':x['sound_class_requested'],'dry_asset_id':None,'foa_anchor_glb_m':x['mouth_local_glb_m'],'static_only':True,'walk_allowed':False}for x in flat if x['static_only']]})
 parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>v7 人物双端验收</title><style>body{font:15px system-ui;background:#f2f3f5;color:#222;margin:24px}section{background:white;padding:16px;margin:20px 0;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}img{width:100%;height:auto}article{border-top:1px solid #ddd;padding:12px 0}.missing{padding:80px 10px;background:#ddd}code{overflow-wrap:anywhere}.meta{color:#555;font-size:13px}input{padding:10px;width:50%}</style><h1>v7 人物资产验收页</h1><p>'+html.escape(json.dumps(summary,ensure_ascii=False))+'</p><p>左：MP3D / Habitat；中：HM3D / Habitat；右：酷家乐 / UE。图片均内嵌。UE 不以贴图色差准入。走、坐和嘴部验证状态按清单单列。</p><input id="search" placeholder="筛选人物或颜色"><p>Habitat 渲染色差最小 CIE76 ΔE：'+str(colors.get('minimum_delta_e','未完成'))+'；阈值 20。</p>']
 for m in majors.values():
  parts.append('<section data-text="'+html.escape(m['name_zh']+' '+m['major_id'],quote=True)+'"><h2>'+html.escape(m['name_zh'])+'</h2>')
  for x in m['minor_variants']:
   aid=x['major_id'];c=x['color'];parts.append('<article><b>'+html.escape(x['minor_name_zh'])+'</b><div class="meta">'+html.escape(x['variant_id']+' | Habitat '+x['habitat']['status']+' | UE '+x['ue']['status']+' | Habitat 嘴部 '+x['mouth']['review_status']+' | UE 嘴部 '+x['mouth']['ue_visual_review'])+'</div><div class="grid">')
   for p in [T/'renders/habitat/mp3d'/aid/c/'mouth.png',T/'renders/habitat/hm3d'/aid/c/'mouth.png',T/'renders/ue'/aid/c/'mouth.png']:parts.append(embed(p))
   parts.append('</div></article>')
  parts.append('</section>')
 parts.append('<section><h2>动作抽样与嘴部复核</h2><p>行走：源动作首、中、末帧；坐姿：源动作首帧，尚需匹配座椅支撑面。以下包含 MP3D/HM3D 原生 Habitat 和酷家乐原生 UE；UE 使用真正 SkeletalMesh/AnimSequence，嘴点由动态骨骼读回。坐姿仍需匹配支撑座椅。</p><div class="grid">')
 for p in [T/'evidence/mask_contact_sheet.jpg',T/'evidence/mouth_review_lx_corrected.jpg',T/'evidence/ue_mouth_review_sheet.jpg']:parts.append(embed(p))
 parts.append('</div>')
 for aid in sorted(selected):
  parts.append('<article><b>'+html.escape(title(aid,selected))+'</b><div class="grid">')
  for label in sorted(p.name for p in (T/'renders/habitat_actions/mp3d'/aid).iterdir()if p.is_dir()):
   for folder in [T/'renders/habitat_actions/mp3d'/aid/label,T/'renders/habitat_actions/hm3d'/aid/label,T/'renders/ue_actions'/aid/label]:parts.append(embed(folder/'mouth.png'))
  parts.append('</div></article>')
 parts.append('</section>')
 parts.append('<script>document.querySelector("#search").oninput=e=>{const s=e.target.value.toLowerCase();document.querySelectorAll("section").forEach(x=>x.hidden=!x.textContent.toLowerCase().includes(s))}</script></html>');(T/'reports/review.html').write_text(''.join(parts))
 rows=['# 人物资产清单','',f"本轮 {summary['major_count']} 个大类、{summary['minor_count']} 个小类；双端原生静态验收通过 {summary['dual_backend_usable_major']} 个大类、{summary['dual_backend_usable_minor']} 个小类。",'','| 大类 | 小类 | Habitat | UE | 动作 | Habitat 嘴部 | UE 嘴部 |','|---|---|---|---|---|---|---|']
 for x in flat:rows.append('| '+' | '.join([x['major_name_zh'],x['minor_name_zh'],x['habitat']['status'],x['ue']['status'],'仅静态'if x['static_only']else'站；走需适配；坐需支撑',x['mouth']['review_status'],x['mouth']['ue_visual_review']])+' |')
 (T/'reports/INVENTORY_zh.md').write_text('\n'.join(rows)+'\n')
 json_write(T/'evidence/delivery_summary.json',summary);print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
