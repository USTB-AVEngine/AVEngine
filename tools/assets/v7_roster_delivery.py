"""Build Chinese inventories, a sourced report and one JPEG-embedded review page."""
import argparse,base64,html,io,json,collections
from pathlib import Path
from PIL import Image
from v7_roster_disk import check
from v7_human_roster_materials import json_write

def read(p,default=None):
 return json.loads(p.read_text()) if p.exists() else ([] if default is None else default)
def label(b):return '通过' if b else '未准入'
def table(columns,rows):
 return '| '+' | '.join(columns)+' |\n|'+ '|'.join(['---']*len(columns))+'|\n'+''.join('| '+' | '.join(str(x).replace('|','/') for x in row)+' |\n'for row in rows)
def run(T):
 R=T/'round3';D=T/'reports';D.mkdir(exist_ok=True);a=read(T/'registrations/v7_asset_roster.json',{});s=a['summary'];names={m['major_id']:m['name_zh']for m in a['humans']['majors']};sty={'dining':'餐椅','office':'办公椅','armchair':'扶手椅','stool':'凳子','sofa_chair':'沙发椅'};cap=read(R/'evidence/ue_native_capture_qa.json');seats=read(R/'evidence/seated_native_results.json');hx=read(R/'evidence/habitat_extra_action_results.json');ud=read(R/'evidence/ue_dog_motion_results.json');bd=read(R/'evidence/bird_native_results.json');full=read(R/'evidence/scene_motion_collision_results.json');extra=read(R/'evidence/extra_action_results.json');source=read(R/'evidence/source_geometry_invariance.json');fits=read(R/'evidence/chair_body_compatibility.json')
 humanrows=[]
 for m in a['humans']['majors']:
  c=m['movement_capabilities'];v=m['minor_variants'];sound=m.get('sound_anchor',{});humanrows.append([m['name_zh'],m['major_id'],'、'.join(x['minor_name_zh']for x in v),len(v),'Habitat/UE','、'.join({'stand':'站','walk':'走','sit':'坐','static':'静态'}.get(x,x)for x in c['movement']),'固定坐姿/自带座椅'if c['static_only']and c['has_sitting_pose']else label(c['sit']),label(c['seated_talk']),label(c['standing_talk_gestures']),label(c['turning']),'、'.join(sty.get(x,x)for x in m['seat_types'])or'不适用',sound.get('primary_part','MUpperLip；头盔身份看各嘴部复核资格')])
 text=f"人物共 {s['major_count']} 个大类、{s['minor_count']} 个小类双端静态可用；走路准入 {s['walking_major_admitted']} 个底模，坐姿准入 {s['seated_major_admitted']} 个底模。\n\n大类是底模或独立身份；颜色是实例小类，不能用同一底模的多个颜色伪装不同长相。逐小类 ID、路径和状态见 [总登记 JSON](../registrations/v7_asset_roster.json) 与 [总 CSV](../registrations/v7_asset_roster.csv)。\n\n"
 text+=table(['大类','major_id','小类','数量','后端','运动能力','坐姿','坐着讲话','站着讲话手势','转身','可坐座位','嘴部/主声源'],humanrows)
 text+="\n运动通过范围是本包的两间 Habitat 场景和酷家乐原生地图指定路线/座位。新 episode 必须重做地面、碰撞、导航与声源随动验收。坐着讲话目前完整检查了凳子配对；其他静态座椅兼容性不等于相应手势全片段也通过。LX 只静态，钢琴人物原网格含琴凳、手鼓人物含折叠椅；姿态细节见登记的 seat_contact_qualification。\n\n"
 for major in a['animals']:
  text+=f"## {major['name_zh']}\n\n"+table(['小类','asset_id','双端可用','运动能力','走路','嘴/鸟嘴','许可及资格'],[[v['name_zh'],v['asset_id'],label(v['dual_backend_static_usable']),'、'.join({'stand':'站','walk':'走','sit':'坐','static':'静态'}.get(x,x)for x in v['movement']),label(v.get('walk_allowed',False)),str(v.get('beak_local_glb_m','Xtra Mouth；逐帧随动')),v.get('source_license',v.get('reason',''))]for v in major['minor_variants']])+'\n'
 (D/'INVENTORY_zh.md').write_text(text)
 chairtext="本包提供五种静态、无声、哑光座椅；优先选已有语义座位，确无可用座位时才补放。\n\n"+table(['ID','类型','座高 m','座宽 m','座深 m','Habitat','UE','发声'],[[c['asset_id'],sty[c['style']],c['seat_height_m'],c['seat_width_m'],c['seat_depth_m'],c['habitat_status'],c['ue_status'],'否']for c in a['chairs']])
 chairtext+="\n30 人 × 5 款的静态几何配对共 150 项；通过 147 项。医护男 01 的低座高不支持餐椅/办公椅；警察女 01 与办公椅扶手冲突。每人至少有一种可用类型，逐项偏移、座高实例缩放和接触点见 round3/evidence/chair_body_compatibility.json。不能仅按类别名放椅子。\n\n材质 metallic=0、roughness=0.85、specular IOR level=0.15；高光采用原生图像饱和像素比例代理，不能解释为物理镜面反射严格为零。新场景的墙/家具、主通道及连通性需重新检查。\n"
 (D/'CHAIR_INVENTORY_zh.md').write_text(chairtext)
 report=f"人物 {s['dual_backend_usable_major']} 个大类、{s['dual_backend_usable_minor']} 个小类双端静态可用；其中 {s['walking_major_admitted']} 个 Rocketbox 底模可走，{s['seated_major_admitted']} 个可坐；另有 {s['dog_minor_dual_backend_usable']} 种狗、{s['chair_styles_dual_backend_usable']} 款无声座椅，家养鸟目标仍未完成。\n\n"
 report+=f"全部文件位于 48g（cw-SYS-4029GP-TRT3）的 {T}。独立检出 worktree，分支 codex/v7-human-roster-20261004。原库、LX tar、其他检出、共享环境只读；未修改 v7-spec 或 v7-families，未发给 owner。提交号见 evidence/commit_round3.txt。\n\n"
 report+=table(['交付组','大类','小类','说明'],[['便装换色',16,96,'女8/男8，每个红绿蓝黄白黑六色'],['职业身份',14,14,'女7/男7，消防/医护/警察/建筑/商务/飞行员/厨师/园丁；不换制服'],['LX静态',10,10,'不加走路；10个对应干声类别由 families 准入'],['狗',1,2,'比格与德国牧羊犬；原目标几何、兼容动作'],['鸟',1,s['bird_library_fallback_dual_backend_usable'],'麻雀仅兜底；鹦鹉2D候选未成3D'],['无声座椅',5,5,'餐椅、办公椅、扶手椅、凳子、沙发椅']])
 report+="\n登记入口 [v7_asset_roster.json](../registrations/v7_asset_roster.json)，人工清单 [INVENTORY_zh.md](INVENTORY_zh.md)、[座椅清单](CHAIR_INVENTORY_zh.md)，接入见 [INTEGRATION_zh.md](INTEGRATION_zh.md)，许可见 [LICENSES_zh.md](LICENSES_zh.md)。截图页 [review.html](review.html) 是单文件 JPEG 内嵌，未发送 owner。\n\n"
 report+="换色口径为 replace_base_color_preserve_pbr_detail_v1，仅 shirt_main_color 遮罩换基色，保留 PBR 亮度细节、条纹/印花/纽扣，排除皮肤。96 个派生和14个原色身份材质检查均通过；遮罩外/保护区域变更像素均为0，来源 evidence/material_variant_equivalence.json。严格 schema 每属性最多3值，拆32个profile、96个请求，共用16个底模 lineage；原共享注册第六段未执行，来源 evidence/strict_profile_validation.json。\n\n"
 report+="Habitat 两房 192 次渲染颜色测量、480 组色差，标称色名与色差均0失败，最小 CIE76 ΔE=40.198260（阈值20）；来源 [rendered_color_check.json](../evidence/rendered_color_check.json)。UE 不使用贴图色差判定。\n\n"
 hb=[read(T/'round2/evidence/native_habitat'/h/(m['major_id']+'.json'),{})for m in a['humans']['majors']if not m['major_id'].startswith('v7_lx_')for h in ['mp3d','hm3d']]
 er=read(R/'evidence/ue_extra_action_results.json');dr=read(R/'evidence/habitat_dog_motion_all.json');cr=read(R/'evidence/chair_native_placements.json')
 results=[['静态原生', 'Habitat240/240；UE120/120', 'evidence/habitat_native_results.json；ue_static_qa_summary.json'],['人物完整行走',f"Habitat {sum(x.get('status')=='pass'for x in hb)}/{len(hb)}；酷家乐真实USD三角碰撞 {sum(x['status']=='pass'for x in full)}/{len(full)}",'round2/evidence/native_habitat；round3/evidence/scene_motion_collision_results.json'],['UE完整姿态回读',f"{len(read(R/'evidence/ue_motion_results.json'))}条行走；{sum(x['status']=='pass'for x in er)}/{len(er)}手势/转身/坐着讲话",'round3/evidence/ue_motion_results.json；ue_extra_action_results.json'],['坐姿配座位',f"{sum(x['status']=='pass'for x in seats)}/{len(seats)}",'round3/evidence/seated_native_results.json'],['原生Habitat额外动作抽样',f"{sum(x['status']=='pass'for x in hx)}/{len(hx)}，720 JPEG",'round3/evidence/habitat_extra_action_results.json'],['UE实际Actor截图',f"{sum(x['qa_status']=='pass'for x in cap)}/{len(cap)}",'round3/evidence/ue_native_capture_qa.json'],['狗完整原生回放',f"Habitat {sum(x['status']=='pass'for x in dr)}/{len(dr)}；UE骨骼{sum(x['status']=='pass'for x in ud)}/{len(ud)}",'round3/evidence/habitat_dog_motion_all.json；ue_dog_motion_results.json'],['座椅原生摆放',f"Habitat {sum(x['status']=='pass_candidate'for x in cr)}/{len(cr)}；UE5款",'round3/evidence/chair_native_placements.json；ue_native_capture_qa.json'],['原始人物几何',f"{sum(x['status']=='pass'for x in source)}/{len(source)}，原动画表面误差最大0m",'round3/evidence/source_geometry_invariance.json']]
 report+=table(['验收','结果','出处'],results)
 seatmax=max(abs(x['seat_contact_gap_m'])for x in seats);footmax=max(x['maximum_foot_contact_gap_m']for x in seats);legmax=max([abs(q)for x in seats for q in x['chair_support_deltas_m']]or[0]);navmax=max(x['protected_route_navmesh_horizontal_error_m']for x in seats)
 report+=f"\n坐姿臀部支撑代理最大间隙 {seatmax*1000:.3f}mm、脚底/物理地面最大间隙 {footmax*1000:.3f}mm、椅腿支撑最大间隙 {legmax*1000:.3f}mm；重新加入静态人/椅后保护路线水平导航误差最大 {navmax:.9g}m，来源 round3/evidence/seated_native_results.json。这里验证的是座面范围内最低衣服表面，未独立分割隐藏的解剖臀部。\n"
 report+="\n实际语义座位经原生 RGBD 重新估计座面，拒绝把沙发垫当成脚底地面。MP3D 28人用已有座位、2人补放；HM3D 2人用已有座位、28人补放。座位候选11/7个，来源 round3/evidence/existing_seats_native.json。接触、家具避让和主路线保留均在本包指定摆放点验收；没有证明全屋所有通道或0–8人拥挤场景均通过。\n"
 report+="\n站着讲话使用官方性别匹配 talk clip 的6秒片段；转身为官方90°动作；坐着讲话使用官方坐姿手势的手臂/颈部链并保持已验收坐姿下半身/脊柱。坐着讲话的凳子配对完整检查；不要将147个静态座椅组合当成147个全动态手势组合。站立待机首帧可用，原长 Standing_Idle 序列保留，长循环尚未完整接触准入。\n"
 report+="\n人行走速度实测1.017294–1.193242m/s，使用本人源动作重定时及脚部IK贴地；嘴部随 MUpperLip逐帧移动。狗比格0.3m/s、德牧0.8m/s，嘴部随Xtra Mouth；德牧体长较大，酷家乐路线缩短到2.210945371m，避免末段墙碰撞。人和狗的原生完整回放与源场景三角BVH检查互相补充；导入全屋网格的UE原生物理射线不可用，没有把源USD FCL验收说成UE物理验收。数值出处：round3/evidence/walk_metric_summary.json、round2/assets/humans/*/walking_<house>.json、round2/assets/dogs/*/motion_summary.json、round3/registrations/dog_paths/german_shepherd/kujiale.json。\n"
 report+="\n90条行走轨迹最大接触/地面误差0.997794mm、支撑点锁定误差0.229839mm、接触区域顶点水平漂移0.033923mm；转弯角速度最大10.976535°/s。来源 round3/evidence/walk_metric_summary.json。新增60/60原生可见性抽样相机通过，240张JPEG的嘴部和双脚15px邻域有原生人物语义像素，见 round3/evidence/habitat_walk_visual_results.json；这是抽样可见性代理，完整接触证据仍由逐帧回放给出。\n"
 report+="\n脚步接触表共有689个事件：599次新接触、90个初始支撑，来源 round3/registrations/foot_contact_events.json。未准入/合成脚步声音频，所以步伐与录音同步尚未验证；families只应在采用这张事件表并复核音轨后启用脚步声题族。LX主声源按嘴、乐器或手掌登记，10项中远景三角表面锚点见 round3/registrations/lx_sound_anchors.json；非精确辐射模型。\n"
 report+="\n110个可动小类的完整PBR外观并入动画master内存副本均通过：原始关节/皮肤/动画元数据不变，同一行走姿态表面差0m，所选小类纹理字节完全相等。见 round3/evidence/motion_colour_binding.json；函数 v7_roster_colour_motion.apply_appearance，不另存110份动画master。\n"
 report+="\n高光是原生RGB饱和像素比例代理，阈值2%，并非镜面AOV。UE按已记录的室内补光条件采样，不做贴图色差。全部新验收图为JPEG质量85；动作只有抽样帧，没有PNG序列。历史换色量化的无损RGB/遮罩PNG保留以便复算。\n"
 report+="\n资源操作异常记录：LX重试时未等旧UE退出，本任务两个UE进程曾短时重叠，精确GPU同时运行时长未验证；随后核对PID/命令并停止自有进程，新增任务内互斥锁及回归测试。见 round3/evidence/gpu_retry_overlap_audit.json。没有因此终止或修改其他任务。\n"
 report+="\n家养虎皮鹦鹉已完成FLUX.2-klein-4B参考图和ISNet分割，未进入Pixal3D。项目生成动物约定明确要求 owner 审核2D品种及解剖后才能重建（worktree/docs/assets/GENERATED_ANIMAL_ASSET_AND_INSTANCE_CONTRACT.md：“Hard stop for project-owner review.”）。本轮无人可确认，因此没有自批或将麻雀改名为鹦鹉。麻雀是任务书明确允许的静态兜底；不满足家养鸟目标。ISNet独立ONNX权重/输出许可范围仍未确认；其他许可及DINOv3论文致谢见许可记录。\n"
 report+="\n磁盘前后对比、已回收及可再回收项见 [DISK_zh.md](DISK_zh.md)。曾观察共享/data因并发占用跌破阈值，守护停止本任务写入并仅清理自有已验证中间件；不能声称任意时刻共享磁盘都由本任务保证≥50GiB。\n\n复现实际执行入口位于 round3/scripts/：habitat_extra.sh、habitat_bird.sh、ue_import_gpu.sh、ue_capture_gpu.sh、contact_tests.sh、fast_suite_final.sh；版本 Blender4.2.1 LTS、UE5.5.4、原生Habitat runtime前缀记录在原生证据。最终测试与进程核对见 evidence/final_delivery_checks.json；阶段记录见 PROGRESS_zh.md。\n"
 (D/'REPORT_zh.md').write_text(report)
 # One review page; every bitmap is JPEG, never an external image dependency.
 check(T,256*1024**2);pictures=[];seen=set()
 def add(path,title,group,status=''):
  path=Path(path)
  if not path.is_absolute():path=T/path
  if not path.is_file()or str(path)in seen:return
  seen.add(str(path));pictures.append((path,title,group,status))
 for r in read(T/'evidence/habitat_native_results.json'):add(r['rgb'],names.get(r['asset_id'],r['asset_id'])+' / '+r['color']+' / '+r['house'],'人物静态 Habitat',r['status'])
 for r in read(T/'evidence/ue_native_results.json'):
  for k in ['rgb','image']:
   if k in r:add(r[k],names.get(r['asset_id'],r['asset_id'])+' / '+r.get('color','')+' / Kujiale','人物静态 UE',r.get('status','native'))
 for p in (R/'evidence/habitat_walk_visual').rglob('*.json'):
  r=read(p,{});[add(f['image'],names.get(r['asset_id'],r['asset_id'])+' / '+r['house']+' / Walking / '+str(f['frame']),'行走可见性 Habitat',r['status'])for f in r['frames']if 'image'in f]
 for r in read(R/'evidence/lx_primary_ue_results.json'):add(r['image'],names.get(r['asset_id'],r['asset_id'])+' / '+r['primary_part'],'LX主声源 UE',r['status'])
 for r in cap:add(r['rgb'],names.get(r['asset_id'],r['asset_id'])+' / '+r['family']+' / '+str(r.get('frame','pose')),'新动作/座椅/动物 UE',r['qa_status'])
 for r in seats:add(r['image'],names.get(r['asset_id'],r['asset_id'])+' / '+r['house']+' / '+r['seat_source']+' '+r['seat_type'],'人物配座位 Habitat',r['status'])
 for p in (R/'evidence/habitat_extra').rglob('*.json'):
  r=read(p,{});[add(f['image'],names.get(r['asset_id'],r['asset_id'])+' / '+r['house']+' / '+r['family']+' / '+str(f['frame']),'额外动作 Habitat',r['status'])for f in r['frames']]
 for r in cr:add(r['image'],r['asset_id']+' / '+r['house'],'座椅 Habitat',r['status'])
 for r in dr:
  for f in r['images']:add(f['path'],r['asset_id']+' / '+r['house']+' / '+str(f['frame']),'狗 Habitat',r['status'])
 for r in bd:add(r['image'],r['house']+' / 麻雀兜底（非家养鸟）','鸟兜底 Habitat',r['status'])
 for r in read(R/'registrations/lx_sound_anchors.json'):add(r['review_jpeg'],names.get(r['asset_id'],r['asset_id'])+' / '+r['primary_part'],'LX主声源',r['status'])
 for p in (R/'images/lx_seat_review').rglob('*.jpg'):add(p,str(p.relative_to(R/'images/lx_seat_review')),'LX座椅复核','native')
 add(T/'round2/assets/bird_2d/candidate.png','虎皮鹦鹉2D候选；未审核、未成3D、不可采样','家养鸟候选','未准入')
 parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>v7 人物、动物和座椅验收</title><style>body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#18212e}input{padding:10px;width:70%}summary{padding:12px;background:white;cursor:pointer}section{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}figure{margin:0;background:white;padding:10px}img{width:100%;height:auto}figcaption{font-size:14px;overflow-wrap:anywhere}.bad{color:#b02020}details{margin:18px 0}header{position:sticky;top:0;background:#f5f6f8;padding:10px;z-index:1}</style><header><h1>v7 人物、动物、无声座椅</h1><p>所有画面来自原生 Habitat 或 UE；静态全颜色、抽样动作与接触配座位分开标明。家养鸟候选未准入；麻雀只作兜底。高光是饱和像素代理，UE不做贴图色差；完整逐帧几何与嘴部回读见JSON证据。</p><input id="filter" placeholder="搜索身份、颜色、动作或后端"><p id="counter"></p></header>']
 groups=collections.defaultdict(list)
 for p,title,group,status in pictures:
  if p.suffix.lower()in ['.jpg','.jpeg']:raw=p.read_bytes()
  else:
   buf=io.BytesIO();Image.open(p).convert('RGB').save(buf,format='JPEG',quality=85);raw=buf.getvalue()
  caption=html.escape(title+' — '+status);key=html.escape(title+' '+group+' '+status,quote=True);encoded=base64.b64encode(raw).decode();groups[group].append(f'<figure data-key="{key}"><img loading="lazy" alt="{caption}" src="data:image/jpeg;base64,{encoded}"><figcaption>{caption}</figcaption></figure>')
 for group,tiles in groups.items():parts.append('<details><summary>'+html.escape(group)+f'（{len(tiles)}）</summary><section>'+''.join(tiles)+'</section></details>')
 parts.append('<script>const rows=[...document.querySelectorAll("figure")];function filter(){let q=document.querySelector("#filter").value.toLowerCase(),n=0;for(const r of rows){const ok=r.dataset.key.toLowerCase().includes(q);r.hidden=!ok;if(ok)n++;}document.querySelector("#counter").textContent=n+" / "+rows.length+" 张 JPEG";}document.querySelector("#filter").oninput=filter;filter();</script></html>')
 page=''.join(parts);(D/'review.html').write_text(page);json_write(R/'evidence/review_page_manifest.json',{'image_count':len(pictures),'html_bytes':len(page.encode()),'groups':{k:len(v)for k,v in groups.items()},'image_format':'embedded JPEG, quality85 for new captures/inline PNG conversions','external_image_dependencies':0});print('DELIVERY_REVIEW',len(pictures),len(page.encode()),'SUMMARY',s)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);run(p.parse_args().task_root)
