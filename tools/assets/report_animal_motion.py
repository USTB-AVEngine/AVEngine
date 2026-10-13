"""Chinese, self-contained per-individual motion admissions and native evidence."""
import argparse,base64,collections,datetime,html,json,subprocess
from pathlib import Path

def collect_admissions(r):
 r=Path(r);inv=json.loads((r/'diagnostics/inventory.json').read_text())['assets'];native=[];audio=[];ue_native=[];raw_audio=[];requests={};request_records=[]
 for path in sorted([*r.glob('habitat/*/native_motion.json'),*r.glob('requests/*/*/native_motion.json')],key=lambda x:x.stat().st_mtime):
  d=json.loads(path.read_text());d['_path']=str(path)
  if d.get('checks',{}).get('status')=='pass' and all(f.get('complete_silhouette_in_frame') for f in d['frames']):native.append(d)
 for path in sorted(r.glob('ue/*/native_motion.json'),key=lambda x:x.stat().st_mtime):
  d=json.loads(path.read_text());d['_path']=str(path)
  if d.get('checks',{}).get('status')=='pass' and all(f.get('complete_silhouette_in_frame') for f in d['frames']):ue_native.append(d)
 for path in [*r.glob('habitat/*/audio.json'),*r.glob('requests/*/*/audio/audio.json'),*r.glob('requests/*/audio.json'),*r.glob('ue/*/audio.json')]:
  d=json.loads(path.read_text());d['_path']=str(path)
  raw_audio.append(d)
  if d.get('status')=='pass':
   ref=Path(d['motion_receipt']);d['_motion']=json.loads(ref.read_text());audio.append(d)
 for path in sorted(r.glob('requests/*/admission.json'),key=lambda x:x.stat().st_mtime):
  d=json.loads(path.read_text());d['_path']=str(path);requests[d['asset_id']]=d;request_records.append(d)
 ue_summary_path=r/'ue/UE_COVERAGE_STATUS.json'
 ue_summary=json.loads(ue_summary_path.read_text()) if ue_summary_path.exists() else {}
 ue_by_asset={x['asset_id']:x for x in ue_summary.get('assets',[])}
 external_reviews={}
 for review_path in sorted(r.glob('diagnostics/external_independent_review/*/review.json')):
  review=json.loads(review_path.read_text())
  if review.get('status')!='pass':continue
  npath=Path(review['native_motion']);apath=Path(review['audio']);n=json.loads(npath.read_text());ad=json.loads(apath.read_text())
  if n['asset_id']!=review['asset_id'] or n.get('checks',{}).get('status')!='pass' or ad.get('status')!='pass':raise ValueError('external_review_receipt_no_longer_matches')
  n.update(_path=str(npath),_external_review_path=str(review_path));ad.update(_path=str(apath),_motion=json.loads(Path(ad['motion_receipt']).read_text()))
  native.append(n);audio.append(ad);external_reviews[review['asset_id']]=review
 external_summary_path=r/'diagnostics/external_rig_progress_readonly.json'
 external_summary=json.loads(external_summary_path.read_text()) if external_summary_path.exists() else {}
 rows=[]
 def identity(n):return (n['house'],n['listener_m'],[(f['time_s'],f['mouth_world_m']) for f in n['frames']])
 for x in inv:
  ns=[n for n in native if n['asset_id']==x['asset_id']]
  qualified=[(n,a) for n in ns for a in audio if identity(n)==identity(a['_motion'])]
  animated=[pair for pair in qualified if pair[0].get('motion_mode','walk_animation')=='walk_animation']
  original_pairs=[pair for pair in qualified if not pair[0].get('_external_review_path')];original_animated=[pair for pair in original_pairs if pair[0].get('motion_mode','walk_animation')=='walk_animation'];original_pair=(original_animated or original_pairs)[-1] if original_pairs else None
  pair=(animated or qualified)[-1] if qualified else None
  n=pair[0] if pair else (ns[-1] if ns else None)
  matches=[pair[1]] if pair else []
  mode=n.get('motion_mode','walk_animation') if matches else 'cannot_move'
  reason='完整原生几何和移动RIR准入通过' if matches else ('几何通过，移动RIR未通过或未完成' if n else '本轮完整运动准入未完成；不等于已证明个体永远不能移动')
  row={'asset_id':x['asset_id'],'species':x['species'],'source_has_skeleton':x['has_skeleton'],'source_has_Walking':x['has_walking'],'motion_mode':mode,'reason':reason,'habitat_geometry_status':'pass' if n else 'not_admitted','habitat_motion':n['_path'] if n else None,'audio':matches[-1]['_path'] if matches else None,'ue_status':'not_admitted','scope':'本轮指定房间和路线；未验证的新房间/路线仍须重新准入'}
  row['original_configuration_motion_mode']=original_pair[0].get('motion_mode','walk_animation') if original_pair else 'cannot_move'
  if n and n.get('_external_review_path'):
   review=external_reviews[x['asset_id']]
   row['derived_binding']={'independent_review':n['_external_review_path'],'external_admission':review['external_admission'],'derived_source':review['derived_source'],'candidate':n.get('source_provenance',{}).get('candidate'),'manifest':n.get('source_provenance',{}).get('manifest'),'original_configuration_retained':True,'original_registry_modified':False}
   row['reason']='原始无骨来源的外部派生Walking，经本任务逐帧与原始RIR真实卷积独立复核完整通过；原来源与原合格滑行证据保留'
  if mode=='slide_no_animation':
   linked=[v for v in request_records if v['asset_id']==x['asset_id'] and v.get('status')=='pass' and v.get('motion_mode')==mode and v.get('receipts',{}).get(mode,{}).get('native')==n['_path'] and v.get('receipts',{}).get(mode,{}).get('audio')==matches[-1]['_path']]
   admission=linked[-1] if linked else None
   row['animation_unavailable_reason']=admission.get('motion_reason') if admission else ('registered_target_has_no_skeleton_or_Walking' if not x['has_walking'] or not x['has_skeleton'] else 'Walking未通过，见原生/声学失败收据')
   row['reason']='滑行原生几何和移动RIR完整通过；动画不可用原因：'+str(row['animation_unavailable_reason'])
   if admission:row['admission_evidence']=admission['_path'];row['animation_attempts']=admission.get('motion_attempts',[])
  if n:row['habitat_clip']={'start_s':n['frames'][0]['time_s'],'end_s':n['frames'][-1]['time_s'],'frame_count':len(n['frames']),'selection_reason':n.get('selection_reason'),'original_longer_audio_failure':n.get('original_longer_audio_failure')}
  if n:row['metrics']={k:n['checks'][k] for k in ('frame_count','max_all_four_paw_gap_m','max_stance_drift_m','max_stance_speed_m_s','min_full_skin_floor_m')};row['metrics']['max_heading_deg']=n['checks']['direction']['maximum_angle_deg'];row['metrics']['max_mouth_error_m']=n.get('max_emitter_error_m',max(f['native_emitter_readback_error_m'] for f in n['frames']))
  if matches:row['audio_reuse_identity']={'original_asset':matches[-1]['_motion']['asset_id'],'every_actual_mouth_time_position_and_listener_exactly_equal':True,'same_static_room_acoustic_geometry':True};row['audio_metrics']={k:matches[-1].get(k) for k in ('maximum_native_direction_error_deg','maximum_segment_direction_sweep_deg','maximum_abs_energy_error_db')}
  if not matches:
   rejected=requests.get(x['asset_id'])
   if rejected and rejected.get('motion_mode')=='cannot_move':
    row['reason']=rejected.get('motion_reason','运动准入未通过');row['failure_evidence']=rejected['_path']
   else:
    rejected_audio=[v for v in raw_audio if v.get('asset_id')==x['asset_id'] and v.get('status')!='pass']
    if rejected_audio:
     bad=rejected_audio[-1];row['reason']='几何通过；声学未通过：'+str(bad.get('error') or bad.get('qualification',{}).get('failures'));row['failure_evidence']=bad['_path']
  row['responsibility']='原有骨骼17只' if x['has_walking'] else '新绑骨由其他代理负责；本任务保留滑行兜底'
  if row.get('derived_binding'):row['responsibility']='外部代理新绑定派生副本；本任务只读独立复核后纳入'
  uns=[u for u in ue_native if u['asset_id']==x['asset_id']]
  upairs=[(u,a) for u in uns for a in audio if identity(u)==identity(a['_motion'])]
  un=upairs[-1][0] if upairs else (uns[-1] if uns else None)
  row['ue_status']='pass' if upairs else ('geometry_pass_audio_pending' if un else 'not_run_or_not_admitted')
  if un:row['ue_motion']=un['_path'];row['ue_video']=un.get('video',{}).get('path')
  if upairs:row['ue_audio']=upairs[-1][1]['_path']
  row['ue_geometry_status']='pass' if un else 'not_admitted'
  row['ue_audio_status']='pass' if upairs else 'not_admitted'
  ue_record=ue_by_asset.get(x['asset_id'])
  if ue_record:
   row['ue_execution_summary']=ue_record
   if not upairs:row['ue_status']=ue_record['status'] if ue_record['status']!='pass' else 'claimed_pass_without_complete_receipt_pair'
  rows.append(row)
 counts={s:dict(collections.Counter(x['motion_mode'] for x in rows if x['species']==s)) for s in ('dog','cat')};result={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts':counts,'assets':rows,'registry_files_modified':False,'training_tables_exported':False,'ue_coverage':ue_summary,'ue_admitted_asset_ids':[x['asset_id'] for x in rows if x['ue_status']=='pass']}
 result['original_rig_counts']={species:dict(collections.Counter(x['motion_mode'] for x in rows if x['species']==species and x['source_has_Walking'])) for species in ('dog','cat')}
 result['original_static_counts']={species:dict(collections.Counter(x['motion_mode'] for x in rows if x['species']==species and not x['source_has_Walking'])) for species in ('dog','cat')}
 result['new_binding_admitted_asset_ids']=[x['asset_id'] for x in rows if x.get('derived_binding')]
 result['external_rig_progress']=external_summary
 return result,native

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--json-only',action='store_true',help='Collect receipts for final audit without writing HTML');a=p.parse_args();r=a.root
 result,native=collect_admissions(r);rows=result['assets'];counts=result['counts'];ue_summary=result['ue_coverage'];external_summary=result['external_rig_progress']
 (r/'motion_admissions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 if a.json_only:
  print(json.dumps({'counts':counts,'admissions':str(r/'motion_admissions.json')},ensure_ascii=False));return
 def esc(x):return html.escape(str(x))
 def picture(path,caption):
  p=Path(path)
  return '<figure><img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"><figcaption>'+esc(caption)+'</figcaption></figure>' if p.exists() else ''
 mode_names={'walk_animation':'有动画且完整通过','slide_no_animation':'只能滑行且完整通过','cannot_move':'本轮不能移动 / 未准入'}
 lead='；'.join({'dog':'狗','cat':'猫'}[s]+'：'+', '.join(mode_names[k]+str(counts[s].get(k,0)) for k in mode_names) for s in counts)
 content=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>v7 狗猫运动修复</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1380px;margin:32px auto;padding:0 20px;color:#1b2530}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border:1px solid #ccd5df;overflow-wrap:anywhere}th{background:#edf3f8}figure{display:inline-block;width:46%;vertical-align:top;margin:1%}img{max-width:100%}.note{background:#fff6df;padding:18px}code{overflow-wrap:anywhere}details{margin:12px 0}h1,h2{line-height:1.3}</style><h1>v7 狗猫运动：原生证据报告</h1><p><b>Habitat 完整准入：'+esc(lead)+'</b></p>',
 '<p>原17个带骨骼来源：'+esc(json.dumps(result['original_rig_counts'],ensure_ascii=False))+'；原46个静态来源：'+esc(json.dumps(result['original_static_counts'],ensure_ascii=False))+'。新增绑定只有本任务逐帧与原始RIR真实消费独立复核通过后才升级主表；每行保留原始骨骼标志、原配置结果和外部派生路径。</p>',
 '<p class="note"><b>工程边界：</b>本轮CLI/API会实际执行单源、指定房间路线的原生渲染与准入；合格RIR已由引擎真实卷积。旧多源30秒生产视觉调度尚未消费 native_frame_source，未默认接入本轮动作。QA与题目标记仅在调用方显式传入 motion_metadata 时传播；完整生产调度对接留给后续integration工作，不能把短片直接替代30秒场景。本轮没有导出训练表。</p>',
 '<p class="note">完整通过须同时满足：逐帧头尾前向与平移速度夹角≤15°；支撑脚固定材料顶点漂移≤3 mm、滑速≤30 mm/s；四爪（含本轮8 mm摆动）距地≤12 mm；全皮肤穿地不超过原1 mm+10 µm数值余量；原始场景三角净空&gt;20 mm；NavMesh水平偏差≤2 mm；嘴部原生读回≤0.1 mm；RIR方向/分段扫角≤5°，原1 dB能量、DRR、到达时间及衰减门槛全部通过。滑行明确不作步态/锁脚动画声明。未验证不计入通过。</p>',
 '<p>原输入：37狗、26猫。实读GLB确认12狗、5猫有骨骼和Walking；25狗、21猫无骨骼。17个原Walking经过登记规范化后的方向未发现180°反向；新接口显式统一规范化矩阵，关闭人物默认180°翻转，并逐帧从实际根读回验证。独立180°/90°负例包含真实旋转的Habitat实体。</p>',
 '<p>猫接地修复：原41帧缓存有空支撑标签，并允许约8–12 mm整身穿地；回到原目标骨骼，先做四爪支撑面校正，再重建慢步支撑轨迹和IK。德牧/英短锁踝后仍有脚趾卷曲，分别抬起固定支撑点约21.2/16.1 mm；锁定踝以下的目标自身Idle脚形解决该问题。原17个骨骼源的网格拓扑和权重保持；原始主资产与登记均未修改。原静态个体新增绑定使用外部派生副本，逐项列明来源。</p>',
 '<p>声学修复：原生持久上下文有同位置重复波动；新进程重算可复现。宽带相干交叉淡化还会压低高频能量；本轮新增按频带、仅依赖端点的相干权重，并只对仍失败的区间增加原生采样。参考RIR不参与拟合，不平均参考，不放宽判定器。</p>',
 '<p>实测范围：Habitat 为 HM3D hm3d_train_00272_kA2nG18hCAr 指定路线；UE 为下列已完成原生和声学准入的个体在酷家乐房间的4秒回放；CPU导出与预检单列。柯基采用同次原生回放的前4秒121帧，原4.8秒尾段声学失败保留；UE同样只对前4秒作完整准入。其他房间、路线与更长片段仍须重新验证。</p>',
 '<p>报告时间：'+esc(result['generated_utc'])+'。这是逐房间运动准入，未生成或导出训练表。UE结果只有原生完成后才计入；当前未准入状态不能写成通过。</p><h2>Habitat 逐个体记录（63只），UE 独立列示</h2><input id="q" placeholder="按资产名、类别、结论筛选" style="width:70%;padding:10px"><table id="assets"><thead><tr><th>个体</th><th>类别</th><th>骨骼 / Walking</th><th>Habitat 运动准入</th><th>UE 独立状态</th><th>原因 / 证据</th></tr></thead><tbody>']
 for x in rows:
  content.append('<tr><td>'+esc(x['asset_id'])+'</td><td>'+esc(x['species'])+'</td><td>'+esc(str(x['source_has_skeleton'])+' / '+str(x['source_has_Walking']))+'</td><td>'+esc(mode_names[x['motion_mode']])+'</td><td>'+esc(x['ue_status'])+'</td><td>'+esc(x['reason'])+'<details><summary>数值与来源</summary><pre>'+esc(json.dumps(x,ensure_ascii=False,indent=2))+'</pre></details></td></tr>')
 content.append('</tbody></table><h2>原生逐帧截图与修复前后</h2><p>绿色箭头：尾到头；红色箭头：实际平移速度。下图来自真实房间原生渲染。前后图分别为原Walking规范化姿态与本轮接触修复，未用替代模型。</p>')
 shown=set()
 for selected_row in rows:
  if not selected_row.get('habitat_motion'):continue
  n=next(item for item in native if item['_path']==selected_row['habitat_motion'])
  if n['asset_id'] in shown:continue
  shown.add(n['asset_id']);content.append('<h3>'+esc(n['asset_id'])+'</h3>')
  for path in n.get('images',[])[::2]:content.append(picture(path,'修复后：'+Path(path).name))
  for path in n.get('before_images',[])[::2]:content.append(picture(path,'原Walking（未做本轮接触修复）：'+Path(path).name))
  content.append('<details><summary>逐帧机器检查</summary><pre>'+esc(json.dumps({k:v for k,v in n['checks'].items() if k!='direction'},ensure_ascii=False,indent=2))+'</pre><code>'+esc(n['_path'])+'</code></details>')
 content.append('<h2>酷家乐 UE / SPEAR</h2><p>以下结果限定实测房间与路线。几何和视频通过、声学尚未通过时仍不计完整准入。</p>')
 if ue_summary:
  content.append('<h3>优先目标的 UE 阶段与失败记录</h3><p>'+esc(ue_summary.get('scope',''))+'；导出与CPU预检通过不计原生准入。</p><table><tr><th>个体</th><th>UE状态</th><th>阶段说明与证据</th></tr>')
  for source in ue_summary.get('assets',[]):
   row=next(x for x in rows if x['asset_id']==source['asset_id'])
   content.append('<tr><td>'+esc(source['asset_id'])+'</td><td>'+esc(row['ue_status'])+'</td><td>'+esc(source.get('reason',''))+'<details><summary>本轮阶段、原失败与收据</summary><pre>'+esc(json.dumps(source,ensure_ascii=False,indent=2))+'</pre></details></td></tr>')
  content.append('</table>')
 for x in rows:
  if x.get('ue_motion'):
   u=json.loads(Path(x['ue_motion']).read_text())
   content.append('<h3>'+esc(x['asset_id'])+'：'+esc(x['ue_status'])+'</h3><code>'+esc(x['ue_motion'])+'</code>')
   for path in u.get('images',[]):content.append(picture(path,'UE 实际回放：'+Path(path).name))
   content.append('<details><summary>UE 数值与视频路径</summary><pre>'+esc(json.dumps({'checks':{k:v for k,v in u['checks'].items() if k!='direction'},'minimum_unoccluded_fraction':min((f.get('unoccluded_fraction',0) for f in u['frames']),default=0),'clip_end_s':u['frames'][-1]['time_s'],'measurement_source':u.get('measurement_source'),'selection_reason':u.get('selection_reason'),'original_longer_clip_failure':u.get('original_4p8s_audio_failure'),'video':u.get('video'),'audio':x.get('ue_audio')},ensure_ascii=False,indent=2))+'</pre></details>')
 controls=r/'diagnostics/native_direction_controls_v1/controls.json'
 if controls.exists():
  cc=json.loads(controls.read_text());content.append('<h2>真正反向与横向的负例</h2><p>'+esc(cc['status'])+'；这些实例故意旋转，只用于证明检测器拒绝，不计准入。</p>')
  for c in cc['controls']:
   for f in c['frames']:
    if f.get('image'):content.append(picture(f['image'],c['asset_id']+' / 故意旋转 '+str(c['body_rotation_error_deg'])+'° / 检测 '+c['check']['status']))
 ue_controls=r/'ue/beagle_direction_controls_v1/native_motion.json'
 if ue_controls.exists():
  cc=json.loads(ue_controls.read_text());content.append('<h2>UE 实际方向负例</h2><p>'+esc(cc['status'])+'；180°和90°均在原生房间里实际旋转骨骼实例后读回。</p>')
  for c in cc.get('controls',[]):
   if len(c['frames'])>1:content.append(picture(c['frames'][1]['image'],'UE 故意旋转 '+str(c['body_rotation_error_deg'])+'°；检测 '+c['check']['status']))
 if external_summary:
  content.append('<h2>原静态46只的外部绑定分工</h2><p>只读快照时间 '+esc(external_summary.get('read_at_utc'))+'。外部声称通过与本任务独立复核分别列示；仅几何通过不计完整Walking。派生副本不会改写原登记或默认接入旧生产调度。</p><table><tr><th>个体</th><th>外部阶段</th><th>本任务独立复核</th><th>来源与原因</th></tr>')
  for source in external_summary.get('assets',[]):
   content.append('<tr><td>'+esc(source['asset_id'])+'</td><td>'+esc(str(source.get('reported_motion_mode'))+' / '+str(source.get('reported_status')))+'<br>几何 '+esc(source.get('native_receipt_status'))+'；声学 '+esc(source.get('audio_receipt_status'))+'</td><td>'+esc(source.get('independent_review',{}).get('status','尚未独立复核'))+'</td><td>'+esc(source.get('reason'))+'<details><summary>证据与绑定尝试</summary><pre>'+esc(json.dumps(source,ensure_ascii=False,indent=2))+'</pre></details></td></tr>')
  content.append('</table>')
 resource_path=r/'RESOURCE_USAGE.json'
 if resource_path.exists():
  resource=json.loads(resource_path.read_text());summary={k:v for k,v in resource.items() if k not in ('jobs','gpu_intervals')}
  content.append('<h2>资源账单</h2><p>已记录CPU下界 '+esc(round(resource['recorded_cpu_hours'],3))+' 核时；GPU0相关UE作业墙钟并集 '+esc(round(resource['gpu_0_related_job_wall_seconds_union']/3600,3))+' 小时（包括启动、等待和清理，不是kernel时间）。本输出目录占用 '+esc(round(resource['own_output_allocated_bytes']/1e9,3))+' GB，较预算基线新增 '+esc(round(resource['additional_bytes_since_latest_budget']/1e9,3))+' GB。外部绑定代理成本不计入本任务目录账单。</p><details><summary>统计范围、缺失项与证据</summary><pre>'+esc(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><code>'+esc(resource_path)+'</code></details>')
 content.append('<h2>失败与未完成</h2>')
 for pth in sorted((r/'derived').glob('rig_*/rigging.json')):
  content.append('<pre>'+esc(pth.read_text())+'</pre>')
 failures=[x for x in rows if x['motion_mode']=='cannot_move'];content.append('<p>'+esc(str(len(failures))+'只未完成完整准入；每只原因见表。几何通过不替代声学通过，导入成功不替代UE回放。')+'</p>')
 jobs=json.loads((r/'JOBS.json').read_text());content.append('<details><summary>所有作业、命令与资源记录</summary><pre>'+esc(json.dumps(jobs,ensure_ascii=False,indent=2))+'</pre></details><script>document.getElementById("q").oninput=function(){let q=this.value.toLowerCase();for(let r of document.querySelectorAll("#assets tbody tr"))r.style.display=r.textContent.toLowerCase().includes(q)?"":"none"}</script></html>')
 (r/'REPORT.html').write_text(''.join(content));print(json.dumps({'counts':counts,'report':str(r/'REPORT.html'),'native_geometry_assets':len(shown)},ensure_ascii=False))
if __name__=='__main__':main()
