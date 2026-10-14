"""Read-only evidence requalification and a compact self-contained Chinese report.

Native room/heading/ground/attachment evidence is recomputed from every stored
frame. Complete same-source surface measurements are evaluated under the named
owner policy. Historical acoustics are preserved as history, never as a gate.
"""
import argparse,base64,collections,datetime,hashlib,html,json
from pathlib import Path
from avengine.assets.animal_motion.motion_only_policy import qualify_motion,rigid_slide_surface_proof,validation_policy

MODES={'walk_animation':'有动画','slide_no_animation':'只能滑行','cannot_move':'不能移动（本轮路线未准入）'}
def read(path):return json.loads(Path(path).read_text())
def hashed(path):
 p=Path(path);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def evaluate(native_path,*,surface_path=None,cpu_path=None,controls_path=None,preservation_paths=(),source_label=''):
 n=read(native_path);mode=n.get('motion_mode');sf=None;association={};sources=[hashed(native_path)]
 if mode=='slide_no_animation':sf=rigid_slide_surface_proof(n)
 elif cpu_path:
  cp=read(cpu_path);sf=cp.get('surface_checks');sources.append(hashed(cpu_path))
  same=cp.get('source_master')==n.get('source_master') and cp.get('asset_id')==n.get('asset_id')
  a=cp.get('frames',[]);b=n.get('frames',[])
  clock=len(a)==len(b) and all(x['time_s']==y['time_s'] and x.get('source_phase_s')==y.get('source_phase_s') for x,y in zip(a,b))
  recipes=all(not f.get('pose_reconstruction') or (f['pose_reconstruction'].get('speed_m_s')==cp.get('speed_m_s') and f['pose_reconstruction'].get('period_s')==cp.get('period_s')) for f in b)
  association={'same_target_source':same,'complete_clock_and_source_phase':clock,'same_speed_and_period':recipes,
               'surface_scope':'flat compiled target skin; full per-frame native room/IK checks are independently retained',
               'native_pose_buffers_transient':any(f.get('pose_buffer_is_transient') for f in b),
               'all_native_vertex_surface_equivalence_independently_verified':False}
  if not(same and clock and recipes):sf=None
 elif surface_path:
  sf=read(surface_path);sources.append(hashed(surface_path));meta=sf.get('stream_metadata')
  if meta:
   md=read(meta);sources.append(hashed(meta));same=md.get('native_motion') in (None,str(native_path)) and md.get('source_master')==n.get('source_master')
   if not same:sf=None
   association={'same_target_source':same,'surface_scope':'actual saved native float32 skin','all_native_vertex_surface_equivalence_independently_verified':same}
 if controls_path=='embedded_in_native_motion':controls_path=None
 controls=read(controls_path) if controls_path else None
 if controls_path:sources.append(hashed(controls_path))
 q=qualify_motion(n,sf,controls)
 placement_proof=Path(native_path).parent.parent/'shape_preservation.json'
 if placement_proof.is_file():sources.append(hashed(placement_proof))
 preserved=[]
 for p in preservation_paths:
  d=read(p)
  if d.get('derived_glb')==n.get('source_master') and d.get('status')=='pass':preserved.append(str(p));sources.append(hashed(p))
 return {'asset_id':n['asset_id'],'motion_mode':mode,'status':q['status'],'qualification':q,'native_motion':str(native_path),
         'surface':str(surface_path) if surface_path else None,'cpu_checks':str(cpu_path) if cpu_path else None,
         'controls':str(controls_path) if controls_path else None,'source_preservation':preserved,'rigid_placement_proof':str(placement_proof) if placement_proof.is_file() else None,'surface_association':association,
         'source_label':source_label,'sources':sources,'frame_count':len(n.get('frames',[])),
         'duration_s':n['frames'][-1]['time_s'] if n.get('frames') else 0,'house':n.get('house'),
         'images':n.get('images',[]),'video':n.get('video',{}),'source_master':n.get('source_master')}

def collect(root,external_index,ue_path=None):
 inv=read(root/'diagnostics/inventory.json')['assets'];by={x['asset_id']:[] for x in inv};errors=[];seen=set()
 def add(native,**kw):
  key=(str(native),str(kw.get('surface_path')),str(kw.get('cpu_path')),str(kw.get('controls_path')))
  if key in seen:return
  seen.add(key)
  try:v=evaluate(native,**kw);by[v['asset_id']].append(v)
  except (OSError,ValueError,KeyError,TypeError) as e:errors.append({'native_motion':str(native),'reason':repr(e),'source_label':kw.get('source_label')})
 old=read(root/'inputs/original17_motion_evidence_owner093x.json')['assets'];controls=root/'diagnostics/original17_native_direction_owner093x_v1/controls.json'
 for x in old:
  surface=root/'motion_only/original17'/x['asset_id']/'surface_v3/surface.json'
  add(x['native_motion'],surface_path=surface if surface.exists() else None,controls_path=controls if controls.exists() else None,source_label='original_rig_walking')
 # New explicit requests carry exact same-source/same-route reused evidence.
 for path in sorted((root/'motion_only/repairs').glob('**/admission/admission.json')):
  d=read(path)
  case=path.parent.parent;preservation=case/'source_preservation.json';trial=read(case/'trial.json') if (case/'trial.json').exists() else {}
  preserved=[str(preservation)] if preservation.exists() else ([trial['reused_source_preservation']] if trial.get('reused_source_preservation') else [])
  for mode,e in d.get('receipts',{}).items():
   if e.get('native'):add(e['native'],surface_path=e.get('surface'),controls_path=e.get('controls'),preservation_paths=preserved,source_label='owned_CC0_repair:'+str(path))
 slide_index=root/'motion_only/original17_slides_owner112x_v1/index.json'
 if slide_index.exists():
  for x in read(slide_index).get('assets',{}).values():
   if x.get('native_motion') and Path(x['native_motion']).is_file():add(x['native_motion'],controls_path=x.get('direction_controls') or x.get('controls') or x.get('physical_controls'),source_label='owned_original17_registered_slide')
 ext=read(external_index)
 for x in ext['assets']:
  for ca in x['walking_candidates']:
   controls_paths=ca.get('direction_controls') or [None]
   for cp in ca.get('cpu_checks',[]):
    for ctl in controls_paths:add(ca['native_motion'],cpu_path=cp,controls_path=ctl,preservation_paths=ca.get('source_preservation') or x.get('source_preservation_candidates',[]),source_label='external_derived_walking')
  for ca in x['slide_fallback_candidates']:
   for ctl in ca.get('direction_controls') or [None]:add(ca['native_motion'],controls_path=ctl,source_label='external_registered_standing_slide')
 # Old own static slide receipts are still legitimate geometry evidence. Actual
 # controls must be embedded or explicitly supplied, otherwise remain unverified.
 for path in sorted(root.glob('requests/*/*/native_motion.json')):
  n=read(path)
  if n.get('motion_mode')=='slide_no_animation':add(path,source_label='historical_owned_slide')
 ue=read(ue_path) if ue_path else {}
 rows=[]
 for a in inv:
  candidates=by[a['asset_id']];good=[x for x in candidates if x['status']=='pass'];walk=[x for x in good if x['motion_mode']=='walk_animation'];slide=[x for x in good if x['motion_mode']=='slide_no_animation'];chosen=(walk or slide or [None])[-1]
  mode=chosen['motion_mode'] if chosen else 'cannot_move'
  rejected=[x for x in candidates if x['motion_mode']=='walk_animation' and x['status']!='pass']
  reason='完整同源动作表面检查、实际原生方向/支撑/房间几何及真实反向负例通过；声音交量产' if mode=='walk_animation' else ('没有通过新标准的Walking；已登记站姿的实际平移和方向负例通过' if mode=='slide_no_animation' else '本轮房间路线的运动证据未完整通过；不表示其它房间永远不能移动')
  if mode=='slide_no_animation' and rejected:reason+='；Walking失败/未验证项：'+','.join(sorted({k for x in rejected for k in x['qualification'].get('failures',[])+x['qualification'].get('missing_evidence',[])}))
  if chosen and chosen.get('rigid_placement_proof'):reason+='；原站姿全身刚体找平，网格形状/拓扑/PBR保持，见刚体证明'
  rows.append({'asset_id':a['asset_id'],'species':a['species'],'source_has_skeleton':a['has_skeleton'],'source_has_walking':a['has_walking'],
               'motion_mode':mode,'classification':MODES[mode],'reason':reason,'selected':chosen,'candidate_evidence':candidates,
               'audio_status':'deferred_to_production','admission_scope':'motion_only','scope':'本轮指定HM3D房间/路线/短片，未验证全部房间'})
 assert len(rows)==63 and len({x['asset_id'] for x in rows})==63
 counts={s:{m:sum(x['species']==s and x['motion_mode']==m for x in rows) for m in MODES} for s in ['dog','cat']}
 assert sum(counts['dog'].values())==37 and sum(counts['cat'].values())==26
 return {'schema':'v7_animal_motion_only_final_owner093x_v1','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'validation_policy':validation_policy(),'counts':counts,'assets':rows,'external_index':hashed(external_index),'ue':ue,
         'read_errors':errors,'original_registrations_modified':False,'per_asset_audio_required':False,'audio_status':'deferred_to_production',
         'original17_counts':{s:{m:sum(x['species']==s and x['source_has_walking'] and x['motion_mode']==m for x in rows) for m in MODES} for s in ['dog','cat']},
         'boundary':'默认30秒多源视觉调度尚未消费native_frame_source；本轮显式CLI/API可产实际单源运动，motion_metadata须调用方传入题目；未声称全部房间或默认量产已接通'}

def render(result,root,filename):
 def esc(v):return html.escape(str(v))
 def pic(path,label):
  p=Path(path)
  if not p.is_file():return ''
  return '<figure><img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"><figcaption>'+esc(label)+'</figcaption></figure>'
 lead='；'.join(('狗' if s=='dog' else '猫')+'：'+str(c['walk_animation'])+' 有动画 / '+str(c['slide_no_animation'])+' 只能滑行 / '+str(c['cannot_move'])+' 本轮不能移动' for s,c in result['counts'].items())
 body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>v7 狗猫动作最终验收</title><style>body{max-width:1200px;margin:28px auto;padding:0 18px;font:15px/1.55 system-ui;color:#172434}table{width:100%;border-collapse:collapse;font-size:12px}th,td{border:1px solid #dce2e8;padding:5px;vertical-align:top;overflow-wrap:anywhere}th{background:#eef4f8}.note{background:#fff5dc;padding:12px}figure{display:inline-block;width:30%;vertical-align:top;margin:1%}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{max-width:680px}.ok{color:#126438}.fail{color:#9d241e}</style><h1>v7 狗猫动作验收（10月14日新标准）</h1><p><b>'+esc(lead)+'</b></p>',
 '<p>标准：朝向≤20°，真实180°/90°负例必须被拒；支撑爪≤12 mm、沿用原穿地容差；跨部位新增穿插面积每帧合计≤400 mm²、深度≤10 mm（同部位忽略）；每步支撑滑动≤15 mm、拉伸≤2倍。逐只RIR不再验，旧结果保留，声音由量产出题器筛选。滑行速度上限0.08 m/s且明确不声称步态。</p>',
 '<p class="note">'+esc(result['boundary'])+'。本页的表面证据包含同源平地编译的完整逐帧表面检测及本房间原生接触/方向检测；对外部8槽复用回放，没有把最后8个GLB冒充全部帧，关联范围逐项见JSON。MP3D及其它路线未验证。</p>',
 '<p>方向修复：统一动物前向轴和glTF/Habitat基变换，关闭人物默认180°翻转，移除片段水平根运动并以路线驱动；归一化原片段未发现整体倒放。猫着地修复：弃用空支撑标签的旧缓存，按原骨骼四爪支撑面重建慢步IK，锁住踝以下自身脚形。新增加的拉伸检查发现原17来源的旧Walking局部蒙皮超过2倍，不能沿用历史“全过”结论；本轮CC0派生与滑行结果以下表为准。原文件与登记均未修改。</p>',
 '<p>时间 '+esc(result['generated_utc'])+'；每行的详细检查和原始文件SHA256见 <code>motion_admissions_owner093x_final.json</code>。</p><input id="q" placeholder="筛选资产/类别/结论" style="padding:8px;width:70%"><table id="assets"><thead><tr><th>资产</th><th>类</th><th>最终结果</th><th>范围 / 原因 / 证据</th></tr></thead><tbody>']
 for x in result['assets']:
  chosen=x['selected'];short=(str(chosen['frame_count'])+'帧 / '+str(chosen['duration_s'])+'秒；') if chosen else ''
  body.append('<tr><td>'+esc(x['asset_id'])+'</td><td>'+('狗' if x['species']=='dog' else '猫')+'</td><td>'+esc(x['classification'])+'</td><td>'+esc(short+x['reason'])+'<details><summary>检查与证据</summary><pre>'+esc(json.dumps(chosen or {'candidate_evidence':x['candidate_evidence']},ensure_ascii=False,indent=2))+'</pre></details></td></tr>')
 body.append('</tbody></table><h2>实际原生截图</h2><p>绿箭头为尾→头；红箭头为实际速度。下面只用验收所需代表帧，图片已内嵌。</p>')
 selected=[]
 for species in ['dog','cat']:
  vals=[x for x in result['assets'] if x['species']==species and x['selected'] and x['motion_mode']=='walk_animation']
  if vals:selected.append(vals[0]['selected'])
 for n in selected:
  row=next(x for x in result['assets'] if x['asset_id']==n['asset_id'])
  before=next((v for v in row['candidate_evidence'] if v['source_label']=='original_rig_walking' and v.get('images')),None)
  if before:body.append(pic(before['images'][0],n['asset_id']+' 原Walking：新增表面检查未过'))
  for p in n.get('images',[])[::2][:3]:body.append(pic(p,n['asset_id']+' '+Path(p).name))
  ctrl=n.get('controls')
  if ctrl:
   for c in read(ctrl).get('controls',[]):
    if c.get('asset_id')!=n['asset_id']:continue
    for f in c.get('frames',[]):
     if f.get('image'):body.append(pic(f['image'],'故意旋转'+str(c['body_rotation_error_deg'])+'°：新门槛拒绝'))
 body.append('<h2>UE与剩余范围</h2>')
 ue=result['ue']
 if ue.get('status')=='pass':
  body.append('<p>酷家乐 kujiale_0020：新绑定比格1只，4秒/121帧按新动作标准通过；真实180°/90°负例被拒，未新增RIR。UE表面由逐帧实测骨骼矩阵和目标原网格权重在CPU重建，并非GPU顶点缓冲读回。旧10只原生几何及8只声学证据保留；另外9只的新表面检查仍未完成，不能写为新标准通过。</p>')
  for p in ue.get('images',[])[:2]:body.append(pic(p,'UE 比格实际回放 '+Path(p).name))
 body.append('<details><summary>UE逐项状态与原生证据（点击展开）</summary><pre>'+esc(json.dumps(ue,ensure_ascii=False,indent=2))+'</pre></details>')
 resources=root/'RESOURCE_USAGE.json'
 if resources.exists():
  d=read(resources);brief={k:d.get(k) for k in ['generated_utc','recorded_cpu_hours','gpu_0_related_job_wall_seconds_union','own_output_allocated_bytes','additional_bytes_since_latest_budget']}
  body.append('<p>资源为记录下界/快照；GPU数字为物理0相关作业墙钟并集，包含等待，非kernel计时。</p><pre>'+esc(json.dumps(brief,ensure_ascii=False,indent=2))+'</pre>')
 body.append('<script>document.getElementById("q").oninput=function(){let q=this.value.toLowerCase();for(let r of document.querySelectorAll("#assets tbody tr"))r.style.display=r.textContent.toLowerCase().includes(q)?"":"none"}</script></html>')
 filename.write_text(''.join(body))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--external-index',type=Path);p.add_argument('--ue-admissions',type=Path);p.add_argument('--html',action='store_true');a=p.parse_args();root=a.root
 result=collect(root,a.external_index or root/'inputs/external_motion_only_evidence_owner093x.json',a.ue_admissions)
 path=root/'motion_admissions_owner093x_final.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
 if a.html:render(result,root,root/'REPORT_MOTION_ONLY.html')
 print(json.dumps({'counts':result['counts'],'original17':result['original17_counts'],'read_errors':result['read_errors'],'output':str(path)},ensure_ascii=False))
if __name__=='__main__':main()
