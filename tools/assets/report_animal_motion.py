"""Chinese, self-contained per-individual motion admissions and native evidence."""
import argparse,base64,collections,datetime,html,json,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;inv=json.loads((r/'diagnostics/inventory.json').read_text())['assets'];native=[];audio=[]
 for path in [*r.glob('habitat/*/native_motion.json'),*r.glob('requests/*/*/native_motion.json')]:
  d=json.loads(path.read_text());d['_path']=str(path)
  if d.get('checks',{}).get('status')=='pass' and all(f.get('complete_silhouette_in_frame') for f in d['frames']):native.append(d)
 for path in [*r.glob('habitat/*/audio.json'),*r.glob('requests/*/*/audio/audio.json')]:
  d=json.loads(path.read_text());d['_path']=str(path)
  if d.get('status')=='pass':
   ref=Path(d['motion_receipt']);d['_motion']=json.loads(ref.read_text());audio.append(d)
 rows=[]
 def identity(n):return (n['house'],n['listener_m'],[(f['time_s'],f['mouth_world_m']) for f in n['frames']])
 for x in inv:
  ns=[n for n in native if n['asset_id']==x['asset_id']];n=ns[-1] if ns else None;matches=[s for s in audio if n and identity(n)==identity(s['_motion'])]
  mode=n.get('motion_mode','walk_animation') if matches else 'cannot_move'
  reason='完整原生几何和移动RIR准入通过' if matches else ('几何通过，移动RIR未通过或未完成' if n else '本轮完整运动准入未完成；不等于已证明个体永远不能移动')
  row={'asset_id':x['asset_id'],'species':x['species'],'source_has_skeleton':x['has_skeleton'],'source_has_Walking':x['has_walking'],'motion_mode':mode,'reason':reason,'habitat_geometry_status':'pass' if n else 'not_admitted','habitat_motion':n['_path'] if n else None,'audio':matches[-1]['_path'] if matches else None,'ue_status':'not_admitted','scope':'本轮指定房间和路线；未验证的新房间/路线仍须重新准入'}
  if n:row['metrics']={k:n['checks'][k] for k in ('frame_count','max_all_four_paw_gap_m','max_stance_drift_m','max_stance_speed_m_s','min_full_skin_floor_m')};row['metrics']['max_heading_deg']=n['checks']['direction']['maximum_angle_deg'];row['metrics']['max_mouth_error_m']=n.get('max_emitter_error_m',max(f['native_emitter_readback_error_m'] for f in n['frames']))
  if matches:row['audio_reuse_identity']={'original_asset':matches[-1]['_motion']['asset_id'],'every_actual_mouth_time_position_and_listener_exactly_equal':True,'same_static_room_acoustic_geometry':True};row['audio_metrics']={k:matches[-1].get(k) for k in ('maximum_native_direction_error_deg','maximum_segment_direction_sweep_deg','maximum_abs_energy_error_db')}
  rows.append(row)
 counts={s:dict(collections.Counter(x['motion_mode'] for x in rows if x['species']==s)) for s in ('dog','cat')};result={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts':counts,'assets':rows,'registry_files_modified':False,'training_tables_exported':False}
 (r/'motion_admissions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 def esc(x):return html.escape(str(x))
 def picture(path,caption):
  p=Path(path)
  return '<figure><img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"><figcaption>'+esc(caption)+'</figcaption></figure>' if p.exists() else ''
 mode_names={'walk_animation':'有动画且完整通过','slide_no_animation':'只能滑行且完整通过','cannot_move':'本轮不能移动 / 未准入'}
 lead='；'.join(s+'：'+', '.join(mode_names[k]+str(counts[s].get(k,0)) for k in mode_names) for s in counts)
 content=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>v7 狗猫运动修复</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1380px;margin:32px auto;padding:0 20px;color:#1b2530}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border:1px solid #ccd5df;overflow-wrap:anywhere}th{background:#edf3f8}figure{display:inline-block;width:46%;vertical-align:top;margin:1%}img{max-width:100%}.note{background:#fff6df;padding:18px}code{overflow-wrap:anywhere}details{margin:12px 0}h1,h2{line-height:1.3}</style><h1>v7 狗猫运动：原生证据报告</h1><p><b>'+esc(lead)+'</b></p>',
 '<p class="note">完整通过须同时满足：逐帧头尾前向与平移速度夹角≤15°；支撑脚固定材料顶点漂移≤3 mm、滑速≤30 mm/s；四爪（含本轮8 mm摆动）距地≤12 mm；全皮肤穿地不超过原1 mm+10 µm数值余量；原始场景三角净空&gt;20 mm；NavMesh水平偏差≤2 mm；嘴部原生读回≤0.1 mm；RIR方向/分段扫角≤5°，原1 dB能量、DRR、到达时间及衰减门槛全部通过。滑行明确不作步态/锁脚动画声明。未验证不计入通过。</p>',
 '<p>原输入：37狗、26猫。实读GLB确认12狗、5猫有骨骼和Walking；25狗、21猫无骨骼。17个原Walking经过登记规范化后的方向未发现180°反向；新接口显式统一规范化矩阵，关闭人物默认180°翻转，并逐帧从实际根读回验证。独立180°/90°负例包含真实旋转的Habitat实体。</p>',
 '<p>猫接地修复：原41帧缓存有空支撑标签，并允许约8–12 mm整身穿地；回到原目标骨骼，先做四爪支撑面校正，再重建慢步支撑轨迹和IK。德牧/英短锁踝后仍有脚趾卷曲，分别抬起固定支撑点约21.2/16.1 mm；锁定踝以下的目标自身Idle脚形解决该问题。原网格拓扑、权重和登记均未修改。</p>',
 '<p>声学修复：原生持久上下文有同位置重复波动；新进程重算可复现。宽带相干交叉淡化还会压低高频能量；本轮新增按频带、仅依赖端点的相干权重，并只对仍失败的区间增加原生采样。参考RIR不参与拟合，不平均参考，不放宽判定器。</p>',
 '<p>报告时间：'+esc(result['generated_utc'])+'。这是逐房间运动准入，未生成或导出训练表。UE结果只有原生完成后才计入；当前未准入状态不能写成通过。</p><h2>逐个体记录（63只）</h2><input id="q" placeholder="按资产名、类别、结论筛选" style="width:70%;padding:10px"><table id="assets"><thead><tr><th>个体</th><th>类别</th><th>骨骼 / Walking</th><th>运动准入</th><th>原因 / 证据</th></tr></thead><tbody>']
 for x in rows:
  content.append('<tr><td>'+esc(x['asset_id'])+'</td><td>'+esc(x['species'])+'</td><td>'+esc(str(x['source_has_skeleton'])+' / '+str(x['source_has_Walking']))+'</td><td>'+esc(mode_names[x['motion_mode']])+'</td><td>'+esc(x['reason'])+'<details><summary>数值与来源</summary><pre>'+esc(json.dumps(x,ensure_ascii=False,indent=2))+'</pre></details></td></tr>')
 content.append('</tbody></table><h2>原生逐帧截图与修复前后</h2><p>绿色箭头：尾到头；红色箭头：实际平移速度。下图来自真实房间原生渲染。前后图分别为原Walking规范化姿态与本轮接触修复，未用替代模型。</p>')
 shown=set()
 for n in native:
  if n['asset_id'] in shown:continue
  shown.add(n['asset_id']);content.append('<h3>'+esc(n['asset_id'])+'</h3>')
  for path in n.get('images',[])[::2]:content.append(picture(path,'修复后：'+Path(path).name))
  for path in n.get('before_images',[])[::2]:content.append(picture(path,'原Walking（未做本轮接触修复）：'+Path(path).name))
  content.append('<details><summary>逐帧机器检查</summary><pre>'+esc(json.dumps({k:v for k,v in n['checks'].items() if k!='direction'},ensure_ascii=False,indent=2))+'</pre><code>'+esc(n['_path'])+'</code></details>')
 controls=r/'diagnostics/native_direction_controls_v1/controls.json'
 if controls.exists():
  cc=json.loads(controls.read_text());content.append('<h2>真正反向与横向的负例</h2><p>'+esc(cc['status'])+'；这些实例故意旋转，只用于证明检测器拒绝，不计准入。</p>')
  for c in cc['controls']:
   for f in c['frames']:
    if f.get('image'):content.append(picture(f['image'],c['asset_id']+' / 故意旋转 '+str(c['body_rotation_error_deg'])+'° / 检测 '+c['check']['status']))
 content.append('<h2>失败与未完成</h2>')
 for pth in sorted((r/'derived').glob('rig_*/rigging.json')):
  content.append('<pre>'+esc(pth.read_text())+'</pre>')
 failures=[x for x in rows if x['motion_mode']=='cannot_move'];content.append('<p>'+esc(str(len(failures))+'只未完成完整准入；每只原因见表。几何通过不替代声学通过，导入成功不替代UE回放。')+'</p>')
 jobs=json.loads((r/'JOBS.json').read_text());content.append('<details><summary>所有作业、命令与资源记录</summary><pre>'+esc(json.dumps(jobs,ensure_ascii=False,indent=2))+'</pre></details><script>document.getElementById("q").oninput=function(){let q=this.value.toLowerCase();for(let r of document.querySelectorAll("#assets tbody tr"))r.style.display=r.textContent.toLowerCase().includes(q)?"":"none"}</script></html>')
 (r/'REPORT.html').write_text(''.join(content));print(json.dumps({'counts':counts,'report':str(r/'REPORT.html'),'native_geometry_assets':len(shown)},ensure_ascii=False))
if __name__=='__main__':main()
