"""Merge qualified original, v6, and extra animals into this task's independent roster."""
import argparse,base64,collections,copy,datetime,hashlib,html,itertools,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from v7_animal_variants_v6_candidates import sha,dump
from v7_animal_variants_register import COLORS,NAMES,verify_image
NAMES={**NAMES,'abyssinian':'阿比西尼亚猫','burmese':'缅甸猫','siamese':'暹罗猫','border_collie':'边境牧羊犬','pembroke_welsh_corgi':'彭布罗克威尔士柯基','shiba_inu':'柴犬','jack_russell_terrier':'杰克罗素梗','chihuahua':'吉娃娃','pomeranian':'博美','american_akita':'美国秋田','domestic_shorthair':'中华田园短毛猫','norwegian_forest':'挪威森林猫','cockatiel':'玄凤鹦鹉','rose_ringed_parakeet':'印度环颈鹦鹉'}
VALID={'dog':{'black','white','gray','brown','golden'},'cat':{'black','white','gray','brown','orange'},'bird':{'green','blue','yellow','white','gray','orange'}}

def find_native(T,m):
 cases={}
 for house in ['mp3d','hm3d']:
  candidates=list((T/'append_20261012/evidence/native').glob('*/'+house+'/'+m['asset_id']+'.json'))+[T/'evidence/habitat'/house/(m['asset_id']+'.json')]
  matches=[]
  for p in candidates:
   if not p.exists():continue
   x=json.loads(p.read_text())
   if x.get('runtime_sha256')==m['runtime_sha256'] and x.get('color_calibration_sha256')==sha(T/'evidence/color_calibration.json'):matches.append((p,x))
  if not matches:raise ValueError('没有当前资产的原生验收 '+m['asset_id']+' '+house)
  good=[(p,x) for p,x in matches if x['status']=='pass']
  if not good:raise ValueError('当前资产原生检查失败 '+m['asset_id']+' '+house+':'+str(matches[-1][1].get('error')))
  p,x=max(good,key=lambda pair:pair[0].stat().st_mtime);assert len(x['screenshots'])==4
  for pic in x['screenshots']:verify_image(pic)
  cases[house]={'receipt':str(p),'receipt_sha256':sha(p),**x}
 return cases

def crossing(rows):
 out={};errors=[]
 for cat in ['dog','cat','bird']:
  g=[r for r in rows if r['category']==cat];labels=sorted({r['label_color'] for r in g if r['label_color']});bodies=sorted({r['body_id'] for r in g});table={b:{l:sum(r['body_id']==b and r['label_color']==l for r in g) for l in labels} for b in bodies};untag={b:sum(r['body_id']==b and r['label_color'] is None for r in g) for b in bodies};coverage={l:sum(table[b][l]>0 for b in bodies) for l in labels};out[cat]={'individuals':len(g),'labels':labels,'body_counts':table,'untagged_by_body':untag,'different_bodies_per_label':coverage,'untagged_total':sum(untag.values()),'untagged_companions':sum(untag.values())}
  if any(n<(2 if cat=='bird' else 3) for n in coverage.values()):errors.append(cat+' 色族身体覆盖不足')
  for b in bodies:
   n=sum(x>0 for x in table[b].values())
   if n==1:errors.append(cat+' '+b+' 只有一种标签颜色')
 return out,errors

def label_size(rows):
 table=[]
 for cat in ['dog','cat','bird']:
  for label in sorted({x['label_color'] for x in rows if x['category']==cat and x['label_color']}):
   group=[x for x in rows if x['category']==cat and x['label_color']==label];key='body_height_without_perch_m' if cat=='bird' else 'shoulder_proxy_m';v=[x['size_measurements'][key] for x in group];table.append({'category':cat,'label':label,'individuals':len(v),'measurement':'bird_body_height_perch_excluded' if cat=='bird' else 'measured_torso_fur_surface_shoulder_proxy','median_m':float(np.median(v)),'minimum_m':float(min(v)),'maximum_m':float(max(v)),'small_large_overlap_claim':'observed range only; no statistical independence claimed'})
 return table

def small_large_coverage(rows):
 limits={'dog':(.30,.50),'cat':(.26,.29),'bird':(.20,.28)};out=[]
 for cat in ['dog','cat','bird']:
  for label in sorted({r['label_color'] for r in rows if r['category']==cat and r['label_color']}):
   g=[r for r in rows if r['category']==cat and r['label_color']==label];small,large=limits[cat];key='principal_body_length_m' if cat=='bird' else 'shoulder_proxy_m'
   a=sorted({r['body_id'] for r in g if r['size_measurements'][key]<=small+1e-6});b=sorted({r['body_id'] for r in g if r['size_measurements'][key]>=large-1e-6});out.append({'category':cat,'label':label,'metric':key,'small_threshold_m':small,'large_threshold_m':large,'small_bodies':a,'large_bodies':b,'status':'pass' if a and b else 'not_met','interpretation':'predeclared task-size bins for observed assets; cats representative estimate, bird length not shoulder height'})
 return out

def build(T):
 P=T/'append_20261012';base=json.loads((P/'previous_delivery/registrations/v7_asset_roster_animal_variants_20261012.json').read_text());old={r['asset_id']:r for r in base['animals']};rows=[];failures=[];rejected=[];paths=json.loads((P/'reports/all_final_metadata.json').read_text());walk_path=P/'reports/v6_walking_candidates_20261012.json';walk=json.loads(walk_path.read_text()) if walk_path.exists() else {};walk_rows={x['asset_id']:x for x in walk.get('candidates',[])}
 for path in paths:
  m=json.loads(Path(path).read_text());aid=m['asset_id']
  try:
   assert sha(m['runtime_glb'])==m['runtime_sha256'];assert m['triangles']<=100000 and Path(m['runtime_glb']).stat().st_size<=40000000;assert m['size_consistency']['status']=='pass';native=find_native(T,m);r={**copy.deepcopy(old.get(aid,{})),**m};r['asset_metadata']=path;r['native_qa']=native;r['habitat_glb']=m['runtime_glb'];r['status']='admitted';r['body_name_zh']=NAMES[m['body_id']];r['downstream_ai_training_admission']='granted';r['license']['downstream_ai_training_admission']='granted';r['walk_allowed']=bool(m.get('walk_allowed',False));r['measured_color_by_house']={h:x['pooled_body_color'] for h,x in native.items()};labels={x['dominant_label'] for x in native.values()};label=next(iter(labels)) if len(labels)==1 and all(x['pooled_body_color']['status']=='pass' for x in native.values()) else None
   if label not in VALID[m['category']]:label=None
   if m.get('cross_requirement_exempt_companion') and m['mode']!='v6_idle_static_derivative':label=None
   r['measured_native_family']={h:q['dominant_label'] for h,q in native.items()};r['label_color']=label;r['measured_color_label']=label;r['color_relabelled']=label is not None and label!=m.get('intended_color_label');r['color_label_decision']='native calibrated family and dominance passed' if label else '无标签花色：场景不一致、主色不足、受限色族或原有陪衬'
   if m['mode']=='v6_idle_static_derivative':
    r['admission_basis']='owner_authorized_all_11_v6_provenance_exception_20261012; unchanged native technical gates';r['walk_allowed']=False;r['movement']='静态 Idle 派生站姿';w=walk_rows.get(aid)
    if w:
     r['walking_attempt']=w
     if w['status']=='admitted':r['walk_allowed']=True;r['movement']='静态站姿 + 逐帧整体竖直修正 Walking 原生缓存';r['walk_delivery_form']='qualified native scene pose sequence, all 41 rigid GLBs retained per house; source skin and all clips unchanged; primary runtime.glb remains static';r['walking_frame_cache']={house:str(P/'walking_frames'/aid/'native'/house/'Walking/frames') for house in ['mp3d','hm3d']}
   elif aid not in old:
    D=Path(m.get('generation_asset_root',Path(m['size_consistency']['source_metadata']).parent));ref=json.loads((D/'reference/generation_receipt.json').read_text());seg=json.loads((D/'reference/segmentation_receipt.json').read_text());gens=[json.loads(p.read_text()) for p in (D/'generation').glob('seed_*/generation_receipt.json')];selected=[g for g in gens if g.get('raw_sha256')==m['raw_sha256']];assert len(selected)==1 and selected[0]['backend']=='Pixal3D' and seg['status']=='pass';assert sha(ref['image'])==ref['image_sha256'];assert sha(D/'reference/input_rgba_isnet.png')==seg['rgba_sha256'];r['reference_generation']=ref;r['segmentation']=seg;r['mesh_generation']=selected[0];r['mesh_attempts']=gens;r['previous_failed_color_generation']=m.get('previous_failed_color_generation');r['previous_color_generation_roots']=m.get('previous_color_generation_roots',[m['previous_failed_color_generation']] if m.get('previous_failed_color_generation') else []);r['previous_failed_color_attempts']=[json.loads(p.read_text()) for root in r['previous_color_generation_roots'] for p in Path(root).glob('generation/seed_*/generation_receipt.json')];r['model_revisions']={'FLUX':ref['revision'],'Pixal3D':selected[0]['model_revision'],'DINOv3':selected[0]['dino_revision']};r['previous_color_attempt_intermediate_sha256']=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for root in r['previous_color_generation_roots'] for p in Path(root).rglob('*') if p.is_file()];r['intermediate_sha256']=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file()];r['movement']='静态站姿/停架';r['append_new_generation']=True
   rows.append(r)
  except Exception as ex:
   detail={'asset_id':aid,'error':repr(ex),'metadata':path};rejected.append(detail)
   if aid in old or m['mode']=='v6_idle_static_derivative':failures.append(detail)
 # Tag admission is independent of static asset admission. Never weaken measured gates.
 decisions=[]
 for _ in range(len(rows)*3+1):
  cross,_=crossing(rows);changed=False
  for r in rows:
   label=r['label_color']
   if label is None:continue
   c=cross[r['category']];body=r['body_id'];reason=None
   if c['different_bodies_per_label'][label]<(2 if r['category']=='bird' else 3):reason='实测色族未覆盖足够不同身体'
   elif sum(v>0 for v in c['body_counts'][body].values())<2:reason='该身体没有两种合格真实标签色，保留无标签陪衬'
   if reason:
    r['label_color']=None;r['measured_color_label']=None;r['color_label_decision']=reason;decisions.append({'asset_id':r['asset_id'],'dropped_label':label,'reason':reason});changed=True
  if changed:continue
  conflict=None
  for cat in ['dog','cat','bird']:
   g=[r for r in rows if r['category']==cat and r['label_color']]
   for a,b in itertools.combinations(g,2):
    if a['label_color']==b['label_color']:continue
    delta=min(float(np.linalg.norm(np.array(x['pooled_body_color']['dominant_median_lab'])-np.array(y['pooled_body_color']['dominant_median_lab']))) for x,y in itertools.product(a['native_qa'].values(),b['native_qa'].values()))
    if delta<15:conflict=(a,b,delta);break
   if conflict:break
  if not conflict:break
  a,b,delta=conflict
  # Preserve original B labels when a new candidate conflicts; ties keep greater dominance.
  drop=min([a,b],key=lambda r:(int(r['asset_id'] in old),min(q['pooled_body_color']['dominant_fraction'] for q in r['native_qa'].values()),r['asset_id']))
  other=b if drop is a else a;label=drop['label_color'];reason='不同标签实测主色差不足 ΔE76=15，保留个体并撤掉标签'
  decisions.append({'asset_id':drop['asset_id'],'dropped_label':label,'reason':reason,'conflicting_asset':other['asset_id'],'minimum_delta_e76':delta});drop['label_color']=None;drop['measured_color_label']=None;drop['color_label_decision']=reason
 contrast={}
 for cat in ['dog','cat','bird']:
  pairs=[];g=[r for r in rows if r['category']==cat and r['label_color']]
  for a,b in itertools.combinations(g,2):
   if a['label_color']==b['label_color']:continue
   delta=min(float(np.linalg.norm(np.array(x['pooled_body_color']['dominant_median_lab'])-np.array(y['pooled_body_color']['dominant_median_lab']))) for x,y in itertools.product(a['native_qa'].values(),b['native_qa'].values()));pairs.append({'asset_a':a['asset_id'],'asset_b':b['asset_id'],'delta_e76':delta})
  contrast[cat]={'minimum_delta_e76':min((p['delta_e76'] for p in pairs),default=None),'pairs_below_15':[p for p in pairs if p['delta_e76']<15]}
  if contrast[cat]['pairs_below_15']:failures.append({'category':cat,'error':'tagged color contrast below unchanged 15 threshold','pairs':contrast[cat]['pairs_below_15']})
 cross,cross_errors=crossing(rows);failures.extend({'error':x} for x in cross_errors);v6=[r for r in rows if r['mode']=='v6_idle_static_derivative'];assert len(v6)==11,'all 11 mandatory standing assets must be admitted';counts=dict(collections.Counter(r['category'] for r in rows));summary={'status':'pass' if not failures else 'fail','counts':counts,'bird_families':dict(collections.Counter(r.get('sound_appearance_family') for r in rows if r['category']=='bird')),'overall_status':'pass' if not failures else 'fail','v6_standing_admitted':len(v6),'v6_walk_allowed':sum(r['walk_allowed'] for r in v6),'new_breeds':sorted({r['body_id'] for r in rows if r.get('append_new_breed')}),'cross_tables':cross,'color_contrast':contrast,'label_size_table':label_size(rows),'label_small_large_body_coverage':small_large_coverage(rows),'label_decisions':decisions,'rejected_optional_candidates':rejected,'failures':failures};dump(P/'reports/append_final_acceptance.json',summary)
 if failures:return rows,summary
 roster={**base,'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'animals':rows,'v6_idle_animals':v6,'append_new_animals':[r for r in rows if r.get('append_new_generation')],'summary':summary,'source_code':{'worktree':str(Path(__file__).resolve().parents[2]),'branch':'codex/v7-animal-variants-20261012','base_commit':'dc4588b829097933c6c95fe725f8d9577c7def5d','final_commit_receipt':str(P/'evidence/final_handoff.json')},'production_sampler_integration':False,'audio_bound':False,'owner_provenance_exception':{'applies_only_to_v6_11':True,'A_table_unchanged':True,'unverified_3d_receipt_preserved':True},'license_notice':'新生成流程 FLUX/ISNet Apache-2.0、Pixal3D MIT、DINOv3 自定义许可（任务接受）；v6 11只来源链按 owner 2026-10-12 新授权接受，缺项照留，原包许可字段不改。'};dump(T/'registrations/v7_asset_roster_animal_variants_20261012.json',roster);dump(Path(__file__).resolve().parents[2]/'registrations/v7_asset_roster_animal_variants_20261012.json',roster);return rows,summary

def h(v):return html.escape(str(v))
def picture(path):
 from PIL import Image
 import io
 buf=io.BytesIO();im=Image.open(path).convert('RGB');im.thumbnail((480,480));im.save(buf,format='JPEG',quality=78);return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+h(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+h(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
def render(T,rows,s):
 P=T/'append_20261012';parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v7 动物外观库 · 追加验收</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f5f5f2;color:#222;max-width:1400px;margin:auto;padding:24px}h1,h2{line-height:1.3}table{border-collapse:collapse;background:white;display:block;overflow:auto;margin:16px 0}td,th{padding:8px 12px;border:1px solid #ddd;text-align:left;font-size:14px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}.card{background:white;border:1px solid #ddd;border-radius:8px;padding:12px}.card img{width:100%;height:260px;object-fit:contain}.views{display:flex;gap:4px}.views img{width:49%;height:180px}code{overflow-wrap:anywhere;font-size:12px}.pass{color:#17633a}.note{background:#e8efe7;padding:14px;border-radius:8px}small{color:#555}details{margin:16px 0}</style><h1>这批动物，机器已经验过</h1>']
 parts.append('<div class="note"><b class="pass">入册 '+h(sum(s['counts'].values()))+' 只：狗 '+h(s['counts'].get('dog',0))+'、猫 '+h(s['counts'].get('cat',0))+'、鸟 '+h(s['counts'].get('bird',0))+'。v6 的 11 只全部有合格站姿。</b><br>这页按机器指标判定，不需要人看片或听声打分。原有 42 只保留，尺寸超差用自己的副本修正。'+h(s['v6_walk_allowed'])+' 只 v6 额外通过走路修正。</div>')
 parts.append('<p>标准一直没改：v6 四爪离地/入地 ≤12 mm；新狗猫 ≤15 mm；站架底座 ≤5 mm。高光占比 ≤2%；颜色主色占比 ≥55%，不同标签的实测主色 ΔE76 ≥15；同身体的前后爪间距与头部高度代理各离中位数 ≤15%。嘴/喙点跟随头部，双场景完整可见；面数 ≤100k，GLB ≤40 MB。猫的肩高是代表性尺寸估计，肩高表用毛发外表面代理，不声称精确解剖测量。</p>')
 for cat,title in [('dog','狗'),('cat','猫'),('bird','鸟')]:
  parts.append('<h2>'+title+'总览</h2><div class="grid">')
  for r in [x for x in rows if x['category']==cat]:
   qa=r['native_qa'];mp=qa['mp3d']['screenshots'];hm=qa['hm3d']['screenshots'];main=next(x for x in mp if x['name']=='side');minimum=min(q['pooled_body_color']['dominant_fraction'] for q in qa.values());label=COLORS.get(r['label_color'],r['label_color']);dimensions=' × '.join(f'{x*100:.1f}' for x in r['dimensions_m']);permission='owner 已授权复用；来源缺项保留' if r['mode']=='v6_idle_static_derivative' else ('Rocketbox MIT；Quaternius 动作 CC0；换色派生保留骨骼' if r['mode']=='rocketbox_coat_variant' else 'FLUX/ISNet Apache-2.0；Pixal MIT；DINOv3 自定义许可（任务接受）')
   parts.append('<article class="card"><b>'+h(r['body_name_zh'])+' · '+h(label)+'</b><img src="'+picture(main['rgb_png'])+'"><p>实测主色占比：'+f'{minimum:.1%}'+'<br>整体尺寸：'+dimensions+' cm<br>能走：'+('能，已有原生逐帧证据' if r['walk_allowed'] else '不能；使用静态站姿/停架')+'<br>许可：'+h(permission)+'</p><small>'+h(r['color_label_decision'])+'</small><div class="views"><img src="'+picture(mp[1]['rgb_png'])+'"><img src="'+picture(hm[1]['rgb_png'])+'"></div><small>左 MP3D，右 HM3D；各场景实际验过四视图。</small><details><summary>机器记录</summary><code>'+h(r['asset_id'])+'</code><p>最大贴地差 '+f'{max(q["maximum_support_delta_m"] for q in qa.values())*1000:.2f}'+' mm；最高高光占比 '+f'{max(pic["production_appearance"]["highlight_ratio"] for q in qa.values() for pic in q["screenshots"]):.4%}'+'。</p><code>'+h(r['asset_metadata'])+'</code></details></article>')
  parts.append('</div>')
 parts.append('<h2>v6 的 11 只，具体怎么站住的</h2>');v6=[r for r in rows if r['mode']=='v6_idle_static_derivative'];parts.append(table(['个体','Idle 帧 / 秒','俯仰 / 侧滚','局部刚体 Y 平移 cm','末端骨骼调整','缩放 XYZ','最大双场景爪差 mm','能走'],[[r['asset_id'],f'{r["idle_derivative"]["selected_frame"]} / {r["idle_derivative"]["selected_time_s"]:.6f}',f'{r["idle_derivative"]["pitch_deg"]}° / {r["idle_derivative"]["roll_deg"]}°',f'{r["idle_derivative"]["translation_m"][1]*100:.3f}',str(r['idle_derivative']['terminal_joint_edits'] or '无'),' / '.join(f'{x:.4f}' for x in r['size_consistency']['scale_xyz']),f'{max(q["maximum_support_delta_m"] for q in r["native_qa"].values())*1000:.3f}','是' if r['walk_allowed'] else '否'] for r in v6]))
 parts.append('<h2>身体 × 标签颜色</h2>')
 for cat,c in s['cross_tables'].items():parts.append('<h3>'+h({'dog':'狗','cat':'猫','bird':'鸟'}[cat])+'</h3>'+table(['身体']+[COLORS.get(x,x) for x in c['labels']]+['无标签'],[[NAMES.get(b,b)]+[c['body_counts'][b][l] for l in c['labels']]+[c['untagged_by_body'][b]] for b in c['body_counts']])+ '<p>每色不同身体数：'+h({COLORS.get(k,k):v for k,v in c['different_bodies_per_label'].items()})+'</p>')
 parts.append('<h2>同身体体型：缩放前后</h2>');parts.append('<p>狗猫用前后两组爪支撑中心的平均间距，以及头部上表面的 99.8 分位高度；v6 头部来自原骨骼语义区域，生成资产用前爪前方的上部区域，排除尾巴。鸟用身体 PCA 主轴长度和身体高度，站架不计入。先等比缩放；等比无法同时达标时，派生网格才做明确的轴向比例修正。原始包和原有产物都留着。</p>')
 parts.append(table(['个体','间距/主轴 前→后 cm','头部/身体高 前→后 cm','缩放 XYZ','修正后最大偏离'],[[r['asset_id'],f'{r["size_consistency"]["before"]["principal_body_length_m" if r["category"]=="bird" else "fore_hind_paw_span_m"]*100:.2f} → {r["size_measurements"]["principal_body_length_m" if r["category"]=="bird" else "fore_hind_paw_span_m"]*100:.2f}',f'{r["size_consistency"]["before"]["body_height_without_perch_m" if r["category"]=="bird" else "robust_top_height_m"]*100:.2f} → {r["size_measurements"]["body_height_without_perch_m" if r["category"]=="bird" else "robust_top_height_m"]*100:.2f}',' / '.join(f'{x:.4f}' for x in r['size_consistency']['scale_xyz']),f'{max(abs(v-1) for v in r["size_consistency"]["relative_after"].values()):.1%}'] for r in rows]))
 parts.append('<h2>标签颜色与体型的实际范围</h2><p>表里是这批资产的观测范围，用来检查不同颜色里都有多种体型；它不等于统计独立性证明。鸟没有可比的肩高，列停架以上的身体高度。</p>'+table(['动物','标签','数量','肩高代理中位 cm','范围 cm'],[[{'dog':'狗','cat':'猫','bird':'鸟'}[x['category']],COLORS.get(x['label'],x['label']),x['individuals'],f'{x["median_m"]*100:.2f}',f'{x["minimum_m"]*100:.2f}–{x["maximum_m"]*100:.2f}'] for x in s['label_size_table']]))
 parts.append('<h2>每个标签里的小身体和大身体</h2><p>本轮固定代理分组：狗肩高 ≤30 / ≥50 cm；猫肩高 ≤26 / ≥29 cm（代表性估计）；鸟身体主轴长度 ≤20 / ≥28 cm。它用于检查资产的体型覆盖，不是生物分类标准。</p>'+table(['动物','标签','小身体','大身体','结果'],[[x['category'],COLORS.get(x['label'],x['label']),', '.join(NAMES.get(b,b) for b in x['small_bodies']),', '.join(NAMES.get(b,b) for b in x['large_bodies']),x['status']] for x in s['label_small_large_body_coverage']]))
 A=json.loads((T/'evidence/A_provenance.json').read_text());parts.append('<h2>来源和许可，照实保留</h2><p>A_provenance.md 保持原样。v6 的参考图来自 Codex、早期链有真照片，以及 standard_sable 缺精确 3D 调用记录，都在登记里保留；这次按 owner 2026-10-12 对 11 只的明确授权复用，技术门槛相同。新生成使用 FLUX → ISNet → Pixal3D，没有训练新模型。</p>'+table(['v6 个体','参考图','3D 来源','原 A 分类'],[[x['name'],x['reference_generator'],x['generator_3d'],x['classification']] for x in A['animals']]))
 ue=P/'ue_retry/evidence/ue_smoke_complete.json';ue_data=json.loads(ue.read_text()) if ue.exists() else {'status':'not_run'};parts.append('<h2>失败、未通过和可选项</h2><p>UE 导入结果：'+h(ue_data)+'</p>')
 for name in ['candidate_failures.json','walking_status.json','walking_build_results.json']:
  p=P/'reports'/name
  if p.exists():parts.append('<details><summary>'+h(name)+'</summary><pre>'+h(p.read_text())+'</pre></details>')
 parts.append('<details><summary>标签降级记录</summary><pre>'+h(json.dumps(s['label_decisions'],ensure_ascii=False,indent=2))+'</pre></details><p>生产采样器和声音没有接入；原始包、既有名册和其他检出保持原样。</p></html>');(T/'index.html').write_text(''.join(parts))

def main():
 p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);a=p.parse_args();rows,s=build(a.task_root);print(json.dumps({k:s[k] for k in ['status','counts','v6_standing_admitted','new_breeds','failures']},ensure_ascii=False),flush=True)
 if s['status']=='pass':render(a.task_root,rows,s)
 return int(s['status']!='pass')
if __name__=='__main__':raise SystemExit(main())
