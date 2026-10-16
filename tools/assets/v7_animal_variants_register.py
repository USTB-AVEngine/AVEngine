"""Admit only measured native candidates, validate crossing, and emit a self-contained Chinese review."""
import argparse,base64,collections,datetime,hashlib,html,itertools,json,sys
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image

NAMES={'standard_poodle':'标准贵宾','labrador':'拉布拉多','newfoundland':'纽芬兰犬','miniature_schnauzer':'迷你雪纳瑞','havanese':'哈瓦那犬','beagle':'比格','german_shepherd':'德国牧羊犬','british_shorthair':'英国短毛猫','exotic_shorthair':'异国短毛猫','persian':'波斯猫','maine_coon':'缅因猫','budgerigar':'虎皮鹦鹉','fischer_lovebird':'费氏牡丹鹦鹉','border_canary':'博德金丝雀','gloster_corona':'冠羽格洛斯特金丝雀'}
COLORS={'black':'黑','white':'白','gray':'灰','brown':'棕','green':'绿','blue':'蓝','yellow':'黄','orange':'橙','golden':'金黄',None:'无标签'}
REAL={'standard_poodle':{'black','white','brown'},'labrador':{'black','brown','golden'},'newfoundland':{'black','brown'},'miniature_schnauzer':{'black','white','gray'},'havanese':{'black','white','brown','gray'},'german_shepherd':{'black','white'},'british_shorthair':{'black','white','gray'},'exotic_shorthair':{'black','white','gray'},'persian':{'black','white','gray'},'maine_coon':{'black','white','gray'},'budgerigar':{'green','blue','yellow','white','gray'},'fischer_lovebird':{'green','blue','yellow','white'},'border_canary':{'yellow','white'},'gloster_corona':{'yellow','white'}}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def img(p):return 'data:image/jpeg;base64,'+base64.b64encode(Path(p).read_bytes()).decode()
def e(v):return html.escape(str(v))
def verify_image(row):
    for k,h in [('rgb_png','rgb_sha256'),('body_mask','body_mask_sha256'),('feature_exclusion','feature_exclusion_sha256'),('original_texture_reference','original_texture_reference_sha256')]:
        if not row.get(k) or sha(row[k])!=row.get(h):raise ValueError('截图或掩膜 SHA256 不一致：'+k)

def qualify(T,item,cal_sha):
    D=T/'assets'/item['asset_id'];path=D/'asset.json'
    if not path.exists():return None,['没有完成资产元数据']
    meta=json.loads(path.read_text());problems=[];cases={};budgets={};motion=[]
    reference_path=D/'reference/generation_receipt.json'
    if not reference_path.exists():problems.append('缺少 FLUX 参考图生成记录')
    else:
        ref=json.loads(reference_path.read_text())
        if ref.get('model')!='black-forest-labs/FLUX.2-klein-4B' or ref.get('revision')!='e7b7dc27f91deacad38e78976d1f2b499d76a294' or ref.get('license') not in ['Apache-2.0','Apache-2.0; Rocketbox source texture MIT']:problems.append('参考图模型或许可没有匹配固定快照')
        image_key='image' if 'image' in ref else 'output_texture'
        hash_key='image_sha256' if 'image_sha256' in ref else 'output_texture_sha256'
        if ref.get(image_key) and sha(ref[image_key])!=ref.get(hash_key):problems.append('参考图摘要不一致')
    if item['mode']=='generated_static':
        segp=D/'reference/segmentation_receipt.json'
        if not segp.exists():problems.append('没有 ISNet 抠图记录')
        else:
            seg=json.loads(segp.read_text())
            if seg.get('weight_license')!='Apache-2.0' or seg.get('checkpoint_sha256')!='60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a' or seg.get('status')!='pass':problems.append('ISNet 模型或抠图记录未过')
            if sha(D/'reference/input_rgba_isnet.png')!=seg.get('rgba_sha256'):problems.append('ISNet RGBA 摘要不一致')
        selected=[]
        for gp in (D/'generation').glob('seed_*/generation_receipt.json'):
            gr=json.loads(gp.read_text())
            if gr.get('raw_sha256')==meta.get('raw_sha256'):selected.append((gp,gr))
        if len(selected)!=1:problems.append('未找到唯一的选中 3D 生成记录')
        else:
            gp,gr=selected[0]
            if gr.get('backend')!='Pixal3D' or gr.get('model_revision')!='0b31f9160aa400719af409098bff7936a932f726' or gr.get('dino_revision')!='3c276edd87d6f6e569ff0c4400e086807d0f3881' or sha(gr['raw_glb'])!=gr['raw_sha256']:problems.append('3D 模型快照或原始网格摘要不一致')
    if meta.get('status')=='rejected_normalization':problems.append('姿势归一化未通过')
    for part in ['runtime.glb','body_only.glb','motion_runtime.glb']:
        p=D/part
        if not p.exists():
            if part!='motion_runtime.glb':problems.append('缺少 '+part)
            continue
        scene=trimesh.load(p,force='scene',process=False);triangles=sum(len(g.faces) for g in scene.geometry.values());budgets[part]={'path':str(p),'bytes':p.stat().st_size,'triangles':triangles,'sha256':sha(p)}
        if triangles>100000 or p.stat().st_size>40000000:problems.append(part+' 超过面数或体积限制')
    if budgets.get('runtime.glb',{}).get('sha256')!=meta.get('runtime_sha256'):problems.append('运行资产摘要与元数据不一致')
    for house in ['mp3d','hm3d']:
        p=T/'evidence/habitat'/house/(item['asset_id']+'.json')
        if not p.exists():problems.append(house+' 没有原生验收');continue
        x=json.loads(p.read_text());cases[house]=x
        if x.get('status')!='pass':problems.append(house+' 原生未过：'+str(x.get('error') or x.get('pooled_body_color',{})))
        if x.get('qa_protocol') not in ['native_four_views_v6','native_four_views_v7']:problems.append(house+' 验收版本过旧')
        if x.get('runtime_sha256')!=meta.get('runtime_sha256') or x.get('color_calibration_sha256')!=cal_sha:problems.append(house+' 不是当前资产和颜色校准的验收')
        if len(x.get('screenshots',[]))!=4:problems.append(house+' 四视图不全')
        for v in x.get('screenshots',[]):
            try:verify_image(v)
            except (ValueError,FileNotFoundError) as ex:problems.append(str(ex))
        if x.get('emit_head_status')!='pass' or x.get('emit_readback_error_m',1)>1e-4 or x.get('emit_translation_and_yaw_follow_error_m',1)>1e-4:problems.append(house+' 发声点检查未过')
    companion=bool(item.get('cross_requirement_exempt_companion'));labels={x.get('dominant_label') for x in cases.values()}
    label=None if companion else (next(iter(labels)) if len(labels)==1 else None)
    if not companion and (label is None or label not in REAL.get(item['body_id'],set())):problems.append('两个场景的颜色标签不同，或实测颜色不属于此身体真实颜色')
    if not companion and any(x.get('pooled_body_color',{}).get('status')!='pass' for x in cases.values()):problems.append('颜色主色占比未过')
    if item['mode']=='rocketbox_coat_variant':
        proof_path=D/'coat_equivalence.json'
        if not proof_path.exists():problems.append('缺少换色几何/骨骼/动作等价证据')
        else:
            from v7_human_roster_materials import geometry_animation_digest,read_glb
            for proof in json.loads(proof_path.read_text()):
                part=D/('motion_runtime.glb' if proof['part']=='motion' else 'runtime.glb');doc,blob=read_glb(part)
                if not proof.get('unchanged') or geometry_animation_digest(doc,blob)!=proof['source_geometry_rig_animation_sha256']:problems.append('原骨骼/几何/全部动作摘要不一致：'+proof['part'])
        for house,family in itertools.product(['mp3d','hm3d'],['Idle','Walking']):
            p=T/'evidence/dog_motion_native'/house/(item['asset_id']+'_'+family+'.json')
            if not p.exists():problems.append('缺少 '+house+' '+family+' 逐帧验收');continue
            x=json.loads(p.read_text());motion.append({'receipt':str(p),'sha256':sha(p),'frame_count':x.get('frame_count'),'status':x.get('status'),'motion_sha256':x.get('new_motion_sha256'),'maximum_native_contact_delta_m':x.get('maximum_native_contact_delta_m'),'maximum_emitter_readback_error_m':x.get('maximum_emitter_readback_error_m')})
            if x.get('status')!='pass' or x.get('new_motion_sha256')!=budgets.get('motion_runtime.glb',{}).get('sha256'):problems.append(house+' '+family+' 逐帧验收不属于当前动作文件或未过')
    meta.update(measured_color_label=label,color_relabelled=not companion and label!=item['intended_color_label'],body_name_zh=NAMES[item['body_id']])
    return {'meta':meta,'cases':cases,'budgets':budgets,'motion':motion,'label':label},sorted(set(problems))

def cross_check(animals):
    tables={};errors=[]
    for cat,minimum in [('dog',16),('cat',12),('bird',10)]:
        rows=[x for x in animals if x['category']==cat];labels=sorted({x['label_color'] for x in rows if x['label_color']});bodies=sorted({x['body_id'] for x in rows if x['label_color']});counts={body:{label:sum(x['body_id']==body and x['label_color']==label for x in rows) for label in labels} for body in bodies};coverage={label:sum(counts[body][label]>0 for body in bodies) for label in labels};tables[cat]={'individuals':len(rows),'labels':labels,'body_counts':counts,'different_bodies_per_label':coverage,'untagged_companions':sum(x['label_color'] is None for x in rows)}
        if len(rows)<minimum:errors.append(f'{cat} 只有 {len(rows)}，下限 {minimum}')
        if any(v<(2 if cat=='bird' else 3) for v in coverage.values()):errors.append(cat+' 某些标签没有覆盖足够不同身体')
        if any(sum(v>0 for v in counts[body].values())<2 for body in bodies):errors.append(cat+' 某些身体不足两种标签颜色')
    birds=[x for x in animals if x['category']=='bird'];sub=collections.Counter(x['sound_appearance_family'] for x in birds)
    if sub['parrot']<6 or sub['canary']<4:errors.append('鹦鹉类或金丝雀类数量不足')
    return tables,dict(sub),errors

def pairwise(animals,threshold):
    results={};errors=[]
    for cat in ['dog','cat','bird']:
        rows=[x for x in animals if x['category']==cat and x['label_color']];pairs=[]
        for a,b in itertools.combinations(rows,2):
            if a['label_color']==b['label_color']:continue
            candidates=[(float(np.linalg.norm(np.array(a['native_qa'][ha]['pooled_body_color']['dominant_median_lab'])-np.array(b['native_qa'][hb]['pooled_body_color']['dominant_median_lab']))),ha,hb) for ha,hb in itertools.product(['mp3d','hm3d'],repeat=2)];distance,ha,hb=min(candidates);pairs.append({'first':a['asset_id'],'second':b['asset_id'],'first_label':a['label_color'],'second_label':b['label_color'],'first_house':ha,'second_house':hb,'delta_e76':distance})
        minimum=min((x['delta_e76'] for x in pairs),default=None);bad=[x for x in pairs if x['delta_e76']<threshold];results[cat]={'status':'pass' if not bad and pairs else 'fail','minimum_delta_e76':minimum,'threshold':threshold,'different_label_pairs':len(pairs),'failed_pairs':bad,'closest_pairs':sorted(pairs,key=lambda x:x['delta_e76'])[:10]}
        if bad:errors.append(cat+' 不同标签主色的最小色差不足 '+str(threshold))
    return results,errors

def make_html(T,roster,A,cal):
    animals=roster['animals'];summary=roster['summary'];parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v7 动物外观库验收</title><style>body{margin:auto;max-width:1440px;padding:24px;font-family:system-ui;background:#f5f6f8;color:#222}h1{font-size:26px}h2{margin-top:32px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}.card{background:white;padding:14px;border-radius:12px;box-shadow:0 2px 7px #0001}img{width:100%;border-radius:8px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}small{color:#555;word-break:break-all}table{border-collapse:collapse;background:white;width:100%;margin:12px 0}th,td{border:1px solid #ddd;padding:9px;text-align:left}code{word-break:break-all}.ok{color:#146b35}.bad{color:#a32424}details{background:white;padding:12px;border-radius:8px;margin:10px 0}</style><body>']
    title='通过全部数量和交叉要求' if summary['overall_status']=='pass' else '尚未满足整批验收标准'
    parts+=[f'<h1>v7 动物外观库：{title}</h1>',f'<p class="{ "ok" if summary["overall_status"]=="pass" else "bad"}">这次实际入册：狗 {summary["counts"]["dog"]} 只、猫 {summary["counts"]["cat"]} 只、鸟 {summary["counts"]["bird"]} 只。鹦鹉类 {summary["bird_families"].get("parrot",0)} 只，金丝雀类 {summary["bird_families"].get("canary",0)} 只。只做资产和机器验收，还没有接生产采样器，也没有配声音。</p>', '<p>入册标准：每只在 MP3D 和 HM3D 都有四个完整可见视角；犬猫四脚支撑误差 ≤15 毫米，鸟站架底座 ≤5 毫米，底面落地误差 &lt;0.01 毫米；高光比 ≤2%；头部发声点随平移/转向的误差 &lt;0.1 毫米；面数 ≤100k、每个 GLB ≤40 MB。换色犬还要逐帧检查站/走的四脚接触及嘴部节点。没有要求人看片或听声来做判定。</p>',f'<p>颜色来自原生图像里的身体掩膜，先排除小面积眼、鼻、喙，再按 CIELAB D65 分色族。主色占比 ≥{cal["thresholds"]["dominant_fraction_min"]:.0%}；黑色 L*≤{cal["thresholds"]["black_l_max"]:g}，白色 L*≥{cal["thresholds"]["white_l_min"]:g}，中间低彩度为灰；不同标签任意个体、任意场景的主色中位数 ΔE76 ≥{cal["thresholds"]["min_pair_delta_e76"]:g}。人物上衣校准 {cal["groups"]["calibration"]["correct"]}/{cal["groups"]["calibration"]["count"]}，保留身份验证 {cal["groups"]["held_out"]["correct"]}/{cal["groups"]["held_out"]["count"]}。</p>', '<p>生成链路是 FLUX.2 klein → ISNet → Pixal3D；必要的纹理修整仍用 FLUX 参考引导，保留所有几何、骨骼和动作，最多原始参考图加两次 UV 种子。没有用 Hunyuan、Codex 画图或新的 TokenRig。FLUX/ISNet/NAF 为 Apache-2.0，Pixal3D 为 MIT；DINOv3 是单独的 DINOv3 License，按任务书允许输出用于训练，不能写成 MIT。换色犬使用 Rocketbox MIT 和 Quaternius CC0 动作。</p>', '<p>尺寸说明：犬的尺寸按公开品种范围取代表值；猫的厘米数是明确标注的代表性估值，未验证为品种官方肩高；金丝雀资料的头顶–尾尖与这里的喙尖–尾尖并非同一种测量。几何缩放和测量本身都有机器记录。</p>']
    if summary['unmet_requirements']:parts.append('<p class="bad">还没过：'+e('；'.join(summary['unmet_requirements']))+'</p>')
    for cat,name in [('dog','狗'),('cat','猫'),('bird','鸟')]:
        group=[a for a in animals if a['category']==cat];parts.append('<h2>'+name+'的总览</h2><div class="grid">')
        for a in group:
            views=a['native_qa'];fraction=min(v['pooled_body_color']['dominant_fraction'] for v in views.values());physical=a['physical_size'];size=(f'{physical.get("target_m",0)*100:.1f} 厘米 '+('喙到尾' if cat=='bird' else '肩高') if physical.get('target_m') else '继承已验收原模型尺寸');label=COLORS[a['label_color']];v=views['mp3d']['screenshots'][1];parts.append(f'<article class="card"><img src="{img(v["image"])}" alt="{e(a["body_name_zh"])}"><b>{e(a["body_name_zh"])} · {label}</b><p>实测主色占比最低 {fraction:.1%} · {size}<br>{"能走、能站" if a["walk_allowed"] else "静态"} · 许可已通过</p><small>{e(a["asset_id"])}</small>'+('<p>花色陪衬：没有标签，不用于颜色指代。</p>' if a['label_color'] is None else '')+('<p>按实测改标签，原意图为 '+COLORS[a['intended_color_label']]+'。</p>' if a['color_relabelled'] else '')+'</article>')
        parts.append('</div>');table=summary['cross_tables'][cat];labels=table['labels'];parts.append('<h3>标签颜色 × 身体（数字是入册个体数）</h3><table><tr><th>身体</th>'+''.join('<th>'+COLORS[c]+'</th>' for c in labels)+'</tr>')
        for body,counts in table['body_counts'].items():parts.append('<tr><td>'+NAMES[body]+'</td>'+''.join('<td>'+str(counts[c])+'</td>' for c in labels)+'</tr>')
        parts.append('<tr><th>覆盖的不同身体数</th>'+''.join('<th>'+str(table['different_bodies_per_label'][c])+'</th>' for c in labels)+'</tr></table>');parts.append('<h3>放进 MP3D / HM3D 房间里的样子</h3><div class="grid">')
        for a in group:
            parts.append('<article class="card"><b>'+e(a['body_name_zh'])+' · '+COLORS[a['label_color']]+'</b><div class="pair">'+''.join('<div><p>'+h.upper()+'</p><img src="'+img(a['native_qa'][h]['screenshots'][0]['image'])+'" alt="原生场景"></div>' for h in ['mp3d','hm3d'])+'</div><small>'+e(a['asset_id'])+'</small></article>')
        parts.append('</div>')
    parts.append('<h2>A：v6 的 11 只动物来源</h2><p>7 只链路许可干净，2 只用了 Codex 自带画图，2 只证据不足。这只是来源盘点，不等于已经通过 v7 视觉验收；原包里的 project_owner_review_required 没有改。</p><table><tr><th>动物 / 标识</th><th>参考图</th><th>3D / 骨骼</th><th>结论</th></tr>')
    for a in A['animals']:parts.append('<tr><td>'+e(a['name'])+'<br><small>'+e(a['asset_id'])+'</small></td><td>'+e(a['reference_generator'])+'</td><td>'+e(a['generator_3d'])+' / '+e(a['rig'])+'</td><td>'+e(a['classification'])+'</td></tr>')
    parts.append('</table><p>逐项许可、lineage 路径与 SHA256 见 A_provenance.md 和 evidence/A_provenance.json。</p><h2>失败和没有做完的项目</h2>')
    for r in roster['rejected_candidates']:parts.append('<details><summary>'+e(r['asset_id'])+'：未入册</summary><p>'+e('；'.join(r['reasons']))+'</p></details>')
    parts.append('<details><summary>生成和检查中失败过的尝试</summary><pre>'+e(json.dumps(roster['attempt_failures'],ensure_ascii=False,indent=2))+'</pre></details>')
    C_status=T/'reports/C_status.json'
    if C_status.exists():
        cs=json.loads(C_status.read_text());parts.append('<h3>C：许可干净的 v6 动物独立候选</h3><p>实际检查 '+str(cs['clean_v6_candidates'])+' 只，入册 '+str(cs['admitted'])+' 只，剔除 '+str(cs['rejected'])+' 只；原生回放 '+str(cs['native_cases'])+' 段、'+str(cs['native_frames'])+' 个源帧。原始包没有改，源几何、骨骼和所有动作保持不变。保留失败结果，许可干净不替代姿势验收。</p>')
        cr=json.loads(Path(cs['candidate_roster']).read_text())
        parts.append('<table><tr><th>v6 个体</th><th>候选结论</th><th>实测接触最大误差</th></tr>')
        for candidate in cr['candidates']:
            deltas=[x['maximum_contact_delta_m'] for x in candidate['native_cases'] if x.get('maximum_contact_delta_m') is not None]
            parts.append('<tr><td><small>'+e(candidate['asset_id'])+'</small></td><td>'+('通过' if candidate['status']=='admitted' else '未入册')+'</td><td>'+('%.1f 毫米'%(max(deltas)*1000) if deltas else '未验证')+'</td></tr>')
        parts.append('</table>')
    parts.append('<p>UE 抽检准备了狗、猫、鸟各一个；编辑器在导入开始前 SIGSEGV（返回码 -11），实际完成导入 0 个，未验证 UE 可导入。这项失败不挡 Habitat 主线。C 的 v6 独立候选验收状态见 reports/C_status.json，原始包只读。</p><p>完整登记：registrations/v7_asset_roster_animal_variants_20261012.json。模型 revision、提示词、种子、中间文件和图片摘要都已保留；作业记录在 JOBS.json。提交、资源和测试的最终记录见 evidence/final_handoff.json。</p></body></html>');(T/'index.html').write_text(''.join(parts))

def main():
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--worktree',type=Path,required=True);p.add_argument('--draft',action='store_true');a=p.parse_args();T=a.task_root;plan=json.loads((T/'plan.json').read_text());calp=T/'evidence/color_calibration.json';cal=json.loads(calp.read_text());assert cal['status']=='pass';cal_sha=sha(calp);dependency=json.loads((T/'evidence/dependency_licenses.json').read_text());assert all(x['downstream_output_training_permitted_under_task'] for x in dependency['models']);animals=[];rejected=[];attempts=[]
    for item in plan['animals']:
        q,problems=qualify(T,item,cal_sha)
        if problems:rejected.append({'asset_id':item['asset_id'],'category':item['category'],'body_id':item['body_id'],'intended_color_label':item['intended_color_label'],'reasons':problems,'downstream_ai_training_admission':'not_granted'});continue
        meta=q['meta'];meta['status']='pending_batch_acceptance';meta['license']['downstream_ai_training_admission']='pending_batch_acceptance'
        animal={**meta,'label_color':q['label'],'habitat_glb':meta['runtime_glb'],'asset_metadata':str(T/'assets'/item['asset_id']/'asset.json'),'movement':'Idle, Walking' if meta['walk_allowed'] else 'static standing/perched','native_qa':q['cases'],'native_qa_receipts':{h:{'path':str(T/'evidence/habitat'/h/(item['asset_id']+'.json')),'sha256':sha(T/'evidence/habitat'/h/(item['asset_id']+'.json'))} for h in q['cases']},'file_budgets':q['budgets'],'dog_motion_qa':q['motion'],'downstream_ai_training_admission':'granted','audio_binding':'not_configured','production_sampler_integration':'not_performed','model_revisions':{k:plan[k] for k in ['reference_revision','mesh_revision','dino_revision']},'dependency_license_evidence':{'path':str(T/'evidence/dependency_licenses.json'),'sha256':sha(T/'evidence/dependency_licenses.json')},'intermediate_sha256_ledger':str(T/'assets'/item['asset_id']/'sha256_ledger.json')};animals.append(animal)
    tables,families,errors=cross_check(animals);distances,contrast_errors=pairwise(animals,cal['thresholds']['min_pair_delta_e76']);errors+=contrast_errors
    # An individual cannot be marked admitted if a contrast gate is unresolved.
    if contrast_errors and not a.draft:raise RuntimeError('Different-label pairwise color gate failed; repair or explicitly reject candidates before registration:'+str(distances))
    for p in (T/'evidence').glob('normalization*results*.json'):
        data=json.loads(p.read_text());rows=data if isinstance(data,list) else data.get('results',[])
        attempts.extend({'source':str(p),'asset_id':r.get('asset_id'),'status':r.get('status'),'reason':r.get('error')} for r in rows if r.get('status','').startswith(('reject','fail')))
    catalog=T/'evidence/attempt_failure_catalog.json'
    if catalog.exists():attempts=json.loads(catalog.read_text())
    counts=dict(collections.Counter(x['category'] for x in animals));counts={c:counts.get(c,0) for c in ['dog','cat','bird']};summary={'overall_status':'pass' if not errors else 'incomplete','counts':counts,'bird_families':families,'cross_tables':tables,'pairwise_color_distance':distances,'unmet_requirements':errors,'planned_candidates':len(plan['animals']),'admitted_candidates':len(animals),'rejected_candidates':len(rejected),'full_native_motion_frames':sum(r['frame_count'] for x in animals for r in x['dog_motion_qa'])}
    roster={'schema':'avengine_v7_animal_variants_20261012_v1','artifact_root':str(T),'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'humans':[],'animals':animals,'chairs':[],'summary':summary,'rejected_candidates':rejected,'attempt_failures':attempts,'integration_owner':'independent candidate registration; no existing roster changed','audio_admission':'not_configured','geometry_qa_scope':'native MP3D/HM3D, machine color/material/grounding/head emitter; current-sha frame replay for walking variants','license_notice':'DINOv3 custom license accepted by supplied task; not described as MIT; original v6 packages untouched','source_code':{'base_commit':'455875b1d644ca861a8337aaa487cca6b30aeb6a','branch':'codex/v7-animal-variants-20261012','worktree':str(a.worktree),'final_commit_receipt':str(T/'evidence/final_code_commit.json')},'color_calibration':{'path':str(calp),'sha256':cal_sha,'groups':cal['groups'],'thresholds':cal['thresholds']}}
    for animal in animals:
        D=T/'assets'/animal['asset_id']
        meta=json.loads((D/'asset.json').read_text());meta['status']='admitted';meta['license']['downstream_ai_training_admission']='granted';meta['measured_color_label']=animal['label_color'];meta['color_relabelled']=animal['color_relabelled'];meta['body_name_zh']=animal['body_name_zh'];dump(D/'asset.json',meta);animal['status']='admitted';animal['license']['downstream_ai_training_admission']='granted'
        entries=[{'relative_path':str(p.relative_to(D)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file() and p.name!='sha256_ledger.json'];dump(D/'sha256_ledger.json',{'asset_id':animal['asset_id'],'all_task_owned_intermediates':entries});animal['intermediate_sha256_ledger_sha256']=sha(D/'sha256_ledger.json')
    filename='v7_asset_roster_animal_variants_20261012.json';dump(T/'registrations'/filename,roster);dump(a.worktree/'registrations'/filename,roster);dump(T/'evidence/final_acceptance_summary.json',summary);make_html(T,roster,json.loads((T/'evidence/A_provenance.json').read_text()),cal);print(json.dumps(summary,ensure_ascii=False,indent=2));return 0 if not errors else 1
if __name__=='__main__':raise SystemExit(main())
