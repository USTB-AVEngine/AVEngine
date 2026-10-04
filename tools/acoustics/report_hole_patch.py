#!/usr/bin/env python3
"""Build a Chinese report and one-page offline review from native trial evidence."""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import csv
from datetime import datetime, timezone
import html
import io
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO)]
import numpy as np
from avengine.acoustics.hole_patch import write_json
from tools.acoustics.run_hole_patch import assert_frozen


def read(p):return json.loads(Path(p).read_text())
def distribution(values):
    v=np.asarray([x for x in values if x is not None and np.isfinite(x)],float)
    if not len(v):return {'n':0,'min':None,'p05':None,'median':None,'p95':None,'max':None,'mean':None}
    return dict(n=len(v),min=float(v.min()),p05=float(np.quantile(v,.05)),median=float(np.median(v)),
                p95=float(np.quantile(v,.95)),max=float(v.max()),mean=float(v.mean()))

def leak_label(value):
    return '未测' if value is None else f'{100*value:.5f}%'


def paired_changes(base,local):
    by={x['probe']:x for x in base['rirs']};rt=[];drr=[];foa=[];attenuation=[]
    for r in local['rirs']:
        old=by.get(r['probe'])
        if not old:continue
        if old.get('status')=='valid' and r.get('status')=='valid' and old.get('rt60_mid_s') is not None and r.get('rt60_mid_s') is not None:
            rt.append(r['rt60_mid_s']-old['rt60_mid_s'])
        if old.get('drr_db') is not None and r.get('drr_db') is not None:drr.append(r['drr_db']-old['drr_db'])
    by={x['probe']:x for x in base.get('foa_checks',[])}
    for r in local.get('foa_checks',[]):
        old=by.get(r['probe'])
        if old and old['check']['angle_deg'] is not None and r['check']['angle_deg'] is not None:foa.append(r['check']['angle_deg']-old['check']['angle_deg'])
    by={x['room_label']:x for x in base.get('distance_attenuation',[])}
    for r in local.get('distance_attenuation',[]):
        old=by.get(r['room_label'])
        if old and old['status']=='valid' and r['status']=='valid':attenuation.append(r['pressure_exponent']-old['pressure_exponent'])
    return {'rt60_s':rt,'drr_db':drr,'foa_deg':foa,'attenuation_exponent':attenuation}


def leakage_changes(base,local):
    if not base.get('leakage') or not local.get('leakage'):
        return {'status':'not_measured','checks':[],'total_rays':0,'before_escape_count':None,
                'after_escape_count':None,'closed_escaped_rays':None,'new_escaped_rays':None}
    by={(r['room_label'],r['role']):r for r in base['leakage']};checks=[]
    for after in local['leakage']:
        before=by[(after['room_label'],after['role'])]
        a={r['direction_index'] for r in before['escaped_rays']}
        b={r['direction_index'] for r in after['escaped_rays']}
        checks.append({'room_label':after['room_label'],'role':after['role'],'ray_count':after['ray_count'],
                       'before_escape_count':before['escape_count'],'after_escape_count':after['escape_count'],
                       'removed_escape_direction_indices':sorted(a-b),'new_escape_direction_indices':sorted(b-a)})
    return {'checks':checks,'total_rays':sum(r['ray_count'] for r in checks),
            'before_escape_count':sum(r['before_escape_count'] for r in checks),
            'after_escape_count':sum(r['after_escape_count'] for r in checks),
            'closed_escaped_rays':sum(len(r['removed_escape_direction_indices']) for r in checks),
            'new_escaped_rays':sum(len(r['new_escape_direction_indices']) for r in checks)}


def draw(record,house_input,context_rooms=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Polygon as Patch
    from shapely.geometry import shape
    fig,ax=plt.subplots(figsize=(6.4,4.0))
    for r in (context_rooms if context_rooms is not None else house_input['acoustic_rooms']):
        value=r.get('floor_polygon')
        if value:
            g=shape(value)
            polys=list(g.geoms) if g.geom_type=='MultiPolygon' else [g]
            for poly in polys:
                xy=np.asarray(poly.exterior.coords);ax.fill(xy[:,0],xy[:,1],color='#e8eef2',edgecolor='#64748b',linewidth=.45)
            camera=(r.get('placement') or {}).get('camera_m')
            if camera is not None:
                p=np.asarray(camera);ax.scatter(p[0],p[2],s=12,color='#0f172a');xz=[p[0],p[2]]
            else:
                center=g.representative_point();xz=[center.x,center.y]
            ax.text(xz[0],xz[1],r['room_label'],fontsize=7)
    for cluster in record['exit_clusters']:
        ids=cluster['ray_ids'];step=max(1,int(np.ceil(len(ids)/25)))
        for i in ids[::step]:
            ray=record['escaped_rays'][i];a=np.asarray(ray['origin_m']);b=np.asarray(ray['aabb_exit_m'])
            ax.plot([a[0],b[0]],[a[2],b[2]],color='#e24d48',alpha=.13,linewidth=.6)
        p=np.asarray(cluster['centroid_m']);ax.scatter(p[0],p[2],s=10,color='#b91c1c',marker='x')
    for c in record['candidates']:
        q=np.asarray(c['quad_m'])[:,[0,2]]
        accepted=c['status']=='accepted';color='#0284c7' if accepted else '#c29c38'
        ax.add_patch(Patch(q,closed=True,facecolor=color if accepted else 'none',edgecolor=color,
                          linewidth=1.2,alpha=.65 if accepted else .5))
        if accepted:
            p=q.mean(0);ax.text(p[0],p[1],c['candidate_id'],color=color,fontsize=6)
    for q in record['shell_trial']['candidate_quads']:
        xy=np.asarray(q['quad_m'])[:,[0,2]]
        ax.plot(xy[:,0],xy[:,1],color='#8b5cf6',alpha=.3,linestyle='--',linewidth=.6)
    ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('World X / m',fontsize=8);ax.set_ylabel('World Z / m',fontsize=8)
    ax.set_title(record['house'],fontsize=8);ax.tick_params(labelsize=7);ax.grid(alpha=.12)
    fig.text(.05,.015,'Blue: applied patch | ochre: review candidate | red: escaped ray | purple: withheld shell',fontsize=7)
    fig.tight_layout(rect=(0,.04,1,1));b=io.BytesIO();fig.savefig(b,format='png',dpi=110);plt.close(fig)
    return b.getvalue()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output
    assert_frozen(root)
    request=root/'evidence/diagnostic_reference_requested.json'
    if request.exists():
        ref=read(request);deadline=time.monotonic()+12000
        while not Path(ref['complete_path']).exists() and not Path(ref['error_path']).exists():
            if time.monotonic()>deadline:raise RuntimeError('Supplementary diagnostic reference did not finish; leave primary errors intact')
            time.sleep(5)
    selection=read(root/'inputs/selection.json');originals={r['house']:r for r in read(root/'inputs/all_house_results.json')}
    ids=selection['blocked']+selection['controls'];control=set(selection['controls']);pilot=set(selection['pilot'])
    # All available original registry footprints provide whole-house plan context.
    # This presentation data never participates in patch admission decisions.
    room_context={}
    for original_registry,snapshot in selection['registry_snapshots'].items():
        with Path(snapshot).open() as stream:
            for line in stream:
                if not line.strip():continue
                r=json.loads(line);h=r.get('house')
                if h in ids and originals[h]['input']['registry_source']==original_registry:
                    metrics=r.get('metrics') or {}
                    if metrics.get('floor_polygon'):
                        room_context.setdefault(h,[]).append({'room_label':r['room_label'],
                            'floor_polygon':metrics['floor_polygon'],'placement':metrics.get('placement') or {}})

    (root/'records').mkdir(exist_ok=False);(root/'images').mkdir(exist_ok=False)
    rows=[];ctrl={k:[] for k in ('rt60_s','drr_db','foa_deg','attenuation_exponent')};seal={k:[] for k in ctrl};no_op={k:[] for k in ctrl}
    ctrl_pairs=[];seal_pairs=[];veto=Counter();patches=Counter();consistency=Counter();missing=[];gallery=[]
    candidate_checks=Counter();candidate_semantics=Counter();control_transitions=Counter()
    for h in ids:
        stage='pilot' if h in pilot else 'batch';dest=root/stage/'houses'/h
        if not (dest/'complete.json').exists():missing.append(h);continue
        complete=read(dest/'complete.json');record=read(dest/'patch_record.json')
        base=read(complete['runs']['baseline']['result_path']);local=read(complete['runs']['local']['result_path']);old=originals[h]
        changes=paired_changes(base,local);leak_changes=leakage_changes(base,local);room_summary_path=root/'room_verification'/h/'complete.json'
        rv=read(room_summary_path) if room_summary_path.exists() else None
        errors={method:run.get('exception_traceback') for method,run in (('baseline',base),('local',local)) if run.get('exception_traceback')}
        primary_complete=not errors and bool(base.get('leakage') and local.get('leakage'))
        diagnostic_path=root/'diagnostic_reference'/h/'complete.json'
        diagnostic=read(diagnostic_path) if diagnostic_path.exists() else None
        actual_room_rescue=sum(x['rescued'] for x in rv['rooms']) if rv else 0
        acoustic_house_rescue=h not in control and primary_complete and record['patch_count']>0 and local['status']=='pass' and bool(rv and rv['all_local_rooms_pass'])
        sample_house_rescue=h not in control and primary_complete and record['patch_count']>0 and local['status']=='pass'
        visual_status=(old.get('visual') or {}).get('status','not_run')
        row={'house':h,'family':old['family'],'group':'control' if h in control else 'blocked','stage':stage,
             'candidate_rooms':old['input']['candidate_room_count'],'original_sampled_rooms':len(old['input']['acoustic_rooms']),
             'primary_native_measurement_complete':primary_complete,'native_upload_error_count':len(errors),
             'diagnostic_reference_path':str(diagnostic_path) if diagnostic else None,'diagnostic_reference_production_qualified':False if diagnostic else None,
             'patch_count':record['patch_count'],'patch_area_m2':record['patch_area_m2'],'candidate_patch_count':len(record['candidates']),
             'before_status':base['status'],'after_status':local['status'],'before_leak':base.get('max_escape_fraction'),'after_leak':local.get('max_escape_fraction'),
             'before_escaped_ray_count':leak_changes['before_escape_count'],'after_escaped_ray_count':leak_changes['after_escape_count'],
             'closed_escaped_rays':leak_changes['closed_escaped_rays'],'new_escaped_rays':leak_changes['new_escaped_rays'],'leakage_ray_count':leak_changes['total_rays'],
             'before_rt60_median_s':base.get('rt60_mid_median_s'),'after_rt60_median_s':local.get('rt60_mid_median_s'),
             'sample_house_acoustic_rescued':sample_house_rescue,'all_rooms_acoustic_rescued':acoustic_house_rescue,
             'independently_rescued_rooms':actual_room_rescue,'verified_candidate_rooms':rv['room_count'] if rv else 0,
             'previous_visual_status':visual_status,'holdout_reserved':old['input'].get('holdout_reserved',False),
             'can_enter_training_by_all_current_checks':acoustic_house_rescue and local.get('batch_eligible',False) and visual_status=='pass' and not old['input'].get('holdout_reserved',False),
             'patch_record_path':str(dest/'patch_record.json'),'baseline_result_path':complete['runs']['baseline']['result_path'],
             'local_result_path':complete['runs']['local']['result_path'],'room_verification_path':str(room_summary_path) if rv else None,
             'shell_status':record['shell_trial']['status'],'shell_applied_patches':0}
        rows.append(row)
        if h in control:
            control_transitions[base['status']+' → '+local['status']]+=1
            for k in ctrl:ctrl[k].extend(changes[k])
            ctrl_pairs.append({'house':h,'changes':changes})
        if record['patch_count']:
            for k in seal:seal[k].extend(changes[k])
            seal_pairs.append({'house':h,'changes':changes,'leakage_changes':leak_changes})
        else:
            for k in no_op:no_op[k].extend(changes[k])
        for c in record['candidates']:
            candidate_checks['candidate_count']+=1
            candidate_semantics[c['semantic_label']]+=1
            if 'direct_consistency' in c:
                candidate_checks['direct_checked_candidates']+=1
                candidate_checks['direct_pairs_checked']+=c['direct_consistency']['checked_pairs']
                candidate_checks['candidate_direct_obstructions']+=c['direct_consistency']['blocked_pair_count']
            if 'navigation_consistency' in c:
                candidate_checks['navigation_checked_candidates']+=1
                candidate_checks['candidate_navigation_intersections']+=c['navigation_consistency']['intersected_native_triangle_prisms']
            if 'continuous_opening_cone_check' in c:
                candidate_checks['opening_cone_evaluated_candidates']+=1
                candidate_checks['candidate_opening_cone_overlaps']+=c['continuous_opening_cone_check']['overlap_count']
            if c.get('opening_visibility',{}).get('status') not in (None,'not_run'):
                candidate_checks['additional_raw_visibility_checked_candidates']+=1
            if c['status']=='accepted':
                patches[c['semantic_label']]+=1
                consistency['accepted_patch_count']+=1
                consistency['direct_pairs_checked']+=c['direct_consistency']['checked_pairs']
                consistency['opening_rays_checked']+=c['opening_visibility']['tested_opening_rays']
                consistency['accepted_direct_obstructions']+=c['direct_consistency']['blocked_pair_count']
                consistency['accepted_navigation_intersections']+=c['navigation_consistency']['intersected_native_triangle_prisms']
                consistency['accepted_visible_opening_obstructions']+=c['opening_visibility']['visible_opening_violation_count']
                consistency['accepted_opening_cone_overlaps']+=c['continuous_opening_cone_check']['overlap_count']
            for reason in c['reasons']:veto[reason]+=1
        if record.get('consistency_error'):consistency['house_consistency_errors']+=1
        exported={'patch_record':record,'house_summary':row,'metric_changes':changes,'leakage_changes':leak_changes,'primary_native_errors':errors,'diagnostic_reference':diagnostic}
        write_json(root/'records'/(h+'.json'),exported)
        png=draw(record,old['input'],room_context.get(h));(root/'images'/(h+'.png')).write_bytes(png)
        meta=f"{record['patch_count']} 块 / {record['patch_area_m2']:.3f} m²；候选 {len(record['candidates'])}；外壳待审"
        if errors:meta+='；严格原生装载失败，诊断仅参考'
        leak=leak_label(base.get('max_escape_fraction'))+' → '+leak_label(local.get('max_escape_fraction'))
        data_uri='data:application/json;base64,'+base64.b64encode(json.dumps(exported,ensure_ascii=False).encode()).decode()
        gallery.append(f'<tr data-family="{old["family"]}" data-id="{html.escape(h)}"><td><b>{html.escape(h)}</b><br>{row["group"]}<br>{html.escape(meta)}<br>漏声 {leak}<br>原生 {base["status"]} → {local["status"]}<br>独立救回 {actual_room_rescue} 间<br><a download="{h}.json" href="{data_uri}">下载补洞记录</a></td><td><img alt="{html.escape(h)} top-down" src="data:image/png;base64,{base64.b64encode(png).decode()}"></td></tr>')
    family_summary={}
    for f in ('hm3d','mp3d'):
        group=[r for r in rows if r['family']==f and r['group']=='blocked']
        family_summary[f]={'blocked_houses':len(group),'candidate_rooms':sum(r['candidate_rooms'] for r in group),
            'original_sampled_rooms':sum(r['original_sampled_rooms'] for r in group),
            'patched_houses':sum(r['patch_count']>0 for r in group),'patch_count':sum(r['patch_count'] for r in group),
            'patch_area_m2':sum(r['patch_area_m2'] for r in group),
            'rescued_houses_all_rooms':sum(r['all_rooms_acoustic_rescued'] for r in group),
            'rescued_houses_original_sample_scope':sum(r['sample_house_acoustic_rescued'] for r in group),
            'rescued_rooms_independent':sum(r['independently_rescued_rooms'] for r in group),
            'verified_rooms':sum(r['verified_candidate_rooms'] for r in group),
            'training_eligible_houses_preserving_prior_visual_and_holdout':sum(r['can_enter_training_by_all_current_checks'] for r in group)}
    summary={'generated_at':datetime.now(timezone.utc).isoformat(),'family_summary':family_summary,'completed_houses':len(rows),'missing_houses':missing,
             'frozen_at':read(root/'rules.frozen.json')['frozen_at'],'control_houses':len([r for r in rows if r['group']=='control']),
             'control_patch_count':sum(r['patch_count'] for r in rows if r['group']=='control'),
             'control_status_transitions':dict(control_transitions),
             'candidate_checks':dict(candidate_checks),'candidate_semantics':dict(candidate_semantics),
             'no_op_houses':sum(r['patch_count']==0 for r in rows),
             'primary_native_completed_houses':sum(r['primary_native_measurement_complete'] for r in rows),
             'primary_native_failed_houses':[r['house'] for r in rows if not r['primary_native_measurement_complete']],
             'supplementary_diagnostic_houses':[r['house'] for r in rows if r['diagnostic_reference_path']],
             'equal_baseline_local_leak_houses':sum(r['primary_native_measurement_complete'] and r['before_leak']==r['after_leak'] for r in rows),
             'original_sample_scope_native_measurement_attempts':2*len(rows),'original_sample_scope_native_measurements':2*sum(r['primary_native_measurement_complete'] for r in rows),
             'control_change_distributions':{k:distribution(v) for k,v in ctrl.items()},
             'patched_house_change_distributions':{k:distribution(v) for k,v in seal.items()},
             'no_op_noise_distributions':{k:distribution(v) for k,v in no_op.items()},
             'patch_material_counts':dict(patches),'candidate_veto_reasons':dict(veto),'consistency':dict(consistency),
             'shells_admitted':0,'native_full_shell_measurements':0,'shells_withheld':len(rows),
             'qualification':'research acoustic preprocessing pilot; original visual and semantic assets unchanged',
             'room_count_definition':'same original criteria per individual candidate room, eight RIRs and two listener positions; all-room house rescue requires every candidate room to pass'}
    write_json(root/'summary.json',summary);write_json(root/'all_house_results.json',rows)
    write_json(root/'control_changes.json',{'pairs':ctrl_pairs,'distributions':summary['control_change_distributions']})
    write_json(root/'sealed_rt60_changes.json',{'pairs':seal_pairs,'distributions':summary['patched_house_change_distributions']})
    if missing:raise ValueError('Report incomplete: '+', '.join(missing))
    with (root/'house_results.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    hm,mp=family_summary['hm3d'],family_summary['mp3d']
    conclusion=f"结论：按冻结规则及原声学验收门槛，HM3D 被卡房屋声学救回 {hm['rescued_houses_all_rooms']} 套、{hm['rescued_rooms_independent']} 间；MP3D 声学救回 {mp['rescued_houses_all_rooms']} 套、{mp['rescued_rooms_independent']} 间。"
    report=[conclusion,'',f"机器 48g；基线 `{selection['base_commit']}`；独立检出 `{REPO}`，分支 `claude/v7-acoustic-hole-patch-20261004`。产物 `{root}`。未 push，未发送 owner。当前提交见 `evidence/git_commit.txt`。",'',
        '## 实测范围与数字出处','',
        '输入以 `inputs/all_house_results.json` 的固定快照为准，原始路径及 SHA 在 `inputs/selection.json`。79 套漏声被卡房屋与 10 套合格对照均经过同一流水线。5 套试点先完成，规则在 `rules.frozen.json` 所列 UTC 时间冻结；批次开始后未按结果调整规则。',
        '房屋级原抽样口径保留其原先至多 3 间；有实际补片的房屋额外逐间测全部规则候选房，单间仍采用原验收代码的一房模式（8 条 RIR、2 个听者、3 个 FOA 位置）。第一句的“套”要求整屋原抽样及所有候选房声学均通过；“间”只数独立前后重测由非 pass 变成 pass 的房间。未逐间测到的候选房不会计入救回。既有视觉待审与 holdout 预留继续保留，不等同已具备训练生产资格。', '',
        '|家族|被卡套|规则候选间|原抽样间|实际补片套 / 块|逐间验证间|原抽样声学救回套|全候选声学救回套|独立救回间|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for f in ('hm3d','mp3d'):
        x=family_summary[f];report.append(f"|{f.upper()}|{x['blocked_houses']}|{x['candidate_rooms']}|{x['original_sampled_rooms']}|{x['patched_houses']} / {x['patch_count']}|{x['verified_rooms']}|{x['rescued_houses_original_sample_scope']}|{x['rescued_houses_all_rooms']}|{x['rescued_rooms_independent']}|")
    report += ['', '## 候选放行与待审原因', '',
        f"检测到 {candidate_checks['candidate_count']} 个候选；{candidate_checks['direct_checked_candidates']} 个进入原摆位直达线和原导航可走体检查。候选类别及检查计数见 `summary.json/candidate_semantics` 与 `summary.json/candidate_checks`。",
        f"新版本中 {summary['no_op_houses']} 套为 no-op；{summary['equal_baseline_local_leak_houses']} 套的漏声比例与同轮旧包完全相等。几何不变的副本不能产生补洞救回，独立 Monte Carlo 的声学状态变化不会计作救回。待审候选表示本轮规则没有得到足够可信的扫描缺失证据，不能推广为扫描破洞无法修复。", '',
        '|候选待审原因（同一候选可有多条）|计数|','|---|---:|']
    for reason,count in veto.most_common():report.append(f'|{reason}|{count}|')
    report += ['', '数字出处：`summary.json/family_summary`、`house_results.csv`、`all_house_results.json`；每项原生测量都给出独立 `baseline_result_path` / `local_result_path`，每间补充验证见 `room_verification/<house>/complete.json`。', '',
        '## 实际补片的漏声变化','',
        '射线按相同 room_label / role / direction_index 配对；以下总数只用于说明局部几何变化，整屋漏声门槛仍取原代码各测点 escape_fraction 的最大值。',
        '|房屋|测量射线总数|漏出条数前 → 后|封住原漏出射线|新增漏出射线|最高漏声前 → 后|',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['patch_count']:
            report.append(f"|{r['house']}|{r['leakage_ray_count']}|{r['before_escaped_ray_count']} → {r['after_escaped_ray_count']}|{r['closed_escaped_rays']}|{r['new_escaped_rays']}|{100*r['before_leak']:.5f}% → {100*r['after_leak']:.5f}%|")
    report += ['', '出处：每套 records/<house>.json/leakage_changes 与 sealed_rt60_changes.json/pairs；保留封住及新增的全部方向身份。原抽样房间与当前规则候选房可能不是同一组，口径在房屋表及 input/candidate_room_labels 中分别保留，不将换房抽样算作补洞改善。', '',
        '## 冻结补洞规则与材质','',
        '1. 漏出射线按原方向与身份保留；在扫描整屋 AABB 的同一面上以 0.75 m 邻接聚类。AABB 穿出点仅用作定位，不能直接当真实洞口。',
        '2. 用编译后的语义表面网格检测边界：仅检测时以 10 µm 焊接类别接缝，原顶点不改；闭环要求平面残差 ≤3 cm、洞面积 0.02–20 m²、无表面面心落在环内，边界至少 95% 为结构语义。门窗及无法确定语义的环待审；大于 0.2 m² 的竖直墙开口一律待审。',
        '3. 断裂的水平边界另由漏出射线投影聚类：平面必须来自同一房间既有 floor/ceiling 语义三角形，绝不只依赖房间包围盒。要求平面外围八处样本至少六处得到邻近结构面支撑，样本须留在原地面多边形的 0.20 m 范围内；证据不足不补。全部常数及实现身份见 `rules.frozen.json` 与源码。',
        '4. 平面四边形覆盖候选范围，留 5 cm 边缘余量，并向房间外偏移 3 cm。整块四边形与原导航三角形的 1.5 m 高、0.10 m 半径可走体连续裁剪；同时检查原摆位算法全部 128/256 上限候选点之间的直达线超集。',
        '5. 门窗语义几何先分为连通体保守包围盒。从所有相机候选位置做连续视锥与补片裁剪；任何可能遮住门窗通透区域的视锥交叠都待审。额外 CPU 原始 GLB 射线网格检查只在没有连续视锥疑点时运行；它保留全部原始三角形，不创建 GPU 渲染器。无补片时几何不变，一致性由 no-op 保证，不伪称运行了逐相机检查。',
        '6. 材质继承环或支撑平面的相邻语义类别，直接复用旧声学包的原 material_id、类别、映射记录和系数，不重新随机材质。窗户只能使用原映射中的 glass；当前窗户缺口均须先通过真实开口/视线检查，不能因玻璃会反射就跳过一致性要求。每块记录保存实际 material_category_record / material_mapping_record。',
        '7. 仅另存声学版本。实际补片版只追加 4 顶点、2 三角形/块；旧顶点、旧面、旧材质 ID 逐项相等，并经原包契约和 RLR 原生装载。无补片版是新目录中的完整只读输入副本，标记 operation=no_op。原 GLB、原语义、navmesh、旧声学包、train-houses 检出和产物均未写入。', '',
        '## 外包壳方法的结果','',
        '每套生成“扫描 AABB 外扩 3 cm”的六面封闭壳候选并记录面积和位置。只要仍有不能确认为扫描缺失的漏出射线，或不能证明外门/窗口及立面材质、视线被保持，整个壳拒绝放行。全部壳的否决证据见每套 `patch_record.json/shell_trial`。',
        f"本轮 {len(rows)} 套壳均未放行，实际应用 0 块；完整外包壳的 RLR 传播 **未运行**。因此这里比较的是局部规则的实测结果与整壳方法的安全资格，不虚构整壳 RT60/救回数据。缺少可靠的外部开口、立面及材质证书时，不能满足‘真实开口不封’而继续生成闭壳声学版本。", '',
        '## 原生装载失败与诊断参考','',
        f"严格原包/派生包双测完成 {summary['primary_native_completed_houses']} 套；原生失败房屋为 {summary['primary_native_failed_houses']}。失败不会伪记为漏声 0%，表中显示“未测”，也不计入救回或成对声学变化。",
        'mp3d_gTV8FGcVJC9 的旧包有两张零面积面，原数据汇总本身来自 diagnostic_without_zero_surface_faces 的非生产诊断。当前严格原包与原样 no-op 派生副本均保留旧面，RLR 装载失败；原报错保存在对应 baseline/local/result.json。未删除原始扫描、旧包或其三角形。',
        '原验收工具以原有 --omit-degenerate-for-diagnostics 另跑参考，仅在独立上下文忽略严格零面积面，所有判据与 e3dc246 首到达窗不变。诊断自身仍保留 DEGENERATE_INPUT_GEOMETRY_DIAGNOSTIC_ONLY 和 production_qualified=false；不会覆盖严格失败结果，也不是合格补洞包。',
        '|房屋|诊断原包 → 新包|诊断最高漏声前 → 后|忽略面数|生产资格|',
        '|---|---|---:|---:|---|']
    for r in rows:
        if r['diagnostic_reference_path']:
            ref=read(r['diagnostic_reference_path']);b=read(ref['runs']['baseline']['result_path']);n=read(ref['runs']['local']['result_path'])
            report.append(f"|{r['house']}|{b['status']} → {n['status']}|{leak_label(b.get('max_escape_fraction'))} → {leak_label(n.get('max_escape_fraction'))}|{b['geometry_diagnostic']['omitted_faces']} / {n['geometry_diagnostic']['omitted_faces']}|false|")
    report += ['', '出处：evidence/zero_area_original_context.json、diagnostic_reference/<house>/complete.json 与逐方法原生 result.json。诊断统计单独记账，不混入严格对照或补洞 RT60 分布。', '',
        '## 对照组的变化分布','',
        f"10 套名单在试点前固定（`inputs/selection.json/controls`），实际补片 {summary['control_patch_count']} 块。每套均独立跑旧包基线和新版本，保留新 RIR。Δ = 新版本 − 同轮原包；原 8 线程 Monte Carlo 设置保持，因此 no-op 的数值波动表示独立传播重复性，不能当成补洞效果。", '',
        '|指标 / 单位|有效成对项|最小|P05|中位|P95|最大|', '|---|---:|---:|---:|---:|---:|---:|']
    for k,d in summary['control_change_distributions'].items():
        vals=[f'{d[x]:.8g}' if d[x] is not None else '未测到' for x in ('min','p05','median','p95','max')]
        report.append('|'+k+'|'+str(d['n'])+'|'+'|'.join(vals)+'|')
    report += ['', f"同轮对照状态前后计数：`{dict(control_transitions)}`。该计数是独立传播的重复性结果；原始对照入组资格来自冻结输入，不用本轮随机状态重新选择名单。"]
    report += ['', '出处：`control_changes.json` 保存逐套逐项差值，`summary.json/control_change_distributions` 保存分位统计；RT60 仅比较两侧均 valid 的同一 RIR，DRR/FOA/衰减分别按 probe 或 room_label 配对。', '',
        '## RT60 与一致性','',
        f"实际补片材质块数：`{dict(patches)}`；实测面积共 {sum(r['patch_area_m2'] for r in rows):.6f} m²。封洞相关变化仅汇总真实应用补片的房屋，数据在 `sealed_rt60_changes.json`；无补片房屋的变化另列 `summary.json/no_op_noise_distributions`。",]
    if not seal_pairs:report.append('本轮没有实际补片被放行，因此封洞导致的 RT60 延长 **未验证**，不能用 no-op 随机传播波动代替封洞效应。真实房屋的追加补片编译与原生装载分支未运行；本轮实测覆盖 no-op 新包，追加几何只经过单元回归。')
    else:
        d=summary['patched_house_change_distributions']['rt60_s'];report.append(f"真实补片房屋的成对有效 RT60：n={d['n']}，中位变化 {d['median']} s，P05/P95={d['p05']}/{d['p95']} s。此值仍含独立 Monte Carlo 波动；对照与 no-op 分布给出重复性参照，未声称恢复了实测真实房间 RT60。")
    report += ['', f"所有候选的一致性检查统计：`{dict(candidate_checks)}`；被拒绝候选保留命中的直达线、导航三角形和开口视锥示例。字段出现表示检查入口已到达；有更早否决时，后续视线步骤为 not_run，不能将其写成通过。"]
    report += ['', f"已放行补片的一致性计数：`{dict(consistency)}`。字段出处为各套候选的 direct_consistency、navigation_consistency、opening_visibility、continuous_opening_cone_check。没有放行的候选仍保留否决记录，不能把其待审写成通过。", '',
        '原声学验收文件及判据未改：漏声 ≤5% / 5–15% 待审 / >15% fail，FOA ≤15°，衰减指数 0.7–1.3，≥6 RIR、≥2 听者、直达偏差 ≤3 ms；原 RT60 有效比例、范围、尾部及其他门禁也完整重跑。直达度量使用基线中与 e3dc246 相同的首到达 +0.5 ms 窗，没有换回 ±2 ms 峰值窗。', '',
        '## 每套记录','', '|房屋|组|块 / 面积 m²|漏声前 → 后|原生前 → 后|独立救回间|', '|---|---|---:|---:|---|---:|']
    for r in rows:
        report.append(f"|{r['house']}|{r['group']}|{r['patch_count']} / {r['patch_area_m2']:.4f}|{leak_label(r['before_leak'])} → {leak_label(r['after_leak'])}|{r['before_status']} → {r['after_status']}|{r['independently_rescued_rooms']}|")
    test_receipt=read(root/'evidence/test_execution.json')
    test_tail=Path(test_receipt['log']).read_text().splitlines()
    test_summary=next((line for line in reversed(test_tail) if ' passed' in line),'见测试日志')
    report += ['', '完整 fast_unit：`'+test_summary+'`，退出码 '+str(test_receipt['returncode'])+'；出处 `evidence/test_execution.json` 与 `logs/tests_fast_unit.log`。']
    report += ['', '自包含逐套 JSON：`records/<house>.json`。原记录：`pilot/houses/<house>/patch_record.json` 或 `batch/houses/<house>/patch_record.json`。这些文件保留全部漏出射线、聚类、边界统计、候选四边形、面积、材质、对应射线身份以及否决原因。', '',
        '离线一页：`review_offline.html`，PNG 与每套可下载 JSON 均内嵌，不需要网络；俯视图展示登记快照中全部可用房间的原始地面多边形与世界坐标几何，每聚类最多画 25 条射线，完整射线保留在内嵌 JSON。蓝色表示实际补片，土黄表示待审，紫色表示未放行外壳。', '',
        '## 运行、验证与未完成事项','',
        '命令入口：`hole_patch_trial.py prepare/allocate`、`run_hole_patch.py evaluate/freeze/batch`、`verify_hole_patch_rooms.py`、`report_hole_patch.py`。具体脚本、日志和 PID 位于 scripts/、logs/、evidence/ 与逐套 *_process.json；CPU 分配取当时 /proc/stat 一秒采样，初段 2 个 worker，内存事件后续跑 1 个；再次采样空闲资源并在完整房屋边界重启后恢复 2 个 worker；每个限制在 8 个逻辑核，RLR 保留原 numThreads=8 参数（库可另建后台线程），BLAS/OMP 1 线程，CUDA_VISIBLE_DEVICES 为空。',
        '共享服务器出现 MemAvailable 约 296 MiB、swap 2 GiB 已满（scripts/kernel_resource.sh 当次输出），本任务按记录核对 PID/命令后停止自己的调度与工作进程；未处理其他人的进程。两套未完成产物完整保留在 resource_interrupted_20261004，搬移对照表为 evidence/resource_resume.json。内存恢复后，以 1 worker、8 核亲和和 nice=10 续跑，并用独立内存监控只停止本任务 PID。随后再次确认空闲核与内存充足，暂停本任务调度，等待当前房屋原包/新包均完整结束后，仅终止自己的旧调度，保留全部已完成房屋结果，恢复为 2 个独立 8 核 CPU worker；一秒空闲核采样仍留至少半数空闲核。单 worker 阶段的驱动与监控日志完整保留在 resource_rescale_20261004，房屋输出仍在原目录；记录见 evidence/controlled_cpu_rescale_20261004.json 与 evidence/cpu_allocation_rescale.json。原规则、材料和验收判据不变；中断的 RIR 不混入正式配对统计。实际事件、启动身份与结束状态见 evidence/owned_stop_memory_pressure_20261004.log、evidence/resource_resume.json、evidence/batch_memory_guard.json 及 logs/batch_resume_driver.log。',
        '试点初次序列化失败产物保留在 pilot_attempt_v0；修复后重新完成五套试点才冻结。专项与完整 fast_unit 实际命令、退出码、计数见 logs/tests_*；原验收源码身份和最终作业状态见 evidence/final_validation.json。',
        '未完成/未验证：完整外包壳传播因真实开口证据不足未运行；mp3d_gTV8FGcVJC9 的严格几何包无法原生装载，追加诊断不赋予生产资格；本轮未替换视觉资产，正式 Habitat RGB 未重渲；真实建筑的外部开口类别、实际材料与实测 RT60 真值未验证；未应用补片的房屋未做本轮全部候选房逐间传播，已有原抽样结果与另一 train-houses 的进行中逐间结果保持独立。没有清理别人进程、缓存或改共享环境。',
        '若需要更高覆盖，缺的是逐洞开口类型/外立面语义与边界证据；应新建后续规则版本再重做试点，不能在本轮冻结规则下按救回数调参。']
    (root/'REPORT_zh.md').write_text('\n'.join(report)+'\n')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>声学补洞试点</title><style>body{font:15px system-ui,sans-serif;background:#f5f7fa;color:#172033;margin:24px auto;max-width:1180px;padding:0 18px}h1{font-size:24px}p{line-height:1.7}input,select{padding:8px;margin:4px}table{width:100%;border-collapse:collapse;background:white}td{border-bottom:1px solid #ddd;padding:14px;vertical-align:top}td:first-child{width:35%;word-break:break-word;line-height:1.9}img{width:100%;height:auto;max-width:704px}a{color:#0369a1}.legend{background:#e8eef7;padding:12px;border-radius:8px}</style><h1>声学网格补洞试点</h1><p>'''+html.escape(conclusion)+'''</p><p class="legend">画面及原始扫描保持原样。蓝色：实际补片；土黄：待审候选；红色：漏出射线；紫色虚线：未放行整屋外壳。俯视图为真实世界坐标几何，非生产 RGB。外包壳传播未运行。每组射线最多显示 25 条，完整记录可下载。</p><label>筛选 <select id="family"><option value="">全部</option><option>hm3d</option><option>mp3d</option></select></label><input id="search" placeholder="房屋 ID" aria-label="房屋 ID"><table><tbody>'''+''.join(gallery)+'''</tbody></table><script>function filter(){let f=document.getElementById('family').value,q=document.getElementById('search').value.toLowerCase();document.querySelectorAll('tr[data-id]').forEach(r=>r.hidden=!!((f&&r.dataset.family!==f)||!r.dataset.id.toLowerCase().includes(q)))}document.getElementById('family').onchange=filter;document.getElementById('search').oninput=filter;</script></html>'''
    (root/'review_offline.html').write_text(page)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
