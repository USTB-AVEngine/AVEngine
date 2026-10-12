#!/usr/bin/env python3
"""Compile per-individual native admission and one self-contained Chinese HTML."""
from __future__ import annotations
import argparse,base64,collections,datetime,hashlib,html,io,json,os,stat
from pathlib import Path
from PIL import Image
from avengine.qa.kujiale_native_depth import REVISION

def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    temporary.replace(path)

def thumbnail(path,width=480):
    p=Path(path)
    if not p.is_file():return ""
    with Image.open(p) as im:
        im=im.convert("RGB");im.thumbnail((width,int(width*9/16)))
        buf=io.BytesIO();im.save(buf,format="JPEG",quality=85)
    return "data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()

def appearance(view):
    """Display a crop of the unchanged native frame; metrics use full pixels."""
    if not view.get("mask") or not Path(view["mask"]).is_file():return thumbnail(view["rgb"])
    with Image.open(view["mask"]) as mask:box=mask.getbbox()
    if box is None:return thumbnail(view["rgb"])
    with Image.open(view["rgb"]) as source:
        source=source.convert("RGB");pad=max(8,int(max(box[2]-box[0],box[3]-box[1])*.08))
        box=(max(0,box[0]-pad),max(0,box[1]-pad),min(source.width,box[2]+pad),min(source.height,box[3]+pad))
        source=source.crop(box);source.thumbnail((480,320));buf=io.BytesIO();source.save(buf,format="JPEG",quality=90)
    return "data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()

def motion_metrics(value):
    frames=value.get("frames",[])
    soles=[p for f in frames for p in f.get("sole_depth",{}).get("four_paws",[]) if p.get("floor_gap_m") is not None]
    support=[min(abs(p["floor_gap_m"]) for p in f.get("sole_depth",{}).get("four_paws",[]) if p.get("floor_gap_m") is not None) for f in frames if len(f.get("sole_depth",{}).get("four_paws",[]))==4 and any(p.get("floor_gap_m") is not None for p in f["sole_depth"]["four_paws"])]
    return {"maximum_body_axis_tilt_deg":max((f.get("body_axis_tilt_deg",0) for f in frames),default=None),
            "maximum_support_sole_gap_m":max(support,default=None),
            "minimum_all_paw_gap_m":min((p["floor_gap_m"] for p in soles),default=None),
            "maximum_declared_contact_gap_m":max((abs(p["floor_gap_m"]) for p in soles if p.get("expected_in_contact")),default=None),
            "maximum_native_bone_error_m":max((f.get("bone_readback_max_error_m",0) for f in frames),default=None),
            "maximum_mouth_emitter_error_m":max((f.get("mouth_emitter_error_m",0) for f in frames),default=None)}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def gpu_usage(jobs):
    per={}
    for gpu in [2,3]:
        spans=[]
        for job in jobs:
            if job.get("gpu_physical")!=gpu or not job.get("finished_utc"):continue
            start=datetime.datetime.fromisoformat(job["started_utc"]).timestamp()
            end=datetime.datetime.fromisoformat(job["finished_utc"]).timestamp()
            spans.append((start,end))
        merged=[]
        for start,end in sorted(spans):
            if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
            else:merged.append([start,end])
        per[str(gpu)]={"reserved_job_union_seconds":sum(b-a for a,b in merged),
                       "intervals":merged,"definition":"own tracked job wall intervals, merged per physical GPU; not measured CUDA/kernel time"}
    return per

def compile_report(root):
    manifest=read(root/"inputs/assets_final.json")
    assets=[]
    for source in manifest["assets"]:
        aid=source["asset_id"];imported=read(root/"evidence/import/final"/(aid+".json"),{})
        candidates={}
        for house in ["kujiale_0003","kujiale_0009","kujiale_0020"]:
            p=root/"evidence/native/final"/house/(aid+".json")
            native=read(p)
            if native and native.get("measurement_revision")==REVISION:
                native["evidence_path"]=str(p);candidates[house]=native
        passed={k:v for k,v in candidates.items() if v["status"]=="pass_static_native"}
        best=next(iter(passed.values()),next(iter(candidates.values()),{}))
        motion={}
        if source["walk_allowed"]:
            for family in ["Idle","Walking"]:
                entries=[]
                for house in ["kujiale_0003","kujiale_0009","kujiale_0020"]:
                    p=root/"evidence/native_motion"/house/(aid+"_"+family+".json")
                    value=read(p)
                    if value and value.get("measurement_revision")==REVISION:
                        value["evidence_path"]=str(p);entries.append(value)
                motion[family]=next((x for x in entries if x["status"]=="pass_native_motion"),entries[0] if entries else {"status":"not_run"})
        visual=bool(passed)
        motion_ok=not source["walk_allowed"] or all(v.get("status")=="pass_native_motion" for v in motion.values())
        qualified=visual and motion_ok
        views=[x for x in best.get("views",[]) if x.get("status")=="pass"][:3]
        if not views:views=best.get("views",[])[:3]
        placement=best.get("placement",{})
        color=best.get("color_tag_status","not_run")
        decision="pass" if qualified and color in ["match","unlabelled_companion"] else ("native_pass_color_owner_decision" if qualified else ("pending" if not candidates else "fail"))
        cause=[]
        if not visual:cause.append(best.get("error","三视角原生检查尚未完成"))
        if not motion_ok:cause.append("可走狗原生逐帧检查未通过或未完成")
        row={k:source.get(k) for k in ["asset_id","category","body_id","object_type","form_factor","attachment_surface","dimensions_m","emitter_local_m","silent_decor","label_color","walk_allowed","source_walk_allowed","source_walk_delivery_form","source_walking_frame_cache","source_registration","source_glb","bake_receipt"]}
        row.update(admission_decision=decision,ue_native_physical_status="pass" if qualified else "fail_or_pending",
                   ue_import_status=imported.get("status","not_run"),
                   ue_project=str(root/"ue_v1/runtime/SpearSim/SpearSim.uproject"),
                   ue_asset_paths=[x["path"] for x in imported.get("static",[]) if x["class"]=="StaticMesh"],
                   ue_skeletal_assets=imported.get("motion",[]),
                   native_house_admissions=passed,
                   native_attempt_evidence={h:v.get("evidence_path") for h,v in candidates.items()},
                   primary_native_evidence=best.get("evidence_path"),
                   source_sha256=source["bake_receipt"]["source_sha256"],
                   checks={"three_complete_views":{"status":"pass" if visual else "fail_or_pending","passed_views":len([x for x in best.get("views",[]) if x.get("status")=="pass"]),"evidence":views},
                           "attachment":{"status":placement.get("status","not_run"),"surface":source["attachment_surface"],"deltas_m":placement.get("contact_deltas_m"),"tolerance_m":placement.get("contact_tolerance_m")},
                           "furniture_and_wall_clearance":placement.get("collision",{"status":"not_run"}),
                           "known_bad_buried_collision":placement.get("known_bad_3cm_buried_collision",{"status":"not_run"}),
                           "static_four_paws":placement.get("paw_contacts",{"status":"not_run"}),
                           "highlight":{"status":"pass" if visual else "fail_or_pending","maximum_selected_ratio":max((x.get("highlight_ratio",0) or 0 for x in views),default=None),"limit":.02},
                           "emitter":best.get("emitter",{"status":"not_run"}),
                           "size":best.get("size",{"status":"not_run"}),
                           "native_skeletal_motion":{k:dict({z:v.get(z) for z in ["status","frame_count","failed_frames","evidence_path","route_mode","scope","error","native_skin_floor_calibration","rig_calibration"]},native_metrics=motion_metrics(v),preview_rgb=next((f.get("rgb") for f in v.get("frames",[]) if f.get("rgb")),None)) for k,v in motion.items()},
                           "color":{"status":color,"registered":source["label_color"],"native_measurement":best.get("measured_color"),"registry_labels_modified":False}},
                   failure_reasons=cause,
                   repair="依据失败视角/接触/动作证据修派生资产或选择原生几何允许的摆放；保持原门槛，颜色差异由 owner 决定。" if cause else None)
        assets.append(row)
    physical=sum(x["ue_native_physical_status"]=="pass" for x in assets)
    imported_count=sum(x["ue_import_status"]=="pass" for x in assets)
    visual_count=sum(x["checks"]["three_complete_views"]["status"]=="pass" for x in assets)
    colors=[x for x in assets if x["checks"]["color"]["status"]=="mismatch_owner_decision"]
    pending=[x for x in assets if x["admission_decision"] in ["fail","pending"]]
    trials=[v for h in ["kujiale_0003","kujiale_0009","kujiale_0020"] for v in read(root/"evidence"/("trial_"+h+".json"),[])]
    trial_pass=[x for x in trials if x["status"]=="pass"]
    jobs=read(root/"JOBS.json",[])
    usage=gpu_usage(jobs)
    sources=[{"path":p,"sha256":sha(p)} for p in manifest["registration_paths"]]
    total_bytes=0
    for directory,_,names in os.walk(root,followlinks=False):
        for name in names:
            try:
                value=(Path(directory)/name).lstat()
                if stat.S_ISREG(value.st_mode):total_bytes+=value.st_size
            except FileNotFoundError:continue
    floor=read(root/"evidence/floor_native_reproduction.json",{})
    stats={"total_individuals":len(assets),"categories":dict(collections.Counter(x["category"] for x in assets)),
           "native_import_pass":imported_count,"native_static_visual_pass":visual_count,
           "native_physical_pass_including_required_motion":physical,"native_failed_or_pending":len(pending),
           "color_mismatch_owner_decisions":len(colors),"native_color_match":sum(x["checks"]["color"]["status"]=="match" for x in assets),"native_color_unlabelled_companions":sum(x["checks"]["color"]["status"]=="unlabelled_companion" for x in assets),"native_motion_clips_passed":sum(v["status"]=="pass_native_motion" for x in assets for v in x["checks"]["native_skeletal_motion"].values()),"native_motion_frames_passed":sum((v.get("frame_count") or 0) for x in assets for v in x["checks"]["native_skeletal_motion"].values() if v["status"]=="pass_native_motion"),"walking_dogs":sum(x["walk_allowed"] for x in assets),
           "animal_body_count":len({x["body_id"] for x in assets if x["category"]!="object"}),
           "object_kind_count":len({x["object_type"] for x in assets if x["category"]=="object"}),
           "trial_pass_shots":len(trial_pass),"trial_pass_houses":sorted({x["house"] for x in trial_pass}),
           "trial_covered_keys":sorted({x["coverage_key"] for x in trial_pass}),
           "full_admission_color_agree_or_unlabelled":sum(x["admission_decision"]=="pass" for x in assets),
           "source_integrity":read(root/"evidence/final_source_integrity.json",{}).get("all_sources_unchanged"),
           "floor_fix_native_status":floor.get("status"),"floor_fix_native_cases":len(floor.get("cases",[])),
           "own_bytes_before_report":total_bytes,"gpu_usage":usage}
    test_path=root/"evidence/unit_fixture_restored_final_summary.json"
    if not test_path.exists():test_path=root/"evidence/final_unit_test_summary.json"
    result={"schema":"v7_kujiale_native_asset_admission_20261012_v3","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "measurement_revision":REVISION,"code_identity":read(root/"evidence/final_commit.json",{}),"crash_root_evidence":str(root/"evidence/crash_root_summary.json"),"unit_test_evidence":str(test_path),"native_production_binding_evidence":str(root/"evidence/native_production_binding_smoke.json"),"status":"complete_native_physical" if physical==len(assets) else "incomplete_native_physical",
            "source_registrations":sources,"source_registrations_modified":False,"source_glbs_modified":False,
            "labels_modified":False,"native_runtime":{"uproject":str(root/"ue_v1/runtime/SpearSim/SpearSim.uproject"),"output_root":str(root),"ue_environment_file":str(root/"ue_env.json"),"renderer":"native UE/SPEAR Vulkan offscreen","owned_editor_gpu_wrappers":{"2":str(root/"scripts/ue_gpu2"),"3":str(root/"scripts/ue_gpu3")}},"summary":stats,"approved_native_ue_references":read(root/"evidence/approved_ue_color_references.json",[]),
            "scope":"Native individual admission and 3-house trial only; no questions, production rerun, ACP or CCI.",
            "assets":assets}
    path=root/"registrations/v7_kujiale_ue_asset_admission_20261012.json";dump(path,result)
    dump(root/"evidence/delivery_summary.json",stats)
    conclusion=f"结论：{len(assets)} 个个体已入本轮清单；UE 导入通过 {imported_count}，三视角通过 {visual_count}，连同必需动作检查的原生实体通过 {physical}。失败或未完成 {len(pending)}，颜色一致 {stats['native_color_match']}，源登记无标签陪衬 {stats['native_color_unlabelled_companions']}，颜色待 owner 决定 {len(colors)}。站床修复原生复现：{floor.get('status','未验证')}；试渲 {len(trial_pass)} 张、{len(stats['trial_pass_houses'])} 套房。"
    escape=lambda s:html.escape(str(s))
    cards=[]
    for row in assets:
        images=row["checks"]["three_complete_views"]["evidence"]
        figures="".join(f'<figure><img loading="lazy" src="{thumbnail(v["rgb"])}"><figcaption>视角 {i+1}：可见 {v.get("visible_fraction",0):.2%}；高光 {(v.get("highlight_ratio") or 0):.2%}</figcaption></figure>' for i,v in enumerate(images) if v.get("rgb"))
        closeup=(f'<figure><img loading="lazy" src="{appearance(images[0])}"><figcaption>原生画面局部展示；下列三视角指标基于完整帧。</figcaption></figure>' if images else "")
        if not figures:figures='<p class="bad">无有效原生画面；不得当成已验收资产。</p>'
        color=row["checks"]["color"];measure=color.get("native_measurement") or {}
        gaps=row["checks"]["attachment"].get("deltas_m") or []
        contact=max((abs(x)*1000 for x in gaps),default=None)
        size=row["checks"]["size"].get("native_dimensions_m") or []
        dimension=" × ".join(f"{x:.3f}" for x in size) if size else "未验证"
        emitter=row["checks"]["emitter"]
        contact_text=f"{contact:.3f}" if contact is not None else "未验证"
        size_error=max(row["checks"]["size"].get("relative_errors") or [0])
        emitter_error=emitter.get("maximum_error_m")
        emitter_text=f"{emitter_error*1000:.6g} mm" if emitter_error is not None else emitter.get("status","未验证")
        metric=f'原生尺寸 {dimension} m，最大尺寸误差 {size_error:.2%}；贴附最大误差 {contact_text} mm；发声点误差 {emitter_text}。'
        motion=row["checks"]["native_skeletal_motion"]
        motion_text=[]
        for k,v in motion.items():
            values=v.get("native_metrics") or {}
            gap=values.get("maximum_declared_contact_gap_m")
            gap_text=f"{gap*1000:.3f} mm" if gap is not None else "未验证"
            tilt=values.get("maximum_body_axis_tilt_deg")
            tilt_text=f"{tilt:.3f}°" if tilt is not None else "未验证"
            motion_text.append(f'{k}：{v.get("status")}，{v.get("frame_count") or 0} 帧；支撑爪最大误差 {gap_text}，身体最大倾角 {tilt_text}，固定根校准 {(v.get("native_skin_floor_calibration") or {}).get("fixed_root_y_offset_m",0)*1000:.3f} mm')
        extra="；".join(motion_text) or "静态个体"
        motion_figures="".join(f'<figure><img loading="lazy" src="{thumbnail(v["preview_rgb"])}"><figcaption>原生骨骼 {escape(k)} · 逐帧指标见登记</figcaption></figure>' for k,v in motion.items() if v.get("preview_rgb"))
        cause="<br>".join(escape(x.splitlines()[-1] if x else "未完成") for x in row["failure_reasons"])
        cards.append(f'<article class="card" data-search="{escape(row["asset_id"]+" "+str(row["body_id"])+" "+str(row["object_type"]))}" data-group="{row["category"]}"><h3>{escape(row["asset_id"])}</h3><p class="{ "good" if row["ue_native_physical_status"]=="pass" else "bad"}">{escape(row["admission_decision"])}</p>{closeup}<div class="views">{figures}</div><p>{escape(metric)}</p><p>附着面：{escape(row["attachment_surface"])}；家具/墙穿插：{escape(row["checks"]["furniture_and_wall_clearance"].get("status"))}；四爪：{escape(row["checks"]["static_four_paws"].get("status"))}。</p><p>登记颜色：{escape(color["registered"])}；UE 主色族：{escape(measure.get("label") or "无稳定主色族")}；{escape(color["status"])}；主色占比 {measure.get("main_share") or 0:.2%}、一致视角 {measure.get("dominant_agreeing_views",0)}。</p><p>{escape(extra)}</p><div class="views">{motion_figures}</div><details><summary>UE 资产路径与证据</summary><pre>{escape(chr(10).join(row["ue_asset_paths"]))}</pre><p>{escape(row["primary_native_evidence"])}</p></details>{f"<p class=bad>{cause}</p>" if cause else ""}</article>')
    groups=[]
    names={"dog":"狗","cat":"猫","bird":"鸟（连架子）","object":"物件（含 2 台静音空调）"}
    for category in ["dog","cat","bird","object"]:
        selected=[c for c,row in zip(cards,assets) if row["category"]==category]
        groups.append(f'<section><h2>{names[category]} · {len(selected)} 个体</h2><div class="grid">{"".join(selected)}</div></section>')
    failrows="".join(f'<tr><td>{escape(x["asset_id"])}</td><td>{escape("；".join(t.splitlines()[-1] for t in x["failure_reasons"]))}</td><td>{escape(x["repair"])}</td></tr>' for x in pending)
    colorrows="".join(f'<tr><td>{escape(x["asset_id"])}</td><td>{escape(x["label_color"])}</td><td>{escape((x["checks"]["color"].get("native_measurement") or {}).get("label"))}</td></tr>' for x in colors)
    floorcards=[]
    for case in floor.get("cases",[]):
        figures="".join(f'<figure><img loading="lazy" src="{thumbnail(v["rgb"])}"><figcaption>{ "修复前" if v["phase"]=="before" else "修复后"} · 根高度 {v["native_actor_root_m"][1]:.6f} m · 脚底 {escape(v.get("native_sole",{}).get("status","未验证"))}</figcaption></figure>' for v in case["images"])
        floorcards.append(f'<article class="card"><h3>{escape(case["scene_id"])} · {escape(case["asset_id"])}</h3><div class="views">{figures}</div><p>房间地板语义门禁拒绝旧点；修复状态 {escape(case["status"])}；下沉 10 cm 反例 {escape(case.get("known_bad_10cm_lowered_native_sole",{}).get("status","未验证"))}。</p></article>')
    trialcards="".join(f'<article class="card"><h3>{escape(t["house"])} · {escape(t["coverage_key"])}</h3><img loading="lazy" src="{thumbnail(t["rgb"],640)}"><p>{escape(t["asset_id"])}；实体与人同场：原生深度读回 {escape(t["status"])}。</p></article>' for t in trial_pass)
    referencecards="".join(f'<figure><img loading="lazy" src="{thumbnail(v["reference_copy"])}"><figcaption>已准入 UE 参照 · {escape(v["house"])} · {escape(v["reference_room"])}</figcaption></figure>' for v in result["approved_native_ue_references"])
    crashes=read(root/"evidence/crash_root_summary.json",{"drained_exit":"仍在验证"})
    tests=read(test_path,{"status":"未汇总","note":"历史完整测试有 99 失败、7 错误；原始 b38de68 对同一节点复测相同。新增相关测试见日志。"})
    commit=read(root/"evidence/final_commit.json",{})
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>酷家乐 UE 动物与物件验收</title><style>body{{font:15px/1.6 system-ui,sans-serif;background:#f3f5f7;color:#17212d;margin:24px auto;max-width:1600px;padding:0 20px}}h1{{font-size:27px}}h2{{margin-top:32px}}.conclusion{{background:#fff;border-left:6px solid #32766d;padding:18px;font-size:19px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(350px,1fr));gap:16px}}.card{{background:white;border:1px solid #d4dce2;border-radius:8px;padding:14px;overflow:hidden}}h3{{font-size:14px;overflow-wrap:anywhere}}.views{{display:flex;gap:8px;flex-wrap:wrap}}figure{{margin:0;flex:1;min-width:100px}}img{{width:100%;height:auto;border-radius:4px}}figcaption{{font-size:12px;color:#52616b}}.good{{color:#146c52}}.bad{{color:#a33a37}}table{{border-collapse:collapse;background:white;width:100%;font-size:13px}}th,td{{border:1px solid #ccd4da;padding:8px;text-align:left;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}}input{{font:inherit;padding:10px;width:min(90%,650px)}}details{{font-size:12px}}code{{overflow-wrap:anywhere}}</style><h1>v7 酷家乐 UE 动物与物件验收</h1><p class="conclusion">{escape(conclusion)}</p><p>结论由原生 UE/SPEAR 图像、量测网格、射线和发声点读回产生，图片用于证据展示，无人工看片判定步骤。仅验收和试渲，未量产、未出新题、未提交 CCI/ACP。</p><p>门槛：至少三个完整视角，可见比例 ≥99%，高光 ≤2%；动物贴地 12 mm、物件贴附 15 mm；尺寸误差 ≤15%；发声点误差 ≤0.5 mm。深度为原生径向米，遮挡容差 5 mm。人物站立保持既有 20 mm 脚底门槛，并新增房间地板面身份检查。</p><p>颜色按 UE 画面主色族量测；原登记未修改。下面参照为任务书认可的 ue_round13 房间，使用同一生产白平衡与曝光规则。可走狗的动作验收明确记录是否只覆盖原地动作，不替未来空间路线提供准入。</p><div class="views">{referencecards}</div><p><label>查找个体或种类 <input id="search" placeholder="输入资产 ID、身体或物件类型"></label></p>{"".join(groups)}<h2>失败或未完成清单</h2><table><tr><th>个体</th><th>原因</th><th>修法</th></tr>{failrows or '<tr><td colspan="3">无</td></tr>'}</table><h2>颜色差异：交 owner 决定，标签未改</h2><table><tr><th>个体</th><th>登记标签</th><th>UE 量测族</th></tr>{colorrows or '<tr><td colspan="3">无</td></tr>'}</table><h2>站床修复：kujiale_0003 原生复现</h2><div class="grid">{"".join(floorcards)}</div><h2>三套房子试渲：每种身体与物件、可见人物同场</h2><div class="grid">{trialcards}</div><h2>启动崩溃、验证与交付记录</h2><p>原动物工程的 -NullRHI -NoShaderCompile 在导入前触发编辑器分析模块读取未初始化的 shader 编译统计互斥锁。差分实验在同一私有环境复现该断言；移除 NoShaderCompile 消除启动断言。本轮改用生产 Vulkan/offscreen 方式，最终四个导入批次均退出 0。另一次 NullRHI 退出内存错误的精确 C++ 根因未验证，不将其写成已修复。动作读取器另一次快速退出的 SIGSEGV 已由 gdb 定位到 ShaderJobCache 后台 SubmitJobs → PreprocessShader → GetInputHash。修法是在 Python 返回前执行 Editor.AsyncAssetCompilationFinishAll，排空 shader/texture/mesh 编译；是否验证退出 0 见下方崩溃记录，不能把尚在运行的验证算通过。</p><pre>{escape(json.dumps(crashes,ensure_ascii=False,indent=2))}</pre><pre>{escape(json.dumps(tests,ensure_ascii=False,indent=2))}</pre><p>登记：<code>{escape(path)}</code>；提交：{escape(commit.get("commit","尚未提交"))}；记录与原始指标均在本轮 evidence、JOBS.json、PROGRESS.md。磁盘生成前 {total_bytes/1e9:.3f} GB；GPU 时间按分卡任务区间合并统计，非内核运行时间：{escape(json.dumps(usage,ensure_ascii=False))}。</p><p>酷家乐场景许可限非商业研究，本交付为内部研究产物。</p><script>document.getElementById('search').addEventListener('input',e=>{{const q=e.target.value.toLowerCase();document.querySelectorAll('[data-search]').forEach(x=>x.hidden=!x.dataset.search.toLowerCase().includes(q));}})</script></html>'''
    (root/"ADMISSION_zh.html").write_text(page)
    dump(root/"evidence/report_build.json",{"html":str(root/"ADMISSION_zh.html"),"html_bytes":len(page.encode()),"embedded_images":page.count("data:image/jpeg;base64,"),"registration":str(path),"summary":stats})
    print(json.dumps(stats,ensure_ascii=False))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",required=True,type=Path);args=p.parse_args();compile_report(args.root)

if __name__=="__main__":main()
