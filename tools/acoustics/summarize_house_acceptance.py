#!/usr/bin/env python3
"""Summarize CPU house acceptance evidence, plots and whole-house recommendations."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime
import html
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    with Path(path).open("x", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def compact_number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def collect(manifest, measurements):
    results = []
    for h in manifest["houses"]:
        out = measurements / "houses" / h["house"]
        if (out / "result.json").exists():
            r = read(out / "result.json")
        else:
            r = {"house": h["house"], "family": h["family"], "status": "review",
                 "reasons": ["NATIVE_MEASUREMENT_MISSING"], "rirs": []}
        visual = read(out / "visual_checks.json") if (out / "visual_checks.json").exists() else {"status": "review", "reasons": ["VISUAL_CHECK_MISSING"]}
        status = r["status"]
        reasons = list(r.get("reasons", []))
        r["distinct_listener_positions"] = len({tuple(m["listener_m"]) for m in r.get("rirs", [])})
        if r["distinct_listener_positions"] < manifest["criteria"].get("minimum_listeners", 2):
            if status != "fail":
                status = "review"
            if "MULTIPLE_LISTENER_POSITIONS_MISSING" not in reasons:
                reasons.append("MULTIPLE_LISTENER_POSITIONS_MISSING")
        occluded = {p["room_label"]+"_"+p["source_label"] for p in r.get("occluded_probe_lines", [])}
        for m in r.get("rirs", []):
            m["direct_path_line_clear"] = m["probe"] not in occluded
            m["direct_metrics_validated"] = m["probe"] not in occluded
        package = Path(r.get("acoustic_package_manifest", out / "package" / "manifest.json"))
        coverage = package.parent / "semantic_material_coverage.json"
        if coverage.exists():
            r["material_coverage_path_resolved"] = str(coverage)
        if r.get("geometry_diagnostic"):
            r["geometry_diagnostic"]["source_geometry_changed"] = r["geometry_diagnostic"]["omitted_faces"] > 0
        if visual["status"] != "pass":
            if status != "fail":
                status = "review"
            reasons += ["CPU_VISUAL_CHECK_PENDING"] + visual.get("reasons", [])
        # No CPU diagnostic is a production visual-runtime claim.
        r.update({"input": h, "native_status": r["status"], "visual": visual,
                  "status": status, "reasons": reasons,
                  "known_exposure": h["known_exposure"],
                  "batch_eligible": status == "pass" and not h["known_exposure"],
                  "result_directory": str(out)})
        if h["known_exposure"]:
            r["batch_exclusion"] = "KNOWN_SO_HOUSE_EXPOSURE"
        results.append(r)
    return results


def diverse_subset(items, count):
    if len(items) <= count:
        return sorted(items, key=lambda r: r["house"])
    # Features and selection have no model-answer inputs.
    area = np.log1p([r["input"]["candidate_area_sum_m2"] for r in items])
    rt = np.array([r["rt60_mid_median_s"] for r in items])
    floors = np.array([r["input"]["floor_count_proxy"] for r in items])
    values = np.stack([area, rt, floors], axis=1)
    values = (values-values.min(axis=0)) / np.maximum(values.max(axis=0)-values.min(axis=0), 1e-9)
    chosen = [int(np.argmin(area))]
    while len(chosen) < count:
        distances = np.min(np.linalg.norm(values[:,None,:]-values[chosen][None,:,:], axis=2),axis=1)
        distances[chosen] = -1
        best = max(range(len(items)), key=lambda i: (float(distances[i]), items[i]["house"]))
        chosen.append(best)
    return [items[i] for i in chosen]


def recommend(results):
    eligible = [r for r in results if r["batch_eligible"] and r["family"] == "hm3d"]
    train = [r for r in eligible if r["input"]["original_split"] == "train"]
    val = [r for r in eligible if r["input"]["original_split"] == "val"]
    for n in range(16,11,-1):
        nt = round(n*.7); nd = round(n*.15); ne = n-nt-nd
        if len(train) >= nt and len(val) >= nd+ne and nd>=2 and ne>=2:
            ts = diverse_subset(train,nt)
            vs = sorted(diverse_subset(val,nd+ne),key=lambda r:(r["rt60_mid_median_s"],r["house"]))
            # Interleave val RT ranks so neither holdout gets only one end of this pool.
            order = vs[::2]+vs[1::2]
            dev,test=order[:nd],order[nd:]
            groups = {"train":ts,"development":dev,"test":test}
            rows = [{"house":r["house"],"split":name,
                     "rt60_mid_median_s":r["rt60_mid_median_s"],
                     "candidate_area_sum_m2":r["input"]["candidate_area_sum_m2"],
                     "candidate_room_count":r["input"]["candidate_room_count"],
                     "room_types":r["input"]["room_types"],
                     "original_split":r["input"]["original_split"],
                     "result_path":str(Path(r["result_directory"])/"result.json"),
                     "visual_path":str(Path(r["result_directory"])/"visual_checks.json")}
                    for name,rs in groups.items() for r in rs]
            return {"target_met":True,"houses":rows,"house_counts":{k:len(v) for k,v in groups.items()},
                    "house_ratios":{k:len(v)/n for k,v in groups.items()},
                    "independent_window_target":3000,
                    "window_targets":{"train":2100,"development":450,"test":450},
                    "known_exposure_excluded":True,"anonymous_exposure_verified":False,
                    "native_habitat_rgb_verified":False,"production_readiness_verified":False,
                    "target_met_scope":"12-16 clean houses and whole-house split only; not complete v7 production",
                    "method":"eligible acoustic and CPU visual pass; whole houses; original val held out; area/RT/floor diversity; no model results"}
    return {"target_met":False,"houses":[],
            "eligible_houses":[r["house"] for r in eligible],
            "eligible_train_count":len(train),"eligible_val_count":len(val),
            "reason":"Fewer than 12 eligible houses with at least four original val houses; no review house promoted"}


def make_plots(results, report):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    passed = [r for r in results if r["status"]=="pass" and r.get("rt60_mid_median_s") is not None]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    bins=np.array([.2,.35,.5,.65,.8,1.,1.25,1.5,2.,2.5])
    for family,color in (("hm3d","steelblue"),("mp3d","darkorange")):
        rs=[r for r in passed if r["family"]==family]
        if rs:axes[0].hist([r["rt60_mid_median_s"] for r in rs],bins=bins,alpha=.65,label=f"{family}: {len(rs)} houses",color=color)
    axes[0].axvspan(1.5,2.5,color="orange",alpha=.12)
    axes[0].set(title="All diagnostic-passed houses: equal house weight",xlabel="500/1k/2k Hz median RT60 (s)",ylabel="houses",xlim=(.2,2.5))
    axes[0].legend()
    ordered=sorted(passed,key=lambda r:(r["rt60_mid_median_s"],r["house"]))
    for i,r in enumerate(ordered):
        vals=[m["rt60_mid_s"] for m in r["rirs"] if m.get("status")=="valid" and m.get("rt60_mid_s") is not None]
        axes[1].scatter(vals,np.full(len(vals),i),alpha=.45,s=14)
        axes[1].scatter([r["rt60_mid_median_s"]],[i],marker="D",s=28,color="black")
    axes[1].set(yticks=list(range(len(ordered))),yticklabels=[r["house"].split("_")[-1] for r in ordered],
                xlabel="per-position RT60 (s); black diamond = house median",title="Position coverage; correlated positions are not houses")
    fig.tight_layout();fig.savefig(report/"passed_rt60_distribution.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,max(5,len(results)*.26)))
    states={"pass":0,"review":1,"fail":2};colors=["#70ad81","#ecc269","#ce6f65"]
    from matplotlib.colors import ListedColormap
    matrix=np.array([[states[r["native_status"]],states[r["visual"]["status"]],states[r["status"]]] for r in results])
    ax.imshow(matrix,cmap=ListedColormap(colors),vmin=-.5,vmax=2.5,aspect="auto")
    ax.set(xticks=[0,1,2],xticklabels=["RIR + rays","CPU image check","combined diagnosis"],
           yticks=np.arange(len(results)),yticklabels=[r["house"] for r in results])
    for i,row in enumerate(matrix):
        for j,v in enumerate(row):ax.text(j,i,["pass","review","fail"][v],ha="center",va="center",fontsize=7)
    fig.tight_layout();fig.savefig(report/"house_statuses.png",dpi=150);plt.close(fig)


def qa_catalog():
    tree=ast.parse((REPOSITORY/"src/avengine/qa/unified_catalog.py").read_text())
    rows={}
    for node in ast.walk(tree):
        if not isinstance(node,ast.Dict):continue
        fields={k.value:v for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant) and isinstance(k.value,str)}
        if all(k in fields and isinstance(fields[k],ast.Constant) for k in ("qa_id","title","question_family")):
            rows[fields["qa_id"].value]={k:fields[k].value for k in ("title","question_family")}
    return dict(sorted(rows.items()))


def summarize(args):
    report=args.report
    report.mkdir(parents=True,exist_ok=False)
    manifest=read(args.manifest)
    results=collect(manifest,args.measurements)
    recommendations=recommend(results)
    timing=np.array([m["native_simulation_wall_seconds"] for r in results for m in r.get("rirs",[]) if "native_simulation_wall_seconds" in m])
    passed=[r for r in results if r["status"]=="pass"]
    primary=[r for r in passed if r.get("rt60_coverage_bin")=="primary"]
    long_tail=[r for r in passed if r.get("rt60_coverage_bin")=="long_tail"]
    medians=[r["rt60_mid_median_s"] for r in passed if r.get("rt60_mid_median_s") is not None]
    stats={"generated_at":datetime.now().astimezone().isoformat(),
           "houses":len(results),"by_family_status":{f:dict(Counter(r["status"] for r in results if r["family"]==f)) for f in ("hm3d","mp3d")},
           "native_by_family_status":{f:dict(Counter(r["native_status"] for r in results if r["family"]==f)) for f in ("hm3d","mp3d")},
           "rirs":sum(len(r.get("rirs",[])) for r in results),
           "houses_with_rirs":sum(bool(r.get("rirs")) for r in results),
           "rir_status_counts":dict(Counter(m.get("status") for r in results for m in r.get("rirs",[]))),
           "unresolved_tail_rirs":sum(bool(m.get("tail_truncated")) for r in results for m in r.get("rirs",[])),
           "baseline_ray_count":sum(sum(l["ray_count"] for l in r.get("leakage",[])) for r in results),
           "cpu_views":sum(len(r["visual"].get("checks",[])) for r in results),
           "passed_rt60_house_median_range_s":[min(medians),max(medians)] if medians else None,
           "passed_primary_houses":len(primary),"passed_long_tail_houses":len(long_tail),
           "coverage_has_long_tail":bool(long_tail),
           "native_rir_seconds":{"n":len(timing),"median":float(np.median(timing)) if len(timing) else None,
                                 "mean":float(timing.mean()) if len(timing) else None,
                                 "p90":float(np.quantile(timing,.9)) if len(timing) else None,
                                 "sum":float(timing.sum())},
           "native_measure_house_wall_seconds_sum":sum(r.get("wall_seconds",0) for r in results),
           "known_exposure_houses":[r["house"] for r in results if r["known_exposure"]],
           "manifest_source":str(args.manifest),"measurement_root":str(args.measurements),
           "formal_visual_runtime":"not_run: CPU-only task, no Habitat GPU capture",
           "recommendations":recommendations}
    test_path=args.manifest.parent/"evidence"/"cpu_test_summary.json"
    stats["cpu_tests"]=read(test_path) if test_path.exists() else {"status":"not_supplied"}
    stats["complete_v7_readiness_verified"]=False
    save(report/"summary.json",stats);save(report/"recommended_houses.json",recommendations)
    save(report/"all_house_results.json",results);save(report/"v6_qa_catalog.json",qa_catalog())
    columns=["house","family","original_split","candidate_area_sum_m2","candidate_room_count","native_status",
             "visual_status","status","batch_eligible","rt60_mid_median_s","max_escape_fraction","max_foa_angle_deg","reasons","result_directory"]
    with (report/"house_results.csv").open("x",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in results:
            writer.writerow({**{k:r.get(k) for k in columns if k not in ("original_split","candidate_area_sum_m2","candidate_room_count","visual_status","reasons")},
                             **{k:r["input"].get(k) for k in ("original_split","candidate_area_sum_m2","candidate_room_count")},
                             "visual_status":r["visual"]["status"],"reasons":"; ".join(r["reasons"]) + ("; KNOWN_SO_EXPOSURE" if r["known_exposure"] else "")})
    make_plots(results,report)
    labels={"pass":"通过","review":"待审","fail":"不通过"}
    lines=["# v7第一批房屋声学验收汇总","",
           "结论："+(f"已找到{len(recommendations['houses'])}套CPU诊断通过的干净房屋；长尾覆盖与生产验收仍有缺口。" if recommendations["target_met"] else "当前通过集合不足首批12套及整屋留出要求；没有把待审屋提升为通过。"),"",
           f"本轮检查{len(results)}套房屋，其中{stats['houses_with_rirs']}套完成RIR测量，共{stats['rirs']}个RIR、{stats['baseline_ray_count']}条球面射线，软件视图{stats['cpu_views']}张。数字来源：本目录summary.json及逐屋result.json/visual_checks.json。RLR设置与v6最终生产收据一致；显式退化面诊断会单列并保持待审。只运行CPU，无GPU、push或原始数据修改。","",
           "acoustic pass为CPU诊断通过，不代表正式Habitat视频、物理材质实测或Episode生产准入。已知SO/M1房屋排除；匿名SO和跨数据集别名不能给出绝对未见保证。","",
           "## 全部房屋","",
           "| 房屋 | 来源 | RIR/射线 | CPU视图 | 综合 | 中频RT60中位秒 | 最差逃逸% | 最大FOA误差° | 原因/排除 |",
           "|---|---|---|---|---|---|---:|---:|---|"]
    for r in results:
        why="; ".join(r["reasons"]) or "已完成本轮检查"
        if r.get("geometry_diagnostic"):
            why += f"；诊断省略{r['geometry_diagnostic']['omitted_faces']}个原生退化面，原包不改"
        if r["known_exposure"]:why+="；已知SO暴露，不进首批"
        lines.append(f"| {r['house']} | {r['family']} | {labels[r['native_status']]} | {labels[r['visual']['status']]} | {labels[r['status']]} | {compact_number(r.get('rt60_mid_median_s'))} | {compact_number(100*r['max_escape_fraction'] if r.get('max_escape_fraction') is not None else None,2)} | {compact_number(r.get('max_foa_angle_deg'))} | {why} |")
    lines+=["","每屋证据在输入measurement_root/houses/<house>/：result.json、rirs.npz、leakage.json、acoustics.png、foa_projection.png、visual_checks.json、cpu_view_1..3.png、hole_locators.json、raw_boundary_edges.npz。图中洞口位置是扫描边界候选，不是已恢复/修补的完整破洞。","",
            "## 推荐名单与整屋划分",""]
    if recommendations["target_met"]:
        lines += [f"共{len(recommendations['houses'])}套，整屋数量{recommendations['house_counts']}，比例{recommendations['house_ratios']}。默认独立场景窗口3000（2100/450/450），这是后续生产目标，本轮没有生成训练窗口。","",
                  "| 划分 | 房屋 | 中频RT60秒 | 候选房间面积和m² | 房型 |","|---|---|---:|---:|---|"]
        for h in recommendations["houses"]:
            lines.append(f"| {h['split']} | {h['house']} | {h['rt60_mid_median_s']:.3f} | {h['candidate_area_sum_m2']:.1f} | {', '.join(h['room_types'])} |")
    else:
        lines += [recommendations["reason"],"可进一步复核的通过屋："+", ".join(recommendations["eligible_houses"])]
    lines+=["","## RT60覆盖","",
            f"全部综合通过屋的中频屋中位数范围：{stats['passed_rt60_house_median_range_s']}秒；主范围屋{len(primary)}，1.5–2.5秒长尾屋{len(long_tail)}。","",
            ("有长尾屋，但需按名单检查留出覆盖。" if long_tail else "没有屋中位数落入1.5–2.5秒长尾；本轮尚未满足完整宽覆盖目标。不能把0.2–1.5配置范围当实际全覆盖。后续按结构与语义材料的合理范围增加profile或候选，并重新验收；不从STARSS波形拟合。"),
            "","![全部通过屋RT60分布](passed_rt60_distribution.png)","","## 吞吐与验证","",
            f"本轮v6基线RIR原生simulate耗时：n={len(timing)}，中位{compact_number(stats['native_rir_seconds']['median'])}秒、平均{compact_number(stats['native_rir_seconds']['mean'])}秒、P90={compact_number(stats['native_rir_seconds']['p90'])}秒。这是固定屋中的单源/固定听者IR，含既有持久上下文，不含编译、软件视图、60秒连续卷积或视频。","",
            "完整渲染预算见V7_SPEC_zh.md第9节。旧combined finalize不是单独IR计时；本轮不启动GPU生产或训练。测试日志位于产物根logs/tests_focused.log、tests_fast_unit_final.log、tests_geometry_and_receipt.log、tests_full_hermetic.log；完整扩展测试未全部通过，不能只引用快速测试声明全套通过。逐项计数见summary.json.cpu_tests与evidence/cpu_test_summary.json，Git交付记录另列。","",
            "## 限制与后续","",
            "CPU原始GLB视图用于坐标/轴/标记像素检查，纹理按三角形近似采样；尚未验证正式Habitat GPU RGB。材质是v6语义近似，不等同实测吸声。通过只覆盖抽样房间/位置，不能证明整屋每个角落都合格；生产必须使用被检查的房间，并对新位置做几何/RIR抽查。","",
            "MP3D原生加载失败时保留待审与错误记录，不省略正面积面来强行加载；没有RIR就没有RT60/FOA通过结论。输入阻断屋的缺失数据在表中为—。","",
            "严格主线干声和标签来源、逐素材许可、匿名历史暴露、navmesh来源、人物覆盖、60秒生成器/时变卷积/v0.2适配及独立窗口生产仍需后续落实；规格设计和本轮房屋验收不能替代这些实现。",""]
    (report/"REPORT_zh.md").write_text("\n".join(lines),encoding="utf8")
    html_lines=["<!doctype html><meta charset='utf-8'><title>v7 CPU acoustic acceptance</title>",
                "<style>body{font:14px system-ui;max-width:1100px;margin:auto}img{max-width:100%}details{border:1px solid #ccc;margin:10px;padding:10px}code{word-break:break-all}</style>",
                "<h1>v7 CPU acoustic acceptance</h1><p>Diagnostic software views; no native Habitat GPU RGB.</p>",
                "<img src='passed_rt60_distribution.png'>"]
    import os
    for r in results:
        directory=Path(r["result_directory"]);relative=os.path.relpath(directory,report)
        html_lines.append("<details><summary>"+html.escape(r["house"]+" / "+r["status"])+"</summary>")
        html_lines.append("<p>"+html.escape("; ".join(r["reasons"]))+"</p>")
        for name in ["acoustics.png","cpu_view_1.png","cpu_view_2.png","cpu_view_3.png"]:
            if (directory/name).exists():html_lines.append(f"<img src='{html.escape(relative+'/'+name)}'>")
        html_lines.append(f"<p><a href='{html.escape(relative+'/result.json')}'>RIR result</a> · <a href='{html.escape(relative+'/hole_locators.json')}'>Hole candidates</a></p></details>")
    (report/"review.html").write_text("\n".join(html_lines),encoding="utf8")
    print(json.dumps({k:v for k,v in stats.items() if k not in ("recommendations",)},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("manifest","measurements","report"):p.add_argument("--"+key,type=Path,required=True)
    summarize(p.parse_args())


if __name__=="__main__":main()
