#!/usr/bin/env python3
"""Assemble a machine-readable Kujiale room/category whitelist and short Chinese report."""
from __future__ import annotations
import argparse, base64, datetime, html, json, os, subprocess, time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from avengine.capture.kujiale_room_inputs import prepare_room_inputs, admission_floor_reference
from avengine.qa.kujiale_asset_geometry import rotation
import numpy as np

COARSE=("human","dog","cat","bird_with_perch","floor_object","table_object","wall_object","ceiling_object")
LABELS={"human":"站立人","dog":"狗","cat":"猫","bird_with_perch":"鸟连木架",
        "floor_object":"落地物件","table_object":"台面物件","wall_object":"挂墙物件","ceiling_object":"吸顶物件"}

def read(path):
    return json.loads(Path(path).read_text())

def combine_room(cpu,native):
    if (cpu["house"],cpu["room_id"],cpu["ue_map_path"])!=(native["house"],native["room_id"],native["ue_map_path"]):
        raise ValueError("CPU and native room identities differ")
    by_key={c["category_key"]:c for c in native["category_views"]}
    if len(by_key)!=len(native["category_views"]):
        raise ValueError("duplicate native room/category checks")
    cells=[]
    for original in cpu["category_checks"]:
        cell=deepcopy(original);cell["CPU_status"]=cell.pop("status")
        current=by_key.get(cell["category_key"])
        cell["native_point_check"]=deepcopy(current) if current else {"status":"not_run"}
        if cell["CPU_status"]!="available_CPU_native_LOD_witness":
            cell["status"]="unavailable_in_bounded_search"
        elif current is None:
            cell.update(status="not_verified",reason="CPU witness has no current native point readback")
        elif current["representative_asset_id"]!=cell["representative_asset_id"]:
            raise ValueError("native point check is for a different representative asset")
        elif current["status"]!="pass":
            cell.update(status="unavailable_native_visibility",reason=current.get("reason"))
        else:
            selected=current["selected_view"]
            target=selected["placement_point_visibility"].get("point_m")
            if not any(np.max(np.abs(np.array(v["target_m"])-target))<1e-8 for v in cell["placement"]["camera_candidates"]):
                raise ValueError("native selected point differs from CPU placement witness")
            if any(selected[k]["status"]!="pass" for k in ("camera_readback","placement_point_visibility","native_render_availability")):
                raise ValueError("native room check marked pass without all actual point checks")
            cell["status"]="usable_for_tested_representative"
            cell["geometry_attachment_max_error_m"]=max(abs(x) for x in cell["placement"]["contact_deltas_m"])
            cell["geometry_intersecting_triangles"]=cell["placement"]["collision"]["intersecting_native_triangles"]
            if cell["geometry_intersecting_triangles"] or cell["geometry_attachment_max_error_m"]>cell["placement"]["contact_tolerance_m"]:
                raise ValueError("CPU placement marked pass with failed physical evidence")
            cell["visibility_scope"]="Native room pixels and empty placement point; individual asset masks/actions are production gates"
        cells.append(cell)
    return {"house":cpu["house"],"room_id":cpu["room_id"],"split":cpu["split"],
            "ue_map_path":cpu["ue_map_path"],"floor_support_check":deepcopy(cpu["floor_support_check"]),
            "native_floor_surface":deepcopy(cpu["native_floor_surface"]),
            "original_native_floor_admission":deepcopy(cpu["original_native_floor_admission"]),
            "CPU_planning_floor_reference":deepcopy(cpu["CPU_planning_floor_reference"]),
            "native_room_render_status":native["native_room_render_status"],
            "categories":cells,
            "coarse_categories":{key:{"status":"usable_for_tested_representative" if any(c["status"]=="usable_for_tested_representative" for c in cells if c["coarse_category"]==key) else "unavailable_in_bounded_search",
                "usable_category_keys":[c["category_key"] for c in cells if c["coarse_category"]==key and c["status"]=="usable_for_tested_representative"],
                "unavailable_category_keys":[c["category_key"] for c in cells if c["coarse_category"]==key and c["status"]!="usable_for_tested_representative"]} for key in COARSE},
            "scope":"single representative placement witness; simultaneous scene, motions and questions remain unqualified by this precheck"}

def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as f:
        f.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+"\n")

def run(args):
    root=args.root.resolve();runtime=read(args.runtime);rooms=[]
    for house in runtime["houses"]:
        for original in house["checked_rooms"]:
            label=house["house"]+"_"+original["room_label"]
            cpu_path=args.cpu_rooms/(label+".json");native_path=args.native_rooms/(label+".json")
            cpu=read(cpu_path);native=read(native_path)
            retained=prepare_room_inputs(original,cpu_floor_reference=cpu["CPU_planning_floor_reference"])
            if admission_floor_reference(retained)!=cpu["original_native_floor_admission"]:
                raise ValueError("original native floor admission changed")
            room=combine_room(cpu,native);room.update(CPU_evidence=str(cpu_path),native_evidence=str(native_path))
            rooms.append(room)
    cells=[c for room in rooms for c in room["categories"]]
    usable=[c for c in cells if c["status"]=="usable_for_tested_representative"]
    views=[c["native_point_check"]["selected_view"] for c in usable]
    counts=Counter(c["status"] for c in cells)
    summary={"house_count":len(runtime["houses"]),"room_count":len(rooms),
        "room_floor_pass":sum(r["floor_support_check"]["status"]=="pass" for r in rooms),
        "native_room_render_pass":sum(r["native_room_render_status"]=="pass" for r in rooms),
        "detailed_category_count":len(rooms[0]["categories"]) if rooms else 0,
        "room_category_cells":len(cells),"cell_status_counts":dict(counts),
        "coarse_usable_room_counts":{k:sum(r["coarse_categories"][k]["status"]=="usable_for_tested_representative" for r in rooms) for k in COARSE},
        "maximum_attachment_error_m":max((c["geometry_attachment_max_error_m"] for c in usable),default=None),
        "native_intersecting_triangles_in_usable_witnesses":sum(c["geometry_intersecting_triangles"] for c in usable),
        "maximum_camera_position_error_m":max((v["camera_readback"]["position_max_error_m"] for v in views),default=None),
        "maximum_camera_yaw_error_deg":max((v["camera_readback"]["yaw_error_deg"] for v in views),default=None),
        "maximum_camera_pitch_error_deg":max((v["camera_readback"]["pitch_error_deg"] for v in views),default=None),
        "native_visible_placement_points":len(views),"new_parent_scenes":0,"new_questions":0}
    repo=Path(__file__).resolve().parents[2]
    report={"schema":"v7_kujiale_room_asset_category_legality_20261014_v1",
            "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status":"complete_room_precheck" if len(rooms)==33 and summary["room_floor_pass"]==33 and summary["native_room_render_pass"]==33 else "incomplete_or_room_failure",
            "summary":summary,"rooms":rooms,"runtime_input":str(args.runtime),
            "code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
            "physical_gpu_dispatch":[2],"physical_GPU1":"not_verified_current_campaign",
            "original_asset_registrations_modified":False,"all_273_asset_appearance_coverage":"cancelled_by_owner_latest_scope",
            "retained_formal_smoke_parent_count":4,
            "production_requirements":["original individual three-view native admission",
                "actual visible physical emitters","original highlight <=2%",
                "attachment <=15mm for objects and retained species/human contact gates",
                "room floor identity for humans","native emitter readback","RIR direction <=5deg",
                "current-scene color match before any color question","independent question truth recomputation"],
            "limits":["Representative witnesses do not certify every appearance or body action.",
                "Unavailable means this bounded support/geometry/camera search found no qualified witness; physical impossibility is not proven.",
                "Native images show the room and point visibility; final individual and joint-scene asset masks must be checked by production.",
                "Human reserved volume comes from the explicitly referenced native standing pose bounds; other bodies and animation need new qualification."]}
    output=root/"registrations/v7_kujiale_room_asset_category_legality_20261014.json"
    write_new(output,report)
    rows=[]
    for room in rooms:
        values="".join("<td>"+("可用" if room["coarse_categories"][k]["status"]=="usable_for_tested_representative" else "禁用本轮候选")+"</td>" for k in COARSE)
        failed=[c for c in room["categories"] if c["status"]!="usable_for_tested_representative"]
        details="<br>".join(html.escape(c["category_key"]+"："+str(c.get("reason"))) for c in failed) or "无"
        rows.append("<tr><td>"+html.escape(room["house"]+"/"+room["room_id"])+"</td>"+values+"<td><details><summary>"+str(len(failed))+"项</summary>"+details+"</details></td></tr>")
    images=[]
    for room in rooms:
        good=[c for c in room["categories"] if c["status"]=="usable_for_tested_representative"]
        if good and len(images)<6 and room["room_id"]=="R0":
            path=Path(good[0]["native_point_check"]["selected_view"]["rgb"])
            images.append("<figure><img src='data:image/jpeg;base64,"+base64.b64encode(path.read_bytes()).decode()+"'><figcaption>"+html.escape(room["house"]+"/"+room["room_id"])+" UE 原生房间样张；用于摆放点可见性预检</figcaption></figure>")
    title="33间房合法性预检完成，供出题器按合法候选选房和资产。" if report["status"]=="complete_room_precheck" else "房间合法性预检未全过，见机器清单中的限制。"
    document="<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>酷家乐 UE 房间合法清单</title><style>body{font:15px system-ui;margin:24px;line-height:1.6}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}img{max-width:720px;width:100%}details{max-width:450px}code{word-break:break-all}</style><body><h1>"+title+"</h1><p>19套房、"+str(len(rooms))+"间训练房。原地板检查 "+str(summary["room_floor_pass"])+"/33，当前GPU2原生出图 "+str(summary["native_room_render_pass"])+"/33。细分"+str(summary["detailed_category_count"])+"类，共"+str(summary["room_category_cells"])+"格，可用候选"+str(len(usable))+"格。保留4个正式通过父场景作冒烟；本轮没有新增父场景或题。</p><p>每个可用结论只覆盖机器清单明确列出的代表个体、尺寸、支撑点和相机。禁用本轮候选仅表示该有界搜索未找到合规证据；未证明物理上绝对放不下。UE样张渲染房间并验证空摆放点，最终同场资产遮罩、动作、发声点、RIR、颜色和真值门禁继续由出题器执行。</p><p>贴附最大 "+str(summary["maximum_attachment_error_m"]*1000)+" mm；可用候选穿插 "+str(summary["native_intersecting_triangles_in_usable_witnesses"])+" 个三角形；相机最大位置误差 "+str(summary["maximum_camera_position_error_m"]*1000)+" mm，最大偏航 "+str(summary["maximum_camera_yaw_error_deg"])+"°，最大俯仰 "+str(summary["maximum_camera_pitch_error_deg"])+"°。</p><p>修复：UE地板原验收与CPU新点分离（3f5aa283）；Vulkan实渲染器UUID与辅助上下文分开核对、统一NVIDIA ICD（bf848ca5、2131c1c4）。本轮只分配GPU2；GPU1未验证。原始名册、标签和GLB未改。音频沿用已封存冒烟输入，本轮不录音、不出题。</p><p>机器清单：<code>"+str(output)+"</code>；提交 <code>"+report["code_commit"]+"</code>。数字来源：逐房CPU和Native证据路径均在清单内。</p><table><tr><th>房间</th>"+"".join("<th>"+LABELS[k]+"</th>" for k in COARSE)+"<th>细类限制</th></tr>"+"".join(rows)+"</table>"+"".join(images)+"</body></html>"
    page=root/"ROOM_LEGALITY_REPORT_zh.html";write_new(page,document)
    md=title+"\n\n"+json.dumps(summary,ensure_ascii=False,indent=2)+"\n\n机器清单："+str(output)+"\n页面："+str(page)+"\n"
    write_new(root/"ROOM_LEGALITY_REPORT_zh.md",md)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","runtime","cpu-rooms","native-rooms"):
        p.add_argument("--"+name,type=Path,required=True)
    run(p.parse_args())

if __name__=="__main__":
    main()
