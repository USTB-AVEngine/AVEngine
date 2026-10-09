"""Owner/smy pages and a new HM3D draft from accepted v5 + native cleanup."""
from __future__ import annotations
import argparse,base64,copy,csv,datetime,html,json,resource
from pathlib import Path
from collections import Counter,defaultdict
import shapely
from shapely.geometry import shape,Point
from tools.rooms.room_split_auto import capped_review as cr,revision_review as rr,native_followup as nf
from tools.rooms.room_split_auto.seam_connectivity import Connectivity,exterior
from tools.rooms.room_selection.measurements import navmesh_triangles
from tools.rooms.room_split_auto.pipeline import load_native,nav_scope_at
dump=nf.dump

def csv_write(path,rows,fields):
    with path.open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader()
        for row in rows:
            record={k:(json.dumps(row[k],ensure_ascii=False) if isinstance(row.get(k),(dict,list)) else row.get(k)) for k in fields}
            writer.writerow(record)

def panel(reg,root,out,tag):
    # Historic below-min admissions are pending, visually use the unresolved grid.
    rendered=copy.deepcopy(reg)
    for block in rendered["blocks"]:
        if block["decision"]=="pending_legacy":block["decision"]="unresolved"
    images=[];missing=[];blocks=['<article><h2>'+html.escape(reg["house"]+"/"+reg["source_region"])+'</h2>']
    for fid in sorted({b["floor_id"] for b in reg["blocks"] if b["floor_area_m2"]>=.5}):
        frame=cr.frame(reg,fid,root)
        if frame is None:missing.append(fid);blocks.append('<p>真实CPU图缺失，未验证</p>');continue
        dest=out/"media"/(reg["house"]+"__"+reg["source_region"]+"__"+fid+"__"+tag+".jpg")
        with dest.open("xb") as f:rr.overlay(rendered,fid,frame,tag).save(f,format="JPEG",quality=85)
        blocks.append('<img width="900" height="900" loading="lazy" decoding="async" style="width:100%;aspect-ratio:1;object-fit:contain" data-src="'+cr.embedded(dest)+'">')
        images.append(str(dest))
    blocks.append(cr.table(reg)+"</article>")
    return "".join(blocks),dict(source=reg["house"]+"/"+reg["source_region"],images=images,missing=missing)

def native_review(root):
    out=root/"existing_rooms_connectivity_v1";done=json.loads((out/"completed.json").read_text());metrics=json.loads((out/"metrics.json").read_text())
    rooms={p.stem:json.loads(p.read_text()) for p in (out/"regions").glob("*.json")}
    affected=[r for r in rooms.values() if r["metrics"]["affected"]]
    blocks=[b for r in rooms.values() for b in r["blocks"]]
    kept=[b for b in blocks if b["decision"]=="retain"];pending=[b for b in blocks if b["decision"] in ("unresolved","pending_legacy")]
    failures=[];hs=load_native();cached={};plan=json.loads((out/"plan.json").read_text());native={r["room_id"]:r for j in plan["jobs"] for r in j["rooms"]}
    for b in kept:
        g=shape(b["floor_polygon_xz_m"]);src=native[b["original_room_id"]];house=b["house"]
        if house not in cached:cached[house]=navmesh_triangles(hs,Path(src["navmesh_source"]))[1:]
        np_,ny=cached[house];ctx=Connectivity(nav_scope_at(np_,ny,b["floor_y_m"],exterior(shape(src["floor_polygon_xz_m"])).buffer(.3),plan["parameters"]))
        certificate=b["connectivity_certificate"]
        if certificate.get("partition_cell_xz_m"):ctx.masks[g.wkb]=shape(certificate["partition_cell_xz_m"])
        checks=dict(area=6-1e-8<=g.area<=35+1e-8,short_side=b["short_side_m"]>=2.4-1e-8,
            connectivity=ctx.count(g)==1,black=b["black_fraction"] is not None and b["black_fraction"]<=.15,
            witness=b["placement_witness"].get("found") is True,
            witness_inside=nf.witness_in(g,b["placement_witness"]))
        for check,passed in checks.items():
            if not passed:failures.append(dict(id=b["id"],check=check))
    # Exact raw-ground conservation for non-delegated native rooms.
    for reg in rooms.values():
        if reg["metrics"]["status"]=="replaced_by_v5_cap_cut":continue
        raw=shape(reg["source_geometry"]["floors"][0]["floor_polygon"])
        union=shapely.union_all([shape(b["floor_polygon_xz_m"]) for b in reg["blocks"]])
        if raw.symmetric_difference(union).area>1e-6:failures.append(dict(id=reg["house"]+"/"+reg["source_region"],check="AREA_PARTITION"))
    width_sorted=sorted(metrics,key=lambda r:(r["body_width_m"],r["room_id"]))[:20]
    ring=next((r for r in metrics if "00172_" in r["house"] and r["room_label"]=="R6"),None)
    summary=dict(native_expected=plan["native_rooms"],native_measured=len(metrics),native_errors=done["errors"],
        native_gt35_replaced_by_v5=sum(r["status"]=="replaced_by_v5_cap_cut" for r in metrics),
        affected_subcap_originals=len(affected),native_retained=len(kept),native_retained_main=sum(b["native_main"] for b in kept),
        new_detached_candidates_retained=sum(not b["native_main"] for b in kept),retained_area_m2=sum(b["floor_area_m2"] for b in kept),
        discard_count=sum(b["decision"]=="discard" for b in blocks),discard_area_m2=sum(b["floor_area_m2"] for b in blocks if b["decision"]=="discard"),
        unresolved_count=sum(b["decision"]=="unresolved" for b in blocks),legacy_pending_count=sum(b["decision"]=="pending_legacy" for b in blocks),
        old_detached_fragment_area_now_retained_m2=sum(r["old_detached_fragment_area_now_retained_m2"] for r in metrics),
        old_detached_fragment_original_rooms_saved=sum(r["old_detached_fragment_area_now_retained_m2"]>1e-8 for r in metrics),
        old_geometric_minimum_fragment_area_now_retained_m2=sum(r["old_frozen_geometric_fragment_area_now_retained_m2"] for r in metrics),
        old_geometric_minimum_original_rooms_saved=sum(r["old_frozen_geometric_fragment_area_now_retained_m2"]>1e-8 for r in metrics),
        old_raw_separation_avoided_rooms=sum(r["old_separation_avoided"] for r in metrics),
        corridor_like_original_count=sum(r["body_width_m"]<1.5 for r in metrics),
        corridor_diagnostic_only_no_original_discard_by_width=True,
        courtyard_ring_R6=ring,validation_passed=not failures and len(metrics)==plan["native_rooms"] and not done["errors"],
        validation_failures=failures,source_native_polygon_manifest=str(nf.PREP),lists_modified=False,acoustics="not_run")
    dump(out/"summary.json",summary);dump(out/"validation.json",dict(passed=summary["validation_passed"],failures=failures))
    dump(out/"pending.json",pending);dump(out/"narrowest20.json",width_sorted)
    csv_write(out/"rooms.csv",blocks,["id","house","source_region","floor_id","floor_y_m","floor_area_m2","short_side_m","black_fraction","decision","discard_reasons","unresolved_reasons","native_main","source_selection","placement_witness"])
    csv_write(out/"width_all846.csv",metrics,[k for k in metrics[0] if k not in ("old_raw_component_areas_m2","new_direct_component_areas_m2")])
    index=[]
    with (out/"review.html").open("x") as f:
        f.write('<!doctype html><html lang="zh"><meta charset="utf-8"><title>原有房间去飞地</title>'+cr.STYLE+
            '<p>原有房间按接缝和局部navmesh规则去真飞地；原生≤35不按面积切，不新增圆形或80%门槛。走廊外观仅列供owner判断，未据此删除原有房间。</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
        for reg in sorted(affected,key=lambda r:r["house"]+r["source_region"]):
            text,entry=panel(reg,root,out,"原有去飞地");f.write(text);index.append(entry)
        f.write(cr.LAZY+'</html>')
    with (out/"review_corridor_like.html").open("x") as f:
        f.write('<!doctype html><html lang="zh"><meta charset="utf-8"><title>原有房间最窄20间</title>'+cr.STYLE+
            '<p>原有房间主体宽度诊断，不据此丢房；列最窄20间和00172/R6中庭环廊。小扫描洞在宽度代理中填掉，≥6m²原始大空洞保留，同时列原始带洞和全部填洞的宽度。</p>')
        chosen=list(width_sorted)
        if ring and ring not in chosen:chosen.append(ring)
        for row in chosen:
            f.write('<pre>'+html.escape(json.dumps(row,ensure_ascii=False,indent=2))+'</pre>')
            reg=rooms[row["house"]+"__"+row["room_label"]]
            if not reg["blocks"]:
                src=native[row["room_id"]];ctx=Connectivity(exterior(shape(src["floor_polygon_xz_m"])))
                reg=dict(reg,blocks=[nf.native_block(src,shape(src["floor_polygon_xz_m"]),0,"unchanged",{},dict(black_fraction=None),[],[],ctx,True)])
            text,entry=panel(reg,root,out,"宽度诊断");f.write(text);index.append(entry)
        f.write(cr.LAZY+'</html>')
    dump(out/"html_validation.json",dict(index=index,missing=[e for e in index if e["missing"]],embedded=True,strict_lazy=True))
    report=f"原有846间连接重判完成={summary['validation_passed']}；受影响≤35原生{len(affected)}间，新增飞地候选保留{summary['new_detached_candidates_retained']}间。\\n\\n"
    report+=f"旧规则确定会按小碎块丢掉、而新规则保住地面{summary['old_detached_fragment_area_now_retained_m2']:.6f}m²，涉及{summary['old_detached_fragment_original_rooms_saved']}间原有房；依据原始part<6m²与新保留多边形的精确交集，不把≥6m²仍可当候选的整块面积说成丢失。metrics.json、summary.json。按原始part量面积或短边必定不过、现在保住的总下界{summary['old_geometric_minimum_fragment_area_now_retained_m2']:.6f}m²/{summary['old_geometric_minimum_original_rooms_saved']}间，包含前述小碎块，不能再相加。原始多part通过接缝/navmesh合为一组{summary['old_raw_separation_avoided_rooms']}间。\\n\\n"
    report+=f"原有保留主块{summary['native_retained_main']}间，另保留飞地候选{summary['new_detached_candidates_retained']}间；丢弃{summary['discard_count']}块/{summary['discard_area_m2']:.3f}m²；未判定{summary['unresolved_count']}块；原先名单已存在的冻门槛例外{summary['legacy_pending_count']}间单列pending，不修改原名单，也不冒称符合6m²。54个原生>35由v5代替，未再重复加入草稿。summary.json、pending.json。\\n\\n"
    report+=f"像走廊的原有房{summary['corridor_like_original_count']}间（宽度<1.5m），先不按此丢；最窄20见narrowest20.json和review_corridor_like.html。width_all846.csv报告所有846间三种宽度，扫描洞可使原始宽度偏低，大中庭洞不能全填后当作大厅。00172/R6：{json.dumps(ring,ensure_ascii=False)}。\\n\\n"
    report+="受影响地面重跑冻结摆放搜索（同高度、净空、距离、相机预算、整屋CPU光线检查），未新增圆/80%准入；未受影响原生使用既有冻结见证与黑区测量，逐间保留出处。独立验证包括真实地面内见证、新连通判据、面积守恒；原名单与另两个任务检出/产物只读。\\n"
    with (out/"REPORT_zh.md").open("x") as f:f.write(report.replace("\\n","\n"))
    return rooms,summary

def smy_review(root,native_rooms):
    out=root/"delivery_all_v5";targets=[json.loads(p.read_text()) for p in (out/"final_v1/regions").glob("*.json")]
    targets=[r for r in targets if r["requires_split"]]
    targets+= [r for r in native_rooms.values() if r["metrics"]["affected"]]
    entries=[];controls=0
    media=out/"media_smy";media.mkdir(exist_ok=False)
    with (out/"review_for_smy.html").open("x") as f:
        f.write('<!doctype html><html lang="zh"><meta charset="utf-8"><title>smy房间切分复核</title>'+cr.STYLE+
            '<p>请判断每间是不是一间说得通的房间、切线位置是否合适。每间选「对 / 不对 / 不确定」并写原因；自动存到本浏览器本地，完成后导出JSON。本页上限35m²，整屋仍完整渲染，切分只限定站位。</p><button id="export">导出 JSON</button><p id="state"></p>')
        for reg in sorted(targets,key=lambda r:r["house"]+r["source_region"]):
            f.write('<article><h2>'+html.escape(reg["house"]+"/"+reg["source_region"])+'</h2>')
            for fid in sorted({b["floor_id"] for b in reg["blocks"] if b["floor_area_m2"]>=.5}):
                fr=cr.frame(reg,fid,root)
                if fr is None:continue
                p=media/(reg["house"]+"__"+reg["source_region"]+"__"+fid+".jpg")
                with p.open("xb") as dst:rr.overlay(reg,fid,fr,"smy复核v5").save(dst,format="JPEG",quality=85)
                f.write('<img loading="lazy" decoding="async" width="900" height="900" style="width:100%;aspect-ratio:1;object-fit:contain" data-src="'+cr.embedded(p)+'">')
            f.write(cr.table(reg))
            ordered=sorted(reg["blocks"],key=lambda b:(b["decision"]!="retain",-b["floor_area_m2"]))
            for no,b in enumerate(ordered,1):
                if b["floor_area_m2"]<6:continue
                bid=html.escape(b["id"],quote=True)
                f.write('<div data-room="'+bid+'"><b>'+f'{no}号 {b["floor_area_m2"]:.2f}m² '+html.escape(b["decision"])+'</b> ')
                for val in ("对","不对","不确定"):
                    f.write('<label><input type="radio" name="'+bid+'" value="'+val+'">'+val+'</label> ')
                f.write('<input type="text" class="reason" placeholder="原因" style="width:45%" aria-label="原因"></div>')
                entries.append(dict(id=b["id"],source_region=reg["house"]+"/"+reg["source_region"],floor_area_m2=b["floor_area_m2"],decision=b["decision"]));controls+=1
            f.write('</article>')
        f.write('<script>const rooms='+json.dumps(entries,ensure_ascii=False).replace("</","<\\/")+';const key="hm3d-room-split-v5-smy-review";let saved={};try{saved=JSON.parse(localStorage.getItem(key)||"{}")}catch(e){};function collect(){const records={};document.querySelectorAll("[data-room]").forEach(d=>{const c=d.querySelector("input:checked");records[d.dataset.room]={choice:c?c.value:null,reason:d.querySelector(".reason").value}});return records}function persist(){saved=collect();try{localStorage.setItem(key,JSON.stringify(saved));document.getElementById("state").textContent="已保存到此浏览器"}catch(e){document.getElementById("state").textContent="本地存储失败，请导出JSON"}}document.querySelectorAll("[data-room]").forEach(d=>{const s=saved[d.dataset.room]||{};d.querySelectorAll("input[type=radio]").forEach(i=>{i.checked=i.value===s.choice});d.querySelector(".reason").value=s.reason||""});document.addEventListener("input",persist);document.addEventListener("change",persist);document.getElementById("export").onclick=()=>{const records=collect();const payload={schema:"smy_room_split_v5_review",created_at:new Date().toISOString(),rooms:rooms.map(r=>({...r,...records[r.id]}))};const u=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=u;a.download="smy_room_split_v5_review.json";a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};</script>')
        f.write(cr.LAZY+'</html>')
    dump(out/"smy_review_validation.json",dict(source_regions=len(targets),room_controls=controls,embedded=True,strict_lazy=True,browser_storage_and_export_implementation_verified_by_code=True,browser_interaction_execution="unverified"))

def draft(root,native_rooms,native_summary):
    out=root/"final_room_list_draft_v1";out.mkdir(exist_ok=False)
    v5=[json.loads(p.read_text()) for p in (root/"delivery_all_v5/final_v1/regions").glob("*.json")]
    native=[b for r in native_rooms.values() for b in r["blocks"] if b["decision"]=="retain"]
    new=[b for r in v5 if r["requires_split"] for b in r["blocks"] if b["decision"]=="retain"]
    originals=json.loads(nf.PREP.read_text())["rooms"];oldkey={(r["house"],r["room_label"]):r for r in originals}
    records=[];pending=[];duplicates=[]
    for b in native+new:
        original=oldkey.get((b["house"],b["source_region"]))
        record=copy_record(b)
        if b in new:
            record["source"]="new_cut";record["origin_list"]=("strict" if "strict" in original["source_list_name"] else "review band") if original else "new"
            record["source_selection"]=original["source_list_name"] if original else None
        records.append(record)
    ids=Counter(r["id"] for r in records)
    if any(v>1 for v in ids.values()):raise RuntimeError("duplicate room ids")
    # Sources cap-split by v5 must not also contribute an original native room.
    native_keys={(b["house"],b["source_region"]) for b in native}
    v5_keys={(b["house"],b["source_region"]) for b in new}
    overlaps=native_keys&v5_keys
    if overlaps:raise RuntimeError("native/cap duplicate source requires explicit floor-aware audit: "+repr(overlaps))
    for reg in native_rooms.values():pending.extend(b for b in reg["blocks"] if b["decision"] in ("unresolved","pending_legacy"))
    for reg in v5:
        if reg["requires_split"]:pending.extend(b for b in reg["blocks"] if b["decision"]=="unresolved")
    eligible=[r for r in records if 6-1e-8<=r["floor_area_m2"]<=35+1e-8]
    if len(eligible)!=len(records):raise RuntimeError("draft admitted outside cap35")
    dump(out/"room_list_draft_v1.json",dict(schema="HM3D_room_list_draft_v1_cap35",rooms=records,leakage="pending separate acoustics task",production_integrated=False))
    dump(out/"pending.json",pending)
    csv_write(out/"rooms.csv",records,["id","house","source_region","source_region_id","floor_id","floor_y_m","floor_area_m2","short_side_m","black_fraction","source","origin_list","source_selection","floor_polygon_xz_m","placement_witness","leakage"])
    bins={label:sum(lo<=r["floor_area_m2"]<(hi if hi<35 else hi+1e-8) for r in records) for label,lo,hi in [("6_10",6,10),("10_20",10,20),("20_30",20,30),("30_35",30,35)]}
    summary=dict(total_rooms=len(records),total_area_m2=sum(r["floor_area_m2"] for r in records),area_bins=bins,
        inherited_native_main=sum(r["source"]=="original" for r in records),
        native_detached_new=sum(r["source"]=="native_detached_candidate" for r in records),
        cap_cut_new=sum(r["source"]=="new_cut" for r in records),native_connectivity_affected_originals=native_summary["affected_subcap_originals"],
        pending=len(pending),native_legacy_pending=native_summary["legacy_pending_count"],max_room_area_m2=35,lists_modified=False,
        production_integrated=False,acoustics="not_run",leakage_columns_empty=True,duplicate_source_keys=len(overlaps),
        retained_native_room_circle_not_new_gate=True,source_native_polygon_manifest=str(nf.PREP),
        sources=[str(root/"delivery_all_v5/final_v1"),str(root/"existing_rooms_connectivity_v1")])
    dump(out/"summary.json",summary)
    with (out/"REPORT_zh.md").open("x") as f:
        f.write(f"HM3D最终名单草稿{len(records)}间，全部6–35m²；仅写新草稿，未改现有名单、未接入生产，漏声为空。\n\n"+json.dumps(summary,ensure_ascii=False,indent=2)+"\n\n原生历史<6m²例外、未判定几何单列pending.json；未把它们冒称冻门槛合格。原生≤35的见证与黑区可继承，去飞地改变的块重跑冻结检查；新切块完整见证见来源JSON。原生来源与v5新切来源没有重复。\n")
    return summary

def copy_record(b):
    keys=["id","house","source_region","source_region_id","floor_id","floor_y_m","floor_polygon_xz_m","floor_area_m2","short_side_m","black_fraction","placement_witness","source","origin_list","source_selection","connectivity_certificate"]
    r={k:b.get(k) for k in keys};r["leakage"]=None;return r

def run(root):
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));root=Path(root)
    rooms,summary=native_review(root)
    if not summary["validation_passed"]:raise RuntimeError("native acceptance failed: "+repr(summary["validation_failures"][:20]))
    smy_review(root,rooms);ds=draft(root,rooms,summary)
    with (root/"PROGRESS_zh.md").open("a") as f:f.write("\nNATIVE_FOLLOWUP_READY "+str(root/"existing_rooms_connectivity_v1/review.html")+"\nDRAFT_READY "+str(root/"final_room_list_draft_v1/rooms.csv")+f"；总间数={ds['total_rooms']}；pending={ds['pending']}。\n")
    print("NATIVE_FOLLOWUP_READY",ds,flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",required=True);v=a.parse_args();run(v.root)
