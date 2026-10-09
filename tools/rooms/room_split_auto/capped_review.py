"""Independent cap35 acceptance, immutable v3 comparisons and native-room gallery."""
from __future__ import annotations
import argparse,base64,csv,datetime,html,json,math,resource
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import shapely
from shapely.geometry import Polygon,Point,LineString,shape
from tools.rooms.room_selection.media import polygons
from tools.rooms.room_split_auto import revision_review as rr
dump=rr.dump

def exterior(g):
    return shapely.union_all([Polygon(p.exterior) for p in polygons(g)])

def connectivity(g):
    clean=g.buffer(0);raw=list(polygons(clean));ext=exterior(clean)
    opened=ext.buffer(-.3,join_style=2).buffer(.3,join_style=2)
    return len(raw),len([p for p in polygons(opened) if p.area>1e-8]),len([p for p in polygons(opened) if p.area>=.05])

def disk_check(g,certificate=None):
    ext=exterior(g);centre=(certificate or {}).get("centre_xz_m")
    if centre:
        point=Point(centre);radius=ext.boundary.distance(point) if ext.covers(point) else 0.
    else:
        try:
            line=shapely.maximum_inscribed_circle(ext,tolerance=.002);point=Point(line.coords[0])
            radius=ext.boundary.distance(point) if ext.covers(point) else 0.
        except Exception:return dict(fits=False,radius_m=None,centre_xz_m=None)
    return dict(fits=radius>=1.2-1e-7,radius_m=float(radius),centre_xz_m=[point.x,point.y])

def corridor_wide_parts(g):
    ext=exterior(g)
    restored=ext.buffer(-1.2,join_style=1).buffer(1.2,join_style=1)
    return [p.area for p in polygons(g.intersection(restored)) if p.area>=6-1e-8 and disk_check(p)["fits"]]

def cut_angles(line,axis):
    delta=np.diff(np.asarray(line.coords),axis=0)
    return [abs((math.degrees(math.atan2(v[1],v[0]))-axis+45)%90-45)
            for v in delta if np.linalg.norm(v)>1e-9]

def frame(reg,fid,root):
    reference=reg.get("floor_overheads",{}).get(fid)
    if reference:
        meta_path=Path(reference["metadata_path"]);p=Path(reference["image_path"])
        if meta_path.exists() and p.exists():
            meta=json.loads(meta_path.read_text())
            if "images" not in meta:meta=dict(meta,images=[dict(path=meta["path"],projection=meta["projection"])])
            sensor=next((s for s in meta["images"] if s["path"].endswith("_overview.png")),meta["images"][0])
            return meta,sensor,p,reference
    return rr.frame_for(reg,fid,root)

def table(reg):
    ordered=sorted(reg["blocks"],key=lambda b:(b["decision"]!="retain",-b["floor_area_m2"]))
    out=['<table><tr><th>编号/块</th><th>面积/短边/圆半径 m</th><th>类型/结论</th><th>原因</th></tr>']
    for no,b in enumerate(ordered,1):
        if b["floor_area_m2"]<.5:continue
        r=b.get("inscribed_circle",{}).get("radius_m");rad=f'{r:.2f}' if r is not None else "—"
        why=b.get("discard_reasons") or b.get("unresolved_reasons") or []
        out.append('<tr><td>'+str(no)+' / '+html.escape(b["id"].split("__")[-1])+'</td><td>'+
                   f'{b["floor_area_m2"]:.2f} / {b.get("short_side_m",0):.2f} / '+rad+'</td><td>'+
                   html.escape(str(b.get("room_type"))+"/"+b["decision"])+'</td><td>'+
                   html.escape(", ".join(why))+'</td></tr>')
    out.append('</table>')
    tiny=[b for b in ordered if b["floor_area_m2"]<.5]
    if tiny:out.append(f'<p>小碎面 {len(tiny)} 块 / {sum(b["floor_area_m2"] for b in tiny):.3f} m²；逐块仍在 JSON/CSV。</p>')
    return "".join(out)

def original_native(reg):
    base=dict(reg,cut_lines=[],blocks=[])
    for f in reg["source_geometry"]["floors"]:
        base["blocks"].append(dict(id=reg["house"]+"__"+reg["source_region"]+"__"+f["floor_id"]+"__NATIVE",
            floor_id=f["floor_id"],floor_area_m2=f["floor_area_m2"],floor_polygon_xz_m=f["floor_polygon"],
            short_side_m=f["short_side_m"],decision="unchanged",room_type=reg.get("native_selection",{}).get("room_type"),
            discard_reasons=[],unresolved_reasons=[]))
    return base

STYLE='<style>body{font:16px sans-serif;margin:18px;background:#eee;color:#17212b}article{background:white;margin:18px 0;padding:16px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pair img{width:100%;aspect-ratio:1;object-fit:contain;background:#ddd}table{border-collapse:collapse;width:100%}td,th{padding:5px;border:1px solid #ddd}pre{white-space:pre-wrap}.note{background:#fff4d8;padding:12px}</style>'
LAZY='<script>const observer=new IntersectionObserver(es=>{for(const e of es){if(e.isIntersecting){const i=e.target;i.src=i.dataset.src;delete i.dataset.src;observer.unobserve(i)}}},{rootMargin:"0px",threshold:0});document.querySelectorAll("img[data-src]").forEach(i=>observer.observe(i));</script>'
def embedded(p):return "data:image/jpeg;base64,"+base64.b64encode(p.read_bytes()).decode()

def render_comparisons(root,out,old,new,summary):
    media=out/"media_compare";media.mkdir(exist_ok=False);index=[]
    chosen={name:r for name,r in new.items() if r["requires_split"]}
    page=out/"review_compare_v3_v4.html";native_page=out/"review_native_35_50.html"
    with page.open("x") as f,native_page.open("x") as n:
        for dst,title in [(f,"HM3D v3 / v4（35 m²）"),(n,"原生35–50 m²房间切分（35 m²）")]:
            dst.write('<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+title+'</title>'+STYLE)
            dst.write('<p>v4 上限35 m²；新房的切线只沿主墙两轴，飞地单列，宽房间部分保留检查。房子仍使用整屋原始几何，切分只限定站位；未跑声学。</p>')
            dst.write('<p>每间一色，正中编号与面积；丢弃斜线、未判定网格。左为v3或原生未切，右为v4；图像为真实CPU俯视图，平铺严格懒加载。</p>')
            dst.write('<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
        for name,reg in sorted(chosen.items(),key=lambda x:(bool(x[1].get("native_selection")),x[1]["house"],int(x[1]["source_region"][1:]))):
            native=bool(reg.get("native_selection"));previous=old.get(name,original_native(reg))
            # A source new to the list has no v3 split; show the exact original
            # semantic-ground floor in the same real full-house frame.
            left_label="v3" if name in old else "原生未切"
            row=['<article><h2>'+html.escape(reg["house"]+"/"+reg["source_region"])+
                 f' — {reg["source_floor_area_m2"]:.2f} m²'+(' / 现有名单原生35–50' if native else '')+'</h2>']
            entry=dict(source=name,native_35_50=native,images=[],missing_frames=[])
            for fid in sorted({b["floor_id"] for b in reg["blocks"]}):
                if not any(b["floor_area_m2"]>=.5 for b in reg["blocks"] if b["floor_id"]==fid):continue
                fr=frame(reg,fid,root) or frame(previous,fid,root)
                if fr is None:row.append('<p>'+fid+'：真实图缺失，未补造图。</p>');entry["missing_frames"].append(fid);continue
                row.append('<div class="pair">')
                for label,record in [(left_label,previous),("v4",reg)]:
                    key=Path(name).stem+"__"+fid+"__"+("before" if label!="v4" else "v4")
                    path=media/(key+".jpg");im=rr.overlay(record,fid,fr,label)
                    with path.open("xb") as dst:im.save(dst,format="JPEG",quality=83)
                    row.append('<div><b>'+html.escape(label+" "+fid)+'</b><img loading="lazy" decoding="async" width="900" height="900" data-src="'+embedded(path)+'" alt="'+html.escape(key)+'"></div>')
                    entry["images"].append(dict(floor_id=fid,version=label,path=str(path)))
                row.append('</div>')
            row.append('<p><b>'+left_label+'</b></p>'+table(previous)+'<p><b>v4</b></p>'+table(reg)+'</article>')
            block="".join(row);f.write(block)
            if native:n.write(block)
            index.append(entry)
        f.write(LAZY+'</html>');n.write(LAZY+'</html>')
    dump(out/"overlay_compare_index.json",index)
    with (out/"review.html").open("x") as f:f.write(page.read_text())
    return index

def run(root,delivery_dir="delivery_all_v4"):
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));root=Path(root);out=root/delivery_dir
    if not (out/"completed.json").exists():raise RuntimeError("native run not complete")
    plan=json.loads((root/"delivery_all_v4/revision_plan.json").read_text());cap=plan["max_room_area_m2"]
    old={p.name:json.loads(p.read_text()) for p in (root/"delivery_all_v3/final_v1/regions").glob("*.json")}
    new={p.name:json.loads(p.read_text()) for p in (out/"regions").glob("*.json")}
    failed_sources=[name for name,reg in new.items() if reg.get("traceback")]
    if failed_sources:raise RuntimeError("native implementation exceptions require immutable repair before review: "+repr(failed_sources))
    failures=[];kept=[];unresolved=[];cuts=[];stairs=[];partition=[];preserved=[];corridors=[]
    for name,r in new.items():
        if not r["requires_split"]:
            ok=r==old.get(name);preserved.append(dict(source=name,unchanged=ok))
            if not ok:failures.append(dict(source=name,check="PROTECTED_ORIGINAL_CHANGED"))
            continue
        grouped=defaultdict(list)
        for b in r["blocks"]:
            g=shape(b["floor_polygon_xz_m"]);grouped[b["floor_id"]].append(g)
            if b["decision"]=="unresolved":unresolved.append(dict(id=b["id"],area_m2=b["floor_area_m2"],reasons=b.get("unresolved_reasons")))
            if b["decision"]=="discard" and any("CORRIDOR" in s for s in b.get("discard_reasons",[])):
                wide=corridor_wide_parts(g);corridors.append(dict(id=b["id"],area_m2=g.area,wide_ge6_disk_parts_m2=wide))
                if wide:failures.append(dict(id=b["id"],check="BROAD_ROOM_DISCARDED_AS_CORRIDOR",areas_m2=wide))
            if b["decision"]!="retain":continue
            raw,n,n_owner=connectivity(g);circle=disk_check(g,b.get("inscribed_circle"))
            outside=[]
            for key in ["camera_m","source_1_m","source_2_m"]:
                p=b["placement_witness"].get(key)
                if not p or not g.buffer(1e-7).covers(Point(p[0],p[2])):outside.append(key)
            checks=dict(area=6-1e-8<=g.area<=cap+1e-8,area_matches=abs(g.area-b["floor_area_m2"])<1e-6,
                        short_side=b["short_side_m"]>=2.4-1e-8,raw_one_polygon=raw==1,filled_mitre_one_component=n==1,
                        owner_ge0_05_one_component=n_owner==1,disk_2_4=circle["fits"],
                        black=b["black_fraction"] is not None and b["black_fraction"]<=.15,
                        witness=b["placement_witness"].get("found") is True,witness_inside=not outside)
            row=dict(id=b["id"],area_m2=g.area,short_side_m=b["short_side_m"],circle=circle,
                     raw_components=raw,opened_components=n,owner_opened_components=n_owner,checks=checks)
            kept.append(row)
            for c,ok in checks.items():
                if not ok:failures.append(dict(id=b["id"],check=c))
        for floor in r["source_geometry"]["floors"]:
            fid=floor["floor_id"];g=shape(floor["floor_polygon"]);pieces=grouped[fid];union=shapely.union_all(pieces)
            err=g.symmetric_difference(union).area;overlap=sum(p.area for p in pieces)-union.area
            partition.append(dict(source=name,floor_id=fid,area_m2=g.area,symmetric_difference_m2=err,overlap_m2=max(0.,overlap)))
            if err>1e-6 or overlap>1e-6:failures.append(dict(source=name,floor_id=fid,check="AREA_PARTITION"))
        for ex in r.get("stair_exact_exclusion",[]):
            actual=shapely.union_all([shape(b["floor_polygon_xz_m"]) for b in r["blocks"] if b["floor_id"]==ex["floor_id"] and "STAIRS" in b.get("discard_reasons",[])])
            exact=shape(ex["geometry_xz_m"]);err=actual.symmetric_difference(exact).area
            stairs.append(dict(source=name,floor_id=ex["floor_id"],exact_semantic_area_m2=exact.area,discarded_area_m2=actual.area,difference_m2=err,margin_m=ex["periphery_expansion_m"]))
            if err>1e-6 or ex["periphery_expansion_m"]!=0:failures.append(dict(source=name,floor_id=ex["floor_id"],check="STAIR_EXACT_ONLY"))
        for c in r["cut_lines"]:
            line=LineString(c["line_xz_m"]);axis=r["wall_axes"][c["floor_id"]]["primary_deg"];angles=cut_angles(line,axis)
            cuts.append(dict(source_region=r["house"]+"/"+r["source_region"],floor_id=c["floor_id"],cut_id=c["id"],type=c["type"],
                stage=c.get("stage"),segment_count=len(line.coords)-1,wall_axis_deg=axis,max_axis_angle_error_deg=max(angles,default=0),
                active=c.get("active_in_final_partition"),furniture_intersection_length_m=c["furniture_intersection_length_m"],
                furniture_intersection_strip_area_m2=c["furniture_intersection_area_m2"],
                strip_width_m=c["intersection_area_band_width_m"],fallback_crosses_furniture=c["fallback_crosses_furniture"],
                furniture_intersections=c.get("furniture_intersections")))
            if len(line.coords)-1>3 or max(angles,default=0)>1e-6:failures.append(dict(source=name,cut_id=c["id"],check="CUT_AXIS_OR_SEGMENTS"))
    before=rr.stats(list(old.values()));after=rr.stats(list(new.values()))
    original_keys={x["source"] for x in plan["scope_manifest"] if x["origin"]=="original_gt50"}
    after_same=rr.stats([new[k] for k in original_keys])
    native_keys={x["source"] for x in plan["scope_manifest"] if x["origin"]=="existing_training_native_gt35"}
    native_stats=rr.stats([new[k] for k in native_keys])
    selected=json.loads((root/"size_gallery_30_50_v1/selection_rows_v1.json").read_text());train_houses={r["house"] for r in selected}
    retained_houses={r["house"] for r in new.values() for b in r["blocks"] if r["requires_split"] and b["decision"]=="retain"}
    summary=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),max_room_area_m2=cap,
        scope=dict(original_gt50_sources=len(original_keys),native_35_50_sources=len(native_keys),native_35_40=plan["native_35_40"],
                   native_40_50=plan["native_40_50"],total_recut=len(original_keys|native_keys)),
        before_v3=before,after_v4=after,after_v4_same_original50=after_same,after_v4_native35_50=native_stats,
        retained_house_count=len(retained_houses),already_existing_training_house_count=len(retained_houses&train_houses),
        lists_modified=False,cpu_only=True,acoustics="not_run",
        visibility_policy=plan["visibility_policy"],scope_note="278 source records include174 untouched prior reference sources; new-retained metrics use104 authorized recut sources",
        global_minimum_partition_unverified=True)
    validation=dict(passed=not failures,failures=failures,retained_new_rooms=len(kept),
        retained_area_min_m2=min((r["area_m2"] for r in kept),default=None),retained_area_max_m2=max((r["area_m2"] for r in kept),default=None),
        retained_short_min_m=min((r["short_side_m"] for r in kept),default=None),
        retained_disconnected=sum(r["opened_components"]!=1 or r["raw_components"]!=1 for r in kept),
        owner_disconnected=sum(r["owner_opened_components"]!=1 for r in kept),
        retained_without_disk=sum(not r["circle"]["fits"] for r in kept),
        cut_count=len(cuts),active_cut_count=sum(bool(c["active"]) for c in cuts),
        max_cut_segments=max((c["segment_count"] for c in cuts),default=0),
        max_wall_angle_error_deg=max((c["max_axis_angle_error_deg"] for c in cuts),default=0),
        designed_cuts_crossing_furniture=sum(c["furniture_intersection_length_m"]>1e-7 for c in cuts),
        stairs_area_m2=sum(r["discarded_area_m2"] for r in stairs),exact_stairs_area_m2=sum(r["exact_semantic_area_m2"] for r in stairs),
        stair_difference_m2=sum(r["difference_m2"] for r in stairs),
        discarded_corridors_with_broad_ge6_disk_part=sum(bool(r["wide_ge6_disk_parts_m2"]) for r in corridors),
        untouched_prior_sources=len(preserved),all_untouched_preserved=all(r["unchanged"] for r in preserved),
        retained_checks=kept,stair_checks=stairs,partition_checks=partition,discarded_corridor_checks=corridors,
        join_style="mitre; Claude checker reference claude_check_split_delivery_v2.py; raw islands independently separated",
        room_circle_basis="filled exterior geometric shape, exact centre-to-boundary >=1.2m; raw holes kept in area",
        cut_segment_note="design physical polyline <=3 axis segments; raw scan holes can break one segment into multiple visible interface fragments")
    dump(out/"summary.json",summary);dump(out/"validation.json",validation);dump(out/"unresolved.json",unresolved);dump(out/"cut_furniture_intersections.json",cuts)
    for filename,rows,fields in [
      ("cut_furniture_intersections.csv",cuts,[k for k in cuts[0] if k!="furniture_intersections"] if cuts else ["cut_id"]),
      ("rooms_summary.csv",[b for r in new.values() for b in r["blocks"]],
       ["id","house","source_region","floor_id","floor_y_m","floor_area_m2","short_side_m","body_width_m_proxy","nav_walkable_area_m2","black_fraction","decision","room_type","discard_reasons","unresolved_reasons","visibility_coverage_fraction","connectivity_core_count"])]:
        with (out/filename).open("x",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    index=render_comparisons(root,out,old,new,summary)
    dump(out/"html_validation.json",dict(compare_sources=len(index),native_sources=sum(x["native_35_50"] for x in index),
        missing_frames=[r for r in index if r["missing_frames"]],image_count=sum(len(r["images"]) for r in index),
        embedded=True,strict_lazy_initial_src_absent=True))
    report=f"v4按35m²和主墙两轴重新切分；验收通过={validation['passed']}，未判定明确单列，未跑声学。\n\n"
    report+=f"范围：旧50大来源+现有名单54原生35–50来源=104；54中35–40=26、40–50=28，278来源记录中174旧参考原样保存。数据出处revision_plan.json；未改现有名单。\n\n"
    report+=f"全部v4新保留{after['retained_new_rooms']}间 / {after['retained_new_area_m2']:.3f}m²；丢弃{after['discarded_blocks']}块 / {after['discarded_area_m2']:.3f}m²；未判定{after['unresolved_blocks']}块 / {after['unresolved_area_m2']:.3f}m²。summary.json。\n\n"
    report+=f"同50来源v3→v4：保留{before['retained_new_rooms']}→{after_same['retained_new_rooms']}间，面积{before['retained_new_area_m2']:.3f}→{after_same['retained_new_area_m2']:.3f}m²；未判定{before['unresolved_blocks']}→{after_same['unresolved_blocks']}。新增54原生房输出保留{native_stats['retained_new_rooms']}间/{native_stats['retained_new_area_m2']:.3f}m²。summary.json。\n\n"
    report+=f"机器验收：保留面积{validation['retained_area_min_m2']}–{validation['retained_area_max_m2']}m²，最短边最低{validation['retained_short_min_m']}m；飞地{validation['retained_disconnected']}、无2.4m圆{validation['retained_without_disk']}；切线{len(cuts)}（active {validation['active_cut_count']}），最大{validation['max_cut_segments']}段，主墙方向最大偏差{validation['max_wall_angle_error_deg']:.10g}°；楼梯丢弃{validation['stairs_area_m2']:.9f}m²，直接语义掩膜{validation['exact_stairs_area_m2']:.9f}m²，差{validation['stair_difference_m2']:.9g}m²；丢弃走廊仍含宽房部分{validation['discarded_corridors_with_broad_ge6_disk_part']}。逐条出处validation.json。\n\n"
    report+="内部洞只在连通、窄口和形状圆判据中填掉；面积、多边形、黑区、可走面与摆放见证均用原始带洞地面。2.4m圆只挑新切块，不淘汰原生≤35房。直角收放对齐Claude独立检查，修复v3圆角收放造成的2间假连通。每条设计折线≤3个轴向直段；扫描洞可能使实际可见接口断成多个共线片段，保留原始接口供检查，未把断片误报为新增折线。\n\n"
    report+="owner不新增80%地面额外切分/准入门槛：≤35原生不切，完成≤35的块不因可见覆盖不足再切或拒绝；>35处理仍先楼层、门/窄口，再在主墙可行切线中用整屋网格可见性辅助选择。可见性记录为诊断，未声称全部房间覆盖80%。全局最少房间、家具相交全局最小值未验证。\n\n"
    report+=f"家具逐条精确相交长度及0.25m条带面积见cut_furniture_intersections.csv/json，设计相交{validation['designed_cuts_crossing_furniture']}条；独立全场景实际接口复量见cut_scene_furniture_audit_v1/。切线候选先避家具，无零相交可行解时取候选内最小相交，未声称数学全局最小。\n\n"
    report+="同50来源丢弃原因面积：\n\n|原因|v3 m²|v4 m²|\n|---|---:|---:|\n"
    for reason in sorted(set(before["discard_primary_reason_area_m2"])|set(after_same["discard_primary_reason_area_m2"])):
        report+=f"|{reason}|{before['discard_primary_reason_area_m2'].get(reason,0):.3f}|{after_same['discard_primary_reason_area_m2'].get(reason,0):.3f}|\n"
    report+="\n未判定（非微小碎面）：\n\n"
    for b in unresolved:
        if b["area_m2"]>=.5:report+=f"- {b['id']}：{b['area_m2']:.3f}m²；{', '.join(b['reasons'] or [])}。\n"
    report+="\n真实CPU图复用前轮整屋缓存，所有旧产物只读；v3/v4对照review_compare_v3_v4.html，54原生房组review_native_35_50.html，图片内嵌严格懒加载。未跑声学、未接入生产、未改名单、未重复留出IoU或调参。\n"
    with (out/"REPORT_zh.md").open("x") as f:f.write(report)
    with (root/"PROGRESS_zh.md").open("a") as f:f.write("\nV4_READY "+str(out/"review_compare_v3_v4.html")+f"；验收={validation['passed']}；新保留={len(kept)}；未判定块={after['unresolved_blocks']}。\n")
    print("V4_REPORT_DONE",validation["passed"],"KEPT",len(kept),"UNRESOLVED",len(unresolved),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",required=True);a.add_argument("--delivery-dir",default="delivery_all_v4");v=a.parse_args();run(v.root,v.delivery_dir)
