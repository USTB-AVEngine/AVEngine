"""Cap35 v5 acceptance with seam/native-navmesh connectivity and immutable comparisons."""
from __future__ import annotations
import argparse,base64,csv,datetime,html,json,math,resource
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import shapely
from shapely.geometry import Polygon,Point,LineString,shape
from tools.rooms.room_selection.media import polygons
from tools.rooms.room_split_auto import revision_review as rr
from tools.rooms.room_split_auto.seam_connectivity import Connectivity,exterior as safe_exterior
from tools.rooms.room_selection.measurements import navmesh_triangles
from tools.rooms.room_split_auto.pipeline import load_native,nav_scope_at
dump=rr.dump

def exterior(g):
    return safe_exterior(g)

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

def render_comparisons(root,out,old,new,summary, v4=None):
    media=out/"media_compare";media.mkdir(exist_ok=False);index=[]
    chosen={name:r for name,r in new.items() if r["requires_split"]}
    page=out/"review_compare_v3_v5.html";native_page=out/"review_native_35_50.html"
    with page.open("x") as f,native_page.open("x") as n:
        for dst,title in [(f,"HM3D v3 / v5（35 m²）"),(n,"原生35–50 m²房间切分（35 m²）")]:
            dst.write('<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+title+'</title>'+STYLE)
            dst.write('<p>v5 上限35 m²；优先主墙两轴，无可行解才在±15°内或顺墙两段切，飞地按接缝与navmesh新规则判断。房子仍使用整屋原始几何，切分只限定站位；未跑声学。</p>')
            dst.write('<p>每间一色，正中编号与面积；丢弃斜线、未判定网格。左为v3或原生未切，右为v5；图像为真实CPU俯视图，平铺严格懒加载。</p>')
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
                for label,record in [(left_label,previous),("v5",reg)]:
                    key=Path(name).stem+"__"+fid+"__"+("before" if label!="v5" else "v5")
                    path=media/(key+".jpg");im=rr.overlay(record,fid,fr,label)
                    with path.open("xb") as dst:im.save(dst,format="JPEG",quality=83)
                    row.append('<div><b>'+html.escape(label+" "+fid)+'</b><img loading="lazy" decoding="async" width="900" height="900" data-src="'+embedded(path)+'" alt="'+html.escape(key)+'"></div>')
                    entry["images"].append(dict(floor_id=fid,version=label,path=str(path)))
                row.append('</div>')
            row.append('<p><b>'+left_label+'</b></p>'+table(previous)+'<p><b>v5</b></p>'+table(reg)+'</article>')
            block="".join(row);f.write(block)
            if native:n.write(block)
            index.append(entry)
        f.write(LAZY+'</html>');n.write(LAZY+'</html>')
    dump(out/"overlay_compare_index.json",index)
    with (out/"review.html").open("x") as f:f.write(page.read_text())
    return index

def run(root,delivery_dir="delivery_all_v5"):
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));root=Path(root);out=root/delivery_dir
    if not (out/"completed.json").exists():raise RuntimeError("native run not complete")
    plan=json.loads((out/"revision_plan.json").read_text());cap=plan["max_room_area_m2"]
    old={p.name:json.loads(p.read_text()) for p in (root/"delivery_all_v3/final_v1/regions").glob("*.json")}
    new={p.name:json.loads(p.read_text()) for p in (out/"regions").glob("*.json")}
    failed_sources=[name for name,reg in new.items() if reg.get("traceback")]
    if failed_sources:raise RuntimeError("v5 geometry/implementation exceptions require immutable repair before review: "+repr(failed_sources))
    hs=load_native();nav_cache={}
    failures=[];kept=[];unresolved=[];cuts=[];stairs=[];partition=[];preserved=[];corridors=[]
    for name,r in new.items():
        if not r["requires_split"]:
            ok=r==old.get(name);preserved.append(dict(source=name,unchanged=ok))
            if not ok:failures.append(dict(source=name,check="PROTECTED_ORIGINAL_CHANGED"))
            continue
        grouped=defaultdict(list);ctx_by_floor={}
        house=r["house"]
        if house not in nav_cache:nav_cache[house]=navmesh_triangles(hs,Path(r["source_geometry"]["navmesh_source"]))[1:]
        np_,ny=nav_cache[house]
        for floor in r["source_geometry"]["floors"]:
            fg=shape(floor["floor_polygon"])
            ctx_by_floor[floor["floor_id"]]=Connectivity(nav_scope_at(np_,ny,floor["floor_y_m"],exterior(fg).buffer(.3),plan["parameters"]))

        for b in r["blocks"]:
            g=shape(b["floor_polygon_xz_m"]);grouped[b["floor_id"]].append(g)
            ctx=ctx_by_floor[b["floor_id"]];cert=b.get("connectivity_certificate",{})
            if cert.get("partition_cell_xz_m"):ctx.masks[g.wkb]=shape(cert["partition_cell_xz_m"])
            ext=ctx.envelope(g)
            if b["decision"]=="unresolved":unresolved.append(dict(id=b["id"],area_m2=b["floor_area_m2"],reasons=b.get("unresolved_reasons")))
            if b["decision"]=="discard" and any("CORRIDOR" in s for s in b.get("discard_reasons",[])):
                restored=ext.buffer(-1.2,join_style=1).buffer(1.2,join_style=1)
                wide=[q.area for q in polygons(g.intersection(restored)) if q.area>=6-1e-8 and disk_check(q)["fits"]];corridors.append(dict(id=b["id"],area_m2=g.area,wide_ge6_disk_parts_m2=wide))
                if wide:failures.append(dict(id=b["id"],check="BROAD_ROOM_DISCARDED_AS_CORRIDOR",areas_m2=wide))
            if b["decision"]!="retain":continue
            record=ctx.record(g);raw=record["raw_part_count"];n=ctx.count(g);n_owner=record["opened_count"]
            centre=b.get("inscribed_circle",{}).get("centre_xz_m");point=Point(centre) if centre else None
            radius=ext.boundary.distance(point) if point is not None and ext.covers(point) else 0.
            circle=dict(fits=radius>=1.2-1e-7,radius_m=radius,centre_xz_m=centre)

            outside=[]
            for key in ["camera_m","source_1_m","source_2_m"]:
                p=b["placement_witness"].get(key)
                if not p or not g.buffer(1e-7).covers(Point(p[0],p[2])):outside.append(key)
            checks=dict(area=6-1e-8<=g.area<=cap+1e-8,area_matches=abs(g.area-b["floor_area_m2"])<1e-6,
                        short_side=b["short_side_m"]>=2.4-1e-8,
                        direct_walkable_group=record["direct_group_count"]==1,filled_mitre_one_component=n==1,
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
            if len(line.coords)-1>3 or max(angles,default=0)>15+1e-6:failures.append(dict(source=name,cut_id=c["id"],check="CUT_AXIS_OR_SEGMENTS"))
    before=rr.stats(list(old.values()));after=rr.stats(list(new.values()))
    v4={p.name:json.loads(p.read_text()) for p in (root/"delivery_all_v4/final_v1/regions").glob("*.json")}
    recovery=[]
    for name,r in new.items():
        if not r["requires_split"] or name not in v4:continue
        prior=v4[name];saved=defaultdict(float);destinations=set()
        for b in r["blocks"]:
            if b["decision"]!="retain":continue
            g=shape(b["floor_polygon_xz_m"])
            for prev in prior["blocks"]:
                if abs(prev["floor_y_m"]-b["floor_y_m"])>.3 or prev["decision"] not in ("discard","unresolved"):continue
                area=g.intersection(shape(prev["floor_polygon_xz_m"])).area
                if area>1e-6:saved[prev["decision"]]+=area;destinations.add(b["id"])
        if saved:recovery.append(dict(source=name,discarded_ground_recovered_m2=saved["discard"],previously_unresolved_ground_now_retained_m2=saved["unresolved"],retained_destination_rooms=len(destinations)))
    baseline_failures=[dict(source=p.name,reasons=json.loads(p.read_text()).get("unresolved_reasons"),area_m2=json.loads(p.read_text())["source_floor_area_m2"]) for p in (root/"delivery_all_v4/regions").glob("*.json") if json.loads(p.read_text()).get("traceback")]
    dump(out/"v4_failure_diagnosis.json",dict(metadata_exception_sources=len(baseline_failures),metadata_exception_area_m2=sum(x["area_m2"] for x in baseline_failures),fixed_sensor_span=True,exceptions=baseline_failures,
        corrected_v4_unresolved_sources=[dict(source=name,blocks=[dict(id=b["id"],area_m2=b["floor_area_m2"],reasons=b.get("unresolved_reasons")) for b in r["blocks"] if b["decision"]=="unresolved"]) for name,r in v4.items() if r["requires_split"] and any(b["decision"]=="unresolved" for b in r["blocks"])],
        v5_remaining_implementation_exceptions=failed_sources))
    dump(out/"v4_v5_ground_recovery.json",recovery)
    original_keys={x["source"] for x in plan["scope_manifest"] if x["origin"]=="original_gt50"}
    after_same=rr.stats([new[k] for k in original_keys])
    native_keys={x["source"] for x in plan["scope_manifest"] if x["origin"]=="existing_training_native_gt35"}
    native_stats=rr.stats([new[k] for k in native_keys])
    selected=json.loads((root/"size_gallery_30_50_v1/selection_rows_v1.json").read_text());train_houses={r["house"] for r in selected}
    retained_houses={r["house"] for r in new.values() for b in r["blocks"] if r["requires_split"] and b["decision"]=="retain"}
    summary=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),max_room_area_m2=cap,
        scope=dict(original_gt50_sources=len(original_keys),native_35_50_sources=len(native_keys),native_35_40=plan["native_35_40"],
                   native_40_50=plan["native_40_50"],total_recut=len(original_keys|native_keys)),
        before_v3=before,before_v4=rr.stats(list(v4.values())),after_v5=after,after_v5_same_original50=after_same,after_v5_native35_50=native_stats,
        old_v4_discarded_ground_recovered_m2=sum(x["discarded_ground_recovered_m2"] for x in recovery),
        old_v4_discarded_sources_recovered=sum(x["discarded_ground_recovered_m2"]>1e-6 for x in recovery),
        old_v4_unresolved_ground_now_retained_m2=sum(x["previously_unresolved_ground_now_retained_m2"] for x in recovery),
        recovery_attribution="same-world-ground intersections vs corrected v4, all algorithm changes combined; not attributed solely to connectivity",
        retained_house_count=len(retained_houses),already_existing_training_house_count=len(retained_houses&train_houses),
        lists_modified=False,cpu_only=True,acoustics="not_run",
        visibility_policy=plan["visibility_policy"],scope_note="278 source records include174 untouched prior reference sources; new-retained metrics use104 authorized recut sources",
        global_minimum_partition_unverified=True)
    validation=dict(passed=not failures,failures=failures,retained_new_rooms=len(kept),
        retained_area_min_m2=min((r["area_m2"] for r in kept),default=None),retained_area_max_m2=max((r["area_m2"] for r in kept),default=None),
        retained_short_min_m=min((r["short_side_m"] for r in kept),default=None),
        retained_disconnected=sum(r["opened_components"]!=1 for r in kept),
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
        join_style="filled joined exterior; seams<=.05 and local navmesh in expanded0.3m; mitre opening0.6m; original raw parts may be multiple",
        room_circle_basis="filled joined exterior with authorized seam/local-nav support, exact centre-to-boundary>=1.2m; raw ground unchanged",
        cut_segment_note="design physical polyline <=3 axis segments; raw scan holes can break one segment into multiple visible interface fragments")
    dump(out/"summary.json",summary);dump(out/"validation.json",validation);dump(out/"unresolved.json",unresolved);dump(out/"cut_furniture_intersections.json",cuts)
    for filename,rows,fields in [
      ("cut_furniture_intersections.csv",cuts,[k for k in cuts[0] if k!="furniture_intersections"] if cuts else ["cut_id"]),
      ("rooms_summary.csv",[b for r in new.values() for b in r["blocks"]],
       ["id","house","source_region","floor_id","floor_y_m","floor_area_m2","short_side_m","body_width_m_proxy","nav_walkable_area_m2","black_fraction","decision","room_type","discard_reasons","unresolved_reasons","visibility_coverage_fraction","connectivity_core_count"])]:
        with (out/filename).open("x",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    index=render_comparisons(root,out,old,new,summary)
    compare_v4=out/"review_compare_v4_v5.html"
    with compare_v4.open("x") as dst:
        dst.write('<!doctype html><html lang="zh"><meta charset="utf-8"><title>v4/v5对照</title>'+STYLE+'<p>v5按35m²和新接缝/navmesh连通规则完成；左v4右v5，整屋原图，丢弃斜线、未判定网格。</p>')
        for name,r in sorted(new.items()):
            if not r["requires_split"]:continue
            prior=v4.get(name,original_native(r));dst.write('<article><h2>'+html.escape(r["house"]+"/"+r["source_region"])+'</h2>')
            for fid in sorted({b["floor_id"] for b in r["blocks"] if b["floor_area_m2"]>=.5}):
                fr=frame(r,fid,root)
                if fr is None:continue
                dest=out/"media_compare"/(Path(name).stem+"__"+fid+"__v4.jpg")
                with dest.open("xb") as f:rr.overlay(prior,fid,fr,"v4").save(f,format="JPEG",quality=83)
                right=out/"media_compare"/(Path(name).stem+"__"+fid+"__v5.jpg")
                dst.write('<div class="pair"><img loading="lazy" width="900" height="900" data-src="'+embedded(dest)+'"><img loading="lazy" width="900" height="900" data-src="'+embedded(right)+'"></div>')
            dst.write(table(prior)+table(r)+'</article>')
        dst.write(LAZY+'</html>')
    dump(out/"html_validation.json",dict(compare_sources=len(index),native_sources=sum(x["native_35_50"] for x in index),
        missing_frames=[r for r in index if r["missing_frames"]],image_count=sum(len(r["images"]) for r in index),
        embedded=True,strict_lazy_initial_src_absent=True))
    report=f"v5新切房验收通过={validation['passed']}；保留{after['retained_new_rooms']}间，未判定{after['unresolved_blocks']}块；不跑声学，不改现有名单。\n\n"
    report+=f"范围：旧50大来源+现有名单54原生35–50=104，278记录含174旧参考；原生35–40=26，40–50=28。来源revision_plan.json。\n\n"
    report+=f"v3→v5同50来源：保留{before['retained_new_rooms']}→{after_same['retained_new_rooms']}，面积{before['retained_new_area_m2']:.3f}→{after_same['retained_new_area_m2']:.3f}m²；v4→v5全104来源：保留{summary['before_v4']['retained_new_rooms']}→{after['retained_new_rooms']}，面积{summary['before_v4']['retained_new_area_m2']:.3f}→{after['retained_new_area_m2']:.3f}m²。summary.json。\n\n"
    report+=f"丢弃{after['discarded_blocks']}块/{after['discarded_area_m2']:.3f}m²；未判定{after['unresolved_blocks']}块/{after['unresolved_area_m2']:.3f}m²。各原因面积见summary.json，逐块见rooms_summary.csv、unresolved.json。\n\n"
    report+=f"机器验收：面积{validation['retained_area_min_m2']:.6f}–{validation['retained_area_max_m2']:.6f}m²；短边最小{validation['retained_short_min_m']:.6f}m；按新接缝/navmesh/.6m规则不连通{validation['retained_disconnected']}间；无2.4m圆{validation['retained_without_disk']}；切线{len(cuts)}条，最多{validation['max_cut_segments']}段，最大偏墙角{validation['max_wall_angle_error_deg']:.6f}°。楼梯丢弃{validation['stairs_area_m2']:.9f}m²=直接语义楼梯{validation['exact_stairs_area_m2']:.9f}m²；走廊丢弃中含≥6m²且容2.4m圆的宽部{validation['discarded_corridors_with_broad_ge6_disk_part']}。validation.json逐项。\n\n"
    report+="原始地面面片、内部洞、面积和黑区保持原量法；连通/窄口/挑切线形状先填洞。距离≤.05m接缝直接合组；否则仅两块各扩.3m后与同楼层navmesh求交的局部直接通道可连接，不能借整屋远路；合组外轮廓再做.6m收放。虚拟接缝/导航桥只作连通与形状判断，不增加地面面积。独立验收重读原navmesh并使用保存的切分单元边界，未将原始MultiPolygon部分数直接当飞地。\n\n"
    report+="切线先尝试主墙两轴；轴向直线无满足两侧2.4m圆的方案，才允许±15°直线，再尝试顺墙两段折线；局部走廊用有限墙到墙轴向线。无法构造的宽块列未判定，不整块丢。每条设计线与家具精确交长及.25m条带面积在cut_furniture_intersections.csv；全场景最终实际接口家具审计在cut_scene_furniture_audit_v1。设计线被洞截断产生的共线片段不当额外转折。候选集内最少相交、全局最优未验证。\n\n"
    report+=f"v4旧根目录38个元数据异常来自新图缓存缺span_m，已在不可覆盖的final_v1修复；本轮v5再按新规则全部重跑。修正v4剩余几何未判定→v5的具体变化见v4_failure_diagnosis.json。v4已丢地面追回{summary['old_v4_discarded_ground_recovered_m2']:.3f}m²/{summary['old_v4_discarded_sources_recovered']}来源；另有旧未判定{summary['old_v4_unresolved_ground_now_retained_m2']:.3f}m²转保留，不能把后者说成曾丢弃；v4_v5_ground_recovery.json，统计包含本轮全部算法变化，并非仅归因接缝。\n\n"
    report+="owner取消额外80%地面切分门槛，原生≤35不按面积切，也不新增圆/可见性准入；>35先楼层、门/窄口，再按面积构造，整屋视线仅作候选排序/合并诊断。未重复留出IoU、未调冻门槛、未跑声学。全局最少房间未验证。图片复用真实CPU整屋渲染，保留房间外内容，图片内嵌严格懒加载。\n\n"
    report+="未判定块（≥.5m²）：\n\n"
    for row in unresolved:
        if row["area_m2"]>=.5:report+=f"- {row['id']}：{row['area_m2']:.3f}m²；{', '.join(row['reasons'] or [])}。\n"
    with (out/"REPORT_zh.md").open("x") as f:f.write(report)
    with (root/"PROGRESS_zh.md").open("a") as f:f.write( "\nV5_READY "+str(out/"review_compare_v3_v5.html")+f"；验收={validation['passed']}；新保留={len(kept)}；未判定块={after['unresolved_blocks']}。\n")
    print("V5_REPORT_DONE",validation["passed"],"KEPT",len(kept),"UNRESOLVED",len(unresolved),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",required=True);a.add_argument("--delivery-dir",default="delivery_all_v5");v=a.parse_args();run(v.root,v.delivery_dir)
