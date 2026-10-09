"""Independent v3 geometry acceptance and flat v2/v3 embedded review."""
from __future__ import annotations
from pathlib import Path
import argparse,base64,csv,datetime,html,io,json,math,resource
from collections import Counter,defaultdict
import numpy as np
import shapely
from shapely.geometry import shape,Polygon,LineString,GeometryCollection
from PIL import Image,ImageDraw,ImageFont
from tools.rooms.room_selection.media import polygons
from tools.rooms.room_split_auto.camera_view_proof import project
PALETTE=[(31,119,230),(240,120,30),(40,170,90),(150,70,210),(230,190,0),(0,180,200),(220,60,150),(140,90,40),(100,150,30),(60,60,200)]
FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

def dump(p,d):
    with Path(p).open("x") as f:json.dump(d,f,ensure_ascii=False,allow_nan=False,indent=2)

def exterior(g):
    return shapely.union_all([Polygon(p.exterior) for p in polygons(g)])

def independent_connectivity(g):
    raw=exterior(g)
    parts=list(polygons(raw))
    if len(parts)!=1:return len(parts)
    return len([p for p in polygons(raw.buffer(-.3,quad_segs=8).buffer(.3,quad_segs=8)) if p.area>1e-8])

def stats(regions):
    blocks=[b for r in regions if r["requires_split"] for b in r["blocks"]];kept=[b for b in blocks if b["decision"]=="retain"];reason_area=defaultdict(float);primary=defaultdict(float);counts=Counter()
    for b in blocks:
        if b["decision"]!="discard":continue
        reasons=b.get("discard_reasons") or ["UNSPECIFIED"]
        primary[reasons[0]]+=b["floor_area_m2"]
        for reason in reasons:reason_area[reason]+=b["floor_area_m2"];counts[reason]+=1
    return dict(source_regions=len(regions),large_source_regions=sum(r["requires_split"] for r in regions),unchanged_source_regions=sum(not r["requires_split"] for r in regions),
        retained_new_rooms=len(kept),retained_new_area_m2=sum(b["floor_area_m2"] for b in kept),split_blocks=len(blocks),discarded_blocks=sum(b["decision"]=="discard" for b in blocks),discarded_area_m2=sum(b["floor_area_m2"] for b in blocks if b["decision"]=="discard"),
        unresolved_blocks=sum(b["decision"]=="unresolved" for b in blocks),unresolved_source_regions=sum(any(b["decision"]=="unresolved" for b in r["blocks"]) for r in regions),
        unresolved_area_m2=sum(b["floor_area_m2"] for b in blocks if b["decision"]=="unresolved"),discard_primary_reason_area_m2=dict(sorted(primary.items())),discard_reason_area_m2_inclusive=dict(sorted(reason_area.items())),discard_reason_counts_inclusive=dict(sorted(counts.items())))

def hatch(size,cross=False):
    im=Image.new("RGBA",size);d=ImageDraw.Draw(im);w,h=size
    for k in range(-h,w,14):
        d.line([(k,0),(k+h,h)],fill=(35,35,35,215),width=3)
        if cross:d.line([(k,h),(k+h,0)],fill=(35,35,35,215),width=3)
    return im

def frame_for(reg,fid,root):
    ref=reg.get("floor_overheads",{}).get(fid)
    attempts=[ref] if ref else []
    key=reg["house"]+"__"+reg["source_region"]+"__"+fid
    for directory in ["overhead_cpu_render_v2","overhead_cpu_render_fallback_v1","overhead_cpu_atlas_v1","overhead_cpu_atlas_v3"]:
        p=root/directory/(key+".json")
        if p.exists():attempts.append(dict(metadata_path=str(p),image_path=str(p.with_suffix(".png"))))
    for reference in attempts:
        if not reference:continue
        p=Path(reference["metadata_path"]);impath=Path(reference["image_path"])
        if not p.exists() or not impath.exists():continue
        meta=json.loads(p.read_text());sensor=next((x for x in meta["images"] if x["path"].endswith("_overview.png")),meta["images"][0])
        if "size_px" not in meta:
            with Image.open(impath) as image:meta["size_px"]=list(image.size)
        return meta,sensor,impath,reference
    return None

def overlay(reg,fid,frame,version):
    meta,sensor,p,_=frame;base=Image.open(p).convert("RGBA");w,h=base.size;fy=meta["floor_y_m"]
    blocks=sorted(reg["blocks"],key=lambda b:(b["decision"]!="retain",-b["floor_area_m2"]))
    numbering={b["id"]:i+1 for i,b in enumerate(blocks)}
    colors={b["id"]:PALETTE[i%len(PALETTE)] for i,b in enumerate(blocks) if b["decision"] in ("retain","unchanged")}
    def prj(coords):
        a=np.asarray(coords,float);xyz=np.c_[a[:,0],np.full(len(a),fy),a[:,1]]
        px,_=project(xyz,np.asarray(sensor["projection"]),np.asarray(meta["view"]),w,h)
        return [tuple(x) for x in px]
    over=Image.new("RGBA",base.size);d=ImageDraw.Draw(over);drop=Image.new("L",base.size);unknown=Image.new("L",base.size)
    fl=[b for b in blocks if b["floor_id"]==fid]
    for b in fl:
        g=shape(b["floor_polygon_xz_m"])
        target=d if b["id"] in colors else ImageDraw.Draw(unknown if b["decision"]=="unresolved" else drop)
        for part in polygons(g):
            target.polygon(prj(part.exterior.coords),fill=colors[b["id"]]+(115,) if b["id"] in colors else 255)
            for ring in part.interiors:target.polygon(prj(ring.coords),fill=(0,0,0,0) if b["id"] in colors else 0)
    for mask,cross in [(drop,False),(unknown,True)]:
        layer=Image.new("RGBA",base.size);layer.paste(hatch(base.size,cross),(0,0),mask);over=Image.alpha_composite(over,layer)
    d=ImageDraw.Draw(over)
    for b in fl:
        color=colors.get(b["id"],(45,45,45))
        for part in polygons(shape(b["floor_polygon_xz_m"])):
            if part.area<.01:continue
            d.line(prj(part.exterior.coords),fill=color+(255,),width=4)
    # Thin black actual final cuts; native scan exterior remains unaltered.
    for cut in reg.get("cut_lines",[]):
        if cut["floor_id"]!=fid or not cut.get("active_in_final_partition"):continue
        g=shape(cut.get("final_interface_geometry_xz_m",cut.get("line_geometry_xz_m")))
        def lines(geom):
            if geom.geom_type=="LineString":yield geom
            elif hasattr(geom,"geoms"):
                for sub in geom.geoms:yield from lines(sub)
        for line in lines(g):
            if line.length>.02:d.line(prj(line.coords),fill=(10,10,10,210),width=2)
    image=Image.alpha_composite(base,over);d=ImageDraw.Draw(image);big=ImageFont.truetype(FONT,46,index=2);small=ImageFont.truetype(FONT,26,index=2)
    for b in fl:
        if b["floor_area_m2"]<.5:continue
        parts=sorted(polygons(shape(b["floor_polygon_xz_m"])),key=lambda g:-g.area)
        if not parts:continue
        q=parts[0].representative_point();x,y=prj([(q.x,q.y)])[0];no=numbering[b["id"]];col=colors.get(b["id"],(60,60,60))
        if b["id"] in colors:
            d.text((x,y-18),str(no),font=big,fill="white",stroke_width=5,stroke_fill=col,anchor="mm")
            d.text((x,y+24),f'{b["floor_area_m2"]:.1f} m²',font=small,fill=(20,20,20),stroke_width=4,stroke_fill="white",anchor="mm")
            for part in parts[1:]:
                if part.area<.5:continue
                q=part.representative_point();xy=prj([(q.x,q.y)])[0];d.text(xy,str(no),font=small,fill="white",stroke_width=4,stroke_fill=col,anchor="mm")
        else:
            mark="?" if b["decision"]=="unresolved" else "×"
            d.text((x,y),mark+str(no),font=small,fill="white",stroke_width=4,stroke_fill=col,anchor="mm")
    d.rectangle((0,0,w,44),fill=(20,28,38));d.text((12,3),f'{version} {reg["source_region"]}/{fid}  {reg["source_floor_area_m2"]:.1f} m²',font=small,fill="white")
    return image.convert("RGB").resize((900,900),Image.Resampling.LANCZOS)

def embedded(p):
    return "data:image/jpeg;base64,"+base64.b64encode(Path(p).read_bytes()).decode()

def run(root,delivery_dir="delivery_all_v3"):
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));root=Path(root);out=root/delivery_dir
    if not (out/"completed.json").exists():raise RuntimeError("native v3 must finish before acceptance")
    old={p.name:json.loads(p.read_text()) for p in (root/"delivery_all_v2/regions").glob("*.json")};new={p.name:json.loads(p.read_text()) for p in (out/"regions").glob("*.json")}
    cap=json.loads((out/"revision_plan.json").read_text())["max_room_area_m2"]
    failures=[];kept=[];cuts=[];stair_rows=[];preserved=[];area_rows=[];unresolved=[]
    for name,reg in new.items():
        if not reg["requires_split"]:
            preserved.append(dict(source=name,unchanged=reg==old[name]))
            if reg!=old[name]:failures.append(dict(source=name,check="ORIGINAL_LE_CAP_CHANGED"))
            continue
        byfloor=defaultdict(list)
        for b in reg["blocks"]:
            byfloor[b["floor_id"]].append(shape(b["floor_polygon_xz_m"]))
            if b["decision"]=="unresolved":unresolved.append(dict(id=b["id"],area_m2=b["floor_area_m2"],reasons=b["unresolved_reasons"]))
            if b["decision"]!="retain":continue
            g=shape(b["floor_polygon_xz_m"]);n=independent_connectivity(g)
            item=dict(id=b["id"],independent_filled_outline_components=n,raw_ground_components=len(list(polygons(g))),ground_area_m2=g.area,reported_area_m2=b["floor_area_m2"],short_side_m=b["short_side_m"],black_fraction=b["black_fraction"],placement_found=b["placement_witness"].get("found"),outside_witness_points=[])
            for key in ["camera_m","source_1_m","source_2_m"]:
                pt=b["placement_witness"].get(key)
                if pt and not g.buffer(1e-7).covers(shapely.Point(pt[0],pt[2])):item["outside_witness_points"].append(key)
            checks=dict(one_filled_outline_component=n==1,area_bounds=6-1e-8<=g.area<=cap+1e-8,area_matches=abs(g.area-b["floor_area_m2"])<1e-6,short_side=b["short_side_m"]>=2.4-1e-8,black=b["black_fraction"] is not None and b["black_fraction"]<=.15,placement=item["placement_found"] is True,placement_in_ground=not item["outside_witness_points"])
            item["checks"]=checks;kept.append(item)
            for key,ok in checks.items():
                if not ok:failures.append(dict(id=b["id"],check=key))
        for floor in reg["source_geometry"]["floors"]:
            fid=floor["floor_id"];scope=shape(floor["floor_polygon"]);pieces=byfloor[fid];union=shapely.union_all(pieces)
            err=union.symmetric_difference(scope).area;overlap=sum(g.area for g in pieces)-union.area
            area_rows.append(dict(source=name,floor_id=fid,original_area_m2=scope.area,partition_area_m2=union.area,symmetric_difference_m2=err,overlap_area_m2=max(0.,overlap)))
            if err>1e-6 or overlap>1e-6:failures.append(dict(source=name,floor_id=fid,check="AREA_CONSERVATION"))
        for exclusion in reg.get("stair_exact_exclusion",[]):
            fid=exclusion["floor_id"];actual=shapely.union_all([shape(b["floor_polygon_xz_m"]) for b in reg["blocks"] if b["floor_id"]==fid and "STAIRS" in b.get("discard_reasons",[])])
            exact=shape(exclusion["geometry_xz_m"]);diff=actual.symmetric_difference(exact).area
            stair_rows.append(dict(source=name,floor_id=fid,semantic_stair_only_area_m2=exact.area,discarded_as_stairs_area_m2=actual.area,difference_area_m2=diff,expansion_m=exclusion["periphery_expansion_m"]))
            if diff>1e-6 or exclusion["periphery_expansion_m"]>.3:failures.append(dict(source=name,floor_id=fid,check="STAIR_EXCLUSION_NOT_EXACT"))
        for c in reg["cut_lines"]:
            n=len(c["line_xz_m"])-1
            cuts.append(dict(source_region=reg["house"]+"/"+reg["source_region"],floor_id=c["floor_id"],cut_id=c["id"],type=c["type"],stage=c.get("stage"),segment_count=n,active_in_final_partition=c.get("active_in_final_partition"),furniture_intersection_area_m2=c.get("furniture_intersection_area_m2"),furniture_intersection_length_m=c.get("furniture_intersection_length_m"),cut_band_width_m=c.get("intersection_area_band_width_m"),fallback_crosses_furniture=c.get("fallback_crosses_furniture"),furniture_intersections=c.get("furniture_intersections",[])))
            if n>3 or n<1:failures.append(dict(source=name,cut_id=c["id"],check="CUT_SEGMENT_COUNT"))
    before=stats(list(old.values()));after=stats(list(new.values()))
    selected=json.loads((root/"size_gallery_30_50_v1/selection_rows_v1.json").read_text());training_houses={r["house"] for r in selected};retained_houses={b["house"] for reg in new.values() for b in reg["blocks"] if b["decision"]=="retain"}
    summary=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_v2=before,after_v3=after,max_room_area_m2=cap,retained_house_count=len(retained_houses),retained_houses_in_existing_training_list=len(retained_houses&training_houses),lists_modified=False,source_regions_missing=sorted(set(old)-set(new)),native_house_job_failures=sum("error" in r for r in json.loads((out/"native_receipt.json").read_text())),owner_connectivity_definition="fill all internal holes for exterior connectivity and narrow-neck detection; erosion/dilation +/-0.3m; original holed geometry retained for area, cuts, black",minimum_partition_global_optimum="未验证：有限角度/0.25m偏移、最多36次回溯构造；部分块未判定，未声称数学全局最优")
    dump(out/"summary.json",summary)
    acceptance=dict(passed=not failures and set(old)==set(new),failures=failures,source_regions=len(new),retained_new_rooms=len(kept),retained_with_more_than_one_filled_outline_component=sum(r["independent_filled_outline_components"]!=1 for r in kept),retained_with_raw_detached_ground_parts=sum(r["raw_ground_components"]!=1 for r in kept),retained_area_min_m2=min((r["ground_area_m2"] for r in kept),default=None),retained_area_max_m2=max((r["ground_area_m2"] for r in kept),default=None),stairs_discarded_area_m2=sum(r["discarded_as_stairs_area_m2"] for r in stair_rows),exact_semantic_stair_area_m2=sum(r["semantic_stair_only_area_m2"] for r in stair_rows),stairs_symmetric_difference_m2=sum(r["difference_area_m2"] for r in stair_rows),cut_count=len(cuts),active_cut_count=sum(c["active_in_final_partition"] for c in cuts),max_cut_segments=max((c["segment_count"] for c in cuts),default=0),cuts_crossing_furniture=sum(bool(c["fallback_crosses_furniture"]) for c in cuts),unchanged_original_regions=len(preserved),all_original_regions_le_cap_unchanged=all(r["unchanged"] for r in preserved),all_partition_area_conserved=all(r["symmetric_difference_m2"]<1e-6 and r["overlap_area_m2"]<1e-6 for r in area_rows),retained_checks=kept,stair_checks=stair_rows,partition_checks=area_rows,original_preservation_checks=preserved)
    dump(out/"validation.json",acceptance);dump(out/"cut_furniture_intersections.json",cuts);dump(out/"unresolved.json",unresolved)
    with (out/"cut_furniture_intersections.csv").open("x",newline="") as f:
        fields=[k for k in cuts[0] if k!="furniture_intersections"] if cuts else ["cut_id"];w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(cuts)
    blocks=[b for r in new.values() for b in r["blocks"]]
    fields=["id","house","source_region","floor_id","floor_y_m","floor_area_m2","short_side_m","nav_walkable_area_m2","black_fraction","decision","room_type","discard_reasons","unresolved_reasons","visibility_coverage_fraction","connectivity_core_count"]
    with (out/"rooms_summary.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(blocks)
    media=out/"media_compare";media.mkdir(exist_ok=False);overlay_index=[];image_count=0
    with (out/"review_compare_v2_v3.html").open("x") as f:
        f.write('<!doctype html><html lang="zh"><meta charset="utf-8"><title>HM3D v2/v3 切分对照</title><style>body{font:16px sans-serif;margin:18px;background:#eee;color:#17212b}article{background:white;margin:18px 0;padding:16px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pair img{width:100%;aspect-ratio:1;object-fit:contain;background:#ddd}table{border-collapse:collapse;width:100%}td,th{padding:5px;border:1px solid #ddd}pre{white-space:pre-wrap}</style>')
        f.write('<p>v3 以填洞外轮廓确保新保留房间连通，原始带洞地面用于面积；原生 ≤50 m² 区域照旧保持。未判定和家具相交切线均明列，未跑声学。</p>')
        f.write('<p>左 v2、右 v3；每间一色，大号编号与面积；丢弃斜线，未判定网格；两侧均复用真实整屋 CPU 图。图片仅进入视口后解码，全部平铺。</p>')
        f.write('<pre>'+html.escape(json.dumps({k:v for k,v in summary.items() if k not in ["source_regions_missing"]},ensure_ascii=False,indent=2))+'</pre>')
        f.write('<p>独立机器验收 '+str(acceptance["passed"])+'；新保留 '+str(len(kept))+'，有飞地 '+str(acceptance["retained_with_more_than_one_filled_outline_component"])+'；楼梯 '+f'{acceptance["stairs_discarded_area_m2"]:.3f}'+' m²；切线最大 '+str(acceptance["max_cut_segments"])+' 段。</p>')
        for name,reg in sorted(new.items(),key=lambda x:(not x[1]["requires_split"],x[1]["house"],int(x[1]["source_region"][1:]))):
            previous=old[name];entry=dict(source=name,images=[],unchanged=not reg["requires_split"]);f.write('<article><h2>'+html.escape(reg["house"]+"/"+reg["source_region"])+f' — {reg["source_floor_area_m2"]:.2f} m²</h2>')
            if not reg["requires_split"]:f.write('<p>原生≤50 m²，v2/v3 JSON语义完全一致；未加连通或80%过滤。</p>')
            for fid in sorted({b["floor_id"] for b in reg["blocks"]}):
                if not any(b["floor_area_m2"]>=.5 for b in reg["blocks"] if b["floor_id"]==fid):continue
                frame=frame_for(previous,fid,root) or frame_for(reg,fid,root)
                if frame is None:
                    f.write('<p>'+html.escape(fid)+'：缺少已有真实俯视图，未补造图；几何见JSON。</p>');entry.setdefault("missing_frames",[]).append(fid);continue
                f.write('<div class="pair">')
                for version,r in [("v2",previous),("v3",reg)]:
                    stem=Path(name).stem+"__"+fid+"__"+version;path=media/(stem+".jpg")
                    image=overlay(r,fid,frame,version)
                    with path.open("xb") as dst:image.save(dst,format="JPEG",quality=82)
                    f.write('<div><b>'+version+' '+html.escape(fid)+'</b><img loading="lazy" decoding="async" data-src="'+embedded(path)+'" width="900" height="900" alt="'+html.escape(stem)+'"></div>')
                    entry["images"].append(str(path));image_count+=1
                f.write('</div>')
            for version,r in [("v2",previous),("v3",reg)]:
                ordered=sorted(r["blocks"],key=lambda b:(b["decision"]!="retain",-b["floor_area_m2"]));tiny=[b for b in ordered if b["floor_area_m2"]<.5]
                f.write('<p><b>'+version+'</b></p><table><tr><th>编号/块</th><th>m² /短边m</th><th>类型/结论</th><th>原因</th></tr>')
                for no,b in enumerate(ordered,1):
                    if b["floor_area_m2"]<.5:continue
                    why=b.get("discard_reasons") or b.get("unresolved_reasons") or []
                    f.write('<tr><td>'+str(no)+' / '+html.escape(b["id"].split("__")[-1])+'</td><td>'+f'{b["floor_area_m2"]:.2f} / {b.get("short_side_m",0):.2f}'+'</td><td>'+html.escape(str(b.get("room_type"))+"/"+b["decision"])+'</td><td>'+html.escape(", ".join(why))+'</td></tr>')
                f.write('</table>')
                if tiny:f.write('<p>&lt;0.5m² 碎面 '+str(len(tiny))+' 块，共 '+f'{sum(b["floor_area_m2"] for b in tiny):.3f}'+' m²；逐块信息均在 rooms/ JSON 与 CSV，图上保留其范围。</p>')
            f.write('</article>');overlay_index.append(entry)
        f.write('<script>const ob=new IntersectionObserver(es=>{for(const e of es){if(e.isIntersecting){const i=e.target;i.src=i.dataset.src;delete i.dataset.src;ob.unobserve(i)}}},{rootMargin:"0px",threshold:0});document.querySelectorAll("img[data-src]").forEach(i=>ob.observe(i));</script></html>')
    dump(out/"overlay_compare_index.json",overlay_index)
    report="v3 的新保留房间通过填洞外轮廓0.6m连通判据，原生≤50m²保持；未判定未伪装成保留。\n\n"
    report+=f"机器验收 passed={acceptance['passed']}；{len(new)} 来源区域，其中 {after['large_source_regions']} 需切、{len(preserved)} 原样保持。\n\n"
    report+=f"v2→v3：新保留 {before['retained_new_rooms']}→{after['retained_new_rooms']} 间，保留面积 {before['retained_new_area_m2']:.3f}→{after['retained_new_area_m2']:.3f} m²；未判定块 {before['unresolved_blocks']}→{after['unresolved_blocks']}，未判定来源区域 {before['unresolved_source_regions']}→{after['unresolved_source_regions']}。出处 summary.json。\n\n"
    report+=f"连通验收：新保留有不相连部分 {acceptance['retained_with_more_than_one_filled_outline_component']} 间；原始地面有分立部分 {acceptance['retained_with_raw_detached_ground_parts']} 间；面积范围 {acceptance['retained_area_min_m2']:.4f}–{acceptance['retained_area_max_m2']:.4f} m²。每个保留块独立复算、带洞面积与取点见证也核验，出处 validation.json。\n\n"
    report+=f"楼梯仅显式语义楼梯水平投影与原地面交集，无周边扩张：丢弃 {acceptance['stairs_discarded_area_m2']:.6f} m²，精确语义掩膜 {acceptance['exact_semantic_stair_area_m2']:.6f} m²，差 {acceptance['stairs_symmetric_difference_m2']:.9f} m²。周边平地正常检查。出处 validation.json.stair_checks 和 regions/*.json.stair_exact_exclusion，直接语义投影代码 pipeline.py:135–151。\n\n"
    report+=f"切线 {len(cuts)} 条（active {acceptance['active_cut_count']}），最多 {acceptance['max_cut_segments']} 段；穿家具 {acceptance['cuts_crossing_furniture']} 条，逐条列出 cut_furniture_intersections.json/csv。线本身二维面积为零，因此另以冻结网格0.25m宽条带测家具相交面积，同时列精确线穿家具长度；此量法只为审计，不改变门槛。可行切线族中按家具交叉优先、段数≤3，有限候选最小不等于数学全局最小，未验证全局最优。家具使用同层语义面保持形状投影，cut guidance 层高2.4m仅涵盖家具，不修改 placement 参数。\n\n"
    report+="参数：上限50m²（--max-room-area-m2）、下限6m²、短边2.4m、黑区15%、原冻结摆放见证；仅>50的残余块走原80%/6m可见性检查，原生≤50完全复制200区域。内部洞只在连通/窄口/主体净宽判断时填，原地面多边形/面积/黑区均未填洞。\n\n"
    report+="丢弃原因面积（首原因互斥；inclusive表不可求和）：\n\n| 原因 | v2 m² | v3 m² |\n|---|---:|---:|\n"
    for reason in sorted(set(before["discard_primary_reason_area_m2"])|set(after["discard_primary_reason_area_m2"])):report+=f"| {reason} | {before['discard_primary_reason_area_m2'].get(reason,0):.3f} | {after['discard_primary_reason_area_m2'].get(reason,0):.3f} |\n"
    report+=f"\n新保留涉及 {len(retained_houses)} 套，其中 {len(retained_houses&training_houses)} 套已在现有训练名单；仅报告不改名单。\n\n"
    report+="未判定逐块：\n\n"
    for b in unresolved:
        if b["area_m2"]<6:continue
        report+=f"- {b['id']}：{b['area_m2']:.3f} m²；{', '.join(b['reasons'])}。\n"
    report+="\n首轮00626候选切线搜索耗时过长，核对worker PID2467638、父进程2466833与启动标识后只SIGTERM该自有worker；原记录在revision_connectivity_20261009_v1/native_cpu_budget_stop_v1.json。框架回收其自有进程池，缺失区域另补算；首轮所有产物保存在delivery_all_v3_native_attempt_v1。补算候选轴各最多8个（均匀覆盖全部轴界值）只限制计算规模，不修改准入标准；来源及选择记录refinement_receipt.json。全局最优未验证。\n"
    report+="\n真实整屋图复用readonly CPU缓存；对照页 review_compare_v2_v3.html，平铺内嵌严格懒加载。未跑声学，未接入生产，未重调参数或重复留出IoU；原对照成绩不变。动态路线受真实形状约束及全局最少切块未验证。\n"
    with (out/"REPORT_zh.md").open("x") as f:f.write(report)
    with (out/"review.html").open("x") as f:f.write((out/"review_compare_v2_v3.html").read_text())
    with (root/"PROGRESS_zh.md").open("a") as f:f.write("\nB_READY "+str(out/"review_compare_v2_v3.html")+"；验收="+str(acceptance["passed"])+"，新保留="+str(len(kept))+"，未判定块="+str(after["unresolved_blocks"])+"。\n")
    print("V3_ACCEPTANCE",acceptance["passed"],"KEPT",len(kept),"IMAGES",image_count,flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",required=True);a.add_argument("--delivery-dir",default="delivery_all_v3");v=a.parse_args();run(v.root,v.delivery_dir)
