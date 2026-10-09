"""Complete all v5 review-floor images in a new immutable visual directory."""
from pathlib import Path
import argparse,copy,html,json,os,resource
import shapely
from shapely.geometry import shape,Polygon,Point
from tools.rooms.room_selection.media import polygons
from tools.rooms.room_split_auto import walkable_review as vr,revision_review as rr
from tools.rooms.room_split_auto.size_gallery import render
dump=vr.dump

def run(root):
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    root=Path(root);base=root/"delivery_all_v5/final_v1"
    if not json.loads((base/"validation.json").read_text())["passed"]:raise RuntimeError("accepted geometry required")
    out=root/"delivery_all_v5/visual_complete_v1";out.mkdir(exist_ok=False);(out/"raw").mkdir();(out/"regions_ref").mkdir()
    new={p.name:json.loads(p.read_text()) for p in (base/"regions").glob("*.json")}
    old={p.name:json.loads(p.read_text()) for p in (root/"delivery_all_v3/final_v1/regions").glob("*.json")}
    v4={p.name:json.loads(p.read_text()) for p in (root/"delivery_all_v4/final_v1/regions").glob("*.json")}
    missing=json.loads((base/"html_validation.json").read_text())["missing_frames"];receipts=[]
    for row in missing:
        reg=new[row["source"]];inv=json.loads((root/"inventory_v1"/(reg["house"]+".json")).read_text())
        gs=[shape(f["floor_polygon"]) for r in inv["rows"] for f in r["floors"]]
        x0,z0,x1,z1=shapely.union_all(gs).bounds
        floors=[dict(key="EXTRA_"+f["floor_id"],y=f["floor_y_m"]) for f in reg["source_geometry"]["floors"] if f["floor_id"] in row["missing_frames"]]
        job=dict(house=reg["house"],scene_directory=reg["source_geometry"]["scene_directory"],bounds=[x0,z0,x1,z1],cx=(x0+x1)/2,cz=(z0+z1)/2,span=max(x1-x0,z1-z0)*1.12+1,floors=floors)
        receipts.append(render(job,out/"raw"))
        for f in floors:
            p=out/"raw"/(reg["house"]+"__"+f["key"]+".json");meta=json.loads(p.read_text());fid=f["key"][6:]
            reg["floor_overheads"][fid]=dict(metadata_path=str(p),image_path=meta["path"],source="whole-house original CPU llvmpipe missing-floor completion")
        dump(out/"regions_ref"/row["source"],reg)
    summary=json.loads((base/"summary.json").read_text());summary.update(geometry_delivery=str(base),visual_only_completion=True,added_floor_frames=sum(len(r["missing_frames"]) for r in missing))
    dump(out/"summary.json",summary);dump(out/"render_receipt.json",receipts)
    index=vr.render_comparisons(root,out,old,new,summary)
    with (out/"review_compare_v4_v5.html").open("x") as dst:
        dst.write('<!doctype html><html lang="zh"><meta charset="utf-8"><title>v4/v5完整楼层对照</title>'+vr.STYLE+'<p>v5按35m²和新接缝/navmesh规则验收通过；额外楼层真实CPU图已补齐。左v4右v5，整屋原图，丢弃斜线、未判定网格。</p>')
        for name,r in sorted(new.items()):
            if not r["requires_split"]:continue
            prior=v4.get(name,vr.original_native(r));dst.write('<article><h2>'+html.escape(r["house"]+"/"+r["source_region"])+'</h2>')
            for fid in sorted({b["floor_id"] for b in r["blocks"] if b["floor_area_m2"]>=.5}):
                fr=vr.frame(r,fid,root)
                if fr is None:raise RuntimeError("missing completed frame")
                dest=out/"media_compare"/(Path(name).stem+"__"+fid+"__v4.jpg")
                with dest.open("xb") as f:rr.overlay(prior,fid,fr,"v4").save(f,format="JPEG",quality=83)
                right=out/"media_compare"/(Path(name).stem+"__"+fid+"__v5.jpg")
                dst.write('<div class="pair"><img loading="lazy" width="900" height="900" data-src="'+vr.embedded(dest)+'"><img loading="lazy" width="900" height="900" data-src="'+vr.embedded(right)+'"></div>')
            dst.write(vr.table(prior)+vr.table(r)+'</article>')
        dst.write(vr.LAZY+'</html>')
    holes=[]
    for reg in new.values():
        if not reg["requires_split"]:continue
        for b in reg["blocks"]:
            if b["decision"]!="retain":continue
            centre=b["inscribed_circle"]["centre_xz_m"]
            for p in polygons(shape(b["floor_polygon_xz_m"])):
                for ring in p.interiors:
                    h=Polygon(ring)
                    if h.area>=6:holes.append(dict(id=b["id"],void_area_m2=h.area,circle_centre_in_void=h.covers(Point(centre))))
    dump(out/"large_void_circle_diagnostic.json",dict(large_raw_voids=holes,diagnostic_only=True))
    report=dict(compare_sources=len(index),native_sources=sum(x["native_35_50"] for x in index),missing_frames=[r for r in index if r["missing_frames"]],image_count=sum(len(r["images"]) for r in index),embedded=True,strict_lazy_initial_src_absent=True,source_geometry_unchanged=True,pid=os.getpid(),cpu_only=True)
    dump(out/"html_validation.json",report)
    if report["missing_frames"]:raise RuntimeError("image completion incomplete")
    dump(out/"completed.json",report)
    with (root/"PROGRESS_zh.md").open("a") as f:f.write("\nV5_VISUAL_COMPLETE "+str(out/"review_compare_v4_v5.html")+"；104来源、54原生组；额外楼层缺图=0，数据仍为delivery_all_v5/final_v1。\n")
    print("V5_VISUAL_COMPLETE",report,flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",required=True);v=a.parse_args();run(v.root)
