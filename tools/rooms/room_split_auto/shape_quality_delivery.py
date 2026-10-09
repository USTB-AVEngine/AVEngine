"""Immutable v6 assembly, source-by-source quality comparison and draft v2."""
from __future__ import annotations
import argparse,copy,csv,hashlib,html,json,math,resource,shutil,datetime
from pathlib import Path
from collections import Counter
import shapely
from shapely.geometry import shape
from tools.rooms.room_split_auto.shape_quality_geometry import own_outline,defects,wrap_count,disk
from tools.rooms.room_split_auto import walkable_review as cr,revision_review as rr
from tools.rooms.room_split_auto.native_finalize import copy_record
def dump(p,x):
    with Path(p).open("x") as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def key(r):return r["house"]+"/"+r["source_region"]
def score(r):
    kept=[b for b in r["blocks"] if b["decision"]=="retain"]
    kinds=Counter(NECK=0,CORRIDOR=0,WRAP=0,FURNITURE=0)
    for b in kept:
        d=defects(shape(b["floor_polygon_xz_m"]));kinds["NECK"]+=d["neck_count"];kinds["CORRIDOR"]+=d["corridor_count"]
    for fid in {b["floor_id"] for b in kept}:
        kinds["WRAP"]+=wrap_count([shape(b["floor_polygon_xz_m"]) for b in kept if b["floor_id"]==fid])
    kinds["FURNITURE"]=sum(c.get("active_in_final_partition",True) and (c.get("furniture_intersection_length_m") or 0)>.5 for c in r["cut_lines"])
    return dict(counts=dict(kinds),total=sum(kinds.values()),own_circle_missing_count=sum(not disk(shape(b["floor_polygon_xz_m"]))["fits"] for b in kept),retained=len(kept),area_m2=sum(b["floor_area_m2"] for b in kept),
        furniture_bad_length_m=sum(c.get("furniture_intersection_length_m") or 0 for c in r["cut_lines"] if c.get("active_in_final_partition",True) and (c.get("furniture_intersection_length_m") or 0)>.5))
def canonical_interfaces(reg):
    """Persist the same surviving design span used by furniture and review."""
    r=copy.deepcopy(reg)
    for c in r["cut_lines"]:
        current=c.get("current_design_span_geometry_xz_m")
        if current is None:continue
        c.setdefault("legacy_measured_interface_geometry_xz_m",c.get("final_interface_geometry_xz_m"))
        c["final_interface_geometry_xz_m"]=current
    return r

def same_layout(a,b):
    aa=[x for x in a["blocks"] if x["decision"] in ("retain","unresolved")]
    bb=[x for x in b["blocks"] if x["decision"] in ("retain","unresolved")]
    if len(aa)!=len(bb):return False
    used=set()
    for x in aa:
        matches=[i for i,y in enumerate(bb) if i not in used and x["decision"]==y["decision"] and x["floor_id"]==y["floor_id"] and shape(x["floor_polygon_xz_m"]).symmetric_difference(shape(y["floor_polygon_xz_m"])).area<=1e-8]
        if not matches:return False
        used.add(matches[0])
    return True

def construction_lost(old,new):
    # A previously admitted room may not disappear merely because repair fails.
    lost=[]
    kept=[x for x in old["blocks"] if x["decision"]=="retain"]
    for b in new["blocks"]:
        if b["decision"]!="unresolved":continue
        g=shape(b["floor_polygon_xz_m"])
        overlap=sum(g.intersection(shape(a["floor_polygon_xz_m"])).area for a in kept if a["floor_id"]==b["floor_id"])
        if overlap>.05:lost.append(dict(block=b["id"],previously_retained_overlap_m2=overlap,reasons=b.get("unresolved_reasons",[])))
    return lost
def assemble(root,attempts=("attempt_v2",),name="final_v1"):
    root=Path(root);base=root/"delivery_all_v5/final_v1";out=root/"delivery_all_v6"/name
    out.mkdir();(out/"regions").mkdir();(out/"rooms").mkdir()
    plan=json.loads((root/"delivery_all_v6/revision_plan_v1.json").read_text());targets=set(plan["shape_targets"]);retried=set(plan["retry_sources"])
    old={p.name:json.loads(p.read_text()) for p in (base/"regions").glob("*.json")}
    new={};selection=[];changed=[];sha=[]
    for filename,before in sorted(old.items()):
        source=key(before);best=before;origin="v5_unchanged";bs=score(before) if before.get("requires_split") else None;reasons=[]
        if source in targets:
            for attempt in attempts:
                p=root/"delivery_all_v6"/attempt/"regions"/filename
                if not p.exists():reasons.append(dict(attempt=attempt,reason="MISSING_RESULT"));continue
                candidate=json.loads(p.read_text());cs=score(candidate)
                if candidate.get("v6_attempt_failure"):reasons.append(dict(attempt=attempt,reason=candidate["v6_attempt_failure"]["reason"]));continue
                lost=construction_lost(before,candidate)
                if lost:reasons.append(dict(attempt=attempt,reason="REPAIR_FAILURE_MUST_KEEP_V5",lost=lost));continue
                if before.get("retained_new_rooms",bs["retained"])>0 and cs["retained"]==0:
                    reasons.append(dict(attempt=attempt,reason="DO_NOT_DROP_WHOLE_ADMITTED_SOURCE"));continue
                bscore=score(best)
                if (cs["total"],cs["own_circle_missing_count"],cs["furniture_bad_length_m"],-cs["area_m2"])<(bscore["total"],bscore["own_circle_missing_count"],bscore["furniture_bad_length_m"],-bscore["area_m2"]):
                    best=candidate;origin=attempt
            if best is not before and same_layout(before,best):
                best=before;origin="v5_unchanged"
        if source in retried and source not in targets:
            reasons.append(dict(reason="RETRY_PROPOSAL_SEPARATE_TO_PRESERVE_65_SOURCE_BYTES"))
        ischanged=best is not before
        if ischanged:best=canonical_interfaces(best)
        new[filename]=best
        if ischanged:
            changed.append(filename);dump(out/"regions"/filename,best)
            for b in best["blocks"]:dump(out/"rooms"/(b["id"]+".json"),b)
        else:
            shutil.copyfile(base/"regions"/filename,out/"regions"/filename)
            for b in before["blocks"]:
                f=b["id"]+".json";shutil.copyfile(base/"rooms"/f,out/"rooms"/f)
        if before.get("requires_split"):
            selection.append(dict(source=source,filename=filename,shape_target=source in targets,retry=source in retried,changed=ischanged,producer=origin,
                before=bs,after=score(best),attempt_rejection_reasons=reasons))
        if before.get("requires_split") and source not in targets:
            for b in before["blocks"]:
                f=b["id"]+".json";original=base/"rooms"/f;dest=out/"rooms"/f
                sha.append(dict(source=source,file=f,v5_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                    v6_sha256=hashlib.sha256(dest.read_bytes()).hexdigest() if dest.exists() else None,
                    identical=dest.exists() and original.read_bytes()==dest.read_bytes()))
    active=[r for r in new.values() if r.get("requires_split")]
    rooms=[b for r in active for b in r["blocks"]];kept=[b for b in rooms if b["decision"]=="retain"]
    summary=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_regions=len(new),split_sources=len(active),target_sources=39,
        retried_sources=5,retries_outside39_proposals_only=3,changed_sources=len(changed),unchanged_split_sources=sum(not x["changed"] for x in selection),
        retained_rooms=len(kept),retained_area_m2=sum(b["floor_area_m2"] for b in kept),
        discarded_area_m2=sum(b["floor_area_m2"] for b in rooms if b["decision"]=="discard"),
        unresolved=[dict(id=b["id"],area_m2=b["floor_area_m2"],reasons=b.get("unresolved_reasons")) for b in rooms if b["decision"]=="unresolved"],
        shape_counts={k:sum(score(r)["counts"][k] for r in active) for k in ["NECK","CORRIDOR","WRAP","FURNITURE"]},
        unchanged65_sha256_rows=len(sha),unchanged65_identical=all(x["identical"] for x in sha),
        max_room_area_m2=35,cpu_only=True,acoustics="not_run",production_integrated=False,lists_modified=False)
    dump(out/"summary.json",summary);dump(out/"source_selection.json",selection);dump(out/"changed_sources.json",changed)
    dump(out/"unchanged65_sha256.json",sha)
    with (out/"unchanged65_sha256.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["source","file","v5_sha256","v6_sha256","identical"]);w.writeheader();w.writerows(sha)
    fields=["id","house","source_region","floor_id","floor_y_m","floor_area_m2","short_side_m","nav_walkable_area_m2","black_fraction","decision","discard_reasons","unresolved_reasons","floor_polygon_xz_m","placement_witness"]
    csv_records(out/"rooms_summary.csv",rooms,fields)
    dump(out/"unresolved.json",summary["unresolved"])
    return old,new,summary,selection
def csv_records(path,rows,fields):
    with Path(path).open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader()
        for r in rows:w.writerow({k:json.dumps(r.get(k),ensure_ascii=False) if isinstance(r.get(k),(dict,list)) else r.get(k) for k in fields})
def review(root,out,old,new,summary):
    media=out/"media_compare";media.mkdir();page=out/"review_compare_v5_v6.html";index=[]
    changed=json.loads((out/"changed_sources.json").read_text())
    with page.open("x") as f:
        f.write('<!doctype html><html lang="zh"><meta charset="utf-8"><title>HM3D v5/v6形状修正</title>'+cr.STYLE)
        f.write('<p>v6 仅修授权来源的形状；整屋几何照旧，切分只限定站位，未跑声学。本页只列实际变化的来源，左v5、右v6；图片内嵌并严格懒加载。</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
        for name in changed:
            a,b=old[name],new[name];entry=dict(source=key(b),images=[],missing=[])
            f.write('<article><h2>'+html.escape(key(b))+'</h2>')
            for fid in sorted({x["floor_id"] for x in b["blocks"] if x["floor_area_m2"]>=.5}):
                fr=cr.frame(b,fid,root) or cr.frame(a,fid,root)
                if fr is None:
                    extra=root/"delivery_all_v5/visual_complete_v1/regions_ref"/name
                    if extra.exists():fr=cr.frame(json.loads(extra.read_text()),fid,root)
                if fr is None:entry["missing"].append(fid);f.write('<p>'+fid+' 缺底图，未造图</p>');continue
                f.write('<div class="pair">')
                for label,reg in [("v5",a),("v6",b)]:
                    image=media/(Path(name).stem+"__"+fid+"__"+label+".jpg")
                    with image.open("xb") as dst:rr.overlay(reg,fid,fr,label).save(dst,format="JPEG",quality=86)
                    f.write('<div><b>'+label+' '+fid+'</b><img loading="lazy" decoding="async" width="900" height="900" data-src="'+cr.embedded(image)+'"></div>')
                    entry["images"].append(str(image))
                f.write('</div>')
            f.write('<p>v5</p>'+cr.table(a)+'<p>v6</p>'+cr.table(b)+'</article>');index.append(entry)
        f.write(cr.LAZY+'</html>')
    dump(out/"html_validation.json",dict(changed_sources=len(changed),image_count=sum(len(x["images"]) for x in index),missing=[x for x in index if x["missing"]],embedded=True,strict_lazy_initial_src_absent=True,index=index))
def draft(root,new):
    previous=json.loads((root/"final_room_list_draft_v1/room_list_draft_v1.json").read_text())
    native=[x for x in previous["rooms"] if x["source"]!="new_cut"]
    assert len(native)==775
    originals={(x["house"],x["source_region"]):x for x in previous["rooms"] if x["source"]=="new_cut"}
    added=[]
    for reg in new.values():
        if not reg.get("requires_split"):continue
        for b in reg["blocks"]:
            if b["decision"]!="retain":continue
            r=copy_record(b);r["source"]="new_cut";old=originals.get((b["house"],b["source_region"]))
            r["origin_list"]=old["origin_list"] if old else "new";r["source_selection"]=old["source_selection"] if old else None
            added.append(r)
    records=native+added;assert len({x["id"] for x in records})==len(records)
    native_source={(r["house"],r["source_region"]) for r in native};new_source={(r["house"],r["source_region"]) for r in added}
    assert not native_source&new_source
    pending=[x for x in json.loads((root/"final_room_list_draft_v1/pending.json").read_text()) if x.get("decision")=="pending_legacy"]
    pending += [b for r in new.values() if r.get("requires_split") for b in r["blocks"] if b["decision"]=="unresolved"]
    out=root/"final_room_list_draft_v2";out.mkdir()
    dump(out/"room_list_draft_v2.json",dict(schema="HM3D_room_list_draft_v2_cap35",rooms=records,leakage="pending separate acoustics task",production_integrated=False))
    fields=["id","house","source_region","source_region_id","floor_id","floor_y_m","floor_area_m2","short_side_m","black_fraction","source","origin_list","source_selection","floor_polygon_xz_m","placement_witness","leakage"]
    csv_records(out/"rooms.csv",records,fields);dump(out/"pending.json",pending)
    summary=dict(total_rooms=len(records),total_area_m2=sum(r["floor_area_m2"] for r in records),native_rooms=775,new_cut_rooms=len(added),
        native_records_identical_to_v1=native==[x for x in previous["rooms"] if x["source"]!="new_cut"],
        native_records_sha256=hashlib.sha256(json.dumps(native,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
        pending=len(pending),max_room_area_m2=35,leakage_columns_empty=True,production_integrated=False,lists_modified=False,acoustics="not_run")
    dump(out/"summary.json",summary)
    return summary
def retry_proposals(root,attempts):
    root=Path(root);out=root/"delivery_all_v6/unresolved_retry5_v1";out.mkdir();(out/"regions").mkdir()
    plan=json.loads((root/"delivery_all_v6/revision_plan_v1.json").read_text())
    targets=set(plan["shape_targets"]);sources=set(plan["retry_sources"]);rows=[]
    for path in sorted((root/"delivery_all_v5/final_v1/regions").glob("*.json")):
        before=json.loads(path.read_text());source=key(before)
        if source not in sources:continue
        best=before;producer="v5_unchanged";rejected=[]
        for attempt in attempts:
            f=root/"delivery_all_v6"/attempt/"regions"/path.name
            if not f.exists():continue
            candidate=json.loads(f.read_text())
            if candidate.get("v6_attempt_failure"):
                rejected.append(dict(attempt=attempt,reason=candidate["v6_attempt_failure"]["reason"]));continue
            if construction_lost(before,candidate):continue
            a,b=score(best),score(candidate)
            if (b["total"],b["furniture_bad_length_m"],-b["area_m2"])<(a["total"],a["furniture_bad_length_m"],-a["area_m2"]):
                best=candidate;producer=attempt
        dump(out/"regions"/path.name,best)
        rows.append(dict(source=source,inside_authorized39=source in targets,in_main_delivery=source in targets,
            producer=producer,before=score(before),after=score(best),remaining_unresolved=[
                dict(id=b["id"],area_m2=b["floor_area_m2"],reasons=b.get("unresolved_reasons")) for b in best["blocks"] if b["decision"]=="unresolved"],
            rejected_attempts=rejected))
    dump(out/"summary.json",dict(sources=rows,policy="three retries outside the 39 shape targets are proposals only; the other 65 source room bytes remain unchanged"))
    return rows

def run(root,attempts,name="final_v1",make_draft=True):
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));root=Path(root)
    old,new,summary,selection=assemble(root,attempts,name);out=root/"delivery_all_v6"/name
    review(root,out,old,new,summary)
    if make_draft:
        retry_proposals(root,attempts)
        ds=draft(root,new);print("DRAFT_V2",json.dumps(ds,ensure_ascii=False))
    print("V6_ASSEMBLED",json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",required=True);a.add_argument("--attempt",action="append",default=[]);a.add_argument("--name",default="final_v1");a.add_argument("--no-draft",action="store_true");v=a.parse_args()
    run(v.root,v.attempt or ["attempt_v2"],v.name,not v.no_draft)
