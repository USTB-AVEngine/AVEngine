"""Preserve successful straight visibility leaves when a residual sibling fails."""
from pathlib import Path
import argparse,copy,datetime,json,math,multiprocessing,os,resource,shutil,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from tools.rooms.room_split_auto import connected_split as cs
PARTIAL="VISIBILITY_PARTIAL_STRAIGHT_CONSTRUCTION_UNRESOLVED"

def solve_partial(self,g,depth=0):
    if g.area<=self.cap+1e-8:
        vis=self.visibility(g)
        if vis["meets_visibility"]:return [(g,vis)],[],None
    if depth>=8 or g.area<12-1e-8 or self.attempts>=36:return [(g,None)],[],"VISIBILITY_STRAIGHT_PARTITION_UNRESOLVED"
    choices,count=cs.line_candidates(g,self.furniture,self.cap,max_dogleg_axis=8)
    if not choices:return [(g,None)],[],"NO_FEASIBLE_STRAIGHT_OR_THREE_SEGMENT_CUT"
    best=None;partial=None
    for score,line,children,m in choices[:5]:
        if self.attempts>=36:break
        self.attempts+=1;leaves=[];cuts=[];failed=False
        for child in children:
            parts,conncuts=cs.connected_parts(child,self.furniture)
            for c,cm,reason in conncuts:cuts.append((c,cm,"narrow",reason,count))
            for part in parts:
                if part.area<6:leaves.append((part,None));continue
                ll,cc,err=solve_partial(self,part,depth+1);leaves.extend(ll);cuts.extend(cc)
                if err:failed=True
        cuts.append((line,m,"visibility","straight_or_three_segment_visibility_partition",count))
        unresolved_area=sum(x.area for x,v in leaves if x.area>=6 and v is None)
        if failed:
            value=(unresolved_area,len([x for x,v in leaves if x.area>=6 and v is None]),len(leaves),sum(c[1]["furniture_intersection_area_m2"] for c in cuts))
            if partial is None or value<partial[0]:partial=(value,leaves,cuts)
            continue
        value=(len([x for x,v in leaves if x.area>=6]),sum(c[1]["furniture_intersection_area_m2"] for c in cuts))
        if best is None or value<best[0]:best=(value,leaves,cuts)
        if value[0]==math.ceil((g.area-1e-8)/self.cap):break
    self.audit.append(dict(floor_area_m2=g.area,candidate_count=count,attempts_total=self.attempts,feasible=best is not None,partial_valid_construction=partial is not None,partial_unresolved_area_m2=partial[0][0] if partial else None,candidate_minimum_furniture_overlap_area_m2=min(x[3]["furniture_intersection_area_m2"] for x in choices),global_minimum_unverified=True,dogleg_axis_candidate_limit=8))
    if best:return best[1],best[2],None
    if partial and partial[0][0]<g.area-1e-7:return partial[1],partial[2],PARTIAL
    return [(g,None)],[],"VISIBILITY_STRAIGHT_PARTITION_NO_VALID_CONSTRUCTION_WITHIN_CPU_BUDGET"

def worker(job,root,out,p,parameter,cap,cache):
    cs.VisibilityCutter.solve=solve_partial
    original=cs.process
    def process(*args,**kwargs):
        result=original(*args,**kwargs)
        for b in result["blocks"]:
            if b["decision"]!="unresolved" or not b["visibility"].get("meets_visibility"):continue
            b["unresolved_reasons"]=[r for r in b["unresolved_reasons"] if r!=PARTIAL]
            b["unverified_diagnostics"]=[r for r in b["unverified_diagnostics"] if r!=PARTIAL]
            if not b["unresolved_reasons"]:b["decision"]="retain"
        result["retained_new_rooms"]=sum(b["decision"]=="retain" for b in result["blocks"])
        result["status"]="partially_unresolved" if any(b["decision"]=="unresolved" for b in result["blocks"]) else "processed"
        result["partial_refinement"]="valid leaves preserve original frozen visibility/area/shape/black/placement checks; failed sibling alone unresolved"
        return result
    cs.process=process
    return cs.worker(job,root,out,p,parameter,cap,cache)

def run(root,workers=2):
    resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,10*1024**3));root=Path(root);native=root/"delivery_all_v3"
    if not (native/"completed.json").exists():raise RuntimeError("native run must finish first")
    trial=root/"revision_connectivity_20261009_v1/partial_refine_v1";trial.mkdir(exist_ok=False);(trial/"regions").mkdir()
    plan=json.loads((root/"processing_plan_v1.json").read_text());parameter=json.loads((root/"selected_parameter_v1.json").read_text())["selected_parameter"]
    jobs={}
    for path in (native/"regions").glob("*.json"):
        d=json.loads(path.read_text())
        if not any(b["decision"]=="unresolved" and any("VISIBILITY_STRAIGHT_PARTITION" in reason or reason=="NATIVE_HOUSE_JOB_FAILURE" for reason in b.get("unresolved_reasons",[])) for b in d["blocks"]):continue
        if d["house"] not in jobs:
            j=next(x.copy() for x in plan["jobs"] if x["house"]==d["house"]);j.pop("rows",None);j["sources"]=[];jobs[d["house"]]=j
        jobs[d["house"]]["sources"].append(str(root/"delivery_all_v2/regions"/path.name))
    receipts=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        fut={pool.submit(worker,j,root,trial,plan["parameters"],parameter,50.,plan["overhead_cache"]):j["house"] for j in jobs.values()}
        for f in as_completed(fut):
            try:receipts.append(f.result())
            except Exception as e:receipts.append(dict(house=fut[f],error=repr(e)))
    cs.dump(trial/"runtime_receipt.json",receipts)
    chosen={};selection=[]
    for path in (trial/"regions").glob("*.json"):
        d=json.loads(path.read_text());previous=json.loads((native/"regions"/path.name).read_text())
        before=sum(b["floor_area_m2"] for b in previous["blocks"] if b["decision"]=="retain");after=sum(b["floor_area_m2"] for b in d["blocks"] if b["decision"]=="retain")
        use=after>before+1e-7 and d.get("area_partition_error_m2",1)<1e-6
        selection.append(dict(source=path.name,before_retained_area_m2=before,after_retained_area_m2=after,selected=use))
        if use:chosen[path.name]=path
    cs.dump(trial/"selection.json",selection)
    preserved=root/"delivery_all_v3_native_attempt_v1"
    if preserved.exists():raise RuntimeError("refuse to overwrite preserved native attempt")
    native.rename(preserved)
    native.mkdir();(native/"regions").mkdir();(native/"rooms").mkdir()
    for p in (preserved/"regions").glob("*.json"):
        source=chosen.get(p.name,p);d=json.loads(source.read_text());cs.dump(native/"regions"/p.name,d)
        for b in d["blocks"]:cs.dump(native/"rooms"/(b["id"]+".json"),b)
    for name in ["revision_plan.json","native_receipt.json","completed.json"]:shutil.copyfile(preserved/name,native/name)
    cs.dump(native/"refinement_receipt.json",dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),preserved_native_attempt=str(preserved),partial_refinement_trial=str(trial),selected=selection,workers=workers,source_code="tools/rooms/room_split_auto/connected_refine.py",pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0)))
    print("V3_REFINED",selection,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--workers",type=int,default=2);a=p.parse_args();run(a.root,a.workers)
