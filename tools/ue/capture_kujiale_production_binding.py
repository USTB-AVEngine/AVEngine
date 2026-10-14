"""Exercise the production physical-source spawn method in native UE."""
from __future__ import annotations
import json
from avengine.capture.kujiale_registered_environment import environment_bindings,require_binding_project
from avengine.capture.v7_registered_ue import NativeRegisteredUEVideo
from avengine.qa.kujiale_asset_geometry import import_vertices
from capture_kujiale_admission import write

def run(c):
    path=c.root/"registrations/v7_kujiale_ue_asset_admission_20261012.json"
    book=json.loads(path.read_text());original={x["asset_id"]:x for x in c.manifest["assets"]}
    results=[]
    for surface in ["floor","table","wall","ceiling"]:
        candidates=[x for x in book["assets"] if x["category"]=="object" and x["attachment_surface"]==surface and not x["silent_decor"] and c.args.house_id in x["native_house_admissions"] and x["ue_native_physical_status"]=="pass"]
        if not candidates:raise ValueError("no admitted production binding candidate on "+surface)
        row=candidates[0];native=row["native_house_admissions"][c.args.house_id];aid=row["asset_id"]
        spec={"sources":{"env":{"actor_id":"binding_"+surface,"native_asset_id":aid,"trajectory":[{"time_s":0.,"position_m":native["emitter"]["actual_m"]}]}}}
        bindings=environment_bindings(spec,c.args.house_id,path,set());require_binding_project(bindings,c.ue)
        result={"asset_id":aid,"surface":surface,"status":"fail","production_method":"NativeRegisteredUEVideo._spawn_physical_environment","registration":str(path),"views":[]}
        try:
            def spawn():
                node=NativeRegisteredUEVideo._spawn_physical_environment(c,"env",bindings["env"])
                c.active=node["runtimes"]
                result["human"]=c.spawn_human(c.plan_human(native["placement"]))
            c.frame(spawn)
            source=original[aid];receipt=json.loads((c.root/"evidence/import/final"/(aid+".json")).read_text());vertices=import_vertices(receipt)
            result["emitter"]=c.read_emitter(source,native["placement"])
            for index,witness in enumerate(native["views"][:3]):
                result["views"].append(c.view(source,witness["view"],native["placement"],vertices,c.root/"renders/production_binding"/c.args.house_id/aid,index,240 if not c.warmed else 24));c.warmed=True
            result["status"]="pass" if result["emitter"]["status"]=="pass" and len(result["views"])==3 and all(v["status"]=="pass" for v in result["views"]) else "fail"
        finally:c.cleanup()
        results.append(result);print("PRODUCTION_BINDING",surface,aid,result["status"],flush=True)
        write(c.root/"evidence/native_production_binding_smoke.json",{"status":"pass" if len(results)==4 and all(v["status"]=="pass" for v in results) else "incomplete_or_fail","house":c.args.house_id,"scope":"actual production environmental spawn method; four attachment surfaces, native views and emitter readbacks; no question generation or timeline production","cases":results})
    if any(v["status"]!="pass" for v in results):raise RuntimeError("production binding native smoke failed")
