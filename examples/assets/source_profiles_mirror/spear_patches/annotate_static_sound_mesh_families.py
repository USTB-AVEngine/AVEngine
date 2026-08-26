#!/usr/bin/env python3
"""Add the 26 AudioSet classes -> 14 logical mesh families mapping to the shared index."""
import argparse,json
from pathlib import Path

FAMILIES={
"doorbell":({"doorbell_chime_unit"},["Doorbell","Ding-dong","Chime"]),
"landline_phone":({"desk_telephone","landline_phone"},["Telephone","Telephone bell ringing","Telephone dialing DTMF","Dial tone","Busy signal"]),
"cellphone":({"cellphone"},["Ringtone","Cellphone buzz vibrating alert"]),
"alarm_clock":({"alarm_clock"},["Alarm clock","Buzzer"]),
"smoke_detector":({"smoke_detector"},["Smoke detector smoke alarm","Fire alarm"]),
"air_conditioner":({"air_conditioner"},["Air conditioning"]),
"blender":({"blender"},["Blender"]),
"microwave_oven":({"microwave_oven"},["Microwave oven"]),
"printer":({"printer"},["Printer"]),
"toilet":({"toilet"},["Toilet flush"]),
"sink_with_tap":({"sink_with_tap"},["Water tap faucet","Sink (filling or washing)"]),
"bathtub":({"bathtub"},["Bathtub (filling or washing)"]),
"floor_drain":({"floor_drain"},["Drip","Gurgling"]),
"fireplace":({"fireplace"},["Fire","Crackle"]),
}
p=argparse.ArgumentParser();p.add_argument("--root",required=True,type=Path);p.add_argument("--receipt",required=True,type=Path);a=p.parse_args();ip=a.root/"index.json";idx=json.loads(ip.read_text());ours=[x for x in idx["assets"] if x.get("pipeline")=="flux2_pixal3d_static_v1" and x.get("category")!="audio_playback"]
assert len(ours)==28 and len(FAMILIES)==14
groups={}
for family,(types,classes) in FAMILIES.items():
 assets=sorted(x["asset_id"] for x in ours if x["identity"]["object_type"] in types)
 assert assets, family
 groups[family]={"object_types":sorted(types),"audioset_classes":classes,"asset_ids":assets}
all_ids=[i for g in groups.values() for i in g["asset_ids"]];assert len(all_ids)==28 and len(set(all_ids))==28
all_classes=[i for g in groups.values() for i in g["audioset_classes"]];assert len(all_classes)==26 and len(set(all_classes))==26
mesh=dict(idx.get("mesh_families") or {});key="static_sound_sources_20260826";assert key not in mesh;mesh[key]={"status":"research_only","episode_counted":False,"families":groups};idx["mesh_families"]=mesh;ip.write_text(json.dumps(idx,ensure_ascii=False,indent=1))
assert not a.receipt.exists();a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps({"index":str(ip),"mapping_key":key,"family_count":14,"audioset_class_count":26,"asset_count":28,"families":groups},ensure_ascii=False,indent=1)+"\n");print(a.receipt)
