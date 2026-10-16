#!/usr/bin/env python3
"""Create two final toilet forms whose declared attributes match stable outputs."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path

LOW_SOURCE="plumbing_fixture_toilet_floor_flushometer_product_view_v1"
HIGH_SOURCE="plumbing_fixture_toilet_high_tank_pull_chain_product_view_v1"
LOW_ID="plumbing_fixture_toilet_exposed_flush_pipe_tank_product_view_v1"
HIGH_ID="plumbing_fixture_toilet_elevated_tank_exposed_pipe_product_view_v1"

def main():
 p=argparse.ArgumentParser();p.add_argument("--spear-root",required=True,type=Path);p.add_argument("--avengine-root",required=True,type=Path);a=p.parse_args();sys.path.insert(0,str(a.spear_root));from tools import controlled_source_asset_schema as c
 sd=a.spear_root/"data/controlled_source_attributes_v1/candidate_profiles/static_object";md=a.avengine_root/"examples/assets/source_profiles_mirror/candidate_profiles/static_object"
 def load(i):
  l=sd/f"{i}.json";r=md/f"{i}.json";assert l.read_bytes()==r.read_bytes();return json.loads(l.read_text())
 low=load(LOW_SOURCE);high=load(HIGH_SOURCE)
 def make(src,i,rev,form,label,pos,neg,h):
  x=json.loads(json.dumps(src));x["profile_schema_id"]=i;x["profile_revision"]=rev;x["fixed_attributes"]["form_factor"]=form;g=x["generation_contract"];g["prompt_template_id"]=f"static_product_view_t2i_v1_{form}";g["positive_template"]=pos;g["negative_prompt"]+=", "+neg;g["value_labels"]["form_factor"]={form:label};t=x["target_physical_profiles"];t["profile_id"]=f"{i}_physical_candidate_v1";t["reference_value_cm"]=h;t["reference_provenance"]["source_id"]=f"{form}_toilet_typical_retail_dimension_v1";t["values"]["fixed"]={"target_value_cm":h,"tolerance_cm":15};x["qa_contract"]["subject_label"]=label;c.validate_attribute_profile(x);return x
 lo=make(low,LOW_ID,"2026_08_26_v1_exposed_flush_pipe_tank_form","exposed_flush_pipe_tank","floor toilet with exposed flush pipe and tank","One {body_color} {material} {form_factor} {object_type}, a household {category}: one floor-standing toilet bowl on a ceramic pedestal with one rectangular rear cistern. A prominent exposed chrome flush valve and curved chrome pipe are fully visible between the cistern bottom and a visible inlet at the rear of the open bowl. The seat and lid are raised and the complete valve-to-pipe-to-bowl water path faces the camera.","hidden pipe, pipe behind tank, missing pipe, tank touching bowl with no visible water path, closed lid, lowered seat, wall-hung bowl",100)
 hi=make(high,HIGH_ID,"2026_08_26_v1_elevated_tank_exposed_pipe_form","elevated_tank_exposed_pipe","elevated-tank toilet with exposed downpipe","One {body_color} {material} {form_factor} {object_type}, a household {category}: one complete toilet assembly with a rectangular cistern visibly elevated well above the floor-standing bowl. One long exposed vertical chrome downpipe spans the clear air gap from the tank bottom to a visible rear bowl inlet. The seat and lid are raised; the whole separated tank, pipe and bowl fit in frame. No pull chain is required.","low tank, tank touching bowl, close-coupled toilet, hidden pipe, missing downpipe, disconnected pipe, wall, bathroom scene, closed lid",150)
 for x in (lo,hi):
  n=f"{x['profile_schema_id']}.json";assert not (sd/n).exists() and not (md/n).exists()
 for x in (lo,hi):
  s=json.dumps(x,ensure_ascii=False,indent=1)+"\n";n=f"{x['profile_schema_id']}.json";(sd/n).write_text(s);(md/n).write_text(s)
 rec={"owner_scope":"static_sound_sources_first_pass_20260826","supersedes_adjustment_record":"examples/assets/source_profiles_mirror/static_sound_form_adjustments_20260826_v2.json","adjustments":[{"replaced_profiles":[LOW_SOURCE,HIGH_SOURCE],"replacement_profiles":[LOW_ID,HIGH_ID],"reason":"The prior adjusted candidates exposed the water path but contradicted only the tankless/pull-chain labels. These final forms retain the stable visible flush-pipe geometry and remove claims the generator did not satisfy.","old_profiles_retained_as_rejected_method_evidence":True}]};path=a.avengine_root/"examples/assets/source_profiles_mirror/static_sound_form_adjustments_20260826_v3.json";assert not path.exists();path.write_text(json.dumps(rec,ensure_ascii=False,indent=1)+"\n");print(path)
if __name__=="__main__":raise SystemExit(main())
