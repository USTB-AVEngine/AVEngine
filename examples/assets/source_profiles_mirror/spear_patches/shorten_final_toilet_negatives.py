#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
IDS={
"plumbing_fixture_toilet_exposed_flush_pipe_tank_product_view_v1":"human, hand, text, logo, watermark, multiple objects, cropped object, floating object, cluttered background, bathroom scene, hidden pipe, missing pipe, tank touching bowl with no visible water path, closed lid, lowered seat, wall-hung bowl",
"plumbing_fixture_toilet_elevated_tank_exposed_pipe_product_view_v1":"human, hand, text, logo, watermark, multiple objects, cropped object, floating object, cluttered background, bathroom scene, low tank, tank touching bowl, hidden pipe, missing downpipe, disconnected pipe, closed lid, lowered seat",
}
p=argparse.ArgumentParser();p.add_argument("--spear-root",required=True,type=Path);p.add_argument("--avengine-root",required=True,type=Path);a=p.parse_args();sys.path.insert(0,str(a.spear_root));from tools import controlled_source_asset_schema as c
sd=a.spear_root/"data/controlled_source_attributes_v1/candidate_profiles/static_object";md=a.avengine_root/"examples/assets/source_profiles_mirror/candidate_profiles/static_object"
for i,n in IDS.items():
 l=sd/f"{i}.json";r=md/f"{i}.json";assert l.read_bytes()==r.read_bytes();x=json.loads(l.read_text());x["generation_contract"]["negative_prompt"]=n;c.validate_attribute_profile(x);s=json.dumps(x,ensure_ascii=False,indent=1)+"\n";l.write_text(s);r.write_text(s);print(i)
