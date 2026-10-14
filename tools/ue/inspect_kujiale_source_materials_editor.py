import json,os
from pathlib import Path
import unreal
r=json.loads(Path(os.environ['AVENGINE_ROUND6_MATERIAL_INSPECT']).read_text())
lib=unreal.MaterialEditingLibrary
source=json.loads(Path(r['repair_request']).read_text())
records=[]
for entry in source['repairs']:
 path=entry['native_material']
 if any(x['native_material']==path for x in records):continue
 m=unreal.load_asset(path);row={'native_material':path,'asset_id':entry['asset_id']}
 row['scalars']={str(n):lib.get_material_instance_scalar_parameter_value(m,n) for n in lib.get_scalar_parameter_names(m)}
 row['vectors']={str(n):str(lib.get_material_instance_vector_parameter_value(m,n)) for n in lib.get_vector_parameter_names(m)}
 row['switches']={str(n):lib.get_material_instance_static_switch_parameter_value(m,n) for n in lib.get_static_switch_parameter_names(m)}
 textures=[]
 for n in lib.get_texture_parameter_names(m):
  t=lib.get_material_instance_texture_parameter_value(m,n)
  if t:textures.append({'parameter':str(n),'path':t.get_path_name(),'srgb':t.get_editor_property('srgb'),'compression':str(t.get_editor_property('compression_settings'))})
 row['textures']=textures
 parent=m.get_editor_property('parent');row['parent']=parent.get_path_name()
 records.append(row)
Path(r['output']).write_text(json.dumps({'records':records,'writes_to_assets':False},indent=2)+'\n')
