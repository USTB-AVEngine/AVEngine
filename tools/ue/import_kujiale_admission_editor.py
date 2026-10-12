"""UE 5.5 editor import of all staged admitted individuals; no source edits."""
import json,os,time,traceback
from pathlib import Path
import unreal
ROOT=Path(os.environ['V7_KUJIALE_ASSET_ROOT']);MANIFEST=Path(os.environ['V7_KUJIALE_IMPORT_INPUTS']);V=json.loads(MANIFEST.read_text());REV=V['revision'];LABEL=V.get('batch_label',REV)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def xyz(v):return [v.x,v.y,v.z]

def run():
    rows=[];tools=unreal.AssetToolsHelpers.get_asset_tools()
    for row in V['assets']:
        ev=ROOT/'evidence/import'/REV/(row['asset_id']+'.json')
        if ev.exists() and json.loads(ev.read_text()).get('status')=='pass':rows.append(json.loads(ev.read_text()));continue
        rec={'asset_id':row['asset_id'],'status':'fail','static':[],'motion':[],'input_manifest':str(MANIFEST)};start=time.time()
        try:
            for kind,key in [('static','static_glb'),('motion','motion_glb')]:
                if not row.get(key):continue
                task=unreal.AssetImportTask();task.filename=row[key];task.destination_path=row['destination_path']+'/'+kind;task.automated=True;task.save=True;task.replace_existing=False;task.async_=True
                tools.import_asset_tasks([task])
                for obj in task.get_objects():
                    cl=obj.get_class().get_name();item={'path':obj.get_path_name(),'class':cl,'name':obj.get_name()}
                    if cl=='StaticMesh':
                        settings=obj.get_editor_property('nanite_settings');item['nanite_before']=str(settings)
                        settings.set_editor_property('enabled',False);obj.set_editor_property('nanite_settings',settings)
                        unreal.EditorAssetLibrary.save_loaded_asset(obj)
                        box=obj.get_bounding_box();item['bounds_cm']={'min':xyz(box.min),'max':xyz(box.max)}
                        mats=obj.get_editor_property('static_materials');item['materials']=[m.material_interface.get_path_name() if m.material_interface else None for m in mats]
                        # LOD0 native geometry, used for actual triangle contact/intersection checks.
                        points=[];faces=[]
                        for section in range(len(mats)):
                            verts,indices,*_=unreal.ProceduralMeshLibrary.get_section_from_static_mesh(obj,0,section);offset=len(points);points.extend(xyz(p) for p in verts);faces.extend([offset+indices[i+j] for j in range(3)] for i in range(0,len(indices),3))
                        geom=ROOT/'evidence/import_geometry'/REV/row['asset_id']/(obj.get_name()+'.json');write(geom,{'vertices_ue_cm':points,'triangles':faces});item['geometry_path']=str(geom)
                    if cl=='AnimSequence':item['length_s']=float(obj.get_editor_property('sequence_length'))
                    rec[kind].append(item)
                unreal.EditorAssetLibrary.save_directory(task.destination_path,only_if_is_dirty=True,recursive=True)
            if not any(x['class']=='StaticMesh' for x in rec['static']):raise RuntimeError('static import produced no StaticMesh')
            if row['walk_allowed']:
                if not any(x['class']=='SkeletalMesh' for x in rec['motion']):raise RuntimeError('walker import produced no SkeletalMesh')
                names=[x['name'] for x in rec['motion'] if x['class']=='AnimSequence']
                if not any('Idle' in n for n in names) or not any('Walking' in n for n in names):raise RuntimeError('walker is missing idle/walking sequences: '+repr(names))
            rec['status']='pass'
        except Exception:rec['error']=traceback.format_exc()
        rec['elapsed_seconds']=time.time()-start;write(ev,rec);rows.append(rec);write(ROOT/'evidence'/f'import_{LABEL}.json',rows);unreal.log('KUJIALE_ADMISSION_IMPORT '+row['asset_id']+' '+rec['status'])
    write(ROOT/'evidence'/f'import_{LABEL}_complete.json',{'status':'pass' if all(x['status']=='pass' for x in rows) and len(rows)==len(V['assets']) else 'fail','requested':len(V['assets']),'completed':len(rows),'results_path':str(ROOT/'evidence'/f'import_{LABEL}.json')})
try:run()
except Exception:write(ROOT/'evidence'/f'import_{LABEL}_fatal.json',{'error':traceback.format_exc()});raise
if not os.environ.get('V7_COMMANDLET'):
    unreal.SystemLibrary.quit_editor()
