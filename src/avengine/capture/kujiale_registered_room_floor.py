"""Wire the legal room's verified source-floor lineage into native production.

Material-name deduplication may hide real floor faces. Existing exact source
triangle lineage supplies identity; it never relabels furniture or changes rays.
"""
from copy import deepcopy
from pathlib import Path
import json
from avengine.registry.kujiale_ue_measurements import read_registration,sha256
IDENTITY=('native_editor_audit','native_editor_audit_sha256','native_vertices_sha256','native_triangles_sha256')

def bind_room_floor(house,room_id,registration=None,contract=None):
    registration=registration if registration is not None else read_registration()[0]
    record=next((r for r in registration.get('rooms',[])if r['house']==house['house']and r['room_id']==room_id),{})
    declaration=record.get('native_floor_surface')
    if declaration is None and contract is not None:
        document=json.loads(Path(contract).read_text())if isinstance(contract,(str,Path))else contract;source=document['inputs']['room_kujiale'];path=Path(source['path'])
        if sha256(path)!=source['sha256']:raise ValueError('formal room legality input changed')
        document=json.loads(path.read_text());entries=document['rooms'].values()if isinstance(document['rooms'],dict)else document['rooms'];record=next(r for r in entries if r['house']==house['house']and r['room_id']==room_id)
        if record['native_room_render_status']!='pass'or record['floor_support_check']['status']!='pass':raise ValueError('room floor lacks original native qualification')
        declaration=record['native_floor_surface']
    if declaration is None:return deepcopy(house)
    original=house['input']['ue_config']['native_floor_surface']
    if any(str(original[k])!=str(declaration[k])for k in IDENTITY):raise ValueError('room floor lineage belongs to different native geometry')
    result=deepcopy(house);result['input']['ue_config']['native_floor_surface']=deepcopy(declaration)
    result['input']['ue_config']['room_floor_registration']={'house':house['house'],'room_id':room_id,'source':'formal native room legality / exact source-floor triangle lineage','floor_ray_and_contact_thresholds_changed':False}
    return result
