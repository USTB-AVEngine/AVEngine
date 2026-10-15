"""Versioned production UE basis/package registration, independent of run copies."""
from pathlib import Path
import hashlib,json,re
import numpy as np

SCHEMA='v7-kujiale-ue-native-measurements-v1'
DEFAULT_REGISTRATION=Path(__file__).resolve().parents[3]/'registrations/v7_kujiale_ue_native_measurements_20261015_v2.json'

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def read_registration(path=None):
    path=Path(path or DEFAULT_REGISTRATION)
    data=json.loads(path.read_text())
    if data.get('schema')!=SCHEMA:raise ValueError('unsupported native measurement registration')
    return data,{'path':str(path),'sha256':sha256(path),'revision':data['revision']}

def native_asset(asset,registration):
    row=next((r for r in registration['assets']if r['asset_id']==asset['asset_id']),None)
    if row is None:raise ValueError('asset has no measured production UE registration: '+asset['asset_id'])
    matrix=np.asarray(row['glb_to_ue_local_cm'],float)
    if matrix.shape!=(3,3)or not np.isfinite(matrix).all():raise ValueError('invalid registered native basis')
    if not np.allclose(matrix.T@matrix,np.eye(3)*10000,atol=1e-6):raise ValueError('native registration basis is not centimetre orthogonal')
    if row['actual_class_path']!=asset['ue_generated_class']:raise ValueError('formal native Blueprint differs')
    if sorted(row['actual_mesh_paths'])!=sorted(asset['registered_native_ue']['mesh_paths']):raise ValueError('formal native meshes differ')
    if not np.allclose(row['mouth_local_glb_m'],asset['mouth_local_glb_m'],atol=1e-12):raise ValueError('formal native mouth differs')
    return row,matrix

def loaded_map_package(project,map_path,launch_directory):
    if not map_path.startswith('/Game/'):raise ValueError('native map outside Game mount')
    content=(Path(project).parent/'Content').resolve();package=(content/(map_path[6:]+'.umap')).resolve()
    if not package.is_relative_to(content)or not package.is_file():raise ValueError('native map package missing')
    log=Path(launch_directory)/'UnrealEditor.log'
    loads=re.findall(r'LogLoad: LoadMap: ([^?\s]+)',log.read_text(errors='replace'))
    if not loads or loads[-1]!=map_path:raise ValueError('actual UE loaded map differs from requested map')
    return {'status':'pass','actual_map_path':loads[-1],'original_package_path':str(package),'sha256':sha256(package),
            'native_load_log':str(log),'native_load_log_sha256':sha256(log),'method':'UE LogLoad final map and original mounted umap bytes'}
