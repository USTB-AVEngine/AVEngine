"""Backend-specific native asset checks, preserving original load identities."""
from functools import lru_cache
import json
from pathlib import Path

@lru_cache(maxsize=128)
def _ue_receipt(path):
    from avengine.capture.v7_ue_registered_readback import digest
    proof=json.loads(Path(path).read_text())
    if proof['status']!='pass' or proof['backend']!='ue_spear':raise ValueError('native UE load not qualified')
    for item in proof['files']:
        if digest(item['path'])!=item['sha256'] or digest(item['source'])!=item['sha256']:
            raise ValueError('actual/original UE package changed after native load')
    if digest(proof['original_glb'])!=proof['original_glb_sha256']:
        raise ValueError('original UE import GLB changed')
    return proof

def require_loaded_asset(actor,asset):
    if actor.get('native_load_backend')=='ue_spear':
        proof=_ue_receipt(actor['native_load_receipt'])
        if (proof['asset_id']!=asset['asset_id'] or proof['source_registry_sha256']!=actor['source_registry_sha256']
                or proof['actual_class_path']!=asset['ue_generated_class']
                or actor['loaded_artifact_sha256']!=proof['loaded_artifact_sha256']
                or actor['original_glb_sha256']!=proof['original_glb_sha256']
                or proof['original_glb_sha256']!=asset['actual_artifact_sha256'][asset['habitat_glb']]):
            raise ValueError('actual loaded UE actor differs from original formal registration')
    elif actor['loaded_artifact_sha256']!=asset['actual_artifact_sha256'][asset['habitat_glb']]:
        raise ValueError('actual native GLB differs from original formal asset')
