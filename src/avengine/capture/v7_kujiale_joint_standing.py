"""Keep both appearance members at one completely measured standing layout."""
from pathlib import Path
import hashlib,json


def completed_shared_standing(parent,assets):
    """Reuse geometry of this fresh parent only; all current gates still rerun."""
    parent=Path(parent);target={a['endpoint_id']:a['asset_id']for a in assets}
    for visual in ('v0','v1'):
        marker=parent/f'visual_{visual}_completed.json'
        if not marker.is_file():continue
        saved=json.loads(marker.read_text());path=parent/saved['attempt']/'native_registered_visual_evidence.json'
        if not path.resolve().is_relative_to(parent.resolve()):raise ValueError('shared standing evidence escaped this parent')
        data=json.loads(path.read_text());frames=data['frames'];first={a['endpoint_id']:a for a in frames[0]['actors']if a.get('endpoint_id')in target};source={k:v['asset_id']for k,v in first.items()}
        if source==target:continue
        if set(source)!=set(target)or sorted(source.values())!=sorted(target.values()):raise ValueError('shared standing reference is not the same appearance quartet')
        if data['frame_count']!=450 or len(frames)!=450 or data['native_body_foot_status']!='pass'or data['native_source_material_status']!='pass':raise ValueError('shared standing requires complete passed native geometry')
        if len(data['standing_admission'])!=len(assets)or any(a['status']!='pass'or not a['known_bad_detected']for a in data['standing_admission']):raise ValueError('shared standing lacks unchanged contact and negative control')
        for frame in frames:
            observed={a['endpoint_id']:a for a in frame['actors']if a.get('endpoint_id')in target}
            if set(observed)!=set(first)or any(observed[k]['position_m']!=first[k]['position_m']for k in first):raise ValueError('shared standing reference is not actually static')
        return {'source_visual':visual,'native_evidence':str(path),'native_evidence_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'actors':{k:{'position_m':a['position_m'],'native_support_floor_y_m':a['native_support_floor_y_m']}for k,a in first.items()},'current_native_contact_collision_visibility_and_acoustics_must_rerun':True,'thresholds_changed':False}
    return None
