"""Construct fixed root offsets from actual native floor/sole observations."""
from copy import deepcopy
from pathlib import Path
import math,os
from avengine.timeline.continuous_pilot import write_json

class NativeContactConstructionRequired(ValueError):
    def __init__(self,frame_index,actors,tolerance):
        self.frame_index=frame_index;self.actors=deepcopy(actors);self.tolerance=tolerance
        super().__init__('native human contact needs a measured fixed root correction')


def fixed_root_offset(observed_gaps,tolerance):
    values=[float(v)for v in observed_gaps]
    if not values or not all(math.isfinite(v)for v in values):raise ValueError('actual finite native sole gaps required')
    lower=max(-tolerance-v for v in values);upper=min(tolerance-v for v in values)
    if lower>upper:raise ValueError('one fixed root cannot satisfy these measured support views; choose another legal position')
    return (lower+upper)/2


def encode_with_contact_construction(video,scene,path,review_frames,collect_pixel_visibility,encode):
    """Keep every failed trial; use hardlinks for the final measured payload."""
    path=Path(path);trial=0
    while True:
        directory=path.parent/f'native_contact_try_{trial:03d}';directory.mkdir()
        try:result=encode(scene,directory/path.name,review_frames,collect_pixel_visibility=collect_pixel_visibility)
        except NativeContactConstructionRequired as error:
            packet={'frame_index':error.frame_index,'tolerance_m':error.tolerance,'actors':error.actors,'source':'actual native target-only depth and native UE floor support'}
            write_json(directory/'native_contact_correction_request.json',packet)
            by={c['endpoint_id']:c for c in video.alpha_contact_checks}
            for actor in error.actors:
                ep=actor['endpoint_id'];gap=actor['native_foot_contact']['gap_m']
                if abs(gap)<=error.tolerance:continue
                offset=video.vertical_root_offsets.get(ep,0.);history=video.contact_gap_history.setdefault(ep,[])
                history.append(gap-offset)
                history.extend(v['gap_m']-offset for v in by[ep]['native_sole_depth']['views'])
                try:updated=fixed_root_offset(history,error.tolerance)
                except ValueError:
                    video.rejected_contact_positions.setdefault(ep,[]).append(actor['position_m'][:1]+actor['position_m'][2:])
                    video.placement_overrides.pop(ep,None);video.vertical_root_offsets.pop(ep,None);video.contact_gap_history.pop(ep,None)
                else:video.vertical_root_offsets[ep]=updated
            receipt={'status':'constructed_pending_same_contact_recheck','source_trial':str(directory),'tolerance_m':error.tolerance,
                'vertical_root_offsets_m':video.vertical_root_offsets,'baseline_measured_gaps_m':video.contact_gap_history,
                'rejected_native_positions_xz_m':video.rejected_contact_positions,'entities_removed':False}
            write_json(video.contact_construction_path,receipt);trial+=1;continue
        # All original contact, negative-control, visibility and native-pose gates
        # have passed on this capture. Original failed bytes are never changed.
        for source in directory.rglob('*'):
            if not source.is_file():continue
            target=path.parent/source.relative_to(directory);target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():os.link(source,target)
        write_json(video.contact_construction_path,{'status':'same_original_contact_check_passed','tolerance_m':.015,'vertical_root_offsets_m':video.vertical_root_offsets,'baseline_measured_gaps_m':video.contact_gap_history,'rejected_native_positions_xz_m':video.rejected_contact_positions,'source_trial':str(directory)})
        result['native_vertical_contact_construction']={'status':'same_original_contact_check_passed',
            'tolerance_m':.015,'vertical_root_offsets_m':deepcopy(video.vertical_root_offsets),
            'source':str(video.contact_construction_path),'successful_trial':str(directory),'failed_trials_preserved':trial}
        return result
