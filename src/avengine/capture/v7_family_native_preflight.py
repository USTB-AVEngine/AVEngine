"""Run family-owned witnesses on the final native world before media encoding."""
from pathlib import Path
from avengine.timeline.continuous_pilot import write_json


def run_family_native_preflights(video,scene,entries,directory):
    scenario=scene.get('production_scenario')or{};requests=scenario.get('motion_requests')or{}
    construction=scene.get('automatic_construction')or{};requested=[]
    if any(r.get('joint_entry_route')for r in requests.values()):requested.append('QA-07')
    if construction.get('qa21_target_asset_ids')or construction.get('qa21_target_asset_id'):requested.append('QA-21')
    if any(r.get('qa09_demand')for r in scene.get('native_dynamic_occlusion_relations',[])):requested.append('QA-09')
    if not requested:return {}
    if {id(e)for e in entries}!={id(e)for e in video.actors}:
        raise ValueError('native family preflight requires every final placed actor')
    reports={}
    for family in requested:
        if family=='QA-07':
            from avengine.construction.joint.families.qa07 import native_entry_preflight
            report=native_entry_preflight(video,scene,entries)
        elif family=='QA-21':
            from avengine.construction.joint.families.qa21 import native_candidate_check
            report=native_candidate_check(video,scene,entries,video._native_pair_pixels(scene,entries))
        else:
            from avengine.capture.v7_dynamic_occlusion_proposals import measure_dynamic_relations
            measured=measure_dynamic_relations(video,scene,entries)
            report={**measured,'status':'pass'if measured['satisfies_request']else'fail'}
        reports[family]=report
        write_json(Path(directory)/(family.lower().replace('-','_')+'_native_preflight.json'),report)
        if report.get('status')!='pass':
            raise ValueError(family+' final native preflight failed: '+str(report.get('reasons',report.get('reason',report))))
    return reports
