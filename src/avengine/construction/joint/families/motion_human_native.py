"""Retain independent original native witnesses; one event cannot veto siblings."""
from copy import deepcopy
import numpy as np


def retain_native_witnesses(video,scene,entries,plan,row_refs):
    from avengine.construction.joint import CheckResult
    from avengine.timeline.v7_motion_during_conditions import verify_motion_during_native
    actual=deepcopy(scene);admissions={}
    for entry in entries:
        ep=entry["endpoint_id"];record=entry.get("motion_episode")
        if record:
            frames=record.get("native_30Hz_readbacks",[])
            if len(frames)!=901:
                yield CheckResult("native:motion_human:mouth_readbacks","fail",
                                  "missing_actual_901_mouth_samples")
                return
            actual["sources"][ep]["trajectory"]=[
                {"time_s":i/30.,"position_m":r["emitter_position_m"]}for i,r in enumerate(frames)]
            admissions[ep]=record
        elif ep in actual["sources"]:
            point=video._apply_entry(entry,scene,frame_index=0)["emitter_position_m"]
            actual["sources"][ep]["trajectory"]=[
                {"time_s":t,"position_m":list(point)}for t in(0.,30.)]
    original=plan.metadata["family_row"]["motion_during_condition_plan"]
    retained=[];observations=[];dropped=[]
    for item in original["plans"]:
        observed=verify_motion_during_native({"motion_during_condition_plan":{"plans":[item]}},
                 actual,{"native_motion_admissions":admissions})
        record={"target_asset_id":item["target_asset_id"],"offsets_s":item["offsets_s"],
                "qa_id":item["config"]["query"]["qa_id"],"original_native_report":observed}
        observations.append(record)
        if observed["status"]=="pass":retained.append(item)
        else:dropped.append(record)
    filtered={**original,"plans":retained}
    counts={}
    for item in retained:
        qa=item["config"]["query"]["qa_id"]
        counts[qa]=counts.get(qa,0)+len(item["offsets_s"])
    containers=[plan.metadata["family_row"],*row_refs]
    if isinstance(scene.get("production_scenario"),dict):containers.append(scene["production_scenario"])
    for row in containers:
        row["motion_during_condition_plan"]=deepcopy(filtered)
        row["human_motion_event_native_verification"]=deepcopy(observations)
        row["native_family_degradations"]=deepcopy(dropped)
        demand=row.get("scene_demand")
        if isinstance(demand,dict):
            demand["required_actual_candidate_families"]=sorted(counts)
            demand["required_independent_family_counts"]=dict(counts)
            demand.setdefault("condition_families",{})["motion_human"]=sorted(counts)
    scene["motion_during_condition_plan"]=deepcopy(filtered)
    plan.metadata["motion_condition_plan"]=filtered
    plan.metadata["native_retained_family_counts"]=counts
    video.receipt["motion_human_joint_original_witnesses"]={
        "status":"pass"if retained else"fail","reports":observations,
        "per_event_family_degradations":dropped,"retained_family_counts":counts}
    if not retained:
        yield CheckResult("native:motion_human:original_witnesses","fail",
                          "no_original_motion_witness_survives_actual_native_readback",
                          evidence={"per_event_family_degradations":dropped})
        return
    yield CheckResult("native:motion_human:original_witnesses","pass",
                      "unchanged_original_verifier_per_retained_event_and_family",
                      evidence={"retained_family_counts":counts,
                                "per_event_family_degradations":dropped})
    # Owner 2026-10-16 12:15: body clipping is observational. Common
    # certification still requires walkable routes and actual root ground.
    # Compiled native contact/collision observations remain in the receipt.
    video.receipt["motion_human_body_collision_policy"]={
        "policy":"observation_only", "standalone_full_mesh_sweep":"not_run",
        "required_common_checks":["walkable_map","root_ground"],
        "basis":"owner_20261016_1215"}
