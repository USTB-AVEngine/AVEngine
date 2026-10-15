"""The unchanged original 5-degree position gate on this prepared native world."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .types import CheckResult


def native_acoustic_checks(video, scene, runtime, directory, *, entries=None, gate_factory=None):
    """Measure actual frame-zero mouth/attached-foot emitters, never planned proxies.

    Caller runtime is passed intact to the original gate. CPU/RIR dispatch belongs
    to the production caller; this helper never pins workers to another task.
    """
    from avengine.timeline.continuous_pilot import write_json
    if gate_factory is None:
        from avengine.timeline.v7_position_admission import PositionAdmission
        gate_factory=PositionAdmission
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    receipt=directory/"acoustic_placement_admission.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    active={e["endpoint_id"]for e in scene.get("events",[])}
    points={}
    for entry in video.actors if entries is None else entries:
        observed=(video._apply_motion_entry(entry,0)if entry.get("motion_episode")
                  else video._apply_entry(entry,scene))
        ep=entry["endpoint_id"]
        if ep in active and observed.get("native_emitter_present",not entry["declaration"].get("silent_decor",False)):
            points[ep]=np.asarray(observed["emitter_position_m"],float)
        for attached in observed.get("attached_emitter_readbacks",[]):
            if attached["endpoint_id"]in active:
                points[attached["endpoint_id"]]=np.asarray(attached["position_m"],float)
    checks=[]
    for ep in sorted(active-set(points)):
        checks.append(CheckResult("native:acoustic:"+ep,"unknown","active_emitter_has_no_actual_native_readback"))
    if active:
        gate=gate_factory(scene,video.motion_house,runtime,directory/"acoustic_rir_cache",receipt)
        try:
            for ep,point in sorted(points.items()):
                if point.shape!=(3,) or not np.isfinite(point).all():
                    checks.append(CheckResult("native:acoustic:"+ep,"fail","invalid_actual_emitter_coordinate"))
                    continue
                measured=gate.check(ep,point)
                status={"pass":"pass","blocked":"fail"}.get(measured.get("status"),"unknown")
                checks.append(CheckResult("native:acoustic:"+ep,status,
                    "unchanged_original_5deg_position_admission",evidence=measured))
        finally:
            gate.close()
    else:
        write_json(receipt,{"status":"not_applicable_no_active_emitters","checks":[],"unchanged_gate_deg":5.})
        checks.append(CheckResult("native:acoustic:none","pass","no_active_emitters"))
    report={"status":"pass"if all(c.status=="pass"for c in checks)else"fail",
            "receipt_path":str(receipt),"gate":"avengine.timeline.v7_position_admission.PositionAdmission",
            "unchanged_gate_deg":5.,"actual_points_m":{ep:p.tolist()for ep,p in points.items()},
            "active_endpoints":sorted(active),"listener_m":list(scene["listener_m"]),
            "camera_forward":list(scene["camera_forward"]),
            "scope":"actual prepared frame-zero emitters; full moving RIR checks remain unchanged"}
    scene["joint_acoustic_preflight"]=report
    write_json(directory/"joint_acoustic_preflight.json",report)
    return tuple(checks)
